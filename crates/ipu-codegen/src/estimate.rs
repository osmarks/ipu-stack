//! Memory, communication, and capacity estimates shared by planning policies.

use crate::cost::IPU21_TARGET_COSTS;
use crate::graph::TensorShape;
use crate::mid::{
    Layout, MemoryClass, MemoryEstimate, MemoryPeaks, MemoryUsage, MidOperation, MidValue,
    MidValueId, OperatorDispatch, Precision, TensorAxis, TensorType,
};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn physical_elements(shape: &TensorShape, layout: &Layout) -> u64 {
    layout
        .padded_shape(shape)
        .map_or_else(|_| shape.elements(), |shape| shape.elements())
        .saturating_mul(u64::from(layout.tiling.replicas))
}

pub(crate) fn maximum_shard_bytes(tensor: &TensorType) -> u64 {
    let Ok(padded) = tensor.format.layout.padded_shape(&tensor.shape) else {
        return u64::MAX;
    };
    padded
        .0
        .iter()
        .enumerate()
        .map(|(index, &extent)| {
            tensor
                .format
                .layout
                .tiling
                .axes
                .iter()
                .find(|axis| axis.axis.resolve(padded.0.len()) == Ok(index))
                .map_or(u64::from(extent), |axis| {
                    let blocks = extent / axis.block_size;
                    u64::from(blocks.div_ceil(u32::from(axis.partitions)) * axis.block_size)
                })
        })
        .product::<u64>()
        .saturating_mul(tensor.format.precision.bytes())
}

pub(crate) fn maximum_axis_shard_extent(tensor: &TensorType, axis: usize) -> u64 {
    (0..tensor.format.layout.tiling.tile_count)
        .filter_map(|tile| tile_axis_range(tensor, axis, tile))
        .map(|range| u64::from(range.end - range.start))
        .max()
        .unwrap_or(u64::MAX)
}

fn tensor_memory(tensor: &TensorType) -> MemoryUsage {
    let mut usage = MemoryUsage::default();
    usage.add_class(
        tensor.format.layout.memory_class,
        maximum_shard_bytes(tensor),
    );
    usage
}

pub(crate) fn operator_memory_estimate(
    dispatch: &OperatorDispatch,
    inputs: &[TensorType],
    output: &TensorType,
) -> MemoryEstimate {
    let live = inputs.iter().fold(tensor_memory(output), |usage, input| {
        usage.saturating_add(tensor_memory(input))
    });
    let mut temporary = MemoryUsage::default();
    if let (
        OperatorDispatch::BlockedGemm {
            inner_block,
            output_column_block,
            ..
        },
        Some(right),
    ) = (dispatch, inputs.get(1))
        && right.format.precision == Precision::F16
        && gemm_requires_staging(dispatch, right, output)
    {
        // One packed panel is reused across K phases. Exchange receive storage
        // is a separately reserved architectural window and is not counted as
        // planned data SRAM.
        temporary.interleaved = u64::from(*inner_block)
            .saturating_mul(u64::from(*output_column_block))
            .saturating_mul(right.format.precision.bytes());
    }
    MemoryEstimate {
        live,
        temporary,
        peak: live.saturating_add(temporary),
    }
}

pub(crate) fn gemm_requires_staging(
    dispatch: &OperatorDispatch,
    right: &TensorType,
    output: &TensorType,
) -> bool {
    if right.format.layout.memory_class == MemoryClass::Ipu21Interleaved {
        return false;
    }
    let OperatorDispatch::BlockedGemm { inner_block, .. } = dispatch else {
        return false;
    };
    let rank = right.shape.0.len();
    let output_rank = output.shape.0.len();
    if rank < 2 || output_rank < 2 {
        return true;
    }
    let streamed = right
        .format
        .layout
        .tiling
        .axes
        .iter()
        .any(|axis| axis.axis == TensorAxis::FromEnd(2) && axis.partitions > 1);
    let k = right.shape.0[rank - 2];
    let columns = maximum_axis_shard_extent(output, output_rank - 1);
    streamed || (k > *inner_block && columns > 16)
}

pub(crate) fn gemm_exchange_bytes_per_cycle(inputs: &[TensorType]) -> u64 {
    let Some(right) = inputs.get(1) else {
        return IPU21_TARGET_COSTS.exchange_bytes_per_cycle;
    };
    let inner_sharded = right
        .format
        .layout
        .tiling
        .axes
        .iter()
        .any(|axis| axis.axis == TensorAxis::FromEnd(2) && axis.partitions > 1);
    let column_partitions = right
        .format
        .layout
        .tiling
        .axes
        .iter()
        .find(|axis| axis.axis == TensorAxis::FromEnd(1))
        .map_or(1, |axis| axis.partitions);
    // IPU21 pairs adjacent tiles on one shared exchange bus. With one output
    // column partition, a streamed K panel is a full broadcast to consecutive
    // tiles and both receivers can consume it in the same cycle.
    if inner_sharded && column_partitions == 1 {
        IPU21_TARGET_COSTS.exchange_bytes_per_cycle * 2
    } else {
        IPU21_TARGET_COSTS.exchange_bytes_per_cycle
    }
}

pub(crate) fn conversion_memory_estimate(
    input: &TensorType,
    output: &TensorType,
) -> MemoryEstimate {
    let live = tensor_memory(input).saturating_add(tensor_memory(output));
    MemoryEstimate {
        live,
        temporary: MemoryUsage::default(),
        peak: live,
    }
}

pub(crate) fn region_peak_memory(
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
) -> MemoryPeaks {
    let mut uses = BTreeMap::<MidValueId, u32>::new();
    for input in operations.iter().flat_map(|operation| &operation.inputs) {
        *uses.entry(*input).or_default() += 1;
    }
    for output in outputs {
        *uses.entry(*output).or_default() += 1;
    }
    let mut live_values = BTreeSet::new();
    let mut live = MemoryUsage::default();
    for id in initial {
        if live_values.insert(*id) {
            live = live.saturating_add(tensor_memory(&values[id.index() as usize].tensor_type));
        }
    }
    let mut peaks = MemoryPeaks::default();
    peaks.observe(live);
    for operation in operations {
        let mut during = live.saturating_add(operation.memory.temporary);
        for result in &operation.results {
            if !live_values.contains(result) {
                during = during
                    .saturating_add(tensor_memory(&values[result.index() as usize].tensor_type));
            }
        }
        peaks.observe(during);
        for input in &operation.inputs {
            if let Some(remaining) = uses.get_mut(input) {
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 && live_values.remove(input) {
                    let bytes = tensor_memory(&values[input.index() as usize].tensor_type);
                    live.standard = live.standard.saturating_sub(bytes.standard);
                    live.interleaved = live.interleaved.saturating_sub(bytes.interleaved);
                }
            }
        }
        for result in &operation.results {
            if uses.get(result).copied().unwrap_or(0) != 0 && live_values.insert(*result) {
                live = live
                    .saturating_add(tensor_memory(&values[result.index() as usize].tensor_type));
            }
        }
    }
    peaks.observe(live);
    peaks
}

pub(crate) fn gemm_remote_bytes_per_tile(inputs: &[TensorType], output: &TensorType) -> u64 {
    let [left, right] = inputs else {
        return u64::MAX;
    };
    if left.shape.0.len() < 2 || right.shape.0.len() < 2 || output.shape.0.len() < 2 {
        return u64::MAX;
    }
    let tiles = output.format.layout.tiling.tile_count;
    let output_row_axis = output.shape.0.len() - 2;
    let output_column_axis = output.shape.0.len() - 1;
    let left_row_axis = left.shape.0.len() - 2;
    let right_column_axis = right.shape.0.len() - 1;
    let right_inner_axis = right.shape.0.len() - 2;
    let k = left.shape.0[left.shape.0.len() - 1];
    (0..tiles).fold(0u64, |maximum, tile| {
        let Some(output_rows) = tile_axis_range(output, output_row_axis, tile) else {
            return u64::MAX;
        };
        let Some(output_columns) = tile_axis_range(output, output_column_axis, tile) else {
            return u64::MAX;
        };
        let Some(left_rows) = tile_axis_range(left, left_row_axis, tile) else {
            return u64::MAX;
        };
        let Some(right_columns) = tile_axis_range(right, right_column_axis, tile) else {
            return u64::MAX;
        };
        let Some(right_inner) = tile_axis_range(right, right_inner_axis, tile) else {
            return u64::MAX;
        };
        let left_remote = if left_rows.start > output_rows.start || left_rows.end < output_rows.end
        {
            u64::from(output_rows.end - output_rows.start)
                .saturating_mul(u64::from(k))
                .saturating_mul(left.format.precision.bytes())
        } else {
            0
        };
        let right_remote = if right_columns.start > output_columns.start
            || right_columns.end < output_columns.end
            || right_inner.start != 0
            || right_inner.end < k
        {
            u64::from(output_columns.end - output_columns.start)
                .saturating_mul(u64::from(k))
                .saturating_mul(right.format.precision.bytes())
        } else {
            0
        };
        maximum.max(left_remote.saturating_add(right_remote))
    })
}

fn tile_axis_range(tensor: &TensorType, axis: usize, tile: u16) -> Option<std::ops::Range<u32>> {
    let layout = &tensor.format.layout;
    let padded = layout.padded_shape(&tensor.shape).ok()?;
    let Some((tiling, stride)) = layout
        .tiling
        .axes
        .iter()
        .zip(layout.tiling.axis_strides().ok()?)
        .find(|(tiling, _)| tiling.axis.resolve(padded.0.len()) == Ok(axis))
    else {
        return Some(0..padded.0[axis]);
    };
    let blocks = padded.0[axis] / tiling.block_size;
    let partitions = u32::from(tiling.partitions);
    let coordinate = (u32::from(tile) / stride) % partitions;
    let short = blocks / partitions;
    let long = blocks % partitions;
    let start_blocks = coordinate * short + coordinate.min(long);
    let shard_blocks = short + u32::from(coordinate < long);
    Some(start_blocks * tiling.block_size..(start_blocks + shard_blocks) * tiling.block_size)
}
