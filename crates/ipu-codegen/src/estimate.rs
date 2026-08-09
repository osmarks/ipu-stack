//! Memory, communication, and capacity estimates shared by planning policies.

use crate::cost::IPU21_TARGET_COSTS;
use crate::graph::TensorShape;
use crate::mid::{
    AmpOrder, ElementOrder, Layout, MemoryClass, MemoryEstimate, MemoryOperand, MemoryPeaks,
    MemoryRelation, MemoryUsage, MidOperation, MidValue, MidValueId, OperatorDispatch, Precision,
    TensorAxis, TensorType,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ConversionTraffic {
    pub source_payload_bytes: u64,
    pub remote_fragments: u64,
    pub maximum_destination_bytes: u64,
    pub maximum_local_bytes: u64,
    pub maximum_intersections: u64,
    pub maximum_local_intersections: u64,
}

pub(crate) fn conversion_traffic(
    shape: &TensorShape,
    precision: Precision,
    from: &Layout,
    to: &Layout,
) -> Option<ConversionTraffic> {
    let sources = layout_extents(shape, from)?;
    let destinations = layout_extents(shape, to)?;
    let element_bytes = precision.bytes();
    let mut remote = BTreeSet::<(u16, Vec<(u32, u32)>)>::new();
    let mut traffic = ConversionTraffic::default();
    for (destination_tile, destination) in destinations.iter().enumerate() {
        let mut intersections = BTreeMap::<Vec<(u32, u32)>, u16>::new();
        for (source_tile, source) in sources.iter().enumerate() {
            let Some(extents) = intersect_ranges(source, destination) else {
                continue;
            };
            let source_tile = u16::try_from(source_tile).ok()?;
            let selected = intersections.entry(extents).or_insert(source_tile);
            if usize::from(source_tile) == destination_tile {
                *selected = source_tile;
            }
        }
        let mut destination_bytes = 0u64;
        let mut local_bytes = 0u64;
        let mut local_intersections = 0u64;
        for (extents, source_tile) in &intersections {
            let bytes = range_elements(extents).saturating_mul(element_bytes);
            destination_bytes = destination_bytes.saturating_add(bytes);
            if usize::from(*source_tile) == destination_tile {
                local_bytes = local_bytes.saturating_add(bytes);
                local_intersections += 1;
            } else {
                remote.insert((*source_tile, extents.clone()));
            }
        }
        traffic.maximum_destination_bytes =
            traffic.maximum_destination_bytes.max(destination_bytes);
        traffic.maximum_local_bytes = traffic.maximum_local_bytes.max(local_bytes);
        traffic.maximum_intersections = traffic
            .maximum_intersections
            .max(intersections.len() as u64);
        traffic.maximum_local_intersections =
            traffic.maximum_local_intersections.max(local_intersections);
    }
    traffic.remote_fragments = remote.len() as u64;
    traffic.source_payload_bytes = remote
        .iter()
        .map(|(_, extents)| range_elements(extents).saturating_mul(element_bytes))
        .sum();
    Some(traffic)
}

fn layout_extents(shape: &TensorShape, layout: &Layout) -> Option<Vec<Vec<(u32, u32)>>> {
    let padded = layout.padded_shape(shape).ok()?;
    let rank = shape.0.len();
    let strides = layout.tiling.axis_strides().ok()?;
    let axes = layout
        .tiling
        .axes
        .iter()
        .zip(strides)
        .map(|(axis, stride)| Some((axis.axis.resolve(rank).ok()?, axis, stride)))
        .collect::<Option<Vec<_>>>()?;
    (0..layout.tiling.tile_count)
        .map(|tile| {
            (0..rank)
                .map(|axis| {
                    if let Some((_, tiling, stride)) =
                        axes.iter().find(|(index, _, _)| *index == axis)
                    {
                        let coordinate = (u32::from(tile) / *stride) % u32::from(tiling.partitions);
                        let blocks = padded.0[axis] / tiling.block_size;
                        let partitions = u32::from(tiling.partitions);
                        let short_size = blocks / partitions;
                        let long_shards = blocks % partitions;
                        let start_blocks = coordinate * short_size + coordinate.min(long_shards);
                        let shard_blocks = short_size + u32::from(coordinate < long_shards);
                        let start = start_blocks * tiling.block_size;
                        let end = (start + shard_blocks * tiling.block_size)
                            .min(shape.0[axis])
                            .max(start);
                        (start, end)
                    } else {
                        (0, shape.0[axis])
                    }
                })
                .collect()
        })
        .collect::<Vec<_>>()
        .into()
}

fn intersect_ranges(left: &[(u32, u32)], right: &[(u32, u32)]) -> Option<Vec<(u32, u32)>> {
    if left.len() != right.len() {
        return None;
    }
    left.iter()
        .zip(right)
        .map(|(&(left_start, left_end), &(right_start, right_end))| {
            let start = left_start.max(right_start);
            let end = left_end.min(right_end);
            (start < end).then_some((start, end))
        })
        .collect()
}

fn range_elements(extents: &[(u32, u32)]) -> u64 {
    extents.iter().fold(1u64, |elements, &(start, end)| {
        elements.saturating_mul(u64::from(end - start))
    })
}

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

pub(crate) fn tensor_memory(tensor: &TensorType) -> MemoryUsage {
    let mut usage = MemoryUsage::default();
    usage.add_class(
        tensor.format.layout.memory_class,
        maximum_shard_bytes(tensor),
    );
    usage
}

#[derive(Clone, Copy, Debug, Default)]
struct AllocationRequirement {
    access_tail: u64,
    distinct_element: bool,
}

fn allocation_memory(tensor: &TensorType, requirement: AllocationRequirement) -> MemoryUsage {
    let mut bytes = maximum_shard_bytes(tensor).saturating_add(requirement.access_tail);
    if requirement.distinct_element {
        let element = match tensor.format.layout.memory_class {
            MemoryClass::Ipu21Standard => ipu_package::TILE_MEMORY_ELEMENT_SIZE,
            MemoryClass::Ipu21Interleaved => ipu_package::IPU21_INTERLEAVED_ELEMENT_SIZE,
        };
        bytes = bytes.div_ceil(u64::from(element)) * u64::from(element);
    }
    let mut usage = MemoryUsage::default();
    usage.add_class(tensor.format.layout.memory_class, bytes);
    usage
}

fn allocation_requirements(
    operations: &[MidOperation],
) -> BTreeMap<MidValueId, AllocationRequirement> {
    let mut requirements = BTreeMap::<MidValueId, AllocationRequirement>::new();
    for operation in operations {
        if let Some(plan) = &operation.operator_plan {
            for (&id, operand) in operation.inputs.iter().zip(&plan.requirements.inputs) {
                let requirement = requirements.entry(id).or_default();
                requirement.access_tail = requirement
                    .access_tail
                    .max(u64::from(operand.access_tail_bytes));
            }
            if let Some(&id) = operation.results.first() {
                let requirement = requirements.entry(id).or_default();
                requirement.access_tail = requirement
                    .access_tail
                    .max(u64::from(plan.requirements.output.access_tail_bytes));
            }
            for relation in &plan.requirements.memory_relations {
                let MemoryRelation::DistinctElements(operands) = relation;
                for operand in operands {
                    let id = match operand {
                        MemoryOperand::Output => operation.results.first().copied(),
                        MemoryOperand::Input(index) => {
                            operation.inputs.get(usize::from(*index)).copied()
                        }
                    };
                    if let Some(id) = id {
                        requirements.entry(id).or_default().distinct_element = true;
                    }
                }
            }
        }
        if let Some(plan) = &operation.conversion_plan {
            if let Some(&id) = operation.inputs.first() {
                let requirement = requirements.entry(id).or_default();
                requirement.access_tail = requirement
                    .access_tail
                    .max(u64::from(plan.input.access_tail_bytes));
            }
            if let Some(&id) = operation.results.first() {
                let requirement = requirements.entry(id).or_default();
                requirement.access_tail = requirement
                    .access_tail
                    .max(u64::from(plan.output.access_tail_bytes));
            }
        }
    }
    requirements
}

fn value_allocation(
    id: MidValueId,
    values: &[MidValue],
    requirements: &BTreeMap<MidValueId, AllocationRequirement>,
) -> MemoryUsage {
    allocation_memory(
        &values[id.index() as usize].tensor_type,
        requirements.get(&id).copied().unwrap_or_default(),
    )
}

fn maximum_standard_allocation(
    ids: &BTreeSet<MidValueId>,
    values: &[MidValue],
    requirements: &BTreeMap<MidValueId, AllocationRequirement>,
) -> u64 {
    ids.iter()
        .map(|&id| value_allocation(id, values, requirements).standard)
        .max()
        .unwrap_or(0)
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
        && gemm_uses_panel_buffer(dispatch, right, output)
    {
        // Each local output-column panel has one final kernel buffer reused
        // across K phases. Remote bytes can be exchanged directly into it.
        let output_columns =
            maximum_axis_shard_extent(output, output.shape.0.len().saturating_sub(1));
        let panels = output_columns.div_ceil(u64::from(*output_column_block));
        temporary.interleaved = panels
            .saturating_mul(u64::from(*inner_block))
            .saturating_mul(u64::from(*output_column_block))
            .saturating_mul(right.format.precision.bytes());
    }
    MemoryEstimate {
        live,
        temporary,
        peak: live.saturating_add(temporary),
    }
}

pub(crate) fn gemm_uses_panel_buffer(
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

pub(crate) fn gemm_requires_panel_repacking(
    dispatch: &OperatorDispatch,
    right: &TensorType,
    output: &TensorType,
) -> bool {
    gemm_uses_panel_buffer(dispatch, right, output)
        && right.format.layout.order != ElementOrder::Amp(AmpOrder::RightK64)
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
        IPU21_TARGET_COSTS.exchange_bytes_per_cycle * IPU21_TARGET_COSTS.exchange_bus_sharing
    } else {
        IPU21_TARGET_COSTS.exchange_bytes_per_cycle
    }
}

pub(crate) fn gemm_exchange_phase_count(
    dispatch: &OperatorDispatch,
    inputs: &[TensorType],
    output: &TensorType,
) -> u64 {
    if gemm_remote_bytes_per_tile(inputs, output) == 0 {
        return 0;
    }
    let OperatorDispatch::BlockedGemm { inner_block, .. } = dispatch else {
        return 0;
    };
    let Some(left) = inputs.first() else {
        return 0;
    };
    let Some(&inner) = left.shape.0.last() else {
        return 0;
    };
    u64::from(inner).div_ceil(u64::from(*inner_block))
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
    let requirements = allocation_requirements(operations);
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
            live = live.saturating_add(value_allocation(*id, values, &requirements));
        }
    }
    let mut peaks = MemoryPeaks::default();
    peaks.observe(
        live,
        maximum_standard_allocation(&live_values, values, &requirements),
    );
    for operation in operations {
        let mut during = live.saturating_add(operation.memory.temporary);
        let mut during_values = live_values.clone();
        for result in &operation.results {
            if !live_values.contains(result) {
                during = during.saturating_add(value_allocation(*result, values, &requirements));
                during_values.insert(*result);
            }
        }
        peaks.observe(
            during,
            maximum_standard_allocation(&during_values, values, &requirements),
        );
        for input in &operation.inputs {
            if let Some(remaining) = uses.get_mut(input) {
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 && live_values.remove(input) {
                    let bytes = value_allocation(*input, values, &requirements);
                    live.standard = live.standard.saturating_sub(bytes.standard);
                    live.interleaved = live.interleaved.saturating_sub(bytes.interleaved);
                }
            }
        }
        for result in &operation.results {
            if uses.get(result).copied().unwrap_or(0) != 0 && live_values.insert(*result) {
                live = live.saturating_add(value_allocation(*result, values, &requirements));
            }
        }
    }
    peaks.observe(
        live,
        maximum_standard_allocation(&live_values, values, &requirements),
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn randomized_conversion_traffic_counts_fragmented_multicasts() {
        let mut random = fastrand::Rng::with_seed(0x6672_6167_6d65_6e74);
        for case in 0..32 {
            let row_partitions = 1_u16 << random.u32(1..=4);
            let column_partitions = 1_u16 << random.u32(1..=4);
            let tiles = row_partitions * column_partitions;
            let rows = u32::from(row_partitions.max(column_partitions)) * random.u32(1..=4);
            let columns = u32::from(row_partitions.max(column_partitions)) * random.u32(1..=4) * 64;
            let shape = TensorShape(vec![rows, columns]);
            let fragmented_source =
                Layout::amp_output_grid(tiles, row_partitions, column_partitions);
            let aligned_source = Layout::amp_output_grid(tiles, column_partitions, row_partitions);
            let destination =
                Layout::amp_output_replicated_grid(tiles, column_partitions, row_partitions);
            let fragmented =
                conversion_traffic(&shape, Precision::F16, &fragmented_source, &destination)
                    .unwrap();
            let aligned =
                conversion_traffic(&shape, Precision::F16, &aligned_source, &destination).unwrap();

            assert_eq!(
                fragmented.maximum_destination_bytes, aligned.maximum_destination_bytes,
                "case {case}"
            );
            assert!(
                fragmented.remote_fragments >= aligned.remote_fragments,
                "case {case}: {fragmented:?} {aligned:?}"
            );
            assert!(
                fragmented.maximum_intersections >= aligned.maximum_intersections,
                "case {case}: {fragmented:?} {aligned:?}"
            );
        }
    }
}
