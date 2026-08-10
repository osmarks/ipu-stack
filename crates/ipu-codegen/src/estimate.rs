//! Memory, communication, and capacity estimates shared by planning policies.

use crate::cost::IPU21_TARGET_COSTS;
use crate::graph::TensorShape;
use crate::mid::{
    AmpOrder, ElementOrder, Layout, MemoryClass, MemoryEstimate, MemoryOperand, MemoryPeaks,
    MemoryRelation, MemoryUsage, MidOperation, MidOperationKind, MidValue, MidValueId,
    OperandMaterialization, OperatorDispatch, OperatorRequirements, Precision, TensorAxis,
    TensorType,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ConversionTraffic {
    pub source_payload_bytes: u64,
    pub remote_fragments: u64,
    pub maximum_routed_fragments: u64,
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
    let mut source_groups = HashMap::<Vec<(u32, u32)>, Vec<u16>>::new();
    for (tile, extents) in sources.into_iter().enumerate() {
        source_groups
            .entry(extents)
            .or_default()
            .push(u16::try_from(tile).ok()?);
    }
    let mut destination_groups = HashMap::<Vec<(u32, u32)>, Vec<u16>>::new();
    for (tile, extents) in destinations.into_iter().enumerate() {
        destination_groups
            .entry(extents)
            .or_default()
            .push(u16::try_from(tile).ok()?);
    }
    let mut remote = HashSet::<(u16, Vec<(u32, u32)>)>::new();
    let mut traffic = ConversionTraffic::default();
    for (destination, destination_tiles) in &destination_groups {
        let mut intersections = Vec::with_capacity(source_groups.len());
        for (source, source_tiles) in &source_groups {
            let Some(extents) = intersect_ranges(source, destination) else {
                continue;
            };
            intersections.push((extents, source_tiles));
        }
        let mut destination_bytes = 0u64;
        let mut local_bytes = 0u64;
        let mut local_intersections = 0u64;
        for (extents, source_tiles) in &intersections {
            let bytes = range_elements(extents).saturating_mul(element_bytes);
            destination_bytes = destination_bytes.saturating_add(bytes);
            if destination_tiles
                .iter()
                .any(|tile| source_tiles.binary_search(tile).is_ok())
            {
                local_bytes = local_bytes.max(bytes);
                local_intersections = 1;
            }
            if destination_tiles
                .iter()
                .any(|tile| source_tiles.binary_search(tile).is_err())
            {
                remote.insert((source_tiles[0], extents.clone()));
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
    traffic.maximum_routed_fragments = if from.order == to.order {
        traffic.maximum_intersections
    } else {
        traffic
            .maximum_destination_bytes
            .saturating_sub(traffic.maximum_local_bytes)
            .div_ceil(4)
    };
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
    let Some(plan) = tile_axis_plan(tensor, axis) else {
        return u64::MAX;
    };
    u64::from(
        plan.blocks
            .div_ceil(plan.partitions)
            .saturating_mul(plan.block_size),
    )
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
    requirements: &OperatorRequirements,
    inputs: &[TensorType],
    output: &TensorType,
) -> MemoryEstimate {
    let live = inputs.iter().zip(&requirements.inputs).fold(
        tensor_memory(output),
        |usage, (input, requirement)| {
            if requirement.materialization == OperandMaterialization::DispatchSlices {
                usage
            } else {
                usage.saturating_add(tensor_memory(input))
            }
        },
    );
    let mut temporary = MemoryUsage::default();
    if let (OperatorDispatch::BlockedGemm { inner_block, .. }, Some(left), Some(requirement)) =
        (dispatch, inputs.first(), requirements.inputs.first())
        && requirement.materialization == OperandMaterialization::DispatchSlices
    {
        let inner = left.shape.0.last().copied().map_or(1, u64::from).max(1);
        let bytes = maximum_shard_bytes(left)
            .div_ceil(inner)
            .saturating_mul(u64::from(*inner_block))
            .saturating_add(u64::from(requirement.access_tail_bytes));
        temporary.add_class(left.format.layout.memory_class, bytes);
    }
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
        exchange_row_bytes: 0,
    }
}

pub(crate) fn gemm_uses_panel_buffer(
    dispatch: &OperatorDispatch,
    right: &TensorType,
    output: &TensorType,
) -> bool {
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
    if streamed {
        return true;
    }
    if right.format.layout.memory_class == MemoryClass::Ipu21Interleaved {
        return false;
    }
    let k = right.shape.0[rank - 2];
    let columns = maximum_axis_shard_extent(output, output_rank - 1);
    k > *inner_block && columns > 16
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
        exchange_row_bytes: 0,
    }
}

pub(crate) fn region_peak_memory(
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
) -> MemoryPeaks {
    region_peak_memory_with_multiplicity(initial, operations, outputs, values, &BTreeMap::new())
}

pub(crate) fn region_peak_memory_with_multiplicity(
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
    allocation_multiplicity: &BTreeMap<MidValueId, u32>,
) -> MemoryPeaks {
    let requirements = allocation_requirements(operations);
    let streamed_aliases = operations
        .iter()
        .filter_map(|operation| {
            let plan = operation.conversion_plan.as_ref()?;
            if plan.output.materialization != OperandMaterialization::DispatchSlices {
                return None;
            }
            Some((*operation.results.first()?, *operation.inputs.first()?))
        })
        .collect::<BTreeMap<_, _>>();
    let mut uses = BTreeMap::<MidValueId, u32>::new();
    for input in operations.iter().flat_map(operation_value_inputs) {
        *uses.entry(*input).or_default() += 1;
    }
    for output in outputs {
        *uses.entry(*output).or_default() += 1;
    }
    let mut live_values = BTreeSet::new();
    for id in initial {
        live_values.insert(*id);
    }
    let mut peaks = MemoryPeaks::default();
    let observe = |peaks: &mut MemoryPeaks, ids: &BTreeSet<MidValueId>, temporary: MemoryUsage| {
        let roots = ids
            .iter()
            .map(|id| allocation_root(*id, &streamed_aliases))
            .collect::<BTreeSet<_>>();
        let live = roots.iter().fold(MemoryUsage::default(), |usage, id| {
            let allocation = value_allocation(*id, values, &requirements);
            let copies = u64::from(allocation_multiplicity.get(id).copied().unwrap_or(1));
            usage.saturating_add(MemoryUsage {
                standard: allocation.standard.saturating_mul(copies),
                interleaved: allocation.interleaved.saturating_mul(copies),
            })
        });
        peaks.observe(
            live.saturating_add(temporary),
            maximum_standard_allocation(&roots, values, &requirements),
        );
    };
    observe(&mut peaks, &live_values, MemoryUsage::default());
    for operation in operations {
        let mut during_values = live_values.clone();
        for result in &operation.results {
            during_values.insert(*result);
        }
        observe(&mut peaks, &during_values, operation.memory.temporary);
        for input in operation_value_inputs(operation) {
            if let Some(remaining) = uses.get_mut(input) {
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 {
                    live_values.remove(input);
                }
            }
        }
        for result in &operation.results {
            if uses.get(result).copied().unwrap_or(0) != 0 {
                live_values.insert(*result);
            }
        }
    }
    observe(&mut peaks, &live_values, MemoryUsage::default());
    let exchange_rows = operations
        .iter()
        .map(|operation| operation.memory.exchange_row_bytes)
        .fold(0u64, u64::saturating_add);
    peaks.exchange_rows = exchange_rows;
    peaks.standard = peaks.standard.saturating_add(exchange_rows);
    peaks.total = peaks.total.saturating_add(exchange_rows);
    peaks
}

fn operation_value_inputs(operation: &MidOperation) -> Vec<&MidValueId> {
    let mut inputs = operation.inputs.iter().collect::<Vec<_>>();
    if let MidOperationKind::Repeat(repeat) = &operation.kind {
        inputs.extend(repeat.iterated_inputs.iter().flatten());
    }
    inputs
}

fn allocation_root(mut id: MidValueId, aliases: &BTreeMap<MidValueId, MidValueId>) -> MidValueId {
    while let Some(source) = aliases.get(&id) {
        id = *source;
    }
    id
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
    let Some(output_rows) = tile_axis_plan(output, output_row_axis) else {
        return u64::MAX;
    };
    let Some(output_columns) = tile_axis_plan(output, output_column_axis) else {
        return u64::MAX;
    };
    let Some(left_rows) = tile_axis_plan(left, left_row_axis) else {
        return u64::MAX;
    };
    let Some(right_columns) = tile_axis_plan(right, right_column_axis) else {
        return u64::MAX;
    };
    let Some(right_inner) = tile_axis_plan(right, right_inner_axis) else {
        return u64::MAX;
    };
    let outer_rows = output.shape.0[..output_row_axis]
        .iter()
        .fold(1u64, |elements, &extent| {
            elements.saturating_mul(u64::from(extent))
        });
    (0..tiles).fold(0u64, |maximum, tile| {
        let output_rows = output_rows.range(tile);
        let output_columns = output_columns.range(tile);
        let left_rows = left_rows.range(tile);
        let right_columns = right_columns.range(tile);
        let right_inner = right_inner.range(tile);
        let required_left_elements = outer_rows
            .saturating_mul(u64::from(output_rows.end - output_rows.start))
            .saturating_mul(u64::from(k));
        let local_left_rows = overlap_length(&output_rows, &left_rows);
        let local_left_elements = outer_rows
            .saturating_mul(u64::from(local_left_rows))
            .saturating_mul(u64::from(k));
        let left_remote = required_left_elements
            .saturating_sub(local_left_elements)
            .saturating_mul(left.format.precision.bytes());

        let required_right_elements =
            u64::from(output_columns.end - output_columns.start).saturating_mul(u64::from(k));
        let local_right_elements = if tile < right.format.layout.tiling.tile_count {
            u64::from(overlap_length(&output_columns, &right_columns))
                .saturating_mul(u64::from(overlap_length(&(0..k), &right_inner)))
        } else {
            0
        };
        let right_remote = required_right_elements
            .saturating_sub(local_right_elements)
            .saturating_mul(right.format.precision.bytes());
        maximum.max(left_remote.saturating_add(right_remote))
    })
}

fn overlap_length(left: &std::ops::Range<u32>, right: &std::ops::Range<u32>) -> u32 {
    left.end
        .min(right.end)
        .saturating_sub(left.start.max(right.start))
}

#[derive(Clone, Copy)]
struct TileAxisPlan {
    blocks: u32,
    block_size: u32,
    partitions: u32,
    stride: u32,
}

impl TileAxisPlan {
    fn range(self, tile: u16) -> std::ops::Range<u32> {
        let coordinate = (u32::from(tile) / self.stride) % self.partitions;
        let short = self.blocks / self.partitions;
        let long = self.blocks % self.partitions;
        let start_blocks = coordinate * short + coordinate.min(long);
        let shard_blocks = short + u32::from(coordinate < long);
        start_blocks * self.block_size..(start_blocks + shard_blocks) * self.block_size
    }
}

fn tile_axis_plan(tensor: &TensorType, axis: usize) -> Option<TileAxisPlan> {
    let layout = &tensor.format.layout;
    let padded = layout.padded_shape(&tensor.shape).ok()?;
    let Some((tiling, stride)) = layout
        .tiling
        .axes
        .iter()
        .zip(layout.tiling.axis_strides().ok()?)
        .find(|(tiling, _)| tiling.axis.resolve(padded.0.len()) == Ok(axis))
    else {
        return Some(TileAxisPlan {
            blocks: 1,
            block_size: padded.0[axis],
            partitions: 1,
            stride: 1,
        });
    };
    Some(TileAxisPlan {
        blocks: padded.0[axis] / tiling.block_size,
        block_size: tiling.block_size,
        partitions: u32::from(tiling.partitions),
        stride,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conversion_traffic_reference(
        shape: &TensorShape,
        precision: Precision,
        from: &Layout,
        to: &Layout,
    ) -> ConversionTraffic {
        let sources = layout_extents(shape, from).unwrap();
        let destinations = layout_extents(shape, to).unwrap();
        let mut remote = BTreeSet::<(u16, Vec<(u32, u32)>)>::new();
        let mut traffic = ConversionTraffic::default();
        for (destination_tile, destination) in destinations.iter().enumerate() {
            let mut intersections = BTreeMap::<Vec<(u32, u32)>, u16>::new();
            for (source_tile, source) in sources.iter().enumerate() {
                let Some(extents) = intersect_ranges(source, destination) else {
                    continue;
                };
                let source_tile = source_tile as u16;
                let selected = intersections.entry(extents).or_insert(source_tile);
                if usize::from(source_tile) == destination_tile {
                    *selected = source_tile;
                }
            }
            let mut destination_bytes = 0;
            let mut local_bytes = 0;
            let mut local_intersections = 0;
            for (extents, source_tile) in &intersections {
                let bytes = range_elements(extents) * precision.bytes();
                destination_bytes += bytes;
                if usize::from(*source_tile) == destination_tile {
                    local_bytes += bytes;
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
        traffic.maximum_routed_fragments = if from.order == to.order {
            traffic.maximum_intersections
        } else {
            traffic
                .maximum_destination_bytes
                .saturating_sub(traffic.maximum_local_bytes)
                .div_ceil(4)
        };
        traffic.source_payload_bytes = remote
            .iter()
            .map(|(_, extents)| range_elements(extents) * precision.bytes())
            .sum();
        traffic
    }

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
                Layout::amp_output_grid(64, tiles, row_partitions, column_partitions);
            let aligned_source =
                Layout::amp_output_grid(64, tiles, column_partitions, row_partitions);
            let destination =
                Layout::amp_output_replicated_grid(tiles, column_partitions, row_partitions);
            let fragmented =
                conversion_traffic(&shape, Precision::F16, &fragmented_source, &destination)
                    .unwrap();
            let aligned =
                conversion_traffic(&shape, Precision::F16, &aligned_source, &destination).unwrap();

            assert_eq!(
                fragmented,
                conversion_traffic_reference(
                    &shape,
                    Precision::F16,
                    &fragmented_source,
                    &destination,
                ),
                "case {case}"
            );
            assert_eq!(
                aligned,
                conversion_traffic_reference(&shape, Precision::F16, &aligned_source, &destination,),
                "case {case}"
            );

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
