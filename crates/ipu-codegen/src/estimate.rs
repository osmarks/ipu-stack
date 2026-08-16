//! Memory, communication, and capacity estimates shared by planning policies.

use crate::graph::TensorShape;
use crate::mid::{
    AMP_COLUMN_MICRO, AMP_INNER_BLOCK, ElementOrder, GemmDistribution, Layout, MemoryClass,
    MemoryEstimate, MemoryOperand, MemoryPeaks, MemoryRelation, MemoryUsage, MidOperation,
    MidOperationKind, MidValue, MidValueId, OperandMaterialization, OperatorDispatch,
    OperatorRequirements, Precision, TensorAxis, TensorType,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ConversionTraffic {
    pub source_payload_bytes: u64,
    pub maximum_source_payload_bytes: u64,
    pub remote_fragments: u64,
    pub maximum_source_fragments: u64,
    pub maximum_source_bus_payload_bytes: u64,
    pub maximum_source_bus_fragments: u64,
    pub maximum_routed_fragments: u64,
    pub maximum_destination_bytes: u64,
    pub maximum_remote_destination_bytes: u64,
    pub maximum_remote_destination_fragments: u64,
    pub maximum_local_bytes: u64,
    pub maximum_intersections: u64,
    pub maximum_local_intersections: u64,
    pub source_bus_loads: Vec<ExchangeEndpointLoad>,
    pub remote_destination_loads: Vec<ExchangeEndpointLoad>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExchangeEndpointLoad {
    pub bytes: u64,
    pub fragments: u64,
}

impl ExchangeEndpointLoad {
    fn add(&mut self, bytes: u64, fragments: u64) {
        self.bytes = self.bytes.saturating_add(bytes);
        self.fragments = self.fragments.saturating_add(fragments);
    }
}

/// Resource-indexed work for one or more transfers which share an exchange
/// phase. Sends from an adjacent tile pair occupy one shared bus; receives are
/// independent per tile. Keeping those roles separate allows independently
/// produced traffic estimates to be combined before finding the bottleneck.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExchangeEndpointTraffic {
    pub outgoing_buses: Vec<ExchangeEndpointLoad>,
    pub incoming_tiles: Vec<ExchangeEndpointLoad>,
}

impl ExchangeEndpointTraffic {
    pub(crate) fn from_conversion(traffic: &ConversionTraffic) -> Self {
        Self {
            outgoing_buses: traffic.source_bus_loads.clone(),
            incoming_tiles: traffic.remote_destination_loads.clone(),
        }
    }

    pub(crate) fn from_maxima(
        outgoing_bytes: u64,
        incoming_bytes: u64,
        outgoing_fragments: u64,
        incoming_fragments: u64,
    ) -> Self {
        let mut traffic = Self::default();
        traffic.add_outgoing(0, outgoing_bytes, outgoing_fragments);
        traffic.add_incoming(0, incoming_bytes, incoming_fragments);
        traffic
    }

    pub(crate) fn add_outgoing(&mut self, bus: u16, bytes: u64, fragments: u64) {
        add_endpoint_load(&mut self.outgoing_buses, bus, bytes, fragments);
    }

    pub(crate) fn add_incoming(&mut self, tile: u16, bytes: u64, fragments: u64) {
        add_endpoint_load(&mut self.incoming_tiles, tile, bytes, fragments);
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        for (bus, load) in other.outgoing_buses.iter().copied().enumerate() {
            self.add_outgoing(bus as u16, load.bytes, load.fragments);
        }
        for (tile, load) in other.incoming_tiles.iter().copied().enumerate() {
            self.add_incoming(tile as u16, load.bytes, load.fragments);
        }
    }

    pub(crate) fn maximum_outgoing_bytes(&self) -> u64 {
        self.outgoing_buses
            .iter()
            .map(|load| load.bytes)
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn maximum_incoming_bytes(&self) -> u64 {
        self.incoming_tiles
            .iter()
            .map(|load| load.bytes)
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn maximum_payload_bytes(&self) -> u64 {
        self.maximum_outgoing_bytes()
            .max(self.maximum_incoming_bytes())
    }

    pub(crate) fn maximum_outgoing_fragments(&self) -> u64 {
        self.outgoing_buses
            .iter()
            .map(|load| load.fragments)
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn maximum_incoming_fragments(&self) -> u64 {
        self.incoming_tiles
            .iter()
            .map(|load| load.fragments)
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn maximum_fragments(&self) -> u64 {
        self.maximum_outgoing_fragments()
            .max(self.maximum_incoming_fragments())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.maximum_payload_bytes() == 0
    }
}

fn add_endpoint_load(
    loads: &mut Vec<ExchangeEndpointLoad>,
    endpoint: u16,
    bytes: u64,
    fragments: u64,
) {
    if bytes == 0 && fragments == 0 {
        return;
    }
    loads.resize(
        loads.len().max(usize::from(endpoint).saturating_add(1)),
        ExchangeEndpointLoad::default(),
    );
    loads[usize::from(endpoint)].add(bytes, fragments);
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
    for (tile, extents) in sources {
        source_groups.entry(extents).or_default().push(tile);
    }
    let mut destination_groups = HashMap::<Vec<(u32, u32)>, Vec<u16>>::new();
    for (tile, extents) in destinations {
        destination_groups.entry(extents).or_default().push(tile);
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
        for (extents, _) in &intersections {
            let bytes = range_elements(extents).saturating_mul(element_bytes);
            destination_bytes = destination_bytes.saturating_add(bytes);
        }
        traffic.maximum_destination_bytes =
            traffic.maximum_destination_bytes.max(destination_bytes);
        traffic.maximum_intersections = traffic
            .maximum_intersections
            .max(intersections.len() as u64);
        for &destination_tile in destination_tiles {
            let mut remote_bytes = 0u64;
            let mut remote_fragments = 0u64;
            let mut local_bytes = 0u64;
            let mut local_intersections = 0u64;
            for (extents, source_tiles) in &intersections {
                let bytes = range_elements(extents).saturating_mul(element_bytes);
                if source_tiles.binary_search(&destination_tile).is_ok() {
                    local_bytes = local_bytes.saturating_add(bytes);
                    local_intersections = local_intersections.saturating_add(1);
                } else {
                    remote_bytes = remote_bytes.saturating_add(bytes);
                    remote_fragments = remote_fragments.saturating_add(1);
                    remote.insert((source_tiles[0], extents.clone()));
                }
            }
            add_endpoint_load(
                &mut traffic.remote_destination_loads,
                destination_tile,
                remote_bytes,
                remote_fragments,
            );
            traffic.maximum_remote_destination_bytes =
                traffic.maximum_remote_destination_bytes.max(remote_bytes);
            traffic.maximum_remote_destination_fragments = traffic
                .maximum_remote_destination_fragments
                .max(remote_fragments);
            traffic.maximum_local_bytes = traffic.maximum_local_bytes.max(local_bytes);
            traffic.maximum_local_intersections =
                traffic.maximum_local_intersections.max(local_intersections);
        }
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
    let mut source_roles = HashMap::<u16, (u64, u64)>::new();
    for (source, extents) in &remote {
        let role = source_roles.entry(*source).or_default();
        role.0 = role
            .0
            .saturating_add(range_elements(extents).saturating_mul(element_bytes));
        role.1 = role.1.saturating_add(1);
    }
    for (bytes, fragments) in source_roles.into_values() {
        traffic.maximum_source_payload_bytes = traffic.maximum_source_payload_bytes.max(bytes);
        traffic.maximum_source_fragments = traffic.maximum_source_fragments.max(fragments);
    }
    let mut source_buses = HashMap::<u16, (u64, u64)>::new();
    for (source, extents) in &remote {
        let role = source_buses.entry(*source / 2).or_default();
        role.0 = role
            .0
            .saturating_add(range_elements(extents).saturating_mul(element_bytes));
        role.1 = role.1.saturating_add(1);
    }
    for (bus, (bytes, fragments)) in source_buses {
        traffic.maximum_source_bus_payload_bytes =
            traffic.maximum_source_bus_payload_bytes.max(bytes);
        traffic.maximum_source_bus_fragments = traffic.maximum_source_bus_fragments.max(fragments);
        add_endpoint_load(&mut traffic.source_bus_loads, bus, bytes, fragments);
    }
    Some(traffic)
}

fn layout_extents(shape: &TensorShape, layout: &Layout) -> Option<Vec<(u16, Vec<(u32, u32)>)>> {
    let padded = layout.padded_shape(shape).ok()?;
    let rank = shape.0.len();
    if let Some(grain) = layout.tiling.linear_grain() {
        let elements = shape.elements();
        let grains = elements / u64::from(grain);
        let tiles = u64::from(layout.tiling.tile_count);
        let width = u64::from(*shape.0.last()?);
        let mut regions = Vec::new();
        for tile in 0..layout.tiling.tile_count {
            let start_grain =
                u64::from(tile) * (grains / tiles) + u64::from(tile).min(grains % tiles);
            let tile_grains = grains / tiles + u64::from(u64::from(tile) < grains % tiles);
            let start = start_grain * u64::from(grain);
            let end = start + tile_grains * u64::from(grain);
            for row in start / width..end.div_ceil(width) {
                let mut coordinates = vec![0u32; rank.saturating_sub(1)];
                let mut remaining = row;
                for axis in (0..rank.saturating_sub(1)).rev() {
                    let extent = u64::from(shape.0[axis]);
                    coordinates[axis] = u32::try_from(remaining % extent).ok()?;
                    remaining /= extent;
                }
                let first = row == start / width;
                let last = row + 1 == end.div_ceil(width);
                let mut extents = coordinates
                    .into_iter()
                    .map(|coordinate| (coordinate, coordinate + 1))
                    .collect::<Vec<_>>();
                extents.push((
                    if first {
                        u32::try_from(start % width).ok()?
                    } else {
                        0
                    },
                    if last && end % width != 0 {
                        u32::try_from(end % width).ok()?
                    } else {
                        u32::try_from(width).ok()?
                    },
                ));
                regions.push((tile, extents));
            }
        }
        return Some(regions);
    }
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
            let extents = (0..rank)
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
                .collect();
            (tile, extents)
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
    let Ok(padded) = layout.padded_shape(shape) else {
        return shape
            .elements()
            .saturating_mul(u64::from(layout.tiling.replicas));
    };
    padded
        .0
        .iter()
        .enumerate()
        .map(|(index, &extent)| {
            layout
                .tiling
                .axes
                .iter()
                .find(|axis| axis.axis.resolve(padded.0.len()) == Ok(index))
                .map_or(u64::from(extent), |axis| {
                    (0..u32::from(axis.partitions))
                        .map(|coordinate| {
                            axis.shard_bounds(extent, shape.0[index], coordinate)
                                .map_or(u64::MAX, |(start, _, end)| u64::from(end - start))
                        })
                        .fold(0_u64, u64::saturating_add)
                })
        })
        .fold(1_u64, u64::saturating_mul)
        .saturating_mul(u64::from(layout.tiling.replicas))
}

pub(crate) fn maximum_shard_bytes(tensor: &TensorType) -> u64 {
    if let Some(grain) = tensor.format.layout.tiling.linear_grain() {
        let grains = tensor.shape.elements().div_ceil(u64::from(grain));
        return grains
            .div_ceil(u64::from(tensor.format.layout.tiling.tile_count))
            .saturating_mul(u64::from(grain))
            .saturating_mul(tensor.format.precision.bytes());
    }
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
                    (0..u32::from(axis.partitions))
                        .filter_map(|coordinate| {
                            axis.shard_bounds(extent, tensor.shape.0[index], coordinate)
                                .ok()
                                .map(|(start, _, end)| u64::from(end - start))
                        })
                        .max()
                        .unwrap_or(u64::MAX)
                })
        })
        .product::<u64>()
        .saturating_mul(tensor.format.precision.bytes())
}

/// Mean physical storage assigned to one active spatial tile. Replicated
/// layouts include each replica in both the total storage and tile count.
pub(crate) fn average_shard_bytes(tensor: &TensorType) -> u64 {
    let tiles = u64::from(tensor.format.layout.tiling.tile_count).max(1);
    physical_elements(&tensor.shape, &tensor.format.layout)
        .saturating_mul(tensor.format.precision.bytes())
        .div_ceil(tiles)
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

pub(crate) fn gemm_partial_tensor(dispatch: &OperatorDispatch, output: &TensorType) -> TensorType {
    let OperatorDispatch::BlockedGemm {
        output_column_block,
        orientation,
        distribution:
            GemmDistribution::ParallelReduction {
                row_partitions,
                column_partitions,
                ..
            },
        ..
    } = dispatch
    else {
        return output.clone();
    };
    TensorType {
        shape: output.shape.clone(),
        format: crate::mid::TensorFormat {
            precision: output.format.precision,
            layout: match (orientation, output.format.layout.order) {
                (
                    crate::GemmOrientation::Normal,
                    crate::ElementOrder::Amp(crate::AmpOrder::Left),
                ) => Layout::amp_left_result_grid(
                    *output_column_block,
                    row_partitions.saturating_mul(*column_partitions),
                    *row_partitions,
                    *column_partitions,
                    crate::mid::GridOrder::ColumnsFast,
                ),
                (
                    crate::GemmOrientation::Swapped,
                    crate::ElementOrder::Amp(crate::AmpOrder::TransposedLeft),
                ) => Layout::amp_transposed_left_result_grid(
                    *output_column_block,
                    row_partitions.saturating_mul(*column_partitions),
                    *row_partitions,
                    *column_partitions,
                    crate::mid::GridOrder::ColumnsFast,
                ),
                (crate::GemmOrientation::Normal, _) => Layout::amp_output_grid(
                    *output_column_block,
                    row_partitions.saturating_mul(*column_partitions),
                    *row_partitions,
                    *column_partitions,
                    crate::mid::GridOrder::ColumnsFast,
                ),
                (crate::GemmOrientation::Swapped, _) => Layout::amp_transposed_output_grid(
                    *output_column_block,
                    row_partitions.saturating_mul(*column_partitions),
                    *row_partitions,
                    *column_partitions,
                    crate::mid::GridOrder::ColumnsFast,
                ),
            },
        },
    }
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
    let mut maximum_standard_temporary_allocation = 0u64;
    if let (
        OperatorDispatch::BlockedGemm {
            orientation,
            distribution:
                GemmDistribution::ParallelReduction {
                    column_partitions,
                    inner_partitions,
                    result_row_partitions,
                    result_column_partitions,
                    reduction_staging,
                    ..
                },
            output_column_block,
            ..
        },
        Some(first),
        Some(second),
    ) = (dispatch, inputs.first(), inputs.get(1))
    {
        let (left, right, left_requirement) = match orientation {
            crate::GemmOrientation::Normal => (first, second, requirements.inputs.first()),
            crate::GemmOrientation::Swapped => (second, first, requirements.inputs.get(1)),
        };
        let right_rank = right.shape.0.len();
        let (right_inner_axis, right_column_axis) = match orientation {
            crate::GemmOrientation::Normal => (right_rank - 2, right_rank - 1),
            crate::GemmOrientation::Swapped => (right_rank - 1, right_rank - 2),
        };
        let inner_blocks = right.shape.0[right_inner_axis].div_ceil(AMP_INNER_BLOCK);
        let column_blocks = right.shape.0[right_column_axis].div_ceil(*output_column_block);
        let right_staging = u64::from(inner_blocks.div_ceil(u32::from(*inner_partitions)))
            .saturating_mul(u64::from(AMP_INNER_BLOCK))
            .saturating_mul(u64::from(
                column_blocks.div_ceil(u32::from(*column_partitions)),
            ))
            .saturating_mul(u64::from(*output_column_block))
            .saturating_mul(right.format.precision.bytes());
        let mut convolution = MemoryUsage::default();
        convolution.add_class(MemoryClass::Ipu21Interleaved, right_staging);
        if left_requirement.is_some_and(|requirement| {
            requirement.materialization == OperandMaterialization::DispatchSlices
        }) {
            let requirement = left_requirement.expect("checked requirement");
            let mut left_staging =
                maximum_shard_bytes(left).saturating_add(u64::from(requirement.access_tail_bytes));
            let left_must_be_distinct = requirements.memory_relations.iter().any(|relation| {
                let MemoryRelation::DistinctElements(operands) = relation;
                operands.contains(&MemoryOperand::Input(match orientation {
                    crate::GemmOrientation::Normal => 0,
                    crate::GemmOrientation::Swapped => 1,
                }))
            });
            if left_must_be_distinct {
                left_staging = left_staging
                    .div_ceil(u64::from(ipu_package::TILE_MEMORY_ELEMENT_SIZE))
                    .saturating_mul(u64::from(ipu_package::TILE_MEMORY_ELEMENT_SIZE));
            }
            convolution.add_class(left.format.layout.memory_class, left_staging);
            if left.format.layout.memory_class == MemoryClass::Ipu21Standard {
                maximum_standard_temporary_allocation =
                    maximum_standard_temporary_allocation.max(left_staging);
            }
        }
        // Compute retains one local partial alongside operand staging. The
        // later reduction ping-pongs an accumulator and result while its
        // staging policy bounds the simultaneously resident remote partials.
        let partial_bytes = maximum_shard_bytes(&gemm_partial_tensor(dispatch, output));
        let reduction_partial_bytes =
            if (*result_row_partitions, *result_column_partitions) != (1, 1) {
                maximum_shard_bytes(output)
            } else {
                partial_bytes
            };
        convolution.interleaved = convolution.interleaved.saturating_add(partial_bytes);
        let staged_remote_partials = match reduction_staging {
            crate::ReductionStaging::Complete => inner_partitions.saturating_sub(1),
            crate::ReductionStaging::Streamed => 1,
        };
        let reduction = MemoryUsage {
            standard: reduction_partial_bytes
                .saturating_mul(u64::from(staged_remote_partials).saturating_add(2)),
            interleaved: partial_bytes,
        };
        temporary = MemoryUsage {
            standard: convolution.standard.max(reduction.standard),
            interleaved: convolution.interleaved.max(reduction.interleaved),
        };
    }
    if let (OperatorDispatch::BlockedGemm { inner_block, .. }, Some(left), Some(requirement)) =
        (dispatch, inputs.first(), requirements.inputs.first())
        && requirement.materialization == OperandMaterialization::DispatchSlices
        && !matches!(
            dispatch,
            OperatorDispatch::BlockedGemm {
                distribution: GemmDistribution::ParallelReduction { .. },
                ..
            }
        )
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
        && !matches!(
            dispatch,
            OperatorDispatch::BlockedGemm {
                distribution: GemmDistribution::ParallelReduction { .. },
                ..
            }
        )
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
    if let OperatorDispatch::BlockedAttention {
        query_block_rows,
        key_block_rows,
        padded_query_dimension,
        padded_value_dimension,
        ..
    } = dispatch
    {
        let element_bytes = inputs.first().map_or(Precision::F16.bytes(), |input| {
            input.format.precision.bytes()
        });
        let key_rows = inputs
            .get(1)
            .and_then(|key| key.shape.0.get(key.shape.0.len().saturating_sub(2)))
            .copied()
            .map_or(1, u64::from);
        let blocks = key_rows.div_ceil(u64::from(*key_block_rows).max(1));
        let panels_per_block = u64::from(
            padded_query_dimension
                .div_ceil(AMP_COLUMN_MICRO)
                .saturating_add(padded_value_dimension.div_ceil(AMP_COLUMN_MICRO)),
        );
        let query_rows = output
            .shape
            .0
            .get(output.shape.0.len().saturating_sub(2))
            .copied()
            .map_or(1, u64::from);
        let query_partitions = query_rows
            .div_ceil(u64::from(*query_block_rows).max(1))
            .max(1);
        let prepared_panels_per_owner = blocks
            .saturating_mul(panels_per_block)
            .div_ceil(query_partitions);
        let panel_bytes = u64::from(*key_block_rows)
            .saturating_mul(u64::from(AMP_COLUMN_MICRO))
            .saturating_mul(element_bytes);
        // Every attention tile retains the current K and V panels. Prepared
        // panels are spread over query-tile owners; each needs both its
        // row-major gather buffer and its packed source until consumption.
        temporary.standard = temporary.standard.saturating_add(
            u64::from(*key_block_rows)
                .saturating_mul(u64::from(
                    padded_query_dimension.saturating_add(*padded_value_dimension),
                ))
                .saturating_mul(element_bytes)
                .saturating_add(
                    prepared_panels_per_owner
                        .saturating_mul(panel_bytes)
                        .saturating_mul(2),
                )
                .saturating_add(
                    u64::from(*query_block_rows)
                        .saturating_mul(u64::from(key_block_rows.saturating_add(16)))
                        .saturating_mul(element_bytes),
                ),
        );
        temporary.interleaved = temporary.interleaved.saturating_add(
            u64::from(*query_block_rows)
                .saturating_mul(u64::from((*padded_value_dimension).max(*key_block_rows)))
                .saturating_mul(Precision::F32.bytes()),
        );
    }
    if let OperatorDispatch::MaterializedAttention {
        query_block_rows,
        padded_key_rows,
        padded_query_dimension,
        padded_value_dimension,
        ..
    } = dispatch
    {
        let element_bytes = inputs.first().map_or(Precision::F16.bytes(), |input| {
            input.format.precision.bytes()
        });
        let key_rows = inputs
            .get(1)
            .and_then(|key| key.shape.0.get(key.shape.0.len().saturating_sub(2)))
            .copied()
            .map_or(1, u64::from);
        let blocks = key_rows.div_ceil(u64::from(AMP_INNER_BLOCK));
        let panels_per_block = u64::from(
            padded_query_dimension
                .div_ceil(AMP_COLUMN_MICRO)
                .saturating_add(padded_value_dimension.div_ceil(AMP_COLUMN_MICRO)),
        );
        let query_rows = output
            .shape
            .0
            .get(output.shape.0.len().saturating_sub(2))
            .copied()
            .map_or(1, u64::from);
        let query_partitions = query_rows
            .div_ceil(u64::from(*query_block_rows).max(1))
            .max(1);
        let prepared_panels_per_owner = blocks
            .saturating_mul(panels_per_block)
            .div_ceil(query_partitions);
        let panel_bytes = u64::from(AMP_INNER_BLOCK)
            .saturating_mul(u64::from(AMP_COLUMN_MICRO))
            .saturating_mul(element_bytes);
        let operand_staging = u64::from(*padded_key_rows)
            .saturating_mul(u64::from(
                (*padded_query_dimension).max(*padded_value_dimension),
            ))
            .saturating_mul(element_bytes);
        maximum_standard_temporary_allocation =
            maximum_standard_temporary_allocation.max(operand_staging);
        let probability_state = u64::from(*query_block_rows)
            .saturating_mul(u64::from(padded_key_rows + AMP_COLUMN_MICRO))
            .saturating_mul(element_bytes);
        temporary.standard = temporary.standard.saturating_add(
            operand_staging
                .saturating_add(
                    prepared_panels_per_owner
                        .saturating_mul(panel_bytes)
                        .saturating_mul(2),
                )
                .saturating_add(
                    (probability_state > operand_staging)
                        .then_some(probability_state)
                        .unwrap_or(0),
                ),
        );
        temporary.interleaved = temporary.interleaved.saturating_add(
            u64::from(*query_block_rows)
                .saturating_mul(u64::from((*padded_value_dimension).max(*padded_key_rows)))
                .saturating_mul(element_bytes),
        );
    }
    MemoryEstimate {
        live,
        temporary,
        peak: live.saturating_add(temporary),
        exchange_row_bytes: 0,
        maximum_standard_temporary_allocation,
    }
}

pub(crate) fn gemm_uses_panel_buffer(
    dispatch: &OperatorDispatch,
    right: &TensorType,
    output: &TensorType,
) -> bool {
    let OperatorDispatch::BlockedGemm {
        inner_block,
        orientation,
        ..
    } = dispatch
    else {
        return false;
    };
    let rank = right.shape.0.len();
    let output_rank = output.shape.0.len();
    if rank < 2 || output_rank < 2 {
        return true;
    }
    let streamed = right.format.layout.tiling.axes.iter().any(|axis| {
        axis.axis
            == match orientation {
                crate::GemmOrientation::Normal => TensorAxis::FromEnd(2),
                crate::GemmOrientation::Swapped => TensorAxis::FromEnd(1),
            }
            && axis.partitions > 1
    });
    if streamed {
        return true;
    }
    if right.format.layout.memory_class == MemoryClass::Ipu21Interleaved {
        return false;
    }
    let k = right.shape.0[rank
        - match orientation {
            crate::GemmOrientation::Normal => 2,
            crate::GemmOrientation::Swapped => 1,
        }];
    let columns = maximum_axis_shard_extent(
        output,
        output_rank
            - match orientation {
                crate::GemmOrientation::Normal => 1,
                crate::GemmOrientation::Swapped => 2,
            },
    );
    k > *inner_block && columns > 16
}

pub(crate) fn gemm_requires_panel_repacking(
    dispatch: &OperatorDispatch,
    right: &TensorType,
    output: &TensorType,
) -> bool {
    gemm_uses_panel_buffer(dispatch, right, output)
        && !matches!(right.format.layout.order, ElementOrder::BlockMajor(_))
}

pub(crate) fn gemm_exchange_phase_count(
    dispatch: &OperatorDispatch,
    inputs: &[TensorType],
    _output: &TensorType,
) -> u64 {
    let OperatorDispatch::BlockedGemm {
        inner_block,
        orientation,
        ..
    } = dispatch
    else {
        return 0;
    };
    let Some(left) = inputs.get(match orientation {
        crate::GemmOrientation::Normal => 0,
        crate::GemmOrientation::Swapped => 1,
    }) else {
        return 0;
    };
    let Some(&inner) = left.shape.0.get(
        left.shape.0.len()
            - match orientation {
                crate::GemmOrientation::Normal => 1,
                crate::GemmOrientation::Swapped => 2,
            },
    ) else {
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
        maximum_standard_temporary_allocation: 0,
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
        let roots = during_values
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
            live.saturating_add(operation.memory.temporary),
            maximum_standard_allocation(&roots, values, &requirements)
                .max(operation.memory.maximum_standard_temporary_allocation),
        );
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

pub(crate) fn gemm_exchange_endpoint_traffic(
    dispatch: &OperatorDispatch,
    inputs: &[TensorType],
    compute_output: &TensorType,
) -> Option<ExchangeEndpointTraffic> {
    let OperatorDispatch::BlockedGemm {
        orientation,
        distribution,
        ..
    } = dispatch
    else {
        return Some(ExchangeEndpointTraffic::default());
    };
    let [first, second] = inputs else {
        return None;
    };
    let (left, right) = match orientation {
        crate::GemmOrientation::Normal => (first, second),
        crate::GemmOrientation::Swapped => (second, first),
    };
    if let GemmDistribution::ParallelReduction {
        row_partitions,
        column_partitions,
        ..
    } = distribution
    {
        // The parallel dispatch grid contains a K axis which is deliberately
        // absent from `compute_output`: every K group produces a partial with
        // the same logical output extent. Consequently, matching operand and
        // partial-output tile numbers cannot determine locality. The physical
        // left operand is replicated across output-column groups and the
        // physical right operand across output-row groups; only a shortfall in
        // those explicit replica counts creates operator-internal traffic.
        return Some(parallel_gemm_operand_traffic(
            left,
            *column_partitions,
            right,
            *row_partitions,
        ));
    }
    let left_rank = left.shape.0.len();
    let right_rank = right.shape.0.len();
    let output_rank = compute_output.shape.0.len();
    if left_rank < 2 || right_rank < 2 || output_rank < 2 {
        return None;
    }
    let (left_row_axis, left_inner_axis, right_inner_axis, right_column_axis) = match orientation {
        crate::GemmOrientation::Normal => {
            (left_rank - 2, left_rank - 1, right_rank - 2, right_rank - 1)
        }
        crate::GemmOrientation::Swapped => {
            (left_rank - 1, left_rank - 2, right_rank - 1, right_rank - 2)
        }
    };
    let (output_row_axis, output_column_axis) = match orientation {
        crate::GemmOrientation::Normal => (output_rank - 2, output_rank - 1),
        crate::GemmOrientation::Swapped => (output_rank - 1, output_rank - 2),
    };
    let output_plans = (0..output_rank)
        .map(|axis| tile_axis_plan(compute_output, axis))
        .collect::<Option<Vec<_>>>()?;
    let left_plan = GemmOperandTrafficPlan::new(
        left,
        compute_output,
        left_row_axis,
        left_inner_axis,
        output_row_axis,
    )?;
    let right_plan = GemmOperandTrafficPlan::new(
        right,
        compute_output,
        right_column_axis,
        right_inner_axis,
        output_column_axis,
    )?;
    let transfer_bytes = u64::from(ipu_exchange::MAX_TRANSFER_WORDS) * 4;
    let mut traffic = ExchangeEndpointTraffic::default();
    let mut left_is_remote = false;
    let mut right_is_remote = false;
    for tile in 0..compute_output.format.layout.tiling.tile_count {
        let left_remote = left_plan.remote_bytes(tile, &output_plans, None)?;
        let right_remote = right_plan.remote_bytes(tile, &output_plans, None)?;
        left_is_remote |= left_remote != 0;
        right_is_remote |= right_remote != 0;
        let incoming = left_remote.saturating_add(right_remote);
        traffic.add_incoming(tile, incoming, incoming.div_ceil(transfer_bytes));
    }
    add_operand_outgoing_bus_work(&mut traffic, left, left_is_remote);
    add_operand_outgoing_bus_work(&mut traffic, right, right_is_remote);
    Some(traffic)
}

fn parallel_gemm_operand_traffic(
    left: &TensorType,
    left_required_replicas: u16,
    right: &TensorType,
    right_required_replicas: u16,
) -> ExchangeEndpointTraffic {
    let mut traffic = ExchangeEndpointTraffic::default();
    for (operand, required_replicas) in [
        (left, left_required_replicas),
        (right, right_required_replicas),
    ] {
        if operand.format.layout.tiling.replicas >= required_replicas {
            continue;
        }
        let stored_replicas = operand.format.layout.tiling.replicas.max(1);
        let base_tiles = operand
            .format
            .layout
            .tiling
            .tile_count
            .checked_div(stored_replicas)
            .unwrap_or(0);
        let Some(tile_count) = base_tiles.checked_mul(required_replicas) else {
            return ExchangeEndpointTraffic::from_maxima(
                u64::MAX / 16,
                u64::MAX / 16,
                u64::MAX / 16,
                u64::MAX / 16,
            );
        };
        let mut consumer_layout = operand.format.layout.clone();
        consumer_layout.tiling.tile_count = tile_count;
        consumer_layout.tiling.replicas = required_replicas;
        let Some(replication) = replica_shortfall_traffic(operand, &consumer_layout) else {
            return ExchangeEndpointTraffic::from_maxima(
                u64::MAX / 16,
                u64::MAX / 16,
                u64::MAX / 16,
                u64::MAX / 16,
            );
        };
        traffic.merge(&replication);
    }
    traffic
}

/// Replicating an otherwise unchanged layout only matches identical shard
/// extents. Account for those roles directly instead of running the general
/// all-pairs layout-intersection algorithm for every GEMM candidate.
fn replica_shortfall_traffic(
    operand: &TensorType,
    consumer_layout: &Layout,
) -> Option<ExchangeEndpointTraffic> {
    let sources = layout_extents(&operand.shape, &operand.format.layout)?;
    let destinations = layout_extents(&operand.shape, consumer_layout)?;
    let mut source_groups = HashMap::<Vec<(u32, u32)>, Vec<u16>>::new();
    for (tile, extents) in sources {
        source_groups.entry(extents).or_default().push(tile);
    }
    let mut destination_groups = HashMap::<Vec<(u32, u32)>, Vec<u16>>::new();
    for (tile, extents) in destinations {
        destination_groups.entry(extents).or_default().push(tile);
    }

    let mut traffic = ExchangeEndpointTraffic::default();
    for (extents, destination_tiles) in destination_groups {
        let source_tiles = source_groups.get(&extents)?;
        let remote_destinations = destination_tiles
            .into_iter()
            .filter(|tile| source_tiles.binary_search(tile).is_err())
            .collect::<Vec<_>>();
        if remote_destinations.is_empty() {
            continue;
        }
        let bytes = range_elements(&extents).saturating_mul(operand.format.precision.bytes());
        traffic.add_outgoing(source_tiles[0] / 2, bytes, 1);
        for tile in remote_destinations {
            traffic.add_incoming(tile, bytes, 1);
        }
    }
    Some(traffic)
}

struct GemmOperandTrafficPlan<'a> {
    operand: &'a TensorType,
    operand_plans: Vec<TileAxisPlan>,
    operand_spatial_axis: usize,
    operand_inner_axis: usize,
    output_spatial_axis: usize,
    rank_offset: usize,
}

impl<'a> GemmOperandTrafficPlan<'a> {
    fn new(
        operand: &'a TensorType,
        output: &TensorType,
        operand_spatial_axis: usize,
        operand_inner_axis: usize,
        output_spatial_axis: usize,
    ) -> Option<Self> {
        let rank_offset = output.shape.0.len().checked_sub(operand.shape.0.len())?;
        let operand_plans = (0..operand.shape.0.len())
            .map(|axis| tile_axis_plan(operand, axis))
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            operand,
            operand_plans,
            operand_spatial_axis,
            operand_inner_axis,
            output_spatial_axis,
            rank_offset,
        })
    }

    fn remote_bytes(
        &self,
        tile: u16,
        output_plans: &[TileAxisPlan],
        inner_plan: Option<TileAxisPlan>,
    ) -> Option<u64> {
        let mut required_elements = 1u64;
        let mut local_elements = 1u64;
        for axis in 0..self.operand.shape.0.len() {
            let extent = self.operand.shape.0[axis];
            let required = if axis == self.operand_spatial_axis {
                clipped_range(output_plans[self.output_spatial_axis].range(tile), extent)
            } else if axis == self.operand_inner_axis {
                inner_plan.map_or(0..extent, |plan| clipped_range(plan.range(tile), extent))
            } else if extent == 1 {
                0..1
            } else {
                let output_axis = axis.checked_add(self.rank_offset)?;
                clipped_range(output_plans[output_axis].range(tile), extent)
            };
            required_elements = required_elements
                .saturating_mul(u64::from(required.end.saturating_sub(required.start)));
            let local_length = if tile < self.operand.format.layout.tiling.tile_count {
                let local = clipped_range(self.operand_plans[axis].range(tile), extent);
                u64::from(
                    required
                        .end
                        .min(local.end)
                        .saturating_sub(required.start.max(local.start)),
                )
            } else {
                0
            };
            local_elements = local_elements.saturating_mul(local_length);
        }
        Some(
            required_elements
                .saturating_sub(local_elements)
                .saturating_mul(self.operand.format.precision.bytes()),
        )
    }
}

fn clipped_range(range: std::ops::Range<u32>, extent: u32) -> std::ops::Range<u32> {
    range.start.min(extent)..range.end.min(extent)
}

fn add_operand_outgoing_bus_work(
    traffic: &mut ExchangeEndpointTraffic,
    operand: &TensorType,
    remote: bool,
) {
    if !remote {
        return;
    }
    let transfer_bytes = u64::from(ipu_exchange::MAX_TRANSFER_WORDS) * 4;
    for tile in 0..operand.format.layout.tiling.tile_count {
        let bytes = maximum_shard_bytes(operand);
        traffic.add_outgoing(tile / 2, bytes, bytes.div_ceil(transfer_bytes));
    }
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

    fn output_stationary_dispatch() -> OperatorDispatch {
        OperatorDispatch::BlockedGemm {
            initialize: crate::TileKernelSpec::Gemm {
                multiply: Precision::F16,
                accumulate: crate::AccumulationPrecision::F16,
                mode: crate::GemmKernelMode::Initialize,
                weights: crate::GemmWeightLoad::Standard,
                inner_block: AMP_INNER_BLOCK,
                output_columns: crate::mid::AMP_OUTPUT_COLUMN_BLOCK,
            },
            accumulate: crate::TileKernelSpec::Gemm {
                multiply: Precision::F16,
                accumulate: crate::AccumulationPrecision::F16,
                mode: crate::GemmKernelMode::Accumulate,
                weights: crate::GemmWeightLoad::Standard,
                inner_block: AMP_INNER_BLOCK,
                output_columns: crate::mid::AMP_OUTPUT_COLUMN_BLOCK,
            },
            inner_block: AMP_INNER_BLOCK,
            output_column_block: crate::mid::AMP_OUTPUT_COLUMN_BLOCK,
            orientation: crate::GemmOrientation::Normal,
            distribution: GemmDistribution::OutputStationary,
        }
    }

    fn parallel_reduction_dispatch(
        row_partitions: u16,
        column_partitions: u16,
        inner_partitions: u16,
    ) -> OperatorDispatch {
        let mut dispatch = output_stationary_dispatch();
        let OperatorDispatch::BlockedGemm { distribution, .. } = &mut dispatch else {
            unreachable!();
        };
        *distribution = GemmDistribution::ParallelReduction {
            row_partitions,
            column_partitions,
            inner_partitions,
            result_row_partitions: 1,
            result_column_partitions: 1,
            reduction_staging: crate::ReductionStaging::Streamed,
        };
        dispatch
    }

    #[test]
    fn randomized_average_shard_storage_covers_spatial_work() {
        let mut random = fastrand::Rng::with_seed(0x7370_6174_6961_6c77);
        for case in 0..32 {
            let row_partitions = 1_u16 << random.u32(0..=4);
            let column_partitions = 1_u16 << random.u32(0..=4);
            let tiles = row_partitions * column_partitions;
            let rows = u32::from(row_partitions) * random.u32(1..=8);
            let columns = u32::from(column_partitions) * 64 * random.u32(1..=4);
            let tensor = TensorType::new(
                [rows, columns],
                Precision::F16,
                Layout::amp_output_grid(
                    64,
                    tiles,
                    row_partitions,
                    column_partitions,
                    crate::mid::GridOrder::ColumnsFast,
                ),
            );
            let total = physical_elements(&tensor.shape, &tensor.format.layout)
                .saturating_mul(tensor.format.precision.bytes());
            let average = average_shard_bytes(&tensor);
            assert!(
                average.saturating_mul(u64::from(tiles)) >= total,
                "case {case}"
            );
            assert!(average <= maximum_shard_bytes(&tensor), "case {case}");
        }
    }

    #[test]
    fn randomized_gemm_endpoint_traffic_tracks_both_exchange_directions() {
        let mut random = fastrand::Rng::with_seed(0x6269_6469_7265_6374);
        for case in 0..32 {
            let row_partitions = 1_u16 << random.u32(1..=3);
            let column_partitions = 1_u16 << random.u32(1..=3);
            let tiles = row_partitions * column_partitions;
            let rows = u32::from(row_partitions) * random.u32(1..=8);
            let inner = AMP_INNER_BLOCK * random.u32(1..=4);
            let columns = u32::from(column_partitions)
                * crate::mid::AMP_OUTPUT_COLUMN_BLOCK
                * random.u32(1..=3);
            let output = TensorType::new(
                [1, rows, columns],
                Precision::F16,
                Layout::amp_output_grid(
                    crate::mid::AMP_OUTPUT_COLUMN_BLOCK,
                    tiles,
                    row_partitions,
                    column_partitions,
                    crate::mid::GridOrder::ColumnsFast,
                ),
            );
            let local_left = TensorType::new(
                [1, rows, inner],
                Precision::F16,
                Layout::amp_left_grid(
                    AMP_INNER_BLOCK as u16,
                    tiles,
                    row_partitions,
                    column_partitions,
                    crate::mid::GridOrder::ColumnsFast,
                ),
            );
            let local_right = TensorType::new(
                [1, inner, columns],
                Precision::F16,
                Layout::block_major_matrix_grid(
                    AMP_INNER_BLOCK as u16,
                    crate::mid::AMP_OUTPUT_COLUMN_BLOCK,
                    tiles,
                    row_partitions,
                    column_partitions,
                    crate::mid::GridOrder::ColumnsFast,
                ),
            );
            let dispatch = output_stationary_dispatch();
            let local =
                gemm_exchange_endpoint_traffic(&dispatch, &[local_left, local_right], &output)
                    .unwrap();
            assert!(local.is_empty(), "case {case}");

            let sharded_left = TensorType::new(
                [1, rows, inner],
                Precision::F16,
                Layout::amp_left(AMP_INNER_BLOCK as u16, row_partitions),
            );
            let sharded_right = TensorType::new(
                [1, inner, columns],
                Precision::F16,
                Layout::block_major_matrix_storage(
                    AMP_INNER_BLOCK as u16,
                    crate::mid::AMP_OUTPUT_COLUMN_BLOCK,
                    column_partitions,
                    1,
                    1,
                    MemoryClass::Ipu21Standard,
                ),
            );
            let remote =
                gemm_exchange_endpoint_traffic(&dispatch, &[sharded_left, sharded_right], &output)
                    .unwrap();
            assert!(remote.maximum_outgoing_bytes() != 0, "case {case}");
            assert!(remote.maximum_incoming_bytes() != 0, "case {case}");
            assert_eq!(
                remote.maximum_payload_bytes(),
                remote
                    .maximum_outgoing_bytes()
                    .max(remote.maximum_incoming_bytes()),
                "case {case}"
            );
        }
    }

    #[test]
    fn randomized_parallel_gemm_traffic_tracks_replica_shortfalls() {
        let mut random = fastrand::Rng::with_seed(0x7265_706c_6963_6173);
        for case in 0..32 {
            let row_partitions = random.u16(2..=5);
            let column_partitions = random.u16(2..=5);
            let inner_partitions = random.u16(2..=5);
            let tiles = row_partitions * column_partitions * inner_partitions;
            let inner_block = 16 * random.u16(1..=4);
            let column_block = 16 * random.u32(1..=4);
            let rows = u32::from(row_partitions) * random.u32(1..=8);
            let inner = u32::from(inner_partitions) * u32::from(inner_block);
            let columns = u32::from(column_partitions) * column_block;
            let left = TensorType::new(
                [1, rows, inner],
                Precision::F16,
                Layout::amp_left_parallel_grid(
                    inner_block,
                    tiles,
                    row_partitions,
                    column_partitions,
                    inner_partitions,
                ),
            );
            let resident_right = TensorType::new(
                [1, inner, columns],
                Precision::F16,
                Layout::block_major_matrix_storage(
                    inner_block,
                    column_block,
                    column_partitions,
                    inner_partitions,
                    row_partitions,
                    MemoryClass::Ipu21Standard,
                ),
            );
            let compute_output = TensorType::new(
                [1, rows, columns],
                Precision::F16,
                Layout::amp_left_result_grid(
                    column_block,
                    row_partitions * column_partitions,
                    row_partitions,
                    column_partitions,
                    crate::mid::GridOrder::ColumnsFast,
                ),
            );
            let dispatch =
                parallel_reduction_dispatch(row_partitions, column_partitions, inner_partitions);
            let resident = gemm_exchange_endpoint_traffic(
                &dispatch,
                &[left.clone(), resident_right],
                &compute_output,
            )
            .unwrap();
            assert!(resident.is_empty(), "case {case}");

            let sharded_right = TensorType::new(
                [1, inner, columns],
                Precision::F16,
                Layout::block_major_matrix_storage(
                    inner_block,
                    column_block,
                    column_partitions,
                    inner_partitions,
                    1,
                    MemoryClass::Ipu21Standard,
                ),
            );
            let expected_incoming = maximum_shard_bytes(&sharded_right);
            let expected_outgoing = expected_incoming.saturating_mul(u64::from(
                sharded_right.format.layout.tiling.tile_count.min(2),
            ));
            let streamed =
                gemm_exchange_endpoint_traffic(&dispatch, &[left, sharded_right], &compute_output)
                    .unwrap();
            assert_eq!(
                streamed.maximum_incoming_bytes(),
                expected_incoming,
                "case {case}"
            );
            assert_eq!(
                streamed.maximum_outgoing_bytes(),
                expected_outgoing,
                "case {case}"
            );
        }
    }

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
        for (destination_tile, destination) in &destinations {
            let mut intersections = BTreeMap::<Vec<(u32, u32)>, u16>::new();
            for (source_tile, source) in &sources {
                let Some(extents) = intersect_ranges(source, destination) else {
                    continue;
                };
                let selected = intersections.entry(extents).or_insert(*source_tile);
                if source_tile == destination_tile {
                    *selected = *source_tile;
                }
            }
            let mut destination_bytes = 0;
            let mut local_bytes = 0;
            let mut local_intersections = 0;
            let mut remote_intersections = 0;
            for (extents, source_tile) in &intersections {
                let bytes = range_elements(extents) * precision.bytes();
                destination_bytes += bytes;
                if source_tile == destination_tile {
                    local_bytes += bytes;
                    local_intersections += 1;
                } else {
                    remote.insert((*source_tile, extents.clone()));
                    remote_intersections += 1;
                }
            }
            traffic.maximum_destination_bytes =
                traffic.maximum_destination_bytes.max(destination_bytes);
            traffic.maximum_remote_destination_bytes = traffic
                .maximum_remote_destination_bytes
                .max(destination_bytes.saturating_sub(local_bytes));
            traffic.maximum_remote_destination_fragments = traffic
                .maximum_remote_destination_fragments
                .max(remote_intersections);
            add_endpoint_load(
                &mut traffic.remote_destination_loads,
                *destination_tile,
                destination_bytes.saturating_sub(local_bytes),
                remote_intersections,
            );
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
        let mut source_roles = BTreeMap::<u16, (u64, u64)>::new();
        for (source, extents) in &remote {
            let role = source_roles.entry(*source).or_default();
            role.0 += range_elements(extents) * precision.bytes();
            role.1 += 1;
        }
        for (bytes, fragments) in source_roles.into_values() {
            traffic.maximum_source_payload_bytes = traffic.maximum_source_payload_bytes.max(bytes);
            traffic.maximum_source_fragments = traffic.maximum_source_fragments.max(fragments);
        }
        let mut source_buses = BTreeMap::<u16, (u64, u64)>::new();
        for (source, extents) in &remote {
            let role = source_buses.entry(*source / 2).or_default();
            role.0 += range_elements(extents) * precision.bytes();
            role.1 += 1;
        }
        for (bus, (bytes, fragments)) in source_buses {
            traffic.maximum_source_bus_payload_bytes =
                traffic.maximum_source_bus_payload_bytes.max(bytes);
            traffic.maximum_source_bus_fragments =
                traffic.maximum_source_bus_fragments.max(fragments);
            add_endpoint_load(&mut traffic.source_bus_loads, bus, bytes, fragments);
        }
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
            let fragmented_source = Layout::amp_output_grid(
                64,
                tiles,
                row_partitions,
                column_partitions,
                crate::mid::GridOrder::ColumnsFast,
            );
            let aligned_source = Layout::amp_output_grid(
                64,
                tiles,
                column_partitions,
                row_partitions,
                crate::mid::GridOrder::ColumnsFast,
            );
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
