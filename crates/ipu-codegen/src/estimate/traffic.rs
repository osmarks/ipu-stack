//! Address-independent exchange traffic and GEMM communication geometry.

use super::*;
use crate::mid::{ResolvedAxis, ResolvedLayout};

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

pub(super) fn add_endpoint_load(
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

pub(super) fn layout_extents(
    shape: &TensorShape,
    layout: &Layout,
) -> Option<Vec<(u16, Vec<(u32, u32)>)>> {
    Some(
        layout
            .shard_extents(shape)
            .ok()?
            .into_iter()
            .map(|(tile, extents)| {
                (
                    tile,
                    extents
                        .into_iter()
                        .map(|extent| (extent.start, extent.logical_end))
                        .collect(),
                )
            })
            .collect(),
    )
}

pub(super) fn intersect_ranges(
    left: &[(u32, u32)],
    right: &[(u32, u32)],
) -> Option<Vec<(u32, u32)>> {
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

pub(super) fn range_elements(extents: &[(u32, u32)]) -> u64 {
    extents.iter().fold(1u64, |elements, &(start, end)| {
        elements.saturating_mul(u64::from(end - start))
    })
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
    let Some(left) = inputs.get(orientation.operand_indices().0) else {
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
    let (left_row_axis, left_inner_axis) = orientation.matrix_axes(left_rank);
    let (right_inner_axis, right_column_axis) = orientation.matrix_axes(right_rank);
    let (output_row_axis, output_column_axis) = orientation.matrix_axes(output_rank);
    let output_layout = compute_output
        .format
        .layout
        .resolve(&compute_output.shape)
        .ok()?;
    let output_plans = output_layout.axes()?;
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
        let left_remote = left_plan.remote_bytes(tile, output_plans)?;
        let right_remote = right_plan.remote_bytes(tile, output_plans)?;
        left_is_remote |= left_remote != 0;
        right_is_remote |= right_remote != 0;
        let incoming = left_remote.saturating_add(right_remote);
        traffic.add_incoming(tile, incoming, incoming.div_ceil(transfer_bytes));
    }
    add_operand_outgoing_bus_work(&mut traffic, left, left_is_remote);
    add_operand_outgoing_bus_work(&mut traffic, right, right_is_remote);
    Some(traffic)
}

pub(super) fn parallel_gemm_operand_traffic(
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
pub(super) fn replica_shortfall_traffic(
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
    operand_layout: ResolvedLayout,
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
        let operand_layout = operand.format.layout.resolve(&operand.shape).ok()?;
        operand_layout.axes()?;
        Some(Self {
            operand,
            operand_layout,
            operand_spatial_axis,
            operand_inner_axis,
            output_spatial_axis,
            rank_offset,
        })
    }

    fn remote_bytes(&self, tile: u16, output_plans: &[ResolvedAxis]) -> Option<u64> {
        let mut required_elements = 1u64;
        let mut local_elements = 1u64;
        for axis in 0..self.operand.shape.0.len() {
            let extent = self.operand.shape.0[axis];
            let required = if axis == self.operand_spatial_axis {
                clipped_range(output_plans[self.output_spatial_axis].range(tile), extent)
            } else if axis == self.operand_inner_axis {
                0..extent
            } else if extent == 1 {
                0..1
            } else {
                let output_axis = axis.checked_add(self.rank_offset)?;
                clipped_range(output_plans[output_axis].range(tile), extent)
            };
            required_elements = required_elements
                .saturating_mul(u64::from(required.end.saturating_sub(required.start)));
            let local_length = if tile < self.operand.format.layout.tiling.tile_count {
                let local = clipped_range(self.operand_layout.axes()?[axis].range(tile), extent);
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

pub(super) fn clipped_range(range: std::ops::Range<u32>, extent: u32) -> std::ops::Range<u32> {
    range.start.min(extent)..range.end.min(extent)
}

pub(super) fn add_operand_outgoing_bus_work(
    traffic: &mut ExchangeEndpointTraffic,
    operand: &TensorType,
    remote: bool,
) {
    if !remote {
        return;
    }
    let transfer_bytes = u64::from(ipu_exchange::MAX_TRANSFER_WORDS) * 4;
    let bytes = maximum_shard_bytes(operand);
    for tile in 0..operand.format.layout.tiling.tile_count {
        traffic.add_outgoing(tile / 2, bytes, bytes.div_ceil(transfer_bytes));
    }
}
