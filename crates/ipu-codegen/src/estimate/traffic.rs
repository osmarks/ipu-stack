//! Address-independent exchange traffic and GEMM communication geometry.

use super::*;
use ipu_target::Target;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ConversionTraffic {
    pub maximum_destination_bytes: u64,
    pub maximum_local_bytes: u64,
    pub maximum_intersections: u64,
    pub maximum_local_intersections: u64,
    pub exchange: ExchangeEndpointTraffic,
    /// Effective maximum lane work after eligible double-width multicasts.
    /// Logical traffic above remains unchanged; controls and row sizes are not discounted.
    pub paired_payload_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExchangeEndpointLoad {
    pub bytes: u64,
    pub fragments: u64,
    pub pointer_resets: u64,
}

/// Resource-indexed work for one or more transfers which share an exchange
/// phase. Ordinary sends and receives have independent lanes per tile; only
/// paired transfers borrow a neighbor's transmit lane. Keeping roles separate allows independently
/// produced traffic estimates to be combined before finding the bottleneck.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExchangeEndpointTraffic {
    pub outgoing_lanes: Vec<ExchangeEndpointLoad>,
    pub incoming_tiles: Vec<ExchangeEndpointLoad>,
}

impl ExchangeEndpointTraffic {
    #[cfg(test)]
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

    pub(crate) fn add_outgoing(&mut self, tile: u16, bytes: u64, fragments: u64) {
        add_endpoint_load(&mut self.outgoing_lanes, tile, bytes, fragments, 0);
    }

    pub(crate) fn add_incoming(&mut self, tile: u16, bytes: u64, fragments: u64) {
        self.add_receive(tile, bytes, fragments, fragments);
    }

    pub(crate) fn add_receive(&mut self, tile: u16, bytes: u64, fragments: u64, resets: u64) {
        add_endpoint_load(&mut self.incoming_tiles, tile, bytes, fragments, resets);
    }

    pub(crate) fn maximum_control_cycles(&self, target: Target) -> u64 {
        // TX/RX payload lanes are independent, but share supervisor issue slots.
        (0..self.outgoing_lanes.len().max(self.incoming_tiles.len()))
            .map(|i| {
                self.outgoing_lanes
                    .get(i)
                    .map_or(0, |l| {
                        l.fragments
                            .saturating_mul(target.costs().send_control_cycles)
                    })
                    .saturating_add(self.incoming_tiles.get(i).map_or(0, |l| {
                        l.fragments
                            .saturating_mul(target.costs().receive_control_cycles)
                            .saturating_add(
                                l.pointer_resets
                                    .saturating_mul(target.costs().receive_pointer_cycles),
                            )
                    }))
            })
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn maximum_payload_bytes(&self) -> u64 {
        self.outgoing_lanes
            .iter()
            .chain(&self.incoming_tiles)
            .map(|load| load.bytes)
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn maximum_fragments(&self) -> u64 {
        self.outgoing_lanes
            .iter()
            .chain(&self.incoming_tiles)
            .map(|load| load.fragments)
            .max()
            .unwrap_or(0)
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
    pointer_resets: u64,
) {
    if bytes == 0 && fragments == 0 && pointer_resets == 0 {
        return;
    }
    loads.resize(
        loads.len().max(usize::from(endpoint).saturating_add(1)),
        ExchangeEndpointLoad::default(),
    );
    let load = &mut loads[usize::from(endpoint)];
    load.bytes = load.bytes.saturating_add(bytes);
    load.fragments = load.fragments.saturating_add(fragments);
    load.pointer_resets = load.pointer_resets.saturating_add(pointer_resets);
}

pub(crate) fn conversion_traffic(
    target: Target,
    input: &MidValue,
    output: &MidValue,
    mapping: &crate::CoordinateMapping,
    tile_count: u16,
) -> Option<ConversionTraffic> {
    // Factor views need nonrectangular preimages. Keep the caller's coarse
    // estimate for those rather than treating them as identity mappings.
    if mapping.view.is_some() || input.tensor_type.shape.0.len() != output.tensor_type.shape.0.len()
    {
        return None;
    }
    let groups = |value: &MidValue, offsets: &[u32]| {
        let mut groups = HashMap::<Vec<(u32, u32)>, Vec<u16>>::new();
        for (tile, mut extents) in
            layout_extents(&value.tensor_type.shape, &value.tensor_type.format.layout)?
        {
            for (axis, (start, end)) in extents.iter_mut().enumerate() {
                let offset = offsets.get(axis).copied().unwrap_or(0);
                *start = start.checked_add(offset)?;
                *end = end.checked_add(offset)?;
            }
            let tile = value.owners.tile(tile, tile_count)?;
            groups.entry(extents).or_default().push(tile);
        }
        for tiles in groups.values_mut() {
            tiles.sort_unstable();
        }
        Some(groups)
    };
    let source_groups = groups(input, &[])?;
    let destination_groups = groups(output, &mapping.offsets)?;
    let element_bytes = input.tensor_type.format.precision.bytes();
    let can_pair = mapping.is_identity()
        && input.tensor_type.format.precision == output.tensor_type.format.precision
        && input.tensor_type.format.layout.order == output.tensor_type.format.layout.order
        && [&input.tensor_type, &output.tensor_type].iter().all(|t| {
            t.format
                .layout
                .resolve(&t.shape)
                .ok()
                .is_some_and(|layout| {
                    layout.axes().is_some_and(|axes| {
                        axes.iter().all(|axis| {
                            axis.partitions()
                                .iter()
                                .all(|e| e.logical_end == e.physical_end)
                        })
                    })
                })
        });
    let mut remote = HashMap::<(u16, Vec<(u32, u32)>), (u64, Vec<u16>, bool)>::new();
    let mut local = HashMap::<u16, [u64; 4]>::new();
    let mut traffic = ConversionTraffic::default();
    for (destination, destination_tiles) in &destination_groups {
        let mut intersections = Vec::with_capacity(source_groups.len());
        let mut destination_bytes = 0u64;
        for (source, source_tiles) in &source_groups {
            let Some(extents) = intersect_ranges(source, destination) else {
                continue;
            };
            let bytes = range_elements(&extents).saturating_mul(element_bytes);
            destination_bytes = destination_bytes.saturating_add(bytes);
            let aligned = can_pair
                && bytes.is_multiple_of(8)
                && destination_tiles.len() >= 2
                && paired_copy_aligned(
                    &input.tensor_type,
                    &output.tensor_type,
                    source,
                    destination,
                    &extents,
                );
            intersections.push((extents, source_tiles, bytes, aligned));
        }
        // Coordinates outside the source are zero-filled by copy lowering.
        // They require local work even when every real value arrives remotely.
        let destination_size = range_elements(destination).saturating_mul(element_bytes);
        let zero_bytes = destination_size.saturating_sub(destination_bytes);
        for &destination_tile in destination_tiles {
            let mut remote_bytes = 0u64;
            let mut remote_fragments = 0u64;
            let mut local_bytes = zero_bytes;
            let mut local_intersections = u64::from(zero_bytes != 0);
            for &(ref extents, source_tiles, bytes, aligned) in &intersections {
                if source_tiles.binary_search(&destination_tile).is_ok() {
                    local_bytes = local_bytes.saturating_add(bytes);
                    local_intersections = local_intersections.saturating_add(1);
                } else {
                    remote_bytes = remote_bytes.saturating_add(bytes);
                    remote_fragments = remote_fragments.saturating_add(1);
                    let entry = remote
                        .entry((source_tiles[0], extents.clone()))
                        .or_insert_with(|| (bytes, Vec::new(), true));
                    entry.1.push(destination_tile);
                    entry.2 &= aligned;
                }
            }
            traffic
                .exchange
                .add_incoming(destination_tile, remote_bytes, remote_fragments);
            for (total, amount) in local.entry(destination_tile).or_default().iter_mut().zip([
                destination_size,
                local_bytes,
                intersections.len() as u64 + u64::from(zero_bytes != 0),
                local_intersections,
            ]) {
                *total = total.saturating_add(amount);
            }
        }
    }
    for ((source, _), (bytes, _, _)) in &remote {
        traffic.exchange.add_outgoing(*source, *bytes, 1);
    }
    let Target::Ipu21 = target;
    let topology = ipu_target::ipu21::fabric::Topology::c600();
    let mut paired = traffic.exchange.clone();
    for ((source, _), (bytes, mut receivers, aligned)) in remote {
        if !aligned {
            continue;
        }
        let Ok(partner) = topology.paired_logical(source) else {
            continue;
        };
        if partner >= tile_count {
            continue;
        }
        receivers.sort_unstable();
        receivers.dedup();
        if !receivers.iter().all(|&tile| {
            topology
                .paired_logical(tile)
                .is_ok_and(|other| receivers.binary_search(&other).is_ok())
        }) {
            continue;
        }
        paired.outgoing_lanes[usize::from(source)].bytes -= bytes / 2;
        paired.add_outgoing(partner, bytes / 2, 0);
        for tile in receivers {
            paired.incoming_tiles[usize::from(tile)].bytes -= bytes / 2;
        }
    }
    // Borrowing a busy sender's partner can make pairing worse. The ordinary
    // schedule is still available, so retain its estimate in that case.
    traffic.paired_payload_bytes = paired
        .maximum_payload_bytes()
        .min(traffic.exchange.maximum_payload_bytes());
    for [destination, bytes, intersections, calls] in local.into_values() {
        traffic.maximum_destination_bytes = traffic.maximum_destination_bytes.max(destination);
        traffic.maximum_local_bytes = traffic.maximum_local_bytes.max(bytes);
        traffic.maximum_intersections = traffic.maximum_intersections.max(intersections);
        traffic.maximum_local_intersections = traffic.maximum_local_intersections.max(calls);
    }
    Some(traffic)
}

fn paired_copy_aligned(
    input: &TensorType,
    output: &TensorType,
    source: &[(u32, u32)],
    destination: &[(u32, u32)],
    region: &[(u32, u32)],
) -> bool {
    use crate::storage::{TensorStorage, byte_traversal};
    let extents = |bounds: &[(u32, u32)]| {
        bounds
            .iter()
            .enumerate()
            .map(|(axis, &(start, end))| crate::ShardExtent {
                axis: axis as u16,
                start,
                logical_end: end,
                physical_end: end,
            })
            .collect::<Vec<_>>()
    };
    let view = extents(region);
    let left = extents(source);
    let right = extents(destination);
    let traversals = byte_traversal(
        TensorStorage {
            format: &input.format,
            extents: &left,
        },
        &view,
        true,
    )
    .and_then(|a| {
        byte_traversal(
            TensorStorage {
                format: &output.format,
                extents: &right,
            },
            &view,
            true,
        )
        .map(|b| (a, b))
    });
    traversals
        .ok()
        .and_then(|(a, b)| a.regular_copy(&b))
        .is_some_and(|(a, b)| {
            [a, b]
                .iter()
                .all(|s| s.bytes != 0 && (s.offset | s.bytes | s.stride).is_multiple_of(8))
        })
}

pub(super) fn layout_extents(
    shape: &TensorShape,
    layout: &Layout,
) -> Option<impl Iterator<Item = (u16, Vec<(u32, u32)>)>> {
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
            }),
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
