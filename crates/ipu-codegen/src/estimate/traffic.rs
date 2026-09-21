//! Address-independent exchange traffic and GEMM communication geometry.

use super::*;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ConversionTraffic {
    pub maximum_destination_bytes: u64,
    pub maximum_local_bytes: u64,
    pub maximum_intersections: u64,
    pub maximum_local_intersections: u64,
    pub exchange: ExchangeEndpointTraffic,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExchangeEndpointLoad {
    pub bytes: u64,
    pub fragments: u64,
    pub controls: u64,
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
        add_endpoint_load(
            &mut self.outgoing_lanes,
            tile,
            bytes,
            fragments,
            fragments.saturating_mul(2),
        );
    }

    pub(crate) fn add_incoming(&mut self, tile: u16, bytes: u64, fragments: u64) {
        self.add_receive(tile, bytes, fragments, fragments);
    }

    pub(crate) fn add_receive(&mut self, tile: u16, bytes: u64, fragments: u64, resets: u64) {
        add_endpoint_load(
            &mut self.incoming_tiles,
            tile,
            bytes,
            fragments,
            fragments.saturating_mul(2).saturating_add(resets),
        );
    }

    pub(crate) fn maximum_controls(&self) -> u64 {
        // TX/RX payload lanes are independent, but share supervisor issue slots.
        (0..self.outgoing_lanes.len().max(self.incoming_tiles.len()))
            .map(|i| {
                self.outgoing_lanes.get(i).map_or(0, |l| l.controls)
                    + self.incoming_tiles.get(i).map_or(0, |l| l.controls)
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
    controls: u64,
) {
    if bytes == 0 && fragments == 0 && controls == 0 {
        return;
    }
    loads.resize(
        loads.len().max(usize::from(endpoint).saturating_add(1)),
        ExchangeEndpointLoad::default(),
    );
    let load = &mut loads[usize::from(endpoint)];
    load.bytes = load.bytes.saturating_add(bytes);
    load.fragments = load.fragments.saturating_add(fragments);
    load.controls = load.controls.saturating_add(controls);
}

pub(crate) fn conversion_traffic(
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
    let mut remote = HashMap::<(u16, Vec<(u32, u32)>), u64>::new();
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
            intersections.push((extents, source_tiles, bytes));
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
            for &(ref extents, source_tiles, bytes) in &intersections {
                if source_tiles.binary_search(&destination_tile).is_ok() {
                    local_bytes = local_bytes.saturating_add(bytes);
                    local_intersections = local_intersections.saturating_add(1);
                } else {
                    remote_bytes = remote_bytes.saturating_add(bytes);
                    remote_fragments = remote_fragments.saturating_add(1);
                    remote.insert((source_tiles[0], extents.clone()), bytes);
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
    for ((source, _), bytes) in remote {
        traffic.exchange.add_outgoing(source, bytes, 1);
    }
    for [destination, bytes, intersections, calls] in local.into_values() {
        traffic.maximum_destination_bytes = traffic.maximum_destination_bytes.max(destination);
        traffic.maximum_local_bytes = traffic.maximum_local_bytes.max(bytes);
        traffic.maximum_intersections = traffic.maximum_intersections.max(intersections);
        traffic.maximum_local_intersections = traffic.maximum_local_intersections.max(calls);
    }
    Some(traffic)
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
