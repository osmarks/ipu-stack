//! Address-independent exchange traffic and GEMM communication geometry.

use super::*;

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

    pub(crate) fn add_outgoing(&mut self, bus: u16, bytes: u64, fragments: u64) {
        add_endpoint_load(&mut self.outgoing_buses, bus, bytes, fragments);
    }

    pub(crate) fn add_incoming(&mut self, tile: u16, bytes: u64, fragments: u64) {
        add_endpoint_load(&mut self.incoming_tiles, tile, bytes, fragments);
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
