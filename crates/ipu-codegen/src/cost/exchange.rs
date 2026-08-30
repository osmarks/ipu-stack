//! Exchange traffic accounting and conversion to target costs.

use crate::metrics::{CostEstimate, ExchangeFootprint};
use ipu_target::hardware::HardwareTarget;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ExchangeEndpointLoad {
    pub bytes: u64,
    pub fragments: u64,
}

impl ExchangeEndpointLoad {
    fn add(&mut self, bytes: u64, fragments: u64) {
        self.bytes = self.bytes.saturating_add(bytes);
        self.fragments = self.fragments.saturating_add(fragments);
    }
}

/// Resource-indexed work for transfers sharing an exchange phase. Sends from
/// an adjacent tile pair occupy one bus; receives are independent per tile.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct ExchangeEndpointTraffic {
    pub outgoing_buses: Vec<ExchangeEndpointLoad>,
    pub incoming_tiles: Vec<ExchangeEndpointLoad>,
}

impl ExchangeEndpointTraffic {
    pub(super) fn from_maxima(
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

    pub(super) fn add_outgoing(&mut self, bus: u16, bytes: u64, fragments: u64) {
        add_endpoint_load(&mut self.outgoing_buses, bus, bytes, fragments);
    }

    pub(super) fn add_incoming(&mut self, tile: u16, bytes: u64, fragments: u64) {
        add_endpoint_load(&mut self.incoming_tiles, tile, bytes, fragments);
    }

    pub(super) fn merge(&mut self, other: &Self) {
        for (bus, load) in other.outgoing_buses.iter().copied().enumerate() {
            self.add_outgoing(bus as u16, load.bytes, load.fragments);
        }
        for (tile, load) in other.incoming_tiles.iter().copied().enumerate() {
            self.add_incoming(tile as u16, load.bytes, load.fragments);
        }
    }

    pub(super) fn maximum_outgoing_bytes(&self) -> u64 {
        endpoint_maxima(&self.outgoing_buses).0
    }

    pub(super) fn maximum_incoming_bytes(&self) -> u64 {
        endpoint_maxima(&self.incoming_tiles).0
    }

    pub(super) fn maximum_payload_bytes(&self) -> u64 {
        self.maximum_outgoing_bytes()
            .max(self.maximum_incoming_bytes())
    }

    pub(super) fn maximum_outgoing_fragments(&self) -> u64 {
        endpoint_maxima(&self.outgoing_buses).1
    }

    pub(super) fn maximum_incoming_fragments(&self) -> u64 {
        endpoint_maxima(&self.incoming_tiles).1
    }

    pub(super) fn maximum_fragments(&self) -> u64 {
        self.maximum_outgoing_fragments()
            .max(self.maximum_incoming_fragments())
    }

    pub(super) fn is_empty(&self) -> bool {
        self.maximum_payload_bytes() == 0
    }
}

fn endpoint_maxima(loads: &[ExchangeEndpointLoad]) -> (u64, u64) {
    loads.iter().fold((0, 0), |(bytes, fragments), load| {
        (bytes.max(load.bytes), fragments.max(load.fragments))
    })
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

pub(super) fn cost(
    traffic: &ExchangeEndpointTraffic,
    phases: u64,
    target: HardwareTarget,
) -> CostEstimate {
    if traffic.is_empty() || phases == 0 {
        return CostEstimate::default();
    }
    let costs = target.costs();
    let payload_cycles = traffic
        .maximum_payload_bytes()
        .div_ceil(costs.exchange_bytes_per_cycle);
    let cutover_cycles = traffic
        .maximum_fragments()
        .saturating_mul(costs.logical_fragment_cycles);
    let cycles = payload_cycles
        .max(cutover_cycles)
        .saturating_add(phases.saturating_mul(costs.exchange_phase_cycles));
    CostEstimate {
        cycles,
        exchange_cycles: cycles,
        exchange_footprint: footprint(traffic, phases, target),
    }
}

fn footprint(
    traffic: &ExchangeEndpointTraffic,
    phases: u64,
    target: HardwareTarget,
) -> ExchangeFootprint {
    let transfer_bytes = u64::from(target.exchange().maximum_transfer_words) * 4;
    ExchangeFootprint {
        phases,
        maximum_transfer_chunks_per_tile: traffic
            .maximum_payload_bytes()
            .div_ceil(transfer_bytes)
            .max(traffic.maximum_fragments())
            .max(phases),
    }
}
