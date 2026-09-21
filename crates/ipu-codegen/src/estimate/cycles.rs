//! Analytical IPU21 cycle estimation used during operator planning.

use crate::estimate::ExchangeEndpointTraffic;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExchangeFootprint {
    pub phases: u64,
    pub maximum_transfer_chunks_per_tile: u64,
    /// Detailed endpoint estimate, when concrete span geometry is available.
    pub encoded_row_bytes: Option<u64>,
}

impl ExchangeFootprint {
    pub const fn estimated_row_bytes(self) -> u64 {
        if let Some(bytes) = self.encoded_row_bytes {
            return bytes;
        }
        // Charge row slots for the supplied fragment count. Mid costing
        // supplies a heuristic count; only the geometry screen supplies
        // concrete span chunks. Compact encoding may use fewer bytes.
        self.phases
            .saturating_add(
                self.maximum_transfer_chunks_per_tile
                    .saturating_mul(crate::exchange::PLAN_WORDS as u64),
            )
            .saturating_mul(4)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ipu21TargetCosts {
    pub exchange_bytes_per_cycle: u64,
    pub local_copy_bytes_per_cycle: u64,
    pub local_copy_call_cycles: u64,
    pub exchange_phase_cycles: u64,
    pub kernel_launch_cycles: u64,
}

// Target::getExchangeBytesPerCycle.
pub const IPU21_TARGET_COSTS: Ipu21TargetCosts = Ipu21TargetCosts {
    exchange_bytes_per_cycle: 4,
    local_copy_bytes_per_cycle: 8,
    // A finalized six-worker local-copy invocation, including supervisor and
    // worker rendezvous overhead, takes 288 tile cycles on IPU21.
    local_copy_call_cycles: crate::kernel::copy::WORKER_CALL_CYCLES,
    // Target::getGlobalSyncCycles.
    exchange_phase_cycles: 600,
    // popops::internal::basicOpSupervisorOverhead(false).
    kernel_launch_cycles: 11,
};

// An ordinary receive needs source selection, neutralization and (unless
// contiguous) a pointer write. These issue alongside the independent payload
// stream, not as a route-latency penalty per fragment. Mid geometry cannot yet
// prove pointer continuation; low geometry counts it explicitly.
pub(crate) const EXCHANGE_FRAGMENT_CONTROLS: u64 = 3;

// Resource-work estimate, not a conflict-free schedule or a guaranteed bound:
// route latency, bank conflicts and dependency chains remain unpriced here.
pub(crate) fn exchange_work_cycles(bytes: u64, controls: u64) -> u64 {
    bytes
        .div_ceil(IPU21_TARGET_COSTS.exchange_bytes_per_cycle)
        .max(controls)
}

pub(super) fn exchange_endpoint_cycles(traffic: &ExchangeEndpointTraffic, phases: u64) -> u64 {
    if traffic.is_empty() || phases == 0 {
        return 0;
    }
    exchange_work_cycles(traffic.maximum_payload_bytes(), traffic.maximum_controls())
        .saturating_add(phases.saturating_mul(IPU21_TARGET_COSTS.exchange_phase_cycles))
}

pub(super) fn exchange_fragment_price(bytes: u64, phases: u64, fragments: u64) -> (u64, u64) {
    if bytes == 0 || phases == 0 {
        return (0, 0);
    }
    let fragments = fragments.max(phases);
    let cycles = exchange_work_cycles(bytes, fragments.saturating_mul(EXCHANGE_FRAGMENT_CONTROLS))
        .saturating_add(phases.saturating_mul(IPU21_TARGET_COSTS.exchange_phase_cycles));
    let rows = ExchangeFootprint {
        phases,
        maximum_transfer_chunks_per_tile: fragments,
        encoded_row_bytes: None,
    }
    .estimated_row_bytes();
    (cycles, rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::ExchangeEndpointLoad;
    use crate::{Layout, Precision, TensorType};

    const CASES: usize = 32;

    #[test]
    fn randomized_exchange_endpoint_costs_overlap_opposite_directions() {
        let mut random = fastrand::Rng::with_seed(0x6675_6c6c_6475_706c);
        for case in 0..CASES {
            let mut traffic = ExchangeEndpointTraffic::default();
            for bus in 0..random.u16(1..=32) {
                traffic.add_outgoing(bus, random.u64(1..=1 << 20), random.u64(1..=256));
            }
            for tile in 0..random.u16(1..=64) {
                traffic.add_incoming(tile, random.u64(1..=1 << 20), random.u64(1..=256));
            }
            let maxima = |loads: &[ExchangeEndpointLoad]| {
                loads.iter().fold((0u64, 0u64), |(bytes, fragments), load| {
                    (bytes.max(load.bytes), fragments.max(load.fragments))
                })
            };
            let (outgoing, outgoing_fragments) = maxima(&traffic.outgoing_lanes);
            let (incoming, incoming_fragments) = maxima(&traffic.incoming_tiles);
            let phases = random.u64(1..=32);
            let fixed = phases.saturating_mul(IPU21_TARGET_COSTS.exchange_phase_cycles);
            let cycles = exchange_endpoint_cycles(&traffic, phases);
            assert_eq!(
                cycles.saturating_sub(fixed),
                outgoing
                    .max(incoming)
                    .div_ceil(IPU21_TARGET_COSTS.exchange_bytes_per_cycle),
                "case {case}"
            );
            let reversed_traffic = ExchangeEndpointTraffic::from_maxima(
                incoming,
                outgoing,
                incoming_fragments,
                outgoing_fragments,
            );
            let reversed = exchange_endpoint_cycles(&reversed_traffic, phases);
            assert_eq!(cycles, reversed, "case {case}");
        }
    }

    #[test]
    fn exchange_controls_overlap_payload_but_share_issue_slots() {
        let mut traffic = ExchangeEndpointTraffic::default();
        // Many short writes: source+neutral controls, one initial pointer.
        traffic.add_receive(0, 400, 100, 1);
        assert_eq!(exchange_endpoint_cycles(&traffic, 1), 600 + 201);
        traffic.add_outgoing(0, 400, 100);
        assert_eq!(exchange_endpoint_cycles(&traffic, 1), 600 + 401);
        // A neighbor has its own TX lane and its own issue slots.
        traffic.add_outgoing(1, 1600, 1);
        assert_eq!(exchange_endpoint_cycles(&traffic, 1), 600 + 401);
        // Strided writes require a pointer for each receive.
        let mut strided = ExchangeEndpointTraffic::default();
        strided.add_receive(0, 400, 100, 100);
        assert_eq!(exchange_endpoint_cycles(&strided, 1), 600 + 300);
        // Long payloads hide these control costs, rather than paying them serially.
        strided.add_receive(0, 4000, 1, 1);
        assert_eq!(exchange_endpoint_cycles(&strided, 1), 600 + 1100);
    }

    #[test]
    fn randomized_pointwise_costs_track_spatial_work_per_tile() {
        let mut random = fastrand::Rng::with_seed(0x706f_696e_7477_6973);
        for case in 0..CASES {
            let tiles = 1_u16 << random.u32(1..=6);
            let rows = u32::from(tiles) * random.u32(1..=8);
            let columns = 16 * random.u32(1..=16);
            let sharded =
                TensorType::new([rows, columns], Precision::F16, Layout::row_sharded(tiles));
            let unsharded =
                TensorType::new([rows, columns], Precision::F16, Layout::row_sharded(1));
            for add in [false, true] {
                let cycles = |tensor: &TensorType| {
                    let mut graph = crate::HighGraph::new();
                    let x = graph.host_input("x", tensor.shape.0.clone()).unwrap();
                    let y = if add { graph.add(x, x) } else { graph.gelu(x) }.unwrap();
                    graph.set_outputs([y]).unwrap();
                    let config =
                        crate::PipelineConfig::new(tiles).with_input(x, tensor.format.clone());
                    let mut layouts = crate::planner::boundary_layouts(&graph, &config);
                    layouts.insert(y, Some(tensor.format.layout.clone()));
                    let program = crate::planner::plan(
                        &graph,
                        &layouts,
                        &config,
                        crate::planner::SearchLimits::default(),
                    )
                    .unwrap();
                    program.validate().unwrap();
                    program.estimated_cycles
                };
                let sharded_cycles = cycles(&sharded);
                let unsharded_cycles = cycles(&unsharded);
                assert!(sharded_cycles <= unsharded_cycles, "case {case}");
            }
        }
    }
}
