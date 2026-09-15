//! Analytical IPU21 cycle estimation used during operator planning.

use crate::estimate::{ExchangeEndpointTraffic, conversion_traffic};
use crate::graph::TensorShape;
use crate::planner::operator::OperatorPlan;
use crate::{CopyPolicy, ElementOrder, Layout, Precision, TensorFormat, TensorType};
use foldhash::fast::FixedState;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub trait CostModel: Sync {
    fn operator_cycle_override(
        &self,
        _plan: &OperatorPlan,
        _inputs: &[TensorType],
        _output: &TensorType,
    ) -> Option<u64> {
        None
    }
    fn cast_format_cycles(&self, input: &TensorType, output: &TensorFormat) -> u64;
    fn rearrangement_cost(
        &self,
        shape: &TensorShape,
        precision: Precision,
        strategy: CopyPolicy,
        from: &Layout,
        to: &Layout,
    ) -> RearrangementCost;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RearrangementCost {
    pub cycles: u64,
    pub exchange_cycles: u64,
}

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

pub(crate) struct MemoizedCostModel<'a, C> {
    inner: &'a C,
    rearrangements: Mutex<RearrangementCache>,
}

type RearrangementKey = (TensorShape, Precision, CopyPolicy, Layout, Layout);
type RearrangementCache = HashMap<RearrangementKey, Arc<OnceLock<RearrangementCost>>, FixedState>;

impl<'a, C> MemoizedCostModel<'a, C> {
    pub(crate) fn new(inner: &'a C) -> Self {
        Self {
            inner,
            rearrangements: Mutex::new(HashMap::default()),
        }
    }
}

impl<C: CostModel> CostModel for MemoizedCostModel<'_, C> {
    fn operator_cycle_override(
        &self,
        plan: &OperatorPlan,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> Option<u64> {
        self.inner.operator_cycle_override(plan, inputs, output)
    }
    fn cast_format_cycles(&self, input: &TensorType, output: &TensorFormat) -> u64 {
        self.inner.cast_format_cycles(input, output)
    }

    fn rearrangement_cost(
        &self,
        shape: &TensorShape,
        precision: Precision,
        strategy: CopyPolicy,
        from: &Layout,
        to: &Layout,
    ) -> RearrangementCost {
        let key = (shape.clone(), precision, strategy, from.clone(), to.clone());
        let cached = self
            .rearrangements
            .lock()
            .unwrap()
            .entry(key)
            .or_default()
            .clone();
        *cached.get_or_init(|| {
            self.inner
                .rearrangement_cost(shape, precision, strategy, from, to)
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Ipu21CostModel;

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

impl CostModel for Ipu21CostModel {
    fn cast_format_cycles(&self, input: &TensorType, output: &TensorFormat) -> u64 {
        let destination = TensorType {
            shape: input.shape.clone(),
            format: output.clone(),
        };
        let (Some(source), Some(destination)) = (
            super::mid::local_tensor(input),
            super::mid::local_tensor(&destination),
        ) else {
            return u64::MAX;
        };
        crate::kernel::KernelCall::select(
            &crate::mid::MidOperationKind::Cast {
                from: input.format.precision,
                to: output.precision,
            },
            &[crate::kernel::Geometry::Tensor(&source)],
            &[crate::kernel::Geometry::Tensor(&destination)],
        )
        .map_or(u64::MAX, |call| call.cycles())
    }

    fn rearrangement_cost(
        &self,
        shape: &TensorShape,
        precision: Precision,
        strategy: CopyPolicy,
        from: &Layout,
        to: &Layout,
    ) -> RearrangementCost {
        let strategy = if strategy == CopyPolicy::Automatic {
            crate::default_copy_policy(from, to)
        } else {
            strategy
        };
        if strategy == CopyPolicy::StageLogicalThenTransform
            && from.order != ElementOrder::RowMajor
            && to.order != ElementOrder::RowMajor
        {
            // This strategy receives into row-major destination staging. It
            // does not yet pack a permuted source locally, so a non-row-major
            // source can expose sub-word logical spans which the exchange
            // hardware cannot send. Do not price an unmaterializable plan.
            return RearrangementCost {
                cycles: u64::MAX / 8,
                exchange_cycles: u64::MAX / 8,
            };
        }
        let Some(traffic) = conversion_traffic(shape, precision, from, to) else {
            return RearrangementCost {
                cycles: u64::MAX / 8,
                exchange_cycles: u64::MAX / 8,
            };
        };
        let direct_retile = strategy == CopyPolicy::DirectRetile;
        let endpoint_traffic = &traffic.exchange;
        let mut exchange_cycles = exchange_endpoint_cycles(endpoint_traffic, 1);
        if direct_retile && !endpoint_traffic.is_empty() {
            let input = TensorType::new(shape.0.clone(), precision, from.clone());
            let output = TensorType::new(shape.0.clone(), precision, to.clone());
            if let Some(fragments) = super::movement::grid_fragments(&input, &output) {
                exchange_cycles =
                    exchange_fragment_price(endpoint_traffic.maximum_payload_bytes(), 1, fragments)
                        .0;
            }
        }
        let (local_bytes, local_calls) = if direct_retile {
            (
                traffic.maximum_local_bytes,
                traffic.maximum_local_intersections,
            )
        } else {
            (
                traffic.maximum_destination_bytes.saturating_mul(2),
                traffic.maximum_intersections,
            )
        };
        let local_cycles = local_bytes
            .div_ceil(IPU21_TARGET_COSTS.local_copy_bytes_per_cycle)
            .saturating_add(local_calls.saturating_mul(IPU21_TARGET_COSTS.local_copy_call_cycles));
        RearrangementCost {
            cycles: exchange_cycles.saturating_add(local_cycles),
            exchange_cycles,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::ExchangeEndpointLoad;
    use crate::planner::operator::OperandRequirement;
    use crate::planner::operator::OperatorFamily;

    const CASES: usize = 32;

    #[test]
    fn memoized_rearrangements_preserve_critical_path_costs() {
        let costs = MemoizedCostModel::new(&Ipu21CostModel);
        // The underlying model already charges the longest local copy, so
        // fewer owners increase latency without a second occupancy multiplier.
        for (shape, owners, cycles) in [
            ([64, 128], 1, 2336),
            ([64, 128], 8, 544),
            ([64, 128], 64, 320),
            ([729, 1152], 96, 2592),
            ([729, 1152], 729, 576),
            ([729, 1152], 1472, 576),
        ] {
            let source = Layout::row_sharded(owners);
            let mut target = source.clone();
            target.memory_class = crate::MemoryClass::Ipu21Interleaved;
            assert_eq!(
                costs.rearrangement_cost(
                    &TensorShape::new(shape),
                    Precision::F16,
                    CopyPolicy::DirectRetile,
                    &source,
                    &target,
                ),
                RearrangementCost {
                    cycles,
                    exchange_cycles: 0
                },
            );
        }
        let shape = TensorShape::new([64, 128]);
        for source_tiles in [1, 2, 8, 64] {
            for target_tiles in [1, 4, 32, 64] {
                let source = Layout::row_sharded(source_tiles);
                let target = Layout::row_major(crate::TensorTiling::sharded(
                    crate::TensorAxis::FromEnd(1),
                    target_tiles,
                ));
                for strategy in [
                    CopyPolicy::DirectRetile,
                    CopyPolicy::StageLogicalThenTransform,
                ] {
                    let expected = Ipu21CostModel.rearrangement_cost(
                        &shape,
                        Precision::F16,
                        strategy,
                        &source,
                        &target,
                    );
                    assert!(expected.cycles > 0);
                    for _ in 0..2 {
                        assert_eq!(
                            costs.rearrangement_cost(
                                &shape,
                                Precision::F16,
                                strategy,
                                &source,
                                &target,
                            ),
                            expected
                        );
                    }
                }
            }
        }
    }

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
            for operator in [OperatorFamily::Gelu, OperatorFamily::Add] {
                let cycles = |tensor: &TensorType| {
                    let arity = if operator == OperatorFamily::Add {
                        2
                    } else {
                        1
                    };
                    let plan = crate::planner::catalogue::ConcreteOperatorCandidate::new(
                        operator,
                        vec![OperandRequirement::new(tensor.format.clone()); arity],
                        tensor.format.clone(),
                    )
                    .plan;
                    let program = crate::planner::fragments::build_fragment(
                        &plan,
                        &vec![tensor.clone(); arity],
                        tensor,
                    )
                    .expect("supported pointwise work");
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
