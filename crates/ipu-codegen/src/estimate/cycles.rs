//! Analytical IPU21 cycle estimation used during operator planning.

use crate::estimate::{ExchangeEndpointTraffic, conversion_traffic, maximum_shard_bytes};
use crate::graph::TensorShape;
use crate::{
    AmpOrder, BlockMajorOrder, ConversionStrategy, ElementOrder, Layout, OperatorPlan, Precision,
    TensorFormat, TensorType,
};
use foldhash::fast::FixedState;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub trait CostModel: Sync {
    fn implementation(
        &self,
        plan: &OperatorPlan,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> Option<Arc<crate::MidProgram>> {
        crate::mid::implementation::implement(plan, inputs, output)
    }
    fn operator_cycle_override(
        &self,
        _plan: &OperatorPlan,
        _inputs: &[TensorType],
        _output: &TensorType,
    ) -> Option<u64> {
        None
    }
    #[cfg(test)]
    fn operator_cycles(
        &self,
        plan: &OperatorPlan,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> u64 {
        self.operator_cycle_override(plan, inputs, output)
            .unwrap_or_else(|| {
                self.implementation(plan, inputs, output)
                    .map_or(u64::MAX, |estimate| estimate.estimated_cycles)
            })
    }
    fn cast_cycles(&self, input: &TensorType, to: Precision) -> u64;
    fn cast_format_cycles(&self, input: &TensorType, output: &TensorFormat) -> u64 {
        let mut packed = input.clone();
        packed.format.layout.order = output.layout.order;
        self.cast_cycles(&packed, output.precision)
    }
    fn rearrangement_cost(
        &self,
        shape: &TensorShape,
        precision: Precision,
        strategy: ConversionStrategy,
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
                    .saturating_mul(ipu_exchange::PLAN_WORDS as u64),
            )
            .saturating_mul(4)
    }
}

pub(crate) struct MemoizedCostModel<'a, C> {
    inner: &'a C,
    spatial_capacity: u16,
    rearrangements: Mutex<RearrangementCache>,
    implementations:
        Mutex<HashMap<(OperatorPlan, Vec<TensorType>, TensorType), Arc<crate::MidProgram>>>,
}

type RearrangementKey = (TensorShape, Precision, ConversionStrategy, Layout, Layout);
type RearrangementCache = HashMap<RearrangementKey, Arc<OnceLock<RearrangementCost>>, FixedState>;

impl<'a, C> MemoizedCostModel<'a, C> {
    pub(crate) fn new(inner: &'a C, spatial_capacity: u16) -> Self {
        Self {
            inner,
            spatial_capacity,
            rearrangements: Mutex::new(HashMap::default()),
            implementations: Mutex::new(HashMap::default()),
        }
    }
}

impl<C: CostModel> CostModel for MemoizedCostModel<'_, C> {
    fn implementation(
        &self,
        plan: &OperatorPlan,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> Option<Arc<crate::MidProgram>> {
        let mut plan = plan.clone();
        plan.deferred_output = None;
        let key = (plan, inputs.to_vec(), output.clone());
        if let Some(retained) = self.implementations.lock().unwrap().get(&key).cloned() {
            return Some(retained);
        }
        // These are compact whole-device fragments. Retain them for this search
        // so shortlist and branch costing do not reconstruct rejected candidates.
        let built = self.inner.implementation(&key.0, &key.1, &key.2)?;
        self.implementations
            .lock()
            .unwrap()
            .insert(key, built.clone());
        Some(built)
    }
    fn operator_cycle_override(
        &self,
        plan: &OperatorPlan,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> Option<u64> {
        self.inner.operator_cycle_override(plan, inputs, output)
    }
    fn cast_cycles(&self, input: &TensorType, to: Precision) -> u64 {
        self.inner.cast_cycles(input, to)
    }
    fn cast_format_cycles(&self, input: &TensorType, output: &TensorFormat) -> u64 {
        self.inner.cast_format_cycles(input, output)
    }

    fn rearrangement_cost(
        &self,
        shape: &TensorShape,
        precision: Precision,
        strategy: ConversionStrategy,
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
            let mut cost = self
                .inner
                .rearrangement_cost(shape, precision, strategy, from, to);
            let active_tiles = from.tiling.tile_count.max(to.tiling.tile_count);
            // The inner model reports occupied work. Reduced-grid conversions
            // leave spatial issue slots idle, so convert that work into a phase
            // horizon using the occupancy of this particular planning target.
            cost.cycles = cost
                .cycles
                .saturating_mul(u64::from(self.spatial_capacity))
                .div_ceil(u64::from(active_tiles));
            cost
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
    local_copy_call_cycles: 288,
    // Target::getGlobalSyncCycles.
    exchange_phase_cycles: 600,
    // popops::internal::basicOpSupervisorOverhead(false).
    kernel_launch_cycles: 11,
};

// Fragmented logical conversions spend most of their critical path changing
// endpoints and receive pointers rather than moving payload. Current IPU21
// schedules sustain about 160 event cycles per independent fragment once
// routing and pointer cutovers are included. This is used to choose between a
// direct word-fragment exchange and one local packed staging pass.
pub(crate) const IPU21_LOGICAL_FRAGMENT_CYCLES: u64 = 160;

pub(super) fn exchange_endpoint_cycles(traffic: &ExchangeEndpointTraffic, phases: u64) -> u64 {
    if traffic.is_empty() || phases == 0 {
        return 0;
    }
    traffic
        .maximum_payload_bytes()
        .div_ceil(IPU21_TARGET_COSTS.exchange_bytes_per_cycle)
        .saturating_add(phases.saturating_mul(IPU21_TARGET_COSTS.exchange_phase_cycles))
}

pub(super) fn exchange_fragment_price(bytes: u64, phases: u64, fragments: u64) -> (u64, u64) {
    if bytes == 0 || phases == 0 {
        return (0, 0);
    }
    let fragments = fragments.max(phases);
    let cycles = bytes
        .div_ceil(IPU21_TARGET_COSTS.exchange_bytes_per_cycle)
        .max(fragments.saturating_mul(IPU21_LOGICAL_FRAGMENT_CYCLES))
        .saturating_add(phases.saturating_mul(IPU21_TARGET_COSTS.exchange_phase_cycles));
    let rows = ExchangeFootprint {
        phases,
        maximum_transfer_chunks_per_tile: fragments,
        encoded_row_bytes: None,
    }
    .estimated_row_bytes();
    (cycles, rows)
}

// Indexed F16 layout transforms execute scalar address arithmetic as well as
// their loads and stores. The transposed-right panel is a contiguous copy:
// its final coefficient permutation is performed by the GEMM's ld*putcs
// sequence. Keep these costs separate from ideal memcpy bandwidth.
const IPU21_INDEXED_F16_TRANSFORM_CYCLES_PER_ELEMENT: u64 = 10;
// Full AMP-left panels use four 64-bit load/store pairs, including pointer
// updates. Allow the panel loop and row setup in the rounded per-element price.
const IPU21_AMP_LEFT_PACK_CYCLES_PER_ELEMENT: u64 = 1;
const IPU21_CONTIGUOUS_PANEL_PACK_CYCLES_PER_ELEMENT: u64 = 3;
pub(crate) fn row_major_pack_cycles(tensor: &TensorType, elements: u64) -> u64 {
    let cycles_per_element = match tensor.format.layout.order {
        ElementOrder::RowMajor => return 0,
        ElementOrder::Amp(AmpOrder::TransposedRight) => {
            IPU21_CONTIGUOUS_PANEL_PACK_CYCLES_PER_ELEMENT
        }
        ElementOrder::Amp(AmpOrder::Left) => IPU21_AMP_LEFT_PACK_CYCLES_PER_ELEMENT,
        ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
            row_block,
            column_block,
        }) if row_block.is_multiple_of(16)
            && column_block == 16
            && tensor
                .shape
                .0
                .iter()
                .rev()
                .nth(1)
                .is_some_and(|&rows| rows <= u32::from(row_block))
            && tensor
                .shape
                .0
                .last()
                .is_some_and(|columns| columns.is_multiple_of(4)) =>
        {
            let columns = u64::from(*tensor.shape.0.last().unwrap());
            let rows = u64::from(row_block);
            return crate::kernel::cost::f16_coefficient_pack_cycles(
                elements.div_ceil(rows * columns.div_ceil(16) * 16),
                rows,
                columns,
            );
        }
        ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. }) | ElementOrder::Amp(_) => {
            IPU21_INDEXED_F16_TRANSFORM_CYCLES_PER_ELEMENT
        }
        ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix { .. }) => {
            IPU21_INDEXED_F16_TRANSFORM_CYCLES_PER_ELEMENT
        }
    };
    elements
        .saturating_mul(cycles_per_element)
        .saturating_add(IPU21_TARGET_COSTS.kernel_launch_cycles)
}

impl CostModel for Ipu21CostModel {
    fn cast_cycles(&self, input: &TensorType, to: Precision) -> u64 {
        let elements = maximum_shard_bytes(input).div_ceil(input.format.precision.bytes());
        let columns = input
            .format
            .layout
            .resolve(&input.shape)
            .ok()
            .and_then(|resolved| {
                resolved
                    .axes()
                    .and_then(|axes| axes.last())
                    .map(|axis| u64::from(axis.maximum_extent()))
            })
            .unwrap_or(elements)
            .max(1);
        let panel_rows = input
            .format
            .layout
            .order
            .fp8_cast_panel_rows(elements / columns, columns);
        super::primitive::cast_cycles(
            input.format.precision,
            to,
            elements,
            if input.format.layout.order == ElementOrder::Amp(AmpOrder::Left)
                && panel_rows == 1
                && columns.is_multiple_of(32)
            {
                0
            } else {
                panel_rows
            },
        )
    }

    fn rearrangement_cost(
        &self,
        shape: &TensorShape,
        precision: Precision,
        strategy: ConversionStrategy,
        from: &Layout,
        to: &Layout,
    ) -> RearrangementCost {
        if strategy == ConversionStrategy::StageLogicalThenTransform
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
        let direct_retile = strategy == ConversionStrategy::DirectRetile;
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
    use crate::{MidOperator, OperatorDispatch};
    use crate::{
        OperandRequirement, OutputAliasing, StorageRequirements, TensorFormat, TileKernelSpec,
    };

    const CASES: usize = 32;

    fn pointwise_dispatch() -> OperatorDispatch {
        OperatorDispatch::Pointwise {
            kernel: TileKernelSpec::Gelu,
        }
    }

    fn pointwise_requirements(format: TensorFormat) -> StorageRequirements {
        StorageRequirements {
            inputs: vec![OperandRequirement::new(format.clone(), 8)],
            output: OperandRequirement::new(format, 8),
            output_aliasing: OutputAliasing::Fresh,
            distinct_elements: Vec::new(),
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
            let outgoing = traffic.maximum_outgoing_bytes();
            let incoming = traffic.maximum_incoming_bytes();
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
                traffic.maximum_incoming_fragments(),
                traffic.maximum_outgoing_fragments(),
            );
            let reversed = exchange_endpoint_cycles(&reversed_traffic, phases);
            assert_eq!(cycles, reversed, "case {case}");
        }
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
            for operator in [MidOperator::Gelu, MidOperator::Add] {
                let sharded_cycles = Ipu21CostModel.operator_cycles(
                    &OperatorPlan {
                        operator,
                        dispatch: pointwise_dispatch(),
                        requirements: pointwise_requirements(sharded.format.clone()),
                        deferred_output: None,
                    },
                    std::slice::from_ref(&sharded),
                    &sharded,
                );
                let unsharded_cycles = Ipu21CostModel.operator_cycles(
                    &OperatorPlan {
                        operator,
                        dispatch: pointwise_dispatch(),
                        requirements: pointwise_requirements(unsharded.format.clone()),
                        deferred_output: None,
                    },
                    std::slice::from_ref(&unsharded),
                    &unsharded,
                );
                assert!(sharded_cycles <= unsharded_cycles, "case {case}");
            }
        }
    }
}
