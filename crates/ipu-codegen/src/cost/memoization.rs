//! Thread-safe memoization for repeated conversion estimates.

use super::kernel::CostModel;
use crate::OperatorSchedule;
use crate::conversion::{ConversionStrategy, DeferredTransform, layout_conversion_strategy};
use crate::graph::TensorShape;
use crate::layout::{Layout, TensorType};
use crate::metrics::{CostEstimate, ExchangeFootprint};
use crate::operator::{MidOperator, OperatorRequirements, Precision};
use foldhash::fast::FixedState;
use ipu_target::hardware::HardwareTarget;
use std::collections::HashMap;
use std::sync::Mutex;

pub(crate) struct MemoizedCostModel<'a, C> {
    inner: &'a C,
    spatial_capacity: u16,
    rearrangements: Mutex<RearrangementCache>,
}

type RearrangementKey = (TensorType, TensorType, ConversionStrategy);
type RearrangementCache = HashMap<RearrangementKey, CostEstimate, FixedState>;

impl<'a, C> MemoizedCostModel<'a, C> {
    pub(crate) fn new(inner: &'a C, spatial_capacity: u16) -> Self {
        Self {
            inner,
            spatial_capacity,
            rearrangements: Mutex::new(HashMap::default()),
        }
    }

    fn rearrangement(
        &self,
        source: TensorType,
        destination: TensorType,
        strategy: ConversionStrategy,
    ) -> CostEstimate
    where
        C: CostModel,
    {
        let active_tiles = source
            .format
            .layout
            .tiling
            .tile_count
            .max(destination.format.layout.tiling.tile_count);
        let key = (source.clone(), destination.clone(), strategy);
        let mut cache = self.rearrangements.lock().unwrap();
        if let Some(&cost) = cache.get(&key) {
            return cost;
        }
        let mut cost = self
            .inner
            .rearrangement_cost(&source, &destination, strategy);
        cost.cycles = cost
            .cycles
            .saturating_mul(u64::from(self.spatial_capacity))
            .div_ceil(u64::from(active_tiles));
        cache.insert(key, cost);
        cost
    }
}

impl<C: CostModel> CostModel for MemoizedCostModel<'_, C> {
    fn target(&self) -> HardwareTarget {
        self.inner.target()
    }

    fn operator_cycles(
        &self,
        operator: MidOperator,
        schedule: &OperatorSchedule,
        requirements: &OperatorRequirements,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> u64 {
        self.inner
            .operator_cycles(operator, schedule, requirements, inputs, output)
    }

    fn cast_cycles(&self, input: &TensorType, to: Precision) -> u64 {
        self.inner.cast_cycles(input, to)
    }

    fn layout_conversion_cost(
        &self,
        shape: &TensorShape,
        precision: Precision,
        from: &Layout,
        to: &Layout,
    ) -> CostEstimate {
        let requested = layout_conversion_strategy(precision, from, to);
        let source = TensorType::new(shape.0.clone(), precision, from.clone());
        let destination = TensorType::new(shape.0.clone(), precision, to.clone());
        self.rearrangement(source, destination, requested)
    }

    fn operator_exchange_cycles(
        &self,
        operator: MidOperator,
        schedule: &OperatorSchedule,
        requirements: &OperatorRequirements,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> u64 {
        self.inner
            .operator_exchange_cycles(operator, schedule, requirements, inputs, output)
    }

    fn operator_exchange_footprint(
        &self,
        operator: MidOperator,
        schedule: &OperatorSchedule,
        requirements: &OperatorRequirements,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> ExchangeFootprint {
        self.inner
            .operator_exchange_footprint(operator, schedule, requirements, inputs, output)
    }

    fn deferred_input_cycles(
        &self,
        transform: DeferredTransform,
        source: &TensorType,
        logical_output: &TensorType,
        consumer_input: &TensorType,
        consumer_dispatch: &OperatorSchedule,
        producer_cycles: u64,
    ) -> u64 {
        self.inner.deferred_input_cycles(
            transform,
            source,
            logical_output,
            consumer_input,
            consumer_dispatch,
            producer_cycles,
        )
    }

    fn deferred_input_exchange_cycles(
        &self,
        transform: DeferredTransform,
        source: &TensorType,
        logical_output: &TensorType,
        consumer_input: &TensorType,
        consumer_dispatch: &OperatorSchedule,
        producer_cycles: u64,
    ) -> u64 {
        self.inner.deferred_input_exchange_cycles(
            transform,
            source,
            logical_output,
            consumer_input,
            consumer_dispatch,
            producer_cycles,
        )
    }

    fn rearrangement_cost(
        &self,
        source: &TensorType,
        destination: &TensorType,
        strategy: ConversionStrategy,
    ) -> CostEstimate {
        self.rearrangement(source.clone(), destination.clone(), strategy)
    }
}
