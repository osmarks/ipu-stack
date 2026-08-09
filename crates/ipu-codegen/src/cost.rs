//! Analytical IPU21 cycle estimation used during operator planning.

use crate::estimate::{
    conversion_traffic, gemm_exchange_bytes_per_cycle, gemm_remote_bytes_per_tile,
    gemm_requires_panel_repacking, gemm_uses_panel_buffer, maximum_axis_shard_extent,
    maximum_shard_bytes, operator_memory_estimate, physical_elements,
};
use crate::graph::TensorShape;
use crate::mid::{
    AmpOrder, ElementOrder, Layout, MemoryClass, MidOperator, OperatorDispatch, Precision,
    TensorAxis, TensorType,
};

pub trait CostModel {
    fn operator_cycles(
        &self,
        operator: MidOperator,
        dispatch: &OperatorDispatch,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> u64;
    fn cast_cycles(&self, input: &TensorType, to: Precision) -> u64;
    fn rearrange_cycles(
        &self,
        shape: &TensorShape,
        precision: Precision,
        from: &Layout,
        to: &Layout,
    ) -> u64;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Ipu21CostModel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ipu21TargetCosts {
    pub exchange_bytes_per_cycle: u64,
    pub standard_load_bytes_per_cycle: u64,
    pub interleaved_load_bytes_per_cycle: u64,
    pub local_copy_bytes_per_cycle: u64,
    pub exchange_fragment_overhead_cycles: u64,
    pub local_copy_call_overhead_cycles: u64,
    pub gemm_call_overhead_cycles: u64,
}

pub const IPU21_TARGET_COSTS: Ipu21TargetCosts = Ipu21TargetCosts {
    exchange_bytes_per_cycle: 8,
    standard_load_bytes_per_cycle: 8,
    interleaved_load_bytes_per_cycle: 16,
    local_copy_bytes_per_cycle: 8,
    // Logical fragments carry route setup and synchronization costs in
    // addition to their byte-transfer time.
    exchange_fragment_overhead_cycles: 104,
    // A local copy launches a worker kernel even for a short intersection.
    local_copy_call_overhead_cycles: 512,
    gemm_call_overhead_cycles: 48,
};

impl CostModel for Ipu21CostModel {
    fn operator_cycles(
        &self,
        operator: MidOperator,
        dispatch: &OperatorDispatch,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> u64 {
        let elements = physical_elements(&output.shape, &output.format.layout);
        match operator {
            MidOperator::Gemm { multiply, .. } => {
                let left_shape = inputs[0]
                    .format
                    .layout
                    .padded_shape(&inputs[0].shape)
                    .unwrap_or_else(|_| inputs[0].shape.clone());
                let k = left_shape.0.last().copied().unwrap_or(1) as u64;
                let flops_per_cycle: u64 = match multiply {
                    Precision::F8F143 { .. } => 256,
                    Precision::F16 => 128,
                    Precision::F32 => 32,
                };
                let output_elements_per_tile =
                    maximum_shard_bytes(output).div_ceil(output.format.precision.bytes());
                let output_columns_per_tile =
                    maximum_axis_shard_extent(output, output.shape.0.len().saturating_sub(1));
                let arithmetic = output_elements_per_tile
                    .saturating_mul(2)
                    .saturating_mul(k)
                    .div_ceil(flops_per_cycle);
                let right = inputs.get(1);
                let right_bytes_consumed = right.map_or(u64::MAX, |right| {
                    output_columns_per_tile
                        .saturating_mul(k)
                        .saturating_mul(right.format.precision.bytes())
                });
                let resident_interleaved_weights = right.is_some_and(|right| {
                    right.format.layout.memory_class == MemoryClass::Ipu21Interleaved
                });
                let staged_weights = right.is_some_and(|right| {
                    gemm_uses_panel_buffer(dispatch, right, output)
                        && right.format.precision == Precision::F16
                });
                let streamed_k64_standard = right.filter(|right| {
                    staged_weights
                        && right.format.layout.memory_class == MemoryClass::Ipu21Standard
                        && right.format.layout.order == ElementOrder::Amp(AmpOrder::RightK64)
                });
                let weight_feed = streamed_k64_standard.map_or_else(
                    || {
                        right_bytes_consumed.div_ceil(
                            if resident_interleaved_weights || staged_weights {
                                IPU21_TARGET_COSTS.interleaved_load_bytes_per_cycle
                            } else {
                                IPU21_TARGET_COSTS.standard_load_bytes_per_cycle
                            },
                        )
                    },
                    |right| {
                        let owners = right
                            .format
                            .layout
                            .tiling
                            .axes
                            .iter()
                            .find(|axis| axis.axis == TensorAxis::FromEnd(2))
                            .map_or(1, |axis| u64::from(axis.partitions));
                        let local = right_bytes_consumed.div_ceil(owners);
                        let remote = right_bytes_consumed.saturating_sub(local);
                        local
                            .div_ceil(IPU21_TARGET_COSTS.standard_load_bytes_per_cycle)
                            .saturating_add(
                                remote
                                    .div_ceil(IPU21_TARGET_COSTS.interleaved_load_bytes_per_cycle),
                            )
                    },
                );
                let packing = if right.is_some_and(|right| {
                    right.format.precision == Precision::F16
                        && gemm_requires_panel_repacking(dispatch, right, output)
                }) {
                    right_bytes_consumed
                        .saturating_mul(2)
                        .div_ceil(IPU21_TARGET_COSTS.local_copy_bytes_per_cycle)
                } else {
                    0
                };
                let exchange = gemm_remote_bytes_per_tile(inputs, output)
                    .div_ceil(gemm_exchange_bytes_per_cycle(inputs));
                let calls = match dispatch {
                    OperatorDispatch::BlockedGemm {
                        inner_block,
                        output_column_block,
                        ..
                    } => k
                        .div_ceil(u64::from(*inner_block))
                        .saturating_mul(output_columns_per_tile)
                        .div_ceil(u64::from(*output_column_block))
                        .saturating_mul(IPU21_TARGET_COSTS.gemm_call_overhead_cycles),
                    OperatorDispatch::Pointwise { .. } => 0,
                };
                let memory = operator_memory_estimate(dispatch, inputs, output);
                let capacity_penalty = if memory.peak.fits_ipu21() {
                    0
                } else {
                    u64::MAX / 8
                };
                arithmetic
                    .max(weight_feed)
                    .saturating_add(packing)
                    .saturating_add(exchange)
                    .saturating_add(calls)
                    .saturating_add(capacity_penalty)
            }
            MidOperator::FlashAttention { .. } => elements.saturating_mul(8).div_ceil(32),
            MidOperator::Gelu => elements.saturating_mul(6).div_ceil(16),
            MidOperator::Add(_) => elements.div_ceil(16),
        }
    }

    fn cast_cycles(&self, input: &TensorType, to: Precision) -> u64 {
        let input_bytes = maximum_shard_bytes(input);
        let output_bytes = input_bytes
            .div_ceil(input.format.precision.bytes())
            .saturating_mul(to.bytes());
        input_bytes
            .saturating_add(output_bytes)
            .div_ceil(IPU21_TARGET_COSTS.local_copy_bytes_per_cycle)
    }

    fn rearrange_cycles(
        &self,
        shape: &TensorShape,
        precision: Precision,
        from: &Layout,
        to: &Layout,
    ) -> u64 {
        let Some(traffic) = conversion_traffic(shape, precision, from, to) else {
            return u64::MAX / 8;
        };
        let direct_retile = from.order == to.order;
        let exchange_cycles = traffic
            .source_payload_bytes
            .div_ceil(IPU21_TARGET_COSTS.exchange_bytes_per_cycle)
            .saturating_add(
                traffic
                    .remote_fragments
                    .saturating_mul(IPU21_TARGET_COSTS.exchange_fragment_overhead_cycles),
            );
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
            .saturating_add(
                local_calls.saturating_mul(IPU21_TARGET_COSTS.local_copy_call_overhead_cycles),
            );
        exchange_cycles.saturating_add(local_cycles)
    }
}
