//! Analytical IPU21 cycle estimation used during operator planning.

use crate::estimate::{
    average_shard_bytes, conversion_traffic, gemm_exchange_bytes_per_cycle,
    gemm_exchange_phase_count, gemm_remote_bytes_per_tile, gemm_requires_panel_repacking,
    gemm_uses_panel_buffer, maximum_axis_shard_extent, maximum_shard_bytes,
    operator_memory_estimate, physical_elements,
};
use crate::graph::TensorShape;
use crate::mid::{
    AmpOrder, ElementOrder, Layout, LocalOperandStaging, MemoryClass, MidOperator,
    OperatorDispatch, OperatorRequirements, Precision, TensorAxis, TensorType,
};
use std::cell::RefCell;
use std::collections::HashMap;

pub trait CostModel {
    fn operator_cycles(
        &self,
        operator: MidOperator,
        dispatch: &OperatorDispatch,
        requirements: &OperatorRequirements,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> u64;
    fn cast_cycles(&self, input: &TensorType, to: Precision) -> u64;
    fn operator_transition_cycles(
        &self,
        operator: MidOperator,
        dispatch: &OperatorDispatch,
        requirements: &OperatorRequirements,
        source_inputs: &[TensorType],
        inputs: &[TensorType],
        output: &TensorType,
    ) -> u64 {
        source_inputs
            .iter()
            .zip(inputs)
            .zip(&requirements.inputs)
            .filter(|((source, input), requirement)| {
                requirement.materialization == crate::OperandMaterialization::DispatchSlices
                    && source.format.layout != input.format.layout
            })
            .fold(
                self.operator_cycles(operator, dispatch, requirements, inputs, output),
                |cycles, ((source, input), _)| {
                    cycles.saturating_add(
                        self.rearrangement_cost(
                            &input.shape,
                            input.format.precision,
                            &source.format.layout,
                            &input.format.layout,
                        )
                        .cycles,
                    )
                },
            )
    }
    fn operator_exchange_footprint(
        &self,
        _operator: MidOperator,
        _dispatch: &OperatorDispatch,
        _requirements: &OperatorRequirements,
        _inputs: &[TensorType],
        _output: &TensorType,
    ) -> ExchangeFootprint {
        ExchangeFootprint::default()
    }
    fn rearrangement_cost(
        &self,
        shape: &TensorShape,
        precision: Precision,
        from: &Layout,
        to: &Layout,
    ) -> RearrangementCost;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RearrangementCost {
    pub cycles: u64,
    pub exchange_row_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExchangeFootprint {
    pub phases: u64,
    pub maximum_transfer_chunks_per_tile: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SpatialOccupancy {
    average_work: u64,
    critical_work: u64,
}

impl SpatialOccupancy {
    fn for_output(output: &TensorType) -> Self {
        let bytes = output.format.precision.bytes();
        Self {
            average_work: average_shard_bytes(output).div_ceil(bytes),
            critical_work: maximum_shard_bytes(output).div_ceil(bytes),
        }
    }

    const fn latency_work(self) -> u64 {
        // Device-wide dependencies make the longest spatial shard determine
        // latency. Keeping the mean alongside it makes the imbalance explicit
        // without incorrectly scoring latency as mean work.
        debug_assert!(self.average_work <= self.critical_work);
        self.critical_work
    }
}

impl ExchangeFootprint {
    pub const fn estimated_row_bytes(self) -> u64 {
        // A primitive plan contributes its synchronization-free body to the
        // consolidated per-phase row. The entry sync and terminal return are
        // shared by the caller and consolidated row respectively. Mid-level
        // byte volume cannot see where independently tiled source and
        // destination spans meet, or the address tables needed by shared
        // executable rows. Six encoded chunks per logical chunk tracks the
        // combined executable, offset, and per-use value storage on IPU21.
        let words_per_chunk = (ipu_exchange::PLAN_WORDS - 2) as u64;
        let encoded_chunks_per_logical_chunk = 6;
        self.phases
            .saturating_add(
                self.maximum_transfer_chunks_per_tile
                    .saturating_mul(words_per_chunk)
                    .saturating_mul(encoded_chunks_per_logical_chunk),
            )
            .saturating_mul(4)
    }
}

pub(crate) struct MemoizedCostModel<'a, C> {
    inner: &'a C,
    spatial_capacity: u16,
    rearrangements: RefCell<HashMap<(TensorShape, Precision, Layout, Layout), RearrangementCost>>,
}

impl<'a, C> MemoizedCostModel<'a, C> {
    pub(crate) fn new(inner: &'a C, spatial_capacity: u16) -> Self {
        Self {
            inner,
            spatial_capacity,
            rearrangements: RefCell::new(HashMap::new()),
        }
    }
}

impl<C: CostModel> CostModel for MemoizedCostModel<'_, C> {
    fn operator_cycles(
        &self,
        operator: MidOperator,
        dispatch: &OperatorDispatch,
        requirements: &OperatorRequirements,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> u64 {
        self.inner
            .operator_cycles(operator, dispatch, requirements, inputs, output)
    }

    fn cast_cycles(&self, input: &TensorType, to: Precision) -> u64 {
        self.inner.cast_cycles(input, to)
    }

    fn operator_exchange_footprint(
        &self,
        operator: MidOperator,
        dispatch: &OperatorDispatch,
        requirements: &OperatorRequirements,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> ExchangeFootprint {
        self.inner
            .operator_exchange_footprint(operator, dispatch, requirements, inputs, output)
    }

    fn rearrangement_cost(
        &self,
        shape: &TensorShape,
        precision: Precision,
        from: &Layout,
        to: &Layout,
    ) -> RearrangementCost {
        let key = (shape.clone(), precision, from.clone(), to.clone());
        if let Some(cost) = self.rearrangements.borrow().get(&key) {
            return *cost;
        }
        let mut cost = self.inner.rearrangement_cost(shape, precision, from, to);
        let active_tiles = from.tiling.tile_count.max(to.tiling.tile_count);
        // The inner model reports occupied work. Reduced-grid conversions
        // leave spatial issue slots idle, so convert that work into a phase
        // horizon using the occupancy of this particular planning target.
        cost.cycles = cost
            .cycles
            .saturating_mul(u64::from(self.spatial_capacity))
            .div_ceil(u64::from(active_tiles));
        self.rearrangements.borrow_mut().insert(key, cost);
        cost
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Ipu21CostModel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ipu21TargetCosts {
    pub exchange_bytes_per_cycle: u64,
    pub exchange_bus_sharing: u64,
    pub standard_load_bytes_per_cycle: u64,
    pub interleaved_load_bytes_per_cycle: u64,
    pub local_copy_bytes_per_cycle: u64,
    pub local_copy_call_cycles: u64,
    pub exchange_phase_cycles: u64,
    pub exchange_transfer_cycles: u64,
    pub kernel_launch_cycles: u64,
}

// Target::getExchangeBytesPerCycle and getTilesPerSharedExchangeBus.
pub const IPU21_TARGET_COSTS: Ipu21TargetCosts = Ipu21TargetCosts {
    exchange_bytes_per_cycle: 4,
    exchange_bus_sharing: 2,
    // Target::getMemcpyBytesPerCycle. Interleaved reads use both memory
    // elements, while an ordinary read or local copy uses one data path.
    standard_load_bytes_per_cycle: 8,
    interleaved_load_bytes_per_cycle: 16,
    local_copy_bytes_per_cycle: 8,
    // A finalized six-worker local-copy invocation, including supervisor and
    // worker rendezvous overhead, takes 288 tile cycles on IPU21.
    local_copy_call_cycles: 288,
    // Target::getGlobalSyncCycles.
    exchange_phase_cycles: 600,
    // Each logical remote span becomes an independently routed transfer. The
    // target's getMaxIPUSyncDelay is a conservative bound for its route/start
    // latency until topology-aware scheduling moves into the search cost.
    exchange_transfer_cycles: 126,
    // popops::internal::basicOpSupervisorOverhead(false).
    kernel_launch_cycles: 11,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AmpKernelCosts {
    call_cycles: u64,
    column_group_width: u64,
    interleaved_column_group_cycles: u64,
    standard_column_group_cycles: u64,
}

// Cycle counts of the generated IPU21 AMP kernel. A column group processes
// sixteen output columns and one 64-element K block. The row term represents
// AMP work; the remaining group cost is dominated by feeding its weights.
const IPU21_AMP_KERNEL_COSTS: AmpKernelCosts = AmpKernelCosts {
    call_cycles: 294,
    column_group_width: 16,
    interleaved_column_group_cycles: 1_157,
    standard_column_group_cycles: 1_280,
};

fn amp_kernel_cycles(
    multiply: Precision,
    dispatch: &OperatorDispatch,
    right: Option<&TensorType>,
    staged_local_weights: bool,
    output_elements_per_tile: u64,
    output_columns_per_tile: u64,
    k: u64,
) -> Option<u64> {
    let OperatorDispatch::BlockedGemm {
        inner_block,
        output_column_block,
        ..
    } = dispatch
    else {
        return None;
    };
    let inner_block = u64::from(*inner_block);
    let output_column_block = u64::from(*output_column_block);
    if inner_block == 0
        || output_column_block == 0
        || output_columns_per_tile == 0
        || !inner_block.is_multiple_of(64)
        || !output_column_block.is_multiple_of(IPU21_AMP_KERNEL_COSTS.column_group_width)
    {
        return None;
    }
    let rows = output_elements_per_tile.div_ceil(output_columns_per_tile);
    let column_groups = output_column_block.div_ceil(IPU21_AMP_KERNEL_COSTS.column_group_width);
    let interleaved = staged_local_weights
        || right
            .is_some_and(|right| right.format.layout.memory_class == MemoryClass::Ipu21Interleaved);
    let (row_cycles, group_cycles) = match multiply {
        Precision::F16 => (
            rows,
            if interleaved {
                IPU21_AMP_KERNEL_COSTS.interleaved_column_group_cycles
            } else {
                IPU21_AMP_KERNEL_COSTS.standard_column_group_cycles
            },
        ),
        // F32 AMP issues one quarter as many operations per cycle and feeds
        // twice as many weight bytes as F16 for the same matrix block.
        Precision::F32 => (
            rows.saturating_mul(4),
            IPU21_AMP_KERNEL_COSTS
                .standard_column_group_cycles
                .saturating_mul(2),
        ),
        Precision::F8F143 { .. } => return None,
    };
    let inner_blocks_per_call = inner_block / 64;
    let call_cycles = IPU21_AMP_KERNEL_COSTS.call_cycles.saturating_add(
        inner_blocks_per_call.saturating_mul(
            output_column_block
                .saturating_mul(row_cycles)
                .saturating_add(column_groups.saturating_mul(group_cycles)),
        ),
    );
    Some(
        k.div_ceil(inner_block)
            .saturating_mul(output_columns_per_tile.div_ceil(output_column_block))
            .saturating_mul(call_cycles),
    )
}

fn standard_to_interleaved_copy_cycles(bytes: u64) -> u64 {
    // The paced parallel helper sustains about six bytes per cycle including
    // worker scheduling (1,308 measured cycles for an 8 KiB panel). Keep this
    // separate from the target's ideal memcpy bandwidth.
    bytes
        .div_ceil(6)
        .saturating_add(IPU21_TARGET_COSTS.kernel_launch_cycles)
}

impl CostModel for Ipu21CostModel {
    fn operator_cycles(
        &self,
        operator: MidOperator,
        dispatch: &OperatorDispatch,
        requirements: &OperatorRequirements,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> u64 {
        let elements = physical_elements(&output.shape, &output.format.layout);
        let spatial_occupancy = SpatialOccupancy::for_output(output);
        let spatial_occupancy_adjusted_elements = spatial_occupancy.latency_work();
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
                let output_elements_per_tile = spatial_occupancy_adjusted_elements;
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
                let staged_local_weights = staged_weights
                    && requirements.inputs.get(1).is_some_and(|requirement| {
                        requirement.local_staging == LocalOperandStaging::MatchRemote
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
                let local_staging = if staged_local_weights {
                    // The source-owner role rotates with the streamed block,
                    // but the critical-path tile performs one local population
                    // for every block it computes rather than one divided share
                    // of the operator's K traffic.
                    standard_to_interleaved_copy_cycles(right_bytes_consumed)
                } else {
                    0
                };
                let exchange = gemm_remote_bytes_per_tile(inputs, output)
                    .div_ceil(gemm_exchange_bytes_per_cycle(inputs))
                    .saturating_add(
                        gemm_exchange_phase_count(dispatch, inputs, output)
                            .saturating_mul(IPU21_TARGET_COSTS.exchange_phase_cycles),
                    );
                let calls = match dispatch {
                    OperatorDispatch::BlockedGemm {
                        inner_block,
                        output_column_block,
                        ..
                    } => k
                        .div_ceil(u64::from(*inner_block))
                        .saturating_mul(output_columns_per_tile)
                        .div_ceil(u64::from(*output_column_block))
                        .saturating_mul(IPU21_TARGET_COSTS.kernel_launch_cycles),
                    OperatorDispatch::Pointwise { .. } => 0,
                };
                let kernel = amp_kernel_cycles(
                    multiply,
                    dispatch,
                    right,
                    staged_local_weights,
                    output_elements_per_tile,
                    output_columns_per_tile,
                    k,
                )
                .unwrap_or_else(|| arithmetic.max(weight_feed).saturating_add(calls));
                let memory = operator_memory_estimate(dispatch, requirements, inputs, output);
                let capacity_penalty = if memory.peak.fits_ipu21() {
                    0
                } else {
                    u64::MAX / 8
                };
                kernel
                    .saturating_add(packing)
                    .saturating_add(local_staging)
                    .saturating_add(exchange)
                    .saturating_add(capacity_penalty)
            }
            MidOperator::FlashAttention { .. } => elements
                .saturating_mul(8)
                .div_ceil(32)
                .saturating_add(IPU21_TARGET_COSTS.kernel_launch_cycles),
            // The exact scalar implementation is compute-bound at roughly
            // ten tile cycles per element across the six workers.
            MidOperator::Gelu => spatial_occupancy_adjusted_elements
                .saturating_mul(10)
                .saturating_add(IPU21_TARGET_COSTS.kernel_launch_cycles),
            MidOperator::Add(_) => spatial_occupancy_adjusted_elements
                .div_ceil(16)
                .saturating_add(IPU21_TARGET_COSTS.kernel_launch_cycles),
        }
    }

    fn operator_exchange_footprint(
        &self,
        _operator: MidOperator,
        dispatch: &OperatorDispatch,
        _requirements: &OperatorRequirements,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> ExchangeFootprint {
        let phases = gemm_exchange_phase_count(dispatch, inputs, output);
        if phases == 0 {
            return ExchangeFootprint::default();
        }
        let remote_bytes = gemm_remote_bytes_per_tile(inputs, output);
        if remote_bytes == u64::MAX {
            return ExchangeFootprint::default();
        }
        let transfer_bytes = u64::from(ipu_exchange::MAX_TRANSFER_WORDS) * 4;
        ExchangeFootprint {
            phases,
            maximum_transfer_chunks_per_tile: remote_bytes.div_ceil(transfer_bytes).max(phases),
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
            .saturating_add(IPU21_TARGET_COSTS.kernel_launch_cycles)
    }

    fn rearrangement_cost(
        &self,
        shape: &TensorShape,
        precision: Precision,
        from: &Layout,
        to: &Layout,
    ) -> RearrangementCost {
        let Some(traffic) = conversion_traffic(shape, precision, from, to) else {
            return RearrangementCost {
                cycles: u64::MAX / 8,
                exchange_row_bytes: u64::MAX / 8,
            };
        };
        let direct_retile = from.order == to.order;
        // Independent source/destination roles execute spatially, but adjacent
        // IPU21 tiles share an exchange bus. Treat at most one paired role as
        // concurrent; the per-role maxima retain skew from uneven layouts.
        let spatial_payload = traffic
            .source_payload_bytes
            .div_ceil(IPU21_TARGET_COSTS.exchange_bus_sharing)
            .max(traffic.maximum_source_payload_bytes)
            .max(traffic.maximum_remote_destination_bytes);
        let spatial_fragments = traffic
            .remote_fragments
            .div_ceil(IPU21_TARGET_COSTS.exchange_bus_sharing)
            .max(traffic.maximum_source_fragments)
            .max(traffic.maximum_routed_fragments);
        let occupied_exchange_cycles = spatial_payload
            .div_ceil(IPU21_TARGET_COSTS.exchange_bytes_per_cycle)
            .saturating_add(
                spatial_fragments.saturating_mul(IPU21_TARGET_COSTS.exchange_transfer_cycles),
            );
        let exchange_cycles =
            occupied_exchange_cycles.saturating_add(if traffic.remote_fragments == 0 {
                0
            } else {
                IPU21_TARGET_COSTS.exchange_phase_cycles
            });
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
        // Runtime latency benefits from spatially concurrent roles, whereas
        // every global fragment still contributes encoded row storage.
        let encoded_work = traffic
            .source_payload_bytes
            .div_ceil(IPU21_TARGET_COSTS.exchange_bytes_per_cycle)
            .saturating_add(
                traffic
                    .remote_fragments
                    .max(traffic.maximum_routed_fragments)
                    .saturating_mul(IPU21_TARGET_COSTS.exchange_transfer_cycles),
            )
            .saturating_add(if traffic.remote_fragments == 0 {
                0
            } else {
                IPU21_TARGET_COSTS.exchange_phase_cycles
            });
        let cycles_per_word = IPU21_TARGET_COSTS
            .exchange_transfer_cycles
            .saturating_mul(16);
        RearrangementCost {
            cycles: exchange_cycles.saturating_add(local_cycles),
            exchange_row_bytes: if from.tiling == to.tiling {
                0
            } else {
                encoded_work.div_ceil(cycles_per_word).saturating_mul(4)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        OperandRequirement, OperatorRequirements, OutputAliasing, PointwiseInputMapping,
        TensorFormat, TileKernelSpec,
    };

    const CASES: usize = 32;

    fn pointwise_dispatch() -> OperatorDispatch {
        OperatorDispatch::Pointwise {
            kernel: TileKernelSpec::Gelu,
            input_mapping: PointwiseInputMapping::TileLocal,
        }
    }

    fn pointwise_requirements(format: TensorFormat) -> OperatorRequirements {
        OperatorRequirements {
            inputs: vec![OperandRequirement::new(format.clone(), 8)],
            output: OperandRequirement::new(format, 8),
            output_aliasing: OutputAliasing::Fresh,
            memory_relations: Vec::new(),
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
            for operator in [
                MidOperator::Gelu,
                MidOperator::Add(crate::AddOptions::default()),
            ] {
                let sharded_cycles = Ipu21CostModel.operator_cycles(
                    operator,
                    &pointwise_dispatch(),
                    &pointwise_requirements(sharded.format.clone()),
                    std::slice::from_ref(&sharded),
                    &sharded,
                );
                let unsharded_cycles = Ipu21CostModel.operator_cycles(
                    operator,
                    &pointwise_dispatch(),
                    &pointwise_requirements(unsharded.format.clone()),
                    std::slice::from_ref(&unsharded),
                    &unsharded,
                );
                assert!(sharded_cycles <= unsharded_cycles, "case {case}");
            }
        }
    }

    #[test]
    fn randomized_conversion_duty_penalizes_fragmented_spatial_work() {
        let mut random = fastrand::Rng::with_seed(0x6475_7479_6672_6167);
        for case in 0..CASES {
            let row_partitions = 1_u16 << random.u32(1..=4);
            let column_partitions = 1_u16 << random.u32(1..=4);
            let tiles = row_partitions * column_partitions;
            let rows = u32::from(row_partitions.max(column_partitions)) * random.u32(1..=4);
            let columns = u32::from(row_partitions.max(column_partitions)) * random.u32(1..=4) * 64;
            let shape = TensorShape(vec![rows, columns]);
            let fragmented = Layout::amp_output_grid(64, tiles, row_partitions, column_partitions);
            let aligned = Layout::amp_output_grid(64, tiles, column_partitions, row_partitions);
            let destination =
                Layout::amp_output_replicated_grid(tiles, column_partitions, row_partitions);
            let fragmented_cycles = Ipu21CostModel
                .rearrangement_cost(&shape, Precision::F16, &fragmented, &destination)
                .cycles;
            let aligned_cycles = Ipu21CostModel
                .rearrangement_cost(&shape, Precision::F16, &aligned, &destination)
                .cycles;
            assert!(
                fragmented_cycles >= aligned_cycles,
                "case {case}: fragmented={fragmented_cycles} aligned={aligned_cycles}"
            );
        }
    }
}
