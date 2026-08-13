//! Analytical IPU21 cycle estimation used during operator planning.

use crate::estimate::{
    average_shard_bytes, conversion_traffic, gemm_exchange_bytes_per_cycle,
    gemm_exchange_phase_count, gemm_partial_tensor, gemm_remote_bytes_per_tile,
    gemm_requires_panel_repacking, gemm_uses_panel_buffer, maximum_axis_shard_extent,
    maximum_shard_bytes, operator_memory_estimate, physical_elements,
};
use crate::graph::TensorShape;
use crate::mid::{
    AmpOrder, BlockMajorOrder, DeferredTransform, ElementOrder, GemmDistribution, Layout,
    LocalOperandStaging, MemoryClass, MidOperator, OperatorDispatch, OperatorRequirements,
    Precision, TensorAxis, TensorType,
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
    /// Cost of producing dispatch-sized consumer slices through a deferred
    /// logical transform. The default preserves the unfused producer estimate;
    /// target models may price the actual fused staging and exchange path.
    fn deferred_input_cycles(
        &self,
        _transform: DeferredTransform,
        _source: &TensorType,
        _logical_output: &TensorType,
        _consumer_input: &TensorType,
        _consumer_dispatch: &OperatorDispatch,
        producer_cycles: u64,
    ) -> u64 {
        producer_cycles
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

    fn deferred_input_cycles(
        &self,
        transform: DeferredTransform,
        source: &TensorType,
        logical_output: &TensorType,
        consumer_input: &TensorType,
        consumer_dispatch: &OperatorDispatch,
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
    pub reduction_output_bytes_per_cycle: u64,
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
    // Reduction-add reads two partials and writes one. Current IPU21 profiles
    // sustain roughly one output byte per cycle after all three interleaved
    // streams and worker imbalance are included.
    reduction_output_bytes_per_cycle: 1,
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
    interleaved_column_group_cycles: 940,
    standard_column_group_cycles: 1_063,
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
        || !inner_block.is_multiple_of(u64::from(crate::mid::AMP_COLUMN_MICRO))
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
    let inner_micro_groups_per_call = inner_block / u64::from(crate::mid::AMP_COLUMN_MICRO);
    let call_cycles = IPU21_AMP_KERNEL_COSTS.call_cycles.saturating_add(
        inner_micro_groups_per_call.saturating_mul(
            output_column_block
                .saturating_mul(row_cycles)
                .div_ceil(4)
                .saturating_add(column_groups.saturating_mul(group_cycles).div_ceil(4)),
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

fn reduction_tree_critical_path(partitions: u16, fan_in: u16) -> (u64, u64) {
    if partitions < 2 || fan_in < 2 {
        return (0, 0);
    }
    let mut remaining = partitions;
    let mut rounds = 0u64;
    let mut additions = 0u64;
    while remaining > 1 {
        additions = additions.saturating_add(u64::from(remaining.min(fan_in) - 1));
        remaining = remaining.div_ceil(fan_in);
        rounds += 1;
    }
    (rounds, additions)
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
                let orientation = match dispatch {
                    OperatorDispatch::BlockedGemm { orientation, .. } => *orientation,
                    _ => crate::GemmOrientation::Normal,
                };
                let (left_index, right_index, left_inner_from_end, output_column_from_end) =
                    match orientation {
                        crate::GemmOrientation::Normal => (0, 1, 1, 1),
                        crate::GemmOrientation::Swapped => (1, 0, 2, 2),
                    };
                let compute_output = gemm_partial_tensor(dispatch, output);
                let output_elements_per_tile =
                    SpatialOccupancy::for_output(&compute_output).latency_work();
                let left_shape = inputs[left_index]
                    .format
                    .layout
                    .padded_shape(&inputs[left_index].shape)
                    .unwrap_or_else(|_| inputs[left_index].shape.clone());
                let k = left_shape
                    .0
                    .get(left_shape.0.len().saturating_sub(left_inner_from_end))
                    .copied()
                    .unwrap_or(1) as u64;
                let compute_k = match dispatch {
                    OperatorDispatch::BlockedGemm {
                        distribution:
                            GemmDistribution::ParallelReduction {
                                inner_partitions, ..
                            },
                        ..
                    } => k.div_ceil(u64::from(*inner_partitions)),
                    _ => k,
                };
                let flops_per_cycle: u64 = match multiply {
                    Precision::F8F143 { .. } => 256,
                    Precision::F16 => 128,
                    Precision::F32 => 32,
                };
                let output_columns_per_tile = maximum_axis_shard_extent(
                    &compute_output,
                    output.shape.0.len().saturating_sub(output_column_from_end),
                );
                let kernel_output_elements = output_elements_per_tile;
                let kernel_output_columns = output_columns_per_tile;
                let arithmetic = kernel_output_elements
                    .saturating_mul(2)
                    .saturating_mul(compute_k)
                    .div_ceil(flops_per_cycle);
                let right = inputs.get(right_index);
                let right_bytes_consumed = right.map_or(u64::MAX, |right| {
                    kernel_output_columns
                        .saturating_mul(compute_k)
                        .saturating_mul(right.format.precision.bytes())
                });
                let resident_interleaved_weights = right.is_some_and(|right| {
                    right.format.layout.memory_class == MemoryClass::Ipu21Interleaved
                });
                let staged_weights = right.is_some_and(|right| {
                    gemm_uses_panel_buffer(dispatch, right, &compute_output)
                        && right.format.precision == Precision::F16
                });
                let staged_local_weights = staged_weights
                    && requirements
                        .inputs
                        .get(right_index)
                        .is_some_and(|requirement| {
                            requirement.local_staging == LocalOperandStaging::MatchRemote
                        });
                let streamed_blocked_standard = right.filter(|right| {
                    staged_weights
                        && right.format.layout.memory_class == MemoryClass::Ipu21Standard
                        && matches!(right.format.layout.order, ElementOrder::BlockMajor(_))
                });
                let weight_feed = streamed_blocked_standard.map_or_else(
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
                            .find(|axis| {
                                axis.axis
                                    == if matches!(
                                        right.format.layout.order,
                                        ElementOrder::BlockMajor(
                                            BlockMajorOrder::TransposedMatrix { .. }
                                        )
                                    ) {
                                        TensorAxis::FromEnd(1)
                                    } else {
                                        TensorAxis::FromEnd(2)
                                    }
                            })
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
                let standard_source_owner_penalty = streamed_blocked_standard.map_or(0, |right| {
                    let owners = right
                        .format
                        .layout
                        .tiling
                        .axes
                        .iter()
                        .find(|axis| {
                            axis.axis
                                == if matches!(
                                    right.format.layout.order,
                                    ElementOrder::BlockMajor(
                                        BlockMajorOrder::TransposedMatrix { .. }
                                    )
                                ) {
                                    TensorAxis::FromEnd(1)
                                } else {
                                    TensorAxis::FromEnd(2)
                                }
                        })
                        .map_or(1, |axis| u64::from(axis.partitions));
                    let local_panel_bytes = right_bytes_consumed.div_ceil(owners);
                    let per_phase_penalty = local_panel_bytes
                        .div_ceil(IPU21_TARGET_COSTS.standard_load_bytes_per_cycle)
                        .saturating_sub(
                            local_panel_bytes
                                .div_ceil(IPU21_TARGET_COSTS.interleaved_load_bytes_per_cycle),
                        );
                    // The source owner changes between phases. Device latency
                    // follows that phase-local critical role rather than the
                    // accumulated work of any one physical tile.
                    per_phase_penalty.saturating_mul(gemm_exchange_phase_count(
                        dispatch,
                        inputs,
                        &compute_output,
                    ))
                });
                let packing = if right.is_some_and(|right| {
                    right.format.precision == Precision::F16
                        && gemm_requires_panel_repacking(dispatch, right, &compute_output)
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
                let exchange = gemm_remote_bytes_per_tile(inputs, &compute_output)
                    .div_ceil(gemm_exchange_bytes_per_cycle(inputs))
                    .saturating_add(
                        gemm_exchange_phase_count(dispatch, inputs, &compute_output)
                            .saturating_mul(IPU21_TARGET_COSTS.exchange_phase_cycles),
                    );
                let exchange = match dispatch {
                    OperatorDispatch::BlockedGemm {
                        output_column_block: _,
                        distribution:
                            GemmDistribution::ParallelReduction {
                                column_partitions,
                                inner_partitions,
                                reduction_fan_in,
                                ..
                            },
                        ..
                    } => {
                        let partitions = u64::from(*inner_partitions);
                        let columns = output
                            .shape
                            .0
                            .get(output.shape.0.len().saturating_sub(output_column_from_end))
                            .copied()
                            .map_or(1, u64::from);
                        let exchange_epochs = 1u64;
                        let local_k = k.div_ceil(partitions);
                        let weight_bytes = local_k
                            .saturating_mul(columns.div_ceil(u64::from(*column_partitions)))
                            .saturating_mul(inputs[right_index].format.precision.bytes());
                        let activation_bytes = maximum_shard_bytes(&inputs[left_index]);
                        let (rounds, reduction_additions) =
                            reduction_tree_critical_path(*inner_partitions, *reduction_fan_in);
                        let partial_bytes = maximum_shard_bytes(&compute_output);
                        weight_bytes
                            .saturating_add(activation_bytes)
                            .div_ceil(IPU21_TARGET_COSTS.exchange_bytes_per_cycle)
                            .saturating_add(
                                reduction_additions
                                    .saturating_mul(partial_bytes)
                                    .div_ceil(IPU21_TARGET_COSTS.exchange_bytes_per_cycle),
                            )
                            .saturating_add(
                                exchange_epochs
                                    .saturating_mul(rounds.saturating_add(1))
                                    .saturating_mul(IPU21_TARGET_COSTS.exchange_phase_cycles),
                            )
                            .saturating_add(
                                rounds.saturating_mul(
                                    partial_bytes
                                        .div_ceil(
                                            IPU21_TARGET_COSTS.reduction_output_bytes_per_cycle,
                                        )
                                        .saturating_add(IPU21_TARGET_COSTS.kernel_launch_cycles),
                                ),
                            )
                    }
                    _ => exchange,
                };
                let calls = match dispatch {
                    OperatorDispatch::BlockedGemm {
                        inner_block,
                        output_column_block,
                        ..
                    } => compute_k
                        .div_ceil(u64::from(*inner_block))
                        .saturating_mul(kernel_output_columns)
                        .div_ceil(u64::from(*output_column_block))
                        .saturating_mul(IPU21_TARGET_COSTS.kernel_launch_cycles),
                    OperatorDispatch::Pointwise { .. } => 0,
                    OperatorDispatch::BlockedAttention { .. } => 0,
                    OperatorDispatch::SplitHeads => 0,
                };
                let kernel = amp_kernel_cycles(
                    multiply,
                    dispatch,
                    right,
                    staged_local_weights,
                    kernel_output_elements,
                    kernel_output_columns,
                    compute_k,
                )
                .unwrap_or_else(|| arithmetic.max(weight_feed).saturating_add(calls));
                let memory = operator_memory_estimate(dispatch, requirements, inputs, output);
                let capacity_penalty = if memory.peak.fits_ipu21() {
                    0
                } else {
                    u64::MAX / 8
                };
                let result_remap = (compute_output.format.layout != output.format.layout)
                    .then(|| {
                        maximum_shard_bytes(&compute_output)
                            .max(maximum_shard_bytes(output))
                            .div_ceil(IPU21_TARGET_COSTS.exchange_bytes_per_cycle)
                            .saturating_add(IPU21_TARGET_COSTS.exchange_phase_cycles)
                    })
                    .unwrap_or(0);
                kernel
                    .saturating_add(standard_source_owner_penalty)
                    .saturating_add(packing)
                    .saturating_add(local_staging)
                    .saturating_add(exchange)
                    .saturating_add(result_remap)
                    .saturating_add(capacity_penalty)
            }
            MidOperator::FlashAttention { .. } => {
                let query = inputs.first().map(|input| &input.shape.0);
                let key = inputs.get(1).map(|input| &input.shape.0);
                let value = inputs.get(2).map(|input| &input.shape.0);
                let key_rows = key
                    .and_then(|shape| shape.get(shape.len().saturating_sub(2)))
                    .copied()
                    .map_or(1, u64::from);
                let query_dimension = query
                    .and_then(|shape| shape.last())
                    .copied()
                    .map_or(1, u64::from);
                let value_dimension = value
                    .and_then(|shape| shape.last())
                    .copied()
                    .map_or(1, u64::from);
                let output_values_per_query = value_dimension.max(1);
                let query_rows = elements.div_ceil(output_values_per_query);
                match dispatch {
                    OperatorDispatch::BlockedAttention { key_block_rows, .. } => {
                        let arithmetic = query_rows
                            .saturating_mul(key_rows)
                            .saturating_mul(query_dimension.saturating_add(value_dimension))
                            .saturating_mul(2)
                            .div_ceil(128);
                        let blocks = key_rows.div_ceil(u64::from(*key_block_rows));
                        arithmetic.saturating_add(
                            blocks
                                .saturating_mul(4)
                                .saturating_mul(IPU21_TARGET_COSTS.kernel_launch_cycles),
                        )
                    }
                    _ => query_rows
                        .saturating_mul(key_rows)
                        .saturating_mul(query_dimension.saturating_add(value_dimension))
                        .saturating_mul(4)
                        .div_ceil(6)
                        .saturating_add(IPU21_TARGET_COSTS.kernel_launch_cycles),
                }
            }
            // The exact scalar implementation is compute-bound at roughly
            // ten tile cycles per element across the six workers.
            MidOperator::Gelu => spatial_occupancy_adjusted_elements
                .saturating_mul(10)
                .saturating_add(IPU21_TARGET_COSTS.kernel_launch_cycles),
            MidOperator::Add(_) => spatial_occupancy_adjusted_elements
                .div_ceil(16)
                .saturating_add(IPU21_TARGET_COSTS.kernel_launch_cycles),
            MidOperator::SplitHeads(_) => {
                let bytes = elements.saturating_mul(output.format.precision.bytes());
                bytes
                    .div_ceil(IPU21_TARGET_COSTS.exchange_bytes_per_cycle)
                    .saturating_add(IPU21_TARGET_COSTS.exchange_phase_cycles)
            }
        }
    }

    fn deferred_input_cycles(
        &self,
        transform: DeferredTransform,
        source: &TensorType,
        logical_output: &TensorType,
        consumer_input: &TensorType,
        _consumer_dispatch: &OperatorDispatch,
        producer_cycles: u64,
    ) -> u64 {
        let DeferredTransform::SplitLastAxisIntoLeading { parts } = transform;
        if parts == 0 || source.shape.0.len() != 3 || logical_output.shape.0.len() != 3 {
            return producer_cycles;
        }
        let bytes = consumer_input.format.precision.bytes().max(1);
        let source_work = maximum_shard_bytes(source).div_ceil(bytes);
        let rank = consumer_input.shape.0.len();
        let rows = consumer_input
            .shape
            .0
            .get(rank.saturating_sub(2))
            .copied()
            .map_or(1, u64::from);
        let columns = consumer_input.shape.0.last().copied().map_or(1, u64::from);
        let panel_columns = columns.min(u64::from(crate::mid::AMP_COLUMN_MICRO));
        let (slices, slice_rows, packing_cycles_per_element) =
            match consumer_input.format.layout.order {
                ElementOrder::Amp(AmpOrder::TransposedRight) => (
                    rows.div_ceil(u64::from(crate::mid::AMP_INNER_BLOCK)),
                    rows.min(u64::from(crate::mid::AMP_INNER_BLOCK)),
                    3,
                ),
                ElementOrder::BlockMajor(BlockMajorOrder::Matrix { row_block, .. }) => (
                    rows.div_ceil(u64::from(row_block)),
                    rows.min(u64::from(row_block)),
                    4,
                ),
                _ => (
                    1,
                    maximum_shard_bytes(consumer_input)
                        .div_ceil(bytes)
                        .div_ceil(columns),
                    2,
                ),
            };
        let panel_elements = slice_rows.saturating_mul(panel_columns);
        let gather = panel_elements.div_ceil(4);
        let pack = panel_elements.saturating_mul(packing_cycles_per_element);
        let broadcast = panel_elements
            .saturating_mul(bytes)
            .div_ceil(IPU21_TARGET_COSTS.exchange_bytes_per_cycle);
        // The producer's packed output must first become an addressable logical
        // view. Thereafter independent panel owners overlap within each slice;
        // only the critical panel and the two exchange epochs contribute.
        source_work.saturating_mul(2).saturating_add(
            slices.saturating_mul(
                gather
                    .saturating_add(pack)
                    .saturating_add(broadcast)
                    .saturating_add(IPU21_TARGET_COSTS.exchange_phase_cycles.saturating_mul(2)),
            ),
        )
    }

    fn operator_exchange_footprint(
        &self,
        _operator: MidOperator,
        dispatch: &OperatorDispatch,
        _requirements: &OperatorRequirements,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> ExchangeFootprint {
        let phases = match dispatch {
            OperatorDispatch::SplitHeads => 1,
            OperatorDispatch::BlockedAttention { key_block_rows, .. } => inputs
                .get(1)
                .and_then(|key| key.shape.0.get(key.shape.0.len().saturating_sub(2)))
                .copied()
                .map_or(0, u64::from)
                .div_ceil(u64::from(*key_block_rows).max(1))
                .saturating_add(2),
            _ => gemm_exchange_phase_count(dispatch, inputs, output),
        };
        let phases = match dispatch {
            OperatorDispatch::BlockedGemm {
                output_column_block: _,
                distribution:
                    GemmDistribution::ParallelReduction {
                        inner_partitions,
                        reduction_fan_in,
                        ..
                    },
                ..
            } => {
                let epochs = 1u64;
                let (rounds, _) =
                    reduction_tree_critical_path(*inner_partitions, *reduction_fan_in);
                epochs.saturating_mul(rounds.saturating_add(1))
            }
            _ => phases,
        };
        if phases == 0 {
            return ExchangeFootprint::default();
        }
        if let OperatorDispatch::BlockedAttention {
            key_block_rows,
            padded_query_dimension,
            padded_value_dimension,
            ..
        } = dispatch
        {
            let key_rows = inputs
                .get(1)
                .and_then(|key| key.shape.0.get(key.shape.0.len().saturating_sub(2)))
                .copied()
                .map_or(0, u64::from);
            let blocks = key_rows.div_ceil(u64::from(*key_block_rows).max(1));
            let panels = u64::from(
                padded_query_dimension
                    .saturating_add(*padded_value_dimension)
                    .div_ceil(crate::mid::AMP_COLUMN_MICRO),
            );
            // Prepared K/V owners receive several independently sharded
            // source fragments in the common gather epoch. Ownership is
            // rotated, so roughly half the panels contribute to the busiest
            // row; retaining this estimate prevents the planner from treating
            // the consolidated exchange program as free SRAM.
            return ExchangeFootprint {
                phases,
                maximum_transfer_chunks_per_tile: blocks.saturating_mul(panels.div_ceil(2)),
            };
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
            let fragmented = Layout::amp_output_grid(
                64,
                tiles,
                row_partitions,
                column_partitions,
                crate::mid::GridOrder::ColumnsFast,
            );
            let aligned = Layout::amp_output_grid(
                64,
                tiles,
                column_partitions,
                row_partitions,
                crate::mid::GridOrder::ColumnsFast,
            );
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
