//! Analytical IPU21 kernel and conversion timing used during operator planning.

use super::exchange::{self, ExchangeEndpointTraffic};
use super::resources::{
    average_shard_bytes, gemm_exchange_endpoint_traffic, gemm_exchange_phase_count,
    gemm_partial_tensor, gemm_requires_panel_repacking, gemm_uses_panel_buffer,
    maximum_axis_shard_extent, maximum_shard_bytes, physical_elements,
};
#[cfg(test)]
use crate::MemorySpaceRequirements;
use crate::conversion::{ConversionStrategy, DeferredTransform, layout_conversion_strategy};
use crate::graph::TensorShape;
use crate::layout::{Layout, MemoryClass, NativeKernelOrder, StorageOrder, TensorAxis, TensorType};
use crate::metrics::{CostEstimate, ExchangeFootprint};
use crate::operator::{
    GemmDistribution, LocalOperandStaging, MidOperator, OperatorDispatch, OperatorRequirements,
    Precision,
};
use ipu_target::cost::HardwareCosts;
use ipu_target::hardware::HardwareTarget;

const IPU21_TARGET_COSTS: &HardwareCosts = HardwareTarget::Ipu21.costs();

pub trait CostModel: Sync {
    fn target(&self) -> HardwareTarget;

    fn operator_cycles(
        &self,
        operator: MidOperator,
        dispatch: &OperatorDispatch,
        requirements: &OperatorRequirements,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> u64;
    fn cast_cycles(&self, input: &TensorType, to: Precision) -> u64;
    fn operator_exchange_cycles(
        &self,
        _operator: MidOperator,
        _dispatch: &OperatorDispatch,
        _requirements: &OperatorRequirements,
        _inputs: &[TensorType],
        _output: &TensorType,
    ) -> u64 {
        0
    }
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
                        self.layout_conversion_cost(
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
    fn operator_transition_exchange_cycles(
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
                self.operator_exchange_cycles(operator, dispatch, requirements, inputs, output),
                |cycles, ((source, input), _)| {
                    cycles.saturating_add(
                        self.layout_conversion_cost(
                            &input.shape,
                            input.format.precision,
                            &source.format.layout,
                            &input.format.layout,
                        )
                        .exchange_cycles,
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
    fn deferred_input_exchange_cycles(
        &self,
        _transform: DeferredTransform,
        _source: &TensorType,
        _logical_output: &TensorType,
        _consumer_input: &TensorType,
        _consumer_dispatch: &OperatorDispatch,
        _producer_cycles: u64,
    ) -> u64 {
        0
    }
    fn rearrangement_cost(
        &self,
        source: &TensorType,
        destination: &TensorType,
        strategy: ConversionStrategy,
    ) -> CostEstimate;

    fn layout_conversion_cost(
        &self,
        shape: &TensorShape,
        precision: Precision,
        from: &Layout,
        to: &Layout,
    ) -> CostEstimate {
        let requested = layout_conversion_strategy(precision, from, to);
        self.rearrangement_cost(
            &TensorType::new(shape.0.clone(), precision, from.clone()),
            &TensorType::new(shape.0.clone(), precision, to.clone()),
            requested,
        )
    }
}

fn impossible_conversion_cost() -> CostEstimate {
    CostEstimate {
        cycles: u64::MAX / 8,
        exchange_cycles: u64::MAX / 8,
        exchange_footprint: ExchangeFootprint {
            phases: u64::MAX / 8,
            maximum_transfer_chunks_per_tile: u64::MAX / 8,
        },
    }
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

pub struct Ipu21CostModel;

fn attention_endpoint_traffic(
    inputs: &[TensorType],
    output: &TensorType,
    dispatch: &OperatorDispatch,
) -> Option<(ExchangeEndpointTraffic, u64)> {
    let OperatorDispatch::Attention(plan) = dispatch else {
        return Some((ExchangeEndpointTraffic::default(), 0));
    };
    let (query_block_rows, key_block_rows, phases) = match plan.blocking {
        crate::AttentionBlocking::Flash {
            query_rows,
            key_rows,
        } => (query_rows, key_rows, None),
        crate::AttentionBlocking::Materialized { query_rows, .. } => {
            (query_rows, crate::layout::AMP_INNER_BLOCK, Some(3))
        }
    };
    let padded_query_dimension = plan.padding.query_dimension;
    let padded_value_dimension = plan.padding.value_dimension;
    let key = inputs.get(1)?;
    let key_rows = key
        .shape
        .0
        .get(key.shape.0.len().checked_sub(2)?)
        .copied()
        .map(u64::from)?;
    let block_rows = u64::from(key_block_rows).max(1);
    let blocks = key_rows.div_ceil(block_rows);
    let panel_columns = u64::from(crate::layout::AMP_COLUMN_MICRO);
    let panels_per_block = u64::from(padded_query_dimension)
        .div_ceil(panel_columns)
        .saturating_add(u64::from(padded_value_dimension).div_ceil(panel_columns));
    let element_bytes = key.format.precision.bytes();
    let panel_bytes = block_rows
        .saturating_mul(panel_columns)
        .saturating_mul(element_bytes);
    let incoming = blocks
        .saturating_mul(panels_per_block)
        .saturating_mul(panel_bytes);
    let query_rows = output
        .shape
        .0
        .get(output.shape.0.len().checked_sub(2)?)
        .copied()
        .map(u64::from)?;
    let query_block_rows = u64::from(query_block_rows).max(1);
    let owners = query_rows.div_ceil(query_block_rows).max(1);
    let owner_panels = blocks.saturating_mul(panels_per_block).div_ceil(owners);
    let outgoing_bus = owner_panels.saturating_mul(panel_bytes).saturating_mul(2);
    Some((
        ExchangeEndpointTraffic::from_maxima(
            outgoing_bus,
            incoming,
            owner_panels.saturating_mul(2),
            blocks.saturating_mul(panels_per_block),
        ),
        phases.unwrap_or_else(|| blocks.saturating_add(2)),
    ))
}

fn maximum_shard_elements(tensor: &TensorType) -> u64 {
    maximum_shard_bytes(tensor).div_ceil(tensor.format.precision.bytes().max(1))
}

pub(crate) fn row_major_pack_cycles(tensor: &TensorType, elements: u64) -> u64 {
    let cycles_per_element = match tensor.format.layout.order {
        StorageOrder::Linear => return 0,
        StorageOrder::Native(NativeKernelOrder::TransposedRight) => {
            IPU21_TARGET_COSTS.contiguous_panel_pack_cycles_per_element
        }
        StorageOrder::Native(NativeKernelOrder::Left) => {
            IPU21_TARGET_COSTS.amp_left_pack_cycles_per_element
        }
        StorageOrder::Blocked(order)
            if order.is_matrix()
                && u32::from(order.block_shape[0]) == crate::layout::AMP_INNER_BLOCK
                && u32::from(order.block_shape[1]) == crate::layout::AMP_COLUMN_MICRO =>
        {
            return elements
                .saturating_mul(IPU21_TARGET_COSTS.block_major_pack_cycles_per_element)
                .saturating_add(IPU21_TARGET_COSTS.block_major_pack_startup_cycles);
        }
        StorageOrder::Blocked(_) | StorageOrder::Native(_) => {
            IPU21_TARGET_COSTS.indexed_f16_transform_cycles_per_element
        }
    };
    elements
        .saturating_mul(cycles_per_element)
        .saturating_add(IPU21_TARGET_COSTS.kernel_launch_cycles)
}

fn amp_unpack_cycles(tensor: &TensorType) -> u64 {
    if !matches!(
        tensor.format.layout.order,
        StorageOrder::Native(NativeKernelOrder::Output | NativeKernelOrder::TransposedLeft)
    ) {
        return 0;
    }
    maximum_shard_elements(tensor)
        .saturating_mul(IPU21_TARGET_COSTS.indexed_f16_transform_cycles_per_element)
        .saturating_add(IPU21_TARGET_COSTS.kernel_launch_cycles)
}

fn split_heads_word_fragment_cycles(output: &TensorType) -> Option<u64> {
    if output.format.precision != Precision::F16
        || !output
            .shape
            .0
            .last()
            .is_some_and(|width| width.is_multiple_of(2))
        || !matches!(
            output.format.layout.order,
            StorageOrder::Native(NativeKernelOrder::Left | NativeKernelOrder::TransposedRight)
        )
    {
        return None;
    }
    let physical_elements = maximum_shard_elements(output);
    let fragments = physical_elements.div_ceil(u64::from(crate::layout::AMP_COLUMN_MICRO));
    let clear_cycles = (output
        .format
        .layout
        .resolve(&output.shape)
        .ok()
        .is_some_and(|resolved| resolved.padded_shape() != &output.shape))
    .then(|| {
        maximum_shard_bytes(output)
            .div_ceil(8 * 6)
            .saturating_add(IPU21_TARGET_COSTS.kernel_launch_cycles)
    })
    .unwrap_or(0);
    Some(
        fragments
            .saturating_mul(IPU21_TARGET_COSTS.logical_fragment_cycles)
            .saturating_add(clear_cycles),
    )
}

fn split_head_panel_exchange_cycles(
    source: &TensorType,
    logical_output: &TensorType,
    destination: &TensorType,
) -> Option<u64> {
    let (&source_columns, &head_columns) = (source.shape.0.last()?, logical_output.shape.0.last()?);
    if head_columns == 0 || !source_columns.is_multiple_of(head_columns) {
        return None;
    }
    let groups = source_columns / head_columns;
    let source_grouped = source
        .format
        .layout
        .tiling
        .axes
        .iter()
        .find(|axis| axis.axis.resolve(source.shape.0.len()).ok() == Some(source.shape.0.len() - 1))
        .is_some_and(|axis| u32::from(axis.padding_groups) == groups);
    let panel = crate::layout::AMP_COLUMN_MICRO;
    let panels_per_group = head_columns.div_ceil(panel);
    let segments = if source_grouped {
        u64::from(groups).saturating_mul(u64::from(panels_per_group))
    } else {
        (0..groups).fold(0_u64, |total, group| {
            let base = group.saturating_mul(head_columns);
            let destination_boundaries = head_columns.saturating_sub(1) / panel;
            let first_source_boundary = (panel - base % panel) % panel;
            let source_boundaries = if first_source_boundary == 0 {
                destination_boundaries
            } else if first_source_boundary >= head_columns {
                0
            } else {
                1 + (head_columns - 1 - first_source_boundary) / panel
            };
            let shared_boundaries = if base.is_multiple_of(panel) {
                destination_boundaries
            } else {
                0
            };
            total.saturating_add(u64::from(
                1 + destination_boundaries + source_boundaries - shared_boundaries,
            ))
        })
    };
    let baseline_segments = u64::from(groups).saturating_mul(u64::from(panels_per_group));
    let baseline_cycles = split_heads_word_fragment_cycles(destination)?;
    Some(
        baseline_cycles
            .saturating_mul(segments)
            .div_ceil(baseline_segments.max(1)),
    )
}

fn amp_kernel_cycles(
    multiply: Precision,
    dispatch: &OperatorDispatch,
    right: Option<&TensorType>,
    staged_local_weights: bool,
    output_elements_per_tile: u64,
    output_columns_per_tile: u64,
    k: u64,
) -> Option<u64> {
    let OperatorDispatch::BlockedGemm(plan) = dispatch else {
        return None;
    };
    let inner_block = u64::from(plan.geometry.block.inner);
    let output_column_block = u64::from(plan.geometry.block.output_columns);
    if inner_block == 0
        || output_column_block == 0
        || output_columns_per_tile == 0
        || !inner_block.is_multiple_of(u64::from(crate::layout::AMP_COLUMN_MICRO))
        || !output_column_block.is_multiple_of(IPU21_TARGET_COSTS.amp_column_group_width)
    {
        return None;
    }
    let rows = output_elements_per_tile.div_ceil(output_columns_per_tile);
    let column_groups = output_column_block.div_ceil(IPU21_TARGET_COSTS.amp_column_group_width);
    let interleaved = staged_local_weights
        || right.is_some_and(|right| right.format.layout.memory_class == MemoryClass::Interleaved);
    let (row_cycles, group_cycles) = match multiply {
        Precision::F16 => (
            rows,
            if interleaved {
                IPU21_TARGET_COSTS.amp_interleaved_column_group_cycles
            } else {
                IPU21_TARGET_COSTS.amp_standard_column_group_cycles
            },
        ),
        // F32 AMP issues one quarter as many operations per cycle and feeds
        // twice as many weight bytes as F16 for the same matrix block.
        Precision::F32 => (
            rows.saturating_mul(4),
            IPU21_TARGET_COSTS
                .amp_standard_column_group_cycles
                .saturating_mul(2),
        ),
        Precision::F8F143 { .. } => return None,
    };
    let inner_micro_groups_per_call = inner_block / u64::from(crate::layout::AMP_COLUMN_MICRO);
    let call_cycles = IPU21_TARGET_COSTS.amp_call_cycles.saturating_add(
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

fn estimated_operator_exchange(
    dispatch: &OperatorDispatch,
    inputs: &[TensorType],
    output: &TensorType,
) -> CostEstimate {
    match dispatch {
        OperatorDispatch::BlockedGemm(crate::BlockedGemmPlan {
            geometry:
                crate::GemmGeometry {
                    result,
                    distribution: GemmDistribution::ParallelReduction(reduction),
                    ..
                },
            ..
        }) => {
            let compute_output = gemm_partial_tensor(dispatch, output);
            let endpoint = gemm_exchange_endpoint_traffic(
                dispatch,
                inputs,
                &compute_output,
                HardwareTarget::Ipu21,
            )
            .unwrap_or_else(|| {
                ExchangeEndpointTraffic::from_maxima(
                    u64::MAX / 16,
                    u64::MAX / 16,
                    u64::MAX / 16,
                    u64::MAX / 16,
                )
            });
            let remote_partials_per_stage = match reduction.staging {
                crate::ReductionStaging::Complete => reduction.compute.inner.saturating_sub(1),
                crate::ReductionStaging::Streamed => 1,
            };
            let reduction_epochs = reduction
                .compute
                .inner
                .saturating_sub(1)
                .div_ceil(remote_partials_per_stage.max(1));
            let reduction_partial_bytes = if *result
                != (crate::GemmResultGrid {
                    rows: reduction.compute.rows,
                    columns: reduction.compute.columns,
                }) {
                maximum_shard_bytes(output)
            } else {
                maximum_shard_bytes(&compute_output)
            };
            let reduction = ExchangeEndpointTraffic::from_maxima(
                reduction_partial_bytes.saturating_mul(2),
                u64::from(reduction.compute.inner.saturating_sub(1))
                    .saturating_mul(reduction_partial_bytes),
                2,
                u64::from(reduction.compute.inner.saturating_sub(1)),
            );
            exchange::cost(&endpoint, 1, HardwareTarget::Ipu21).sequence(exchange::cost(
                &reduction,
                u64::from(reduction_epochs),
                HardwareTarget::Ipu21,
            ))
        }
        OperatorDispatch::BlockedGemm(_) => {
            let compute_output = gemm_partial_tensor(dispatch, output);
            let traffic = gemm_exchange_endpoint_traffic(
                dispatch,
                inputs,
                &compute_output,
                HardwareTarget::Ipu21,
            )
            .unwrap_or_else(|| {
                ExchangeEndpointTraffic::from_maxima(
                    u64::MAX / 16,
                    u64::MAX / 16,
                    u64::MAX / 16,
                    u64::MAX / 16,
                )
            });
            exchange::cost(
                &traffic,
                gemm_exchange_phase_count(dispatch, inputs, &compute_output),
                HardwareTarget::Ipu21,
            )
        }
        OperatorDispatch::Attention(_) => attention_endpoint_traffic(inputs, output, dispatch)
            .map(|(traffic, phases)| exchange::cost(&traffic, phases, HardwareTarget::Ipu21))
            .unwrap_or_else(impossible_conversion_cost),
        OperatorDispatch::Pointwise(_) => CostEstimate::default(),
    }
}

fn deferred_split_input_cycles(
    source: &TensorType,
    logical_output: &TensorType,
    consumer_input: &TensorType,
    consumer_dispatch: &OperatorDispatch,
) -> Option<(u64, u64)> {
    if source.shape.0.len() != 3 || logical_output.shape.0.len() != 3 {
        return None;
    }
    let bytes = consumer_input.format.precision.bytes().max(1);
    let direct_panel_exchange = source
        .format
        .supports_f16_micro_panel_exchange(&consumer_input.format)
        && logical_output
            .shape
            .0
            .last()
            .is_some_and(|width| width.is_multiple_of(2));
    let source_unpack = if direct_panel_exchange {
        0
    } else {
        amp_unpack_cycles(source)
    };
    let rank = consumer_input.shape.0.len();
    let rows = consumer_input
        .shape
        .0
        .get(rank.saturating_sub(2))
        .copied()
        .map_or(1, u64::from);
    if matches!(
        consumer_input.format.layout.order,
        StorageOrder::Native(NativeKernelOrder::Left)
    ) {
        if direct_panel_exchange {
            let exchange =
                split_head_panel_exchange_cycles(source, logical_output, consumer_input)?;
            return Some((exchange, exchange));
        }
        let local_elements = maximum_shard_elements(consumer_input);
        let gather = local_elements
            .saturating_mul(bytes)
            .div_ceil(IPU21_TARGET_COSTS.local_copy_bytes_per_cycle);
        return Some((
            source_unpack
                .saturating_add(gather)
                .saturating_add(row_major_pack_cycles(consumer_input, local_elements)),
            0,
        ));
    }

    let (query_block_rows, key_block_rows) = match consumer_dispatch {
        OperatorDispatch::Attention(plan) => match plan.blocking {
            crate::AttentionBlocking::Flash {
                query_rows,
                key_rows,
            } => (u64::from(query_rows), u64::from(key_rows)),
            crate::AttentionBlocking::Materialized { query_rows, .. } => (
                u64::from(query_rows),
                u64::from(crate::layout::AMP_INNER_BLOCK),
            ),
        },
        _ => (rows, u64::from(crate::layout::AMP_INNER_BLOCK)),
    };
    let block_rows = key_block_rows.max(1);
    let blocks = rows.div_ceil(block_rows);
    let physical_columns = consumer_input
        .format
        .layout
        .resolve(&consumer_input.shape)
        .ok()?
        .padded_shape()
        .0
        .last()
        .copied()
        .map(u64::from)?;
    let panel_columns = u64::from(crate::layout::AMP_COLUMN_MICRO);
    let panels_per_block = physical_columns.div_ceil(panel_columns);
    let owners = rows.div_ceil(query_block_rows.max(1)).max(1);
    let panels_per_owner = blocks.saturating_mul(panels_per_block).div_ceil(owners);
    let panel_elements = block_rows.saturating_mul(panel_columns);
    let gather = if direct_panel_exchange {
        0
    } else {
        panel_elements
            .saturating_mul(bytes)
            .div_ceil(IPU21_TARGET_COSTS.local_copy_bytes_per_cycle)
    };
    let pack = if direct_panel_exchange {
        0
    } else {
        row_major_pack_cycles(consumer_input, panel_elements)
    };
    // A materialized block is one contiguous source span. Deferral assigns its
    // micro-panels to independent owners, so every source change after the
    // first adds another panel-serialization horizon on the shared exchange
    // paths even though the destination byte volume is unchanged.
    let fragmented_exchange = blocks
        .saturating_mul(panels_per_block.saturating_sub(1))
        .saturating_mul(panel_elements)
        .saturating_mul(bytes)
        .div_ceil(IPU21_TARGET_COSTS.exchange_bytes_per_cycle);
    let panel_exchange = direct_panel_exchange
        .then(|| split_head_panel_exchange_cycles(source, logical_output, consumer_input))
        .flatten()
        .unwrap_or(0);
    let fragmented_exchange = fragmented_exchange.saturating_add(panel_exchange);

    Some((
        source_unpack
            .saturating_add(panels_per_owner.saturating_mul(gather.saturating_add(pack)))
            .saturating_add(fragmented_exchange),
        fragmented_exchange,
    ))
}

/// Cheap ranking used before concrete blocked-GEMM layouts are expanded into
/// the full operator candidate set.
impl CostModel for Ipu21CostModel {
    fn target(&self) -> HardwareTarget {
        HardwareTarget::Ipu21
    }

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
                    OperatorDispatch::BlockedGemm(plan) => plan.geometry.orientation,
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
                    .resolve(&inputs[left_index].shape)
                    .map(|resolved| resolved.padded_shape().clone())
                    .unwrap_or_else(|_| inputs[left_index].shape.clone());
                let k = left_shape
                    .0
                    .get(left_shape.0.len().saturating_sub(left_inner_from_end))
                    .copied()
                    .unwrap_or(1) as u64;
                let compute_k = match dispatch {
                    OperatorDispatch::BlockedGemm(plan) => match plan.geometry.distribution {
                        GemmDistribution::ParallelReduction(reduction) => {
                            k.div_ceil(u64::from(reduction.compute.inner))
                        }
                        GemmDistribution::OutputStationary => k,
                    },
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
                // AMP left/output storage deliberately flattens outer axes
                // into its row dimension. Transposed and matrix-major orders
                // instead require one invocation sequence per physical
                // matrix; preserve that distinction in the call estimate.
                let matrices_per_tile = if matches!(
                    compute_output.format.layout.order,
                    StorageOrder::Native(NativeKernelOrder::Left | NativeKernelOrder::Output)
                ) {
                    1
                } else {
                    (0..compute_output.shape.0.len().saturating_sub(2))
                        .map(|axis| maximum_axis_shard_extent(&compute_output, axis))
                        .fold(1u64, u64::saturating_mul)
                        .max(1)
                };
                let kernel_output_elements = output_elements_per_tile.div_ceil(matrices_per_tile);
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
                    right.format.layout.memory_class == MemoryClass::Interleaved
                });
                let staged_weights = right.is_some_and(|right| {
                    gemm_uses_panel_buffer(dispatch, right, &compute_output)
                        && right.format.precision == Precision::F16
                });
                let staging = requirements
                    .inputs
                    .get(right_index)
                    .map_or_else(LocalOperandStaging::default, |requirement| {
                        requirement.local_staging
                    });
                let staged_local_weights = staged_weights && staging.stages_local();
                let streamed_blocked_standard = right.filter(|right| {
                    staged_weights
                        && staging.memory_class() == MemoryClass::Interleaved
                        && !staging.stages_local()
                        && right.format.layout.memory_class == MemoryClass::Standard
                        && matches!(right.format.layout.order, StorageOrder::Blocked(_))
                });
                let weight_feed = streamed_blocked_standard.map_or_else(
                    || {
                        right_bytes_consumed.div_ceil(
                            if resident_interleaved_weights
                                || (staged_weights
                                    && staging.memory_class() == MemoryClass::Interleaved)
                            {
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
                                    == if matches!(right.format.layout.order, StorageOrder::Blocked(order) if order.is_transposed_matrix()) {
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
                                == if matches!(right.format.layout.order, StorageOrder::Blocked(order) if order.is_transposed_matrix()) {
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
                    if staging.memory_class() == MemoryClass::Interleaved {
                        standard_to_interleaved_copy_cycles(right_bytes_consumed)
                    } else {
                        right_bytes_consumed
                            .saturating_mul(2)
                            .div_ceil(IPU21_TARGET_COSTS.local_copy_bytes_per_cycle)
                    }
                } else {
                    0
                };
                let exchange = estimated_operator_exchange(dispatch, inputs, output).cycles;
                let reduction_work = match dispatch {
                    OperatorDispatch::BlockedGemm(crate::BlockedGemmPlan {
                        geometry:
                            crate::GemmGeometry {
                                result,
                                distribution: GemmDistribution::ParallelReduction(reduction),
                                ..
                            },
                        ..
                    }) => {
                        let remote_partials_per_stage = match reduction.staging {
                            crate::ReductionStaging::Complete => {
                                reduction.compute.inner.saturating_sub(1)
                            }
                            crate::ReductionStaging::Streamed => 1,
                        };
                        let reduction_epochs = reduction
                            .compute
                            .inner
                            .saturating_sub(1)
                            .div_ceil(remote_partials_per_stage.max(1));
                        let partial_bytes = maximum_shard_bytes(&compute_output);
                        let reduction_partial_bytes = if *result
                            != (crate::GemmResultGrid {
                                rows: reduction.compute.rows,
                                columns: reduction.compute.columns,
                            }) {
                            maximum_shard_bytes(output)
                        } else {
                            partial_bytes
                        };
                        u64::from(
                            reduction
                                .compute
                                .inner
                                .saturating_sub(1)
                                .saturating_add(reduction_epochs),
                        )
                        .saturating_mul(reduction_partial_bytes)
                        .div_ceil(IPU21_TARGET_COSTS.reduction_output_bytes_per_cycle)
                        .saturating_add(
                            u64::from(reduction_epochs)
                                .saturating_mul(IPU21_TARGET_COSTS.kernel_launch_cycles),
                        )
                    }
                    _ => 0,
                };
                let calls = match dispatch {
                    OperatorDispatch::BlockedGemm(plan) => compute_k
                        .div_ceil(u64::from(plan.geometry.block.inner))
                        .saturating_mul(kernel_output_columns)
                        .div_ceil(u64::from(plan.geometry.block.output_columns))
                        .saturating_mul(matrices_per_tile)
                        .saturating_mul(IPU21_TARGET_COSTS.kernel_launch_cycles),
                    OperatorDispatch::Pointwise(_) => 0,
                    OperatorDispatch::Attention(_) => 0,
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
                .map(|cycles| cycles.saturating_mul(matrices_per_tile))
                .unwrap_or_else(|| arithmetic.max(weight_feed).saturating_add(calls));
                let result_copy = (compute_output.format.layout != output.format.layout)
                    .then(|| standard_to_interleaved_copy_cycles(maximum_shard_bytes(output)))
                    .unwrap_or(0);
                kernel
                    .saturating_add(standard_source_owner_penalty)
                    .saturating_add(packing)
                    .saturating_add(local_staging)
                    .saturating_add(exchange)
                    .saturating_add(reduction_work)
                    .saturating_add(result_copy)
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
                    OperatorDispatch::Attention(crate::AttentionPlan {
                        blocking:
                            crate::AttentionBlocking::Flash {
                                key_rows: key_block_rows,
                                ..
                            },
                        ..
                    }) => {
                        let arithmetic = query_rows
                            .saturating_mul(key_rows)
                            .saturating_mul(query_dimension.saturating_add(value_dimension))
                            .saturating_mul(2)
                            .div_ceil(128);
                        let blocks = key_rows.div_ceil(u64::from(*key_block_rows));
                        let exchange = estimated_operator_exchange(dispatch, inputs, output).cycles;
                        arithmetic
                            .saturating_add(
                                blocks
                                    .saturating_mul(4)
                                    .saturating_mul(IPU21_TARGET_COSTS.kernel_launch_cycles),
                            )
                            .saturating_add(exchange)
                    }
                    OperatorDispatch::Attention(crate::AttentionPlan {
                        blocking: crate::AttentionBlocking::Materialized { .. },
                        ..
                    }) => {
                        let arithmetic = query_rows
                            .saturating_mul(key_rows)
                            .saturating_mul(query_dimension.saturating_add(value_dimension))
                            .saturating_mul(2)
                            .div_ceil(128);
                        arithmetic
                            .saturating_add(
                                4u64.saturating_mul(IPU21_TARGET_COSTS.kernel_launch_cycles),
                            )
                            .saturating_add(
                                estimated_operator_exchange(dispatch, inputs, output).cycles,
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
        }
    }

    fn operator_exchange_cycles(
        &self,
        _operator: MidOperator,
        dispatch: &OperatorDispatch,
        _requirements: &OperatorRequirements,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> u64 {
        estimated_operator_exchange(dispatch, inputs, output).exchange_cycles
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
        let DeferredTransform::SplitLastAxisIntoLeading { parts } = transform;
        if parts == 0 {
            return producer_cycles;
        }
        deferred_split_input_cycles(source, logical_output, consumer_input, consumer_dispatch)
            .map_or(producer_cycles, |cost| cost.0)
    }

    fn deferred_input_exchange_cycles(
        &self,
        transform: DeferredTransform,
        source: &TensorType,
        logical_output: &TensorType,
        consumer_input: &TensorType,
        consumer_dispatch: &OperatorDispatch,
        _producer_cycles: u64,
    ) -> u64 {
        let DeferredTransform::SplitLastAxisIntoLeading { parts } = transform;
        if parts == 0 {
            return 0;
        }
        deferred_split_input_cycles(source, logical_output, consumer_input, consumer_dispatch)
            .map_or(0, |cost| cost.1)
    }

    fn operator_exchange_footprint(
        &self,
        _operator: MidOperator,
        dispatch: &OperatorDispatch,
        _requirements: &OperatorRequirements,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> ExchangeFootprint {
        estimated_operator_exchange(dispatch, inputs, output).exchange_footprint
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
        source: &TensorType,
        destination: &TensorType,
        strategy: ConversionStrategy,
    ) -> CostEstimate {
        if source.format.layout.resolve(&source.shape).is_err()
            || destination
                .format
                .layout
                .resolve(&destination.shape)
                .is_err()
        {
            return impossible_conversion_cost();
        }
        let source_bytes = maximum_shard_bytes(source);
        let destination_bytes = maximum_shard_bytes(destination);
        let changes_ownership = source.format.layout.tiling != destination.format.layout.tiling
            && source
                .format
                .layout
                .tiling
                .tile_count
                .max(destination.format.layout.tiling.tile_count)
                > 1;
        let maximum_chunk_bytes = u64::from(self.target().exchange().maximum_transfer_words) * 4;
        let replica_expansion = u64::from(destination.format.layout.tiling.replicas)
            .div_ceil(u64::from(source.format.layout.tiling.replicas));
        let endpoint_traffic = if changes_ownership {
            let contiguous_bytes = if source.format.layout.order == destination.format.layout.order
                && source.format.layout.order != StorageOrder::Linear
                && (source.format.layout.tiling.linear_grain().is_some()
                    || destination.format.layout.tiling.linear_grain().is_some())
            {
                source
                    .shape
                    .0
                    .len()
                    .checked_sub(1)
                    .map(|axis| {
                        maximum_axis_shard_extent(source, axis)
                            .min(maximum_axis_shard_extent(destination, axis))
                            .saturating_mul(source.format.precision.bytes())
                            .min(maximum_chunk_bytes)
                    })
                    .unwrap_or(maximum_chunk_bytes)
                    .max(4)
            } else {
                maximum_chunk_bytes
            };
            let outgoing_bytes = source_bytes.saturating_mul(replica_expansion);
            ExchangeEndpointTraffic::from_maxima(
                outgoing_bytes,
                destination_bytes,
                outgoing_bytes.div_ceil(contiguous_bytes),
                destination_bytes.div_ceil(contiguous_bytes),
            )
        } else {
            ExchangeEndpointTraffic::default()
        };
        let exchange = exchange::cost(
            &endpoint_traffic,
            u64::from(changes_ownership),
            HardwareTarget::Ipu21,
        );
        let changes_order = source.format.layout.order != destination.format.layout.order;
        let block_elements = |tensor: &TensorType| match tensor.format.layout.order {
            StorageOrder::Linear => maximum_shard_elements(tensor).max(1),
            StorageOrder::Native(NativeKernelOrder::Left | NativeKernelOrder::TransposedLeft) => {
                u64::from(crate::storage::amp_micro_dimension(tensor.format.precision))
            }
            StorageOrder::Native(
                NativeKernelOrder::Output | NativeKernelOrder::TransposedOutput,
            ) => u64::from(crate::AMP_COLUMN_MICRO),
            StorageOrder::Native(NativeKernelOrder::TransposedRight) => {
                u64::from(crate::storage::amp_micro_dimension(tensor.format.precision))
                    .saturating_mul(u64::from(crate::AMP_COLUMN_MICRO))
            }
            StorageOrder::Blocked(order) => {
                u64::from(crate::storage::amp_micro_dimension(tensor.format.precision))
                    .saturating_mul(u64::from(order.block_shape[1]))
            }
        };
        let transform_calls = if changes_order {
            maximum_shard_elements(source)
                .div_ceil(block_elements(source))
                .saturating_add(
                    maximum_shard_elements(destination).div_ceil(block_elements(destination)),
                )
        } else {
            0
        };
        let (local_bytes, local_calls): (u64, u64) = match strategy {
            ConversionStrategy::LocalKernel => (source_bytes.saturating_add(destination_bytes), 1),
            ConversionStrategy::StageLogicalThenTransform => (
                source_bytes.saturating_add(destination_bytes.saturating_mul(2)),
                transform_calls.max(2),
            ),
            ConversionStrategy::DirectLogical | ConversionStrategy::DirectRetile
                if changes_order =>
            {
                (
                    source_bytes.saturating_add(destination_bytes),
                    transform_calls.max(1),
                )
            }
            ConversionStrategy::DirectLogical | ConversionStrategy::DirectRetile
                if !changes_ownership =>
            {
                (source_bytes.saturating_add(destination_bytes), 1)
            }
            ConversionStrategy::DirectLogical | ConversionStrategy::DirectRetile => (0, 0),
        };
        let local_cycles = local_bytes
            .div_ceil(IPU21_TARGET_COSTS.local_copy_bytes_per_cycle)
            .saturating_add(local_calls.saturating_mul(IPU21_TARGET_COSTS.local_copy_call_cycles));
        CostEstimate {
            cycles: exchange.cycles.saturating_add(local_cycles),
            exchange_cycles: exchange.exchange_cycles,
            exchange_footprint: exchange.exchange_footprint,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        OperandRequirement, OperatorRequirements, OutputAliasing, PointwiseInputMapping,
        TensorFormat,
    };

    const CASES: usize = 32;

    fn pointwise_dispatch() -> OperatorDispatch {
        OperatorDispatch::Pointwise(PointwiseInputMapping::TileLocal)
    }

    fn pointwise_requirements(format: TensorFormat) -> OperatorRequirements {
        OperatorRequirements {
            inputs: vec![OperandRequirement::new(format.clone(), 8)],
            output: OperandRequirement::new(format, 8),
            output_aliasing: OutputAliasing::Fresh,
            memory_space: MemorySpaceRequirements::default(),
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
            let estimate = exchange::cost(&traffic, phases, HardwareTarget::Ipu21);
            let cycles = estimate.cycles;
            assert_eq!(
                cycles.saturating_sub(fixed),
                outgoing
                    .max(incoming)
                    .div_ceil(IPU21_TARGET_COSTS.exchange_bytes_per_cycle)
                    .max(
                        traffic
                            .maximum_fragments()
                            .saturating_mul(IPU21_TARGET_COSTS.logical_fragment_cycles)
                    ),
                "case {case}"
            );
            let reversed_traffic = ExchangeEndpointTraffic::from_maxima(
                incoming,
                outgoing,
                traffic.maximum_incoming_fragments(),
                traffic.maximum_outgoing_fragments(),
            );
            let reversed = exchange::cost(&reversed_traffic, phases, HardwareTarget::Ipu21).cycles;
            assert_eq!(cycles, reversed, "case {case}");
            let footprint = estimate.exchange_footprint;
            let transfer_bytes = u64::from(
                ipu_target::hardware::HardwareTarget::Ipu21
                    .exchange()
                    .maximum_transfer_words,
            ) * 4;
            assert_eq!(footprint.phases, phases, "case {case}");
            assert!(
                footprint.maximum_transfer_chunks_per_tile
                    >= outgoing.max(incoming).div_ceil(transfer_bytes),
                "case {case}"
            );
            assert!(
                footprint.maximum_transfer_chunks_per_tile >= traffic.maximum_fragments(),
                "case {case}"
            );
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
    fn randomized_group_padding_never_adds_split_panel_boundaries() {
        let mut random = fastrand::Rng::with_seed(0x6865_6164_5f67_7264);
        let mut improvements = 0;
        for case in 0..CASES * 8 {
            let groups = random.u16(2..=16);
            let head_columns = loop {
                let columns = random.u32(1..=64) * 2;
                if !columns.is_multiple_of(crate::layout::AMP_COLUMN_MICRO) {
                    break columns;
                }
            };
            let panels = head_columns.div_ceil(crate::layout::AMP_COLUMN_MICRO);
            let partitions_per_group = random.u16(1..=u16::try_from(panels).unwrap());
            let column_partitions = groups * partitions_per_group;
            let rows = random.u32(1..=128);
            let batch = random.u32(1..=4);
            let source_shape =
                TensorShape::new([batch, rows, u32::from(groups).saturating_mul(head_columns)]);
            let logical_output_shape =
                TensorShape::new([batch * u32::from(groups), rows, head_columns]);
            let ordinary_layout = Layout::amp_left_result_grid(
                crate::layout::AMP_COLUMN_MICRO,
                column_partitions,
                1,
                column_partitions,
                crate::GridOrder::ColumnsFast,
            );
            let mut grouped_layout = ordinary_layout.clone();
            grouped_layout.tiling.axes[0].padding_groups = groups;
            let ordinary = TensorType::new(source_shape.0.clone(), Precision::F16, ordinary_layout);
            let grouped = TensorType::new(source_shape.0.clone(), Precision::F16, grouped_layout);
            let logical_output = TensorType::new(
                logical_output_shape.0.clone(),
                Precision::F16,
                Layout::attention_query(u16::try_from(logical_output_shape.0[0]).unwrap(), 1),
            );
            let ordinary_cycles =
                split_head_panel_exchange_cycles(&ordinary, &logical_output, &logical_output)
                    .unwrap();
            let grouped_cycles =
                split_head_panel_exchange_cycles(&grouped, &logical_output, &logical_output)
                    .unwrap();
            improvements += usize::from(grouped_cycles < ordinary_cycles);
            assert!(
                grouped_cycles <= ordinary_cycles,
                "random case {case}: groups={groups} columns={head_columns} ordinary={ordinary_cycles} grouped={grouped_cycles}"
            );
        }
        assert!(improvements > 0);
    }
}
