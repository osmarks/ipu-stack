//! Structural memory, capacity, and communication estimates.

use super::exchange::ExchangeEndpointTraffic;
use crate::graph::TensorShape;
use crate::layout::{
    AMP_COLUMN_MICRO, AMP_INNER_BLOCK, Layout, MemoryClass, StorageOrder, TensorRegion, TensorType,
};
use crate::metrics::{MemoryEstimate, MemoryPeaks, MemoryUsage};
use crate::mid::{MidOperation, MidOperationKind, MidValue, MidValueId};
use crate::operator::{
    AllocationRequirements, GemmDistribution, MemoryElementRequirement, MemoryOperand,
    OperandMaterialization, OperatorDispatch, OperatorRequirements, Precision,
};
use ipu_target::hardware::HardwareTarget;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

fn layout_extents(shape: &TensorShape, layout: &Layout) -> Option<Vec<(u16, TensorRegion)>> {
    Some(
        layout
            .resolve(shape)
            .ok()?
            .shard_extents()
            .into_iter()
            .map(|shard| (shard.tile, shard.extents.logical()))
            .collect(),
    )
}

pub(crate) fn physical_elements(shape: &TensorShape, layout: &Layout) -> u64 {
    layout.resolve(shape).map_or_else(
        |_| {
            shape
                .elements()
                .saturating_mul(u64::from(layout.tiling.replicas))
        },
        |resolved| resolved.total_elements(),
    )
}

pub(crate) fn maximum_shard_bytes(tensor: &TensorType) -> u64 {
    let Ok(resolved) = tensor.format.layout.resolve(&tensor.shape) else {
        return u64::MAX;
    };
    resolved
        .maximum_tile_elements()
        .saturating_mul(tensor.format.precision.bytes())
}

/// Mean physical storage assigned to one active spatial tile. Replicated
/// layouts include each replica in both the total storage and tile count.
pub(crate) fn average_shard_bytes(tensor: &TensorType) -> u64 {
    tensor
        .format
        .layout
        .resolve(&tensor.shape)
        .map_or(u64::MAX, |resolved| {
            resolved
                .total_elements()
                .saturating_mul(tensor.format.precision.bytes())
                .div_ceil(u64::from(resolved.tile_count()).max(1))
        })
}

pub(crate) fn maximum_axis_shard_extent(tensor: &TensorType, axis: usize) -> u64 {
    tensor
        .format
        .layout
        .resolve(&tensor.shape)
        .ok()
        .and_then(|resolved| resolved.maximum_axis_extent(axis))
        .map_or(u64::MAX, u64::from)
}

pub(crate) fn gemm_partial_tensor(dispatch: &OperatorDispatch, output: &TensorType) -> TensorType {
    let OperatorDispatch::BlockedGemm(plan) = dispatch else {
        return output.clone();
    };
    plan.partial_tensor(output)
        .unwrap_or_else(|| output.clone())
}

pub(crate) fn tensor_memory(tensor: &TensorType) -> MemoryUsage {
    let mut usage = MemoryUsage::default();
    usage.add_class(
        tensor.format.layout.memory_class,
        maximum_shard_bytes(tensor),
    );
    usage
}

/// Interleaved working-set lower bound used to discard impossible parallel
/// GEMM grids before their concrete layouts and staging policies are expanded.
/// Exact candidates are checked again by [`operator_memory_estimate`].
fn allocation_memory(tensor: &TensorType, requirement: AllocationRequirements) -> MemoryUsage {
    let mut bytes =
        maximum_shard_bytes(tensor).saturating_add(u64::from(requirement.access_tail_bytes));
    if requirement.memory_element == MemoryElementRequirement::Distinct {
        let element = match tensor.format.layout.memory_class {
            MemoryClass::Standard => ipu_target::memory::TILE_MEMORY_ELEMENT_SIZE,
            MemoryClass::Interleaved => ipu_target::memory::IPU21_INTERLEAVED_ELEMENT_SIZE,
        };
        bytes = bytes.div_ceil(u64::from(element)) * u64::from(element);
    }
    let mut usage = MemoryUsage::default();
    usage.add_class(tensor.format.layout.memory_class, bytes);
    usage
}

fn allocation_requirements(
    operations: &[MidOperation],
) -> BTreeMap<MidValueId, AllocationRequirements> {
    let mut requirements = BTreeMap::<MidValueId, AllocationRequirements>::new();
    for operation in operations {
        if let Some(plan) = operation.operator_plan() {
            for (&id, operand) in operation.inputs.iter().zip(&plan.requirements.inputs) {
                let requirement = requirements.entry(id).or_default();
                requirement.merge(operand.allocation);
            }
            if let Some(&id) = operation.results.first() {
                let requirement = requirements.entry(id).or_default();
                requirement.merge(plan.requirements.output.allocation);
            }
            for operands in &plan.requirements.memory_space.distinct_element_groups {
                for operand in operands {
                    let id = match operand {
                        MemoryOperand::Output => operation.results.first().copied(),
                        MemoryOperand::Input(index) => {
                            operation.inputs.get(usize::from(*index)).copied()
                        }
                    };
                    if let Some(id) = id {
                        requirements
                            .entry(id)
                            .or_default()
                            .require_distinct_element();
                    }
                }
            }
        }
        if operation.conversion().is_some() {
            for id in operation
                .inputs
                .first()
                .into_iter()
                .chain(operation.results.first())
            {
                let requirement = requirements.entry(*id).or_default();
                requirement.alignment = requirement.alignment.max(8);
            }
        }
    }
    requirements
}

fn value_allocation(
    id: MidValueId,
    values: &[MidValue],
    requirements: &BTreeMap<MidValueId, AllocationRequirements>,
) -> MemoryUsage {
    allocation_memory(
        &values[id.index() as usize].tensor_type,
        requirements.get(&id).copied().unwrap_or_default(),
    )
}

fn maximum_standard_allocation(
    ids: &BTreeSet<MidValueId>,
    values: &[MidValue],
    requirements: &BTreeMap<MidValueId, AllocationRequirements>,
) -> u64 {
    ids.iter()
        .map(|&id| value_allocation(id, values, requirements).standard)
        .max()
        .unwrap_or(0)
}

pub(crate) fn operator_memory_estimate(
    dispatch: &OperatorDispatch,
    requirements: &OperatorRequirements,
    inputs: &[TensorType],
    output: &TensorType,
) -> MemoryEstimate {
    let live = inputs.iter().zip(&requirements.inputs).fold(
        tensor_memory(output),
        |usage, (input, requirement)| {
            if requirement.materialization == OperandMaterialization::DispatchSlices {
                usage
            } else {
                usage.saturating_add(tensor_memory(input))
            }
        },
    );
    let mut temporary = MemoryUsage::default();
    let mut maximum_standard_temporary_allocation = 0u64;
    if let (OperatorDispatch::BlockedGemm(plan), Some(first), Some(second)) =
        (dispatch, inputs.first(), inputs.get(1))
        && let GemmDistribution::ParallelReduction(reduction) = plan.geometry.distribution
    {
        let orientation = plan.geometry.orientation;
        let output_column_block = plan.geometry.block.output_columns;
        let [left, right] = orientation.physical_order([first, second]);
        let left_requirement = requirements.inputs.get(orientation.physical_left_input());
        let right_rank = right.shape.0.len();
        let right_inner_axis = orientation
            .row_axis()
            .resolve(right_rank)
            .expect("GEMM operand rank validated");
        let right_column_axis = orientation
            .column_axis()
            .resolve(right_rank)
            .expect("GEMM operand rank validated");
        let inner_blocks = right.shape.0[right_inner_axis].div_ceil(AMP_INNER_BLOCK);
        let column_blocks = right.shape.0[right_column_axis].div_ceil(output_column_block);
        let right_staging = u64::from(inner_blocks.div_ceil(u32::from(reduction.compute.inner)))
            .saturating_mul(u64::from(AMP_INNER_BLOCK))
            .saturating_mul(u64::from(
                column_blocks.div_ceil(u32::from(reduction.compute.columns)),
            ))
            .saturating_mul(u64::from(output_column_block))
            .saturating_mul(right.format.precision.bytes());
        let mut convolution = MemoryUsage::default();
        convolution.add_class(MemoryClass::Interleaved, right_staging);
        if left_requirement.is_some_and(|requirement| {
            requirement.materialization == OperandMaterialization::DispatchSlices
        }) {
            let requirement = left_requirement.expect("checked requirement");
            let mut left_staging = maximum_shard_bytes(left)
                .saturating_add(u64::from(requirement.allocation.access_tail_bytes));
            let left_must_be_distinct = requirements
                .memory_space
                .distinct_element_groups
                .iter()
                .any(|operands| {
                    operands.contains(&MemoryOperand::Input(
                        orientation.physical_left_input() as u16
                    ))
                });
            if left_must_be_distinct {
                left_staging = left_staging
                    .div_ceil(u64::from(ipu_target::memory::TILE_MEMORY_ELEMENT_SIZE))
                    .saturating_mul(u64::from(ipu_target::memory::TILE_MEMORY_ELEMENT_SIZE));
            }
            convolution.add_class(left.format.layout.memory_class, left_staging);
            if left.format.layout.memory_class == MemoryClass::Standard {
                maximum_standard_temporary_allocation =
                    maximum_standard_temporary_allocation.max(left_staging);
            }
        }
        // Compute retains one local partial alongside operand staging. The
        // later reduction ping-pongs an accumulator and result while its
        // staging policy bounds the simultaneously resident remote partials.
        let partial_bytes = maximum_shard_bytes(&gemm_partial_tensor(dispatch, output));
        let reduction_partial_bytes = if plan.geometry.result
            != (crate::GemmResultGrid {
                rows: reduction.compute.rows,
                columns: reduction.compute.columns,
            }) {
            maximum_shard_bytes(output)
        } else {
            partial_bytes
        };
        convolution.interleaved = convolution.interleaved.saturating_add(partial_bytes);
        let staged_remote_partials = match reduction.staging {
            crate::ReductionStaging::Complete => reduction.compute.inner.saturating_sub(1),
            crate::ReductionStaging::Streamed => 1,
        };
        let reduction = MemoryUsage {
            standard: reduction_partial_bytes
                .saturating_mul(u64::from(staged_remote_partials).saturating_add(2)),
            interleaved: partial_bytes,
        };
        temporary = MemoryUsage {
            standard: convolution.standard.max(reduction.standard),
            interleaved: convolution.interleaved.max(reduction.interleaved),
        };
    }
    if let (OperatorDispatch::BlockedGemm(plan), Some(left), Some(requirement)) =
        (dispatch, inputs.first(), requirements.inputs.first())
        && requirement.materialization == OperandMaterialization::DispatchSlices
        && !matches!(
            plan.geometry.distribution,
            GemmDistribution::ParallelReduction(_)
        )
    {
        let inner = left.shape.0.last().copied().map_or(1, u64::from).max(1);
        let bytes = maximum_shard_bytes(left)
            .div_ceil(inner)
            .saturating_mul(u64::from(plan.geometry.block.inner))
            .saturating_add(u64::from(requirement.allocation.access_tail_bytes));
        temporary.add_class(left.format.layout.memory_class, bytes);
    }
    if let (OperatorDispatch::BlockedGemm(plan), Some(right), Some(requirement)) =
        (dispatch, inputs.get(1), requirements.inputs.get(1))
        && right.format.precision == Precision::F16
        && !matches!(
            plan.geometry.distribution,
            GemmDistribution::ParallelReduction(_)
        )
        && gemm_uses_panel_buffer(dispatch, right, output)
    {
        // Each local output-column panel has one final kernel buffer reused
        // across K phases. Remote bytes can be exchanged directly into it.
        let output_columns =
            maximum_axis_shard_extent(output, output.shape.0.len().saturating_sub(1));
        let panels = output_columns.div_ceil(u64::from(plan.geometry.block.output_columns));
        let bytes = panels
            .saturating_mul(u64::from(plan.geometry.block.inner))
            .saturating_mul(u64::from(plan.geometry.block.output_columns))
            .saturating_mul(right.format.precision.bytes());
        temporary.add_class(requirement.local_staging.memory_class(), bytes);
    }
    if let OperatorDispatch::Attention(crate::AttentionPlan {
        blocking:
            crate::AttentionBlocking::Flash {
                query_rows: query_block_rows,
                key_rows: key_block_rows,
            },
        padding,
        ..
    }) = dispatch
    {
        let padded_query_dimension = &padding.query_dimension;
        let padded_value_dimension = &padding.value_dimension;
        let element_bytes = inputs.first().map_or(Precision::F16.bytes(), |input| {
            input.format.precision.bytes()
        });
        let key_rows = inputs
            .get(1)
            .and_then(|key| key.shape.0.get(key.shape.0.len().saturating_sub(2)))
            .copied()
            .map_or(1, u64::from);
        let blocks = key_rows.div_ceil(u64::from(*key_block_rows).max(1));
        let panels_per_block = u64::from(
            padded_query_dimension
                .div_ceil(AMP_COLUMN_MICRO)
                .saturating_add(padded_value_dimension.div_ceil(AMP_COLUMN_MICRO)),
        );
        let query_rows = output
            .shape
            .0
            .get(output.shape.0.len().saturating_sub(2))
            .copied()
            .map_or(1, u64::from);
        let query_partitions = query_rows
            .div_ceil(u64::from(*query_block_rows).max(1))
            .max(1);
        let prepared_panels_per_owner = blocks
            .saturating_mul(panels_per_block)
            .div_ceil(query_partitions);
        let panel_bytes = u64::from(*key_block_rows)
            .saturating_mul(u64::from(AMP_COLUMN_MICRO))
            .saturating_mul(element_bytes);
        // Every attention tile retains the current K and V panels. Prepared
        // panels are spread over query-tile owners; each needs both its
        // row-major gather buffer and its packed source until consumption.
        temporary.standard = temporary.standard.saturating_add(
            u64::from(*key_block_rows)
                .saturating_mul(u64::from(
                    padded_query_dimension.saturating_add(*padded_value_dimension),
                ))
                .saturating_mul(element_bytes)
                .saturating_add(
                    prepared_panels_per_owner
                        .saturating_mul(panel_bytes)
                        .saturating_mul(2),
                )
                .saturating_add(
                    u64::from(*query_block_rows)
                        .saturating_mul(u64::from(key_block_rows.saturating_add(16)))
                        .saturating_mul(element_bytes),
                ),
        );
        temporary.interleaved = temporary.interleaved.saturating_add(
            u64::from(*query_block_rows)
                .saturating_mul(u64::from((*padded_value_dimension).max(*key_block_rows)))
                .saturating_mul(Precision::F32.bytes()),
        );
    }
    if let OperatorDispatch::Attention(crate::AttentionPlan {
        blocking:
            crate::AttentionBlocking::Materialized {
                query_rows: query_block_rows,
                padded_key_rows,
            },
        padding,
        ..
    }) = dispatch
    {
        let padded_query_dimension = &padding.query_dimension;
        let padded_value_dimension = &padding.value_dimension;
        let element_bytes = inputs.first().map_or(Precision::F16.bytes(), |input| {
            input.format.precision.bytes()
        });
        let key_rows = inputs
            .get(1)
            .and_then(|key| key.shape.0.get(key.shape.0.len().saturating_sub(2)))
            .copied()
            .map_or(1, u64::from);
        let blocks = key_rows.div_ceil(u64::from(AMP_INNER_BLOCK));
        let panels_per_block = u64::from(
            padded_query_dimension
                .div_ceil(AMP_COLUMN_MICRO)
                .saturating_add(padded_value_dimension.div_ceil(AMP_COLUMN_MICRO)),
        );
        let query_rows = output
            .shape
            .0
            .get(output.shape.0.len().saturating_sub(2))
            .copied()
            .map_or(1, u64::from);
        let query_partitions = query_rows
            .div_ceil(u64::from(*query_block_rows).max(1))
            .max(1);
        let prepared_panels_per_owner = blocks
            .saturating_mul(panels_per_block)
            .div_ceil(query_partitions);
        let panel_bytes = u64::from(AMP_INNER_BLOCK)
            .saturating_mul(u64::from(AMP_COLUMN_MICRO))
            .saturating_mul(element_bytes);
        let operand_staging = u64::from(*padded_key_rows)
            .saturating_mul(u64::from(
                (*padded_query_dimension).max(*padded_value_dimension),
            ))
            .saturating_mul(element_bytes);
        maximum_standard_temporary_allocation =
            maximum_standard_temporary_allocation.max(operand_staging);
        let probability_state = u64::from(*query_block_rows)
            .saturating_mul(u64::from(padded_key_rows + AMP_COLUMN_MICRO))
            .saturating_mul(element_bytes);
        temporary.standard = temporary.standard.saturating_add(
            operand_staging
                .saturating_add(
                    prepared_panels_per_owner
                        .saturating_mul(panel_bytes)
                        .saturating_mul(2),
                )
                .saturating_add(
                    (probability_state > operand_staging)
                        .then_some(probability_state)
                        .unwrap_or(0),
                ),
        );
        temporary.interleaved = temporary.interleaved.saturating_add(
            u64::from(*query_block_rows)
                .saturating_mul(u64::from((*padded_value_dimension).max(*padded_key_rows)))
                .saturating_mul(element_bytes),
        );
    }
    MemoryEstimate {
        live,
        temporary,
        peak: live.saturating_add(temporary),
        maximum_standard_temporary_allocation,
    }
}

pub(crate) fn gemm_uses_panel_buffer(
    dispatch: &OperatorDispatch,
    right: &TensorType,
    output: &TensorType,
) -> bool {
    let OperatorDispatch::BlockedGemm(plan) = dispatch else {
        return false;
    };
    let inner_block = plan.geometry.block.inner;
    let orientation = plan.geometry.orientation;
    let rank = right.shape.0.len();
    let output_rank = output.shape.0.len();
    if rank < 2 || output_rank < 2 {
        return true;
    }
    let streamed = right
        .format
        .layout
        .tiling
        .axes
        .iter()
        .any(|axis| axis.axis == orientation.row_axis() && axis.partitions > 1);
    if streamed {
        return true;
    }
    if right.format.layout.memory_class == MemoryClass::Interleaved {
        return false;
    }
    let k = right.shape.0[orientation.row_axis().resolve(rank).unwrap()];
    let columns = maximum_axis_shard_extent(
        output,
        orientation.column_axis().resolve(output_rank).unwrap(),
    );
    k > inner_block && columns > 16
}

pub(crate) fn gemm_requires_panel_repacking(
    dispatch: &OperatorDispatch,
    right: &TensorType,
    output: &TensorType,
) -> bool {
    gemm_uses_panel_buffer(dispatch, right, output)
        && !matches!(right.format.layout.order, StorageOrder::Blocked(_))
}

pub(crate) fn gemm_exchange_phase_count(
    dispatch: &OperatorDispatch,
    inputs: &[TensorType],
    _output: &TensorType,
) -> u64 {
    let OperatorDispatch::BlockedGemm(plan) = dispatch else {
        return 0;
    };
    let inner_block = plan.geometry.block.inner;
    let orientation = plan.geometry.orientation;
    let Some(left) = inputs.get(orientation.physical_left_input()) else {
        return 0;
    };
    let Some(&inner) = orientation
        .column_axis()
        .resolve(left.shape.0.len())
        .ok()
        .and_then(|axis| left.shape.0.get(axis))
    else {
        return 0;
    };
    u64::from(inner).div_ceil(u64::from(inner_block))
}

pub(crate) fn conversion_memory_estimate(
    input: &TensorType,
    output: &TensorType,
    strategy: crate::ConversionStrategy,
) -> MemoryEstimate {
    let live = tensor_memory(input).saturating_add(tensor_memory(output));
    let mut staging_by_tile = HashMap::<u16, u64>::new();
    let mut maximum_standard_temporary_allocation = 0;
    if strategy == crate::ConversionStrategy::StageLogicalThenTransform {
        for shard in output
            .format
            .layout
            .resolve(&output.shape)
            .into_iter()
            .flat_map(|layout| layout.shard_extents())
        {
            let bytes = shard
                .extents
                .logical_elements()
                .saturating_mul(output.format.precision.bytes());
            let tile_bytes = staging_by_tile.entry(shard.tile).or_default();
            *tile_bytes = tile_bytes.saturating_add(bytes);
            maximum_standard_temporary_allocation =
                maximum_standard_temporary_allocation.max(bytes);
        }
    }
    let geometry_staging = if strategy.uses_intersections()
        && input.format.layout.tiling != output.format.layout.tiling
        && input
            .format
            .layout
            .tiling
            .tile_count
            .max(output.format.layout.tiling.tile_count)
            > 1
        && input.format.layout.order != output.format.layout.order
    {
        let source = maximum_shard_bytes(input);
        let destination = maximum_shard_bytes(output);
        maximum_standard_temporary_allocation = maximum_standard_temporary_allocation
            .max(source)
            .max(destination);
        source.saturating_add(destination)
    } else {
        0
    };
    let temporary = MemoryUsage {
        standard: staging_by_tile
            .into_values()
            .max()
            .unwrap_or(0)
            .saturating_add(geometry_staging),
        interleaved: 0,
    };
    MemoryEstimate {
        live,
        temporary,
        peak: live.saturating_add(temporary),
        maximum_standard_temporary_allocation,
    }
}

pub(crate) fn region_peak_memory(
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
    target: HardwareTarget,
) -> MemoryPeaks {
    region_peak_memory_with_multiplicity(
        initial,
        operations,
        outputs,
        values,
        &BTreeMap::new(),
        target,
    )
}

pub(crate) fn region_peak_memory_with_multiplicity(
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
    allocation_multiplicity: &BTreeMap<MidValueId, u32>,
    target: HardwareTarget,
) -> MemoryPeaks {
    let constraints = target.memory_constraints();
    let requirements = allocation_requirements(operations);
    let streamed_aliases = operations
        .iter()
        .filter_map(|operation| {
            if operation.conversion()?.1 != OperandMaterialization::DispatchSlices {
                return None;
            }
            Some((*operation.results.first()?, *operation.inputs.first()?))
        })
        .collect::<BTreeMap<_, _>>();
    let mut uses = BTreeMap::<MidValueId, u32>::new();
    for input in operations.iter().flat_map(operation_value_inputs) {
        *uses.entry(*input).or_default() += 1;
    }
    for output in outputs {
        *uses.entry(*output).or_default() += 1;
    }
    let mut live_values = BTreeSet::new();
    for id in initial {
        live_values.insert(*id);
    }
    let mut peaks = MemoryPeaks::default();
    let observe = |peaks: &mut MemoryPeaks, ids: &BTreeSet<MidValueId>, temporary: MemoryUsage| {
        let roots = ids
            .iter()
            .map(|id| allocation_root(*id, &streamed_aliases))
            .collect::<BTreeSet<_>>();
        let live = roots.iter().fold(MemoryUsage::default(), |usage, id| {
            let allocation = value_allocation(*id, values, &requirements);
            let copies = u64::from(allocation_multiplicity.get(id).copied().unwrap_or(1));
            usage.saturating_add(MemoryUsage {
                standard: allocation.standard.saturating_mul(copies),
                interleaved: allocation.interleaved.saturating_mul(copies),
            })
        });
        peaks.observe(
            live.saturating_add(temporary),
            maximum_standard_allocation(&roots, values, &requirements),
            constraints,
        );
    };
    observe(&mut peaks, &live_values, MemoryUsage::default());
    for operation in operations {
        let mut during_values = live_values.clone();
        for result in &operation.results {
            during_values.insert(*result);
        }
        let roots = during_values
            .iter()
            .map(|id| allocation_root(*id, &streamed_aliases))
            .collect::<BTreeSet<_>>();
        let live = roots.iter().fold(MemoryUsage::default(), |usage, id| {
            let allocation = value_allocation(*id, values, &requirements);
            let copies = u64::from(allocation_multiplicity.get(id).copied().unwrap_or(1));
            usage.saturating_add(MemoryUsage {
                standard: allocation.standard.saturating_mul(copies),
                interleaved: allocation.interleaved.saturating_mul(copies),
            })
        });
        peaks.observe(
            live.saturating_add(operation.metrics.memory.temporary),
            maximum_standard_allocation(&roots, values, &requirements).max(
                operation
                    .metrics
                    .memory
                    .maximum_standard_temporary_allocation,
            ),
            constraints,
        );
        for input in operation_value_inputs(operation) {
            if let Some(remaining) = uses.get_mut(input) {
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 {
                    live_values.remove(input);
                }
            }
        }
        for result in &operation.results {
            if uses.get(result).copied().unwrap_or(0) != 0 {
                live_values.insert(*result);
            }
        }
    }
    observe(&mut peaks, &live_values, MemoryUsage::default());
    let exchange_rows = operations
        .iter()
        .map(|operation| operation.metrics.cost.exchange_row_bytes(target))
        .fold(0u64, u64::saturating_add);
    peaks.exchange_rows = exchange_rows;
    peaks.standard = peaks.standard.saturating_add(exchange_rows);
    peaks.total = peaks.total.saturating_add(exchange_rows);
    peaks
}

fn operation_value_inputs(operation: &MidOperation) -> Vec<&MidValueId> {
    let mut inputs = operation.inputs.iter().collect::<Vec<_>>();
    if let MidOperationKind::Repeat(repeat) = &operation.kind {
        inputs.extend(repeat.iterated_inputs.iter().flatten());
    }
    inputs
}

fn allocation_root(mut id: MidValueId, aliases: &BTreeMap<MidValueId, MidValueId>) -> MidValueId {
    while let Some(source) = aliases.get(&id) {
        id = *source;
    }
    id
}

pub(crate) fn gemm_exchange_endpoint_traffic(
    dispatch: &OperatorDispatch,
    inputs: &[TensorType],
    compute_output: &TensorType,
    target: HardwareTarget,
) -> Option<ExchangeEndpointTraffic> {
    let OperatorDispatch::BlockedGemm(plan) = dispatch else {
        return Some(ExchangeEndpointTraffic::default());
    };
    let [first, second] = inputs else {
        return None;
    };
    let [left, right] = plan.geometry.orientation.physical_order([first, second]);
    if let GemmDistribution::ParallelReduction(reduction) = plan.geometry.distribution {
        // The parallel dispatch grid contains a K axis which is deliberately
        // absent from `compute_output`: every K group produces a partial with
        // the same logical output extent. Consequently, matching operand and
        // partial-output tile numbers cannot determine locality. The physical
        // left operand is replicated across output-column groups and the
        // physical right operand across output-row groups; only a shortfall in
        // those explicit replica counts creates operator-internal traffic.
        return Some(parallel_gemm_operand_traffic(
            left,
            reduction.compute.columns,
            right,
            reduction.compute.rows,
        ));
    }
    let orientation = plan.geometry.orientation;
    let left_rank = left.shape.0.len();
    let right_rank = right.shape.0.len();
    let output_rank = compute_output.shape.0.len();
    if left_rank < 2 || right_rank < 2 || output_rank < 2 {
        return None;
    }
    let left_row_axis = orientation.row_axis().resolve(left_rank).ok()?;
    let left_inner_axis = orientation.column_axis().resolve(left_rank).ok()?;
    let right_inner_axis = orientation.row_axis().resolve(right_rank).ok()?;
    let right_column_axis = orientation.column_axis().resolve(right_rank).ok()?;
    let output_row_axis = orientation.row_axis().resolve(output_rank).ok()?;
    let output_column_axis = orientation.column_axis().resolve(output_rank).ok()?;
    let output_plans = tile_axis_plans(compute_output)?;
    let left_plan = GemmOperandTrafficPlan::new(
        left,
        compute_output,
        left_row_axis,
        left_inner_axis,
        output_row_axis,
    )?;
    let right_plan = GemmOperandTrafficPlan::new(
        right,
        compute_output,
        right_column_axis,
        right_inner_axis,
        output_column_axis,
    )?;
    let transfer_bytes = u64::from(target.exchange().maximum_transfer_words) * 4;
    let mut traffic = ExchangeEndpointTraffic::default();
    let mut left_is_remote = false;
    let mut right_is_remote = false;
    for tile in 0..compute_output.format.layout.tiling.tile_count {
        let left_remote = left_plan.remote_bytes(tile, &output_plans, None)?;
        let right_remote = right_plan.remote_bytes(tile, &output_plans, None)?;
        left_is_remote |= left_remote != 0;
        right_is_remote |= right_remote != 0;
        let incoming = left_remote.saturating_add(right_remote);
        traffic.add_incoming(tile, incoming, incoming.div_ceil(transfer_bytes));
    }
    add_operand_outgoing_bus_work(&mut traffic, left, left_is_remote, transfer_bytes);
    add_operand_outgoing_bus_work(&mut traffic, right, right_is_remote, transfer_bytes);
    Some(traffic)
}

fn parallel_gemm_operand_traffic(
    left: &TensorType,
    left_required_replicas: u16,
    right: &TensorType,
    right_required_replicas: u16,
) -> ExchangeEndpointTraffic {
    let mut traffic = ExchangeEndpointTraffic::default();
    for (operand, required_replicas) in [
        (left, left_required_replicas),
        (right, right_required_replicas),
    ] {
        if operand.format.layout.tiling.replicas >= required_replicas {
            continue;
        }
        let stored_replicas = operand.format.layout.tiling.replicas.max(1);
        let base_tiles = operand
            .format
            .layout
            .tiling
            .tile_count
            .checked_div(stored_replicas)
            .unwrap_or(0);
        let Some(tile_count) = base_tiles.checked_mul(required_replicas) else {
            return ExchangeEndpointTraffic::from_maxima(
                u64::MAX / 16,
                u64::MAX / 16,
                u64::MAX / 16,
                u64::MAX / 16,
            );
        };
        let mut consumer_layout = operand.format.layout.clone();
        consumer_layout.tiling.tile_count = tile_count;
        consumer_layout.tiling.replicas = required_replicas;
        let Some(replication) = replica_shortfall_traffic(operand, &consumer_layout) else {
            return ExchangeEndpointTraffic::from_maxima(
                u64::MAX / 16,
                u64::MAX / 16,
                u64::MAX / 16,
                u64::MAX / 16,
            );
        };
        traffic.merge(&replication);
    }
    traffic
}

/// Replicating an otherwise unchanged layout only matches identical shard
/// extents. Account for those roles directly instead of running the general
/// all-pairs layout-intersection algorithm for every GEMM candidate.
fn replica_shortfall_traffic(
    operand: &TensorType,
    consumer_layout: &Layout,
) -> Option<ExchangeEndpointTraffic> {
    let sources = layout_extents(&operand.shape, &operand.format.layout)?;
    let destinations = layout_extents(&operand.shape, consumer_layout)?;
    let mut source_groups = HashMap::<TensorRegion, Vec<u16>>::new();
    for (tile, extents) in sources {
        source_groups.entry(extents).or_default().push(tile);
    }
    let mut destination_groups = HashMap::<TensorRegion, Vec<u16>>::new();
    for (tile, extents) in destinations {
        destination_groups.entry(extents).or_default().push(tile);
    }

    let mut traffic = ExchangeEndpointTraffic::default();
    for (extents, destination_tiles) in destination_groups {
        let source_tiles = source_groups.get(&extents)?;
        let remote_destinations = destination_tiles
            .into_iter()
            .filter(|tile| source_tiles.binary_search(tile).is_err())
            .collect::<Vec<_>>();
        if remote_destinations.is_empty() {
            continue;
        }
        let bytes = extents
            .logical_elements()
            .saturating_mul(operand.format.precision.bytes());
        traffic.add_outgoing(source_tiles[0] / 2, bytes, 1);
        for tile in remote_destinations {
            traffic.add_incoming(tile, bytes, 1);
        }
    }
    Some(traffic)
}

struct GemmOperandTrafficPlan<'a> {
    operand: &'a TensorType,
    operand_plans: Vec<TileAxisPlan>,
    operand_spatial_axis: usize,
    operand_inner_axis: usize,
    output_spatial_axis: usize,
    rank_offset: usize,
}

impl<'a> GemmOperandTrafficPlan<'a> {
    fn new(
        operand: &'a TensorType,
        output: &TensorType,
        operand_spatial_axis: usize,
        operand_inner_axis: usize,
        output_spatial_axis: usize,
    ) -> Option<Self> {
        let rank_offset = output.shape.0.len().checked_sub(operand.shape.0.len())?;
        let operand_plans = tile_axis_plans(operand)?;
        Some(Self {
            operand,
            operand_plans,
            operand_spatial_axis,
            operand_inner_axis,
            output_spatial_axis,
            rank_offset,
        })
    }

    fn remote_bytes(
        &self,
        tile: u16,
        output_plans: &[TileAxisPlan],
        inner_plan: Option<&TileAxisPlan>,
    ) -> Option<u64> {
        let mut required_elements = 1u64;
        let mut local_elements = 1u64;
        for axis in 0..self.operand.shape.0.len() {
            let extent = self.operand.shape.0[axis];
            let required = if axis == self.operand_spatial_axis {
                clipped_range(output_plans[self.output_spatial_axis].range(tile), extent)
            } else if axis == self.operand_inner_axis {
                inner_plan.map_or(0..extent, |plan| clipped_range(plan.range(tile), extent))
            } else if extent == 1 {
                0..1
            } else {
                let output_axis = axis.checked_add(self.rank_offset)?;
                clipped_range(output_plans[output_axis].range(tile), extent)
            };
            required_elements = required_elements
                .saturating_mul(u64::from(required.end.saturating_sub(required.start)));
            let local_length = if tile < self.operand.format.layout.tiling.tile_count {
                let local = clipped_range(self.operand_plans[axis].range(tile), extent);
                u64::from(
                    required
                        .end
                        .min(local.end)
                        .saturating_sub(required.start.max(local.start)),
                )
            } else {
                0
            };
            local_elements = local_elements.saturating_mul(local_length);
        }
        Some(
            required_elements
                .saturating_sub(local_elements)
                .saturating_mul(self.operand.format.precision.bytes()),
        )
    }
}

fn clipped_range(range: std::ops::Range<u32>, extent: u32) -> std::ops::Range<u32> {
    range.start.min(extent)..range.end.min(extent)
}

fn add_operand_outgoing_bus_work(
    traffic: &mut ExchangeEndpointTraffic,
    operand: &TensorType,
    remote: bool,
    transfer_bytes: u64,
) {
    if !remote {
        return;
    }
    for tile in 0..operand.format.layout.tiling.tile_count {
        let bytes = maximum_shard_bytes(operand);
        traffic.add_outgoing(tile / 2, bytes, bytes.div_ceil(transfer_bytes));
    }
}

#[derive(Clone)]
struct TileAxisPlan {
    layout: Arc<crate::layout::ResolvedLayout>,
    axis: usize,
}

impl TileAxisPlan {
    fn range(&self, tile: u16) -> std::ops::Range<u32> {
        self.layout
            .tile_axis_range(tile, self.axis)
            .expect("axis plan was resolved for this tile")
    }
}

fn tile_axis_plans(tensor: &TensorType) -> Option<Vec<TileAxisPlan>> {
    let layout = Arc::new(tensor.format.layout.resolve(&tensor.shape).ok()?);
    (0..tensor.shape.0.len())
        .map(|axis| {
            layout.maximum_axis_extent(axis)?;
            Some(TileAxisPlan {
                layout: Arc::clone(&layout),
                axis,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output_stationary_dispatch() -> OperatorDispatch {
        OperatorDispatch::BlockedGemm(crate::BlockedGemmPlan {
            kernel: crate::GemmKernelFamily {
                multiply: Precision::F16,
                accumulate: crate::AccumulationPrecision::F16,
                weights: crate::GemmWeightLoad::Standard,
            },
            geometry: crate::GemmGeometry {
                block: crate::GemmBlockShape {
                    inner: AMP_INNER_BLOCK,
                    output_columns: crate::layout::AMP_OUTPUT_COLUMN_BLOCK,
                },
                orientation: crate::GemmOrientation::Normal,
                result: crate::GemmResultGrid {
                    rows: 1,
                    columns: 1,
                },
                order: crate::GridOrder::ColumnsFast,
                distribution: GemmDistribution::OutputStationary,
            },
        })
    }

    fn parallel_reduction_dispatch(
        row_partitions: u16,
        column_partitions: u16,
        inner_partitions: u16,
    ) -> OperatorDispatch {
        let mut dispatch = output_stationary_dispatch();
        let OperatorDispatch::BlockedGemm(plan) = &mut dispatch else {
            unreachable!();
        };
        plan.geometry.result = crate::GemmResultGrid {
            rows: row_partitions,
            columns: column_partitions,
        };
        plan.geometry.distribution =
            GemmDistribution::ParallelReduction(crate::ParallelReductionPlan {
                compute: crate::GemmGrid {
                    rows: row_partitions,
                    columns: column_partitions,
                    inner: inner_partitions,
                },
                staging: crate::ReductionStaging::Streamed,
            });
        dispatch
    }

    #[test]
    fn randomized_average_shard_storage_covers_spatial_work() {
        let mut random = fastrand::Rng::with_seed(0x7370_6174_6961_6c77);
        for case in 0..32 {
            let row_partitions = 1_u16 << random.u32(0..=4);
            let column_partitions = 1_u16 << random.u32(0..=4);
            let tiles = row_partitions * column_partitions;
            let rows = u32::from(row_partitions) * random.u32(1..=8);
            let columns = u32::from(column_partitions) * 64 * random.u32(1..=4);
            let tensor = TensorType::new(
                [rows, columns],
                Precision::F16,
                Layout::amp_output_grid(
                    crate::GemmOrientation::Normal,
                    64,
                    tiles,
                    row_partitions,
                    column_partitions,
                    crate::operator::GridOrder::ColumnsFast,
                ),
            );
            let total = physical_elements(&tensor.shape, &tensor.format.layout)
                .saturating_mul(tensor.format.precision.bytes());
            let average = average_shard_bytes(&tensor);
            assert!(
                average.saturating_mul(u64::from(tiles)) >= total,
                "case {case}"
            );
            assert!(average <= maximum_shard_bytes(&tensor), "case {case}");
        }
    }

    #[test]
    fn randomized_gemm_endpoint_traffic_tracks_both_exchange_directions() {
        let mut random = fastrand::Rng::with_seed(0x6269_6469_7265_6374);
        for case in 0..32 {
            let row_partitions = 1_u16 << random.u32(1..=3);
            let column_partitions = 1_u16 << random.u32(1..=3);
            let tiles = row_partitions * column_partitions;
            let rows = u32::from(row_partitions) * random.u32(1..=8);
            let inner = AMP_INNER_BLOCK * random.u32(1..=4);
            let columns = u32::from(column_partitions)
                * crate::layout::AMP_OUTPUT_COLUMN_BLOCK
                * random.u32(1..=3);
            let output = TensorType::new(
                [1, rows, columns],
                Precision::F16,
                Layout::amp_output_grid(
                    crate::GemmOrientation::Normal,
                    crate::layout::AMP_OUTPUT_COLUMN_BLOCK,
                    tiles,
                    row_partitions,
                    column_partitions,
                    crate::operator::GridOrder::ColumnsFast,
                ),
            );
            let local_left = TensorType::new(
                [1, rows, inner],
                Precision::F16,
                Layout::amp_left_grid(
                    AMP_INNER_BLOCK as u16,
                    tiles,
                    row_partitions,
                    column_partitions,
                    crate::operator::GridOrder::ColumnsFast,
                ),
            );
            let local_right = TensorType::new(
                [1, inner, columns],
                Precision::F16,
                Layout::block_major_matrix_grid(
                    AMP_INNER_BLOCK as u16,
                    crate::layout::AMP_OUTPUT_COLUMN_BLOCK,
                    tiles,
                    row_partitions,
                    column_partitions,
                    crate::operator::GridOrder::ColumnsFast,
                ),
            );
            let dispatch = output_stationary_dispatch();
            let local = gemm_exchange_endpoint_traffic(
                &dispatch,
                &[local_left, local_right],
                &output,
                HardwareTarget::Ipu21,
            )
            .unwrap();
            assert!(local.is_empty(), "case {case}");

            let sharded_left = TensorType::new(
                [1, rows, inner],
                Precision::F16,
                Layout::amp_left(AMP_INNER_BLOCK as u16, row_partitions),
            );
            let sharded_right = TensorType::new(
                [1, inner, columns],
                Precision::F16,
                Layout::block_major_matrix_storage(
                    crate::GemmOrientation::Normal,
                    AMP_INNER_BLOCK as u16,
                    crate::layout::AMP_OUTPUT_COLUMN_BLOCK,
                    column_partitions,
                    1,
                    1,
                    MemoryClass::Standard,
                ),
            );
            let remote = gemm_exchange_endpoint_traffic(
                &dispatch,
                &[sharded_left, sharded_right],
                &output,
                HardwareTarget::Ipu21,
            )
            .unwrap();
            assert!(remote.maximum_outgoing_bytes() != 0, "case {case}");
            assert!(remote.maximum_incoming_bytes() != 0, "case {case}");
            assert_eq!(
                remote.maximum_payload_bytes(),
                remote
                    .maximum_outgoing_bytes()
                    .max(remote.maximum_incoming_bytes()),
                "case {case}"
            );
        }
    }

    #[test]
    fn randomized_parallel_gemm_traffic_tracks_replica_shortfalls() {
        let mut random = fastrand::Rng::with_seed(0x7265_706c_6963_6173);
        for case in 0..32 {
            let row_partitions = random.u16(2..=5);
            let column_partitions = random.u16(2..=5);
            let inner_partitions = random.u16(2..=5);
            let tiles = row_partitions * column_partitions * inner_partitions;
            let inner_block = 16 * random.u16(1..=4);
            let column_block = 16 * random.u32(1..=4);
            let rows = u32::from(row_partitions) * random.u32(1..=8);
            let inner = u32::from(inner_partitions) * u32::from(inner_block);
            let columns = u32::from(column_partitions) * column_block;
            let left = TensorType::new(
                [1, rows, inner],
                Precision::F16,
                Layout::amp_left_parallel_grid(
                    crate::GemmOrientation::Normal,
                    inner_block,
                    tiles,
                    row_partitions,
                    column_partitions,
                    inner_partitions,
                ),
            );
            let resident_right = TensorType::new(
                [1, inner, columns],
                Precision::F16,
                Layout::block_major_matrix_storage(
                    crate::GemmOrientation::Normal,
                    inner_block,
                    column_block,
                    column_partitions,
                    inner_partitions,
                    row_partitions,
                    MemoryClass::Standard,
                ),
            );
            let compute_output = TensorType::new(
                [1, rows, columns],
                Precision::F16,
                Layout::amp_left_result_grid(
                    crate::GemmOrientation::Normal,
                    column_block,
                    row_partitions * column_partitions,
                    row_partitions,
                    column_partitions,
                    crate::operator::GridOrder::ColumnsFast,
                ),
            );
            let dispatch =
                parallel_reduction_dispatch(row_partitions, column_partitions, inner_partitions);
            let resident = gemm_exchange_endpoint_traffic(
                &dispatch,
                &[left.clone(), resident_right],
                &compute_output,
                HardwareTarget::Ipu21,
            )
            .unwrap();
            assert!(resident.is_empty(), "case {case}");

            let sharded_right = TensorType::new(
                [1, inner, columns],
                Precision::F16,
                Layout::block_major_matrix_storage(
                    crate::GemmOrientation::Normal,
                    inner_block,
                    column_block,
                    column_partitions,
                    inner_partitions,
                    1,
                    MemoryClass::Standard,
                ),
            );
            let expected_incoming = maximum_shard_bytes(&sharded_right);
            let expected_outgoing = expected_incoming.saturating_mul(u64::from(
                sharded_right.format.layout.tiling.tile_count.min(2),
            ));
            let streamed = gemm_exchange_endpoint_traffic(
                &dispatch,
                &[left, sharded_right],
                &compute_output,
                HardwareTarget::Ipu21,
            )
            .unwrap();
            assert_eq!(
                streamed.maximum_incoming_bytes(),
                expected_incoming,
                "case {case}"
            );
            assert_eq!(
                streamed.maximum_outgoing_bytes(),
                expected_outgoing,
                "case {case}"
            );
        }
    }
}
