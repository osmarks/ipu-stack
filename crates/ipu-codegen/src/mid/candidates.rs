//! Shape-dependent implementation choices and local candidate pruning.

use super::*;

mod attention;
mod demand;
pub(super) use demand::{OutputDemand, OutputDemands};
mod search;
pub(super) use search::CandidateSearch;

// Explicit diagnostic plans bypass heuristic memory rejection; concrete
// placement still decides whether their storage fits.
pub(super) fn contains_forced_plan(operations: &[MidOperation], config: &PipelineConfig) -> bool {
    operations.iter().any(|operation| {
        operation.source.is_some_and(|source| {
            config
                .gemm_plan_constraints
                .iter()
                .any(|constraint| constraint.source_operation == source.index())
        })
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GroupedOutputLayout {
    pub(super) groups: u16,
    pub(super) physical_lane_multiple: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ParallelGridProxy {
    pub(super) compute: u64,
    pub(super) communication: u64,
    pub(super) temporary_bytes: u64,
    pub(super) unused_tiles: u64,
    pub(super) row_partitions: u16,
    pub(super) column_partitions: u16,
    pub(super) inner_partitions: u16,
    pub(super) physical_column_groups: u16,
    pub(super) grouped: bool,
}

fn grouped_output_layout(
    consumers: &[Operation],
    operation: &Operation,
    output: &TensorShape,
    value_uses: &BTreeMap<ValueId, usize>,
) -> Option<GroupedOutputLayout> {
    let result = *operation.results.first()?;
    if value_uses.get(&result).copied() != Some(1) {
        return None;
    }
    let consumer = consumers
        .iter()
        .find(|candidate| candidate.inputs.contains(&result))?;
    let OperationKind::View(view) = consumer.kind else {
        return None;
    };
    if view.reversed || view.split_axis != output.0.len().checked_sub(1)? || view.merge_axis != 0 {
        return None;
    }
    let groups = u16::try_from(view.factor).ok()?;
    let width = *output.0.last()?;
    (groups != 0 && width.is_multiple_of(u32::from(groups))).then_some(GroupedOutputLayout {
        groups,
        physical_lane_multiple: AMP_COLUMN_MICRO,
    })
}

fn direct_consumer_layouts(
    consumers: &[Operation],
    result: ValueId,
    output: &TensorShape,
    config: &PipelineConfig,
) -> Vec<Layout> {
    let Ok(streams) = u16::try_from(output.0.first().copied().unwrap_or(0)) else {
        return Vec::new();
    };
    if streams == 0 {
        return Vec::new();
    }
    let Some(&rows) = output.0.get(1) else {
        return Vec::new();
    };
    let query_partitions = u16::try_from(rows)
        .unwrap_or(u16::MAX)
        .min(config.tile_count / streams);
    let key_partitions = u16::try_from(rows.div_ceil(AMP_INNER_BLOCK))
        .unwrap_or(u16::MAX)
        .min(config.tile_count / streams);
    let mut layouts = Vec::new();
    for consumer in consumers {
        for input_index in consumer
            .inputs
            .iter()
            .enumerate()
            .filter_map(|(index, &input)| (input == result).then_some(index))
        {
            let layout = match (&consumer.kind, input_index) {
                (OperationKind::FlashAttention(_), 0) if query_partitions != 0 => {
                    Some(Layout::attention_query(streams, query_partitions))
                }
                (OperationKind::FlashAttention(_), 1) if key_partitions != 0 => {
                    Some(Layout::attention_key(streams, key_partitions))
                }
                (OperationKind::FlashAttention(_), 2) if key_partitions != 0 => Some(
                    Layout::attention_block_major_key_value(streams, key_partitions),
                ),
                _ => None,
            };
            if let Some(layout) = layout
                && !layouts.contains(&layout)
            {
                layouts.push(layout);
            }
        }
    }
    layouts
}

pub(super) fn plans(
    operation: &Operation,
    inputs: &[TensorType],
    parameter_inputs: &[bool],
    output: &TensorShape,
    config: &PipelineConfig,
    costs: &impl CostModel,
    distributed_result_is_useful: bool,
    grouped_output: Option<GroupedOutputLayout>,
    direct_consumer_layouts: &[Layout],
    output_demands: &[OutputDemand],
) -> Vec<OperatorPlan> {
    let mut plans = Vec::new();
    let gemm_constraint = config
        .gemm_plan_constraints
        .iter()
        .find(|constraint| constraint.source_operation == operation.id.index());
    if let OperationKind::View(view) = operation.kind
        && !view.reversed
        && view.split_axis == 2
        && view.merge_axis == 0
        && let [input] = inputs
        // These direct attention layouts use F16/F32 panel geometry. FP8
        // views retain the general row-major path until an FP8 consumer exists.
        && matches!(input.format.precision, Precision::F16 | Precision::F32)
        && output.0.len() == 3
        && let (Ok(streams), Ok(rows)) = (u16::try_from(output.0[0]), u16::try_from(output.0[1]))
        && streams != 0
    {
        let query_partitions = rows.min(config.tile_count / streams);
        let key_partitions = u16::try_from(output.0[1].div_ceil(AMP_INNER_BLOCK))
            .unwrap_or(u16::MAX)
            .min(config.tile_count / streams);
        let layouts = if direct_consumer_layouts.is_empty() {
            [
                (query_partitions != 0).then(|| Layout::attention_query(streams, query_partitions)),
                (key_partitions != 0).then(|| Layout::attention_key(streams, key_partitions)),
                (key_partitions != 0)
                    .then(|| Layout::attention_block_major_key_value(streams, key_partitions)),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
        } else {
            direct_consumer_layouts.to_vec()
        };
        for layout in layouts {
            let plan = OperatorPlan {
                operator: MidOperator::View(view),
                dispatch: OperatorDispatch::View,
                requirements: StorageRequirements {
                    inputs: vec![OperandRequirement::new(input.format.clone(), 8)],
                    output: OperandRequirement::new(
                        TensorFormat {
                            precision: input.format.precision,
                            layout,
                        },
                        8,
                    ),
                    output_aliasing: OutputAliasing::Fresh,
                    distinct_elements: Vec::new(),
                },
                deferred_output: Some(DeferredOutputPlan {
                    source_input: 0,
                    transform: view,
                    unfused_cycles: 0,
                    unfused_exchange_cycles: 0,
                }),
            };
            if !plans.contains(&plan) {
                plans.push(plan);
            }
        }
    }
    if let OperationKind::View(view) = operation.kind
        && plans.is_empty()
        && let [input] = inputs
    {
        let row_major = |shape: &TensorShape| {
            // Keep complete rows, distributing the matrix row axis rather
            // than concentrating an entire batch on one tile.
            let axis = shape.0.len().saturating_sub(2);
            Layout::row_major(TensorTiling::sharded(
                TensorAxis::FromEnd(if axis == 0 && shape.0.len() == 1 {
                    1
                } else {
                    2
                }),
                u16::try_from(shape.0[axis])
                    .unwrap_or(u16::MAX)
                    .min(config.tile_count)
                    .max(1),
            ))
        };
        let source = if input.format.layout.tiling.linear_grain().is_some() {
            TensorFormat {
                precision: input.format.precision,
                layout: row_major(&input.shape),
            }
        } else {
            input.format.clone()
        };
        plans.push(OperatorPlan {
            operator: MidOperator::View(view),
            dispatch: OperatorDispatch::View,
            requirements: StorageRequirements {
                inputs: vec![OperandRequirement::new(source, 8)],
                output: OperandRequirement::new(
                    TensorFormat {
                        precision: input.format.precision,
                        layout: row_major(output),
                    },
                    8,
                ),
                output_aliasing: OutputAliasing::Fresh,
                distinct_elements: Vec::new(),
            },
            deferred_output: None,
        });
    }
    if let OperationKind::Slice(slice) = operation.kind
        && let [input] = inputs
    {
        let mut layouts = vec![input.format.layout.clone()];
        layouts.push(Layout::row_sharded(
            u16::try_from(output.0[output.0.len().saturating_sub(2)])
                .unwrap_or(u16::MAX)
                .min(config.tile_count)
                .max(1),
        ));
        for layout in layouts {
            plans.push(OperatorPlan {
                operator: MidOperator::Slice(slice),
                dispatch: OperatorDispatch::View,
                requirements: StorageRequirements {
                    inputs: vec![OperandRequirement::new(input.format.clone(), 8)],
                    output: OperandRequirement::new(
                        TensorFormat {
                            precision: input.format.precision,
                            layout,
                        },
                        8,
                    ),
                    output_aliasing: OutputAliasing::Fresh,
                    distinct_elements: Vec::new(),
                },
                deferred_output: None,
            });
        }
    }
    if let OperationKind::FlashAttention(options) = operation.kind
        && !options.causal
        && let [query, key, value] = inputs
        && query.shape.0.len() == 3
        && key.shape.0.len() == 3
        && value.shape.0.len() == 3
        && query.shape.0[0] == key.shape.0[0]
        && query.shape.0[0] == value.shape.0[0]
        && let Ok(heads) = u16::try_from(query.shape.0[0])
        && heads != 0
    {
        let query_rows = query.shape.0[1];
        let query_partitions = u16::try_from(query_rows)
            .unwrap_or(u16::MAX)
            .min(config.tile_count / heads);
        if query_partitions != 0 {
            let key_partitions =
                u16::try_from(key.shape.0[1].div_ceil(AMP_INNER_BLOCK)).unwrap_or(u16::MAX);
            if key_partitions == 0 || heads.saturating_mul(key_partitions) > config.tile_count {
                return plans;
            }
            let padded_query_dimension =
                query.shape.0[2].div_ceil(AMP_COLUMN_MICRO) * AMP_COLUMN_MICRO;
            let padded_value_dimension =
                value.shape.0[2].div_ceil(AMP_COLUMN_MICRO) * AMP_COLUMN_MICRO;
            let padded_key_rows = key.shape.0[1].div_ceil(AMP_INNER_BLOCK) * AMP_INNER_BLOCK;
            let query_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::attention_query(heads, query_partitions),
            };
            let key_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::attention_key(heads, key_partitions),
            };
            let value_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::attention_block_major_key_value(heads, key_partitions),
            };
            let output_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::attention_output(heads, query_partitions),
            };
            for (materialized, key_block_rows) in
                [(false, AMP_INNER_BLOCK), (true, padded_key_rows)]
            {
                let plan = OperatorPlan {
                    operator: MidOperator::FlashAttention {
                        options,
                        accumulate: AccumulationPrecision::F32,
                    },
                    dispatch: OperatorDispatch::Attention {
                        query_key: None,
                        probability_value: None,
                        materialized,
                        key_block_rows,
                        padded_query_dimension,
                        padded_value_dimension,
                    },
                    requirements: StorageRequirements {
                        inputs: [&query_format, &key_format, &value_format]
                            .into_iter()
                            .map(|format| {
                                OperandRequirement::new(format.clone(), 8)
                                    .with_materialization(OperandMaterialization::DispatchSlices)
                            })
                            .collect(),
                        output: OperandRequirement::new(output_format.clone(), 8),
                        output_aliasing: OutputAliasing::Fresh,
                        distinct_elements: Vec::new(),
                    },
                    deferred_output: None,
                };
                plans.extend(attention::product_variants(plan, inputs, config));
            }
        }
        match config.attention_strategy {
            AttentionStrategy::Automatic => {}
            AttentionStrategy::Flash => plans.retain(|plan| {
                !matches!(
                    plan.dispatch,
                    OperatorDispatch::Attention {
                        materialized: true,
                        ..
                    }
                )
            }),
            AttentionStrategy::Materialized => plans.retain(|plan| {
                !matches!(
                    plan.dispatch,
                    OperatorDispatch::Attention {
                        materialized: false,
                        ..
                    }
                )
            }),
        }
    }
    if let [input] = inputs
        && input.shape == *output
        && config.conversion_streaming != ConversionStreamingPolicy::Always
    {
        let mut flat_candidates = BTreeMap::new();
        for candidate in config
            .operator_candidates
            .iter()
            .filter_map(OperatorCandidate::concrete)
            .filter(|candidate| {
                operator_matches(&operation.kind, candidate.plan.operator)
                    && candidate.plan.requirements.inputs.len() == 1
                    && matches!(
                        candidate.format_policy,
                        OperatorFormatPolicy::PreserveInputLayout(0)
                    )
                    && matches!(candidate.plan.dispatch, OperatorDispatch::Pointwise { .. })
                    && candidate.plan.requirements.inputs[0].format.precision
                        == input.format.precision
            })
        {
            let grain = candidate.plan.requirements.inputs[0]
                .alignment
                .div_ceil(input.format.precision.bytes() as u32);
            if grain == 0 || !output.elements().is_multiple_of(u64::from(grain)) {
                continue;
            }
            let tiles = candidate
                .plan
                .requirements
                .output
                .format
                .layout
                .tiling
                .tile_count;
            let Some(&width) = output.0.last() else {
                continue;
            };
            let width = u64::from(width);
            let grains = output.elements() / u64::from(grain);
            let splits = (1..tiles)
                .filter(|&tile| {
                    let tile = u64::from(tile);
                    let offset = (tile * (grains / u64::from(tiles))
                        + tile.min(grains % u64::from(tiles)))
                        * u64::from(grain);
                    !offset.is_multiple_of(width)
                })
                .count();
            flat_candidates
                .entry(tiles)
                .or_insert((splits, grain, &candidate.plan));
        }
        // Retain the occupancy/fragmentation Pareto frontier. This keeps the
        // option available without multiplying equivalent pointwise plans.
        let flat_candidates = flat_candidates
            .iter()
            .filter(|(tiles, (splits, _, _))| {
                !flat_candidates
                    .iter()
                    .any(|(other_tiles, (other_splits, _, _))| {
                        other_tiles >= tiles
                            && other_splits <= splits
                            && (other_tiles > tiles || other_splits < splits)
                    })
            })
            .map(|(_, (_, grain, candidate))| (*grain, *candidate))
            .collect::<Vec<_>>();
        for (grain, candidate) in flat_candidates {
            let tiles = candidate
                .requirements
                .output
                .format
                .layout
                .tiling
                .tile_count;
            let mut layouts = vec![Layout::logical_linear(tiles, grain)];
            if let Some(retained_grain) = input
                .format
                .layout
                .order
                .retained_linear_column_grain(input.format.precision)
                .filter(|retained_grain| retained_grain.is_multiple_of(grain))
                .filter(|retained_grain| {
                    output.elements().is_multiple_of(u64::from(*retained_grain))
                        && output
                            .0
                            .last()
                            .is_some_and(|width| width.is_multiple_of(*retained_grain))
                })
            {
                layouts.push(
                    input
                        .format
                        .layout
                        .with_retained_order_linear_ownership(tiles, retained_grain),
                );
            }
            for layout in layouts {
                let format = TensorFormat {
                    precision: input.format.precision,
                    layout,
                };
                let plan = OperatorPlan {
                    operator: candidate.operator,
                    dispatch: candidate.dispatch.clone(),
                    requirements: StorageRequirements {
                        inputs: vec![OperandRequirement::new(format.clone(), 8)],
                        output: OperandRequirement::new(format, 8),
                        output_aliasing: OutputAliasing::MayAliasInputs(vec![0]),
                        distinct_elements: Vec::new(),
                    },
                    deferred_output: None,
                };
                if !plans.contains(&plan) {
                    plans.push(plan);
                }
            }
        }
    }
    for candidate in config
        .operator_candidates
        .iter()
        .filter(|candidate| operator_matches(&operation.kind, candidate.operator()))
    {
        let mut variants = match candidate {
            OperatorCandidate::ParallelGemm { tile_count, .. } => parallel_reduction_candidates(
                candidate.operator(),
                *tile_count,
                inputs,
                output,
                config,
                costs,
                distributed_result_is_useful,
                gemm_constraint,
                grouped_output,
                output_demands,
            ),
            OperatorCandidate::Concrete(concrete) => {
                let ConcreteOperatorCandidate {
                    plan: mut candidate,
                    format_policy,
                } = concrete.clone();
                if let OperatorFormatPolicy::PreserveInputLayout(index) = format_policy {
                    let Some((actual, requirement)) = inputs
                        .get(usize::from(index))
                        .zip(candidate.requirements.inputs.get_mut(usize::from(index)))
                    else {
                        continue;
                    };
                    if actual.format.precision != requirement.format.precision
                        || candidate.requirements.output.format.precision
                            != requirement.format.precision
                    {
                        continue;
                    }
                    requirement.format.layout = actual.format.layout.clone();
                    candidate.requirements.output.format.layout = actual.format.layout.clone();
                }
                vec![candidate]
            }
        };
        if config.gemm_output_packing != GemmOutputPacking::Native
            && (grouped_output.is_some() || config.gemm_output_packing == GemmOutputPacking::Packed)
        {
            let additions = variants
                .iter()
                .filter(|plan| {
                    matches!(
                        plan.dispatch,
                        OperatorDispatch::BlockedGemm {
                            distribution: GemmDistribution::OutputStationary,
                            ..
                        }
                    )
                })
                .filter_map(|plan| packed_gemm_output(plan, output))
                .collect::<Vec<_>>();
            variants.extend(additions);
        }
        for (input_index, _) in parameter_inputs
            .iter()
            .enumerate()
            .filter(|(_, parameter)| **parameter)
        {
            let additions = variants
                .iter()
                .flat_map(|variant| {
                    independent_parameter_storage(variant, inputs, input_index, config)
                })
                .filter(|independent| !variants.contains(independent))
                .collect::<Vec<_>>();
            variants.extend(additions);
        }
        for candidate in variants {
            if !candidate.supports(inputs, output) {
                continue;
            }
            let aliasing = resolved_output_aliasing(&candidate, inputs, output);
            let mut plan = candidate;
            plan.requirements.output_aliasing = aliasing;
            if !plans.contains(&plan) {
                plans.push(plan);
            }
        }
    }
    plans.retain(|plan| {
        (config.gemm_output_packing != GemmOutputPacking::Packed
            || !matches!(plan.operator, MidOperator::Gemm { .. })
            || plan
                .requirements
                .output
                .format
                .layout
                .order
                .gemm_output_group()
                .is_some())
            && plan_fits_operator_memory(plan, inputs, output, config)
    });
    if let Some(constraint) = gemm_constraint {
        plans.retain(|plan| {
            gemm_plan_matches(constraint, &plan.dispatch, &plan.requirements.inputs)
        });
        tracing::info!(
            source_operation = constraint.source_operation,
            matching_plans = plans.len(),
            ?constraint,
            "applied GEMM plan constraint"
        );
    }
    plans
}

pub(super) fn gemm_plan_matches(
    constraint: &GemmPlanConstraint,
    dispatch: &OperatorDispatch,
    inputs: &[OperandRequirement],
) -> bool {
    let OperatorDispatch::BlockedGemm {
        output_column_block,
        orientation,
        distribution:
            GemmDistribution::ParallelReduction {
                row_partitions,
                column_partitions,
                inner_partitions,
                result_row_partitions,
                result_column_partitions,
                reduction_staging,
            },
        ..
    } = dispatch
    else {
        return false;
    };
    let weight_index = orientation.operand_indices().1;
    let Some(weight) = inputs.get(weight_index) else {
        return false;
    };
    *orientation == constraint.orientation
        && *row_partitions == constraint.row_partitions
        && *column_partitions == constraint.column_partitions
        && *inner_partitions == constraint.inner_partitions
        && *result_row_partitions == constraint.result_row_partitions
        && *result_column_partitions == constraint.result_column_partitions
        && *output_column_block == constraint.output_column_block
        && weight.format.layout.memory_class == constraint.weight_memory_class
        && *reduction_staging == constraint.reduction_staging
        && weight.local_staging == constraint.local_weight_staging
}

pub(super) fn independent_parameter_storage(
    candidate: &OperatorPlan,
    inputs: &[TensorType],
    input_index: usize,
    config: &PipelineConfig,
) -> Vec<OperatorPlan> {
    if !matches!(candidate.dispatch, OperatorDispatch::BlockedGemm { .. }) {
        return Vec::new();
    }
    let Some(requirement) = candidate.requirements.inputs.get(input_index) else {
        return Vec::new();
    };
    let ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
        row_block: inner_block,
        column_block: _,
    }) = requirement.format.layout.order
    else {
        return Vec::new();
    };
    let Some(input) = inputs.get(input_index) else {
        return Vec::new();
    };
    let rank = input.shape.0.len();
    let Some(inner_axis) = rank.checked_sub(2) else {
        return Vec::new();
    };
    let Some(&inner) = input.shape.0.get(inner_axis) else {
        return Vec::new();
    };
    let Some(&columns) = input.shape.0.last() else {
        return Vec::new();
    };
    let inner_blocks = inner.div_ceil(u32::from(inner_block));
    let output_column_block = match candidate.dispatch {
        OperatorDispatch::BlockedGemm {
            output_column_block,
            ..
        } => output_column_block,
        OperatorDispatch::Pointwise { .. }
        | OperatorDispatch::Attention { .. }
        | OperatorDispatch::View => {
            return Vec::new();
        }
    };
    if output_column_block < AMP_OUTPUT_COLUMN_BLOCK {
        return Vec::new();
    }
    let column_blocks = columns.div_ceil(output_column_block);
    let mut storage_grids = (1..=inner_blocks.min(u32::from(config.tile_count)))
        .flat_map(|inner_partitions| {
            let maximum_columns = (u32::from(config.tile_count) / inner_partitions)
                .min(column_blocks)
                .min(u32::from(u16::MAX));
            (1..=maximum_columns).map(move |column_partitions| {
                let panels_per_shard = inner_blocks
                    .div_ceil(inner_partitions)
                    .saturating_mul(column_blocks.div_ceil(column_partitions));
                let used = inner_partitions.saturating_mul(column_partitions);
                (
                    panels_per_shard,
                    u32::MAX - used,
                    column_partitions,
                    inner_partitions,
                )
            })
        })
        .collect::<Vec<_>>();
    storage_grids.sort_unstable();
    storage_grids
        .first()
        .and_then(|&(_, _, column_partitions, inner_partitions)| {
            Some((
                u16::try_from(column_partitions).ok()?,
                u16::try_from(inner_partitions).ok()?,
            ))
        })
        .into_iter()
        .map(|(column_partitions, inner_partitions)| {
            let mut independent = candidate.clone();
            independent.requirements.inputs[input_index].format.layout =
                Layout::block_major_matrix_storage(
                    inner_block,
                    output_column_block,
                    column_partitions,
                    inner_partitions,
                    1,
                    requirement.format.layout.memory_class,
                );
            independent
        })
        .collect()
}

/// A packed alternative keeps the compute grid but gives both producers and
/// reductions the same panel order. Native layouts remain separate candidates.
fn packed_gemm_output(plan: &OperatorPlan, output: &TensorShape) -> Option<OperatorPlan> {
    let MidOperator::Gemm {
        multiply: Precision::F16,
        ..
    } = plan.operator
    else {
        return None;
    };
    let OperatorDispatch::BlockedGemm {
        orientation,
        distribution,
        ..
    } = plan.dispatch
    else {
        return None;
    };
    if matches!(distribution, GemmDistribution::ParallelReduction { result_row_partitions, result_column_partitions, .. }
        if result_row_partitions != 1 || result_column_partitions != 1)
    {
        return None;
    }
    let mut packed = plan.clone();
    let (left, right) = orientation.operand_indices();
    let (row, column) = orientation.matrix_axes(output.0.len());
    let pad = |layout: &mut Layout, axis: usize, multiple| -> Option<()> {
        let tiling = layout
            .tiling
            .axes
            .iter_mut()
            .find(|a| a.axis.resolve(output.0.len()) == Ok(axis))?;
        tiling.shard_padding_multiple = multiple;
        tiling.padding = Padding::Zero;
        Some(())
    };
    pad(&mut packed.requirements.inputs[left].format.layout, row, 16)?;
    pad(
        &mut packed.requirements.inputs[right].format.layout,
        column,
        64,
    )?;
    let layout = &mut packed.requirements.output.format.layout;
    pad(layout, row, 16)?;
    pad(layout, column, 64)?;
    layout.order = ElementOrder::BlockMajor(match orientation {
        GemmOrientation::Normal => BlockMajorOrder::TransposedMatrix {
            row_block: 64,
            column_block: 16,
        },
        GemmOrientation::Swapped => BlockMajorOrder::Matrix {
            row_block: 64,
            column_block: 16,
        },
    });
    Some(packed)
}

pub(super) fn parallel_reduction_candidates(
    operator: MidOperator,
    tile_count: u16,
    inputs: &[TensorType],
    output: &TensorShape,
    config: &PipelineConfig,
    costs: &impl CostModel,
    distributed_result_is_useful: bool,
    constraint: Option<&GemmPlanConstraint>,
    grouped_output: Option<GroupedOutputLayout>,
    output_demands: &[OutputDemand],
) -> Vec<OperatorPlan> {
    [GemmOrientation::Normal, GemmOrientation::Swapped]
        .into_iter()
        .flat_map(|orientation| {
            parallel_reduction_candidates_for_orientation(
                operator,
                tile_count,
                inputs,
                output,
                config,
                costs,
                orientation,
                distributed_result_is_useful,
                constraint,
                grouped_output,
                output_demands,
            )
        })
        .collect()
}

pub(super) fn parallel_reduction_candidates_for_orientation(
    operator: MidOperator,
    tile_count: u16,
    inputs: &[TensorType],
    output: &TensorShape,
    config: &PipelineConfig,
    costs: &impl CostModel,
    orientation: GemmOrientation,
    distributed_result_is_useful: bool,
    constraint: Option<&GemmPlanConstraint>,
    grouped_output: Option<GroupedOutputLayout>,
    output_demands: &[OutputDemand],
) -> Vec<OperatorPlan> {
    let MidOperator::Gemm { multiply, .. } = operator else {
        return Vec::new();
    };
    let inner_micro = if matches!(multiply, Precision::F8F143 { .. }) {
        32
    } else {
        16
    };
    let [left, right] = inputs else {
        return Vec::new();
    };
    let rank = left.shape.0.len();
    if rank < 2 || right.shape.0.len() < 2 {
        return Vec::new();
    }
    let Some(&inner) = left.shape.0.last() else {
        return Vec::new();
    };
    let Some(&normal_columns) = right.shape.0.last() else {
        return Vec::new();
    };
    let normal_rows = left.shape.0[left.shape.0.len() - 2];
    let (rows, columns) = match orientation {
        GemmOrientation::Normal => (normal_rows, normal_columns),
        GemmOrientation::Swapped => (normal_columns, normal_rows),
    };
    // Each grid determines its own padded K/C extents and operand layouts.
    let column_groups = columns.div_ceil(AMP_COLUMN_MICRO);
    let inner_groups = inner.div_ceil(inner_micro);
    let Ok(inner_groups) = u16::try_from(inner_groups) else {
        return Vec::new();
    };
    let Ok(column_groups) = u16::try_from(column_groups) else {
        return Vec::new();
    };
    let grouped_column_groups = grouped_output.and_then(|grouping| {
        let groups = u32::from(grouping.groups);
        (orientation == GemmOrientation::Normal && groups != 0 && columns.is_multiple_of(groups))
            .then(|| {
                let columns_per_group = columns / groups;
                columns_per_group
                    .div_ceil(grouping.physical_lane_multiple)
                    .saturating_mul(groups)
            })
    });
    let grouped_column_groups = grouped_column_groups
        .and_then(|groups| u16::try_from(groups).ok())
        .filter(|groups| *groups >= column_groups);
    let outer_rows = left.shape.0[..left.shape.0.len() - 2]
        .iter()
        .copied()
        .fold(1u64, |product, extent| {
            product.saturating_mul(u64::from(extent))
        });
    let mut grids = Vec::new();
    for inner_partitions in 2..=inner_groups.min(tile_count) {
        let maximum_columns = grouped_column_groups
            .unwrap_or(column_groups)
            .min(tile_count / inner_partitions);
        for column_partitions in 1..=maximum_columns {
            let grouped_options = [
                (column_partitions <= column_groups).then_some((false, column_groups)),
                grouped_output.and_then(|grouping| match orientation {
                    GemmOrientation::Normal => column_partitions
                        .is_multiple_of(grouping.groups)
                        .then_some((true, grouped_column_groups?)),
                    GemmOrientation::Swapped => normal_columns
                        .is_multiple_of(u32::from(grouping.groups))
                        .then_some((true, column_groups)),
                }),
            ];
            for (grouped, physical_column_groups) in grouped_options.into_iter().flatten() {
                let mut row_partitions = (tile_count / inner_partitions / column_partitions)
                    .min(u16::try_from(rows).unwrap_or(u16::MAX));
                if grouped && orientation == GemmOrientation::Swapped {
                    let groups = grouped_output.unwrap().groups;
                    row_partitions = row_partitions / groups * groups;
                }
                if row_partitions == 0 {
                    continue;
                }
                let used_tiles = row_partitions
                    .saturating_mul(column_partitions)
                    .saturating_mul(inner_partitions);
                if used_tiles < tile_count.div_ceil(2) || u32::from(row_partitions) > rows {
                    continue;
                }
                let local_rows = if grouped && orientation == GemmOrientation::Swapped {
                    let groups = u32::from(grouped_output.unwrap().groups);
                    (rows / groups)
                        .div_ceil(AMP_COLUMN_MICRO)
                        .div_ceil(u32::from(row_partitions) / groups)
                        * AMP_COLUMN_MICRO
                } else {
                    rows.div_ceil(u32::from(row_partitions))
                };
                let local_columns =
                    u32::from(physical_column_groups).div_ceil(u32::from(column_partitions));
                let local_inner = u32::from(inner_groups).div_ceil(u32::from(inner_partitions));
                if u32::from(inner_partitions - 1).saturating_mul(local_inner)
                    >= u32::from(inner_groups)
                {
                    continue;
                }
                // Retain grids by the generated kernel's actual K16 x C16
                // invocation structure, including its fixed weight-feed and
                // worker/supervisor cost. Pure arithmetic work is almost
                // constant across grids and incorrectly favors tiny row runs.
                let row_run_cycles = crate::kernel::cost::f16_gemm_microgroup_cycles(
                    outer_rows.saturating_mul(u64::from(local_rows)),
                );
                let compute = u64::from(local_columns)
                    .saturating_mul(u64::from(local_inner))
                    .saturating_mul(row_run_cycles);
                let communication = u64::from(local_columns)
                    .saturating_mul(u64::from(local_inner))
                    .saturating_add(u64::from(local_rows).saturating_mul(u64::from(local_inner)))
                    .saturating_add(
                        u64::from(local_rows)
                            .saturating_mul(u64::from(local_columns))
                            .saturating_mul(u64::from(inner_partitions - 1)),
                    );
                let left_bytes = outer_rows
                    .saturating_mul(u64::from(local_rows))
                    .saturating_mul(u64::from(local_inner))
                    .saturating_mul(u64::from(inner_micro))
                    .saturating_mul(multiply.bytes());
                let right_bytes = u64::from(local_columns)
                    .saturating_mul(u64::from(AMP_COLUMN_MICRO))
                    .saturating_mul(u64::from(local_inner))
                    .saturating_mul(u64::from(inner_micro))
                    .saturating_mul(multiply.bytes());
                let partial_bytes = outer_rows
                    .saturating_mul(u64::from(local_rows))
                    .saturating_mul(u64::from(local_columns))
                    .saturating_mul(u64::from(AMP_COLUMN_MICRO))
                    .saturating_mul(Precision::F16.bytes());
                // Operand staging and the local partial coexist during
                // convolution. Complete staging is evaluated later by the
                // ordinary operator-memory model.
                let convolution_bytes = left_bytes
                    .saturating_add(right_bytes)
                    .saturating_add(partial_bytes);
                let reduction_bytes = partial_bytes.saturating_mul(4);
                let temporary_bytes = convolution_bytes.max(reduction_bytes);
                if temporary_bytes > u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES)
                    || right_bytes.saturating_add(partial_bytes)
                        > u64::from(crate::memory::IPU21_INTERLEAVED_REGION_BYTES)
                {
                    continue;
                }
                grids.push(ParallelGridProxy {
                    compute,
                    communication,
                    temporary_bytes,
                    unused_tiles: u64::from(tile_count - used_tiles),
                    row_partitions,
                    column_partitions,
                    inner_partitions,
                    physical_column_groups,
                    grouped,
                });
            }
        }
    }
    let generated_grids = grids.len();
    let grids = if let Some(constraint) = constraint {
        grids
            .into_iter()
            .filter(|grid| {
                orientation == constraint.orientation
                    && grid.row_partitions == constraint.row_partitions
                    && grid.column_partitions == constraint.column_partitions
                    && grid.inner_partitions == constraint.inner_partitions
            })
            .collect::<Vec<_>>()
    } else {
        // Peak temporary storage and unused tiles are independent planning
        // constraints, so retain every non-dominated proxy tradeoff until the
        // precise operator estimator can rank it.
        let dominates = |left: &ParallelGridProxy, right: &ParallelGridProxy| {
            left.inner_partitions == right.inner_partitions
                && left.grouped == right.grouped
                && left.compute <= right.compute
                && left.communication <= right.communication
                && left.temporary_bytes <= right.temporary_bytes
                && left.unused_tiles <= right.unused_tiles
                && (left.compute < right.compute
                    || left.communication < right.communication
                    || left.temporary_bytes < right.temporary_bytes
                    || left.unused_tiles < right.unused_tiles)
        };
        let mut frontier = Vec::new();
        for grid in grids {
            if frontier.iter().any(|kept| dominates(kept, &grid)) {
                continue;
            }
            frontier.retain(|kept| !dominates(&grid, kept));
            frontier.push(grid);
        }
        frontier.sort_by_key(|grid| {
            (
                grid.compute,
                grid.communication,
                grid.temporary_bytes,
                grid.unused_tiles,
                grid.row_partitions,
                grid.column_partitions,
                grid.inner_partitions,
                grid.grouped,
            )
        });
        frontier
    };
    let proxy_frontier_grids = grids.len();
    let mut variants = Vec::new();
    for grid in grids {
        let ParallelGridProxy {
            row_partitions,
            column_partitions,
            inner_partitions,
            physical_column_groups,
            grouped,
            ..
        } = grid;
        let used_tiles = row_partitions
            .saturating_mul(column_partitions)
            .saturating_mul(inner_partitions);
        let kernel_inner_block = u32::from(inner_groups)
            .div_ceil(u32::from(inner_partitions))
            .saturating_mul(inner_micro);
        let kernel_output_columns = u32::from(physical_column_groups)
            .div_ceil(u32::from(column_partitions))
            .saturating_mul(AMP_COLUMN_MICRO);
        let Ok(kernel_inner_block_u16) = u16::try_from(kernel_inner_block) else {
            continue;
        };
        for memory_class in [MemoryClass::Ipu21Standard, MemoryClass::Ipu21Interleaved] {
            let input_layouts = match orientation {
                GemmOrientation::Normal => [
                    Layout::amp_left_parallel_grid(
                        kernel_inner_block_u16,
                        used_tiles,
                        row_partitions,
                        column_partitions,
                        inner_partitions,
                    ),
                    Layout::block_major_matrix_storage(
                        kernel_inner_block_u16,
                        kernel_output_columns,
                        column_partitions,
                        inner_partitions,
                        1,
                        memory_class,
                    ),
                ],
                GemmOrientation::Swapped => [
                    Layout::transposed_block_major_matrix_storage(
                        kernel_inner_block_u16,
                        kernel_output_columns,
                        column_partitions,
                        inner_partitions,
                        row_partitions,
                        memory_class,
                    ),
                    Layout::amp_transposed_left_parallel_grid(
                        kernel_inner_block_u16,
                        used_tiles,
                        row_partitions,
                        column_partitions,
                        inner_partitions,
                    ),
                ],
            };
            let mut input_layouts = input_layouts;
            balance_parallel_gemm_columns(
                &mut input_layouts[orientation.operand_indices().1],
                match orientation {
                    GemmOrientation::Normal => TensorAxis::FromEnd(1),
                    GemmOrientation::Swapped => TensorAxis::FromEnd(2),
                },
            );
            let physical_right_index = orientation.operand_indices().1;
            let local_staging_options: &[_] = match orientation {
                GemmOrientation::Normal => &[LocalOperandStaging::Direct],
                GemmOrientation::Swapped => &[
                    LocalOperandStaging::Direct,
                    LocalOperandStaging::MatchRemote,
                ],
            };
            let maximum_result_rows = u16::try_from(rows / u32::from(row_partitions))
                .unwrap_or(u16::MAX)
                .min(inner_partitions);
            let maximum_result_columns = u16::try_from(
                columns
                    .div_ceil(AMP_COLUMN_MICRO)
                    .checked_div(u32::from(column_partitions))
                    .unwrap_or(0),
            )
            .unwrap_or(u16::MAX)
            .min(inner_partitions);
            let mut result_partition_options = vec![(1, 1)];
            if distributed_result_is_useful {
                for result_rows in 1..=maximum_result_rows {
                    if inner_partitions.is_multiple_of(result_rows) {
                        let result_columns = inner_partitions / result_rows;
                        if result_columns <= maximum_result_columns {
                            result_partition_options.push((result_rows, result_columns));
                        }
                    }
                }
            }
            if let Some(constraint) = constraint {
                let pair = (
                    constraint.result_row_partitions,
                    constraint.result_column_partitions,
                );
                if pair.0 <= maximum_result_rows
                    && pair.1 <= maximum_result_columns
                    && u32::from(pair.0) * u32::from(pair.1) <= u32::from(inner_partitions)
                    && !result_partition_options.contains(&pair)
                {
                    result_partition_options.push(pair);
                }
            }
            for (result_row_partitions, result_column_partitions) in result_partition_options {
                let result_rows = row_partitions.saturating_mul(result_row_partitions);
                let result_columns = column_partitions.saturating_mul(result_column_partitions);
                let result_column_block = if result_column_partitions > 1 {
                    AMP_COLUMN_MICRO
                } else {
                    kernel_output_columns
                };
                for grid_order in [GridOrder::ColumnsFast, GridOrder::RowsFast] {
                    if grid_order == GridOrder::RowsFast
                        && ((result_row_partitions, result_column_partitions) != (1, 1)
                            || result_rows == 1
                            || result_columns == 1)
                    {
                        continue;
                    }
                    let mut result_layout = match orientation {
                        GemmOrientation::Normal => Layout::amp_left_result_grid(
                            result_column_block,
                            result_rows.saturating_mul(result_columns),
                            result_rows,
                            result_columns,
                            grid_order,
                        ),
                        GemmOrientation::Swapped => Layout::amp_transposed_left_result_grid(
                            result_column_block,
                            result_rows.saturating_mul(result_columns),
                            result_rows,
                            result_columns,
                            grid_order,
                        ),
                    };
                    let physical_column_axis = match orientation {
                        GemmOrientation::Normal => TensorAxis::FromEnd(1),
                        GemmOrientation::Swapped => TensorAxis::FromEnd(2),
                    };
                    balance_parallel_gemm_columns(&mut result_layout, physical_column_axis);

                    let (physical_row_axis, physical_rows, physical_left_index) = match orientation
                    {
                        GemmOrientation::Normal => (TensorAxis::FromEnd(2), normal_rows, 0),
                        GemmOrientation::Swapped => (TensorAxis::FromEnd(1), normal_columns, 1),
                    };
                    let result_rows = result_layout
                        .tiling
                        .axes
                        .iter()
                        .find(|axis| axis.axis == physical_row_axis)
                        .map_or(row_partitions, |axis| axis.partitions);
                    // Exchange moves whole 32-bit words. Give every F16 shard an
                    // even physical-row grain rather than allowing a later layout
                    // conversion to discover an unsendable two-byte tail.
                    if u32::from(result_rows) > physical_rows.div_ceil(2) {
                        continue;
                    }
                    pad_axis_to_f16_exchange_word(
                        &mut input_layouts[physical_left_index],
                        physical_row_axis,
                    );
                    pad_axis_to_f16_exchange_word(&mut result_layout, physical_row_axis);
                    if grouped
                        && !grouped_output.is_some_and(|grouping| {
                            apply_grouped_output_layout(
                                &mut input_layouts[1],
                                &mut result_layout,
                                grouping,
                            )
                        })
                    {
                        continue;
                    }
                    let staging_options = constraint.map_or_else(
                        || vec![ReductionStaging::Complete, ReductionStaging::Streamed],
                        |constraint| vec![constraint.reduction_staging],
                    );
                    for &local_staging in local_staging_options {
                        for &reduction_staging in &staging_options {
                            let dispatch = OperatorDispatch::BlockedGemm {
                                inner_block: kernel_inner_block,
                                output_column_block: kernel_output_columns,
                                orientation,
                                distribution: GemmDistribution::ParallelReduction {
                                    row_partitions,
                                    column_partitions,
                                    inner_partitions,
                                    result_row_partitions,
                                    result_column_partitions,
                                    reduction_staging,
                                },
                            };
                            let mut staged = gemm_plan(
                                operator,
                                [
                                    input_layouts[0].clone(),
                                    input_layouts[1].clone(),
                                    result_layout.clone(),
                                ],
                                16,
                                dispatch,
                            );
                            staged.requirements.inputs[physical_right_index].local_staging =
                                local_staging;
                            if config.gemm_output_packing != GemmOutputPacking::Native
                                && (grouped_output.is_some()
                                    || config.gemm_output_packing == GemmOutputPacking::Packed)
                                && let Some(packed) = packed_gemm_output(&staged, output)
                                && (config.gemm_output_packing == GemmOutputPacking::Packed
                                    || output_demands.iter().any(|demand| {
                                        demand.matches(
                                            &packed.requirements.output.format.layout,
                                            output,
                                        )
                                    }))
                            {
                                variants.push(packed);
                            }
                            if config.gemm_output_packing != GemmOutputPacking::Packed {
                                variants.push(staged);
                            }
                        }
                    }
                }
            }
        }
    }
    let generated_variants = variants.len();
    let generated_grouped_variants = variants
        .iter()
        .filter(|candidate| {
            candidate
                .requirements
                .output
                .format
                .layout
                .tiling
                .axes
                .iter()
                .any(|axis| axis.padding_groups > 1)
        })
        .count();
    let retained = if let Some(constraint) = constraint {
        variants
            .into_iter()
            .filter(|candidate| {
                gemm_plan_matches(
                    constraint,
                    &candidate.dispatch,
                    &candidate.requirements.inputs,
                )
            })
            .collect::<Vec<_>>()
    } else {
        retain_operator_candidates_for_demands(
            variants,
            inputs,
            output,
            costs,
            config.planning_beam_width.max(1),
            output_demands,
        )
    };
    tracing::debug!(
        ?orientation,
        generated_grids,
        proxy_frontier_grids,
        generated_variants,
        generated_grouped_variants,
        retained_grouped_variants = retained
            .iter()
            .filter(|candidate| candidate
                .requirements
                .output
                .format
                .layout
                .tiling
                .axes
                .iter()
                .any(|axis| axis.padding_groups > 1))
            .count(),
        retained_variants = retained.len(),
        "retained parallel GEMM candidates"
    );
    retained
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct OperatorCompatibility {
    pub(super) orientation: Option<GemmOrientation>,
    pub(super) reduction_staging: Option<ReductionStaging>,
    // Preserve different reduction fan-ins through local screening: their
    // result ownership and cost to the next consumer differ substantially.
    pub(super) inner_partitions: Option<u16>,
    pub(super) result_partitions: Option<(u16, u16)>,
    pub(super) inputs: Vec<(Precision, ElementOrder, MemoryClass, LocalOperandStaging)>,
    pub(super) output: (
        Precision,
        ElementOrder,
        MemoryClass,
        Vec<(TensorAxis, u16, u32)>,
    ),
}

pub(super) fn operator_candidate_compatibility(candidate: &OperatorPlan) -> OperatorCompatibility {
    let (orientation, reduction_staging, inner_partitions, result_partitions) =
        match candidate.dispatch {
            OperatorDispatch::BlockedGemm {
                orientation,
                distribution:
                    GemmDistribution::ParallelReduction {
                        reduction_staging,
                        inner_partitions,
                        result_row_partitions,
                        result_column_partitions,
                        ..
                    },
                ..
            } => (
                Some(orientation),
                Some(reduction_staging),
                Some(inner_partitions),
                Some((result_row_partitions, result_column_partitions)),
            ),
            OperatorDispatch::BlockedGemm { orientation, .. } => {
                (Some(orientation), None, Some(1), None)
            }
            _ => (None, None, None, None),
        };
    OperatorCompatibility {
        orientation,
        reduction_staging,
        inner_partitions,
        result_partitions,
        inputs: candidate
            .requirements
            .inputs
            .iter()
            .map(|input| {
                (
                    input.format.precision,
                    input.format.layout.order,
                    input.format.layout.memory_class,
                    input.local_staging,
                )
            })
            .collect(),
        output: (
            candidate.requirements.output.format.precision,
            candidate.requirements.output.format.layout.order,
            candidate.requirements.output.format.layout.memory_class,
            candidate
                .requirements
                .output
                .format
                .layout
                .tiling
                .axes
                .iter()
                .map(|axis| (axis.axis, axis.padding_groups, axis.shard_padding_multiple))
                .collect(),
        ),
    }
}

#[cfg(test)]
pub(super) fn retain_operator_candidates(
    candidates: Vec<OperatorPlan>,
    inputs: &[TensorType],
    output: &TensorShape,
    costs: &impl CostModel,
    width: usize,
) -> Vec<OperatorPlan> {
    retain_operator_candidates_for_demands(candidates, inputs, output, costs, width, &[])
}

pub(super) fn retain_operator_candidates_for_demands(
    candidates: Vec<OperatorPlan>,
    inputs: &[TensorType],
    output: &TensorShape,
    costs: &impl CostModel,
    width: usize,
    demands: &[OutputDemand],
) -> Vec<OperatorPlan> {
    if candidates.len() <= width {
        return candidates;
    }
    let ranked = candidates
        .into_par_iter()
        .map(|candidate| {
            let (planned_inputs, planned_output) = candidate.tensor_types(inputs, output);
            // Price the compact whole-device implementation, including staging
            // and reduction work. Boundary bytes are not an execution cost and
            // systematically discard useful larger-K, lower-fan-in GEMM grids.
            let implementation = costs.implementation(&candidate, &planned_inputs, &planned_output);
            let peak = implementation
                .as_ref()
                .map(|p| p.peak_memory)
                .unwrap_or_default();
            let objective = PlanMetrics {
                cycles: costs
                    .operator_cycle_override(&candidate, &planned_inputs, &planned_output)
                    .unwrap_or_else(|| {
                        implementation
                            .as_ref()
                            .map_or(u64::MAX, |p| p.estimated_cycles)
                    }),
                memory: peak,
            };
            let compatibility = operator_candidate_compatibility(&candidate);
            (candidate, objective, compatibility)
        })
        .collect::<Vec<_>>();
    let mut frontier = Vec::<(OperatorPlan, PlanMetrics, OperatorCompatibility)>::new();
    for entry in ranked {
        if frontier
            .iter()
            .any(|(_, kept, compatibility)| *compatibility == entry.2 && kept.dominates(entry.1))
        {
            continue;
        }
        frontier.retain(|(_, kept, compatibility)| {
            *compatibility != entry.2 || !entry.1.dominates(*kept)
        });
        frontier.push(entry);
    }
    let mut ranked = frontier;
    ranked.sort_by_key(|(_, objective, _)| {
        (
            objective.memory.standard_contiguous_overflow(),
            objective.cycles,
            objective.memory.total,
            objective.memory.interleaved,
            objective.memory.exchange_rows,
        )
    });
    let has_feasible = ranked
        .iter()
        .any(|(_, metrics, _)| metrics.memory.standard_contiguous_overflow() == 0);
    let mut selected = BTreeSet::new();
    // Reserve useful output families before fine-grained K/grid diversity.
    // All representatives are ranked by ordinary producer cost.
    if !demands.is_empty() {
        let mut families = BTreeSet::new();
        for (index, (plan, objective, _)) in ranked.iter().enumerate() {
            if selected.len() >= width {
                break;
            }
            if has_feasible && objective.memory.standard_contiguous_overflow() != 0 {
                continue;
            }
            let family = demands
                .iter()
                .map(|d| d.matches(&plan.requirements.output.format.layout, output))
                .collect::<Vec<_>>();
            if families.insert(family) {
                selected.insert(index);
            }
        }
    }
    let mut represented = BTreeSet::new();
    for (index, (_, objective, compatibility)) in ranked.iter().enumerate() {
        if selected.len() >= width {
            break;
        }
        if has_feasible && objective.memory.standard_contiguous_overflow() != 0 {
            continue;
        }
        // Preserve K splits and distinct scatter directions before spending
        // diversity slots on exact result factors or boundary grid extents.
        if represented.insert((
            compatibility.orientation,
            compatibility.inner_partitions,
            compatibility
                .result_partitions
                .map(|(rows, columns)| (rows > 1, columns > 1)),
            compatibility.output.0,
            compatibility.output.1,
            compatibility
                .inner_partitions
                .is_none()
                .then_some(&compatibility.output),
        )) {
            selected.insert(index);
        }
    }
    for (index, (_, objective, _)) in ranked.iter().enumerate() {
        if selected.len() >= width {
            break;
        }
        if !has_feasible || objective.memory.standard_contiguous_overflow() == 0 {
            selected.insert(index);
        }
    }
    // Preserve one tied memory/staging alternative per selected implementation.
    // Sharing its geometry slot avoids trading away a different useful grid.
    // This bounds the local pool at 2 * width; the region beam remains width.
    let representatives = selected.iter().copied().collect::<Vec<_>>();
    for representative in representatives {
        let (plan, metrics, compatibility) = &ranked[representative];
        if let Some((index, _)) =
            ranked
                .iter()
                .enumerate()
                .find(|(_, (candidate, objective, signature))| {
                    candidate.operator == plan.operator
                        && candidate.dispatch == plan.dispatch
                        && candidate.requirements.output == plan.requirements.output
                        && signature.inputs != compatibility.inputs
                        && objective.cycles == metrics.cycles
                        && objective.memory.standard_contiguous_overflow()
                            == metrics.memory.standard_contiguous_overflow()
                })
        {
            selected.insert(index);
        }
    }
    ranked
        .into_iter()
        .enumerate()
        .filter_map(|(index, (candidate, _, _))| selected.contains(&index).then_some(candidate))
        .collect()
}

pub(super) fn balance_parallel_gemm_columns(layout: &mut Layout, axis: TensorAxis) {
    if let Some(columns) = layout
        .tiling
        .axes
        .iter_mut()
        .find(|tiling| tiling.axis == axis)
    {
        columns.block_size = AMP_COLUMN_MICRO;
        columns.padding_multiple = AMP_COLUMN_MICRO;
        columns.padding = Padding::Zero;
    }
}

pub(super) fn apply_grouped_output_layout(
    input: &mut Layout,
    output: &mut Layout,
    grouping: GroupedOutputLayout,
) -> bool {
    if grouping.groups == 0 || grouping.physical_lane_multiple == 0 {
        return false;
    }
    let configure = |layout: &mut Layout| {
        let Some(axis) = layout
            .tiling
            .axes
            .iter_mut()
            .find(|axis| axis.axis == TensorAxis::FromEnd(1))
        else {
            return false;
        };
        if !axis.partitions.is_multiple_of(grouping.groups) {
            return false;
        }
        // Subdivide every semantic group independently. This permits several
        // shards per group while keeping padding at the group boundary rather
        // than inserting it at unrelated grid boundaries.
        axis.block_size = grouping.physical_lane_multiple;
        axis.padding_multiple = grouping.physical_lane_multiple;
        axis.padding_groups = grouping.groups;
        axis.shard_padding_multiple = 1;
        axis.padding = Padding::Zero;
        true
    };
    configure(input) && configure(output)
}

pub(super) fn pad_axis_to_f16_exchange_word(layout: &mut Layout, axis: TensorAxis) {
    if let Some(tiling) = layout
        .tiling
        .axes
        .iter_mut()
        .find(|tiling| tiling.axis == axis)
    {
        tiling.block_size = tiling.block_size.div_ceil(2) * 2;
        tiling.padding_multiple = tiling.padding_multiple.div_ceil(2) * 2;
        tiling.padding = Padding::Zero;
    }
}

pub(super) fn resolved_output_aliasing(
    candidate: &OperatorPlan,
    inputs: &[TensorType],
    output: &TensorShape,
) -> OutputAliasing {
    match &candidate.requirements.output_aliasing {
        OutputAliasing::MayAliasInputs(indices) => OutputAliasing::MayAliasInputs(
            indices
                .iter()
                .copied()
                .filter(|index| {
                    alias_compatible(
                        usize::from(*index),
                        &candidate.requirements.inputs,
                        inputs,
                        &candidate.requirements.output,
                        output,
                    )
                })
                .collect(),
        ),
        aliasing => aliasing.clone(),
    }
}

pub(super) fn operator_matches(operation: &OperationKind, operator: MidOperator) -> bool {
    match (operation, operator) {
        (OperationKind::Gemm(expected), MidOperator::Gemm { options, .. }) => *expected == options,
        (OperationKind::LayerNorm, MidOperator::LayerNorm) => true,
        (OperationKind::Gelu, MidOperator::Gelu) => true,
        (OperationKind::Add, MidOperator::Add) => true,
        (OperationKind::View(expected), MidOperator::View(view)) => *expected == view,
        (OperationKind::Slice(expected), MidOperator::Slice(slice)) => *expected == slice,
        (OperationKind::FlashAttention(expected), MidOperator::FlashAttention { options, .. }) => {
            *expected == options
        }
        _ => false,
    }
}
