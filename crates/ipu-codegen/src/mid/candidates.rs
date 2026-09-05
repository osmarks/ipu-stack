//! Shape-dependent implementation choices and local candidate pruning.

use super::*;

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

pub(super) fn grouped_output_layout(
    source: &[Operation],
    operation_index: usize,
    operation: &Operation,
    output: &TensorShape,
    value_uses: &BTreeMap<ValueId, usize>,
) -> Option<GroupedOutputLayout> {
    let result = *operation.results.first()?;
    if value_uses.get(&result).copied() != Some(1) {
        return None;
    }
    let consumer = source[operation_index + 1..]
        .iter()
        .find(|candidate| candidate.inputs.contains(&result))?;
    let OperationKind::SplitHeads(options) = consumer.kind else {
        return None;
    };
    let groups = u16::try_from(options.heads).ok()?;
    let width = *output.0.last()?;
    (groups != 0 && width.is_multiple_of(u32::from(groups))).then_some(GroupedOutputLayout {
        groups,
        physical_lane_multiple: AMP_COLUMN_MICRO,
    })
}

pub(super) fn direct_consumer_layouts(
    source: &[Operation],
    operation_index: usize,
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
    for consumer in &source[operation_index + 1..] {
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
) -> Vec<OperatorPlan> {
    let mut plans = Vec::new();
    let gemm_constraint = config
        .gemm_plan_constraints
        .iter()
        .find(|constraint| constraint.source_operation == operation.id.index());
    if let OperationKind::SplitHeads(options) = operation.kind
        && let [input] = inputs
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
                operator: MidOperator::View(AxisFactorView::new(2, 0, options.heads)),
                dispatch: OperatorDispatch::View,
                requirements: OperatorRequirements {
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
                    transform: AxisFactorView::new(2, 0, options.heads),
                    unfused_cycles: 0,
                    unfused_exchange_cycles: 0,
                }),
            };
            if !plans.contains(&plan) {
                plans.push(plan);
            }
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
                precision: Precision::F32,
                layout: Layout::attention_output(heads, query_partitions),
            };
            plans.push(OperatorPlan {
                operator: MidOperator::FlashAttention {
                    options,
                    accumulate: AccumulationPrecision::F32,
                },
                dispatch: OperatorDispatch::BlockedAttention {
                    query_key: TileKernelSpec::Gemm {
                        multiply: Precision::F16,
                        accumulate: AccumulationPrecision::F32,
                        mode: GemmKernelMode::Initialize,
                        weights: GemmWeightLoad::Standard,
                        inner_block: padded_query_dimension,
                        output_columns: AMP_INNER_BLOCK,
                    },
                    probability_value: TileKernelSpec::Gemm {
                        multiply: Precision::F16,
                        accumulate: AccumulationPrecision::F32,
                        mode: GemmKernelMode::Initialize,
                        weights: GemmWeightLoad::Standard,
                        inner_block: AMP_INNER_BLOCK,
                        output_columns: padded_value_dimension,
                    },
                    query_block_rows: query_rows.div_ceil(u32::from(query_partitions)),
                    key_block_rows: AMP_INNER_BLOCK,
                    padded_query_dimension,
                    padded_value_dimension,
                },
                requirements: OperatorRequirements {
                    inputs: vec![
                        OperandRequirement::new(query_format.clone(), 8)
                            .with_materialization(OperandMaterialization::DispatchSlices),
                        OperandRequirement::new(key_format.clone(), 8)
                            .with_materialization(OperandMaterialization::DispatchSlices),
                        OperandRequirement::new(value_format.clone(), 8)
                            .with_materialization(OperandMaterialization::DispatchSlices),
                    ],
                    output: OperandRequirement::new(output_format.clone(), 8),
                    output_aliasing: OutputAliasing::Fresh,
                    distinct_elements: Vec::new(),
                },
                deferred_output: None,
            });
            plans.push(OperatorPlan {
                operator: MidOperator::FlashAttention {
                    options,
                    accumulate: AccumulationPrecision::F32,
                },
                dispatch: OperatorDispatch::MaterializedAttention {
                    query_key: TileKernelSpec::Gemm {
                        multiply: Precision::F16,
                        accumulate: AccumulationPrecision::F32,
                        mode: GemmKernelMode::Initialize,
                        weights: GemmWeightLoad::Standard,
                        inner_block: padded_query_dimension,
                        output_columns: padded_key_rows,
                    },
                    probability_value: TileKernelSpec::Gemm {
                        multiply: Precision::F16,
                        accumulate: AccumulationPrecision::F32,
                        mode: GemmKernelMode::Initialize,
                        weights: GemmWeightLoad::Standard,
                        inner_block: padded_key_rows,
                        output_columns: padded_value_dimension,
                    },
                    query_block_rows: query_rows.div_ceil(u32::from(query_partitions)),
                    padded_key_rows,
                    padded_query_dimension,
                    padded_value_dimension,
                },
                requirements: OperatorRequirements {
                    inputs: vec![
                        OperandRequirement::new(query_format, 8)
                            .with_materialization(OperandMaterialization::DispatchSlices),
                        OperandRequirement::new(key_format, 8)
                            .with_materialization(OperandMaterialization::DispatchSlices),
                        OperandRequirement::new(value_format, 8)
                            .with_materialization(OperandMaterialization::DispatchSlices),
                    ],
                    output: OperandRequirement::new(output_format, 8),
                    output_aliasing: OutputAliasing::Fresh,
                    distinct_elements: Vec::new(),
                },
                deferred_output: None,
            });
        }
        match config.attention_strategy {
            AttentionStrategy::Automatic => {}
            AttentionStrategy::Flash => plans.retain(|plan| {
                !matches!(
                    plan.dispatch,
                    OperatorDispatch::MaterializedAttention { .. }
                )
            }),
            AttentionStrategy::Materialized => plans
                .retain(|plan| !matches!(plan.dispatch, OperatorDispatch::BlockedAttention { .. })),
        }
    }
    if let [input] = inputs
        && input.shape == *output
        && config.conversion_streaming != ConversionStreamingPolicy::Always
    {
        let mut flat_candidates = BTreeMap::new();
        for candidate in config.operator_candidates.iter().filter(|candidate| {
            operator_matches(&operation.kind, candidate.plan.operator)
                && candidate.plan.requirements.inputs.len() == 1
                && matches!(
                    candidate.format_policy,
                    OperatorFormatPolicy::PreserveInputLayout(0)
                )
                && matches!(
                    candidate.plan.dispatch,
                    OperatorDispatch::Pointwise {
                        input_mapping: PointwiseInputMapping::TileLocal,
                        ..
                    }
                )
                && candidate.plan.requirements.inputs[0].format.precision == input.format.precision
        }) {
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
                    requirements: OperatorRequirements {
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
        .filter(|candidate| operator_matches(&operation.kind, candidate.plan.operator))
    {
        let OperatorCandidate {
            plan: mut candidate,
            format_policy,
        } = candidate.clone();
        if let OperatorFormatPolicy::PreserveInputLayout(index) = format_policy {
            let Some((actual, requirement)) = inputs
                .get(usize::from(index))
                .zip(candidate.requirements.inputs.get_mut(usize::from(index)))
            else {
                continue;
            };
            if actual.format.precision != requirement.format.precision
                || candidate.requirements.output.format.precision != requirement.format.precision
            {
                continue;
            }
            requirement.format.layout = actual.format.layout.clone();
            candidate.requirements.output.format.layout = actual.format.layout.clone();
        } else if let OperatorFormatPolicy::PreserveInputTiling(index) = format_policy {
            let Some((actual, requirement)) = inputs
                .get(usize::from(index))
                .zip(candidate.requirements.inputs.get_mut(usize::from(index)))
            else {
                continue;
            };
            if actual.format.precision != requirement.format.precision
                || candidate.requirements.output.format.precision != requirement.format.precision
                || actual.format.layout.order != requirement.format.layout.order
            {
                continue;
            }
            requirement.format.layout = actual.format.layout.clone();
            candidate.requirements.output.format.layout.tiling =
                actual.format.layout.tiling.clone();
        }
        let mut variants = vec![candidate.clone()];
        variants.extend(parallel_reduction_candidates(
            &candidate,
            inputs,
            output,
            config,
            costs,
            distributed_result_is_useful,
            gemm_constraint,
            grouped_output,
        ));
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
    plans.retain(|plan| plan_fits_operator_memory(plan, inputs, output, config));
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
        | OperatorDispatch::BlockedAttention { .. }
        | OperatorDispatch::MaterializedAttention { .. }
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

pub(super) fn parallel_reduction_candidates(
    candidate: &OperatorPlan,
    inputs: &[TensorType],
    output: &TensorShape,
    config: &PipelineConfig,
    costs: &impl CostModel,
    distributed_result_is_useful: bool,
    constraint: Option<&GemmPlanConstraint>,
    grouped_output: Option<GroupedOutputLayout>,
) -> Vec<OperatorPlan> {
    [GemmOrientation::Normal, GemmOrientation::Swapped]
        .into_iter()
        .flat_map(|orientation| {
            parallel_reduction_candidates_for_orientation(
                candidate,
                inputs,
                output,
                config,
                costs,
                orientation,
                distributed_result_is_useful,
                constraint,
                grouped_output,
            )
        })
        .collect()
}

pub(super) fn parallel_reduction_candidates_for_orientation(
    candidate: &OperatorPlan,
    inputs: &[TensorType],
    output: &TensorShape,
    config: &PipelineConfig,
    costs: &impl CostModel,
    orientation: GemmOrientation,
    distributed_result_is_useful: bool,
    constraint: Option<&GemmPlanConstraint>,
    grouped_output: Option<GroupedOutputLayout>,
) -> Vec<OperatorPlan> {
    // Residual supervisor, weight-feed, and worker setup cost after retained
    // state, measured on IPU21 independently of the four issue cycles per row.
    const AMP_F16_MICROBLOCK_FIXED_CYCLES: u64 = 160;
    let OperatorDispatch::BlockedGemm {
        output_column_block,
        distribution: GemmDistribution::OutputStationary,
        ..
    } = candidate.dispatch
    else {
        return Vec::new();
    };
    if !matches!(
        candidate.operator,
        MidOperator::Gemm {
            multiply: Precision::F16,
            ..
        }
    ) || output_column_block != AMP_OUTPUT_COLUMN_BLOCK
    {
        return Vec::new();
    }
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
    // Generate the shape-specialized family once from the ordinary C64 seed.
    // Each grid chooses the exact padded local K and C extents, so one tile
    // call traverses all of its AMP micro-groups without fixed K64/C64
    // boundaries.
    let tile_count = candidate
        .requirements
        .output
        .format
        .layout
        .tiling
        .tile_count;
    let column_groups = columns.div_ceil(AMP_COLUMN_MICRO);
    let inner_groups = inner.div_ceil(AMP_COLUMN_MICRO);
    let Ok(inner_groups) = u16::try_from(inner_groups) else {
        return Vec::new();
    };
    let Ok(column_groups) = u16::try_from(column_groups) else {
        return Vec::new();
    };
    let grouped_column_groups = grouped_output.and_then(|grouping| {
        let groups = u32::from(grouping.groups);
        (groups != 0 && columns.is_multiple_of(groups)).then(|| {
            let columns_per_group = columns / groups;
            columns_per_group
                .div_ceil(grouping.physical_lane_multiple)
                .saturating_mul(groups)
        })
    });
    let grouped_column_groups = grouped_column_groups
        .and_then(|groups| u16::try_from(groups).ok())
        .filter(|groups| *groups >= column_groups);
    let output_seed_partitions = candidate
        .requirements
        .output
        .format
        .layout
        .tiling
        .axes
        .iter()
        .find(|axis| axis.axis == TensorAxis::FromEnd(1))
        .map(|axis| axis.partitions);
    if output_seed_partitions != Some(tile_count)
        || candidate.requirements.inputs[1].format.layout.memory_class != MemoryClass::Ipu21Standard
        || candidate.requirements.inputs[1].local_staging != LocalOperandStaging::Direct
    {
        return Vec::new();
    }
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
                grouped_output.and_then(|grouping| {
                    let physical = grouped_column_groups?;
                    column_partitions
                        .is_multiple_of(grouping.groups)
                        .then_some((true, physical))
                }),
            ];
            for (grouped, physical_column_groups) in grouped_options.into_iter().flatten() {
                let row_partitions = (tile_count / inner_partitions / column_partitions)
                    .min(u16::try_from(rows).unwrap_or(u16::MAX));
                let used_tiles = row_partitions
                    .saturating_mul(column_partitions)
                    .saturating_mul(inner_partitions);
                if used_tiles < tile_count.div_ceil(2) || u32::from(row_partitions) > rows {
                    continue;
                }
                let local_rows = rows.div_ceil(u32::from(row_partitions));
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
                let row_run_cycles = outer_rows
                    .saturating_mul(u64::from(local_rows))
                    .saturating_mul(4)
                    .saturating_add(AMP_F16_MICROBLOCK_FIXED_CYCLES);
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
                    .saturating_mul(u64::from(AMP_COLUMN_MICRO))
                    .saturating_mul(
                        candidate.requirements.inputs[orientation.operand_indices().0]
                            .format
                            .precision
                            .bytes(),
                    );
                let right_bytes = u64::from(local_columns)
                    .saturating_mul(u64::from(AMP_COLUMN_MICRO))
                    .saturating_mul(u64::from(local_inner))
                    .saturating_mul(u64::from(AMP_COLUMN_MICRO))
                    .saturating_mul(
                        candidate.requirements.inputs[orientation.operand_indices().1]
                            .format
                            .precision
                            .bytes(),
                    );
                let partial_bytes = outer_rows
                    .saturating_mul(u64::from(local_rows))
                    .saturating_mul(u64::from(local_columns))
                    .saturating_mul(u64::from(AMP_COLUMN_MICRO))
                    .saturating_mul(candidate.requirements.output.format.precision.bytes());
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
            .saturating_mul(AMP_COLUMN_MICRO);
        let kernel_output_columns = u32::from(physical_column_groups)
            .div_ceil(u32::from(column_partitions))
            .saturating_mul(AMP_COLUMN_MICRO);
        let Ok(kernel_inner_block_u16) = u16::try_from(kernel_inner_block) else {
            continue;
        };
        for memory_class in [MemoryClass::Ipu21Standard, MemoryClass::Ipu21Interleaved] {
            let mut variant = candidate.clone();
            match orientation {
                GemmOrientation::Normal => {
                    variant.requirements.inputs[0].format.layout = Layout::amp_left_parallel_grid(
                        kernel_inner_block_u16,
                        used_tiles,
                        row_partitions,
                        column_partitions,
                        inner_partitions,
                    );
                    variant.requirements.inputs[1].format.layout =
                        Layout::block_major_matrix_storage(
                            kernel_inner_block_u16,
                            kernel_output_columns,
                            column_partitions,
                            inner_partitions,
                            1,
                            memory_class,
                        );
                    balance_parallel_gemm_columns(
                        &mut variant.requirements.inputs[1].format.layout,
                        TensorAxis::FromEnd(1),
                    );
                }
                GemmOrientation::Swapped => {
                    let mut physical_left = variant.requirements.inputs[1].clone();
                    physical_left.format.layout = Layout::amp_transposed_left_parallel_grid(
                        kernel_inner_block_u16,
                        used_tiles,
                        row_partitions,
                        column_partitions,
                        inner_partitions,
                    );
                    physical_left.materialization = OperandMaterialization::DispatchSlices;
                    let mut physical_right = variant.requirements.inputs[0].clone();
                    physical_right.format.layout = Layout::transposed_block_major_matrix_storage(
                        kernel_inner_block_u16,
                        kernel_output_columns,
                        column_partitions,
                        inner_partitions,
                        row_partitions,
                        memory_class,
                    );
                    balance_parallel_gemm_columns(
                        &mut physical_right.format.layout,
                        TensorAxis::FromEnd(2),
                    );
                    physical_right.materialization = OperandMaterialization::Complete;
                    variant.requirements.inputs = vec![physical_right, physical_left];
                    variant.requirements.distinct_elements =
                        vec![vec![MemoryOperand::Output, MemoryOperand::Input(1)]];
                }
            }
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
                if inner_partitions <= maximum_result_rows {
                    result_partition_options.push((inner_partitions, 1));
                } else if inner_partitions <= maximum_result_columns {
                    result_partition_options.push((1, inner_partitions));
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
                    let mut result_variant = variant.clone();
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
                    result_variant.requirements.output.format.layout = result_layout;
                    let (physical_row_axis, physical_rows, physical_left_index) = match orientation
                    {
                        GemmOrientation::Normal => (TensorAxis::FromEnd(2), normal_rows, 0),
                        GemmOrientation::Swapped => (TensorAxis::FromEnd(1), normal_columns, 1),
                    };
                    let result_rows = result_variant
                        .requirements
                        .output
                        .format
                        .layout
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
                        &mut result_variant.requirements.inputs[physical_left_index]
                            .format
                            .layout,
                        physical_row_axis,
                    );
                    pad_axis_to_f16_exchange_word(
                        &mut result_variant.requirements.output.format.layout,
                        physical_row_axis,
                    );
                    if grouped
                        && !grouped_output.is_some_and(|grouping| {
                            apply_grouped_output_layout(&mut result_variant, grouping)
                        })
                    {
                        continue;
                    }
                    for &local_staging in local_staging_options {
                        for reduction_staging in
                            [ReductionStaging::Complete, ReductionStaging::Streamed]
                        {
                            let mut staged = result_variant.clone();
                            staged.requirements.inputs[physical_right_index].local_staging =
                                local_staging;
                            staged.dispatch = OperatorDispatch::BlockedGemm {
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
                            variants.push(staged);
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
        retain_precise_operator_candidates(
            variants,
            inputs,
            output,
            costs,
            config.planning_beam_width.max(1),
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
    pub(super) inputs: Vec<(ElementOrderCompatibility, MemoryClass, LocalOperandStaging)>,
    pub(super) output: (
        ElementOrderCompatibility,
        MemoryClass,
        Vec<(TensorAxis, u16, u32)>,
    ),
}

pub(super) fn operator_candidate_compatibility(candidate: &OperatorPlan) -> OperatorCompatibility {
    let (orientation, reduction_staging) = match candidate.dispatch {
        OperatorDispatch::BlockedGemm {
            orientation,
            distribution:
                GemmDistribution::ParallelReduction {
                    reduction_staging, ..
                },
            ..
        } => (Some(orientation), Some(reduction_staging)),
        OperatorDispatch::BlockedGemm { orientation, .. } => (Some(orientation), None),
        _ => (None, None),
    };
    OperatorCompatibility {
        orientation,
        reduction_staging,
        inputs: candidate
            .requirements
            .inputs
            .iter()
            .map(|input| {
                (
                    element_order_compatibility(input.format.layout.order),
                    input.format.layout.memory_class,
                    input.local_staging,
                )
            })
            .collect(),
        output: (
            element_order_compatibility(candidate.requirements.output.format.layout.order),
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

pub(super) fn retain_precise_operator_candidates(
    candidates: Vec<OperatorPlan>,
    inputs: &[TensorType],
    output: &TensorShape,
    costs: &impl CostModel,
    width: usize,
) -> Vec<OperatorPlan> {
    let ranked = candidates
        .into_iter()
        .map(|candidate| {
            let (planned_inputs, planned_output) = candidate.tensor_types(inputs, output);
            let requirements = &candidate.requirements;
            let memory = operator_memory_estimate(
                &candidate.dispatch,
                requirements,
                &planned_inputs,
                &planned_output,
            );
            let exchange =
                costs.operator_exchange_footprint(&candidate, &planned_inputs, &planned_output);
            let objective = PlanMetrics {
                standard_contiguous_overflow: 0,
                cycles: costs.operator_cycles(&candidate, &planned_inputs, &planned_output),
                standard: memory.peak.standard,
                interleaved: memory.peak.interleaved,
                total: memory.peak.total(),
                maximum_standard_allocation: memory.maximum_standard_temporary_allocation,
                exchange_rows: exchange.estimated_row_bytes(),
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
            objective.cycles,
            objective.total,
            objective.interleaved,
            objective.exchange_rows,
        )
    });
    let mut selected = BTreeSet::new();
    let mut represented = BTreeSet::new();
    for (index, (_, _, compatibility)) in ranked.iter().enumerate() {
        if represented.insert(compatibility.clone()) {
            selected.insert(index);
        }
    }
    for index in 0..ranked.len() {
        if selected.len() == width {
            break;
        }
        selected.insert(index);
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
    candidate: &mut OperatorPlan,
    grouping: GroupedOutputLayout,
) -> bool {
    if candidate.requirements.output.format.precision != Precision::F16
        || grouping.groups == 0
        || grouping.physical_lane_multiple == 0
    {
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
    configure(&mut candidate.requirements.inputs[1].format.layout)
        && configure(&mut candidate.requirements.output.format.layout)
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
        (OperationKind::Gelu, MidOperator::Gelu) => true,
        (OperationKind::Add(expected), MidOperator::Add(options)) => *expected == options,
        (OperationKind::SplitHeads(expected), MidOperator::View(view)) => {
            view == AxisFactorView::new(2, 0, expected.heads)
        }
        (OperationKind::FlashAttention(expected), MidOperator::FlashAttention { options, .. }) => {
            *expected == options
        }
        _ => false,
    }
}
