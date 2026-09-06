//! Candidate eligibility and caching for one operation across beam branches.
//! This boundary accepts tensor facts, never mutable planner state.

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PlanCacheKey {
    input_shapes: Vec<TensorShape>,
    parameter_inputs: Vec<bool>,
    format_sensitive_inputs: Vec<(usize, TensorFormat)>,
}

pub(in crate::mid) struct CandidateSearch<'a> {
    operation: &'a Operation,
    consumers: &'a [Operation],
    value_uses: &'a BTreeMap<ValueId, usize>,
    config: &'a PipelineConfig,
    demands: &'a OutputDemands,
    format_sensitive_indices: BTreeSet<usize>,
    distributed_result_is_useful: bool,
    cache: BTreeMap<PlanCacheKey, Vec<OperatorPlan>>,
    pub generated_plan_sets: usize,
    pub plan_cache_hits: usize,
}

impl<'a> CandidateSearch<'a> {
    pub(in crate::mid) fn new(
        operation: &'a Operation,
        consumers: &'a [Operation],
        result_required: bool,
        value_uses: &'a BTreeMap<ValueId, usize>,
        config: &'a PipelineConfig,
        demands: &'a OutputDemands,
    ) -> Self {
        let distributed_result_is_useful = operation.results.first().is_some_and(|result| {
            result_required
                || (value_uses.get(result).copied() == Some(1)
                    && consumers
                        .iter()
                        .find(|consumer| consumer.inputs.contains(result))
                        .is_some_and(|consumer| {
                            config
                                .operator_candidates
                                .iter()
                                .filter(|candidate| {
                                    operator_matches(&consumer.kind, candidate.operator())
                                })
                                .any(|candidate| {
                                    matches!(
                                        candidate.format_policy(),
                                        OperatorFormatPolicy::PreserveInputLayout(_)
                                    )
                                })
                        }))
        });
        let format_sensitive_indices = if matches!(operation.kind, OperationKind::View(_)) {
            (0..operation.inputs.len()).collect::<BTreeSet<_>>()
        } else {
            config
                .operator_candidates
                .iter()
                .filter(|candidate| operator_matches(&operation.kind, candidate.operator()))
                .filter_map(|candidate| match candidate.format_policy() {
                    OperatorFormatPolicy::Concrete => None,
                    OperatorFormatPolicy::PreserveInputLayout(index) => Some(usize::from(index)),
                })
                .collect()
        };
        Self {
            operation,
            consumers,
            value_uses,
            config,
            demands,
            format_sensitive_indices,
            distributed_result_is_useful,
            cache: BTreeMap::new(),
            generated_plan_sets: 0,
            plan_cache_hits: 0,
        }
    }

    pub(in crate::mid) fn generate(
        &mut self,
        input_types: &[TensorType],
        parameter_inputs: &[bool],
        automatic_inputs: &[bool],
        output_shape: &TensorShape,
        costs: &impl CostModel,
    ) -> LoweringResult<Vec<OperatorPlan>> {
        let operation = self.operation;
        let config = self.config;
        if matches!(operation.kind, OperationKind::Gemm(_))
            && input_types.get(1).is_some_and(|right| {
                right.shape.0[..right.shape.0.len().saturating_sub(2)]
                    .iter()
                    .any(|&extent| extent != 1)
            })
        {
            return Err(LoweringError::UnsupportedGemmBatching(operation.id));
        }
        let cache_key = PlanCacheKey {
            input_shapes: input_types
                .iter()
                .map(|input| input.shape.clone())
                .collect(),
            parameter_inputs: parameter_inputs.to_vec(),
            format_sensitive_inputs: self
                .format_sensitive_indices
                .iter()
                .filter_map(|&index| {
                    input_types
                        .get(index)
                        .map(|input| (index, input.format.clone()))
                })
                .collect(),
        };
        let cached = if let Some(cached) = self.cache.get(&cache_key) {
            self.plan_cache_hits += 1;
            cached
        } else {
            self.generated_plan_sets += 1;
            let output_demands = self.demands.get(operation.results[0]);
            let mut groupings = output_demands
                .iter()
                .map(|d| d.column_groups)
                .filter(|&groups| groups > 1)
                .collect::<BTreeSet<_>>();
            if let Some(grouping) =
                grouped_output_layout(self.consumers, operation, output_shape, self.value_uses)
            {
                groupings.insert(grouping.groups);
            }
            let mut groupings = groupings
                .into_iter()
                .map(|groups| {
                    Some(GroupedOutputLayout {
                        groups,
                        physical_lane_multiple: AMP_COLUMN_MICRO,
                    })
                })
                .collect::<Vec<_>>();
            if groupings.is_empty() {
                groupings.push(None);
            }
            let direct_consumer_layouts =
                direct_consumer_layouts(self.consumers, operation.results[0], output_shape, config);
            let mut generated = Vec::new();
            for grouped_output in groupings {
                for plan in plans(
                    operation,
                    input_types,
                    parameter_inputs,
                    output_shape,
                    config,
                    costs,
                    self.distributed_result_is_useful,
                    grouped_output,
                    &direct_consumer_layouts,
                    output_demands,
                ) {
                    if !generated.contains(&plan) {
                        generated.push(plan);
                    }
                }
            }
            self.cache.entry(cache_key).or_insert(generated)
        };
        let candidate_plans = cached
            .iter()
            .filter(|&plan| {
                input_types
                    .iter()
                    .zip(automatic_inputs)
                    .zip(&plan.requirements.inputs)
                    .all(|((input, &automatic), requirement)| {
                        let current = &input.format.layout;
                        automatic
                            || current.order == requirement.format.layout.order
                            || (config.conversion_streaming != ConversionStreamingPolicy::Always
                                && input
                                    .format
                                    .supports_f16_micro_panel_exchange(&requirement.format))
                            || !requirement.format.layout.order.requires_direct_population()
                            || (current.order == ElementOrder::RowMajor
                                && requirement
                                    .format
                                    .layout
                                    .order
                                    .supports_row_major_population())
                    })
            })
            .cloned()
            .collect::<Vec<_>>();
        let candidate_plans = if matches!(operation.kind, OperationKind::Gemm(_)) {
            retain_operator_candidates_for_demands(
                candidate_plans,
                input_types,
                output_shape,
                costs,
                config.planning_beam_width.max(1),
                self.demands.get(operation.results[0]),
            )
        } else {
            candidate_plans
        };
        let candidate_plans = candidate_plans
            .into_iter()
            .flat_map(|plan| {
                let mut complete = plan.clone();
                for requirement in &mut complete.requirements.inputs {
                    requirement.materialization = OperandMaterialization::Complete;
                }
                let panel_population = input_types.iter().zip(&plan.requirements.inputs).any(
                    |(input, requirement)| {
                        input.format.layout.order != requirement.format.layout.order
                            && requirement.format.layout.order.requires_direct_population()
                            && input
                                .format
                                .supports_f16_micro_panel_exchange(&requirement.format)
                    },
                );
                // Cross-order panel exchange is currently implemented for a
                // complete value, not the dispatch-slice staging ABI.
                if panel_population {
                    return vec![complete];
                }
                match config.conversion_streaming {
                    ConversionStreamingPolicy::Never => vec![complete],
                    ConversionStreamingPolicy::Always => vec![plan],
                    ConversionStreamingPolicy::WhenRequired if complete == plan => {
                        vec![complete]
                    }
                    ConversionStreamingPolicy::WhenRequired => vec![complete, plan],
                }
            })
            .collect::<Vec<_>>();
        Ok(candidate_plans)
    }
}
