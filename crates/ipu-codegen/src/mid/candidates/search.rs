//! Candidate eligibility for one whole-device operation.
//! This boundary accepts tensor facts, never mutable planner state.

use super::*;

pub(in crate::mid) struct CandidateSearch<'a> {
    operation: &'a Operation,
    consumers: &'a [Operation],
    value_uses: &'a BTreeMap<ValueId, usize>,
    config: &'a PipelineConfig,
    demands: &'a OutputDemands,
    distributed_result_is_useful: bool,
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
        Self {
            operation,
            consumers,
            value_uses,
            config,
            demands,
            distributed_result_is_useful,
        }
    }

    pub(in crate::mid) fn generate(
        &self,
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
            for mut plan in plans(
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
                if let OutputAliasing::MayAliasInputs(indices) =
                    &mut plan.requirements.output_aliasing
                {
                    // An input is writable only when this is its final use.
                    // Parameters remain live across host inference calls.
                    indices.retain(|&index| {
                        let index = usize::from(index);
                        !parameter_inputs[index]
                            && self.value_uses.get(&operation.inputs[index]) == Some(&1)
                    });
                }
                if !generated.contains(&plan) {
                    generated.push(plan);
                }
            }
        }
        let candidate_plans = generated
            .into_iter()
            .filter(|plan| {
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
                                    .supports_micro_panel_exchange(&requirement.format))
                            || !requirement.format.layout.order.requires_direct_population()
                            || (current.order == ElementOrder::RowMajor
                                && requirement
                                    .format
                                    .layout
                                    .order
                                    .supports_row_major_population())
                    })
            })
            .collect::<Vec<_>>();
        let candidate_plans = if matches!(operation.kind, OperationKind::Gemm(_)) {
            retain_operator_candidates_for_demands(
                candidate_plans,
                input_types,
                output_shape,
                costs,
                config.operator_candidate_limit.max(1),
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
                                .supports_micro_panel_exchange(&requirement.format)
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
