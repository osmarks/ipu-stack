//! Region beam search, explicit conversion insertion, and Repeat planning.

use super::*;

#[cfg(test)]
#[tracing::instrument(
    name = "ipu_codegen.mid.lower",
    skip(graph, config, costs),
    fields(tile_count = config.tile_count, operations = graph.operations().len())
)]
pub fn lower(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
) -> LoweringResult<MidProgram> {
    Ok(plan_finalists(graph, config, costs, 1)?.remove(0))
}

pub(crate) fn plan_finalists(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
    finalist_count: usize,
) -> LoweringResult<Vec<MidProgram>> {
    // Fragment construction allocates substantially more than scalar costing.
    // Use a fixed small pool instead of spreading allocator arenas over every
    // logical CPU on large build hosts. Concurrent planners share this bound.
    static POOL: std::sync::OnceLock<Result<rayon::ThreadPool, String>> =
        std::sync::OnceLock::new();
    let pool = POOL
        .get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(
                    std::thread::available_parallelism()
                        .map_or(1, usize::from)
                        .min(8),
                )
                .thread_name(|index| format!("ipu-plan-{index}"))
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| LoweringError::PlanningThreads(error.clone()))?;
    pool.install(|| plan_in_pool(graph, config, costs, finalist_count))
}

fn plan_in_pool(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
    finalist_count: usize,
) -> LoweringResult<Vec<MidProgram>> {
    if config.tile_count == 0 {
        return Err(LoweringError::EmptyTileGroup);
    }
    let use_shape_aware_counts = config.shape_aware_active_tile_counts
        && config.operator_candidates == default_operator_candidates(config.tile_count);
    let resolved_config = use_shape_aware_counts.then(|| {
        let mut resolved = config.clone();
        for tile_count in
            shape_aware_active_tile_counts(config.tile_count, graph.value_shapes().values())
        {
            resolved
                .operator_candidates
                .extend(operator_candidates_for_tile_count(tile_count));
        }
        let mut unique = Vec::with_capacity(resolved.operator_candidates.len());
        for candidate in resolved.operator_candidates {
            if !unique.contains(&candidate) {
                unique.push(candidate);
            }
        }
        resolved.operator_candidates = unique;
        resolved
    });
    let config = resolved_config.as_ref().unwrap_or(config);
    let mut state = LoweringState::default();
    let costs = MemoizedCostModel::new(costs, config.tile_count);
    let mut values = BTreeMap::new();
    let mut inputs = Vec::with_capacity(graph.inputs().len());
    for input in graph.inputs() {
        let (format, automatic) = if let Some(format) = config.inputs.get(&input.value) {
            (format.clone(), false)
        } else if let Some(&precision) = config.automatic_inputs.get(&input.value) {
            (
                TensorFormat {
                    precision,
                    layout: Layout::row_sharded(config.tile_count),
                },
                true,
            )
        } else {
            return Err(LoweringError::MissingInputType(input.value));
        };
        let tensor_type = TensorType {
            shape: input.shape.clone(),
            format,
        };
        let value = state.value(input.value, tensor_type);
        if input.kind == GraphInputKind::Parameter {
            state.parameter_values.insert(value);
        }
        if automatic {
            state.automatic_inputs.insert(value);
        }
        values.insert(input.value, value);
        inputs.push(MidInput {
            name: input.name.clone(),
            kind: input.kind,
            value,
        });
    }
    let branches = lower_operation_candidates(
        graph.operations(),
        graph.outputs(),
        &mut values,
        graph.value_shapes(),
        graph,
        config,
        &costs,
        &mut state,
        &RegionPlanningConstraints::default(),
    )?;
    let initial = inputs.iter().map(|input| input.value).collect::<Vec<_>>();
    branches
        .into_iter()
        .take(finalist_count.max(1))
        .enumerate()
        .map(|(finalist, branch)| {
            let outputs = graph
                .outputs()
                .iter()
                .map(|value| lookup(&branch.values, *value))
                .collect::<LoweringResult<Vec<_>>>()?;
            let (estimated_cycles, estimated_exchange_cycles, peak_memory) =
                if let Some(Some((program, peak, true))) = branch.analysis.get() {
                    (program.total, program.exchange, *peak)
                } else {
                    (
                        branch.operations.iter().map(|operation| operation.estimated_cycles).sum(),
                        branch.operations.iter().map(|operation| operation.estimated_exchange_cycles).sum(),
                        region_peak_memory(&initial, &branch.operations, &outputs, &branch.state.values),
                    )
                };
            tracing::info!(
                finalist,
                values = branch.state.values.len(),
                operations = branch.operations.len(),
                estimated_cycles,
                estimated_exchange_cycles,
                exchange_row_bytes = peak_memory.exchange_rows,
                peak_standard_bytes = peak_memory.standard,
                peak_interleaved_bytes = peak_memory.interleaved,
                peak_total_bytes = peak_memory.total,
                maximum_standard_allocation_bytes = peak_memory.maximum_standard_allocation,
                active_tile_counts = ?branch.operations
                    .iter()
                    .filter_map(|operation| operation.results.first())
                    .map(|result| branch.state.values[result.index() as usize]
                        .tensor_type
                        .format
                        .layout
                        .tiling
                        .tile_count)
                    .collect::<BTreeSet<_>>(),
                padding_group_counts = ?branch.operations
                    .iter()
                    .flat_map(|operation| operation.results.iter())
                    .flat_map(|result| branch.state.values[result.index() as usize]
                        .tensor_type
                        .format
                        .layout
                        .tiling
                        .axes
                        .iter()
                        .map(|axis| axis.padding_groups))
                    .filter(|groups| *groups > 1)
                    .collect::<BTreeSet<_>>(),
                conversion_sources = ?branch.operations
                    .iter()
                    .filter(|operation| operation.conversion_plan().is_some())
                    .map(|operation| operation.source)
                    .collect::<Vec<_>>(),
                "retained operator-plan finalist"
            );
            tracing::debug!(
                finalist,
                plans = ?branch.operations
                    .iter()
                    .filter_map(|operation| operation.operator_plan().map(|plan| (
                        operation.source,
                        &plan.dispatch,
                        plan.requirements.inputs.iter().map(|input| &input.format.layout).collect::<Vec<_>>(),
                        &plan.requirements.output.format.layout,
                        operation.estimated_cycles,
                        operation.estimated_exchange_cycles,
                    )))
                    .collect::<Vec<_>>(),
                conversions = ?branch.operations
                    .iter()
                    .filter_map(|operation| operation.conversion_plan().map(|plan| (
                        operation.source,
                        &plan.input.format.layout,
                        &plan.output.format.layout,
                        operation.estimated_cycles,
                        operation.estimated_exchange_cycles,
                    )))
                    .collect::<Vec<_>>(),
                "retained operator-plan details"
            );
            let mut graph = MidProgram {
                tile_count: config.tile_count,
                inputs: inputs.clone(),
                values: branch.state.values,
                operations: branch.operations,
                outputs,
                estimated_cycles,
                estimated_exchange_cycles,
                peak_memory,
            };
            graph.assign_parameter_tiles()?;
            Ok(graph)
        })
        .collect()
}

#[derive(Clone, Default)]
pub(super) struct LoweringState {
    pub(super) values: Vec<MidValue>,
    pub(super) automatic_inputs: BTreeSet<MidValueId>,
    pub(super) parameter_values: BTreeSet<MidValueId>,
}

impl LoweringState {
    pub(super) fn value(&mut self, origin: ValueId, tensor_type: TensorType) -> MidValueId {
        let id = MidValueId(self.values.len() as u32);
        self.values.push(MidValue {
            id,
            tile_offset: 0,
            tensor_type,
            origin,
            storage_group: id,
        });
        id
    }

    pub(super) fn value_in_storage_group(
        &mut self,
        origin: ValueId,
        tensor_type: TensorType,
        storage_group: MidValueId,
    ) -> MidValueId {
        let result = self.value(origin, tensor_type);
        self.values[result.index() as usize].storage_group = storage_group;
        result
    }

    pub(super) fn get(&self, id: MidValueId) -> &MidValue {
        &self.values[id.0 as usize]
    }

    pub(super) fn derived_value(
        &mut self,
        source: MidValueId,
        tensor_type: TensorType,
    ) -> MidValueId {
        let origin = self.get(source).origin;
        let storage_group = self.get(source).storage_group;
        let result = self.value_in_storage_group(origin, tensor_type, storage_group);
        if self.parameter_values.contains(&source) {
            self.parameter_values.insert(result);
        }
        result
    }

    pub(super) fn retarget_automatic_input(&mut self, id: MidValueId, layout: Layout) -> bool {
        if !self.automatic_inputs.remove(&id) {
            return false;
        }
        self.values[id.0 as usize].tensor_type.format.layout = layout;
        true
    }
}

type BranchAnalysis = Option<(crate::estimate::ProgramCycles, MemoryPeaks, bool)>;

#[derive(Clone)]
pub(super) struct BeamBranch {
    pub(super) values: BTreeMap<ValueId, MidValueId>,
    pub(super) state: LoweringState,
    pub(super) operations: Vec<MidOperation>,
    pub(super) peak_memory: MemoryPeaks,
    pub(super) analysis: std::sync::OnceLock<BranchAnalysis>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct FutureValueState {
    pub(super) origin: ValueId,
    pub(super) tensor_type: TensorType,
    pub(super) automatic_input: bool,
    pub(super) parameter: bool,
    pub(super) allocation_copies: u32,
    pub(super) storage_class: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct FutureDeferredState {
    pub(super) origin: ValueId,
    pub(super) source_type: TensorType,
    pub(super) source_automatic_input: bool,
    pub(super) source_parameter: bool,
    pub(super) source_storage_class: u32,
    pub(super) transform: AxisFactorView,
    pub(super) unfused_cycles: u64,
    pub(super) claimed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct FutureBeamState {
    pub(super) values: Vec<FutureValueState>,
    pub(super) deferred: Vec<FutureDeferredState>,
    pub(super) equal_formats_satisfied: Vec<(ValueId, ValueId, bool)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PlanMetrics {
    pub(super) cycles: u64,
    pub(super) standard: u64,
    pub(super) interleaved: u64,
    pub(super) total: u64,
    pub(super) maximum_standard_allocation: u64,
    pub(super) standard_contiguous_overflow: u64,
    pub(super) exchange_rows: u64,
}

impl PlanMetrics {
    pub(super) fn dominates(self, other: Self) -> bool {
        let no_worse = self.cycles <= other.cycles
            && self.standard <= other.standard
            && self.interleaved <= other.interleaved
            && self.total <= other.total
            && self.maximum_standard_allocation <= other.maximum_standard_allocation
            && self.standard_contiguous_overflow <= other.standard_contiguous_overflow
            && self.exchange_rows <= other.exchange_rows;
        no_worse && self != other
    }
}

pub(super) struct RankedBeamBranch {
    pub(super) branch: BeamBranch,
    pub(super) objective: PlanMetrics,
    pub(super) compatibility: FutureFormatCompatibility,
    pub(super) order: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct FutureFormatCompatibility(
    Vec<(
        ValueId,
        FutureFormatRole,
        Precision,
        ElementOrderCompatibility,
        MemoryClass,
        Vec<(TensorAxis, u16, u32)>,
    )>,
);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum FutureFormatRole {
    Value,
    DeferredSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum ElementOrderCompatibility {
    RowMajor,
    BlockMajorMatrix,
    BlockMajorTransposedMatrix,
    Amp(AmpOrder),
}

pub(super) fn element_order_compatibility(order: ElementOrder) -> ElementOrderCompatibility {
    match order {
        ElementOrder::RowMajor => ElementOrderCompatibility::RowMajor,
        ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. }) => {
            ElementOrderCompatibility::BlockMajorMatrix
        }
        ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix { .. }) => {
            ElementOrderCompatibility::BlockMajorTransposedMatrix
        }
        ElementOrder::Amp(order) => ElementOrderCompatibility::Amp(order),
    }
}

pub(super) fn future_format_compatibility(
    branch: &BeamBranch,
    future_origins: &BTreeSet<ValueId>,
) -> FutureFormatCompatibility {
    let mut formats = Vec::new();
    for &origin in future_origins {
        let Some(&id) = branch.values.get(&origin) else {
            continue;
        };
        let mut add = |role, format: &TensorFormat| {
            let axes = format
                .layout
                .tiling
                .axes
                .iter()
                .map(|axis| (axis.axis, axis.padding_groups, axis.shard_padding_multiple))
                .collect();
            formats.push((
                origin,
                role,
                format.precision,
                element_order_compatibility(format.layout.order),
                format.layout.memory_class,
                axes,
            ));
        };
        add(
            FutureFormatRole::Value,
            &branch.state.get(id).tensor_type.format,
        );
        let deferred_source = branch
            .operations
            .iter()
            .rev()
            .find(|operation| operation.results.first() == Some(&id))
            .and_then(|operation| {
                let offer = operation.operator_plan()?.deferred_output?;
                operation.inputs.get(offer.source_input).copied()
            });
        if let Some(source) = deferred_source {
            add(
                FutureFormatRole::DeferredSource,
                &branch.state.get(source).tensor_type.format,
            );
        }
    }
    FutureFormatCompatibility(formats)
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct PlanCacheKey {
    pub(super) input_shapes: Vec<TensorShape>,
    pub(super) parameter_inputs: Vec<bool>,
    pub(super) format_sensitive_inputs: Vec<(usize, TensorFormat)>,
}

#[derive(Default)]
pub(super) struct RegionPlanningConstraints {
    /// Number of simultaneously resident blocks represented by a region value.
    pub(super) allocation_copies: BTreeMap<ValueId, u32>,
    /// Value pairs whose formats must agree at a structured-region boundary.
    pub(super) required_equal_formats: Vec<(ValueId, ValueId)>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn lower_operation_candidates(
    source: &[Operation],
    required_outputs: &[ValueId],
    values: &mut BTreeMap<ValueId, MidValueId>,
    shapes: &BTreeMap<ValueId, TensorShape>,
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
    state: &mut LoweringState,
    constraints: &RegionPlanningConstraints,
) -> LoweringResult<Vec<BeamBranch>> {
    if source.is_empty() {
        return Ok(vec![BeamBranch {
            values: values.clone(),
            state: state.clone(),
            operations: Vec::new(),
            peak_memory: MemoryPeaks::default(),
            analysis: std::sync::OnceLock::new(),
        }]);
    }
    let relevant_origins = source
        .iter()
        .flat_map(|operation| operation_graph_inputs(operation, graph))
        .chain(required_outputs.iter().copied())
        .chain(
            constraints
                .required_equal_formats
                .iter()
                .flat_map(|pair| [pair.0, pair.1]),
        )
        .chain(constraints.allocation_copies.keys().copied())
        .collect::<BTreeSet<_>>();
    let initial = relevant_origins
        .iter()
        .filter_map(|origin| values.get(origin).copied())
        .collect::<Vec<_>>();
    let mut value_uses = BTreeMap::<ValueId, usize>::new();
    for value in source
        .iter()
        .flat_map(|operation| operation_graph_inputs(operation, graph))
        .chain(required_outputs.iter().copied())
    {
        *value_uses.entry(value).or_default() += 1;
    }
    let mut beam = vec![BeamBranch {
        values: values.clone(),
        state: state.clone(),
        operations: Vec::new(),
        peak_memory: MemoryPeaks::default(),
        analysis: std::sync::OnceLock::new(),
    }];
    for (operation_index, operation) in source.iter().enumerate() {
        let distributed_result_is_useful = operation.results.first().is_some_and(|result| {
            required_outputs.contains(result)
                || (value_uses.get(result).copied() == Some(1)
                    && source[operation_index + 1..]
                        .iter()
                        .find(|consumer| consumer.inputs.contains(result))
                        .is_some_and(|consumer| {
                            config
                                .operator_candidates
                                .iter()
                                .filter(|candidate| {
                                    operator_matches(&consumer.kind, candidate.plan.operator)
                                })
                                .any(|candidate| {
                                    matches!(
                                        candidate.format_policy,
                                        OperatorFormatPolicy::PreserveInputLayout(_)
                                    )
                                })
                        }))
        });
        let future_origins = source[operation_index + 1..]
            .iter()
            .flat_map(|operation| operation_graph_inputs(operation, graph))
            .chain(required_outputs.iter().copied())
            .chain(
                constraints
                    .required_equal_formats
                    .iter()
                    .flat_map(|pair| [pair.0, pair.1]),
            )
            .chain(constraints.allocation_copies.keys().copied())
            .collect::<BTreeSet<_>>();
        // Compose full region liveness only for this wider shortlist. Each
        // operator already has a compact implementation and execution price.
        let screening_width = config.planning_beam_width.max(1).saturating_mul(2);
        let mut expanded = Vec::new();
        let mut rejected_memory = Vec::new();
        let mut saw_candidate = false;
        let format_sensitive_indices = if matches!(operation.kind, OperationKind::View(_)) {
            (0..operation.inputs.len()).collect::<BTreeSet<_>>()
        } else {
            config
                .operator_candidates
                .iter()
                .filter(|candidate| operator_matches(&operation.kind, candidate.plan.operator))
                .filter_map(|candidate| match candidate.format_policy {
                    OperatorFormatPolicy::Concrete => None,
                    OperatorFormatPolicy::PreserveInputLayout(index) => Some(usize::from(index)),
                })
                .collect()
        };
        let mut plan_cache = BTreeMap::<PlanCacheKey, Vec<OperatorPlan>>::new();
        let mut plan_cache_hits = 0usize;
        let mut generated_plan_sets = 0usize;
        for branch in beam {
            if let OperationKind::Repeat(repeat) = &operation.kind {
                saw_candidate = true;
                let mut next = branch.clone();
                next.analysis.take();
                lower_repeat(
                    operation,
                    repeat,
                    &mut next.values,
                    graph,
                    config,
                    costs,
                    &mut next.state,
                    &mut next.operations,
                )?;
                expanded.push(next);
                continue;
            }
            let input_ids = operation
                .inputs
                .iter()
                .map(|value| lookup(&branch.values, *value))
                .collect::<LoweringResult<Vec<_>>>()?;
            let input_types = input_ids
                .iter()
                .map(|value| branch.state.get(*value).tensor_type.clone())
                .collect::<Vec<_>>();
            if matches!(operation.kind, OperationKind::Gemm(_))
                && input_types.get(1).is_some_and(|right| {
                    right.shape.0[..right.shape.0.len().saturating_sub(2)]
                        .iter()
                        .any(|&extent| extent != 1)
                })
            {
                return Err(LoweringError::UnsupportedGemmBatching(operation.id));
            }
            let output_shape = shapes
                .get(&operation.results[0])
                .cloned()
                .ok_or(LoweringError::MissingShape(operation.results[0]))?;
            let grouped_output = grouped_output_layout(
                source,
                operation_index,
                operation,
                &output_shape,
                &value_uses,
            );
            let direct_consumer_layouts = direct_consumer_layouts(
                source,
                operation_index,
                operation.results[0],
                &output_shape,
                config,
            );
            let parameter_inputs = input_ids
                .iter()
                .map(|id| branch.state.parameter_values.contains(id))
                .collect::<Vec<_>>();
            let cache_key = PlanCacheKey {
                input_shapes: input_types
                    .iter()
                    .map(|input| input.shape.clone())
                    .collect(),
                parameter_inputs: parameter_inputs.clone(),
                format_sensitive_inputs: format_sensitive_indices
                    .iter()
                    .filter_map(|&index| {
                        input_types
                            .get(index)
                            .map(|input| (index, input.format.clone()))
                    })
                    .collect(),
            };
            let cached = if let Some(cached) = plan_cache.get(&cache_key) {
                plan_cache_hits += 1;
                cached
            } else {
                generated_plan_sets += 1;
                let generated = plans(
                    operation,
                    &input_types,
                    &parameter_inputs,
                    &output_shape,
                    config,
                    costs,
                    distributed_result_is_useful,
                    grouped_output,
                    &direct_consumer_layouts,
                );
                plan_cache.entry(cache_key).or_insert(generated)
            };
            let candidate_plans = cached
                .iter()
                .filter(|&plan| {
                    input_ids
                        .iter()
                        .zip(&plan.requirements.inputs)
                        .all(|(id, requirement)| {
                            let current = &branch.state.get(*id).tensor_type.format.layout;
                            branch.state.automatic_inputs.contains(id)
                                || current.order == requirement.format.layout.order
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
                retain_operator_candidates(
                    candidate_plans,
                    &input_types,
                    &output_shape,
                    costs,
                    config.planning_beam_width.max(1),
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
            saw_candidate |= !candidate_plans.is_empty();
            let evaluated = candidate_plans
                .into_par_iter()
                .map(|plan| {
                    let mut next = branch.clone();
                    next.analysis.take();
                    apply_selected_plan(
                        operation,
                        output_shape.clone(),
                        plan,
                        &operation
                            .inputs
                            .iter()
                            .map(|value| value_uses.get(value).copied().unwrap_or(0) == 1)
                            .collect::<Vec<_>>(),
                        costs,
                        &mut next.values,
                        &mut next.state,
                        &mut next.operations,
                    );
                    let boundary = next.operations.last().unwrap();
                    let tensors = boundary
                        .inputs
                        .iter()
                        .chain(&boundary.results)
                        .map(|id| &next.state.get(*id).tensor_type);
                    let mut usage = MemoryUsage::default();
                    let mut maximum_standard = 0;
                    for tensor in tensors {
                        let memory = crate::estimate::tensor_memory(tensor);
                        usage = usage.saturating_add(memory);
                        maximum_standard = maximum_standard.max(memory.standard);
                    }
                    next.peak_memory.observe(usage, maximum_standard);
                    next
                })
                .collect::<Vec<_>>();
            expanded.extend(evaluated);
            // Prune between parent branches as well as at the end of the
            // operation. Executable fragments can be large; retaining every
            // rejected Cartesian-product branch defeats the planning beam.
            if expanded.len() > screening_width.saturating_mul(2) {
                expanded = retain_pareto_beam(
                    expanded,
                    &future_origins,
                    constraints,
                    costs,
                    screening_width,
                )
                .0;
            }
        }
        let shortlisted = retain_pareto_beam(
            expanded,
            &future_origins,
            constraints,
            costs,
            screening_width,
        )
        .0;
        let evaluated = shortlisted
            .into_par_iter()
            .map(|mut branch| {
                // Construct only shortlisted implementations, retaining the
                // fragments that the region builder will bind below.
                for operation in &mut branch.operations {
                    if let MidOperationKind::Operator {
                        plan,
                        implementation,
                        ..
                    } = &mut operation.kind
                        && implementation.is_none()
                    {
                        let inputs = operation
                            .inputs
                            .iter()
                            .map(|id| branch.state.get(*id).tensor_type.clone())
                            .collect::<Vec<_>>();
                        *implementation = costs.implementation(
                            plan,
                            &inputs,
                            &branch.state.get(operation.results[0]).tensor_type,
                        );
                    }
                }
                let peak = beam_memory_peak(
                    costs,
                    &branch,
                    &initial,
                    source,
                    operation_index,
                    required_outputs,
                    graph,
                    &constraints.allocation_copies,
                );
                branch.peak_memory = peak;
                (branch, peak)
            })
            .collect::<Vec<_>>();
        let mut expanded = Vec::new();
        for (branch, peak) in evaluated {
            if peak.fits_ipu21_with_budget(
                config.standard_memory_reservation_bytes,
                config.tile_memory_budget_bytes,
            ) || branch_contains_gemm_constraint(&branch, config)
            {
                expanded.push(branch);
            } else {
                rejected_memory.push(peak);
            }
        }
        if expanded.is_empty() {
            if saw_candidate
                && let Some(peak) = rejected_memory
                    .into_iter()
                    .min_by_key(|peak| (peak.total, peak.interleaved, peak.standard))
            {
                return Err(LoweringError::InsufficientMemory {
                    operation: operation.id,
                    standard: peak.standard,
                    standard_reservation: config.standard_memory_reservation_bytes,
                    interleaved: peak.interleaved,
                    total: peak.total,
                    standard_contiguous_overflow: peak
                        .standard_contiguous_overflow_with_reservation(
                            config.standard_memory_reservation_bytes,
                        ),
                });
            }
            return Err(LoweringError::NoCandidate(operation.id));
        }
        let expanded_count = expanded.len();
        let (expanded, dominated, equivalent, diversity) = retain_pareto_beam(
            expanded,
            &future_origins,
            constraints,
            costs,
            config.planning_beam_width.max(1),
        );
        tracing::debug!(
            operation = operation.id.index(),
            retained = expanded.len(),
            expanded = expanded_count,
            pareto_dominated = dominated,
            equivalent,
            diversity_representatives = diversity,
            best_cycles = deferred_aware_branch_score(&expanded[0], &future_origins),
            generated_plan_sets,
            plan_cache_hits,
            "retained planning beam"
        );
        beam = expanded;
    }
    let final_operation = source.len().saturating_sub(1);
    let beam = beam
        .into_iter()
        .filter_map(|mut branch| {
            if restore_unclaimed_deferred_costs(&mut branch.operations) {
                branch.analysis.take();
            }
            let peak = beam_memory_peak(
                costs,
                &branch,
                &initial,
                source,
                final_operation,
                required_outputs,
                graph,
                &constraints.allocation_copies,
            );
            (peak.fits_ipu21_with_budget(
                config.standard_memory_reservation_bytes,
                config.tile_memory_budget_bytes,
            ) || branch_contains_gemm_constraint(&branch, config))
            .then(|| {
                branch.peak_memory = peak;
                branch
            })
        })
        .collect::<Vec<_>>();
    let mut beam = beam;
    beam.sort_by_key(|branch| {
        deferred_aware_branch_score(branch, &BTreeSet::new()).saturating_add(format_equality_cost(
            branch,
            &constraints.required_equal_formats,
            costs,
        ))
    });
    if beam.is_empty() {
        return Err(LoweringError::NoCandidate(source[0].id));
    }
    Ok(beam)
}

pub(super) fn branch_contains_gemm_constraint(
    branch: &BeamBranch,
    config: &PipelineConfig,
) -> bool {
    branch.operations.iter().any(|operation| {
        operation.source.is_some_and(|source| {
            config
                .gemm_plan_constraints
                .iter()
                .any(|constraint| constraint.source_operation == source.index())
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn lower_operations(
    source: &[Operation],
    required_outputs: &[ValueId],
    values: &mut BTreeMap<ValueId, MidValueId>,
    shapes: &BTreeMap<ValueId, TensorShape>,
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
    state: &mut LoweringState,
    constraints: &RegionPlanningConstraints,
) -> LoweringResult<Vec<MidOperation>> {
    let mut candidates = lower_operation_candidates(
        source,
        required_outputs,
        values,
        shapes,
        graph,
        config,
        costs,
        state,
        constraints,
    )?;
    let best = candidates.remove(0);
    *values = best.values;
    *state = best.state;
    Ok(best.operations)
}

pub(super) fn retain_pareto_beam(
    branches: Vec<BeamBranch>,
    future_origins: &BTreeSet<ValueId>,
    constraints: &RegionPlanningConstraints,
    costs: &impl CostModel,
    width: usize,
) -> (Vec<BeamBranch>, usize, usize, usize) {
    let mut groups = BTreeMap::<FutureBeamState, Vec<RankedBeamBranch>>::new();
    for (order, branch) in branches.into_iter().enumerate() {
        let signature = future_beam_state(&branch, future_origins, constraints);
        let objective = PlanMetrics {
            cycles: deferred_aware_branch_score(&branch, future_origins).saturating_add(
                format_equality_cost(&branch, &constraints.required_equal_formats, costs),
            ),
            standard: branch.peak_memory.standard,
            interleaved: branch.peak_memory.interleaved,
            total: branch.peak_memory.total,
            maximum_standard_allocation: branch.peak_memory.maximum_standard_allocation,
            standard_contiguous_overflow: branch.peak_memory.standard_contiguous_overflow,
            exchange_rows: branch.peak_memory.exchange_rows,
        };
        groups.entry(signature).or_default().push(RankedBeamBranch {
            compatibility: future_format_compatibility(&branch, future_origins),
            branch,
            objective,
            order,
        });
    }

    let mut frontier = Vec::new();
    let mut dominated = 0usize;
    let mut equivalent = 0usize;
    for (_, candidates) in groups {
        let mut group_frontier = Vec::<RankedBeamBranch>::new();
        for candidate in candidates {
            if group_frontier
                .iter()
                .any(|kept| kept.objective == candidate.objective)
            {
                equivalent += 1;
                continue;
            }
            if group_frontier
                .iter()
                .any(|kept| kept.objective.dominates(candidate.objective))
            {
                dominated += 1;
                continue;
            }
            let before = group_frontier.len();
            group_frontier.retain(|kept| !candidate.objective.dominates(kept.objective));
            dominated += before - group_frontier.len();
            group_frontier.push(candidate);
        }
        frontier.extend(group_frontier);
    }
    frontier.sort_by_key(|candidate| (candidate.objective.cycles, candidate.order));

    let mut selected = BTreeSet::new();
    let mut diversity = 0usize;
    if frontier.len() > width {
        // Preserve the cheapest representative of every live format family
        // before retaining secondary memory tradeoffs. Partition counts are
        // intentionally excluded: they are searched within a family, whereas
        // physical order and ownership axes determine which imminent
        // consumers can use a value without a qualitatively different
        // conversion.
        let mut represented = BTreeSet::new();
        for (index, entry) in frontier.iter().enumerate() {
            if selected.len() == width {
                break;
            }
            if represented.insert(entry.compatibility.clone()) && selected.insert(index) {
                diversity += 1;
            }
        }
        let objectives: [fn(&RankedBeamBranch) -> u64; 6] = [
            |entry: &RankedBeamBranch| entry.objective.standard,
            |entry: &RankedBeamBranch| entry.objective.interleaved,
            |entry: &RankedBeamBranch| entry.objective.total,
            |entry: &RankedBeamBranch| entry.objective.maximum_standard_allocation,
            |entry: &RankedBeamBranch| entry.objective.standard_contiguous_overflow,
            |entry: &RankedBeamBranch| entry.objective.exchange_rows,
        ];
        selected.insert(0);
        for objective in objectives {
            if selected.len() == width {
                break;
            }
            let index = frontier
                .iter()
                .enumerate()
                .min_by_key(|(index, entry)| (objective(entry), entry.objective.cycles, *index))
                .map(|(index, _)| index)
                .unwrap();
            if selected.insert(index) {
                diversity += 1;
            }
        }
        for index in 0..frontier.len() {
            if selected.len() == width {
                break;
            }
            selected.insert(index);
        }
    } else {
        selected.extend(0..frontier.len());
    }
    let mut retained = frontier
        .into_iter()
        .enumerate()
        .filter_map(|(index, entry)| selected.contains(&index).then_some(entry.branch))
        .collect::<Vec<_>>();
    retained.sort_by_cached_key(|branch| {
        deferred_aware_branch_score(branch, future_origins).saturating_add(format_equality_cost(
            branch,
            &constraints.required_equal_formats,
            costs,
        ))
    });
    (retained, dominated, equivalent, diversity)
}

pub(super) fn future_beam_state(
    branch: &BeamBranch,
    future_origins: &BTreeSet<ValueId>,
    constraints: &RegionPlanningConstraints,
) -> FutureBeamState {
    let claims = deferred_claims(&branch.operations);
    let mut storage_classes = BTreeMap::<MidValueId, u32>::new();
    let mut next_storage_class = 0u32;
    let mut storage_class = |id: MidValueId| {
        let group = branch.state.get(id).storage_group;
        *storage_classes.entry(group).or_insert_with(|| {
            let class = next_storage_class;
            next_storage_class += 1;
            class
        })
    };
    let mut values = Vec::new();
    let mut deferred_sources = Vec::new();
    for &origin in future_origins {
        let Some(&id) = branch.values.get(&origin) else {
            continue;
        };
        values.push(FutureValueState {
            origin,
            tensor_type: branch.state.get(id).tensor_type.clone(),
            automatic_input: branch.state.automatic_inputs.contains(&id),
            parameter: branch.state.parameter_values.contains(&id),
            allocation_copies: constraints
                .allocation_copies
                .get(&origin)
                .copied()
                .unwrap_or(1),
            storage_class: storage_class(id),
        });
        let offer = branch
            .operations
            .iter()
            .rev()
            .find(|operation| operation.results.first() == Some(&id));
        let Some((operation, offer)) = offer.and_then(|operation| {
            operation
                .operator_plan()
                .and_then(|plan| plan.deferred_output)
                .map(|offer| (operation, offer))
        }) else {
            continue;
        };
        let Some(&source) = operation.inputs.get(offer.source_input) else {
            continue;
        };
        deferred_sources.push((origin, id, source, offer));
    }
    let deferred = deferred_sources
        .into_iter()
        .map(|(origin, result, source, offer)| FutureDeferredState {
            origin,
            source_type: branch.state.get(source).tensor_type.clone(),
            source_automatic_input: branch.state.automatic_inputs.contains(&source),
            source_parameter: branch.state.parameter_values.contains(&source),
            source_storage_class: storage_class(source),
            transform: offer.transform,
            unfused_cycles: offer.unfused_cycles,
            claimed: claims.contains(&result),
        })
        .collect();
    let equal_formats_satisfied = constraints
        .required_equal_formats
        .iter()
        .map(|&(left, right)| {
            let satisfied = branch
                .values
                .get(&left)
                .zip(branch.values.get(&right))
                .is_some_and(|(&left, &right)| {
                    branch.state.get(left).tensor_type.format
                        == branch.state.get(right).tensor_type.format
                });
            (left, right, satisfied)
        })
        .collect();
    FutureBeamState {
        values,
        deferred,
        equal_formats_satisfied,
    }
}

pub(super) fn deferred_claims(operations: &[MidOperation]) -> BTreeSet<MidValueId> {
    operations
        .iter()
        .flat_map(|operation| operation.deferred_inputs().iter().flatten())
        .map(|input| input.producer)
        .collect()
}

pub(super) fn deferred_aware_branch_score(
    branch: &BeamBranch,
    possible_future_consumers: &BTreeSet<ValueId>,
) -> u64 {
    if let Some(Some((program, _, true))) = branch.analysis.get() {
        return program.total;
    }

    let claims = deferred_claims(&branch.operations);
    branch.operations.iter().fold(0u64, |cycles, operation| {
        let pending = operation
            .operator_plan()
            .and_then(|plan| plan.deferred_output)
            .filter(|_| {
                operation.results.first().is_some_and(|result| {
                    !claims.contains(result)
                        && !possible_future_consumers.contains(&branch.state.get(*result).origin)
                })
            })
            .map_or(0, |offer| offer.unfused_cycles);
        cycles
            .saturating_add(operation.estimated_cycles)
            .saturating_add(pending)
    })
}

pub(super) fn restore_unclaimed_deferred_costs(operations: &mut [MidOperation]) -> bool {
    let mut changed = false;
    let claims = deferred_claims(operations);
    for operation in operations {
        let Some(offer) = operation
            .operator_plan()
            .and_then(|plan| plan.deferred_output)
        else {
            continue;
        };
        if operation
            .results
            .first()
            .is_some_and(|result| !claims.contains(result))
        {
            changed = true;
            operation.estimated_cycles = offer.unfused_cycles;
            operation.estimated_exchange_cycles = offer.unfused_exchange_cycles;
            if let MidOperationKind::Operator { plan, .. } = &mut operation.kind {
                plan.deferred_output = None;
            }
        }
    }
    changed
}

pub(super) fn format_equality_cost(
    branch: &BeamBranch,
    equalities: &[(ValueId, ValueId)],
    costs: &impl CostModel,
) -> u64 {
    equalities.iter().fold(0u64, |total, &(source, target)| {
        let Some((&source, &target)) = branch.values.get(&source).zip(branch.values.get(&target))
        else {
            return total;
        };
        let source = &branch.state.get(source).tensor_type;
        let target = &branch.state.get(target).tensor_type;
        let cast = if source.format.precision != target.format.precision {
            costs.cast_cycles(source, target.format.precision)
        } else {
            0
        };
        let rearrange = if source.format.layout != target.format.layout {
            costs
                .rearrangement_cost(
                    &source.shape,
                    target.format.precision,
                    layout_conversion_strategy(&source.format.layout, &target.format.layout),
                    &source.format.layout,
                    &target.format.layout,
                )
                .cycles
        } else {
            0
        };
        total.saturating_add(cast).saturating_add(rearrange)
    })
}

pub(super) fn operation_graph_inputs(operation: &Operation, graph: &ComputeGraph) -> Vec<ValueId> {
    let mut inputs = operation.inputs.clone();
    if let OperationKind::Repeat(repeat) = &operation.kind {
        for sequence in &repeat.iterated_inputs {
            inputs.extend(&graph.sequences()[sequence.index() as usize].values);
        }
    }
    inputs
}

pub(super) fn apply_selected_plan(
    operation: &Operation,
    output_shape: TensorShape,
    mut plan: OperatorPlan,
    single_use_inputs: &[bool],
    costs: &impl CostModel,
    values: &mut BTreeMap<ValueId, MidValueId>,
    state: &mut LoweringState,
    operations: &mut Vec<MidOperation>,
) {
    let input_ids = operation
        .inputs
        .iter()
        .map(|value| values[value])
        .collect::<Vec<_>>();
    let original_input_ids = input_ids.clone();
    let mut source_types = Vec::with_capacity(input_ids.len());
    let mut converted = Vec::with_capacity(input_ids.len());
    for (value, requirement) in input_ids.into_iter().zip(&plan.requirements.inputs) {
        let conversion_start = operations.len();
        let converted_value = ensure_format(
            value,
            requirement.format.clone(),
            requirement.materialization,
            operation.id,
            costs,
            state,
            operations,
        );
        let streamed_source = operations[conversion_start..]
            .last_mut()
            .and_then(|conversion| {
                let streamed = conversion.conversion_plan().is_some_and(|plan| {
                    plan.output.materialization == OperandMaterialization::DispatchSlices
                });
                if streamed {
                    conversion.estimated_cycles = 0;
                    conversion.estimated_exchange_cycles = 0;
                    conversion.inputs.first().copied()
                } else {
                    None
                }
            });
        let source_value = streamed_source.unwrap_or(converted_value);
        source_types.push(state.get(source_value).tensor_type.clone());
        converted.push(converted_value);
    }
    let result = state.value(
        operation.results[0],
        TensorType {
            shape: output_shape,
            format: plan.requirements.output.format.clone(),
        },
    );
    let converted_types = converted
        .iter()
        .map(|value| state.get(*value).tensor_type.clone())
        .collect::<Vec<_>>();
    let implementation =
        costs.implementation(&plan, &converted_types, &state.get(result).tensor_type);
    let mut operator_cycles = costs
        .operator_cycle_override(&plan, &converted_types, &state.get(result).tensor_type)
        .unwrap_or_else(|| {
            implementation
                .as_ref()
                .map_or(u64::MAX, |p| p.estimated_cycles)
        });
    let mut operator_exchange_cycles = implementation
        .as_ref()
        .map_or(0, |p| p.estimated_exchange_cycles);
    // Preliminary transition prices remain useful for custom planning models.
    // Normal detailed ranking evaluates the emitted region, including movement.
    for ((source, input), requirement) in source_types
        .iter()
        .zip(&converted_types)
        .zip(&plan.requirements.inputs)
    {
        if requirement.materialization == OperandMaterialization::DispatchSlices
            && source.format.layout != input.format.layout
        {
            let cost = costs.rearrangement_cost(
                &input.shape,
                input.format.precision,
                layout_conversion_strategy(&source.format.layout, &input.format.layout),
                &source.format.layout,
                &input.format.layout,
            );
            operator_cycles = operator_cycles.saturating_add(cost.cycles);
            operator_exchange_cycles =
                operator_exchange_cycles.saturating_add(cost.exchange_cycles);
        }
    }
    let mut deferred_inputs = vec![None; converted.len()];
    for (input_index, ((&original, &converted), requirement)) in original_input_ids
        .iter()
        .zip(&converted)
        .zip(&plan.requirements.inputs)
        .enumerate()
    {
        let conversion_is_streamed = original == converted
            || operations.iter().any(|candidate| {
                candidate.inputs.as_slice() == [original]
                    && candidate.results.as_slice() == [converted]
                    && candidate.conversion_plan().is_some_and(|conversion| {
                        conversion.output.materialization == OperandMaterialization::DispatchSlices
                    })
            });
        if !conversion_is_streamed
            || !single_use_inputs.get(input_index).copied().unwrap_or(false)
            || requirement.materialization != OperandMaterialization::DispatchSlices
        {
            continue;
        }
        let Some(producer_index) = operations
            .iter()
            .position(|candidate| candidate.results.as_slice() == [original])
        else {
            continue;
        };
        let Some(offered) = operations[producer_index]
            .operator_plan()
            .and_then(|producer| producer.deferred_output)
        else {
            continue;
        };
        let Some(&source) = operations[producer_index].inputs.get(offered.source_input) else {
            continue;
        };
        deferred_inputs[input_index] = Some(DeferredInputPlan {
            producer: original,
            source,
            transform: offered.transform,
        });
    }
    tracing::trace!(
        source = operation.id.index(),
        cycles = operator_cycles,
        dispatch = ?plan.dispatch,
        input_layouts = ?converted_types
            .iter()
            .map(|input| &input.format.layout)
            .collect::<Vec<_>>(),
        output_layout = ?state.get(result).tensor_type.format.layout,
        "costed operator plan"
    );
    if let Some(offer) = &mut plan.deferred_output {
        offer.unfused_cycles = operator_cycles;
        offer.unfused_exchange_cycles = operator_exchange_cycles;
        operator_cycles = 0;
        operator_exchange_cycles = 0;
    }
    operations.push(MidOperation {
        source: Some(operation.id),
        inputs: converted,
        results: vec![result],
        kind: MidOperationKind::Operator {
            plan,
            deferred_inputs,
            implementation,
        },
        estimated_cycles: operator_cycles,
        estimated_exchange_cycles: operator_exchange_cycles,
    });
    values.insert(operation.results[0], result);
}

pub(super) fn beam_memory_peak(
    costs: &impl CostModel,
    branch: &BeamBranch,
    initial: &[MidValueId],
    source: &[Operation],
    operation_index: usize,
    required_outputs: &[ValueId],
    graph: &ComputeGraph,
    allocation_multiplicity: &BTreeMap<ValueId, u32>,
) -> MemoryPeaks {
    let live_origins = source[operation_index + 1..]
        .iter()
        .flat_map(|operation| operation_graph_inputs(operation, graph))
        .chain(required_outputs.iter().copied())
        .collect::<BTreeSet<_>>();
    let live = live_origins
        .iter()
        .filter_map(|origin| branch.values.get(origin).copied())
        .collect::<Vec<_>>();
    let multiplicity = branch
        .state
        .values
        .iter()
        .filter_map(|value| {
            allocation_multiplicity
                .get(&value.origin)
                .map(|copies| (value.id, *copies))
        })
        .collect::<BTreeMap<_, _>>();
    branch
        .analysis
        .get_or_init(|| {
            let (program, peak) = crate::estimate::region_estimate(
                initial,
                &branch.operations,
                &live,
                &branch.state.values,
                &multiplicity,
            )?;
            let primitive = !branch.operations.iter().any(|operation| {
                operation.operator_plan().is_some_and(|plan| {
                    let inputs = operation
                        .inputs
                        .iter()
                        .map(|id| branch.state.get(*id).tensor_type.clone())
                        .collect::<Vec<_>>();
                    costs
                        .operator_cycle_override(
                            plan,
                            &inputs,
                            &branch.state.get(operation.results[0]).tensor_type,
                        )
                        .is_some()
                })
            });
            Some((
                crate::estimate::ProgramCycles {
                    total: program.estimated_cycles,
                    exchange: program.estimated_exchange_cycles,
                },
                peak,
                primitive,
            ))
        })
        .as_ref()
        .map_or_else(crate::estimate::unavailable_memory, |(_, peak, _)| *peak)
}

pub(super) fn plan_fits_operator_memory(
    plan: &OperatorPlan,
    inputs: &[TensorType],
    output: &TensorShape,
    config: &PipelineConfig,
) -> bool {
    let (planned_inputs, planned_output) = plan.tensor_types(inputs, output);
    let peak = planned_inputs
        .iter()
        .chain(std::iter::once(&planned_output))
        .map(crate::estimate::tensor_memory)
        .fold(MemoryUsage::default(), |peak, tensor| MemoryUsage {
            standard: peak.standard.max(tensor.standard),
            interleaved: peak.interleaved.max(tensor.interleaved),
        });
    peak.interleaved <= u64::from(crate::memory::IPU21_INTERLEAVED_REGION_BYTES)
        && peak
            .total()
            .saturating_add(config.standard_memory_reservation_bytes)
            <= config
                .tile_memory_budget_bytes
                .min(u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn lower_repeat(
    operation: &Operation,
    repeat: &Repeat,
    values: &mut BTreeMap<ValueId, MidValueId>,
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
    state: &mut LoweringState,
    operations: &mut Vec<MidOperation>,
) -> LoweringResult<()> {
    let inputs = operation
        .inputs
        .iter()
        .map(|value| lookup(values, *value))
        .collect::<LoweringResult<Vec<_>>>()?;
    let mut argument_types = inputs
        .iter()
        .map(|value| state.get(*value).tensor_type.clone())
        .collect::<Vec<_>>();
    let mut raw_iterated_inputs = Vec::with_capacity(repeat.iterated_inputs.len());
    let mut iterated_parameters = Vec::with_capacity(repeat.iterated_inputs.len());
    let mut iterated_automatic = Vec::with_capacity(repeat.iterated_inputs.len());
    for sequence_id in &repeat.iterated_inputs {
        let sequence = &graph.sequences()[sequence_id.index() as usize];
        let first = lookup(values, sequence.values[0])?;
        let first_type = state.get(first).tensor_type.clone();
        let sequence_values: Vec<_> = sequence
            .values
            .iter()
            .map(|value| lookup(values, *value))
            .collect::<LoweringResult<Vec<_>>>()?;
        let storage_group = state.get(first).storage_group;
        for &value in &sequence_values {
            state.values[value.index() as usize].storage_group = storage_group;
        }
        iterated_parameters.push(
            sequence_values
                .iter()
                .all(|value| state.parameter_values.contains(value)),
        );
        iterated_automatic.push(
            sequence_values
                .iter()
                .all(|value| state.automatic_inputs.contains(value)),
        );
        raw_iterated_inputs.push(sequence_values);
        argument_types.push(first_type);
    }
    let mut body_values = BTreeMap::new();
    let mut arguments = Vec::new();
    for (argument_index, (&origin, tensor_type)) in
        repeat.body.arguments.iter().zip(argument_types).enumerate()
    {
        let storage_group = if argument_index < inputs.len() {
            state.get(inputs[argument_index]).storage_group
        } else {
            state
                .get(raw_iterated_inputs[argument_index - inputs.len()][0])
                .storage_group
        };
        let value = state.value_in_storage_group(origin, tensor_type, storage_group);
        if argument_index < inputs.len() {
            if state.automatic_inputs.contains(&inputs[argument_index]) {
                state.automatic_inputs.insert(value);
            }
            if state.parameter_values.contains(&inputs[argument_index]) {
                state.parameter_values.insert(value);
            }
        } else {
            let iterated_index = argument_index - inputs.len();
            if iterated_automatic[iterated_index] {
                state.automatic_inputs.insert(value);
            }
            if iterated_parameters[iterated_index] {
                state.parameter_values.insert(value);
            }
        }
        body_values.insert(origin, value);
        arguments.push(value);
    }
    let body_allocation_copies = repeat
        .body
        .arguments
        .iter()
        .skip(inputs.len())
        .copied()
        .map(|argument| (argument, repeat.count))
        .collect::<BTreeMap<_, _>>();
    let required_equal_formats = repeat
        .body
        .yields
        .iter()
        .copied()
        .zip(repeat.body.arguments.iter().copied())
        .take(repeat.carried_inputs)
        .collect();
    let body_constraints = RegionPlanningConstraints {
        allocation_copies: body_allocation_copies,
        required_equal_formats,
    };
    let mut body_operations = lower_operations(
        &repeat.body.operations,
        &repeat.body.yields,
        &mut body_values,
        &repeat.body.value_shapes,
        graph,
        config,
        costs,
        state,
        &body_constraints,
    )?;
    for index in 0..repeat.carried_inputs {
        let body_layout = state
            .get(arguments[index])
            .tensor_type
            .format
            .layout
            .clone();
        state.retarget_automatic_input(inputs[index], body_layout);
    }
    let iterated_inputs = raw_iterated_inputs
        .into_iter()
        .enumerate()
        .map(|(index, sequence)| {
            let target = state
                .get(arguments[inputs.len() + index])
                .tensor_type
                .format
                .clone();
            sequence
                .into_iter()
                .map(|value| {
                    ensure_format(
                        value,
                        target.clone(),
                        OperandMaterialization::Complete,
                        operation.id,
                        costs,
                        state,
                        operations,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut yields = Vec::new();
    for (index, high_yield) in repeat.body.yields.iter().enumerate() {
        let value = lookup(&body_values, *high_yield)?;
        let target = state.get(inputs[index]).tensor_type.format.clone();
        yields.push(ensure_format(
            value,
            target,
            OperandMaterialization::Complete,
            operation.id,
            costs,
            state,
            &mut body_operations,
        ));
    }
    let body_cost = body_operations
        .iter()
        .map(|operation| operation.estimated_cycles)
        .sum();
    let body_exchange_cost = body_operations
        .iter()
        .map(|operation| operation.estimated_exchange_cycles)
        .sum::<u64>();
    let body_allocation_multiplicity = arguments
        .iter()
        .skip(inputs.len())
        .copied()
        .map(|argument| (argument, repeat.count))
        .collect::<BTreeMap<_, _>>();
    let body_peak = region_peak_memory_with_multiplicity(
        &arguments,
        &body_operations,
        &yields,
        &state.values,
        &body_allocation_multiplicity,
    );
    let mut results = Vec::new();
    for (origin, input) in operation.results.iter().zip(&inputs) {
        let tensor_type = state.get(*input).tensor_type.clone();
        let storage_group = state.get(*input).storage_group;
        let result = state.value_in_storage_group(*origin, tensor_type, storage_group);
        values.insert(*origin, result);
        results.push(result);
    }
    operations.push(MidOperation {
        source: Some(operation.id),
        inputs,
        results,
        kind: MidOperationKind::Repeat(MidRepeat {
            count: repeat.count,
            carried_inputs: repeat.carried_inputs,
            invariant_inputs: repeat.invariant_inputs,
            iterated_inputs,
            body: MidRegion {
                arguments,
                operations: body_operations,
                yields,
                estimated_cycles: body_cost,
                peak_memory: body_peak,
            },
        }),
        estimated_cycles: body_cost.saturating_mul(u64::from(repeat.count)),
        estimated_exchange_cycles: body_exchange_cost.saturating_mul(u64::from(repeat.count)),
    });
    Ok(())
}

pub(super) fn ensure_format(
    mut value: MidValueId,
    target: TensorFormat,
    materialization: OperandMaterialization,
    source: OperationId,
    costs: &impl CostModel,
    state: &mut LoweringState,
    operations: &mut Vec<MidOperation>,
) -> MidValueId {
    if state.retarget_automatic_input(value, target.layout.clone())
        && state.get(value).tensor_type.format.precision == target.precision
    {
        return value;
    }
    let cast_format = TensorFormat {
        precision: target.precision,
        layout: state.get(value).tensor_type.format.layout.clone(),
    };
    for format in [cast_format, target] {
        let input = state.get(value).tensor_type.clone();
        if input.format == format {
            continue;
        }
        let output = TensorType {
            shape: input.shape.clone(),
            format,
        };
        let cast = input.format.precision != output.format.precision;
        let strategy = if cast {
            ConversionStrategy::LocalKernel
        } else {
            layout_conversion_strategy(&input.format.layout, &output.format.layout)
        };
        let cost = if cast {
            crate::estimate::RearrangementCost {
                cycles: costs.cast_cycles(&input, output.format.precision),
                ..Default::default()
            }
        } else {
            costs.rearrangement_cost(
                &output.shape,
                output.format.precision,
                strategy,
                &input.format.layout,
                &output.format.layout,
            )
        };
        let result = state.derived_value(value, output.clone());
        operations.push(MidOperation {
            source: Some(source),
            inputs: vec![value],
            results: vec![result],
            kind: MidOperationKind::Convert(ConversionPlan {
                input: OperandRequirement::new(input.format, 8),
                output: OperandRequirement::new(output.format, 8).with_materialization(if cast {
                    OperandMaterialization::Complete
                } else {
                    materialization
                }),
                strategy,
            }),
            estimated_cycles: cost.cycles,
            estimated_exchange_cycles: cost.exchange_cycles,
        });
        value = result;
    }
    value
}

pub(super) fn lookup(
    values: &BTreeMap<ValueId, MidValueId>,
    value: ValueId,
) -> LoweringResult<MidValueId> {
    values
        .get(&value)
        .copied()
        .ok_or(LoweringError::UnknownValue(value))
}
