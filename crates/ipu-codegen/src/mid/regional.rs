//! Bounded production region replacements over selected, unresolved mid plans.
//! Physical feasibility and incumbent acceptance belong to package selection.
use super::*;
use std::ops::Range;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionalPlanning {
    pub baseline_attempts: usize,
    pub beam_width: usize,
    pub candidates_per_region: usize,
    pub max_evaluations: usize,
    pub passes: usize,
}
impl Default for RegionalPlanning {
    fn default() -> Self {
        Self {
            baseline_attempts: 3,
            beam_width: 8,
            candidates_per_region: 4,
            max_evaluations: 12,
            passes: 1,
        }
    }
}

pub(crate) fn baseline_config(config: &PipelineConfig, attempt: usize) -> PipelineConfig {
    let mut seed = config.clone();
    seed.regional_planning = None;
    seed.compact_layout_search = true;
    seed.shape_aware_active_tile_counts = false;
    seed.planning_beam_width = [2, 8, 16][attempt.min(2)];
    seed.expanded_plan_finalists = 1;
    seed.placement_finalists = 1;
    seed.exchange_schedule_finalists = 1;
    seed.max_parallel_reductions = 1;
    seed.exchange_table_cost_per_byte = [16, 256, 4096][attempt.min(2)];
    seed
}

pub(crate) fn baseline(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
) -> LoweringResult<MidProgram> {
    let mut seed = config.clone();
    for input in graph
        .inputs()
        .iter()
        .filter(|input| input.kind == GraphInputKind::Host)
    {
        if let Some(&precision) = seed.automatic_inputs.get(&input.value)
            && let Some(layout) = balanced_row_major(&input.shape, precision, seed.tile_count, true)
        {
            seed.inputs
                .insert(input.value, TensorFormat { precision, layout });
            seed.automatic_inputs.remove(&input.value);
        }
    }
    planner::plan_finalists(graph, &seed, costs, 1)?
        .into_iter()
        .next()
        .ok_or(LoweringError::InvalidImplementation)
}

pub(crate) fn regions(graph: &ComputeGraph) -> Vec<Range<usize>> {
    if graph.planning_regions().is_empty() {
        (0..graph.operations().len()).map(|i| i..i + 1).collect()
    } else {
        graph.planning_regions().to_vec()
    }
}

fn references(operation: &MidOperation) -> impl Iterator<Item = MidValueId> + '_ {
    operation.read_values().copied().chain(
        operation
            .deferred_inputs()
            .iter()
            .flatten()
            .flat_map(|d| [d.source, d.producer]),
    )
}

fn remap(operation: &mut MidOperation, replacements: &BTreeMap<MidValueId, MidValueId>) {
    let map = |id: &mut MidValueId| {
        if let Some(&new) = replacements.get(id) {
            *id = new;
        }
    };
    operation
        .inputs
        .iter_mut()
        .chain(&mut operation.results)
        .for_each(map);
    match &mut operation.kind {
        MidOperationKind::Operator {
            deferred_inputs, ..
        } => {
            for deferred in deferred_inputs.iter_mut().flatten() {
                map(&mut deferred.source);
                map(&mut deferred.producer);
            }
        }
        MidOperationKind::Repeat(repeat) => {
            repeat
                .iterated_inputs
                .iter_mut()
                .flatten()
                .chain(&mut repeat.body.arguments)
                .chain(&mut repeat.body.yields)
                .for_each(map);
            for operation in &mut repeat.body.operations {
                remap(operation, replacements);
            }
        }
        _ => {}
    }
}

pub(crate) struct Replacement {
    pub program: MidProgram,
    pub local: Option<MidProgram>,
}

/// Boundaries are actual live mid values, including their format and ownership.
/// Multiple representations of the same origin and escaping deferred producers
/// need a richer contract; leave those incumbents untouched for now.
pub(crate) fn replacements(
    graph: &ComputeGraph,
    incumbent: &MidProgram,
    range: Range<usize>,
    config: &PipelineConfig,
    options: &RegionalPlanning,
    costs: &impl CostModel,
) -> LoweringResult<Vec<Replacement>> {
    planner::in_planning_pool(|| {
        replacements_in_pool(graph, incumbent, range, config, options, costs)
    })
}

fn replacements_in_pool(
    graph: &ComputeGraph,
    incumbent: &MidProgram,
    range: Range<usize>,
    config: &PipelineConfig,
    options: &RegionalPlanning,
    costs: &impl CostModel,
) -> LoweringResult<Vec<Replacement>> {
    let source = graph
        .operations()
        .get(range)
        .ok_or(LoweringError::InvalidImplementation)?;
    if source.is_empty() {
        return Ok(Vec::new());
    }
    let origins = source.iter().map(|op| op.id).collect::<BTreeSet<_>>();
    let belongs = |op: &MidOperation| op.source.is_some_and(|id| origins.contains(&id));
    let Some(start) = incumbent.operations.iter().position(belongs) else {
        return Ok(Vec::new());
    };
    let end = incumbent.operations.iter().rposition(belongs).unwrap() + 1;
    if !incumbent.operations[start..end].iter().all(belongs) {
        return Ok(Vec::new());
    }
    let removed = &incumbent.operations[start..end];
    let defined = removed
        .iter()
        .flat_map(|op| op.results.iter().copied())
        .collect::<BTreeSet<_>>();
    if incumbent.operations[end..]
        .iter()
        .flat_map(MidOperation::deferred_inputs)
        .flatten()
        .any(|d| defined.contains(&d.producer))
    {
        return Ok(Vec::new());
    }
    let live_in = removed
        .iter()
        .flat_map(references)
        .filter(|id| !defined.contains(id))
        .collect::<BTreeSet<_>>();
    let live_out = incumbent.operations[end..]
        .iter()
        .flat_map(references)
        .chain(incumbent.outputs.iter().copied())
        .filter(|id| defined.contains(id))
        .collect::<BTreeSet<_>>();
    let high_results = source
        .iter()
        .flat_map(|op| op.results.iter().copied())
        .collect::<BTreeSet<_>>();
    if live_out
        .iter()
        .any(|id| !high_results.contains(&incumbent.values[id.index() as usize].origin))
    {
        return Ok(Vec::new());
    }
    let mut values = BTreeMap::new();
    for &id in &live_in {
        let value = &incumbent.values[id.index() as usize];
        if values.insert(value.origin, id).is_some() {
            return Ok(Vec::new());
        }
    }
    let outputs = live_out
        .iter()
        .map(|id| incumbent.values[id.index() as usize].origin)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let parameters = graph
        .inputs()
        .iter()
        .filter(|i| i.kind == GraphInputKind::Parameter)
        .map(|i| i.value)
        .collect::<BTreeSet<_>>();
    let mut state = LoweringState {
        values: incumbent.values.clone(),
        automatic_inputs: BTreeSet::new(),
        parameter_values: incumbent
            .values
            .iter()
            .filter(|v| parameters.contains(&v.origin))
            .map(|v| v.id)
            .collect(),
    };
    let mut search = config.clone();
    search.regional_planning = None;
    search.planning_beam_width = options.beam_width.max(1);
    let branches = lower_operation_candidates(
        source,
        &outputs,
        &mut values,
        graph.value_shapes(),
        graph,
        &search,
        costs,
        &mut state,
        &RegionPlanningConstraints::default(),
    )?;
    let mut result = Vec::new();
    for mut branch in branches.into_iter().take(options.candidates_per_region) {
        // No region replacement may change the storage contract of an existing value.
        if branch.state.values[..incumbent.values.len()] != incumbent.values {
            continue;
        }
        let mut bindings = BTreeMap::new();
        for &old in &live_out {
            let boundary = &incumbent.values[old.index() as usize];
            let Some(&new) = branch.values.get(&boundary.origin) else {
                return Err(LoweringError::InvalidImplementation);
            };
            let new = ensure_format(
                new,
                boundary.tensor_type.format.clone(),
                OperandMaterialization::Complete,
                false,
                source.last().unwrap().id,
                costs,
                &mut branch.state,
                &mut branch.operations,
            );
            if branch.state.get(new).tensor_type != boundary.tensor_type
                || branch.state.get(new).tile_offset != boundary.tile_offset
            {
                return Err(LoweringError::InvalidImplementation);
            }
            bindings.insert(old, new);
        }
        let local = live_in
            .iter()
            .all(|id| incumbent.values[id.index() as usize].storage_group == *id)
            .then(|| MidProgram {
                tile_count: incumbent.tile_count,
                inputs: live_in
                    .iter()
                    .map(|&id| MidInput {
                        name: format!("boundary-{}", id.index()),
                        kind: if parameters.contains(&incumbent.values[id.index() as usize].origin)
                        {
                            GraphInputKind::Parameter
                        } else {
                            GraphInputKind::Host
                        },
                        value: id,
                    })
                    .collect(),
                values: branch.state.values.clone(),
                operations: branch.operations.clone(),
                outputs: bindings.values().copied().collect(),
                ..Default::default()
            });
        let mut candidate = incumbent.clone();
        candidate.values = branch.state.values;
        candidate.operations.splice(start..end, branch.operations);
        let suffix = candidate.operations.len() - (incumbent.operations.len() - end);
        for op in &mut candidate.operations[suffix..] {
            remap(op, &bindings);
        }
        for value in &mut candidate.values {
            if let Some(&new) = bindings.get(&value.storage_group) {
                value.storage_group = new;
            }
        }
        for output in &mut candidate.outputs {
            if let Some(&new) = bindings.get(output) {
                *output = new;
            }
        }
        restore_unclaimed_deferred_costs(&mut candidate.operations);
        candidate.estimated_cycles = candidate
            .operations
            .iter()
            .map(|op| op.estimated_cycles)
            .sum();
        candidate.estimated_exchange_cycles = candidate
            .operations
            .iter()
            .map(|op| op.estimated_exchange_cycles)
            .sum();
        candidate.peak_memory = region_peak_memory(
            &candidate.inputs.iter().map(|i| i.value).collect::<Vec<_>>(),
            &candidate.operations,
            &candidate.outputs,
            &candidate.values,
        );
        result.push(Replacement {
            program: candidate,
            local,
        });
    }
    Ok(result)
}

/// A deliberately loose throughput bound, independent of exchange heuristics.
/// 1024 floating operations per tile-cycle exceeds the supported kernels' peak.
/// Ignore non-GEMM work rather than treating heuristic prices as hard bounds.
pub(crate) fn compute_lower_bound(program: &MidProgram) -> u64 {
    fn work(ops: &[MidOperation], values: &[MidValue]) -> (u64, u64) {
        let (mut total, mut bottleneck) = (0u64, 0u64);
        for op in ops {
            if let MidOperationKind::Repeat(r) = &op.kind {
                let (body, local) = work(&r.body.operations, values);
                total = total.saturating_add(body.saturating_mul(u64::from(r.count)));
                bottleneck = bottleneck.max(local.saturating_mul(u64::from(r.count)));
                continue;
            }
            let Some(plan) = op.operator_plan() else {
                continue;
            };
            let MidOperator::Gemm { options, .. } = plan.operator else {
                continue;
            };
            let Some(input) = op
                .inputs
                .first()
                .map(|id| &values[id.index() as usize].tensor_type.shape.0)
            else {
                continue;
            };
            let Some(output) = op
                .results
                .first()
                .map(|id| &values[id.index() as usize].tensor_type.shape.0)
            else {
                continue;
            };
            let Some(axis) = input
                .len()
                .checked_sub(if options.transpose_left { 2 } else { 1 })
            else {
                continue;
            };
            let flops = output
                .iter()
                .fold(2u64, |p, &n| p.saturating_mul(u64::from(n)))
                .saturating_mul(u64::from(input[axis]));
            let active = match plan.dispatch {
                OperatorDispatch::BlockedGemm {
                    distribution:
                        GemmDistribution::ParallelReduction {
                            row_partitions,
                            column_partitions,
                            inner_partitions,
                            ..
                        },
                    ..
                } => {
                    u64::from(row_partitions)
                        * u64::from(column_partitions)
                        * u64::from(inner_partitions)
                }
                _ => u64::from(plan.requirements.output.format.layout.tiling.tile_count),
            };
            total = total.saturating_add(flops);
            bottleneck = bottleneck.max(flops / (active.max(1) * 1024));
        }
        (total, bottleneck)
    }
    let (flops, bottleneck) = work(&program.operations, &program.values);
    (flops / (u64::from(program.tile_count).max(1) * 1024)).max(bottleneck)
}

pub(crate) fn resolve(program: &MidProgram, checkpoints: bool) -> LoweringResult<MidProgram> {
    let resolved =
        implementation::resolve(program.clone()).ok_or(LoweringError::InvalidImplementation)?;
    Ok(if checkpoints {
        resolved
    } else {
        resolved.with_elementwise_fusions().unwrap_or(resolved)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn regional_repeat_preserves_sequences_and_is_deterministic() {
        let mut graph = ComputeGraph::new();
        let x = graph.host_input("x", [16, 16]).unwrap();
        let weights = (0..3)
            .map(|i| graph.parameter(format!("w{i}"), [16, 16]).unwrap())
            .collect::<Vec<_>>();
        let sequence = graph.value_sequence("weights", weights.clone()).unwrap();
        let y = graph
            .repeat(3, [x], [], [sequence], |body, args| {
                Ok(vec![body.gemm(args.carried[0], args.iterated[0])?])
            })
            .unwrap()[0];
        graph.set_outputs([y]).unwrap();
        let mut config = PipelineConfig::new(8).with_automatic_input(x, Precision::F16);
        for weight in weights {
            config = config.with_automatic_input(weight, Precision::F16);
        }
        let seed_config = baseline_config(&config, 0);
        let seed = baseline(&graph, &seed_config, &Ipu21CostModel).unwrap();
        assert_eq!(
            seed,
            baseline(&graph, &seed_config, &Ipu21CostModel).unwrap()
        );
        let alternatives = replacements(
            &graph,
            &seed,
            0..1,
            &config,
            &RegionalPlanning::default(),
            &Ipu21CostModel,
        )
        .unwrap();
        assert!(!alternatives.is_empty());
        for alternative in alternatives {
            let resolved = resolve(&alternative.program, true).unwrap();
            let low = crate::low::expand::expand_tiles(&resolved, true).unwrap();
            assert!(
                matches!(&alternative.program.operations[0].kind, MidOperationKind::Repeat(r) if r.count == 3)
            );
            assert!(compute_lower_bound(&alternative.program) <= low.estimated_cycles);
        }
    }

    #[test]
    fn replacements_preserve_boundary_types_and_expand_with_residual_consumers() {
        let mut graph = ComputeGraph::new();
        let x = graph.host_input("x", [32, 32]).unwrap();
        let w = graph.parameter("w", [32, 32]).unwrap();
        let y = graph.gemm(x, w).unwrap();
        let z = graph.gelu(y).unwrap();
        let out = graph.add(z, y).unwrap();
        graph.set_outputs([out]).unwrap();
        graph.add_planning_region(0..2).unwrap();
        assert!(graph.add_planning_region(1..3).is_err());
        let config = PipelineConfig::new(8)
            .with_automatic_input(x, Precision::F16)
            .with_automatic_input(w, Precision::F16);
        let costs = Ipu21CostModel;
        let seed = baseline(&graph, &baseline_config(&config, 0), &costs).unwrap();
        let proposals = replacements(
            &graph,
            &seed,
            0..2,
            &config,
            &RegionalPlanning::default(),
            &costs,
        )
        .unwrap();
        assert!(!proposals.is_empty());
        for proposal in proposals {
            let proposal = proposal.program;
            for (&a, &b) in seed.outputs.iter().zip(&proposal.outputs) {
                assert_eq!(
                    seed.values[a.index() as usize].tensor_type,
                    proposal.values[b.index() as usize].tensor_type
                );
            }
            assert_eq!(
                proposal.operations.last().unwrap().operator_plan(),
                seed.operations.last().unwrap().operator_plan()
            );
            let resolved = resolve(&proposal, true).unwrap();
            let low = crate::low::expand::expand_tiles(&resolved, true).unwrap();
            assert!(!low.kernel_runs.is_empty());
            assert!(compute_lower_bound(&proposal) <= low.estimated_cycles);
        }
    }
}
