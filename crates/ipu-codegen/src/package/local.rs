//! One validated incumbent; bounded proposals replace it only after packaging.
use super::*;
use crate::mid::baseline::{self, Baseline, Recipe};
use std::sync::Arc;

pub(super) fn optimize<T>(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    tile_mapping: Option<&[u16]>,
    mut finalize: impl FnMut(&mut ScheduledPlan) -> PackageBuildResult<(u64, T)>,
) -> PackageBuildResult<(ScheduledPlan, T)> {
    let costs = crate::estimate::MemoizedCostModel::new(&Ipu21CostModel, config.tile_count);
    let expansions = Arc::new(crate::low::expand::ExpansionCache::default());
    let mut schedules = crate::ExchangeScheduleCache::default();
    let mut incumbent = baseline::lower(graph, config, &costs, &Recipe::default())?;
    let (mut selected, mut cycles, mut artifact) = validate(
        &incumbent.program,
        config,
        tile_mapping,
        Arc::clone(&expansions),
        &mut schedules,
        &mut finalize,
    )?;
    tracing::info!(cycles, "validated canonical baseline");
    // Preserve logical homes, not their physical addresses. All proposals use
    // joint placement after support reservations; scratch has no separate arena.
    let mut fixed = config.clone();
    for (input, planned) in graph.inputs().iter().zip(&incumbent.program.inputs) {
        fixed.inputs.insert(
            input.value,
            incumbent.program.values[planned.value.index() as usize]
                .tensor_type
                .format
                .clone(),
        );
    }
    let mut pending = Vec::<Baseline>::new();
    let mut visited = Vec::new();
    let mut regenerate = true;
    for attempt in 0..config.optimization_steps {
        if regenerate {
            pending.clear();
            for recipe in proposals(graph, &incumbent) {
                if visited.contains(&recipe) {
                    continue;
                }
                let candidate = match baseline::lower(graph, &fixed, &costs, &recipe) {
                    Ok(candidate) => candidate,
                    Err(error) => {
                        tracing::debug!(%error, "discarded invalid local recipe");
                        continue;
                    }
                };
                if candidate.program.estimated_cycles >= incumbent.program.estimated_cycles {
                    continue;
                }
                pending.push(candidate);
                pending.sort_by_key(|candidate| candidate.program.estimated_cycles);
                pending.truncate(config.optimization_steps - attempt);
            }
            regenerate = false;
        }
        if pending.is_empty() {
            break;
        }
        let candidate = pending.remove(0);
        visited.push(candidate.recipe.clone());
        match validate(
            &candidate.program,
            &fixed,
            tile_mapping,
            Arc::clone(&expansions),
            &mut schedules,
            &mut finalize,
        ) {
            Ok((plan, candidate_cycles, built)) if candidate_cycles < cycles => {
                tracing::info!(
                    attempt,
                    before = cycles,
                    after = candidate_cycles,
                    "accepted local layout improvement"
                );
                let mut candidate = candidate;
                for (id, alternatives) in incumbent.alternatives {
                    candidate.alternatives.entry(id).or_insert(alternatives);
                }
                incumbent = candidate;
                selected = plan;
                cycles = candidate_cycles;
                artifact = built;
                regenerate = true;
            }
            Ok((_, candidate_cycles, _)) => tracing::info!(
                attempt,
                candidate_cycles,
                cycles,
                "retained faster incumbent"
            ),
            Err(error) => tracing::info!(attempt, %error, "retained feasible incumbent"),
        }
    }
    Ok((selected, artifact))
}

fn validate<T>(
    mid: &crate::MidProgram,
    config: &PipelineConfig,
    mapping: Option<&[u16]>,
    expansions: Arc<crate::low::expand::ExpansionCache>,
    cache: &mut crate::ExchangeScheduleCache,
    finalize: &mut impl FnMut(&mut ScheduledPlan) -> PackageBuildResult<(u64, T)>,
) -> PackageBuildResult<(ScheduledPlan, u64, T)> {
    let (program, _) = selection::expand_and_screen(mid, config, mapping, expansions)?;
    let placement = place(&program)?;
    let topology = active_topology(config.tile_count)?;
    let exchanges =
        crate::exchange::lower_exchanges_cached(&program, &placement, &topology, false, cache)?;
    let mut plan = ScheduledPlan {
        program,
        placement,
        phases: exchanges.phases,
        cache: cache.clone(),
    };
    let result = finalize(&mut plan);
    *cache = plan.cache.clone();
    let (cycles, artifact) = result?;
    Ok((plan, cycles, artifact))
}

fn proposals(graph: &ComputeGraph, incumbent: &Baseline) -> Vec<Recipe> {
    fn visit<'a>(source: &'a [crate::Operation], operations: &mut Vec<&'a crate::Operation>) {
        for op in source {
            if let crate::OperationKind::Repeat(repeat) = &op.kind {
                visit(&repeat.body.operations, operations);
            } else {
                operations.push(op);
            }
        }
    }
    let mut operations = Vec::new();
    visit(graph.operations(), &mut operations);
    let mut candidates = Vec::new();
    for operation in &operations {
        if let Some(alternatives) = incumbent.alternatives.get(&operation.id) {
            for plan in alternatives
                .iter()
                .filter(|plan| incumbent.recipe.plans.get(&operation.id) != Some(plan))
                .take(4)
            {
                let mut recipe = incumbent.recipe.clone();
                recipe.plans.insert(operation.id, plan.clone());
                candidates.push(recipe);
            }
        }
        let mut recipe = incumbent.recipe.clone();
        if !recipe.late_casts.insert(operation.id) {
            recipe.late_casts.remove(&operation.id);
        }
        candidates.push(recipe);
        for &output in &operation.results {
            if graph.outputs().contains(&output)
                || incumbent.recipe.open_boundaries.contains(&output)
            {
                continue;
            }
            let consumers = operations
                .iter()
                .filter(|op| op.inputs.contains(&output))
                .collect::<Vec<_>>();
            if consumers.is_empty() {
                continue;
            }
            let mut recipe = incumbent.recipe.clone();
            recipe.open_boundaries.insert(output);
            // Re-enumerate consumers against the producer's new format. The
            // external boundary after each consumer remains canonical.
            for consumer in consumers {
                recipe.plans.remove(&consumer.id);
            }
            candidates.push(recipe);
        }
    }
    candidates
}
