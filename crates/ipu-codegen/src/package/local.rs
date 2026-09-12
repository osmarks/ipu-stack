//! One validated incumbent; bounded proposals replace it only after packaging.
mod checkpoint;

use super::*;
use crate::mid::baseline::{self, Baseline, Recipe};
use std::sync::Arc;

pub(super) fn optimize<T: Send>(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    tile_mapping: Option<&[u16]>,
    finalize: impl Fn(&mut ScheduledPlan) -> PackageBuildResult<(u64, T)> + Sync,
) -> PackageBuildResult<(ScheduledPlan, T)> {
    let costs = crate::estimate::MemoizedCostModel::new(&Ipu21CostModel);
    let expansions = Arc::new(crate::low::expand::ExpansionCache::default());
    let mut schedules =
        crate::ExchangeScheduleCache::with_stream_words(config.exchange_stream_words);
    let mut state = checkpoint::State::load(graph, config, tile_mapping)?;
    let mut fixed = config.clone();
    let resuming = config.load_search_state.is_some();
    if resuming {
        fixed.inputs = state.inputs.clone();
    }
    let mut incumbent = baseline::lower(graph, &fixed, &costs, &state.recipe)?;
    if resuming {
        incumbent.alternatives = state.alternatives.clone();
    }
    memory_profile(graph, config, &incumbent.program, "baseline")?;
    let (mut selected, mut cycles, mut artifact) = validate(
        &incumbent.program,
        config,
        state.mapping.as_deref(),
        Arc::clone(&expansions),
        &mut schedules,
        &finalize,
    )?;
    tracing::info!(cycles, resuming, "validated search incumbent");
    let budget_end = state
        .attempts
        .checked_add(config.optimization_steps)
        .ok_or_else(|| invalid("search step budget overflow"))?;
    // Fix logical homes once, including on a baseline-only checkpoint.
    for (input, planned) in graph.inputs().iter().zip(&incumbent.program.inputs) {
        fixed.inputs.insert(
            input.value,
            incumbent.program.values[planned.value.index() as usize]
                .tensor_type
                .format
                .clone(),
        );
    }
    state.save(config, &incumbent, &fixed)?;
    if config.optimization_steps != 0 && !state.mapping_checked && state.mapping.is_none() {
        state.mapping_checked = true;
        let challenger =
            match placement::model_mapping(&selected.program, &selected.placement, true) {
                Ok((_, challenger)) => challenger,
                Err(error) => {
                    tracing::info!(%error, "retained incumbent after mapping estimate failed");
                    None
                }
            };
        if let Some(challenger) = challenger {
            state.attempts += 1;
            match validate(
                &incumbent.program,
                config,
                Some(&challenger),
                Arc::clone(&expansions),
                &mut schedules,
                &finalize,
            ) {
                Ok((plan, improved, built)) if improved < cycles => {
                    tracing::info!(
                        before = cycles,
                        after = improved,
                        "accepted tile mapping improvement"
                    );
                    selected = plan;
                    cycles = improved;
                    artifact = built;
                    state.mapping = Some(challenger);
                }
                Ok(_) => tracing::info!("retained incumbent tile mapping"),
                Err(error) => tracing::info!(%error, "retained feasible tile mapping"),
            }
        }
    }

    state.save(config, &incumbent, &fixed)?;
    while state.attempts < budget_end {
        // Indexed collection preserves proposal order for equal-cost ties.
        let mut pending = proposals(graph, config, &incumbent)
            .par_iter()
            .filter(|recipe| !state.visited.contains(recipe))
            .filter_map(
                |recipe| match baseline::lower(graph, &fixed, &costs, recipe) {
                    Ok(candidate)
                        if !state.visited.contains(&candidate.recipe)
                            && candidate.program.estimated_cycles
                                < incumbent.program.estimated_cycles =>
                    {
                        Some(candidate)
                    }
                    Ok(_) => None,
                    Err(error) => {
                        tracing::debug!(%error, "discarded invalid local recipe");
                        None
                    }
                },
            )
            .collect::<Vec<_>>();
        pending.sort_by_key(|candidate| candidate.program.estimated_cycles);
        pending.dedup_by(|a, b| a.program == b.program);
        pending.truncate(budget_end - state.attempts);
        if pending.is_empty() {
            break;
        }
        for (index, candidate) in pending.iter().enumerate() {
            memory_profile(
                graph,
                &fixed,
                &candidate.program,
                &format!("local-{}-candidate-{index}", state.attempts),
            )?;
        }
        tracing::info!(
            candidates = pending.len(),
            threads = rayon::current_num_threads(),
            "evaluating ordered local candidates concurrently"
        );
        // One Rayon pool serves both candidate builds and their internal work.
        // find_first cancels unstarted later work once the earliest improvement
        // is known. Each speculative build owns its schedule-cache snapshot;
        // only the selected cache becomes the next incumbent's cache.
        let winner = pending
            .par_iter()
            .enumerate()
            .map(|(index, candidate)| {
                let attempt = state.attempts + index;
                let span = tracing::info_span!("local_candidate", attempt);
                let _entered = span.enter();
                let result = validate(
                    &candidate.program,
                    &fixed,
                    state.mapping.as_deref(),
                    Arc::clone(&expansions),
                    &mut schedules.clone(),
                    &finalize,
                );
                match result {
                    Ok((plan, candidate_cycles, built)) if candidate_cycles < cycles => {
                        Some((index, plan, candidate_cycles, built))
                    }
                    Ok((_, candidate_cycles, _)) => {
                        tracing::info!(candidate_cycles, cycles, "retained faster incumbent");
                        None
                    }
                    Err(error) => {
                        tracing::info!(%error, "retained feasible incumbent");
                        None
                    }
                }
            })
            .find_first(Option::is_some)
            .flatten();
        let Some((index, plan, candidate_cycles, built)) = winner else {
            state.attempts += pending.len();
            state
                .visited
                .extend(pending.into_iter().map(|candidate| candidate.recipe));
            state.save(config, &incumbent, &fixed)?;
            // A truncated shortlist may have further unevaluated proposals.
            continue;
        };
        state.visited.extend(
            pending[..=index]
                .iter()
                .map(|candidate| candidate.recipe.clone()),
        );
        let mut candidate = pending.swap_remove(index);
        tracing::info!(
            attempt = state.attempts + index,
            before = cycles,
            after = candidate_cycles,
            changed_operations = ?candidate.recipe.plans.iter().filter_map(|(id, plan)|
                (incumbent.recipe.plans.get(id) != Some(plan)).then_some(id.index())).collect::<Vec<_>>(),
            opened_boundaries = ?candidate.recipe.open_boundaries.difference(&incumbent.recipe.open_boundaries)
                .map(|id| id.index()).collect::<Vec<_>>(),
            "accepted local layout improvement"
        );
        state.attempts += index + 1;
        for (id, alternatives) in incumbent.alternatives {
            candidate.alternatives.entry(id).or_insert(alternatives);
        }
        incumbent = candidate;
        schedules = plan.cache.clone();
        selected = plan;
        cycles = candidate_cycles;
        artifact = built;
        state.save(config, &incumbent, &fixed)?;
    }
    Ok((selected, artifact))
}

fn memory_profile(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    mid: &crate::MidProgram,
    scope: &str,
) -> PackageBuildResult<()> {
    crate::estimate::memory_profile::write(
        scope,
        graph,
        config,
        &mid.inputs
            .iter()
            .map(|input| input.value)
            .collect::<Vec<_>>(),
        &mid.operations,
        &mid.outputs,
        &mid.values,
        &Default::default(),
    )?;
    Ok(())
}

fn validate<T>(
    mid: &crate::MidProgram,
    config: &PipelineConfig,
    mapping: Option<&[u16]>,
    expansions: Arc<crate::low::expand::ExpansionCache>,
    cache: &mut crate::ExchangeScheduleCache,
    finalize: &(impl Fn(&mut ScheduledPlan) -> PackageBuildResult<(u64, T)> + Sync),
) -> PackageBuildResult<(ScheduledPlan, u64, T)> {
    let (program, _) = validation::expand_and_screen(mid, config, mapping, expansions)?;
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

fn proposals(graph: &ComputeGraph, config: &PipelineConfig, incumbent: &Baseline) -> Vec<Recipe> {
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
    // Include the incumbent so its opposite cast-storage policy is considered
    // alongside the paired layout changes below. Visited recipes are filtered.
    let mut candidates = vec![incumbent.recipe.clone()];
    for rows in [32, 64, 128, 256] {
        if incumbent.recipe.packing_rows != Some(rows) {
            let mut recipe = incumbent.recipe.clone();
            recipe.packing_rows = Some(rows);
            candidates.push(recipe);
        }
    }
    for limit in 2..=config.max_parallel_reductions {
        if incumbent.recipe.parallel_reductions != limit {
            let mut recipe = incumbent.recipe.clone();
            recipe.parallel_reductions = limit;
            candidates.push(recipe);
        }
    }
    if !incumbent.recipe.disjoint_copy_sources {
        let mut recipe = incumbent.recipe.clone();
        recipe.disjoint_copy_sources = true;
        candidates.push(recipe);
    }

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
        if !recipe.early_casts.insert(operation.id) {
            recipe.early_casts.remove(&operation.id);
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
            // A store format and its consumer can need to change together;
            // neither isolated change need improve the canonical baseline.
            if let Some(alternatives) = incumbent.alternatives.get(&operation.id) {
                for plan in alternatives
                    .iter()
                    .filter(|plan| incumbent.recipe.plans.get(&operation.id) != Some(plan))
                    .take(4)
                {
                    let mut joint = recipe.clone();
                    joint.plans.insert(operation.id, plan.clone());
                    candidates.push(joint);
                }
            }
            candidates.push(recipe);
        }
    }
    // Storage donation can enable a faster layout even though the cast itself
    // is slower. Evaluate it with each layout proposal, not only as an isolated
    // change that the cycle-improvement screen would always discard.
    candidates
        .into_iter()
        .flat_map(|recipe| {
            let mut alternate = recipe.clone();
            alternate.in_place_casts =
                Some(!recipe.in_place_casts.unwrap_or(config.capacity_baseline));
            [recipe, alternate]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mlp() -> (ComputeGraph, PipelineConfig) {
        let mut graph = ComputeGraph::new();
        let x = graph.host_input("x", [32, 64]).unwrap();
        let up = graph.parameter("up", [64, 128]).unwrap();
        let down = graph.parameter("down", [128, 64]).unwrap();
        let hidden = graph.gemm(x, up).unwrap();
        let hidden = graph.gelu(hidden).unwrap();
        let y = graph.gemm(hidden, down).unwrap();
        graph.set_outputs([y]).unwrap();
        let config = PipelineConfig::new(8)
            .with_automatic_input(x, Precision::F16)
            .with_automatic_input(up, Precision::F16)
            .with_automatic_input(down, Precision::F16);
        (graph, config)
    }

    #[test]
    fn memoization_does_not_change_baseline_selection() {
        let (graph, mut config) = mlp();
        let costs = crate::estimate::MemoizedCostModel::new(&Ipu21CostModel);
        for capacity in [false, true] {
            config.capacity_baseline = capacity;
            let recipe = Recipe::default();
            let direct = baseline::lower(&graph, &config, &Ipu21CostModel, &recipe).unwrap();
            let memoized = baseline::lower(&graph, &config, &costs, &recipe).unwrap();
            assert_eq!(direct.program, memoized.program);
            assert!(direct.recipe == memoized.recipe);
        }
    }

    #[test]
    fn saved_search_resumes_the_same_ordered_path() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                let (graph, mut config) = mlp();
                let path =
                    std::env::temp_dir().join(format!("ipu-search-{}.json", fastrand::u64(..)));
                let cost = |plan: &mut ScheduledPlan| Ok((plan.program.estimated_cycles, ()));
                config.optimization_steps = 4;
                let uninterrupted = optimize(&graph, &config, None, cost).unwrap().0;
                config.optimization_steps = 2;
                config.save_search_state = Some(path.clone());
                optimize(&graph, &config, None, cost).unwrap();
                let before = checkpoint::State::load(
                    &graph,
                    &PipelineConfig {
                        load_search_state: Some(path.clone()),
                        ..config.clone()
                    },
                    None,
                )
                .unwrap();
                assert!(before.attempts > 0);
                config.load_search_state = Some(path.clone());
                let resumed = optimize(&graph, &config, None, cost).unwrap().0;
                assert_eq!(resumed.program, uninterrupted.program);
                assert_eq!(resumed.placement, uninterrupted.placement);
                assert_eq!(resumed.phases, uninterrupted.phases);
                // A zero-step load still rebuilds and validates the saved winner.
                config.optimization_steps = 0;
                let rebuilt = optimize(&graph, &config, None, cost).unwrap().0;
                assert_eq!(rebuilt.program, resumed.program);
                config.exchange_table_budget_bytes += 4;
                assert!(
                    optimize::<()>(&graph, &config, None, |_| panic!(
                        "incompatible state reached validation"
                    ))
                    .is_err()
                );
                std::fs::remove_file(path).unwrap();
            });
    }

    #[test]
    fn rejected_recipes_survive_checkpoint_resume() {
        let (graph, mut config) = mlp();
        let path = std::env::temp_dir().join(format!("ipu-search-{}.json", fastrand::u64(..)));
        config.optimization_steps = 1;
        config.save_search_state = Some(path.clone());
        for _ in 0..2 {
            let first = std::sync::atomic::AtomicBool::new(true);
            optimize(&graph, &config, None, |_| {
                if first.swap(false, std::sync::atomic::Ordering::Relaxed) {
                    Ok((1, ()))
                } else {
                    Err(invalid("test candidate fails placement"))
                }
            })
            .unwrap();
            config.load_search_state = Some(path.clone());
        }
        let state = checkpoint::State::load(&graph, &config, None).unwrap();
        assert_eq!(state.attempts, 2);
        assert_eq!(state.visited.len(), 2);
        assert!(state.visited[0] != state.visited[1]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn complete_validation_is_bounded_and_keeps_the_best_feasible_package() {
        let (graph, mut config) = mlp();
        for budget in [0, 1, 3] {
            config.optimization_steps = budget;
            for behavior in ["reject", "slower", "faster"] {
                let calls = std::sync::atomic::AtomicUsize::new(0);
                let (_, selected) = optimize(&graph, &config, None, |_| {
                    let index = calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if index != 0 && behavior == "reject" {
                        return Err(invalid("test package does not fit support reservations"));
                    }
                    let cycles = if behavior == "faster" {
                        if index == 0 { 100 } else { 90 }
                    } else {
                        100 + index
                    };
                    Ok((cycles as u64, cycles))
                })
                .unwrap();
                let calls = calls.load(std::sync::atomic::Ordering::Relaxed);
                assert!(
                    calls <= 1 + budget * (budget + 1) / 2,
                    "{behavior}: {calls} validations for budget {budget}"
                );
                if budget > 0 {
                    assert!(calls > 1, "test must exercise local proposals");
                }
                assert_eq!(
                    selected,
                    if behavior == "faster" && calls > 1 {
                        90
                    } else {
                        100
                    }
                );
            }
        }
    }

    #[test]
    fn parallel_validation_preserves_ordered_selection() {
        let (graph, mut config) = mlp();
        config.optimization_steps = 5;
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    optimize(&graph, &config, None, |plan| {
                        let cycles = plan.program.estimated_cycles;
                        // Deliberately perturb completion order. Selection must
                        // depend on shortlist order, not callback arrival.
                        std::thread::sleep(std::time::Duration::from_millis(cycles % 7));
                        Ok((cycles, ()))
                    })
                    .unwrap()
                    .0
                })
        };
        let serial = run(1);
        let parallel = run(4);
        assert_eq!(serial.program, parallel.program);
        assert_eq!(serial.placement, parallel.placement);
        assert_eq!(serial.phases, parallel.phases);
    }

    #[test]
    fn infeasible_baseline_does_not_launch_another_search() {
        let (graph, config) = mlp();
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let result = optimize(&graph, &config, None, |_| {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Err::<(u64, ()), _>(invalid("baseline cannot fit"))
        });
        assert!(result.is_err());
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    }
}
