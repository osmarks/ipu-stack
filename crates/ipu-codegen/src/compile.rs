//! Compiler driver: search over executable mid candidates, then compile each
//! through explicit expansion, support sizing, placement and exchange feedback.
//! The accepted result owns one final placement, schedule, package and cache.
pub(crate) mod config;
pub use config::*;
mod benchmark;
mod placement;
mod screen;
pub use benchmark::{ExpansionBenchmark, ExpansionTiming, benchmark_mid_expansion};

use crate::estimate::Ipu21CostModel;
use crate::estimate::memory_profile::write as memory_profile;
use crate::graph::ComputeGraph;
use crate::kernel::KernelBuildPlan;
use crate::low::LowProgram;
use crate::memory::TileMemoryMap;
use crate::package::{
    self, CompiledPackage, DiagnosticCheckpoint, PackageBuildResult, active_topology, build_phase,
    diagnostic_tensor, invalid, package_inputs, package_multiply_precisions, package_precisions,
    validate_tile_count,
};
use crate::planner::{Candidate, Recipe, build};
use crate::planner::{checkpoint, proposals};
use ipu_elf::Toolchain;
use ipu_target::ipu21::fabric::Topology;
use rayon::prelude::*;
use std::{path::PathBuf, sync::Arc};

#[derive(Clone, Debug)]
pub struct PackageConfig {
    /// Host-triggered inference calls after a single parameter upload.
    pub invocations: u32,
    pub toolchain: Toolchain,
    pub runtime_source: PathBuf,
    pub kernel_source_directory: PathBuf,
    pub pipeline: PipelineConfig,
    /// Initial bijection from planned tile indices to execution tile indices.
    /// A supplied map disables automatic ownership-remapping proposals.
    pub tile_mapping: Option<Vec<u16>>,
}

struct EvaluatedCandidate {
    program: LowProgram,
    placement: crate::Placement,
    exchanges: crate::exchange::LoweredExchanges,
    application: ipu_package::Application,
    support_memory: TileMemoryMap,
    exchange_code_base: u32,
    cycles: u64,
    cache: crate::ExchangeScheduleCache,
}

/// Compiles and packages a compute graph into a directly loadable IPU21
/// application.
#[tracing::instrument(
    name = "ipu_codegen.package.build",
    skip(graph, config),
    fields(tile_count = config.pipeline.tile_count, operations = graph.operations().len())
)]
pub fn build_package(
    graph: &ComputeGraph,
    config: &PackageConfig,
) -> PackageBuildResult<CompiledPackage> {
    build_package_with_checkpoints(graph, config, false)
}

/// Builds an ordinary optimized package with resumable PBRK0 traps after each
/// top-level operator and returns the storage map needed for non-invasive
/// numerical inspection.
pub fn build_diagnostic_package(
    graph: &ComputeGraph,
    config: &PackageConfig,
) -> PackageBuildResult<CompiledPackage> {
    build_package_with_checkpoints(graph, config, true)
}

fn build_package_with_checkpoints(
    graph: &ComputeGraph,
    config: &PackageConfig,
    diagnostic: bool,
) -> PackageBuildResult<CompiledPackage> {
    let mut config = config.clone();
    config.pipeline.diagnostic_checkpoints = diagnostic;
    if diagnostic {
        config.pipeline.profiling = false;
    }
    let built = compile_graph(graph, &config)?;
    let low = &built.program;
    let topology = active_topology(low.tile_count)?;
    let inputs = package_inputs(low, &built.placement, &topology)?;
    let outputs = low
        .outputs
        .iter()
        .enumerate()
        .map(|(index, output)| {
            diagnostic_tensor(
                low,
                &built.placement,
                &topology,
                *output,
                Some(format!("output.{index}")),
            )
        })
        .collect::<PackageBuildResult<Vec<_>>>()?;
    let mut checkpoints = Vec::new();
    for (source, results) in low.checkpoints.iter().filter(|_| diagnostic) {
        let source = *source;
        let tensors = results
            .iter()
            .map(|&value| diagnostic_tensor(low, &built.placement, &topology, value, None))
            .collect::<PackageBuildResult<Vec<_>>>()?;
        // An elided view has no independently materialized boundary; its
        // consumer's checkpoint covers the composed mapping instead.
        if tensors.iter().all(|tensor| tensor.shards.is_empty()) {
            continue;
        }
        for tensor in &tensors {
            tracing::debug!(
                operation = source.index(),
                value = tensor.value.index(),
                shape = ?tensor.shape.0,
                precision = ?tensor.precision,
                shards = tensor.shards.len(),
                order = ?tensor.shards.first().map(|shard| &shard.storage.tensor_type.format.layout.order),
                memory_class = ?tensor.shards.first().map(|shard| shard.storage.tensor_type.format.layout.memory_class),
                first_extents = ?tensor.shards.first().map(|shard| &shard.storage.extents),
                "recorded diagnostic tensor"
            );
        }
        checkpoints.push(DiagnosticCheckpoint {
            operation: source,
            breakpoint: (checkpoints.len() & 1) as u8,
            tensors,
        });
    }
    Ok(CompiledPackage {
        application: built.application,
        inputs,
        outputs,
        checkpoints,
        precisions: package_precisions(low),
        multiply_precisions: package_multiply_precisions(low),
        exchange_phases: built.exchanges.phases,
        exchange_schedule: built.exchanges.schedule_snapshot,
        exchange_code_base: built.exchange_code_base,
    })
}

fn compile_graph(
    graph: &ComputeGraph,
    package: &PackageConfig,
) -> PackageBuildResult<EvaluatedCandidate> {
    let config = &package.pipeline;
    let tile_mapping = package.tile_mapping.as_deref();
    validate_tile_count(u32::from(config.tile_count))?;
    let runtime = build_phase("compile_runtime", || {
        let artifact = package
            .toolchain
            .compile(&package.runtime_source, "static_runtime", &[])?;
        Ok(std::fs::read(artifact.object)?)
    })?;
    let selected = build_phase("plan_package", || {
        let costs = crate::estimate::MemoizedCostModel::new(&Ipu21CostModel);
        let fragments = crate::planner::cache::FragmentCache::default();
        let expansions = Arc::new(crate::low::expand::ExpansionCache::default());
        let mut state = checkpoint::State::load(graph, config, tile_mapping)?;
        let mut fixed = config.clone();
        let resuming = config.load_search_state.is_some();
        if resuming {
            fixed.inputs = state.inputs.clone();
        }
        let mut incumbent =
            build::build_candidate(graph, &fixed, &costs, &fragments, &state.recipe)?;
        if resuming {
            incumbent.alternatives = state.alternatives.clone();
        }
        memory_profile(graph, config, &incumbent.program, "baseline")?;
        let mut selected = evaluate_candidate(
            &incumbent.program,
            package,
            Arc::clone(&expansions),
            crate::ExchangeScheduleCache::default(),
            &runtime,
        )?;
        tracing::info!(
            cycles = selected.cycles,
            resuming,
            "validated search incumbent"
        );
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
        while state.attempts < budget_end {
            let traffic = if tile_mapping.is_none() {
                crate::exchange::MappingTraffic::new(&selected.program, &selected.placement)
                    .map_err(
                        |error| tracing::info!(%error, "could not estimate ownership proposals"),
                    )
                    .ok()
            } else {
                None
            };
            let proposed = proposals(graph, config, &incumbent, traffic.as_ref());
            let proposed_count = proposed.len();
            let screened = proposed
                .into_par_iter()
                .enumerate()
                .filter(|(_, proposal)| !state.visited.contains(&proposal.recipe))
                .map(|(proposal, proposed)| {
                    let recipe = proposed.recipe;
                    let span =
                        tracing::debug_span!("local_screen", round = state.attempts, proposal);
                    let _entered = span.enter();
                    let candidate =
                        build::build_candidate(graph, &fixed, &costs, &fragments, &recipe);
                    let candidate = match candidate {
                        Ok(candidate) => {
                            let visited = state.visited.contains(&candidate.recipe);
                            let estimate = proposed
                                .estimated_cycles
                                .unwrap_or(candidate.program.estimated_cycles);
                            let keep = !visited && estimate < incumbent.program.estimated_cycles;
                            tracing::debug!(
                                incumbent_estimate = incumbent.program.estimated_cycles,
                                candidate_estimate = estimate,
                                visited, keep, delta = ?candidate.recipe.changes(&incumbent.recipe),
                                "screened local recipe"
                            );
                            if keep {
                                Ok((candidate, estimate))
                            } else if visited {
                                Err(Skipped::Visited)
                            } else {
                                Err(Skipped::NotCheaper)
                            }
                        }
                        Err(error) => {
                            tracing::debug!(%error, "discarded invalid local recipe");
                            Err(Skipped::Invalid)
                        }
                    };
                    (proposal, recipe, candidate)
                })
                .collect::<Vec<_>>();
            let mut visited = proposed_count - screened.len();
            let mut invalid = 0;
            let mut not_cheaper = 0;
            let mut deduplicated = 0;
            let mut pending: Vec<ShortlistedCandidate> = Vec::new();
            for (proposal, raw, result) in screened {
                let (candidate, estimate) = match result {
                    Ok(candidate) => candidate,
                    Err(reason) => {
                        match reason {
                            Skipped::Invalid => invalid += 1,
                            Skipped::NotCheaper => not_cheaper += 1,
                            Skipped::Visited => {
                                visited += 1;
                                if !state.visited.contains(&raw) {
                                    state.visited.push(raw);
                                }
                            }
                        }
                        continue;
                    }
                };
                if let Some(same) = pending
                    .iter_mut()
                    .find(|old| old.baseline.program == candidate.program)
                {
                    same.estimated_cycles = same.estimated_cycles.min(estimate);
                    same.recipes.extend([raw, candidate.recipe]);
                    deduplicated += 1;
                } else {
                    let recipes = vec![raw, candidate.recipe.clone()];
                    pending.push(ShortlistedCandidate {
                        proposal,
                        estimated_cycles: estimate,
                        baseline: candidate,
                        recipes,
                    });
                }
            }
            pending.sort_by_key(|candidate| candidate.estimated_cycles);
            let truncated = pending.len().saturating_sub(budget_end - state.attempts);
            pending.truncate(budget_end - state.attempts);
            tracing::info!(
                round = state.attempts,
                proposed = proposed_count,
                invalid,
                visited,
                not_cheaper,
                deduplicated,
                truncated,
                shortlisted = pending.len(),
                "screened local search round"
            );
            if pending.is_empty() {
                state.save(config, &incumbent, &fixed)?;
                break;
            }
            for (index, candidate) in pending.iter().enumerate() {
                let scope = format!("local-{}-candidate-{index}", state.attempts);
                if let Err(error) =
                    memory_profile(graph, &fixed, &candidate.baseline.program, &scope)
                {
                    tracing::warn!(%error, scope, "skipped candidate memory profile");
                }
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
                    let span = tracing::info_span!(
                        "local_candidate",
                        attempt,
                        proposal = candidate.proposal
                    );
                    let _entered = span.enter();
                    let result = evaluate_candidate(
                        &candidate.baseline.program,
                        package,
                        Arc::clone(&expansions),
                        selected.cache.clone(),
                        &runtime,
                    );
                    match result {
                        Ok(plan) if plan.cycles < selected.cycles => Some((index, plan)),
                        Ok(plan) => {
                            let candidate_cycles = plan.cycles;
                            tracing::info!(
                                candidate_cycles,
                                cycles = selected.cycles,
                                "retained faster incumbent"
                            );
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
            let Some((index, plan)) = winner else {
                state.attempts += pending.len();
                remember(&mut state.visited, pending.iter());
                state.save(config, &incumbent, &fixed)?;
                // All shortlisted programs failed. A truncated tail is available
                // on resume; otherwise the unchanged incumbent has no new proposals.
                break;
            };
            let candidate_cycles = plan.cycles;
            remember(&mut state.visited, pending[..=index].iter());
            let mut candidate = pending.swap_remove(index).baseline;
            tracing::info!(
                attempt = state.attempts + index,
                before = selected.cycles,
                after = candidate_cycles,
                delta = ?candidate.recipe.changes(&incumbent.recipe),
                "accepted local layout improvement"
            );
            state.attempts += index + 1;
            for (id, alternatives) in incumbent.alternatives {
                candidate.alternatives.entry(id).or_insert(alternatives);
            }
            incumbent = candidate;
            selected = plan;
            state.save(config, &incumbent, &fixed)?;
        }
        Ok(selected)
    })?;
    if let Some(directory) = &config.memory_profile_directory {
        crate::place::profile::write(
            directory,
            &selected.program,
            &selected.placement,
            &selected.support_memory,
            &selected.application,
        )?;
    }
    Ok(selected)
}

enum Skipped {
    Invalid,
    Visited,
    NotCheaper,
}

/// All recipes lowering to one program travel together until validation. Do not
/// mark aliases of a truncated or cancelled candidate as visited.
struct ShortlistedCandidate {
    proposal: usize,
    estimated_cycles: u64,
    baseline: Candidate,
    recipes: Vec<Recipe>,
}

fn remember<'a>(
    visited: &mut Vec<Recipe>,
    candidates: impl IntoIterator<Item = &'a ShortlistedCandidate>,
) {
    for recipe in candidates
        .into_iter()
        .flat_map(|candidate| &candidate.recipes)
    {
        if !visited.contains(recipe) {
            visited.push(recipe.clone());
        }
    }
}

/// Compile one complete candidate. Provisional addresses are local to sizing;
/// only the final placement, exchanges and package escape this procedure.
fn evaluate_candidate(
    mid: &crate::MidProgram,
    package: &PackageConfig,
    expansions: Arc<crate::low::expand::ExpansionCache>,
    mut cache: crate::ExchangeScheduleCache,
    runtime: &[u8],
) -> PackageBuildResult<EvaluatedCandidate> {
    let config = &package.pipeline;
    let (program, _) = screen::expand_and_screen(mid, config, expansions)?;
    let provisional_placement = build_phase("place_provisional_storage", || {
        Ok(crate::place::place(&program)?)
    })?;
    let topology = active_topology(program.tile_count)?;
    let provisional_exchanges = build_phase("schedule_provisional_exchanges", || {
        Ok(crate::exchange::lower_exchanges_cached(
            &program,
            &provisional_placement,
            &topology,
            config.exchange_stream_words,
            false,
            &mut cache,
        )?)
    })?;
    let kernel_plan = build_phase("plan_kernels", || {
        Ok(KernelBuildPlan::from_program(&program)?)
    })?;
    let objects = build_phase("compile_kernels", || {
        let mut objects = vec![runtime.to_vec()];
        for compilation in &kernel_plan.compilations {
            let artifact = package.toolchain.compile(
                package.kernel_source_directory.join(compilation.source),
                &compilation.name,
                &compilation.flags,
            )?;
            objects.push(std::fs::read(&artifact.object)?);
        }
        Ok(objects)
    })?;
    let support = package::size_support(
        &program,
        &provisional_placement,
        &provisional_exchanges.phases,
        config,
        objects,
        kernel_plan,
        package.invocations,
    )?;
    drop(provisional_exchanges);
    drop(provisional_placement);
    let mut placement = build_phase("place_storage", || {
        Ok(crate::place::place_with_auxiliary(
            &program,
            &support.available_ranges,
            0,
            &support.profile_requests,
        )?)
    })?;
    let mut exchanges = build_phase("lower_exchanges", || {
        Ok(crate::exchange::lower_exchanges_cached(
            &program,
            &placement,
            &topology,
            config.exchange_stream_words,
            config.exchange_diagnostics,
            &mut cache,
        )?)
    })?;
    build_phase("optimize_exchange_placement", || {
        if let Some(proposal) = placement::propose_exchange_placement(
            &program,
            &support.available_ranges,
            &support.profile_requests,
            &placement,
        )? {
            // A failed/slower alternative cannot contaminate the accepted cache.
            let mut alternative_cache = cache.clone();
            match crate::exchange::lower_exchanges_cached(
                &program,
                &proposal.placement,
                &topology,
                config.exchange_stream_words,
                false,
                &mut alternative_cache,
            ) {
                Ok(alternative) => {
                    let baseline_cycles = placement::exchange_cycles(&program, &exchanges.phases);
                    let candidate_cycles =
                        placement::exchange_cycles(&program, &alternative.phases);
                    let row_bytes = crate::tile::compact_exchange_table_bytes(
                        &alternative.phases,
                        u16::try_from(Topology::c600().tile_count())?,
                        program.tile_count,
                    )?;
                    let row_capacity = support.exchange_row_capacity();
                    let accepted = candidate_cycles < baseline_cycles && row_bytes <= row_capacity;
                    tracing::info!(offset = proposal.offset, baseline_score = %proposal.baseline_score,
                        score = %proposal.score, baseline_cycles, candidate_cycles, row_bytes,
                        row_capacity, accepted, "evaluated exchange placement candidate");
                    if accepted {
                        placement = proposal.placement;
                        exchanges = alternative;
                        cache = alternative_cache;
                    }
                }
                Err(error) => tracing::info!(offset = proposal.offset, %error,
                    "rejected unschedulable exchange placement"),
            }
        }
        Ok(())
    })?;
    let final_cost =
        crate::estimate::scheduled_program_cycles(&program.program, &exchanges.phases)?;
    tracing::info!(
        final_cycles = final_cost.total,
        final_exchange = final_cost.exchange,
        "costed final placed program"
    );
    let application = package::emit_package(
        &program,
        &placement,
        &exchanges.phases,
        &support,
        config,
        package.invocations,
    )?;
    Ok(EvaluatedCandidate {
        program,
        placement,
        exchanges,
        application,
        support_memory: support.memory,
        exchange_code_base: support.exchange_code_base,
        cycles: final_cost.total,
        cache,
    })
}

/// Capture address-resolved ordinary transfers before scheduling or linking.
/// Capture the canonical baseline; failed placements remain errors.
pub fn capture_exchange_baseline(
    graph: &ComputeGraph,
    config: &PackageConfig,
) -> PackageBuildResult<crate::ExchangeScheduleSnapshot> {
    let planning = &config.pipeline;
    validate_tile_count(u32::from(planning.tile_count))?;
    let costs = crate::estimate::MemoizedCostModel::new(&Ipu21CostModel);
    let mut mid = crate::planner::build_baseline(graph, planning, &costs)?;
    if let Some(mapping) = &config.tile_mapping {
        mid.remap_tiles(mapping)?;
    }
    let (low, placement, _) = screen::expand_and_place(&mid, planning)?;
    Ok(crate::exchange::capture_exchange_schedule(
        &low, &placement,
    )?)
}

#[cfg(test)]
mod tests;
