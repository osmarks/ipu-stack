//! Compiler driver: search over executable mid candidates, then compile each
//! through explicit expansion, support sizing, placement and exchange feedback.
//! The accepted result owns one final placement, schedule, package and cache.
mod config;

use crate::estimate::Ipu21CostModel;
use crate::estimate::memory_profile::write as memory_profile;
use crate::kernel::KernelBuildPlan;
use crate::low::LowProgram;
use crate::memory::TileMemoryMap;
use crate::package::{
    DiagnosticCheckpoint, PackageBuildResult, active_topology, diagnostic_tensor, package_inputs,
    package_multiply_precisions, package_precisions, validate_tile_count,
};
use crate::planner::proposals;
use crate::planner::{Candidate, Recipe, build};
use ipu_elf::Toolchain;
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
    let runtime =
        tracing::info_span!("compile_runtime").in_scope(|| -> PackageBuildResult<_> {
            let artifact =
                package
                    .toolchain
                    .compile(&package.runtime_source, "static_runtime", &[])?;
            Ok(std::fs::read(artifact.object)?)
        })?;
    let selected = tracing::info_span!("plan_package").in_scope(|| -> PackageBuildResult<_> {
        let costs = crate::estimate::MemoizedCostModel::new(&Ipu21CostModel);
        let fragments = crate::planner::cache::FragmentCache::default();
        let expansions = Arc::new(crate::storage::GeometryCache::default());
        let recipe = match tile_mapping {
            Some(mapping) => Recipe::baseline(config).remapped(mapping, config.tile_count)?,
            None => Recipe::baseline(config),
        };
        let mut attempted_recipes = Vec::new();
        let mut attempts = 0;
        let mut fixed = config.clone();
        let mut incumbent = build::build_candidate(graph, &fixed, &costs, &fragments, &recipe)?;
        memory_profile(graph, config, &incumbent.program, "baseline")?;
        let mut selected = evaluate_candidate(
            &incumbent.program,
            package,
            Arc::clone(&expansions),
            crate::ExchangeScheduleCache::default(),
            &runtime,
        )?;
        tracing::info!(cycles = selected.cycles, "validated search incumbent");
        // Fix logical homes for the remaining local search.
        for (input, planned) in graph.inputs().iter().zip(&incumbent.program.inputs) {
            fixed.inputs.insert(
                input.value,
                incumbent.program.values[planned.value.index() as usize]
                    .tensor_type
                    .format
                    .clone(),
            );
        }
        while attempts < config.optimization_steps {
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
                .filter(|(_, proposal)| !attempted_recipes.contains(&proposal.recipe))
                .map(|(proposal, proposed)| {
                    let recipe = proposed.recipe;
                    let span = tracing::debug_span!("local_screen", round = attempts, proposal);
                    let _entered = span.enter();
                    let candidate =
                        build::build_candidate(graph, &fixed, &costs, &fragments, &recipe);
                    let candidate = match candidate {
                        Ok(candidate) => {
                            let visited = attempted_recipes.contains(&candidate.recipe);
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
                                if !attempted_recipes.contains(&raw) {
                                    attempted_recipes.push(raw);
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
            let truncated = pending
                .len()
                .saturating_sub(config.optimization_steps - attempts);
            pending.truncate(config.optimization_steps - attempts);
            tracing::info!(
                round = attempts,
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
                break;
            }
            for (index, candidate) in pending.iter().enumerate() {
                let scope = format!("local-{}-candidate-{index}", attempts);
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
                    let attempt = attempts + index;
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
                remember(&mut attempted_recipes, pending.iter());
                // No shortlisted program improves the incumbent.
                break;
            };
            let candidate_cycles = plan.cycles;
            remember(&mut attempted_recipes, pending[..=index].iter());
            let mut candidate = pending.swap_remove(index).baseline;
            tracing::info!(
                attempt = attempts + index,
                before = selected.cycles,
                after = candidate_cycles,
                delta = ?candidate.recipe.changes(&incumbent.recipe),
                "accepted local layout improvement"
            );
            attempts += index + 1;
            for (id, alternatives) in incumbent.alternatives {
                candidate.alternatives.entry(id).or_insert(alternatives);
            }
            incumbent = candidate;
            selected = plan;
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
    expansions: Arc<crate::storage::GeometryCache>,
    mut cache: crate::ExchangeScheduleCache,
    runtime: &[u8],
) -> PackageBuildResult<EvaluatedCandidate> {
    let config = &package.pipeline;
    let expanded = crate::low::expand::expand_tiles_cached(
        mid,
        config.diagnostic_checkpoints,
        Arc::clone(&expansions),
    )?;
    let footprint = crate::estimate::program_footprint_analyzed(&expanded, &expansions)?;
    if footprint.maximum_transfer_chunks_per_tile > config.exchange_transfer_limit_per_tile {
        return Err(package::PackageBuildError::ExchangeTransferLimitExceeded {
            transfers: footprint.maximum_transfer_chunks_per_tile,
            limit: config.exchange_transfer_limit_per_tile,
        });
    }
    let program = crate::low::lower_to_tiles(&expanded, config.diagnostic_checkpoints);
    drop(expanded);
    let provisional_placement = tracing::info_span!("place_provisional_storage")
        .in_scope(|| -> PackageBuildResult<_> { Ok(crate::place::place(&program)?) })?;
    let topology = active_topology(program.tile_count)?;
    let provisional_exchanges = tracing::info_span!("schedule_provisional_exchanges").in_scope(
        || -> PackageBuildResult<_> {
            Ok(crate::exchange::lower_exchanges_cached(
                &program,
                &provisional_placement,
                &topology,
                config.exchange_stream_words,
                &mut cache,
            )?)
        },
    )?;
    let kernel_plan = tracing::info_span!("plan_kernels")
        .in_scope(|| -> PackageBuildResult<_> { Ok(KernelBuildPlan::from_program(&program)?) })?;
    let objects =
        tracing::info_span!("compile_kernels").in_scope(|| -> PackageBuildResult<_> {
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
    let placement =
        tracing::info_span!("place_storage").in_scope(|| -> PackageBuildResult<_> {
            Ok(crate::place::place_with_auxiliary(
                &program,
                &support.available_ranges,
                0,
                &support.profile_requests,
            )?)
        })?;
    let exchanges =
        tracing::info_span!("lower_exchanges").in_scope(|| -> PackageBuildResult<_> {
            Ok(crate::exchange::lower_exchanges_cached(
                &program,
                &placement,
                &topology,
                config.exchange_stream_words,
                &mut cache,
            )?)
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

pub mod f143;
pub mod runtime_layout;

mod estimate;
pub use estimate::{ExchangeStorageEstimator, estimate_exchange_phase_storage};
pub mod exchange;
pub mod graph;
mod host;
pub mod kernel;
mod low;
mod memory;
mod mid;
pub use planner::optimistic;
mod package;
mod place;
mod planner;
mod storage;
mod tensor;
mod tile;
pub use config::*;
pub(crate) use exchange::*;
pub use exchange::{
    EXCHANGE_SCHEDULE_SNAPSHOT_VERSION, ExchangeActivity, ExchangeActivityKind,
    ExchangeScheduleCache, ExchangeScheduleSnapshot, ExchangeSchedulingPriority,
    PhysicalExchangePhase, inactive_exchange_program, schedule_exchange_problem,
    schedule_exchange_problem_with_priority, select_exchange_schedule, validate_exchange_schedule,
};
pub use graph::{
    AttentionOptions, AttentionScale, AxisFactorView, AxisSlice, ComputeGraph, GemmOptions,
    GraphError, GraphInput, GraphInputKind, Operation, OperationId, OperationKind, Region, Repeat,
    ValueId,
};
pub(crate) use kernel::*;
pub(crate) use low::*;
pub use low::{
    BlockValue, BlockValueId, ShardDefinition, ShardView, logical_view_byte_spans,
    shard_storage_bytes,
};
pub(crate) use memory::*;
pub use mid::ReductionStaging;
pub(crate) use mid::*;
pub use package::{
    CompiledPackage, DiagnosticShard, DiagnosticTensor, TileProgramData, build_tile_program_package,
};
pub use place::profile::render_memory_profile;
pub(crate) use place::*;
pub use supervisor::*;
// Compatibility name for planner choices; this is not an executable mid node.
pub use planner::OperatorFamily as MidOperator;
pub use planner::{GemmOrientation, LocalOperandStaging, OperatorCandidate};
pub(crate) use storage::*;
pub use storage::{amp_matrix_coordinates, block_major_matrix_coordinates};
pub(crate) use tensor::*;
pub use tensor::{
    AMP_COLUMN_MICRO, AmpOrder, BlockMajorOrder, GridOrder, Layout, MemoryClass, OwnerMap,
    Precision, ShardExtent, TensorFormat, TensorType,
};
pub(crate) use tile::*;

mod supervisor;
