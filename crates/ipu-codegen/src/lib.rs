//! Compiler driver: build the baseline mid program, then expand, size support,
//! place storage, schedule exchanges, and emit the application.
mod config;

use crate::estimate::memory_profile::write as memory_profile;
use crate::kernel::KernelObjects;
use crate::low::LowGraph;
use crate::package::{
    DiagnosticCheckpoint, PackageBuildResult, active_topology, diagnostic_tensor, package_inputs,
    package_multiply_precisions, package_precisions, validate_tile_count,
};
use ipu_elf::Toolchain;
use std::{path::PathBuf, sync::Arc};

#[derive(Clone, Debug)]
pub struct PackageConfig {
    /// Host-triggered inference calls after a single parameter upload.
    pub invocations: u32,
    pub toolchain: Toolchain,
    pub runtime_source: PathBuf,
    pub kernel_source_directory: PathBuf,
    pub pipeline: PipelineConfig,
}

struct Compilation {
    program: LowGraph,
    placement: crate::Placement,
    exchanges: Vec<crate::exchange::PhysicalExchangePhase>,
    application: ipu_package::Application,
    exchange_code_base: u32,
}

/// Compiles and packages a compute graph into a directly loadable IPU21
/// application.
#[tracing::instrument(
    name = "ipu_codegen.package.build",
    skip(graph, config),
    fields(tile_count = config.pipeline.tile_count, operations = graph.operations().len())
)]
pub fn build_package(
    graph: &HighGraph,
    config: &PackageConfig,
) -> PackageBuildResult<CompiledPackage> {
    build_package_with_checkpoints(graph, config, false)
}

/// Builds an ordinary optimized package with resumable PBRK0 traps after each
/// top-level operator and returns the storage map needed for non-invasive
/// numerical inspection.
pub fn build_diagnostic_package(
    graph: &HighGraph,
    config: &PackageConfig,
) -> PackageBuildResult<CompiledPackage> {
    build_package_with_checkpoints(graph, config, true)
}

fn build_package_with_checkpoints(
    graph: &HighGraph,
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
    let topology = active_topology(config.pipeline.target, low.tile_count)?;
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
        exchange_phases: built.exchanges,
        exchange_code_base: built.exchange_code_base,
    })
}

fn compile_graph(graph: &HighGraph, package: &PackageConfig) -> PackageBuildResult<Compilation> {
    let config = &package.pipeline;
    validate_tile_count(config.target, u32::from(config.tile_count))?;
    let runtime =
        tracing::info_span!("compile_runtime").in_scope(|| -> PackageBuildResult<_> {
            let artifact = package.toolchain.compile(
                config.target,
                &package.runtime_source,
                "static_runtime",
                &[],
            )?;
            Ok(std::fs::read(artifact.object)?)
        })?;
    let layouts = planner::boundary_layouts(graph, config);
    let mid = tracing::info_span!("construct_mid")
        .in_scope(|| planner::plan(graph, &layouts, config, planner::SearchLimits::default()))?;
    memory_profile(graph, config, &mid, "baseline")?;
    let mut cache = crate::ExchangeScheduleCache::default();
    let expansions = Arc::new(crate::storage::GeometryCache::default());
    let expanded = crate::low::expand::expand_tiles_cached(
        config.target,
        &mid,
        config.diagnostic_checkpoints,
        config.reuse_cast_inputs && !config.diagnostic_checkpoints,
        Arc::clone(&expansions),
    )?;
    let program = crate::low::lower_to_tiles(&expanded, config.diagnostic_checkpoints);
    let provisional_placement = tracing::info_span!("place_provisional_storage").in_scope(
        || -> PackageBuildResult<_> { Ok(crate::place::place(config.target, &program)?) },
    )?;

    let topology = active_topology(config.target, program.tile_count)?;
    let provisional_exchanges = tracing::info_span!("schedule_provisional_exchanges").in_scope(
        || -> PackageBuildResult<_> {
            Ok(crate::exchange::lower_exchanges_cached(
                config.target,
                &program,
                &provisional_placement,
                &topology,
                config.exchange_stream_words,
                &mut cache,
            )?)
        },
    )?;
    let kernel_plan =
        tracing::info_span!("plan_kernels").in_scope(|| -> PackageBuildResult<_> {
            Ok(KernelObjects::from_program(config.target, &program)?)
        })?;
    let objects =
        tracing::info_span!("compile_kernels").in_scope(|| -> PackageBuildResult<_> {
            let mut objects = vec![runtime];
            for compilation in &kernel_plan.compilations {
                let artifact = package.toolchain.compile(
                    config.target,
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
        &provisional_exchanges,
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
                config.target,
                &program,
                &support.available_ranges,
                0,
                &support.profile_requests,
            )?)
        })?;
    let exchanges =
        tracing::info_span!("lower_exchanges").in_scope(|| -> PackageBuildResult<_> {
            Ok(crate::exchange::lower_exchanges_cached(
                config.target,
                &program,
                &placement,
                &topology,
                config.exchange_stream_words,
                &mut cache,
            )?)
        })?;
    let final_cost =
        crate::estimate::scheduled_program_cycles(config.target, &program.program, &exchanges)?;
    tracing::info!(
        final_cycles = final_cost.total,
        final_exchange = final_cost.exchange,
        "costed final placed program"
    );
    let application = package::emit_package(
        &program,
        &placement,
        &exchanges,
        &support,
        config,
        package.invocations,
    )?;
    if let Some(directory) = &config.memory_profile_directory {
        crate::place::profile::write(
            directory,
            &program,
            &placement,
            &support.memory,
            &application,
        )?;
    }
    Ok(Compilation {
        program,
        placement,
        exchanges,
        application,
        exchange_code_base: support.exchange_code_base,
    })
}

pub mod f143;

mod estimate;
pub mod exchange;
pub mod graph;
mod host;
pub mod kernel;
mod low;
mod memory;
mod mid;
mod package;
mod place;
mod planner;
mod storage;
mod tensor;
mod tile;
pub use config::*;
pub(crate) use exchange::*;
pub use exchange::{
    ExchangeActivity, ExchangeActivityKind, ExchangeScheduleCache, PhysicalExchangePhase,
};
pub use graph::{
    AttentionOptions, AttentionScale, AxisFactorView, AxisSlice, GemmOptions, GraphError,
    GraphInput, GraphInputKind, HighGraph, Operation, OperationId, OperationKind, Region, Repeat,
    ValueId,
};
pub(crate) use kernel::*;
pub(crate) use low::*;
pub use low::{
    BlockValue, BlockValueId, ShardDefinition, ShardView, logical_view_byte_spans,
    shard_storage_bytes,
};
pub(crate) use mid::*;
pub use package::{
    CompiledPackage, DiagnosticShard, DiagnosticTensor, TileProgramData, build_tile_program_package,
};
pub use place::profile::render_memory_profile;
pub(crate) use place::*;
pub(crate) use storage::*;
pub use storage::{amp_matrix_coordinates, block_major_matrix_coordinates};
pub use supervisor::*;
pub(crate) use tensor::*;
pub use tensor::{
    AMP_COLUMN_MICRO, AmpOrder, BlockMajorOrder, GridOrder, Layout, MemoryClass, OwnerMap,
    Precision, ShardExtent, TensorFormat, TensorType,
};
pub(crate) use tile::*;

mod supervisor;
