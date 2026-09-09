mod bindings;
use bindings::{PackageBindings, auxiliary_ranges};
mod placement;
mod profile;
mod profile_work;
use profile::{instrument_profile, profile_binding, profile_step_count};
mod benchmark;
mod regional;
mod selection;
pub use benchmark::{ExpansionBenchmark, ExpansionTiming, benchmark_mid_expansion};
mod tile_program;
use selection::ScheduledPlan;
pub use tile_program::build_tile_program_package;

use crate::graph::{ComputeGraph, OperationId, ValueId};
use crate::host;
use crate::low::LowProgram;
use crate::memory::{
    MemoryLayoutError, MemoryRequest, PROFILE_END_CYCLE, PROFILE_START_CYCLE, RUNTIME_STATE_BASE,
    RUNTIME_STATE_BYTES, TileMemoryMap, WORKER_STACK_HEADROOM,
};
use crate::{
    COMPLETE_SYMBOL, COMPLETION_ADDRESS_SYMBOL, CodegenOptions, HOST_RUN_SYMBOL, KernelBuildPlan,
    PRNG_SEED_SYMBOL, PROGRAM_ADDRESS_SYMBOL, REPEAT_CALL_SYMBOL, RUNTIME_ENTRY_SYMBOL,
    SAMPLE_CYCLE_SYMBOL, TileProgram, TileProgramLowering, WORKER_BARRIER_SYMBOL,
    WORKER_STACK_BASE_SYMBOL, WORKER_SYNC_CONTEXT_SYMBOL, emit, lower_to_tiles, place,
    shard_storage_bytes,
};
use crate::{Ipu21CostModel, PipelineConfig, Precision, TileGraph, lower_finalists};
use ipu_driver::{APPLICATION_LOAD_BASE, TILES_PER_BATCH};
use ipu_elf::{ElfError, LinkOptions, LinkedImage, Toolchain, link};
use ipu_exchange::{ExchangeError, Topology, encode_br_m, encode_setzi_m};
use ipu_package::{
    Application, Binding, DEBUG_ALL_TILES, DebugRegion, DebugSymbol, EntryPoint,
    PROFILE_CYCLES_BINDING, PackageError, ProfileExchangeActivity, ProfileExchangeActivityKind,
    ProfileMetadata, ProfileStep, ProfileStepKind, RegionSlice, SEGMENT_EXECUTE, SEGMENT_READ,
    SEGMENT_WRITE, Segment, TILE_MEMORY_BASE, TileImage, TileProfilePlan,
};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::num::TryFromIntError;
use std::path::PathBuf;
use std::time::Instant;

const ENTRY_BYTES: u32 = 8;
const SUPPORT_START: u32 = APPLICATION_LOAD_BASE + ENTRY_BYTES;
const COMPLETION_ADDRESS: u32 = RUNTIME_STATE_BASE;
const RUNTIME_EXECUTABLE_START: u32 =
    (RUNTIME_STATE_BASE + RUNTIME_STATE_BYTES + ipu_package::TILE_MEMORY_ELEMENT_SIZE - 1)
        & !(ipu_package::TILE_MEMORY_ELEMENT_SIZE - 1);

#[derive(Debug, thiserror::Error)]
pub enum PackageBuildError {
    #[error("exchange transfer count exceeds per-tile limit: {transfers} fragments, limit {limit}")]
    ExchangeTransferLimitExceeded { transfers: u64, limit: u64 },
    #[error("exchange tables exceed per-tile budget: {bytes} bytes, limit {budget} bytes")]
    ExchangeBudgetExceeded { bytes: u64, budget: u64 },
    #[error("code generation failed: {0}")]
    Codegen(#[from] crate::CodegenError),
    #[error("ELF processing failed: {0}")]
    Elf(#[from] ElfError),
    #[error("exchange encoding failed: {0}")]
    Exchange(#[from] ExchangeError),
    #[error("package construction failed: {0}")]
    Package(#[from] PackageError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("integer conversion failed: {0}")]
    Integer(#[from] TryFromIntError),
    #[error("invalid package build: {0}")]
    Invalid(String),
    #[error("mid-level lowering failed: {0}")]
    Mid(#[from] crate::LoweringError),
    #[error("tile scheduling failed: {0}")]
    Low(#[from] crate::ExpansionError),
    #[error("kernel planning failed: {0}")]
    Kernel(#[from] crate::KernelAbiError),
    #[error("placement failed: {0}")]
    Placement(#[from] crate::PlacementError),
    #[error("exchange lowering failed: {0}")]
    ExchangeLowering(#[from] crate::ExchangeLoweringError),
    #[error("tile-program lowering failed: {0}")]
    TileLowering(#[from] crate::TileLoweringError),
    #[error("storage layout failed: {0}")]
    Storage(#[from] crate::StorageError),
}

impl From<MemoryLayoutError> for PackageBuildError {
    fn from(error: MemoryLayoutError) -> Self {
        Self::Invalid(error.to_string())
    }
}

pub type PackageBuildResult<T> = std::result::Result<T, PackageBuildError>;

#[derive(Clone, Debug)]
pub struct PackageConfig {
    pub toolchain: Toolchain,
    pub runtime_source: PathBuf,
    pub kernel_source_directory: PathBuf,
    pub pipeline: PipelineConfig,
    /// Optional bijection from planned tile indices to execution tile indices.
    pub tile_mapping: Option<Vec<u16>>,
}

/// Data embedded in one logical tile image for a finalized tile-program package.
#[derive(Clone, Debug)]
pub struct TileProgramData {
    pub tile: u16,
    pub address: u32,
    pub data: Vec<u8>,
}

/// A loadable application together with the optimized physical storage of its
/// graph inputs and outputs.  The storage map lets hosts populate and inspect
/// logical tensors without assuming a particular planner-selected layout.
#[derive(Clone, Debug)]
pub struct CompiledPackage {
    pub application: Application,
    pub inputs: Vec<DiagnosticTensor>,
    pub outputs: Vec<DiagnosticTensor>,
    pub precisions: BTreeMap<ValueId, Precision>,
    /// Selected operand precision of each semantic product (before accumulation).
    pub multiply_precisions: BTreeMap<crate::OperationId, Precision>,
    /// Exact physical exchange schedules retained for low-level diagnostics.
    /// This is build metadata and is not serialized into the application.
    pub exchange_phases: Vec<crate::PhysicalExchangePhase>,
    /// Address-resolved inputs to physical exchange scheduling and row codegen.
    pub exchange_schedule: crate::ExchangeScheduleSnapshot,
    /// Base address used when laying out the compact per-tile exchange table.
    pub exchange_code_base: u32,
}

/// A loadable package plus the exact device storage visible at each semantic
/// operator checkpoint.
#[derive(Clone, Debug)]
pub struct DiagnosticPackage {
    pub application: Application,
    pub inputs: Vec<DiagnosticTensor>,
    pub checkpoints: Vec<DiagnosticCheckpoint>,
    pub precisions: BTreeMap<ValueId, Precision>,
    /// Selected operand precision of each semantic product (before accumulation).
    pub multiply_precisions: BTreeMap<crate::OperationId, Precision>,
    /// Exact physical exchange schedules retained for low-level diagnostics.
    /// This is build metadata and is not serialized into the application.
    pub exchange_phases: Vec<crate::PhysicalExchangePhase>,
    /// Address-resolved inputs to physical exchange scheduling and row codegen.
    pub exchange_schedule: crate::ExchangeScheduleSnapshot,
    /// Base address used when laying out the compact per-tile exchange table.
    pub exchange_code_base: u32,
}

#[derive(Clone, Debug)]
pub struct DiagnosticCheckpoint {
    pub operation: OperationId,
    pub breakpoint: u8,
    pub tensors: Vec<DiagnosticTensor>,
}

#[derive(Clone, Debug)]
pub struct DiagnosticTensor {
    pub name: Option<String>,
    pub value: ValueId,
    pub shape: crate::TensorShape,
    pub precision: Precision,
    pub shards: Vec<DiagnosticShard>,
}

#[derive(Clone, Debug)]
pub struct DiagnosticShard {
    pub physical_tile: u16,
    pub address: u32,
    pub storage: crate::BlockValue,
}

struct BuiltApplication {
    application: Application,
    placement: crate::Placement,
    exchange_phases: Vec<crate::PhysicalExchangePhase>,
    exchange_schedule: crate::ExchangeScheduleSnapshot,
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
    graph: &ComputeGraph,
    config: &PackageConfig,
) -> PackageBuildResult<CompiledPackage> {
    let (built, low) = build_package_artifacts(graph, config, false)?;
    let topology = active_topology(low.tile_count)?;
    let inputs = package_inputs(&low, &built.placement, &topology)?;
    let outputs = low
        .outputs
        .iter()
        .enumerate()
        .map(|(index, output)| {
            diagnostic_tensor(
                &low,
                &built.placement,
                &topology,
                output.value,
                Some(format!("output.{index}")),
            )
        })
        .collect::<PackageBuildResult<Vec<_>>>()?;
    let precisions = package_precisions(&low);
    Ok(CompiledPackage {
        application: built.application,
        inputs,
        outputs,
        precisions,
        multiply_precisions: package_multiply_precisions(&low),
        exchange_phases: built.exchange_phases,
        exchange_schedule: built.exchange_schedule,
        exchange_code_base: built.exchange_code_base,
    })
}

/// Builds an ordinary optimized package with resumable PBRK0 traps after each
/// top-level operator and returns the storage map needed for non-invasive
/// numerical inspection.
pub fn build_diagnostic_package(
    graph: &ComputeGraph,
    config: &PackageConfig,
) -> PackageBuildResult<DiagnosticPackage> {
    let (built, low) = build_package_artifacts(graph, config, true)?;
    let topology = active_topology(low.tile_count)?;
    let inputs = package_inputs(&low, &built.placement, &topology)?;
    let mut checkpoints = Vec::new();
    for (source, results) in &low.checkpoints {
        let source = *source;
        let tensors = results
            .iter()
            .map(|&value| diagnostic_tensor(&low, &built.placement, &topology, value, None))
            .collect::<PackageBuildResult<Vec<_>>>()?;
        // A fully deferred view operation has no device work or independently
        // materialized boundary to stop at; its consumer's checkpoint covers
        // the fused mapping instead.
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
    Ok(DiagnosticPackage {
        application: built.application,
        inputs,
        checkpoints,
        precisions: package_precisions(&low),
        multiply_precisions: package_multiply_precisions(&low),
        exchange_phases: built.exchange_phases,
        exchange_schedule: built.exchange_schedule,
        exchange_code_base: built.exchange_code_base,
    })
}

fn package_multiply_precisions(low: &TileGraph) -> BTreeMap<crate::OperationId, Precision> {
    low.kernel_runs
        .iter()
        .filter_map(|run| {
            let crate::TileKernelSpec::Gemm { multiply, .. } = run.kernel else {
                return None;
            };
            Some((run.provenance.operation?, multiply))
        })
        .collect()
}

fn package_precisions(mid: &TileGraph) -> BTreeMap<ValueId, Precision> {
    let mut precisions = BTreeMap::new();
    // Canonical values precede implementation-local staging and accumulator
    // values, which share their producer's origin for profiling purposes.
    for value in &mid.logical_values {
        precisions
            .entry(value.origin)
            .or_insert(value.tensor_type.format.precision);
    }
    precisions
}

fn package_inputs(
    low: &LowProgram,
    placement: &crate::Placement,
    topology: &Topology,
) -> PackageBuildResult<Vec<DiagnosticTensor>> {
    low.inputs
        .iter()
        .map(|input| {
            diagnostic_tensor(
                low,
                placement,
                topology,
                input.value,
                Some(input.name.clone()),
            )
        })
        .collect()
}

fn build_package_artifacts(
    graph: &ComputeGraph,
    config: &PackageConfig,
    diagnostic_checkpoints: bool,
) -> PackageBuildResult<(BuiltApplication, LowProgram)> {
    validate_tile_count(u32::from(config.pipeline.tile_count))?;
    let mut planning = config.pipeline.clone();
    planning.diagnostic_checkpoints = diagnostic_checkpoints;
    if diagnostic_checkpoints {
        planning.profiling = false;
    }
    let runtime_artifact = build_phase("compile_runtime", || {
        Ok(config
            .toolchain
            .compile(&config.runtime_source, "static_runtime", &[])?)
    })?;
    let (selected, built) = build_phase("select_finalist", || {
        selection::select_graph_finalist(
            graph,
            &planning,
            config.tile_mapping.as_deref(),
            |selected| {
                let low = &selected.program;
                let kernel_plan =
                    build_phase("plan_kernels", || Ok(KernelBuildPlan::from_program(low)?))?;
                let objects = build_phase("compile_kernels", || {
                    let mut objects = vec![fs::read(&runtime_artifact.object)?];
                    for compilation in &kernel_plan.compilations {
                        let artifact = config.toolchain.compile(
                            config.kernel_source_directory.join(compilation.source),
                            &compilation.name,
                            &compilation.flags,
                        )?;
                        objects.push(fs::read(&artifact.object)?);
                    }
                    Ok(objects)
                })?;
                let built =
                    build_package_from_objects(selected, &planning, &objects, &kernel_plan)?;
                Ok(built)
            },
        )
    })?;
    Ok((built, selected.program))
}

fn build_package_from_objects(
    selected: &mut ScheduledPlan,
    config: &PipelineConfig,
    objects: &[Vec<u8>],
    kernel_plan: &KernelBuildPlan,
) -> PackageBuildResult<BuiltApplication> {
    let program = &selected.program;
    let provisional_placement = &selected.placement;
    let provisional_exchanges = &selected.phases;
    let exchange_cache = &mut selected.cache;
    let topology = active_topology(program.tile_count)?;
    let retained_runtime = runtime_retained_symbols(program, config);
    let layout = build_phase("link_runtime", || {
        link_runtime(
            objects,
            runtime_symbols(0, 0, 0)?,
            kernel_plan,
            &retained_runtime,
        )
    })?;
    let linked_end = linked_end(&layout)?;
    let mut memory = TileMemoryMap::new();
    reserve_linked_image(&mut memory, &layout, "linked runtime and kernels")?;
    memory.reserve(
        "host exchange aperture",
        ipu_exchange::EXCHANGE_WINDOW_BASE
            ..ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES,
    )?;
    memory.reserve(
        "runtime state",
        RUNTIME_STATE_BASE..RUNTIME_EXECUTABLE_START,
    )?;

    let execution_tile_count = u16::try_from(Topology::c600().tile_count())?;
    let exchange_table_bytes = crate::tile::compact_exchange_table_bytes(
        provisional_exchanges,
        execution_tile_count,
        program.tile_count,
    )?;
    selection::check_exchange_budget(u64::from(exchange_table_bytes), config)?;
    let profile_samples = config.profiling.then(|| {
        program
            .tiles
            .iter()
            .map(|tile| profile_step_count(program, tile))
            .max()
            .unwrap_or(0)
            .max(program.exchange_phases.len())
            + 1
    });
    let profile_storage = profile_samples
        .map(|samples| -> PackageBuildResult<_> {
            let bytes = u32::try_from(samples)?
                .checked_mul(4)
                .ok_or_else(|| invalid("profile storage size overflow"))?;
            Ok(memory.allocate(MemoryRequest {
                name: "cycle profile samples",
                bytes,
                // Profile samples are transferred to the host while the tile
                // executes its host-readback program. Those accesses must not
                // share a standard-memory element with instruction fetch.
                alignment: ipu_package::TILE_MEMORY_ELEMENT_SIZE,
                bounds: crate::IPU21_DATA_BASE..ipu_package::IPU21_INTERLEAVED_MEMORY_BASE,
                end_alignment: ipu_package::TILE_MEMORY_ELEMENT_SIZE,
                guard_after: 0,
            })?)
        })
        .transpose()?;
    let exchange_rows = (exchange_table_bytes != 0)
        .then(|| {
            memory.allocate(MemoryRequest {
                name: "exchange row tables",
                bytes: exchange_table_bytes,
                // Executed exchange rows may not share an SRAM element with
                // any transfer source or destination. Reserve whole elements
                // at both ends so storage placement cannot use a prefix of the
                // row table's first element.
                alignment: ipu_package::TILE_MEMORY_ELEMENT_SIZE,
                bounds: crate::IPU21_DATA_BASE..ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT,
                end_alignment: ipu_package::TILE_MEMORY_ELEMENT_SIZE,
                guard_after: ipu_package::IPU21_SUPERVISOR_FETCH_LOOKAHEAD,
            })
        })
        .transpose()?;
    let exchange_code_base = exchange_rows
        .as_ref()
        .map_or(crate::IPU21_DATA_BASE, |allocation| allocation.range.start);
    let execution_topology = Topology::c600();
    let mut physical_to_logical = vec![None; usize::from(execution_tile_count)];
    for logical in 0..execution_tile_count {
        let physical = execution_topology.physical(logical)?;
        physical_to_logical[usize::from(physical)] = Some(logical);
    }
    let physical_to_logical = physical_to_logical
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| invalid("execution topology does not cover every physical tile"))?;
    let provisional_bindings = PackageBindings::new(
        program,
        provisional_placement,
        &topology,
        &physical_to_logical,
        profile_storage.as_ref().map(|storage| storage.range.start),
    )?;
    let sizing_host_base = memory.next_free(
        linked_end,
        TILE_MEMORY_BASE..ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT,
        4,
        "host programs",
    )?;
    let provisional_auxiliary_ranges = auxiliary_ranges(
        program,
        provisional_placement,
        &topology,
        execution_tile_count,
        &[(
            crate::IPU21_DATA_BASE,
            TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
        )],
    )?;
    let provisional_host = host::plan(
        &provisional_bindings.weights,
        &provisional_bindings.inputs,
        &provisional_bindings.outputs,
        execution_tile_count,
        sizing_host_base,
        &provisional_auxiliary_ranges,
    )?;
    let host_code_bytes = provisional_host
        .end
        .checked_sub(sizing_host_base)
        .ok_or_else(|| invalid("host program size underflow"))?;
    let host_code = (host_code_bytes != 0)
        .then(|| {
            memory.allocate(MemoryRequest {
                name: "host programs",
                bytes: host_code_bytes,
                alignment: 8,
                bounds: linked_end..ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT,
                end_alignment: 8,
                // The supervisor can fetch beyond the final host row while
                // exchange is active, including into the next SRAM element.
                guard_after: ipu_package::IPU21_SUPERVISOR_FETCH_LOOKAHEAD,
            })
        })
        .transpose()?;
    let host_code_base = host_code
        .as_ref()
        .map_or(sizing_host_base, |code| code.range.start);
    let provisional_host = host::plan(
        &provisional_bindings.weights,
        &provisional_bindings.inputs,
        &provisional_bindings.outputs,
        execution_tile_count,
        host_code_base,
        &provisional_auxiliary_ranges,
    )?;
    let symbols = layout
        .symbols
        .clone()
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let provisional_finalizer = TileProgramLowering::new(
        program,
        provisional_placement,
        provisional_exchanges,
        kernel_plan,
        exchange_code_base,
        execution_tile_count,
        false,
    )?;
    let sizing_code_address = memory.next_free(
        host_code_base + host_code_bytes,
        TILE_MEMORY_BASE..ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT,
        4,
        "generated tile programs",
    )?;
    let generated_code_bytes = build_phase("size_tile_code", || {
        physical_to_logical
            .par_iter()
            .enumerate()
            .map(|(physical, &logical)| {
                let host = &provisional_host.programs[physical];
                let mut tile_program = provisional_finalizer.lower_tile(logical)?;
                if let Some(storage) = &profile_storage {
                    instrument_profile(
                        program,
                        provisional_exchanges,
                        logical,
                        u32::try_from(physical)?,
                        &mut tile_program,
                        storage.range.start,
                    )?;
                }
                // Row sharing can change after final placement. Reserve its
                // optional setup call through the same emitter used below.
                reserve_exchange_setup(&mut tile_program.steps);
                let generated = emit(
                    &tile_program,
                    &symbols,
                    host,
                    &CodegenOptions {
                        code_address: sizing_code_address,
                        initial_profile_address: config.profiling.then_some(PROFILE_START_CYCLE),
                        final_profile_address: config.profiling.then_some(PROFILE_END_CYCLE),
                        ..CodegenOptions::default()
                    },
                )?;
                Ok::<_, PackageBuildError>(u32::try_from(generated.bytes.len())?)
            })
            .collect::<PackageBuildResult<Vec<_>>>()?
            .into_iter()
            .max()
            .ok_or_else(|| invalid("execution topology has no tiles"))
    })?;
    let code_address = if generated_code_bytes == 0 {
        sizing_code_address
    } else {
        memory
            .allocate(MemoryRequest {
                name: "generated tile programs",
                bytes: generated_code_bytes,
                alignment: 4,
                bounds: (host_code_base + host_code_bytes)
                    ..ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT,
                // Host programs and generated tile code form one contiguous
                // executable region. Keep planned standard-memory values out
                // of its final memory element.
                end_alignment: ipu_package::TILE_MEMORY_ELEMENT_SIZE,
                guard_after: 0,
            })?
            .range
            .start
    };
    let standard_ranges =
        memory.free_ranges(crate::IPU21_DATA_BASE..ipu_package::IPU21_INTERLEAVED_MEMORY_BASE);
    tracing::info!(
        linked_end,
        profile_bytes = profile_storage
            .as_ref()
            .map_or(0, |allocation| allocation.range.len()),
        exchange_table_bytes,
        host_code_bytes,
        generated_code_bytes,
        code_address,
        ?standard_ranges,
        "allocated package support memory"
    );
    let placement = build_phase("place_storage", || {
        Ok(crate::place::place_with_standard_ranges(
            program,
            &standard_ranges,
        )?)
    })?;
    let lowered_exchanges = build_phase("lower_exchanges", || {
        Ok(crate::exchange::lower_exchanges_cached(
            program,
            &placement,
            &topology,
            config.exchange_diagnostics,
            exchange_cache,
        )?)
    })?;
    let (placement, lowered_exchanges) = build_phase("optimize_exchange_placement", || {
        placement::improve_exchange_placement(
            program,
            &standard_ranges,
            &topology,
            placement,
            lowered_exchanges,
            exchange_rows.as_ref().map_or(0, |storage| {
                storage.reserved.end
                    - ipu_package::IPU21_SUPERVISOR_FETCH_LOOKAHEAD
                    - storage.range.start
            }),
            exchange_cache,
        )
    })?;
    let final_cost =
        crate::estimate::scheduled_program_cycles(&program.program, &lowered_exchanges.phases)?;
    tracing::info!(
        final_cycles = final_cost.total,
        final_exchange = final_cost.exchange,
        "costed final placed program"
    );
    let exchange_schedule = lowered_exchanges.schedule_snapshot;
    let exchanges = lowered_exchanges.phases;
    let PackageBindings {
        inputs,
        weights,
        outputs,
    } = PackageBindings::new(
        program,
        &placement,
        &topology,
        &physical_to_logical,
        profile_storage.as_ref().map(|storage| storage.range.start),
    )?;
    let mut inactive_auxiliary_ranges = standard_ranges.clone();
    inactive_auxiliary_ranges.push((
        ipu_package::IPU21_INTERLEAVED_MEMORY_BASE,
        TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
    ));
    let host = host::plan(
        &weights,
        &inputs,
        &outputs,
        execution_tile_count,
        host_code_base,
        &auxiliary_ranges(
            program,
            &placement,
            &topology,
            execution_tile_count,
            &inactive_auxiliary_ranges,
        )?,
    )?;
    let final_host_code_bytes = host
        .end
        .checked_sub(host_code_base)
        .ok_or_else(|| invalid("host program end precedes its base"))?;
    if final_host_code_bytes > host_code_bytes {
        return Err(invalid(format!(
            "host program grew after tensor placement: reserved {host_code_bytes}, requires {final_host_code_bytes} bytes"
        )));
    }
    let finalizer = TileProgramLowering::new(
        program,
        &placement,
        &exchanges,
        kernel_plan,
        exchange_code_base,
        execution_tile_count,
        true,
    )?;
    if let Some(storage) = &exchange_rows {
        // Placement can change row sharing and instruction alignment. The
        // entire SRAM element is already excluded from tensor storage, so
        // let final rows use its padding while retaining fetch look-ahead.
        let capacity_end = storage.reserved.end - ipu_package::IPU21_SUPERVISOR_FETCH_LOOKAHEAD;
        tracing::debug!(
            planned_bytes = storage.range.len(),
            final_bytes = finalizer.exchange_code_end() - storage.range.start,
            capacity_bytes = capacity_end - storage.range.start,
            "checked final exchange table capacity"
        );
        selection::check_exchange_budget(
            u64::from(finalizer.exchange_code_end() - storage.range.start),
            config,
        )?;
        if finalizer.exchange_code_end() > capacity_end {
            return Err(invalid(format!(
                "final exchange rows require {} bytes; planned {}, reserved capacity {}",
                finalizer.exchange_code_end() - storage.range.start,
                storage.range.len(),
                capacity_end - storage.range.start,
            )));
        }
    }
    let prepared = build_phase("prepare_tile_code", || {
        physical_to_logical
            .par_iter()
            .enumerate()
            .map(|(physical_tile, &logical)| {
                let mut tile_program = finalizer.lower_tile(logical)?;
                let profile = profile_storage
                    .as_ref()
                    .map(|storage| {
                        instrument_profile(
                            program,
                            &exchanges,
                            logical,
                            u32::try_from(physical_tile)?,
                            &mut tile_program,
                            storage.range.start,
                        )
                    })
                    .transpose()?;
                Ok((tile_program, profile))
            })
            .collect::<PackageBuildResult<Vec<_>>>()
    })?;
    let generate = || {
        prepared
            .iter()
            .zip(&host.programs)
            .map(|((tile_program, _), host)| {
                Ok(emit(
                    tile_program,
                    &symbols,
                    host,
                    &CodegenOptions {
                        code_address,
                        initial_profile_address: config.profiling.then_some(PROFILE_START_CYCLE),
                        final_profile_address: config.profiling.then_some(PROFILE_END_CYCLE),
                        ..CodegenOptions::default()
                    },
                )?)
            })
            .collect::<PackageBuildResult<Vec<_>>>()
    };
    let generated = build_phase("emit_tile_code", generate)?;
    let actual_code_bytes = generated.iter().try_fold(0u32, |maximum, program| {
        Ok::<_, PackageBuildError>(maximum.max(u32::try_from(program.bytes.len())?))
    })?;
    if actual_code_bytes > generated_code_bytes {
        return Err(invalid(format!(
            "generated tile code requires {actual_code_bytes} bytes; reserved {generated_code_bytes}"
        )));
    }
    let profile_tiles = prepared
        .into_iter()
        .filter_map(|(_, profile)| profile)
        .filter(|tile| !tile.steps.is_empty())
        .collect::<Vec<_>>();

    let tile_build = TileBuildContext {
        objects,
        kernel_plan,
        retained_runtime: &retained_runtime,
        code_address,
        host_staging_address: host.staging_address,
    };
    let tiles = build_phase("build_tile_images", || {
        (0..execution_tile_count)
            .map(|physical_tile| {
                build_tile(
                    u32::from(physical_tile),
                    u32::from(physical_to_logical[usize::from(physical_tile)]),
                    &generated[usize::from(physical_tile)],
                    &host.segments[usize::from(physical_tile)],
                    &tile_build,
                )
            })
            .collect::<PackageBuildResult<Vec<_>>>()
    })?;
    let mut application = Application {
        tiles,
        ..Application::default()
    };
    add_linked_debug_map(&mut application, &layout)?;
    for (physical, program) in generated.iter().enumerate() {
        add_generated_debug_map(
            &mut application,
            u32::try_from(physical)?,
            code_address,
            program,
        )?;
        for segment in &host.segments[physical] {
            if segment.flags & SEGMENT_EXECUTE != 0 && segment.memory_size != 0 {
                application.debug_regions.push(DebugRegion {
                    physical_tile: u32::try_from(physical)?,
                    address: segment.address,
                    size: segment.memory_size,
                    name: "host exchange program".into(),
                });
            }
        }
    }
    application
        .tiles
        .sort_unstable_by_key(|tile| tile.physical_tile);
    application.inputs = inputs;
    application.weights = weights;
    application.outputs = outputs;
    application.profile_tiles = profile_tiles;
    application.outputs.push(Binding {
        name: "completion".into(),
        dtype: "u32".into(),
        shape: vec![1],
        slices: vec![RegionSlice {
            tile: 0,
            tile_address: COMPLETION_ADDRESS,
            file_offset: 0,
            size: 4,
        }],
    });
    application.entry_points.push(EntryPoint {
        name: "run".into(),
        command: 0,
        external_syncs: 0,
    });
    application.host_exchange = host.protocol;
    application.validate()?;
    Ok(BuiltApplication {
        application,
        placement,
        exchange_phases: exchanges,
        exchange_schedule,
        exchange_code_base,
    })
}

/// Sizing-only patches account for exchange rows which become structurally
/// shareable after addresses and schedules change. All addresses/counts fit
/// one SETZI; the payload itself is already covered by exchange-row sizing.
fn reserve_exchange_setup(steps: &mut [crate::TileStep]) {
    for step in steps {
        match step {
            crate::TileStep::Exchange(exchange)
                if exchange.active && exchange.setup_patch.is_none() =>
            {
                exchange.setup_patch = Some(crate::ExchangeSetupPatch {
                    offsets: crate::PlacedExchangeRow {
                        address: 0,
                        words: vec![0],
                    },
                    values: crate::PlacedExchangeRow {
                        address: 4,
                        words: vec![0],
                    },
                });
            }
            crate::TileStep::Repeat(repeat) => reserve_exchange_setup(&mut repeat.body),
            _ => {}
        }
    }
}

fn diagnostic_tensor(
    low: &LowProgram,
    placement: &crate::Placement,
    topology: &Topology,
    value: crate::MidValueId,
    name: Option<String>,
) -> PackageBuildResult<DiagnosticTensor> {
    let mid_value = low
        .logical_values
        .get(value.index() as usize)
        .ok_or_else(|| invalid("diagnostic mid-level value is missing"))?;
    let low_value = low.values.iter().find(|candidate| candidate.value == value);
    let shards = if let Some(low_value) = low_value {
        low_value
            .shards
            .iter()
            .filter(|id| {
                low.shards
                    .get(id.index() as usize)
                    .is_some_and(|shard| shard.definition != crate::ShardDefinition::Unmaterialized)
            })
            .map(|id| {
                let storage = low
                    .shards
                    .get(id.index() as usize)
                    .ok_or_else(|| invalid("diagnostic low-level shard is missing"))?;
                let address = placement
                    .shard_addresses
                    .get(id)
                    .copied()
                    .ok_or_else(|| invalid("diagnostic shard placement is missing"))?;
                Ok(DiagnosticShard {
                    physical_tile: topology.physical(storage.tile)?,
                    address,
                    storage: storage.clone(),
                })
            })
            .collect::<PackageBuildResult<Vec<_>>>()?
    } else {
        Vec::new()
    };
    Ok(DiagnosticTensor {
        name,
        value: mid_value.origin,
        shape: mid_value.tensor_type.shape.clone(),
        precision: mid_value.tensor_type.format.precision,
        shards,
    })
}

fn add_linked_debug_map(
    application: &mut Application,
    linked: &LinkedImage,
) -> PackageBuildResult<()> {
    for segment in &linked.segments {
        application.debug_regions.push(DebugRegion {
            physical_tile: DEBUG_ALL_TILES,
            address: segment.address,
            size: u32::try_from(segment.size)?,
            name: "linked executable".into(),
        });
    }
    application.debug_symbols.extend(
        linked
            .symbols
            .iter()
            .filter(|(_, address)| {
                linked.segments.iter().any(|segment| {
                    (segment.address..segment.address.saturating_add(segment.size as u32))
                        .contains(address)
                })
            })
            .map(|(name, &address)| DebugSymbol {
                name: name.clone(),
                address,
            }),
    );
    application
        .debug_symbols
        .sort_unstable_by_key(|symbol| symbol.address);
    Ok(())
}

fn add_generated_debug_map(
    application: &mut Application,
    physical_tile: u32,
    code_address: u32,
    generated: &crate::GeneratedProgram,
) -> PackageBuildResult<()> {
    if !generated.bytes.is_empty() {
        application.debug_regions.push(DebugRegion {
            physical_tile,
            address: code_address,
            size: u32::try_from(generated.bytes.len())?,
            name: "generated tile program".into(),
        });
    }
    for row in &generated.exchange_rows {
        if !row.words.is_empty() {
            application.debug_regions.push(DebugRegion {
                physical_tile,
                address: row.address,
                size: u32::try_from(row.words.len())?
                    .checked_mul(4)
                    .ok_or_else(|| invalid("exchange debug range overflow"))?,
                name: "exchange row".into(),
            });
        }
    }
    Ok(())
}

fn build_phase<T>(
    phase: &'static str,
    build: impl FnOnce() -> PackageBuildResult<T>,
) -> PackageBuildResult<T> {
    let span = tracing::info_span!("ipu_codegen.package.phase", phase);
    let _entered = span.enter();
    let started = Instant::now();
    let result = build();
    tracing::info!(
        phase,
        elapsed_ms = started.elapsed().as_millis() as u64,
        success = result.is_ok(),
        "package build phase finished"
    );
    result
}

fn validate_tile_count(tile_count: u32) -> PackageBuildResult<()> {
    let maximum = Topology::c600().tile_count() as u32;
    if tile_count == 0 || !tile_count.is_multiple_of(TILES_PER_BATCH as u32) || tile_count > maximum
    {
        return Err(invalid(format!(
            "tile count must be a nonzero multiple of {TILES_PER_BATCH} and at most {maximum}"
        )));
    }
    Ok(())
}

fn active_topology(tile_count: u16) -> PackageBuildResult<Topology> {
    Ok(Topology::new(
        (0..tile_count)
            .map(ipu_exchange::c600_logical_to_physical)
            .collect(),
    )?)
}

struct TileBuildContext<'a> {
    objects: &'a [Vec<u8>],
    kernel_plan: &'a KernelBuildPlan,
    retained_runtime: &'a [String],
    code_address: u32,
    host_staging_address: u32,
}

fn build_tile(
    physical_tile: u32,
    logical_tile: u32,
    generated: &crate::GeneratedProgram,
    host_segments: &[Segment],
    context: &TileBuildContext<'_>,
) -> PackageBuildResult<TileImage> {
    let linked = link_runtime(
        context.objects,
        runtime_symbols(
            logical_tile,
            context.code_address,
            context.host_staging_address,
        )?,
        context.kernel_plan,
        context.retained_runtime,
    )?;
    let mut entry = Vec::with_capacity(ENTRY_BYTES as usize);
    entry.extend_from_slice(&encode_setzi_m(0, linked.entry)?.to_le_bytes());
    entry.extend_from_slice(&encode_br_m(0)?.to_le_bytes());
    let mut segments = vec![Segment {
        address: APPLICATION_LOAD_BASE,
        memory_size: ENTRY_BYTES,
        data: entry,
        flags: SEGMENT_READ | SEGMENT_EXECUTE,
    }];
    segments.extend(linked.segments.iter().map(|segment| Segment {
        address: segment.address,
        memory_size: segment.size as u32,
        data: linked.bytes[segment.offset..segment.offset + segment.size].to_vec(),
        flags: SEGMENT_READ | SEGMENT_EXECUTE,
    }));
    let mut exchange_rows = BTreeMap::<u32, Vec<u8>>::new();
    for row in &generated.exchange_rows {
        let bytes = row
            .words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        if exchange_rows.insert(row.address, bytes).is_some() {
            return Err(invalid(
                "duplicate exchange-row address in one tile program",
            ));
        }
    }
    segments.extend(exchange_rows.into_iter().map(|(address, data)| Segment {
        address,
        memory_size: data.len() as u32,
        data,
        flags: SEGMENT_READ | SEGMENT_EXECUTE,
    }));
    segments.extend_from_slice(host_segments);
    segments.push(Segment {
        address: context.code_address,
        memory_size: generated.bytes.len() as u32,
        data: generated.bytes.clone(),
        flags: SEGMENT_READ | SEGMENT_EXECUTE,
    });
    segments.push(Segment {
        address: COMPLETION_ADDRESS,
        memory_size: RUNTIME_STATE_BYTES,
        data: vec![0; 4],
        flags: SEGMENT_READ | SEGMENT_WRITE,
    });
    Ok(TileImage {
        physical_tile,
        entry_point: APPLICATION_LOAD_BASE,
        command_address: 0,
        diagnostic_address: COMPLETION_ADDRESS,
        segments,
    })
}

fn link_runtime(
    objects: &[Vec<u8>],
    externals: HashMap<String, u32>,
    kernel_plan: &KernelBuildPlan,
    retained_runtime: &[String],
) -> PackageBuildResult<LinkedImage> {
    let mut retained_symbols = retained_runtime.to_vec();
    retained_symbols.extend(kernel_plan.retained_symbols().map(str::to_owned));
    Ok(link(
        objects,
        &LinkOptions {
            image_base: TILE_MEMORY_BASE,
            regions: vec![
                (SUPPORT_START, ipu_exchange::EXCHANGE_WINDOW_BASE),
                (
                    RUNTIME_EXECUTABLE_START,
                    ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT,
                ),
            ],
            entry_symbol: RUNTIME_ENTRY_SYMBOL.into(),
            retained_symbols,
            externals,
        },
    )?)
}

fn runtime_retained_symbols(program: &LowProgram, config: &PipelineConfig) -> Vec<String> {
    let mut symbols = vec![COMPLETE_SYMBOL.into()];
    if !program.exchange_phases.is_empty() {
        symbols.push(WORKER_BARRIER_SYMBOL.into());
        symbols.push(crate::PATCH_ROW_SYMBOL.into());
        if !program.repeat_runs.is_empty() {
            symbols.push(crate::PATCH_WORD_SYMBOL.into());
        }
    }
    if config.profiling {
        symbols.push(SAMPLE_CYCLE_SYMBOL.into());
    }
    if !program.inputs.is_empty() || !program.outputs.is_empty() {
        symbols.push(crate::HOST_RUN_SYMBOL.into());
        symbols.push(crate::REPEAT_CALL_SYMBOL.into());
    }
    #[derive(Default)]
    struct CopySymbols {
        local: bool,
        halfword: bool,
        zero: bool,
    }
    let mut copies = CopySymbols::default();
    fn collect(program: &LowProgram, tile: &crate::TileWorkList, copies: &mut CopySymbols) {
        for work in program.work(tile) {
            match work {
                crate::TileWorkRef::LocalCopy(copy) => {
                    copies.local = true;
                    copies.halfword |= crate::tile::local_copy_call(copy)
                        .is_some_and(|(symbol, _)| symbol == crate::COPY_U16_SYMBOL);
                }
                crate::TileWorkRef::Kernel(run) => {
                    copies.zero |= matches!(run.kernel, crate::TileKernelSpec::FillZero { .. });
                }
                crate::TileWorkRef::Repeat(repeat) => collect(program, &repeat.body, copies),
                _ => {}
            }
        }
    }
    for tile in &program.tiles {
        collect(program, tile, &mut copies);
    }
    if copies.halfword {
        symbols.push(crate::COPY_U16_SYMBOL.into());
    }
    if copies.local {
        symbols.extend(
            [
                crate::COPY_U32_SYMBOL,
                crate::COPY_U64_SYMBOL,
                crate::COPY_STRIDED_U64_SYMBOL,
                crate::COPY_STRIDED_U32_SYMBOL,
            ]
            .map(String::from),
        );
    }
    if copies.zero {
        symbols.push(crate::FILL_ZERO_U64_SYMBOL.into());
    }
    symbols
}

fn runtime_symbols(
    physical_tile: u32,
    program_address: u32,
    host_staging_address: u32,
) -> PackageBuildResult<HashMap<String, u32>> {
    let sync_context = physical_tile
        .checked_mul(8)
        .ok_or_else(|| invalid("tile index overflow"))?;
    let prng_seed = physical_tile
        .checked_add(1)
        .and_then(|value| value.checked_mul(8))
        .ok_or_else(|| invalid("PRNG seed overflow"))?;
    Ok(HashMap::from([
        (WORKER_SYNC_CONTEXT_SYMBOL.into(), sync_context),
        (
            WORKER_STACK_BASE_SYMBOL.into(),
            COMPLETION_ADDRESS + WORKER_STACK_HEADROOM,
        ),
        (PRNG_SEED_SYMBOL.into(), prng_seed),
        (PROGRAM_ADDRESS_SYMBOL.into(), program_address),
        (COMPLETION_ADDRESS_SYMBOL.into(), COMPLETION_ADDRESS),
        (crate::HOST_STAGING_SYMBOL.into(), host_staging_address),
    ]))
}

fn linked_end(linked: &LinkedImage) -> PackageBuildResult<u32> {
    linked
        .segments
        .iter()
        .map(|segment| segment.address.checked_add(segment.size as u32))
        .collect::<Option<Vec<_>>>()
        .and_then(|ends| ends.into_iter().max())
        .ok_or_else(|| invalid("linked runtime has no valid segments"))
}

fn reserve_linked_image(
    memory: &mut TileMemoryMap,
    linked: &LinkedImage,
    name: &'static str,
) -> PackageBuildResult<()> {
    for segment in &linked.segments {
        let end = segment
            .address
            .checked_add(u32::try_from(segment.size)?)
            .ok_or_else(|| invalid("linked runtime segment range overflow"))?;
        memory.reserve(name, segment.address..end)?;
    }
    // Instruction fetch conflicts with writes to the same memory element.
    // Linked sections may leave small holes and a partial final element; none
    // of those bytes can be handed to tensor or host-command storage.
    let element = ipu_package::TILE_MEMORY_ELEMENT_SIZE;
    for segment in &linked.segments {
        let start = (segment.address / element * element).max(RUNTIME_EXECUTABLE_START);
        let end = segment.address + u32::try_from(segment.size)?;
        let end = end.div_ceil(element) * element;
        for (start, end) in memory.free_ranges(start..end) {
            memory.reserve("linked executable memory elements", start..end)?;
        }
    }
    Ok(())
}

pub(crate) fn invalid(message: impl Into<String>) -> PackageBuildError {
    PackageBuildError::Invalid(message.into())
}

/// Capture address-resolved ordinary transfers before scheduling or linking.
/// Finalist indices match package-selection diagnostics; failed placements remain errors.
pub fn capture_exchange_finalist(
    graph: &ComputeGraph,
    config: &PackageConfig,
    finalist: usize,
) -> PackageBuildResult<crate::ExchangeScheduleSnapshot> {
    let planning = &config.pipeline;
    validate_tile_count(u32::from(planning.tile_count))?;
    let finalists = lower_finalists(
        graph,
        planning,
        &Ipu21CostModel,
        planning
            .expanded_plan_finalists
            .max(planning.exchange_schedule_finalists),
    )?;
    let mid = finalists
        .get(finalist)
        .ok_or_else(|| invalid("exchange capture finalist out of range"))?;
    let (low, placement, _) =
        selection::expand_and_place(mid, planning, config.tile_mapping.as_deref())?;
    Ok(crate::exchange::capture_exchange_schedule(
        &low, &placement,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linked_sections_protect_their_complete_memory_elements() {
        let base = RUNTIME_EXECUTABLE_START;
        let element = ipu_package::TILE_MEMORY_ELEMENT_SIZE;
        let linked = LinkedImage {
            base,
            entry: base,
            bytes: vec![],
            segments: vec![
                ipu_elf::LinkedSegment {
                    address: base,
                    offset: 0,
                    size: 128,
                },
                ipu_elf::LinkedSegment {
                    address: base + 256,
                    offset: 128,
                    size: 128,
                },
                ipu_elf::LinkedSegment {
                    address: base + element,
                    offset: 256,
                    size: 128,
                },
            ],
            symbols: BTreeMap::new(),
        };
        let mut memory = TileMemoryMap::new();
        reserve_linked_image(&mut memory, &linked, "test code").unwrap();
        assert_eq!(
            memory.free_ranges(base..base + 3 * element),
            vec![(base + 2 * element, base + 3 * element)]
        );
        // Rounding executable sections does not consume permanent runtime state.
        memory
            .reserve("runtime state", RUNTIME_STATE_BASE..base)
            .unwrap();
    }

    #[test]
    fn attention_scratch_does_not_override_result_precision() {
        let mut graph = ComputeGraph::new();
        let q = graph.host_input("q", [2, 4, 16]).unwrap();
        let k = graph.host_input("k", [2, 4, 16]).unwrap();
        let v = graph.host_input("v", [2, 4, 16]).unwrap();
        let y = graph.flash_attention(q, k, v).unwrap();
        graph.set_outputs([y]).unwrap();
        let config = PipelineConfig::new(8)
            .with_automatic_input(q, Precision::F16)
            .with_automatic_input(k, Precision::F16)
            .with_automatic_input(v, Precision::F16);
        let mid = lower_finalists(&graph, &config, &Ipu21CostModel, 1)
            .unwrap()
            .remove(0);
        let low = crate::low::expand::expand_tiles(&mid, false).unwrap();
        assert!(
            low.logical_values
                .iter()
                .any(|value| value.origin == y
                    && value.tensor_type.format.precision == Precision::F32)
        );
        assert_eq!(package_precisions(&low)[&y], Precision::F16);
    }
}
