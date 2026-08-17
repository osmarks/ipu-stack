use crate::emitter::{
    COMPLETE_SYMBOL, COMPLETION_ADDRESS_SYMBOL, CodegenError, CodegenOptions, GeneratedProgram,
    HOST_RUN_SYMBOL, HOST_STAGING_SYMBOL, PATCH_ROW_SYMBOL, PATCH_WORD_SYMBOL, PRNG_SEED_SYMBOL,
    PROGRAM_ADDRESS_SYMBOL, REPEAT_CALL_SYMBOL, RUNTIME_ENTRY_SYMBOL, SAMPLE_CYCLE_SYMBOL,
    WORKER_BARRIER_SYMBOL, WORKER_STACK_BASE_SYMBOL, WORKER_SYNC_CONTEXT_SYMBOL, emit,
};
use crate::graph::{ComputeGraph, OperationId, ValueId};
use crate::host;
use crate::ir::{MidGraph, MidOperationKind};
use crate::low::LowProgram;
use crate::memory::{
    MemoryLayoutError, MemoryRequest, PROFILE_END_CYCLE, PROFILE_START_CYCLE, RUNTIME_STATE_BASE,
    RUNTIME_STATE_BYTES, TileMemoryMap, WORKER_STACK_HEADROOM,
};
use crate::mid::lower_finalists;
use crate::operator::Precision;
use crate::{
    KernelBuildPlan, PipelineConfig, TileProgram, TileProgramLowering, lower_exchanges,
    lower_to_tiles, place, shard_storage_bytes,
};
use ipu_driver::{APPLICATION_LOAD_BASE, TILES_PER_BATCH};
use ipu_elf::{ElfError, LinkOptions, LinkedImage, Toolchain, link};
use ipu_exchange::{ExchangeError, Topology, encode_br_m, encode_setzi_m};
use ipu_package::{
    AddressRegion, Application, Binding, DEBUG_ALL_TILES, DebugRegion, DebugSymbol, EntryPoint,
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
    #[error("code generation failed: {0}")]
    Codegen(#[from] CodegenError),
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
    Low(#[from] crate::LowLoweringError),
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
    pub inputs: Vec<CompiledTensor>,
    pub outputs: Vec<CompiledTensor>,
    pub precisions: BTreeMap<ValueId, Precision>,
    /// Exact physical exchange schedules retained for low-level diagnostics.
    /// This is build metadata and is not serialized into the application.
    pub exchange_phases: Vec<crate::PhysicalExchangePhase>,
    /// Address-resolved inputs to physical exchange scheduling and row codegen.
    pub exchange_schedule: crate::ExchangeScheduleSnapshot,
    /// Base address used when laying out the compact per-tile exchange table.
    pub exchange_code_base: u32,
    pub checkpoints: Vec<DiagnosticCheckpoint>,
    placement: crate::Placement,
}

#[derive(Clone, Debug)]
pub struct DiagnosticCheckpoint {
    pub operation: OperationId,
    pub breakpoint: u8,
    pub tensors: Vec<CompiledTensor>,
}

#[derive(Clone, Debug)]
pub struct CompiledTensor {
    pub name: Option<String>,
    pub value: ValueId,
    pub shape: crate::TensorShape,
    pub precision: Precision,
    pub shards: Vec<CompiledTensorShard>,
}

#[derive(Clone, Debug)]
pub struct CompiledTensorShard {
    pub physical_tile: u16,
    pub address: u32,
    pub storage: crate::LowShard,
}

/// Builds an application from address-resolved tile programs.
///
/// This is the low-level counterpart to [`build_package`]. It deliberately has
/// no tensor bindings: callers supply initialized tile data and inspect it
/// through driver diagnostics. A zero-payload `run` rendezvous starts execution
/// after loading, so breakpoints in the program cannot race the loader.
pub fn build_tile_program_package(
    programs: &[TileProgram],
    data: &[TileProgramData],
    outputs: &[Binding],
    toolchain: &Toolchain,
    runtime_source: &std::path::Path,
) -> PackageBuildResult<Application> {
    let topology = Topology::c600();
    let execution_tiles = u16::try_from(topology.tile_count())?;
    if programs.len() != usize::from(execution_tiles)
        || programs
            .iter()
            .enumerate()
            .any(|(tile, program)| usize::from(program.tile) != tile)
    {
        return Err(invalid(
            "finalized tile programs must cover every C600 logical tile in order",
        ));
    }
    if data
        .iter()
        .any(|segment| segment.tile >= execution_tiles || segment.data.is_empty())
    {
        return Err(invalid(
            "tile-program data has an invalid tile or empty payload",
        ));
    }

    let runtime_artifact = toolchain.compile(runtime_source, "static_runtime", &[])?;
    let objects = vec![fs::read(runtime_artifact.object)?];
    let kernels = KernelBuildPlan::default();
    let mut retained_runtime = vec![
        COMPLETE_SYMBOL.into(),
        HOST_RUN_SYMBOL.into(),
        REPEAT_CALL_SYMBOL.into(),
        WORKER_BARRIER_SYMBOL.into(),
    ];
    for program in programs {
        collect_compute_symbols(&mut retained_runtime, &program.steps);
    }
    retained_runtime.sort_unstable();
    retained_runtime.dedup();
    let layout = link_runtime(
        &objects,
        runtime_symbols(0, 0, 0)?,
        &kernels,
        &retained_runtime,
    )?;
    let symbols = layout
        .symbols
        .clone()
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let linked_end = linked_end(&layout)?;
    let mut memory = TileMemoryMap::new();
    reserve_linked_image(&mut memory, &layout, "linked runtime")?;
    memory.reserve(
        "host exchange aperture",
        AddressRegion::new(
            ipu_exchange::EXCHANGE_WINDOW_BASE,
            ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES,
        ),
    )?;
    memory.reserve(
        "runtime state",
        AddressRegion::new(RUNTIME_STATE_BASE, RUNTIME_EXECUTABLE_START),
    )?;
    let mut tile_data = vec![Vec::<(u32, u32)>::new(); usize::from(execution_tiles)];
    for segment in data {
        let bytes = u32::try_from(segment.data.len())?;
        let end = segment
            .address
            .checked_add(bytes)
            .ok_or_else(|| invalid("tile data range overflow"))?;
        tile_data[usize::from(segment.tile)].push((segment.address, end));
    }
    let mut tile_rows = vec![Vec::<(u32, u32)>::new(); usize::from(execution_tiles)];
    for program in programs {
        let mut rows = BTreeMap::new();
        collect_exchange_rows(&mut rows, &program.steps)?;
        tile_rows[usize::from(program.tile)].extend(rows);
    }
    for tile in 0..execution_tiles {
        for &(data_start, data_end) in &tile_data[usize::from(tile)] {
            if let Some(&(row_start, row_end)) = tile_rows[usize::from(tile)]
                .iter()
                .find(|&&(row_start, row_end)| data_start < row_end && row_start < data_end)
            {
                return Err(invalid(format!(
                    "tile {tile} data at 0x{data_start:x}..0x{data_end:x} overlaps exchange row 0x{row_start:x}..0x{row_end:x}"
                )));
            }
        }
    }
    // Generated and linked code use common addresses on every tile, so choose
    // them against the union of tile-local data and row ranges. Data on one
    // tile may otherwise legally share an address with a row on another tile.
    let mut tile_local_ranges = tile_data
        .into_iter()
        .chain(tile_rows)
        .flatten()
        .collect::<Vec<_>>();
    tile_local_ranges.sort_unstable();
    let mut merged_tile_local = Vec::<(u32, u32)>::new();
    for (start, end) in tile_local_ranges {
        if let Some((_, previous_end)) = merged_tile_local.last_mut()
            && start <= *previous_end
        {
            *previous_end = (*previous_end).max(end);
        } else {
            merged_tile_local.push((start, end));
        }
    }
    for (start, end) in merged_tile_local {
        memory.reserve(
            "tile-local data or exchange rows",
            AddressRegion::new(start, end),
        )?;
    }

    let launch = Binding {
        name: "run-gate".into(),
        dtype: "u32".into(),
        shape: vec![1],
        slices: vec![RegionSlice {
            tile: u32::from(topology.physical(0)?),
            tile_address: COMPLETION_ADDRESS + 4,
            file_offset: 0,
            size: 4,
        }],
    };
    let finish = Binding {
        name: "run-finish".into(),
        dtype: "u32".into(),
        shape: vec![1],
        slices: vec![RegionSlice {
            tile: u32::from(topology.physical(0)?),
            tile_address: COMPLETION_ADDRESS + 8,
            file_offset: 0,
            size: 4,
        }],
    };
    let mut run_outputs = outputs.to_vec();
    run_outputs.push(finish);
    let host_bounds = AddressRegion::new(
        crate::IPU21_DATA_BASE,
        TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
    );
    let sizing_host_base = memory.next_free(
        linked_end,
        AddressRegion::new(TILE_MEMORY_BASE, ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT),
        8,
        "host programs",
    )?;
    let provisional_ranges = memory.free_ranges(host_bounds.clone());
    let provisional_host = host::plan(
        &[],
        std::slice::from_ref(&launch),
        &run_outputs,
        execution_tiles,
        sizing_host_base,
        &vec![provisional_ranges; usize::from(execution_tiles)],
    )?;
    let host_code_bytes = provisional_host
        .end
        .checked_sub(sizing_host_base)
        .ok_or_else(|| invalid("host program size underflow"))?;
    let host_code = memory.allocate(MemoryRequest {
        name: "host programs",
        bytes: host_code_bytes,
        alignment: 8,
        bounds: AddressRegion::new(linked_end, ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT),
        end_alignment: 8,
        guard_after: 0,
    })?;
    let host_ranges = memory.free_ranges(host_bounds.clone());
    let host = host::plan(
        &[],
        std::slice::from_ref(&launch),
        &run_outputs,
        execution_tiles,
        host_code.range.start,
        &vec![host_ranges; usize::from(execution_tiles)],
    )?;
    if host.end - host_code.range.start > host_code_bytes {
        return Err(invalid("host program grew after placement"));
    }
    let mut host_data_ranges = host
        .segments
        .iter()
        .flatten()
        .filter(|segment| segment.flags & SEGMENT_EXECUTE == 0)
        .map(|segment| (segment.address, segment.address + segment.memory_size))
        .collect::<Vec<_>>();
    host_data_ranges.sort_unstable();
    let mut merged_host_data = Vec::<(u32, u32)>::new();
    for (start, end) in host_data_ranges {
        if let Some((_, previous_end)) = merged_host_data.last_mut()
            && start <= *previous_end
        {
            *previous_end = (*previous_end).max(end);
        } else {
            merged_host_data.push((start, end));
        }
    }
    for (start, end) in merged_host_data {
        memory.reserve("host program data", AddressRegion::new(start, end))?;
    }

    let sizing_address = memory.next_free(
        host_code.range.end,
        AddressRegion::new(TILE_MEMORY_BASE, ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT),
        8,
        "generated tile programs",
    )?;
    let maximum_bytes = programs.iter().try_fold(0u32, |maximum, program| {
        let physical = topology.physical(program.tile)?;
        let generated = emit(
            program,
            &symbols,
            &host.programs[usize::from(physical)],
            &CodegenOptions {
                code_address: sizing_address,
                ..CodegenOptions::default()
            },
        )?;
        Ok::<_, PackageBuildError>(maximum.max(u32::try_from(generated.bytes.len())?))
    })?;
    let code_address = memory
        .allocate(MemoryRequest {
            name: "generated tile programs",
            bytes: maximum_bytes,
            alignment: 4,
            bounds: AddressRegion::new(linked_end, ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT),
            // Supervisor instruction fetch and exchange/paired memory access
            // cannot safely use the same standard-memory element. Reserve the
            // rest of the element so subsequently placed tensor data cannot
            // become the source of an exchange while code executes from it.
            end_alignment: ipu_package::TILE_MEMORY_ELEMENT_SIZE,
            guard_after: 0,
        })?
        .range
        .start;
    let generated = programs
        .iter()
        .map(|program| {
            let physical = topology.physical(program.tile)?;
            Ok(emit(
                program,
                &symbols,
                &host.programs[usize::from(physical)],
                &CodegenOptions {
                    code_address,
                    ..CodegenOptions::default()
                },
            )?)
        })
        .collect::<PackageBuildResult<Vec<_>>>()?;

    let mut segments = vec![Vec::new(); usize::from(execution_tiles)];
    for segment in data {
        let physical = topology.physical(segment.tile)?;
        segments[usize::from(physical)].push(Segment {
            address: segment.address,
            memory_size: u32::try_from(segment.data.len())?,
            data: segment.data.clone(),
            flags: SEGMENT_READ | SEGMENT_WRITE,
        });
    }
    for (physical, host_segments) in host.segments.iter().enumerate() {
        segments[physical].extend(host_segments.iter().cloned());
    }
    let context = TileBuildContext {
        objects: &objects,
        kernel_plan: &kernels,
        retained_runtime: &retained_runtime,
        code_address,
        host_staging_address: host.staging_address,
    };
    let mut tiles = Vec::with_capacity(usize::from(execution_tiles));
    for logical in 0..execution_tiles {
        let physical = topology.physical(logical)?;
        tiles.push(build_tile(
            u32::from(physical),
            u32::from(logical),
            &generated[usize::from(logical)],
            &segments[usize::from(physical)],
            &context,
        )?);
    }
    tiles.sort_unstable_by_key(|tile| tile.physical_tile);
    let mut application = Application {
        tiles,
        ..Application::default()
    };
    add_linked_debug_map(&mut application, &layout)?;
    for (logical, program) in generated.iter().enumerate() {
        let physical = u32::from(topology.physical(u16::try_from(logical)?)?);
        add_generated_debug_map(&mut application, physical, code_address, program)?;
    }
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
    application.outputs.extend(run_outputs);
    application.inputs.push(launch);
    application.entry_points.push(EntryPoint {
        name: "run".into(),
        command: 0,
        external_syncs: 0,
    });
    application.host_exchange = host.protocol;
    application.validate()?;
    Ok(application)
}

fn collect_exchange_rows(
    rows: &mut BTreeMap<u32, u32>,
    steps: &[crate::TileStep],
) -> PackageBuildResult<()> {
    for step in steps {
        match step {
            crate::TileStep::Exchange(exchange) => {
                let bytes = u32::try_from(exchange.program.words.len())?
                    .checked_mul(4)
                    .ok_or_else(|| invalid("exchange row size overflow"))?;
                let end = exchange
                    .program
                    .address
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("exchange row range overflow"))?;
                rows.entry(exchange.program.address)
                    .and_modify(|existing| *existing = (*existing).max(end))
                    .or_insert(end);
            }
            crate::TileStep::Repeat(repeat) => collect_exchange_rows(rows, &repeat.body)?,
            crate::TileStep::Compute(_) | crate::TileStep::Checkpoint(_) => {}
        }
    }
    Ok(())
}

fn collect_compute_symbols(symbols: &mut Vec<String>, steps: &[crate::TileStep]) {
    for step in steps {
        match step {
            crate::TileStep::Compute(compute) => symbols.push(compute.symbol.clone()),
            crate::TileStep::Repeat(repeat) => collect_compute_symbols(symbols, &repeat.body),
            crate::TileStep::Exchange(_) | crate::TileStep::Checkpoint(_) => {}
        }
    }
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
    let (built, _, _) = build_package_artifacts(graph, config, false)?;
    Ok(built)
}

/// Builds an ordinary optimized package with resumable PBRK0 traps after each
/// top-level operator and returns the storage map needed for non-invasive
/// numerical inspection.
pub fn build_diagnostic_package(
    graph: &ComputeGraph,
    config: &PackageConfig,
) -> PackageBuildResult<CompiledPackage> {
    let (mut built, mid, low) = build_package_artifacts(graph, config, true)?;
    let topology = active_topology(low.tile_count)?;
    let mut checkpoints = Vec::new();
    for operation in &mid.operations {
        if !matches!(
            operation.kind,
            MidOperationKind::Operator(_) | MidOperationKind::Repeat(_)
        ) {
            continue;
        }
        let Some(source) = operation.source else {
            continue;
        };
        let tensors = operation
            .results
            .iter()
            .map(|&value| compiled_tensor(&mid, &low, &built.placement, &topology, value, None))
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
    built.checkpoints = checkpoints;
    Ok(built)
}

fn package_precisions(mid: &MidGraph) -> BTreeMap<ValueId, Precision> {
    mid.values
        .iter()
        .map(|value| (value.origin, value.tensor_type.format.precision))
        .collect()
}

fn compiled_graph_tensors(
    mid: &MidGraph,
    low: &LowProgram,
    placement: &crate::Placement,
    topology: &Topology,
) -> PackageBuildResult<(Vec<CompiledTensor>, Vec<CompiledTensor>)> {
    let inputs = low
        .inputs
        .iter()
        .map(|input| {
            compiled_tensor(
                mid,
                low,
                placement,
                topology,
                input.value,
                Some(input.name.clone()),
            )
        })
        .collect::<PackageBuildResult<Vec<_>>>()?;
    let outputs = low
        .outputs
        .iter()
        .enumerate()
        .map(|(index, output)| {
            compiled_tensor(
                mid,
                low,
                placement,
                topology,
                output.value,
                Some(format!("output.{index}")),
            )
        })
        .collect::<PackageBuildResult<Vec<_>>>()?;
    Ok((inputs, outputs))
}

fn build_package_artifacts(
    graph: &ComputeGraph,
    config: &PackageConfig,
    diagnostic_checkpoints: bool,
) -> PackageBuildResult<(CompiledPackage, MidGraph, LowProgram)> {
    validate_tile_count(u32::from(config.pipeline.tile_count))?;
    let mut planning = config.pipeline.clone();
    planning.diagnostic_checkpoints = diagnostic_checkpoints;
    if diagnostic_checkpoints {
        planning.profiling = crate::ProfilingConfig::Disabled;
    }
    let cost_model = planning.target.cost_model();
    let finalists = build_phase("lower_mid", || {
        Ok(lower_finalists(
            graph,
            &planning,
            &cost_model,
            planning.exchange_schedule_finalists,
        )?)
    })?;
    let (mid, low) = build_phase("select_finalist", || {
        select_scheduled_finalist(finalists, &planning)
    })?;
    tracing::info!(
        logical_shards = low.shards.len(),
        exchange_phases = low.exchange_phases.len(),
        "lowered graph for package construction"
    );
    let runtime_artifact = build_phase("compile_runtime", || {
        Ok(config
            .toolchain
            .compile(&config.runtime_source, "static_runtime", &[])?)
    })?;
    let kernel_plan = build_phase("plan_kernels", || Ok(KernelBuildPlan::from_program(&low)?))?;
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
    let mut package_config = config.clone();
    package_config.pipeline = planning;
    let built = build_package_from_objects(&mid, &low, &package_config, &objects, &kernel_plan)?;
    Ok((built, mid, low))
}

fn select_scheduled_finalist(
    finalists: Vec<MidGraph>,
    planning: &PipelineConfig,
) -> PackageBuildResult<(MidGraph, LowProgram)> {
    if finalists.len() == 1 {
        let mid = finalists.into_iter().next().unwrap();
        tracing::info!(
            estimated_cycles = mid.metrics.cost.cycles,
            estimated_exchange_cycles = mid.metrics.cost.exchange_cycles,
            "selected analytical operator plan"
        );
        for operation in &mid.operations {
            tracing::debug!(
                source = ?operation.source,
                kind = ?operation.kind,
                memory = ?operation.metrics.memory,
                plan = ?operation.operator_plan(),
                "selected mid-level operation"
            );
        }
        let low = lower_to_tiles(&mid, planning)?;
        return Ok((mid, low));
    }

    let topology = active_topology(planning.tile_count)?;
    let mut ranked = Vec::with_capacity(finalists.len());
    for (index, mid) in finalists.into_iter().enumerate() {
        let low = lower_to_tiles(&mid, planning)?;
        let placement = place(&low)?;
        let exchanges = lower_exchanges(
            &low,
            &placement,
            &topology,
            crate::ExchangeLoweringOptions::default(),
        )?;
        let scheduled_exchange_cycles = exchanges
            .phases
            .iter()
            .map(|phase| u64::from(phase.event_cycles))
            .sum::<u64>()
            .saturating_add(
                (exchanges.phases.len() as u64)
                    .saturating_mul(crate::IPU21_TARGET_COSTS.exchange_phase_cycles),
            );
        let estimated_non_exchange_cycles = mid
            .metrics
            .cost
            .cycles
            .saturating_sub(mid.metrics.cost.exchange_cycles);
        let refined_cycles =
            estimated_non_exchange_cycles.saturating_add(scheduled_exchange_cycles);
        tracing::info!(
            finalist = index,
            analytical_cycles = mid.metrics.cost.cycles,
            analytical_exchange_cycles = mid.metrics.cost.exchange_cycles,
            scheduled_exchange_cycles,
            refined_cycles,
            "scheduled operator-plan finalist"
        );
        ranked.push((refined_cycles, index, mid, low));
    }
    ranked.sort_by_key(|(cycles, index, _, _)| (*cycles, *index));
    let (_, selected, mid, low) = ranked.remove(0);
    tracing::info!(
        selected,
        "selected physically scheduled operator-plan finalist"
    );
    Ok((mid, low))
}

fn build_package_from_objects(
    mid: &MidGraph,
    program: &LowProgram,
    config: &PackageConfig,
    objects: &[Vec<u8>],
    kernel_plan: &KernelBuildPlan,
) -> PackageBuildResult<CompiledPackage> {
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
        AddressRegion::new(
            ipu_exchange::EXCHANGE_WINDOW_BASE,
            ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES,
        ),
    )?;
    memory.reserve(
        "runtime state",
        AddressRegion::new(RUNTIME_STATE_BASE, RUNTIME_EXECUTABLE_START),
    )?;

    let provisional_placement = build_phase("plan_exchange_storage", || Ok(place(program)?))?;
    let provisional_exchanges = build_phase("lower_exchanges_provisional", || {
        Ok(lower_exchanges(
            program,
            &provisional_placement,
            &topology,
            crate::ExchangeLoweringOptions::default(),
        )?)
    })?
    .phases;
    let execution_tile_count = u16::try_from(Topology::c600().tile_count())?;
    let exchange_table_bytes = crate::tile::compact_exchange_table_bytes(
        &provisional_exchanges,
        execution_tile_count,
        program.tile_count,
    )?;
    let profile_samples = config.pipeline.profiling.records_steps().then(|| {
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
                bounds: AddressRegion::new(
                    crate::IPU21_DATA_BASE,
                    ipu_package::IPU21_INTERLEAVED_MEMORY_BASE,
                ),
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
                bounds: AddressRegion::new(
                    crate::IPU21_DATA_BASE,
                    ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT,
                ),
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
    let (provisional_tensors, provisional_output_tensors) =
        compiled_graph_tensors(mid, program, &provisional_placement, &topology)?;
    let provisional_inputs = program
        .inputs
        .iter()
        .zip(&provisional_tensors)
        .filter(|(input, _)| input.kind == crate::GraphInputKind::Host)
        .map(|(_, tensor)| tensor.binding())
        .collect::<PackageBuildResult<Vec<_>>>()?;
    let provisional_weights = program
        .inputs
        .iter()
        .zip(&provisional_tensors)
        .filter(|(input, _)| input.kind == crate::GraphInputKind::Parameter)
        .map(|(_, tensor)| tensor.binding())
        .collect::<PackageBuildResult<Vec<_>>>()?;
    let mut provisional_outputs = provisional_output_tensors
        .iter()
        .map(CompiledTensor::binding)
        .collect::<PackageBuildResult<Vec<_>>>()?;
    if let Some(storage) = &profile_storage {
        provisional_outputs.push(cycle_binding(
            "profile.start-cycle",
            PROFILE_START_CYCLE,
            program.tile_count,
            &topology,
        ));
        provisional_outputs.push(profile_binding(
            program,
            &physical_to_logical,
            storage.range.start,
        )?);
        provisional_outputs.push(cycle_binding(
            "profile.end-cycle",
            PROFILE_END_CYCLE,
            program.tile_count,
            &topology,
        ));
    }
    let sizing_host_base = memory.next_free(
        linked_end,
        AddressRegion::new(TILE_MEMORY_BASE, ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT),
        4,
        "host programs",
    )?;
    let mut provisional_auxiliary_ranges = vec![
        vec![AddressRegion::new(
            crate::IPU21_DATA_BASE,
            TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
        )];
        usize::from(execution_tile_count)
    ];
    for logical in 0..program.tile_count {
        provisional_auxiliary_ranges[usize::from(topology.physical(logical)?)] =
            provisional_placement.tile_auxiliary_ranges[usize::from(logical)].clone();
    }
    let provisional_host = host::plan(
        &provisional_weights,
        &provisional_inputs,
        &provisional_outputs,
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
                bounds: AddressRegion::new(linked_end, ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT),
                end_alignment: 8,
                guard_after: 0,
            })
        })
        .transpose()?;
    let host_code_base = host_code
        .as_ref()
        .map_or(sizing_host_base, |code| code.range.start);
    let provisional_host = host::plan(
        &provisional_weights,
        &provisional_inputs,
        &provisional_outputs,
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
        &provisional_placement,
        &provisional_exchanges,
        kernel_plan,
        exchange_code_base,
        execution_tile_count,
        false,
    )?;
    let sizing_code_address = memory.next_free(
        host_code_base + host_code_bytes,
        AddressRegion::new(TILE_MEMORY_BASE, ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT),
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
                        &provisional_exchanges,
                        logical,
                        u32::try_from(physical)?,
                        &mut tile_program,
                        storage.range.start,
                    )?;
                }
                let generated = emit(
                    &tile_program,
                    &symbols,
                    host,
                    &CodegenOptions {
                        code_address: sizing_code_address,
                        initial_profile_address: config
                            .pipeline
                            .profiling
                            .records_overall_time()
                            .then_some(PROFILE_START_CYCLE),
                        final_profile_address: config
                            .pipeline
                            .profiling
                            .records_overall_time()
                            .then_some(PROFILE_END_CYCLE),
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
                bounds: AddressRegion::new(
                    host_code_base + host_code_bytes,
                    ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT,
                ),
                // Host programs and generated tile code form one contiguous
                // executable region. Keep planned standard-memory values out
                // of its final memory element.
                end_alignment: ipu_package::TILE_MEMORY_ELEMENT_SIZE,
                guard_after: 0,
            })?
            .range
            .start
    };
    let standard_ranges = memory.free_ranges(AddressRegion::new(
        crate::IPU21_DATA_BASE,
        ipu_package::IPU21_INTERLEAVED_MEMORY_BASE,
    ));
    tracing::info!(
        linked_end,
        profile_bytes = profile_storage
            .as_ref()
            .map_or(0, |allocation| allocation.range.size()),
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
        Ok(lower_exchanges(
            program,
            &placement,
            &topology,
            crate::ExchangeLoweringOptions {
                diagnostics: config.pipeline.exchange_diagnostics,
            },
        )?)
    })?;
    let exchange_schedule = lowered_exchanges.schedule_snapshot;
    let exchanges = lowered_exchanges.phases;
    let (tensors, output_tensors) = compiled_graph_tensors(mid, program, &placement, &topology)?;
    let inputs = program
        .inputs
        .iter()
        .zip(&tensors)
        .filter(|(input, _)| input.kind == crate::GraphInputKind::Host)
        .map(|(_, tensor)| tensor.binding())
        .collect::<PackageBuildResult<Vec<_>>>()?;
    let weights = program
        .inputs
        .iter()
        .zip(&tensors)
        .filter(|(input, _)| input.kind == crate::GraphInputKind::Parameter)
        .map(|(_, tensor)| tensor.binding())
        .collect::<PackageBuildResult<Vec<_>>>()?;
    let mut outputs = output_tensors
        .iter()
        .map(CompiledTensor::binding)
        .collect::<PackageBuildResult<Vec<_>>>()?;
    if config.pipeline.profiling.records_overall_time() {
        outputs.push(cycle_binding(
            "profile.start-cycle",
            PROFILE_START_CYCLE,
            program.tile_count,
            &topology,
        ));
        if config.pipeline.profiling.records_steps() {
            outputs.push(profile_binding(
                program,
                &physical_to_logical,
                profile_storage
                    .as_ref()
                    .expect("full profiling allocates per-step storage")
                    .range
                    .start,
            )?);
        }
        outputs.push(cycle_binding(
            "profile.end-cycle",
            PROFILE_END_CYCLE,
            program.tile_count,
            &topology,
        ));
    }
    let mut inactive_auxiliary_ranges = standard_ranges.clone();
    inactive_auxiliary_ranges.push(AddressRegion::new(
        ipu_package::IPU21_INTERLEAVED_MEMORY_BASE,
        TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
    ));
    let host = host::plan(
        &weights,
        &inputs,
        &outputs,
        execution_tile_count,
        host_code_base,
        &{
            let mut ranges =
                vec![inactive_auxiliary_ranges.clone(); usize::from(execution_tile_count)];
            for logical in 0..program.tile_count {
                ranges[usize::from(topology.physical(logical)?)] =
                    placement.tile_auxiliary_ranges[usize::from(logical)].clone();
            }
            ranges
        },
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
    if exchange_rows
        .as_ref()
        .is_some_and(|storage| finalizer.exchange_code_end() > storage.range.end)
    {
        return Err(invalid(
            "final exchange rows exceeded their planned allocation",
        ));
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
                        initial_profile_address: config
                            .pipeline
                            .profiling
                            .records_overall_time()
                            .then_some(PROFILE_START_CYCLE),
                        final_profile_address: config
                            .pipeline
                            .profiling
                            .records_overall_time()
                            .then_some(PROFILE_END_CYCLE),
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
        return Err(invalid(
            "generated tile code exceeded its planned allocation",
        ));
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
    Ok(CompiledPackage {
        application,
        inputs: tensors,
        outputs: output_tensors,
        precisions: package_precisions(mid),
        exchange_phases: exchanges,
        exchange_schedule,
        exchange_code_base,
        checkpoints: Vec::new(),
        placement,
    })
}

fn compiled_tensor(
    mid: &MidGraph,
    low: &LowProgram,
    placement: &crate::Placement,
    topology: &Topology,
    value: crate::MidValueId,
    name: Option<String>,
) -> PackageBuildResult<CompiledTensor> {
    let mid_value = mid
        .values
        .get(value.index() as usize)
        .ok_or_else(|| invalid("diagnostic mid-level value is missing"))?;
    let shards = low
        .values
        .iter()
        .find(|candidate| candidate.value == value)
        .into_iter()
        .flat_map(|value| &value.shards)
        .filter(|id| {
            low.shards
                .get(id.index() as usize)
                .is_some_and(|shard| shard.definition != crate::ShardDefinition::Unmaterialized)
        })
        .map(|id| {
            let storage = low
                .shards
                .get(id.index() as usize)
                .ok_or_else(|| invalid("compiled tensor shard is missing"))?;
            Ok(CompiledTensorShard {
                physical_tile: topology.physical(storage.tile)?,
                address: placement
                    .shard_addresses
                    .get(id)
                    .copied()
                    .ok_or_else(|| invalid("compiled tensor shard placement is missing"))?,
                storage: storage.clone(),
            })
        })
        .collect::<PackageBuildResult<Vec<_>>>()?;
    Ok(CompiledTensor {
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
    generated: &GeneratedProgram,
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
    generated: &GeneratedProgram,
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

fn runtime_retained_symbols(program: &LowProgram, config: &PackageConfig) -> Vec<String> {
    let mut symbols = vec![COMPLETE_SYMBOL.into()];
    if !program.exchange_phases.is_empty() {
        symbols.push(WORKER_BARRIER_SYMBOL.into());
        symbols.push(PATCH_ROW_SYMBOL.into());
        if !program.repeat_runs.is_empty() {
            symbols.push(PATCH_WORD_SYMBOL.into());
        }
    }
    if config.pipeline.profiling.records_overall_time() {
        symbols.push(SAMPLE_CYCLE_SYMBOL.into());
    }
    if !program.inputs.is_empty() || !program.outputs.is_empty() {
        symbols.push(HOST_RUN_SYMBOL.into());
        symbols.push(REPEAT_CALL_SYMBOL.into());
    }
    if program
        .tiles
        .iter()
        .any(|tile| tile_has_local_copy(program, tile))
    {
        if program
            .tiles
            .iter()
            .any(|tile| tile_has_halfword_copy(program, tile))
        {
            symbols.push(crate::emitter::COPY_U16_SYMBOL.into());
        }
        symbols.push(crate::emitter::COPY_U32_SYMBOL.into());
        symbols.push(crate::emitter::COPY_U64_SYMBOL.into());
        symbols.push(crate::emitter::COPY_STRIDED_U64_SYMBOL.into());
    }
    if program
        .tiles
        .iter()
        .any(|tile| tile_has_fill_zero(program, tile))
    {
        symbols.push(crate::emitter::FILL_ZERO_U64_SYMBOL.into());
    }
    symbols
}

fn tile_has_fill_zero(program: &LowProgram, tile: &crate::TileWorkList) -> bool {
    program.work(tile).any(|work| match work {
        crate::TileWorkRef::Kernel(run) => {
            matches!(run.kernel, crate::TileKernelSpec::FillZero)
        }
        crate::TileWorkRef::Repeat(repeat) => tile_has_fill_zero(program, &repeat.body),
        crate::TileWorkRef::Exchange(_)
        | crate::TileWorkRef::LocalCopy(_)
        | crate::TileWorkRef::Checkpoint(..) => false,
    })
}

fn tile_has_halfword_copy(program: &LowProgram, tile: &crate::TileWorkList) -> bool {
    program.work(tile).any(|work| match work {
        crate::TileWorkRef::LocalCopy(copy) => match copy.pattern {
            crate::LocalCopyPattern::Contiguous => !copy.bytes.is_multiple_of(4),
            crate::LocalCopyPattern::Strided { row_bytes, .. } => !row_bytes.is_multiple_of(4),
        },
        crate::TileWorkRef::Repeat(repeat) => tile_has_halfword_copy(program, &repeat.body),
        crate::TileWorkRef::Exchange(_)
        | crate::TileWorkRef::Kernel(_)
        | crate::TileWorkRef::Checkpoint(..) => false,
    })
}

fn tile_has_local_copy(program: &LowProgram, tile: &crate::TileWorkList) -> bool {
    program.work(tile).any(|work| match work {
        crate::TileWorkRef::LocalCopy(_) => true,
        crate::TileWorkRef::Repeat(repeat) => tile_has_local_copy(program, &repeat.body),
        crate::TileWorkRef::Exchange(_)
        | crate::TileWorkRef::Kernel(_)
        | crate::TileWorkRef::Checkpoint(..) => false,
    })
}

fn cycle_binding(name: &str, address: u32, tile_count: u16, topology: &Topology) -> Binding {
    Binding {
        name: name.into(),
        dtype: "u32".into(),
        shape: vec![u32::from(tile_count)],
        slices: (0..tile_count)
            .map(|tile| RegionSlice {
                tile: u32::from(
                    topology
                        .physical(tile)
                        .expect("active topology contains tile"),
                ),
                tile_address: address,
                file_offset: u64::from(tile) * 4,
                size: 4,
            })
            .collect(),
    }
}

fn profile_binding(
    program: &LowProgram,
    physical_to_logical: &[u16],
    address: u32,
) -> PackageBuildResult<Binding> {
    let mut file_offset = 0u64;
    let mut sample_count = 0u32;
    let slices = physical_to_logical
        .iter()
        .enumerate()
        .filter_map(|(physical, &logical)| {
            let steps = if logical < program.tile_count {
                profile_step_count(program, &program.tiles[usize::from(logical)])
            } else {
                inactive_profile_work(program).len()
            };
            (steps != 0).then_some((physical, steps))
        })
        .map(|(physical, steps)| {
            let samples = u32::try_from(steps + 1)?;
            let size = u64::from(samples)
                .checked_mul(4)
                .ok_or_else(|| invalid("profile binding size overflow"))?;
            let slice = RegionSlice {
                tile: u32::try_from(physical)?,
                tile_address: address,
                file_offset,
                size,
            };
            file_offset = file_offset
                .checked_add(size)
                .ok_or_else(|| invalid("profile binding offset overflow"))?;
            sample_count = sample_count
                .checked_add(samples)
                .ok_or_else(|| invalid("profile binding sample count overflow"))?;
            Ok(slice)
        })
        .collect::<PackageBuildResult<Vec<_>>>()?;
    Ok(Binding {
        name: PROFILE_CYCLES_BINDING.into(),
        dtype: "u32".into(),
        shape: vec![sample_count],
        slices,
    })
}

fn instrument_profile(
    program: &LowProgram,
    exchanges: &[crate::PhysicalExchangePhase],
    logical_tile: u16,
    physical_tile: u32,
    tile_program: &mut crate::TileProgram,
    address: u32,
) -> PackageBuildResult<TileProfilePlan> {
    let mut plans = Vec::with_capacity(tile_program.steps.len());
    if logical_tile < program.tile_count {
        let schedule = program
            .work(&program.tiles[usize::from(logical_tile)])
            .collect::<Vec<_>>();
        if schedule.len() != tile_program.steps.len() {
            return Err(invalid("tile profile work does not match finalized steps"));
        }
        for (index, (&work, step)) in schedule.iter().zip(&mut tile_program.steps).enumerate() {
            if index != 0 && profile_work_can_merge(schedule[index - 1], work) {
                continue;
            }
            let following = schedule[index + 1..].iter().find_map(|work| match work {
                crate::TileWorkRef::Kernel(run) => Some(&run.provenance),
                crate::TileWorkRef::Repeat(repeat) => Some(&repeat.provenance),
                crate::TileWorkRef::Exchange(_)
                | crate::TileWorkRef::LocalCopy(_)
                | crate::TileWorkRef::Checkpoint(..) => None,
            });
            step_profile(step).before = Some(profile_address(address, plans.len())?);
            let mut description = profile_step(
                program,
                exchanges,
                logical_tile,
                index,
                work,
                step,
                following,
            )?;
            let invocations = schedule[index + 1..]
                .iter()
                .take_while(|&&next| profile_work_can_merge(work, next))
                .count()
                + 1;
            description.metadata.push(ProfileMetadata {
                name: "invocations".into(),
                value: invocations.to_string(),
            });
            description.local_index = u32::try_from(plans.len())?;
            plans.push(description);
        }
    } else {
        let schedule = inactive_profile_work(program);
        if schedule.len() != tile_program.steps.len() {
            return Err(invalid(
                "inactive tile profile does not match finalized steps",
            ));
        }
        for (index, (work, step)) in schedule
            .into_iter()
            .zip(&mut tile_program.steps)
            .enumerate()
        {
            if let (crate::TileWorkRef::Checkpoint(operation, _), crate::TileStep::Checkpoint(_)) =
                (work, &*step)
            {
                step_profile(step).before = Some(profile_address(address, index)?);
                plans.push(ProfileStep {
                    local_index: u32::try_from(index)?,
                    phase: u32::try_from(index)?,
                    epoch: 0,
                    operation: format!("operation.{}", operation.index()),
                    kind: ProfileStepKind::Idle,
                    kernel: "diagnostic-checkpoint".into(),
                    metadata: Vec::new(),
                    exchange_activities: Vec::new(),
                    exchange_event_cycles: 0,
                });
                continue;
            }
            let (phase, provenance) = match (work, &*step) {
                (crate::TileWorkRef::Exchange(id), crate::TileStep::Exchange(_)) => {
                    let phase = &program.exchange_phases[id.index() as usize];
                    (0x8000_0000 | id.index(), &phase.provenance)
                }
                (crate::TileWorkRef::Repeat(repeat), crate::TileStep::Repeat(_)) => {
                    (u32::try_from(index)?, &repeat.provenance)
                }
                _ => return Err(invalid("inactive tile contains executable work")),
            };
            step_profile(step).before = Some(profile_address(address, index)?);
            plans.push(inactive_tile_description(index, phase, provenance)?);
        }
    }
    if let Some(last) = tile_program.steps.last_mut() {
        step_profile(last).after = Some(profile_address(address, plans.len())?);
    }
    Ok(TileProfilePlan {
        physical_tile,
        steps: plans,
    })
}

fn inactive_profile_work(program: &LowProgram) -> Vec<crate::TileWorkRef<'_>> {
    program
        .tiles
        .first()
        .into_iter()
        .flat_map(|tile| program.work(tile))
        .filter(|work| {
            matches!(
                work,
                crate::TileWorkRef::Exchange(_)
                    | crate::TileWorkRef::Repeat(_)
                    | crate::TileWorkRef::Checkpoint(..)
            )
        })
        .collect()
}

fn profile_step_count(program: &LowProgram, tile: &crate::TileWorkList) -> usize {
    let mut previous = None;
    let mut count = 0;
    for work in program.work(tile) {
        if previous.is_none_or(|previous| !profile_work_can_merge(previous, work)) {
            count += 1;
        }
        previous = Some(work);
    }
    count
}

fn profile_work_can_merge(
    previous: crate::TileWorkRef<'_>,
    current: crate::TileWorkRef<'_>,
) -> bool {
    matches!(
        (previous, current),
        (crate::TileWorkRef::Kernel(previous), crate::TileWorkRef::Kernel(current))
            if previous.kernel == current.kernel && previous.provenance == current.provenance
    ) || matches!(
        (previous, current),
        (
            crate::TileWorkRef::LocalCopy(previous),
            crate::TileWorkRef::LocalCopy(current)
        ) if previous.bytes == current.bytes && previous.pattern == current.pattern
    )
}

#[allow(clippy::too_many_arguments)]
fn profile_step(
    program: &LowProgram,
    exchanges: &[crate::PhysicalExchangePhase],
    logical_tile: u16,
    index: usize,
    work: crate::TileWorkRef<'_>,
    step: &mut crate::TileStep,
    following: Option<&crate::WorkProvenance>,
) -> PackageBuildResult<ProfileStep> {
    match (work, step) {
        (crate::TileWorkRef::Exchange(id), crate::TileStep::Exchange(exchange)) => {
            let phase = &program.exchange_phases[id.index() as usize];
            if !exchange.active {
                exchange_synchronization_description(
                    index,
                    0x8000_0000 | id.index(),
                    &phase.provenance,
                )
            } else {
                let mut description = profile_description(
                    index,
                    0x8000_0000 | id.index(),
                    &phase.provenance,
                    ProfileStepKind::Exchange,
                    "exchange",
                )?;
                let physical = exchanges
                    .get(id.index() as usize)
                    .ok_or_else(|| invalid("profile exchange phase is missing"))?;
                description.exchange_activities = physical
                    .activities
                    .get(usize::from(logical_tile))
                    .ok_or_else(|| invalid("profile exchange tile is missing"))?
                    .iter()
                    .map(|activity| ProfileExchangeActivity {
                        kind: match activity.kind {
                            crate::ExchangeActivityKind::Send => ProfileExchangeActivityKind::Send,
                            crate::ExchangeActivityKind::Receive => {
                                ProfileExchangeActivityKind::Receive
                            }
                            crate::ExchangeActivityKind::PartnerBusy => {
                                ProfileExchangeActivityKind::PartnerBusy
                            }
                        },
                        start_cycle: activity.start_cycle,
                        end_cycle: activity.end_cycle,
                    })
                    .collect();
                description.exchange_event_cycles = physical.event_cycles;
                Ok(description)
            }
        }
        (crate::TileWorkRef::Kernel(run), crate::TileStep::Compute(compute)) => {
            let mut description = profile_description(
                index,
                u32::try_from(index)?,
                &run.provenance,
                ProfileStepKind::Compute,
                &compute.symbol,
            )?;
            description.metadata.push(ProfileMetadata {
                name: "kernelSpec".into(),
                value: format!("{:?}", run.kernel),
            });
            description.metadata.push(ProfileMetadata {
                name: "outputElements".into(),
                value: view_logical_elements(&run.output).to_string(),
            });
            for (operand, input) in run.inputs.iter().enumerate() {
                description.metadata.push(ProfileMetadata {
                    name: format!("input{operand}Elements"),
                    value: input
                        .views
                        .iter()
                        .map(view_logical_elements)
                        .sum::<u64>()
                        .to_string(),
                });
            }
            Ok(description)
        }
        (crate::TileWorkRef::LocalCopy(copy), crate::TileStep::Compute(compute)) => {
            if let Some(provenance) = following {
                let mut description = profile_description(
                    index,
                    u32::try_from(index)?,
                    provenance,
                    ProfileStepKind::Compute,
                    &compute.symbol,
                )?;
                description.metadata[0].value = "LocalCopy".into();
                description.metadata.extend([
                    ProfileMetadata {
                        name: "bytes".into(),
                        value: copy.bytes.to_string(),
                    },
                    ProfileMetadata {
                        name: "pattern".into(),
                        value: format!("{:?}", copy.pattern),
                    },
                ]);
                Ok(description)
            } else {
                Ok(ProfileStep {
                    local_index: u32::try_from(index)?,
                    phase: u32::try_from(index)?,
                    epoch: 0,
                    operation: String::new(),
                    kind: ProfileStepKind::Compute,
                    kernel: compute.symbol.clone(),
                    metadata: vec![
                        ProfileMetadata {
                            name: "reason".into(),
                            value: "LocalCopy".into(),
                        },
                        ProfileMetadata {
                            name: "bytes".into(),
                            value: copy.bytes.to_string(),
                        },
                        ProfileMetadata {
                            name: "pattern".into(),
                            value: format!("{:?}", copy.pattern),
                        },
                    ],
                    exchange_activities: Vec::new(),
                    exchange_event_cycles: 0,
                })
            }
        }
        (crate::TileWorkRef::Repeat(repeat), crate::TileStep::Repeat(_)) => profile_description(
            index,
            u32::try_from(index)?,
            &repeat.provenance,
            ProfileStepKind::Compute,
            "repeat",
        ),
        (crate::TileWorkRef::Checkpoint(operation, _), crate::TileStep::Checkpoint(_)) => {
            Ok(ProfileStep {
                local_index: u32::try_from(index)?,
                phase: u32::try_from(index)?,
                epoch: 0,
                operation: format!("operation.{}", operation.index()),
                kind: ProfileStepKind::Synchronization,
                kernel: "diagnostic-checkpoint".into(),
                metadata: Vec::new(),
                exchange_activities: Vec::new(),
                exchange_event_cycles: 0,
            })
        }
        _ => Err(invalid(
            "tile profile work kind does not match finalized step",
        )),
    }
}

fn view_logical_elements(view: &crate::ShardView) -> u64 {
    view.extents.iter().fold(1u64, |elements, extent| {
        elements.saturating_mul(u64::from(extent.logical_end.saturating_sub(extent.start)))
    })
}

fn exchange_synchronization_description(
    index: usize,
    phase: u32,
    provenance: &crate::WorkProvenance,
) -> PackageBuildResult<ProfileStep> {
    let mut description = profile_description(
        index,
        phase,
        provenance,
        ProfileStepKind::Synchronization,
        "sync",
    )?;
    description.metadata[0].value = "ExchangeBarrier".into();
    Ok(description)
}

fn inactive_tile_description(
    index: usize,
    phase: u32,
    provenance: &crate::WorkProvenance,
) -> PackageBuildResult<ProfileStep> {
    let mut description =
        profile_description(index, phase, provenance, ProfileStepKind::Idle, "idle")?;
    description.metadata[0].value = "InactiveTile".into();
    Ok(description)
}

fn profile_description(
    index: usize,
    phase: u32,
    provenance: &crate::WorkProvenance,
    kind: ProfileStepKind,
    kernel: &str,
) -> PackageBuildResult<ProfileStep> {
    let mut metadata = vec![ProfileMetadata {
        name: "reason".into(),
        value: format!("{:?}", provenance.reason),
    }];
    if let Some(value) = provenance.value {
        metadata.push(ProfileMetadata {
            name: "value".into(),
            value: value.index().to_string(),
        });
    }
    Ok(ProfileStep {
        local_index: u32::try_from(index)?,
        phase,
        epoch: 0,
        operation: provenance
            .operation
            .map(|operation| format!("operation.{}", operation.index()))
            .unwrap_or_default(),
        kind,
        kernel: kernel.into(),
        metadata,
        exchange_activities: Vec::new(),
        exchange_event_cycles: 0,
    })
}

fn step_profile(step: &mut crate::TileStep) -> &mut crate::StepProfile {
    match step {
        crate::TileStep::Exchange(exchange) => &mut exchange.profile,
        crate::TileStep::Compute(compute) => &mut compute.profile,
        crate::TileStep::Repeat(repeat) => &mut repeat.profile,
        crate::TileStep::Checkpoint(checkpoint) => &mut checkpoint.profile,
    }
}

fn profile_address(base: u32, index: usize) -> PackageBuildResult<u32> {
    base.checked_add(
        u32::try_from(index)?
            .checked_mul(4)
            .ok_or_else(|| invalid("profile address overflow"))?,
    )
    .ok_or_else(|| invalid("profile address overflow"))
}

impl CompiledTensor {
    pub fn binding(&self) -> PackageBuildResult<Binding> {
        let name = self
            .name
            .clone()
            .ok_or_else(|| invalid("unnamed compiled tensor cannot be a host binding"))?;
        let dtype = match self.precision {
            crate::Precision::F8F143 { .. } => "f8f143",
            crate::Precision::F16 => "f16",
            crate::Precision::F32 => "f32",
        };
        let mut file_offset = 0u64;
        let slices = self
            .shards
            .iter()
            .map(|shard| {
                let size = u64::from(shard_storage_bytes(&shard.storage)?);
                let slice = RegionSlice {
                    tile: u32::from(shard.physical_tile),
                    tile_address: shard.address,
                    file_offset,
                    size,
                };
                file_offset = file_offset
                    .checked_add(size)
                    .ok_or_else(|| invalid("binding file offset overflow"))?;
                Ok(slice)
            })
            .collect::<PackageBuildResult<Vec<_>>>()?;
        Ok(Binding {
            name,
            dtype: dtype.into(),
            shape: self.shape.0.clone(),
            slices,
        })
    }
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
        (HOST_STAGING_SYMBOL.into(), host_staging_address),
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
        memory.reserve(name, AddressRegion::new(segment.address, end))?;
    }
    Ok(())
}

pub(crate) fn invalid(message: impl Into<String>) -> PackageBuildError {
    PackageBuildError::Invalid(message.into())
}
