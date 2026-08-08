use crate::graph::ComputeGraph;
use crate::host;
use crate::low::{LowProgram, LowValue};
use crate::memory::{
    PROFILE_END_CYCLE, PROFILE_START_CYCLE, RUNTIME_STATE_BASE, RUNTIME_STATE_BYTES,
    WORKER_STACK_HEADROOM,
};
use crate::mid::{Ipu21CostModel, PipelineConfig};
use crate::{
    COMPLETE_SYMBOL, COMPLETION_ADDRESS_SYMBOL, CodegenOptions, KernelBuildPlan, PRNG_SEED_SYMBOL,
    PROGRAM_ADDRESS_SYMBOL, RUNTIME_ENTRY_SYMBOL, SAMPLE_CYCLE_SYMBOL, WORKER_BARRIER_SYMBOL,
    WORKER_STACK_BASE_SYMBOL, WORKER_SYNC_CONTEXT_SYMBOL, emit, lower, lower_exchanges,
    lower_to_tile_programs_with_fill, lower_to_tiles, place, shard_storage_bytes,
};
use ipu_driver::{APPLICATION_LOAD_BASE, TILES_PER_BATCH};
use ipu_elf::{ElfError, LinkOptions, LinkedImage, Toolchain, link};
use ipu_exchange::{ExchangeError, Topology, encode_br_m, encode_setzi_m};
use ipu_package::{
    Application, Binding, EntryPoint, PackageError, RegionSlice, SEGMENT_EXECUTE, SEGMENT_READ,
    SEGMENT_WRITE, Segment, TILE_MEMORY_BASE, TileImage,
};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::num::TryFromIntError;
use std::path::PathBuf;
use std::time::Instant;

const ENTRY_BYTES: u32 = 8;
const SUPPORT_START: u32 = APPLICATION_LOAD_BASE + ENTRY_BYTES;
const COMPLETION_ADDRESS: u32 = RUNTIME_STATE_BASE;

#[derive(Debug, thiserror::Error)]
pub enum PackageBuildError {
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

pub type PackageBuildResult<T> = std::result::Result<T, PackageBuildError>;

#[derive(Clone, Debug)]
pub struct PackageConfig {
    pub toolchain: Toolchain,
    pub runtime_source: PathBuf,
    pub kernel_source_directory: PathBuf,
    pub build_directory: PathBuf,
    pub pipeline: PipelineConfig,
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
) -> PackageBuildResult<Application> {
    validate_tile_count(u32::from(config.pipeline.tile_count))?;
    let mid = build_phase("lower_mid", || {
        Ok(lower(graph, &config.pipeline, &Ipu21CostModel)?)
    })?;
    let low = build_phase("lower_tiles", || {
        Ok(lower_to_tiles(&mid, &config.pipeline)?)
    })?;
    tracing::info!(
        logical_shards = low.shards.len(),
        exchange_phases = low.exchange_phases.len(),
        "lowered graph for package construction"
    );
    let runtime_artifact = build_phase("compile_runtime", || {
        Ok(config.toolchain.compile(
            &config.runtime_source,
            &config.build_directory,
            "static_runtime",
            &[],
        )?)
    })?;
    let kernel_plan = build_phase("plan_kernels", || Ok(KernelBuildPlan::from_program(&low)?))?;
    let objects = build_phase("compile_kernels", || {
        let mut objects = vec![fs::read(&runtime_artifact.object)?];
        for compilation in &kernel_plan.compilations {
            let artifact = config.toolchain.compile(
                config.kernel_source_directory.join(compilation.source),
                &config.build_directory,
                &compilation.name,
                &compilation.flags,
            )?;
            objects.push(fs::read(&artifact.object)?);
        }
        Ok(objects)
    })?;
    build_package_from_objects(&low, config, &objects, &kernel_plan)
}

fn build_package_from_objects(
    program: &LowProgram,
    config: &PackageConfig,
    objects: &[Vec<u8>],
    kernel_plan: &KernelBuildPlan,
) -> PackageBuildResult<Application> {
    let topology = active_topology(program.tile_count)?;
    let placement = build_phase("place_storage", || Ok(place(program)?))?;
    let exchanges = build_phase("lower_exchanges", || {
        Ok(lower_exchanges(program, &placement, &topology)?)
    })?;
    let retained_runtime = runtime_retained_symbols(program, config);
    let layout = build_phase("link_runtime", || {
        link_runtime(
            objects,
            runtime_symbols(0, 0, 0)?,
            kernel_plan,
            &retained_runtime,
        )
    })?;
    let execution_tile_count = u16::try_from(Topology::c600().tile_count())?;
    let finalized = build_phase("finalize_tile_programs", || {
        Ok(lower_to_tile_programs_with_fill(
            program,
            &placement,
            &exchanges,
            kernel_plan,
            linked_end(&layout)?,
            execution_tile_count,
        )?)
    })?;
    let inputs = program
        .inputs
        .iter()
        .filter(|input| input.kind == crate::GraphInputKind::Host)
        .map(|input| input_binding(program, &placement, &topology, input))
        .collect::<PackageBuildResult<Vec<_>>>()?;
    let weights = program
        .inputs
        .iter()
        .filter(|input| input.kind == crate::GraphInputKind::Parameter)
        .map(|input| input_binding(program, &placement, &topology, input))
        .collect::<PackageBuildResult<Vec<_>>>()?;
    let mut outputs = program
        .outputs
        .iter()
        .enumerate()
        .map(|(index, output)| output_binding(program, &placement, &topology, output, index))
        .collect::<PackageBuildResult<Vec<_>>>()?;
    if config.pipeline.profiling.enabled {
        outputs.push(cycle_binding(
            "profile.start-cycle",
            PROFILE_START_CYCLE,
            program.tile_count,
        ));
        outputs.push(cycle_binding(
            "profile.end-cycle",
            PROFILE_END_CYCLE,
            program.tile_count,
        ));
    }
    let host = host::plan(
        &weights,
        &inputs,
        &outputs,
        execution_tile_count,
        finalized.exchange_code_end,
        &(0..execution_tile_count)
            .map(|tile| {
                placement
                    .tile_data_end
                    .get(usize::from(tile))
                    .copied()
                    .unwrap_or(crate::IPU21_DATA_BASE)
            })
            .collect::<Vec<_>>(),
    )?;
    let code_address = align_up(host.end, 4)?;
    let symbols = layout
        .symbols
        .clone()
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let generated = build_phase("emit_tile_code", || {
        Ok(finalized
            .programs
            .iter()
            .zip(&host.programs)
            .map(|(program, host)| {
                emit(
                    program,
                    &symbols,
                    host,
                    &CodegenOptions {
                        code_address,
                        initial_profile_address: config
                            .pipeline
                            .profiling
                            .enabled
                            .then_some(PROFILE_START_CYCLE),
                        final_profile_address: config
                            .pipeline
                            .profiling
                            .enabled
                            .then_some(PROFILE_END_CYCLE),
                        ..CodegenOptions::default()
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?)
    })?;
    if let Some((tile, program)) = generated.iter().enumerate().find(|(_, program)| {
        code_address
            .checked_add(program.bytes.len() as u32)
            .is_none_or(|end| end > ipu_exchange::EXCHANGE_WINDOW_BASE)
    }) {
        let available = ipu_exchange::EXCHANGE_WINDOW_BASE.saturating_sub(code_address);
        let prefix_overflow = code_address.saturating_sub(ipu_exchange::EXCHANGE_WINDOW_BASE);
        return Err(invalid(format!(
            "tile {tile} generated program is {} bytes, but only {available} bytes are available from 0x{code_address:x} to the exchange window at 0x{:x} (linked exchange/support/host prefix already exceeds the window by {prefix_overflow} bytes)",
            program.bytes.len(),
            ipu_exchange::EXCHANGE_WINDOW_BASE,
        )));
    }

    let tiles = build_phase("build_tile_images", || {
        (0..execution_tile_count)
            .map(|physical_tile| {
                build_tile(
                    u32::from(physical_tile),
                    objects,
                    kernel_plan,
                    &retained_runtime,
                    code_address,
                    &generated[usize::from(physical_tile)],
                    (
                        &host.segments[usize::from(physical_tile)],
                        host.staging_address,
                    ),
                )
            })
            .collect::<PackageBuildResult<Vec<_>>>()
    })?;
    let mut application = Application {
        tiles,
        ..Application::default()
    };
    application
        .tiles
        .sort_unstable_by_key(|tile| tile.physical_tile);
    application.inputs = inputs;
    application.weights = weights;
    application.outputs = outputs;
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
    Ok(application)
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
    Ok(Topology::new((0..tile_count).collect())?)
}

fn build_tile(
    physical_tile: u32,
    objects: &[Vec<u8>],
    kernel_plan: &KernelBuildPlan,
    retained_runtime: &[String],
    code_address: u32,
    generated: &crate::GeneratedProgram,
    host: (&[Segment], u32),
) -> PackageBuildResult<TileImage> {
    let (host_segments, host_staging_address) = host;
    let linked = link_runtime(
        objects,
        runtime_symbols(physical_tile, code_address, host_staging_address)?,
        kernel_plan,
        retained_runtime,
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
        address: code_address,
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
            regions: vec![(SUPPORT_START, ipu_exchange::EXCHANGE_WINDOW_BASE)],
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
    }
    if config.pipeline.profiling.enabled {
        symbols.push(SAMPLE_CYCLE_SYMBOL.into());
    }
    if !program.inputs.is_empty() || !program.outputs.is_empty() {
        symbols.push(crate::HOST_RUN_SYMBOL.into());
        symbols.push(crate::REPEAT_CALL_SYMBOL.into());
    }
    if program.tiles.iter().any(tile_has_local_copy) {
        symbols.push(crate::COPY_U32_SYMBOL.into());
        symbols.push(crate::COPY_U64_SYMBOL.into());
    }
    symbols
}

fn tile_has_local_copy(tile: &crate::TileWorkList) -> bool {
    tile.work.iter().any(|work| match work {
        crate::TileWork::LocalCopy(_) => true,
        crate::TileWork::Repeat(repeat) => tile_has_local_copy(&repeat.body),
        crate::TileWork::Exchange(_) | crate::TileWork::Kernel(_) => false,
    })
}

fn input_binding(
    program: &LowProgram,
    placement: &crate::Placement,
    topology: &Topology,
    input: &crate::LowInput,
) -> PackageBuildResult<Binding> {
    binding(
        program,
        placement,
        topology,
        input.name.clone(),
        &input.shards,
    )
}

fn output_binding(
    program: &LowProgram,
    placement: &crate::Placement,
    topology: &Topology,
    output: &LowValue,
    index: usize,
) -> PackageBuildResult<Binding> {
    binding(
        program,
        placement,
        topology,
        format!("output.{index}"),
        &output.shards,
    )
}

fn cycle_binding(name: &str, address: u32, tile_count: u16) -> Binding {
    Binding {
        name: name.into(),
        dtype: "u32".into(),
        shape: vec![u32::from(tile_count)],
        slices: (0..tile_count)
            .map(|tile| RegionSlice {
                tile: u32::from(tile),
                tile_address: address,
                file_offset: u64::from(tile) * 4,
                size: 4,
            })
            .collect(),
    }
}

fn binding(
    program: &LowProgram,
    placement: &crate::Placement,
    topology: &Topology,
    name: String,
    shards: &[crate::LowShardId],
) -> PackageBuildResult<Binding> {
    let first = shards
        .first()
        .and_then(|id| program.shards.get(id.index() as usize))
        .ok_or_else(|| invalid("binding has no shards"))?;
    let dtype = match first.tensor_type.format.precision {
        crate::Precision::F8F143 { .. } => "f8f143",
        crate::Precision::F16 => "f16",
        crate::Precision::F32 => "f32",
    };
    let mut file_offset = 0u64;
    let slices = shards
        .iter()
        .map(|id| {
            let shard = &program.shards[id.index() as usize];
            let size = u64::from(shard_storage_bytes(shard)?);
            let slice = RegionSlice {
                tile: u32::from(topology.physical(shard.tile)?),
                tile_address: *placement
                    .shard_addresses
                    .get(id)
                    .ok_or_else(|| invalid("binding shard is not placed"))?,
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
        shape: first.tensor_type.shape.0.clone(),
        slices,
    })
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

fn align_up(value: u32, alignment: u32) -> PackageBuildResult<u32> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| invalid("address alignment overflow"))
}

pub(crate) fn invalid(message: impl Into<String>) -> PackageBuildError {
    PackageBuildError::Invalid(message.into())
}
