use crate::graph::ComputeGraph;
use crate::host;
use crate::low::{LowProgram, LowValue};
use crate::memory::{
    MemoryLayoutError, MemoryRequest, PROFILE_END_CYCLE, PROFILE_START_CYCLE, RUNTIME_STATE_BASE,
    RUNTIME_STATE_BYTES, TileMemoryMap, WORKER_STACK_HEADROOM,
};
use crate::mid::{Ipu21CostModel, PipelineConfig};
use crate::{
    COMPLETE_SYMBOL, COMPLETION_ADDRESS_SYMBOL, CodegenOptions, KernelBuildPlan, PRNG_SEED_SYMBOL,
    PROGRAM_ADDRESS_SYMBOL, RUNTIME_ENTRY_SYMBOL, SAMPLE_CYCLE_SYMBOL, TileProgramLowering,
    WORKER_BARRIER_SYMBOL, WORKER_STACK_BASE_SYMBOL, WORKER_SYNC_CONTEXT_SYMBOL, emit, lower,
    lower_exchanges, lower_to_tiles, place, shard_storage_bytes,
};
use ipu_driver::{APPLICATION_LOAD_BASE, TILES_PER_BATCH};
use ipu_elf::{ElfError, LinkOptions, LinkedImage, Toolchain, link};
use ipu_exchange::{ExchangeError, Topology, encode_br_m, encode_setzi_m};
use ipu_package::{
    Application, Binding, EntryPoint, PROFILE_CYCLES_BINDING, PackageError, ProfileMetadata,
    ProfileStep, ProfileStepKind, RegionSlice, SEGMENT_EXECUTE, SEGMENT_READ, SEGMENT_WRITE,
    Segment, TILE_MEMORY_BASE, TileImage, TileProfilePlan,
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
    build_package_from_objects(&low, config, &objects, &kernel_plan)
}

fn build_package_from_objects(
    program: &LowProgram,
    config: &PackageConfig,
    objects: &[Vec<u8>],
    kernel_plan: &KernelBuildPlan,
) -> PackageBuildResult<Application> {
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
    memory.reserve(
        "linked runtime and kernels",
        APPLICATION_LOAD_BASE..linked_end,
    )?;
    memory.reserve(
        "host exchange aperture",
        ipu_exchange::EXCHANGE_WINDOW_BASE
            ..ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES,
    )?;
    memory.reserve(
        "runtime state",
        RUNTIME_STATE_BASE..RUNTIME_STATE_BASE + RUNTIME_STATE_BYTES,
    )?;

    let provisional_placement = build_phase("plan_exchange_storage", || Ok(place(program)?))?;
    let provisional_exchanges = lower_exchanges(
        program,
        &provisional_placement,
        &topology,
        crate::ExchangeLoweringOptions::default(),
    )?;
    let exchange_table_bytes = provisional_exchanges
        .iter()
        .try_fold(0u32, |bytes, phase| {
            let words = phase
                .rows
                .iter()
                .map(Vec::len)
                .max()
                .unwrap_or(0)
                .max(ipu_exchange::PLAN_WORDS);
            bytes
                .checked_add(
                    u32::try_from(words)?
                        .checked_mul(4)
                        .ok_or_else(|| invalid("exchange row table size overflow"))?,
                )
                .ok_or_else(|| invalid("exchange row table size overflow"))
        })?;
    let exchange_rows = (exchange_table_bytes != 0)
        .then(|| {
            memory.allocate(MemoryRequest {
                name: "exchange row tables",
                bytes: exchange_table_bytes,
                alignment: 4,
                bounds: crate::IPU21_DATA_BASE..ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT,
                end_alignment: ipu_package::TILE_MEMORY_ELEMENT_SIZE,
                guard_after: ipu_package::IPU21_SUPERVISOR_FETCH_LOOKAHEAD,
            })
        })
        .transpose()?;
    let exchange_code_base = exchange_rows
        .as_ref()
        .map_or(crate::IPU21_DATA_BASE, |allocation| allocation.range.start);
    let profile_samples = config.pipeline.profiling.enabled.then(|| {
        program
            .tiles
            .iter()
            .map(|tile| tile.work.len())
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
                alignment: 4,
                bounds: crate::IPU21_DATA_BASE..ipu_package::IPU21_INTERLEAVED_MEMORY_BASE,
                end_alignment: 4,
                guard_after: 0,
            })?)
        })
        .transpose()?;
    let execution_tile_count = u16::try_from(Topology::c600().tile_count())?;
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
    let provisional_inputs = program
        .inputs
        .iter()
        .filter(|input| input.kind == crate::GraphInputKind::Host)
        .map(|input| input_binding(program, &provisional_placement, &topology, input))
        .collect::<PackageBuildResult<Vec<_>>>()?;
    let provisional_weights = program
        .inputs
        .iter()
        .filter(|input| input.kind == crate::GraphInputKind::Parameter)
        .map(|input| input_binding(program, &provisional_placement, &topology, input))
        .collect::<PackageBuildResult<Vec<_>>>()?;
    let mut provisional_outputs = program
        .outputs
        .iter()
        .enumerate()
        .map(|(index, output)| {
            output_binding(program, &provisional_placement, &topology, output, index)
        })
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
        TILE_MEMORY_BASE..ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT,
        4,
        "host programs",
    )?;
    let mut provisional_auxiliary_ranges = vec![
        vec![(
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
                alignment: 4,
                bounds: linked_end..ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT,
                end_alignment: 4,
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
    )?;
    let sizing_code_address = memory.next_free(
        host_code_base + host_code_bytes,
        TILE_MEMORY_BASE..ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT,
        4,
        "generated tile programs",
    )?;
    let generated_code_bytes = physical_to_logical
        .iter()
        .zip(&provisional_host.programs)
        .enumerate()
        .try_fold(0u32, |maximum, (physical, (&logical, host))| {
            let mut tile_program = provisional_finalizer.lower_tile(logical)?;
            if let Some(storage) = &profile_storage {
                instrument_profile(
                    program,
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
                        .enabled
                        .then_some(PROFILE_START_CYCLE),
                    final_profile_address: config
                        .pipeline
                        .profiling
                        .enabled
                        .then_some(PROFILE_END_CYCLE),
                    ..CodegenOptions::default()
                },
            )?;
            Ok::<_, PackageBuildError>(maximum.max(u32::try_from(generated.bytes.len())?))
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
                end_alignment: 4,
                guard_after: 0,
            })?
            .range
            .start
    };
    let standard_ranges =
        memory.free_ranges(crate::IPU21_DATA_BASE..ipu_package::IPU21_INTERLEAVED_MEMORY_BASE);
    let placement = build_phase("place_storage", || {
        Ok(crate::place::place_with_standard_ranges(
            program,
            &standard_ranges,
        )?)
    })?;
    let exchanges = build_phase("lower_exchanges", || {
        Ok(lower_exchanges(
            program,
            &placement,
            &topology,
            crate::ExchangeLoweringOptions {
                diagnostics: config.pipeline.exchange_diagnostics,
            },
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
            &topology,
        ));
        outputs.push(profile_binding(
            program,
            &physical_to_logical,
            profile_storage
                .as_ref()
                .expect("profiling storage is allocated when profiling is enabled")
                .range
                .start,
        )?);
        outputs.push(cycle_binding(
            "profile.end-cycle",
            PROFILE_END_CYCLE,
            program.tile_count,
            &topology,
        ));
    }
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
    if host.end.checked_sub(host_code_base) != Some(host_code_bytes) {
        return Err(invalid("host program size changed after tensor placement"));
    }
    let finalizer = TileProgramLowering::new(
        program,
        &placement,
        &exchanges,
        kernel_plan,
        exchange_code_base,
        execution_tile_count,
    )?;
    let prepared = physical_to_logical
        .iter()
        .enumerate()
        .map(|(physical_tile, &logical)| -> PackageBuildResult<_> {
            let mut tile_program = finalizer.lower_tile(logical)?;
            let profile = profile_storage
                .as_ref()
                .map(|storage| {
                    instrument_profile(
                        program,
                        logical,
                        u32::try_from(physical_tile)?,
                        &mut tile_program,
                        storage.range.start,
                    )
                })
                .transpose()?;
            Ok((tile_program, profile))
        })
        .collect::<PackageBuildResult<Vec<_>>>()?;
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
                            .enabled
                            .then_some(PROFILE_START_CYCLE),
                        final_profile_address: config
                            .pipeline
                            .profiling
                            .enabled
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
    if program
        .tiles
        .iter()
        .any(|tile| tile_has_local_copy(program, tile))
    {
        symbols.push(crate::COPY_U32_SYMBOL.into());
        symbols.push(crate::COPY_U64_SYMBOL.into());
    }
    symbols
}

fn tile_has_local_copy(program: &LowProgram, tile: &crate::TileWorkList) -> bool {
    program.work(tile).any(|work| match work {
        crate::TileWorkRef::LocalCopy(_) => true,
        crate::TileWorkRef::Repeat(repeat) => tile_has_local_copy(program, &repeat.body),
        crate::TileWorkRef::Exchange(_) | crate::TileWorkRef::Kernel(_) => false,
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
                program.tiles[usize::from(logical)].work.len()
            } else {
                program.exchange_phases.len()
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
            let following = schedule[index + 1..].iter().find_map(|work| match work {
                crate::TileWorkRef::Kernel(run) => Some(&run.provenance),
                crate::TileWorkRef::Repeat(repeat) => Some(&repeat.provenance),
                crate::TileWorkRef::Exchange(_) | crate::TileWorkRef::LocalCopy(_) => None,
            });
            plans.push(profile_step(
                program, index, work, step, address, following,
            )?);
        }
    } else {
        if program.exchange_phases.len() != tile_program.steps.len() {
            return Err(invalid(
                "inactive tile profile does not match exchange phases",
            ));
        }
        for (index, (phase, step)) in program
            .exchange_phases
            .iter()
            .zip(&mut tile_program.steps)
            .enumerate()
        {
            let crate::TileStep::Exchange(exchange) = step else {
                return Err(invalid("inactive tile contains non-exchange work"));
            };
            exchange.profile.before = Some(profile_address(address, index)?);
            plans.push(inactive_tile_description(
                index,
                0x8000_0000 | phase.id.index(),
                &phase.provenance,
            )?);
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

fn profile_step(
    program: &LowProgram,
    index: usize,
    work: crate::TileWorkRef<'_>,
    step: &mut crate::TileStep,
    base: u32,
    following: Option<&crate::WorkProvenance>,
) -> PackageBuildResult<ProfileStep> {
    step_profile(step).before = Some(profile_address(base, index)?);
    match (work, step) {
        (crate::TileWorkRef::Exchange(id), crate::TileStep::Exchange(exchange)) => {
            let phase = &program.exchange_phases[id.index() as usize];
            if exchange.row.first() == Some(&ipu_exchange::SANS_INACTIVE_INSTRUCTION) {
                exchange_synchronization_description(
                    index,
                    0x8000_0000 | id.index(),
                    &phase.provenance,
                )
            } else {
                profile_description(
                    index,
                    0x8000_0000 | id.index(),
                    &phase.provenance,
                    ProfileStepKind::Exchange,
                    "exchange",
                )
            }
        }
        (crate::TileWorkRef::Kernel(run), crate::TileStep::Compute(compute)) => {
            profile_description(
                index,
                u32::try_from(index)?,
                &run.provenance,
                ProfileStepKind::Compute,
                &compute.symbol,
            )
        }
        (crate::TileWorkRef::LocalCopy(_), crate::TileStep::Compute(compute)) => {
            if let Some(provenance) = following {
                let mut description = profile_description(
                    index,
                    u32::try_from(index)?,
                    provenance,
                    ProfileStepKind::Compute,
                    &compute.symbol,
                )?;
                description.metadata[0].value = "LocalCopy".into();
                Ok(description)
            } else {
                Ok(ProfileStep {
                    local_index: u32::try_from(index)?,
                    phase: u32::try_from(index)?,
                    epoch: 0,
                    operation: String::new(),
                    kind: ProfileStepKind::Compute,
                    kernel: compute.symbol.clone(),
                    metadata: vec![ProfileMetadata {
                        name: "reason".into(),
                        value: "LocalCopy".into(),
                    }],
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
        _ => Err(invalid(
            "tile profile work kind does not match finalized step",
        )),
    }
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
    })
}

fn step_profile(step: &mut crate::TileStep) -> &mut crate::StepProfile {
    match step {
        crate::TileStep::Exchange(exchange) => &mut exchange.profile,
        crate::TileStep::Compute(compute) => &mut compute.profile,
        crate::TileStep::Repeat(repeat) => &mut repeat.profile,
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

pub(crate) fn invalid(message: impl Into<String>) -> PackageBuildError {
    PackageBuildError::Invalid(message.into())
}
