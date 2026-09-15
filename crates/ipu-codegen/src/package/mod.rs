//! Package support sizing, address binding and final image emission.
//! Tensor search, placement and exchange scheduling belong to the compiler driver.
use ipu_target::ipu21::fabric::Topology;
use ipu_target::ipu21::instruction::{encode_br_m, encode_setzi_m};
use ipu_target::ipu21::memory::TILE_MEMORY_BASE;
mod bindings;
use bindings::{PackageBindings, auxiliary_ranges};
mod profile;
mod profile_work;
use profile::{instrument_profile, profile_binding};
mod support;
pub(crate) use support::{PackageSupport, size_support};
mod tile_program;
pub use tile_program::build_tile_program_package;

use crate::exchange::ExchangeError;
use crate::graph::{OperationId, ValueId};
use crate::host;
use crate::low::LowProgram;
use crate::memory::{
    MemoryAllocation, MemoryLayoutError, MemoryRequest, PROFILE_END_CYCLE, PROFILE_START_CYCLE,
    RUNTIME_STATE_BASE, RUNTIME_STATE_BYTES, TileMemoryMap, WORKER_STACK_HEADROOM,
};
use crate::runtime_layout::{
    COMPLETE_SYMBOL, COMPLETION_ADDRESS_SYMBOL, HOST_RUN_SYMBOL, PRNG_SEED_SYMBOL,
    PROGRAM_ADDRESS_SYMBOL, REPEAT_CALL_SYMBOL, RUNTIME_ENTRY_SYMBOL, SAMPLE_CYCLE_SYMBOL,
    WORKER_BARRIER_SYMBOL, WORKER_STACK_BASE_SYMBOL, WORKER_SYNC_CONTEXT_SYMBOL,
};
use crate::{
    CodegenOptions, KernelBuildPlan, TileProgram, TileProgramLowering, emit, shard_storage_bytes,
};
use crate::{PipelineConfig, Precision, TileGraph};

use ipu_elf::{ElfError, LinkOptions, LinkedImage, Toolchain, link};
use ipu_package::loader_abi::{APPLICATION_LOAD_BASE, TILES_PER_BATCH};
use ipu_package::{
    Application, Binding, DEBUG_ALL_TILES, DebugRegion, DebugSymbol, EntryPoint,
    PROFILE_CYCLES_BINDING, PackageError, ProfileExchangeActivity, ProfileExchangeActivityKind,
    ProfileMetadata, ProfileStep, ProfileStepKind, RegionSlice, SEGMENT_EXECUTE, SEGMENT_READ,
    SEGMENT_WRITE, Segment, TileImage, TileProfilePlan,
};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::num::TryFromIntError;

const ENTRY_BYTES: u32 = 8;
const SUPPORT_START: u32 = APPLICATION_LOAD_BASE + ENTRY_BYTES;
const COMPLETION_ADDRESS: u32 = RUNTIME_STATE_BASE;
const RUNTIME_EXECUTABLE_START: u32 = (RUNTIME_STATE_BASE
    + RUNTIME_STATE_BYTES
    + ipu_target::ipu21::memory::TILE_MEMORY_ELEMENT_SIZE
    - 1)
    & !(ipu_target::ipu21::memory::TILE_MEMORY_ELEMENT_SIZE - 1);

#[derive(Debug, thiserror::Error)]
pub enum PackageBuildError {
    #[error(transparent)]
    Topology(#[from] ipu_target::ipu21::fabric::TopologyError),
    #[error(transparent)]
    Instruction(#[from] ipu_target::ipu21::instruction::InstructionError),
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
    #[error("planning failed: {0}")]
    Planning(#[from] crate::planner::LoweringError),
    #[error(transparent)]
    Program(#[from] crate::mid::ProgramError),
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
    /// Resumable operator checkpoints; empty for ordinary builds.
    pub checkpoints: Vec<DiagnosticCheckpoint>,
    pub precisions: BTreeMap<ValueId, Precision>,
    /// Selected operand precision of each semantic product (before accumulation).
    pub multiply_precisions: BTreeMap<crate::OperationId, Precision>,
    /// Exact physical exchange schedules retained for low-level diagnostics.
    /// This is build metadata and is not serialized into the application.
    pub exchange_phases: Vec<crate::PhysicalExchangePhase>,
    /// Address-resolved inputs to physical exchange scheduling and row codegen.
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
    pub view: crate::ShardView,
}

pub(crate) fn package_multiply_precisions(
    low: &TileGraph,
) -> BTreeMap<crate::OperationId, Precision> {
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

pub(crate) fn package_precisions(mid: &TileGraph) -> BTreeMap<ValueId, Precision> {
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

pub(crate) fn package_inputs(
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

pub(crate) fn emit_package(
    program: &LowProgram,
    placement: &crate::Placement,
    exchanges: &[crate::PhysicalExchangePhase],
    support: &PackageSupport,
    config: &PipelineConfig,
    invocations: u32,
) -> PackageBuildResult<Application> {
    let topology = active_topology(program.tile_count)?;
    let execution_topology = Topology::c600();
    let execution_tile_count = u16::try_from(execution_topology.tile_count())?;
    let objects = &support.objects;
    let kernel_plan = &support.kernel_plan;
    let retained_runtime = &support.retained_runtime;
    let layout = &support.layout;
    let physical_to_logical = &support.physical_to_logical;
    let code_address = support.code_address;
    let generated_code_bytes = support.generated_code_bytes;
    let host_code_base = support.host_code_base;
    let host_code_bytes = support.host_code_bytes;
    let host_data = &support.host_data;
    let exchange_rows = &support.exchange_rows;
    let exchange_code_base = support.exchange_code_base;
    let available_ranges = &support.available_ranges;
    let profile_addresses = config.profiling.then(|| {
        placement
            .auxiliary_allocations
            .iter()
            .map(|allocations| allocations[0].address)
            .collect::<Vec<_>>()
    });
    let PackageBindings {
        inputs,
        weights,
        outputs,
    } = PackageBindings::new(
        program,
        placement,
        &topology,
        &physical_to_logical,
        profile_addresses.as_deref(),
    )?;
    let inactive_auxiliary_ranges = available_ranges
        .iter()
        .copied()
        .filter(|range| *range != crate::place::HOST_SCRATCH_RANGE)
        .collect::<Vec<_>>();
    let mut host_data_ranges = auxiliary_ranges(
        placement,
        &execution_topology,
        execution_tile_count,
        &inactive_auxiliary_ranges,
    )?;
    if let Some(storage) = host_data {
        for ranges in &mut host_data_ranges {
            ranges.push((storage.range.start, storage.range.end));
            ranges.sort_unstable();
        }
    }
    let mut host = host::plan(
        &weights,
        &inputs,
        &outputs,
        execution_tile_count,
        host_code_base,
        &host_data_ranges,
    )?;
    for call in &mut host.protocol.calls {
        if call.name == "run" {
            call.invocations = invocations;
        }
    }
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
        placement,
        exchanges,
        kernel_plan,
        exchange_code_base,
        execution_tile_count,
        true,
    )?;
    if let Some(storage) = exchange_rows {
        // Placement can change row sharing and instruction alignment. The
        // entire SRAM element is already excluded from tensor storage, so
        // let final rows use its padding while retaining fetch look-ahead.
        let capacity_end =
            storage.reserved.end - ipu_target::ipu21::memory::IPU21_SUPERVISOR_FETCH_LOOKAHEAD;
        tracing::debug!(
            planned_bytes = storage.range.len(),
            final_bytes = finalizer.exchange_code_end() - storage.range.start,
            capacity_bytes = capacity_end - storage.range.start,
            "checked final exchange table capacity"
        );
        check_exchange_budget(
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
    let prepared =
        tracing::info_span!("prepare_tile_code").in_scope(|| -> PackageBuildResult<_> {
            physical_to_logical
                .par_iter()
                .enumerate()
                .map(|(physical_tile, &logical)| {
                    let mut tile_program = finalizer.lower_tile(logical)?;
                    let profile = profile_addresses
                        .as_ref()
                        .map(|addresses| {
                            instrument_profile(
                                program,
                                exchanges,
                                logical,
                                u32::try_from(physical_tile)?,
                                &mut tile_program,
                                addresses[usize::from(logical)],
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
                    &layout.symbols,
                    host,
                    &CodegenOptions {
                        invocations,
                        code_address,
                        initial_profile_address: config.profiling.then_some(PROFILE_START_CYCLE),
                        final_profile_address: config.profiling.then_some(PROFILE_END_CYCLE),
                    },
                )?)
            })
            .collect::<PackageBuildResult<Vec<_>>>()
    };
    let generated = tracing::info_span!("emit_tile_code").in_scope(generate)?;
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
        retained_runtime,
        code_address,
        host_staging_address: host.staging_address,
    };
    let tiles =
        tracing::info_span!("build_tile_images").in_scope(|| -> PackageBuildResult<_> {
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
    let mut application = assemble_application(tiles, outputs, layout, host)?;
    for (physical, program) in generated.iter().enumerate() {
        add_generated_debug_map(
            &mut application,
            u32::try_from(physical)?,
            code_address,
            program,
        )?;
    }
    application.inputs = inputs;
    application.weights = weights;
    application.profile_tiles = profile_tiles;
    application.validate()?;
    Ok(application)
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

pub(crate) fn diagnostic_tensor(
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
    let shards = low
        .value_views(value)
        .iter()
        .filter_map(|view| {
            let storage = low.shards.get(view.shard.index() as usize)?;
            (storage.definition != crate::ShardDefinition::Unmaterialized)
                .then_some((view, storage))
        })
        .map(|(view, storage)| {
            let address = placement
                .shard_addresses
                .get(&view.shard)
                .copied()
                .ok_or_else(|| invalid("diagnostic shard placement is missing"))?;
            Ok(DiagnosticShard {
                physical_tile: topology.physical(storage.tile)?,
                address,
                storage: storage.clone(),
                view: view.clone(),
            })
        })
        .collect::<PackageBuildResult<Vec<_>>>()?;
    Ok(DiagnosticTensor {
        name,
        value: mid_value.origin,
        shape: mid_value.tensor_type.shape.clone(),
        precision: mid_value.tensor_type.format.precision,
        shards,
    })
}

/// Common package metadata for graph lowering and explicit tile programs.
/// Callers attach their bindings and generated-code debug maps before validation.
fn assemble_application(
    mut tiles: Vec<TileImage>,
    mut outputs: Vec<Binding>,
    linked: &LinkedImage,
    host: host::HostPackagePlan,
) -> PackageBuildResult<Application> {
    tiles.sort_unstable_by_key(|tile| tile.physical_tile);
    outputs.push(Binding {
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
    let mut application = Application {
        tiles,
        outputs,
        entry_points: vec![EntryPoint {
            name: "run".into(),
            command: 0,
            external_syncs: 0,
        }],
        host_exchange: host.protocol,
        ..Application::default()
    };
    add_linked_debug_map(&mut application, linked);
    for (physical, segments) in host.segments.iter().enumerate() {
        for segment in segments {
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
    Ok(application)
}

fn add_linked_debug_map(application: &mut Application, linked: &LinkedImage) {
    for segment in &linked.segments {
        application.debug_regions.push(DebugRegion {
            physical_tile: DEBUG_ALL_TILES,
            address: segment.range.start,
            size: segment.range.end - segment.range.start,
            name: "linked executable".into(),
        });
    }
    application.debug_symbols.extend(
        linked
            .symbols
            .iter()
            .filter(|(_, address)| {
                linked
                    .segments
                    .iter()
                    .any(|segment| segment.range.contains(address))
            })
            .map(|(name, &address)| DebugSymbol {
                name: name.clone(),
                address,
            }),
    );
    application
        .debug_symbols
        .sort_unstable_by_key(|symbol| symbol.address);
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

pub(crate) fn validate_tile_count(tile_count: u32) -> PackageBuildResult<()> {
    let maximum = Topology::c600().tile_count() as u32;
    if tile_count == 0 || !tile_count.is_multiple_of(TILES_PER_BATCH as u32) || tile_count > maximum
    {
        return Err(invalid(format!(
            "tile count must be a nonzero multiple of {TILES_PER_BATCH} and at most {maximum}"
        )));
    }
    Ok(())
}

pub(crate) fn active_topology(tile_count: u16) -> PackageBuildResult<Topology> {
    Ok(Topology::new(
        (0..tile_count)
            .map(ipu_target::c600::logical_to_physical)
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
        address: segment.range.start,
        memory_size: segment.range.end - segment.range.start,
        data: linked.bytes[segment.offset..segment.offset + segment.range.len()].to_vec(),
        flags: SEGMENT_READ | SEGMENT_EXECUTE,
    }));
    segments.extend(generated.exchange_rows.iter().map(|row| {
        let data = row
            .words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        Segment {
            address: row.address,
            memory_size: data.len() as u32,
            data,
            flags: SEGMENT_READ | SEGMENT_EXECUTE,
        }
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
                (SUPPORT_START, crate::runtime_layout::EXCHANGE_WINDOW_BASE),
                (
                    RUNTIME_EXECUTABLE_START,
                    ipu_target::ipu21::memory::IPU21_EXECUTABLE_MEMORY_LIMIT,
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
        symbols.push(crate::runtime_layout::PATCH_ROW_SYMBOL.into());
        if !program.repeat_runs.is_empty() {
            symbols.push(crate::runtime_layout::PATCH_REPEAT_TABLES_SYMBOL.into());
            symbols.push(crate::runtime_layout::PATCH_REPEAT_ARITHMETIC_SYMBOL.into());
        }
    }
    if config.profiling {
        symbols.push(SAMPLE_CYCLE_SYMBOL.into());
    }
    if !program.inputs.is_empty() || !program.outputs.is_empty() {
        symbols.push(crate::runtime_layout::HOST_RUN_SYMBOL.into());
        symbols.push(crate::runtime_layout::REPEAT_CALL_SYMBOL.into());
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
        (
            crate::runtime_layout::HOST_STAGING_SYMBOL.into(),
            host_staging_address,
        ),
    ]))
}

// Calls use explicit addresses. Use any free executable hole, including those
// before the highest linked section; unrelated support objects need not follow
// one another. End rounding/guards still isolate instruction fetch from data.
fn allocate_package_code(
    memory: &mut TileMemoryMap,
    name: &'static str,
    bytes: u32,
    end_alignment: u32,
    guard_after: u32,
) -> Result<MemoryAllocation, MemoryLayoutError> {
    memory.allocate(MemoryRequest {
        name,
        bytes,
        alignment: 8,
        bounds: RUNTIME_EXECUTABLE_START..ipu_target::ipu21::memory::IPU21_EXECUTABLE_MEMORY_LIMIT,
        end_alignment,
        guard_after,
    })
}

fn linked_end(linked: &LinkedImage) -> PackageBuildResult<u32> {
    linked
        .segments
        .iter()
        .map(|segment| segment.range.end)
        .max()
        .ok_or_else(|| invalid("linked runtime has no valid segments"))
}

fn reserve_fixed_runtime_memory(memory: &mut TileMemoryMap) -> PackageBuildResult<()> {
    memory.reserve(
        "host exchange aperture",
        crate::runtime_layout::EXCHANGE_WINDOW_BASE
            ..crate::runtime_layout::EXCHANGE_WINDOW_BASE
                + crate::runtime_layout::EXCHANGE_WINDOW_BYTES,
    )?;
    memory.reserve("runtime state", RUNTIME_STATE_BASE..crate::IPU21_DATA_BASE)?;
    Ok(())
}

fn reserve_linked_image(
    memory: &mut TileMemoryMap,
    linked: &LinkedImage,
    name: &'static str,
) -> PackageBuildResult<()> {
    for segment in &linked.segments {
        memory.reserve(name, segment.range.clone())?;
    }
    Ok(())
}

fn protect_executable_elements(
    memory: &mut TileMemoryMap,
    ranges: impl IntoIterator<Item = std::ops::Range<u32>>,
) -> PackageBuildResult<()> {
    // Instruction fetch conflicts with data access in the same element, but
    // distinct executable objects can share it safely.
    let element = ipu_target::ipu21::memory::TILE_MEMORY_ELEMENT_SIZE;
    for range in ranges {
        let start = (range.start / element * element).max(RUNTIME_EXECUTABLE_START);
        let end = range.end.div_ceil(element) * element;
        for (start, end) in memory.free_ranges(start..end) {
            memory.reserve("executable memory elements", start..end)?;
        }
    }
    Ok(())
}

pub(crate) fn invalid(message: impl Into<String>) -> PackageBuildError {
    PackageBuildError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ComputeGraph;
    use crate::estimate::Ipu21CostModel;

    #[test]
    fn package_assembly_maps_host_code_by_physical_tile() {
        let linked = LinkedImage {
            base: SUPPORT_START,
            entry: SUPPORT_START,
            bytes: vec![],
            segments: vec![],
            symbols: BTreeMap::new(),
        };
        let segment = |address, memory_size, flags| Segment {
            address,
            memory_size,
            flags,
            data: vec![],
        };
        let host = host::HostPackagePlan {
            descriptor_bytes: 0,
            programs: vec![],
            segments: vec![
                vec![segment(100, 8, SEGMENT_READ)],
                vec![
                    segment(200, 12, SEGMENT_EXECUTE),
                    segment(300, 0, SEGMENT_EXECUTE),
                ],
            ],
            protocol: Default::default(),
            end: 0,
            staging_address: 0,
        };
        let application = assemble_application(vec![], vec![], &linked, host).unwrap();
        assert_eq!(application.debug_regions.len(), 1);
        let region = &application.debug_regions[0];
        assert_eq!(
            (region.physical_tile, region.address, region.size),
            (1, 200, 12)
        );
        assert_eq!(region.name, "host exchange program");
        assert_eq!(application.outputs[0].name, "completion");
        assert_eq!(
            application.outputs[0].slices[0].tile_address,
            COMPLETION_ADDRESS
        );
        assert_eq!(application.entry_points[0].name, "run");
    }

    #[test]
    fn linked_sections_protect_their_complete_memory_elements() {
        let base = RUNTIME_EXECUTABLE_START;
        let element = ipu_target::ipu21::memory::TILE_MEMORY_ELEMENT_SIZE;
        let linked = LinkedImage {
            base,
            entry: base,
            bytes: vec![],
            segments: vec![
                ipu_elf::LinkedSegment {
                    range: base..base + 128,
                    offset: 0,
                },
                ipu_elf::LinkedSegment {
                    range: base + 256..base + 256 + 128,
                    offset: 128,
                },
                ipu_elf::LinkedSegment {
                    range: base + element..base + element + 128,
                    offset: 256,
                },
            ],
            symbols: BTreeMap::new(),
        };
        let mut memory = TileMemoryMap::new();
        reserve_linked_image(&mut memory, &linked, "test code").unwrap();
        let extra = allocate_package_code(&mut memory, "extra code", 64, 8, 0).unwrap();
        assert_eq!(extra.range.start, base + 128);
        protect_executable_elements(
            &mut memory,
            linked
                .segments
                .iter()
                .map(|segment| segment.range.clone())
                .chain([extra.reserved]),
        )
        .unwrap();
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
    fn package_code_reuses_holes_before_the_last_linked_section() {
        let base = RUNTIME_EXECUTABLE_START;
        let element = ipu_target::ipu21::memory::TILE_MEMORY_ELEMENT_SIZE;
        let limit = ipu_target::ipu21::memory::IPU21_EXECUTABLE_MEMORY_LIMIT;
        let linked = LinkedImage {
            base,
            entry: base,
            bytes: vec![],
            symbols: BTreeMap::new(),
            segments: vec![
                ipu_elf::LinkedSegment {
                    range: base..base + 128,
                    offset: 0,
                },
                ipu_elf::LinkedSegment {
                    range: base + 3 * element..limit,
                    offset: 128,
                },
            ],
        };
        let mut memory = TileMemoryMap::new();
        reserve_linked_image(&mut memory, &linked, "linked code").unwrap();
        assert!(
            memory
                .next_free(linked_end(&linked).unwrap(), base..limit, 8, "old tail")
                .is_err()
        );
        let host = allocate_package_code(
            &mut memory,
            "host programs",
            4096,
            8,
            ipu_target::ipu21::memory::IPU21_SUPERVISOR_FETCH_LOOKAHEAD,
        )
        .unwrap();
        let tile = allocate_package_code(&mut memory, "generated tile programs", 17556, element, 0)
            .unwrap();
        assert_eq!(host.range.start, base + 128);
        assert!(host.reserved.end <= tile.range.start);
        assert_eq!(tile.reserved.end, base + 2 * element);
    }

    #[test]
    fn runtime_state_tail_can_hold_data_but_never_code() {
        let mut memory = TileMemoryMap::new();
        memory
            .reserve("runtime state", RUNTIME_STATE_BASE..crate::IPU21_DATA_BASE)
            .unwrap();
        let data = memory
            .allocate(MemoryRequest {
                name: "host descriptors",
                bytes: 6280,
                alignment: 4,
                bounds: crate::IPU21_DATA_BASE..ipu_package::loader_abi::APPLICATION_LOAD_LIMIT,
                end_alignment: 4,
                guard_after: 0,
            })
            .unwrap();
        assert_eq!(data.range.start, crate::IPU21_DATA_BASE);
        assert!(data.range.end < RUNTIME_EXECUTABLE_START);
        let code = allocate_package_code(&mut memory, "code", 128, 8, 0).unwrap();
        assert_eq!(code.range.start, RUNTIME_EXECUTABLE_START);
    }

    #[test]
    fn attention_scratch_does_not_override_result_precision() {
        let mut graph = ComputeGraph::new();
        let q = graph.host_input("q", [2, 17, 72]).unwrap();
        let k = graph.host_input("k", [2, 73, 72]).unwrap();
        let v = graph.host_input("v", [2, 73, 72]).unwrap();
        let y = graph.flash_attention(q, k, v).unwrap();
        graph.set_outputs([y]).unwrap();
        let config = PipelineConfig::new(64)
            .with_attention_strategy(crate::AttentionStrategy::Flash)
            .with_automatic_input(q, Precision::F16)
            .with_automatic_input(k, Precision::F16)
            .with_automatic_input(v, Precision::F16);
        let mid = crate::planner::build::build_candidate(
            &graph,
            &config,
            &Ipu21CostModel,
            &crate::planner::cache::FragmentCache::default(),
            &crate::planner::Recipe::default(),
        )
        .unwrap()
        .program;
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

pub(crate) fn check_exchange_budget(bytes: u64, config: &PipelineConfig) -> PackageBuildResult<()> {
    if bytes > config.exchange_table_budget_bytes {
        return Err(PackageBuildError::ExchangeBudgetExceeded {
            bytes,
            budget: config.exchange_table_budget_bytes,
        });
    }
    Ok(())
}
