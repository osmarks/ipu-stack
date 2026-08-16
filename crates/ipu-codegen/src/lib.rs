use ipu_exchange::{
    SANS_INACTIVE_INSTRUCTION, SYNC_SUPERVISOR_INSTRUCTION, encode_add_m_immediate, encode_br_m,
    encode_brz_m_immediate, encode_call_m_immediate, encode_ld32_m_immediate, encode_put_special_m,
    encode_setzi_m, encode_shl_m_immediate, encode_st32_m_immediate,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

mod cost;
mod estimate;
pub mod exchange;
pub mod graph;
mod host;
pub mod kernel;
mod layout;
pub mod low;
pub mod memory;
pub mod mid;
mod package;
pub mod place;
pub mod storage;
pub mod tile;
pub use exchange::{
    EXCHANGE_SCHEDULE_SNAPSHOT_VERSION, ExchangeActivity, ExchangeActivityDiagnostic,
    ExchangeActivityKind, ExchangeItemWidth, ExchangeLoweringError, ExchangeLoweringOptions,
    ExchangeMemoryElement, ExchangeScheduleDestination, ExchangeScheduleProblem,
    ExchangeScheduleRun, ExchangeScheduleSnapshot, ExchangeScheduleTransfer,
    ExchangeTileDiagnostic, LoweredExchanges, PhysicalExchangePhase, diagnose_exchange_tile,
    inactive_exchange_program, lower_exchanges, schedule_exchange_problem,
    validate_exchange_schedule,
};
pub use graph::{
    AddOptions, AttentionOptions, AttentionScale, BroadcastMode, ComputeGraph, GemmOptions,
    GraphError, GraphInput, GraphInputKind, GraphResult, Operation, OperationId, OperationKind,
    Region, RegionBuilder, Repeat, RepeatArguments, SplitHeadsOptions, TensorShape, ValueId,
    ValueSequence, ValueSequenceId,
};
pub use kernel::{
    KernelAbi, KernelAbiError, KernelAvailability, KernelBuildPlan, KernelCompilation,
    KernelMaterializationError, KernelSymbols, PlannedKernelCall, ScalarArgument,
    materialize_kernel_run, tile_kernel_abi, validate_kernel_run,
};
pub use layout::{ShardExtent, TensorRegion};
pub use low::{
    ExchangeOrder, ExchangePhase, ExchangePhaseId, KernelOperand, KernelRequirements, KernelRun,
    KernelRunId, KernelRunMetadata, LocalCopy, LocalCopyId, LocalCopyPattern, LogicalExchange,
    LowInput, LowLoweringError, LowLoweringResult, LowProgram, LowShard, LowShardId, LowValue,
    RepeatCarried, RepeatInvariant, RepeatIterated, RepeatRun, RepeatRunId, ShardDefinition,
    ShardView, TileWork, TileWorkList, TileWorkRef, WorkProvenance, WorkReason, lower_to_tiles,
};
pub use memory::{
    IPU21_DATA_BASE, IPU21_INTERLEAVED_REGION_BYTES, IPU21_PLANNED_DATA_BYTES,
    IPU21_STANDARD_FIXED_BYTES,
};
pub use mid::{
    AccumulationPrecision, AllocationRequirements, AmpOrder, AttentionBlocking,
    AttentionKernelFamily, AttentionPadding, AttentionPlan, AttentionStrategy, AxisTiling,
    BlockMajorOrder, BlockedGemmPlan, ConversionPlan, ConversionStrategy,
    ConversionStreamingPolicy, CostModel, ElementOrder, GemmBlockShape, GemmDistribution,
    GemmGeometry, GemmGrid, GemmKernelFamily, GemmKernelMode, GemmOrientation, GemmPlanConstraint,
    GemmResultGrid, GemmWeightLoad, GridOrder, HardwareMemoryConstraints, HardwareTarget,
    IPU21_TARGET_COSTS, Ipu21CostModel, Ipu21TargetCosts, Layout, LayoutError, LocalOperandStaging,
    LoweringError, LoweringResult, MemoryClass, MemoryElementRequirement, MemoryEstimate,
    MemoryOperand, MemoryPeaks, MemorySpaceRequirements, MemoryUsage, MidGraph, MidInput,
    MidOperation, MidOperationKind, MidOperator, MidRegion, MidRepeat, MidValue, MidValueId,
    OperandMaterialization, OperandRequirement, OperatorClass, OperatorDispatch, OperatorPlan,
    OperatorPlanError, OperatorRequirements, OutputAliasing, Padding, ParallelReductionPlan,
    PipelineConfig, PlannerSearchDomain, PointwiseInputMapping, Precision, ProfilingMode,
    ReductionStaging, TensorAxis, TensorFormat, TensorTiling, TensorType, TileKernelSpec, lower,
};
pub use package::{
    CompiledPackage, DiagnosticCheckpoint, DiagnosticPackage, DiagnosticShard, DiagnosticTensor,
    PackageBuildError, PackageBuildResult, PackageConfig, TileProgramData,
    build_diagnostic_package, build_package, build_tile_program_package,
};
pub use place::{Placement, PlacementError, place};
pub use storage::{
    ByteSpan, StorageError, StorageResult, amp_matrix_coordinates, block_major_matrix_coordinates,
    logical_view_byte_spans, shard_storage_bytes, view_byte_spans,
};
pub use tile::{TileLoweringError, TileProgramLowering, compact_exchange_row_address};

const INCOMING_BASE: u8 = 0xa4;
const INCOMING_DCOUNT: u8 = 0xa6;
const INCOMING_MUX: u8 = 0xa0;
const INCOMING_FORMAT: u8 = 0xa3;
const INCOMING_MUXPAIR: u8 = 0xa1;
// Recovered primitive PIC/XPIC plans arm A6 with one; their payload length is
// encoded in the timed instructions rather than this external-stream counter.
// Consolidated phases currently preserve that primitive-plan setting.
const INTERNAL_EXCHANGE_DCOUNT: u32 = 1;
const OUTGOING_BASE: u8 = 0xa7;
const FIRST_INPUT_REGISTER: u8 = 3;
const LAST_VALUE_REGISTER: u8 = 9;

pub const WORKER_BARRIER_SYMBOL: &str = "ipu_stack_static_worker_barrier";
pub const COMPLETE_SYMBOL: &str = "ipu_stack_static_complete";
pub const HOST_RUN_SYMBOL: &str = "ipu_stack_static_host_run";
pub const REPEAT_CALL_SYMBOL: &str = "ipu_stack_static_repeat_call";
pub const SAMPLE_CYCLE_SYMBOL: &str = "ipu_stack_static_sample_cycle";
pub const COPY_U16_SYMBOL: &str = "ipu_stack_static_copy_u16";
pub const COPY_U32_SYMBOL: &str = "ipu_stack_static_copy_u32";
pub const COPY_U64_SYMBOL: &str = "ipu_stack_copy_u64";
pub const COPY_STRIDED_U64_SYMBOL: &str = "ipu_stack_copy_strided_u64";
pub const FILL_ZERO_U64_SYMBOL: &str = "ipu_stack_fill_zero_u64";
pub const PATCH_WORD_SYMBOL: &str = "ipu_stack_static_patch_word";
pub const PATCH_ROW_SYMBOL: &str = "ipu_stack_static_patch_row";
pub const RUNTIME_ENTRY_SYMBOL: &str = "ipu_stack_static_start";
pub const PROGRAM_ADDRESS_SYMBOL: &str = "ipu_stack_static_program";
pub const WORKER_SYNC_CONTEXT_SYMBOL: &str = "ipu_stack_static_worker_sync_context";
pub const WORKER_STACK_BASE_SYMBOL: &str = "ipu_stack_static_worker_stack_base";
pub const PRNG_SEED_SYMBOL: &str = "ipu_stack_static_prng_seed";
pub const HOST_STAGING_SYMBOL: &str = "ipu_stack_static_host_staging";
pub const COMPLETION_ADDRESS_SYMBOL: &str = "ipu_stack_static_completion";
const PATCHED_BREAKPOINT_TRAP_BASE: u32 = 0x4180_1000;

#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    #[error("exchange encoding failed: {0}")]
    Exchange(#[from] ipu_exchange::ExchangeError),
    #[error("invalid tile program: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, CodegenError>;

/// A fully resolved program for one logical tile.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileProgram {
    pub tile: u16,
    pub steps: Vec<TileStep>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TileStep {
    Exchange(ExchangeStep),
    Compute(ComputeStep),
    Repeat(RepeatStep),
    Checkpoint(CheckpointStep),
}

/// A debugger-visible operator boundary using alternating PBRK0/PBRK1 traps.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointStep {
    pub operation: u32,
    pub breakpoint: u8,
    #[serde(default)]
    pub profile: StepProfile,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepeatStep {
    pub count: u32,
    /// Mutable bases used by [`TileAddress::RepeatPointer`] in the body.
    pub iterated_pointers: Vec<RepeatPointer>,
    pub body: Vec<TileStep>,
    #[serde(default)]
    pub profile: StepProfile,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepeatPointer {
    pub initial_address: u32,
    pub stride_bytes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TileAddress {
    Absolute(u32),
    /// The current base of an enclosing repeat plus a constant byte offset.
    RepeatPointer {
        index: u16,
        offset: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExchangeStep {
    /// Whether this tile executes a timed send/receive program after the boundary.
    pub active: bool,
    /// Base address used by point-to-point receive rows.
    pub incoming_base: u32,
    /// Preserve both exchange base registers on entry. Absolute-address paired
    /// rows use the two PIC streams directly and must not reset their state.
    #[serde(default)]
    pub preserve_base_registers: bool,
    /// Ordinary receive source selected outside the timed row when a paired
    /// receive uses the neighbouring sender for its waiting half.
    #[serde(default)]
    pub incoming_mux: Option<u16>,
    /// IPU21 incoming item format: 0 for 32-bit, 1 for the early half of a
    /// paired 64-bit path, and 2 for the waiting half.
    #[serde(default)]
    pub incoming_format: u8,
    /// Fixed source selection for the borrowed half of a paired 64-bit path.
    #[serde(default)]
    pub incoming_mux_pair: Option<u16>,
    /// Override the ordinary internal-exchange down-count. Paired 64-bit
    /// helper tiles execute mux timing while using zero to ignore the value.
    #[serde(default)]
    pub incoming_dcount: Option<u32>,
    /// The exchange row owns its supervisor sync and does not require the
    /// generic down-count setup. This is used by paired-width rows whose SDK
    /// form treats the sync and the following timing program as one unit.
    #[serde(default)]
    pub sync_in_program: bool,
    /// Synchronization-free timed exchange program.
    pub program: PlacedExchangeRow,
    /// Address words applied before invoking a structurally shared row.
    #[serde(default)]
    pub setup_patch: Option<ExchangeSetupPatch>,
    /// Words rewritten before the timed program is invoked inside a structured repeat.
    #[serde(default)]
    pub repeat_patches: Vec<ExchangePatch>,
    #[serde(default)]
    pub profile: StepProfile,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExchangeSetupPatch {
    /// Byte offsets into the shared executable row, reused by its structural shape.
    pub offsets: PlacedExchangeRow,
    /// Replacement instruction words for this use of the row.
    pub values: PlacedExchangeRow,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExchangePatch {
    pub word_offset: u32,
    /// Full replacement instruction words, indexed by repeat iteration.
    pub values: PlacedExchangeRow,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeStep {
    /// Exact linked kernel symbol; no naming convention is applied.
    pub symbol: String,
    pub output_address: TileAddress,
    pub input_addresses: Vec<TileAddress>,
    pub arguments: Vec<u32>,
    #[serde(default)]
    pub profile: StepProfile,
}

/// Optional explicit cycle-counter destinations around a step.
///
/// The addresses belong to caller-managed tile memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepProfile {
    pub before: Option<u32>,
    pub after: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostPhase {
    pub address: u32,
    pub active: bool,
    pub run_table: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostProgram {
    pub initialize: Vec<HostPhase>,
    pub inputs: Vec<HostPhase>,
    pub outputs: Vec<HostPhase>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodegenOptions {
    /// Address where the first emitted byte will be placed.
    pub code_address: u32,
    pub invocations: u32,
    pub initial_profile_address: Option<u32>,
    pub final_profile_address: Option<u32>,
}

impl Default for CodegenOptions {
    fn default() -> Self {
        Self {
            code_address: 0,
            invocations: 1,
            initial_profile_address: None,
            final_profile_address: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedProgram {
    pub bytes: Vec<u8>,
    /// Exchange data retained verbatim for explicit package placement.
    pub exchange_rows: Vec<PlacedExchangeRow>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacedExchangeRow {
    pub address: u32,
    pub words: Vec<u32>,
}

pub fn emit(
    program: &TileProgram,
    symbols: &BTreeMap<String, u32>,
    host: &HostProgram,
    options: &CodegenOptions,
) -> Result<GeneratedProgram> {
    if options.invocations == 0 {
        return Err(invalid("invocation count must be nonzero"));
    }
    validate(program)?;

    let complete = symbol(symbols, COMPLETE_SYMBOL)?;
    let mut code = TileCode::default();
    emit_host_phases(&mut code, symbols, &host.initialize)?;

    if options.invocations > 1 {
        code.add_immediate(11, 11, -8)?;
        code.setzi(0, options.invocations)?;
        code.st32(0, 11, 15, 0)?;
    }
    let invocation_start = code.address(options.code_address)?;
    emit_host_phases(&mut code, symbols, &host.inputs)?;

    if let Some(address) = options.initial_profile_address {
        emit_cycle_sample(&mut code, symbols, address)?;
    }

    let worker_barrier = program
        .steps
        .iter()
        .any(active_exchange)
        .then(|| symbol(symbols, WORKER_BARRIER_SYMBOL))
        .transpose()?;
    let mut exchange_rows = Vec::new();
    emit_steps(
        &mut code,
        program.tile,
        &program.steps,
        symbols,
        worker_barrier,
        &mut exchange_rows,
        None,
        None,
        options.code_address,
    )?;

    if let Some(address) = options.final_profile_address {
        emit_cycle_sample(&mut code, symbols, address)?;
    }
    emit_host_phases(&mut code, symbols, &host.outputs)?;
    if options.invocations > 1 {
        code.ld32(0, 11, 15, 0)?;
        code.add_immediate(0, 0, -1)?;
        code.st32(0, 11, 15, 0)?;
        let done_branch = code.words.len();
        code.brz(0, 0)?;
        code.jump(invocation_start)?;
        let done = code.address(options.code_address)?;
        code.words[done_branch] = encode_brz_m_immediate(0, done)?;
        code.add_immediate(11, 11, 8)?;
    }
    code.jump(complete)?;

    let mut unique_exchange_rows = BTreeMap::new();
    for row in exchange_rows {
        if unique_exchange_rows
            .insert(row.address, row.words.clone())
            .is_some_and(|existing| existing != row.words)
        {
            return Err(invalid("different exchange rows share an address"));
        }
    }
    Ok(GeneratedProgram {
        bytes: code.words.into_iter().flat_map(u32::to_le_bytes).collect(),
        exchange_rows: unique_exchange_rows
            .into_iter()
            .map(|(address, words)| PlacedExchangeRow { address, words })
            .collect(),
    })
}

#[allow(clippy::too_many_arguments)]
fn emit_steps(
    code: &mut TileCode,
    tile: u16,
    steps: &[TileStep],
    symbols: &BTreeMap<String, u32>,
    worker_barrier: Option<u32>,
    exchange_rows: &mut Vec<PlacedExchangeRow>,
    repeat_pointer_count: Option<usize>,
    repeat_count: Option<u32>,
    code_address: u32,
) -> Result<()> {
    let mut index = 0;
    while index < steps.len() {
        let step = &steps[index];
        if let TileStep::Compute(compute) = step
            && compute.symbol == COPY_U16_SYMBOL
            && let Some((source, destination)) = absolute_u16_copy(compute)
        {
            if let Some(address) = compute.profile.before {
                emit_cycle_sample(code, symbols, address)?;
            }
            let mut copies = vec![(source, destination)];
            let mut end = index + 1;
            while end < steps.len()
                && step_compute_profile(&steps[end - 1])
                    .is_none_or(|profile| profile.after.is_none())
            {
                let TileStep::Compute(next) = &steps[end] else {
                    break;
                };
                if next.symbol != COPY_U16_SYMBOL || next.profile.before.is_some() {
                    break;
                }
                let Some(copy) = absolute_u16_copy(next) else {
                    break;
                };
                copies.push(copy);
                end += 1;
                if next.profile.after.is_some() {
                    break;
                }
            }
            code.setzi(
                4,
                u32::try_from(copies.len())
                    .map_err(|_| invalid("halfword copy table is too large"))?,
            )?;
            code.call(symbol(symbols, COPY_U16_SYMBOL)?, 10)?;
            for (source, destination) in copies {
                code.instruction(source);
                code.instruction(destination);
            }
            if let Some(address) = step_compute_profile(&steps[end - 1]).and_then(|p| p.after) {
                emit_cycle_sample(code, symbols, address)?;
            }
            index = end;
            continue;
        }
        match step {
            TileStep::Exchange(exchange) => {
                if let Some(address) = exchange.profile.before {
                    emit_cycle_sample(code, symbols, address)?;
                }
                if let Some(patch) = &exchange.setup_patch {
                    emit_exchange_setup_patch(code, exchange, patch, symbols)?;
                }
                if !exchange.repeat_patches.is_empty() {
                    emit_exchange_patches(
                        code,
                        exchange,
                        repeat_count.ok_or_else(|| invalid("exchange patches outside repeat"))?,
                        symbols,
                    )?;
                }
                if !exchange.preserve_base_registers {
                    code.setzi(8, exchange.incoming_base)?;
                    code.put_special(INCOMING_BASE, 8)?;
                }
                if let Some(source) = exchange.incoming_mux {
                    code.setzi(8, u32::from(source))?;
                    code.put_special(INCOMING_MUX, 8)?;
                }
                if exchange.incoming_format != 0 {
                    code.setzi(8, u32::from(exchange.incoming_format))?;
                    code.put_special(INCOMING_FORMAT, 8)?;
                }
                if let Some(source) = exchange.incoming_mux_pair {
                    code.setzi(8, u32::from(source))?;
                    code.put_special(INCOMING_MUXPAIR, 8)?;
                }
                if !exchange.preserve_base_registers {
                    code.put_special(OUTGOING_BASE, 15)?;
                }
                if exchange.active {
                    code.call(
                        worker_barrier.expect("active exchange phase has worker barrier"),
                        7,
                    )?;
                    if exchange.incoming_dcount.is_some() || !exchange.sync_in_program {
                        code.setzi(
                            8,
                            exchange.incoming_dcount.unwrap_or(INTERNAL_EXCHANGE_DCOUNT),
                        )?;
                        code.put_special(INCOMING_DCOUNT, 8)?;
                    }
                }
                if exchange.active && !exchange.sync_in_program {
                    code.instruction(SYNC_SUPERVISOR_INSTRUCTION);
                } else {
                    if !exchange.active {
                        code.instruction(SANS_INACTIVE_INSTRUCTION);
                        code.instruction(ipu_exchange::SYNC_ANS_INSTRUCTION);
                    }
                }
                code.call(exchange.program.address, 10)?;
                if let Some(address) = exchange.profile.after {
                    emit_cycle_sample(code, symbols, address)?;
                }
                exchange_rows.push(exchange.program.clone());
                if let Some(patch) = &exchange.setup_patch {
                    exchange_rows.push(patch.offsets.clone());
                    exchange_rows.push(patch.values.clone());
                }
                exchange_rows.extend(
                    exchange
                        .repeat_patches
                        .iter()
                        .map(|patch| patch.values.clone()),
                );
            }
            TileStep::Compute(compute) => {
                if let Some(address) = compute.profile.before {
                    emit_cycle_sample(code, symbols, address)?;
                }
                emit_compute(code, tile, compute, symbols, repeat_pointer_count)?;
                if let Some(address) = compute.profile.after {
                    emit_cycle_sample(code, symbols, address)?;
                }
            }
            TileStep::Repeat(repeat) => {
                if let Some(address) = repeat.profile.before {
                    emit_cycle_sample(code, symbols, address)?;
                }
                emit_repeat(
                    code,
                    tile,
                    repeat,
                    symbols,
                    worker_barrier,
                    exchange_rows,
                    code_address,
                )?;
                if let Some(address) = repeat.profile.after {
                    emit_cycle_sample(code, symbols, address)?;
                }
            }
            TileStep::Checkpoint(checkpoint) => {
                code.instruction(PATCHED_BREAKPOINT_TRAP_BASE | u32::from(checkpoint.breakpoint))
            }
        }
        index += 1;
    }
    Ok(())
}

fn absolute_u16_copy(compute: &ComputeStep) -> Option<(u32, u32)> {
    let [TileAddress::Absolute(source)] = compute.input_addresses.as_slice() else {
        return None;
    };
    let TileAddress::Absolute(destination) = compute.output_address else {
        return None;
    };
    (compute.arguments.as_slice() == [1]).then_some((*source, destination))
}

fn step_compute_profile(step: &TileStep) -> Option<&StepProfile> {
    match step {
        TileStep::Compute(compute) => Some(&compute.profile),
        TileStep::Exchange(_) | TileStep::Repeat(_) | TileStep::Checkpoint(_) => None,
    }
}

fn validate(program: &TileProgram) -> Result<()> {
    validate_steps(&program.steps, None, None)
}

fn validate_steps(
    steps: &[TileStep],
    repeat_pointer_count: Option<usize>,
    repeat_count: Option<u32>,
) -> Result<()> {
    for step in steps {
        match step {
            TileStep::Exchange(exchange) => {
                validate_exchange_program(exchange)?;
                if exchange.setup_patch.as_ref().is_some_and(|patch| {
                    patch.offsets.words.is_empty()
                        || patch.offsets.words.len() != patch.values.words.len()
                }) {
                    return Err(invalid("exchange setup patch has an invalid shape"));
                }
                for patch in &exchange.repeat_patches {
                    if repeat_count.is_none_or(|count| patch.values.words.len() != count as usize)
                        || patch.word_offset as usize >= exchange.program.words.len()
                        || patch.values.address & 0b11 != 0
                    {
                        return Err(invalid("exchange patch has invalid shape or address"));
                    }
                }
            }
            TileStep::Compute(compute) => {
                if compute.symbol.is_empty() {
                    return Err(invalid("compute symbol is empty"));
                }
                let values = compute.input_addresses.len() + compute.arguments.len();
                let available = usize::from(LAST_VALUE_REGISTER - FIRST_INPUT_REGISTER + 1);
                if values == 0 || values > available {
                    return Err(invalid(format!(
                        "kernel {} needs {values} input/argument registers; 1..={available} are supported",
                        compute.symbol
                    )));
                }
                validate_address(compute.output_address, repeat_pointer_count)?;
                for &address in &compute.input_addresses {
                    validate_address(address, repeat_pointer_count)?;
                }
            }
            TileStep::Repeat(repeat) => {
                if repeat_pointer_count.is_some() {
                    return Err(invalid("nested finalized repeats are not yet supported"));
                }
                if repeat.count == 0 {
                    return Err(invalid("repeat count must be nonzero"));
                }
                validate_steps(
                    &repeat.body,
                    Some(repeat.iterated_pointers.len()),
                    Some(repeat.count),
                )?;
            }
            TileStep::Checkpoint(checkpoint) => {
                if checkpoint.breakpoint > 1 {
                    return Err(invalid("checkpoint breakpoint must be zero or one"));
                }
            }
        }
    }
    Ok(())
}

fn validate_exchange_program(exchange: &ExchangeStep) -> Result<()> {
    let embedded_sync = exchange
        .program
        .words
        .first()
        .is_some_and(|word| *word == SYNC_SUPERVISOR_INSTRUCTION);
    if exchange.program.address & 0b11 != 0
        || exchange.program.words.last() != Some(&ipu_exchange::RETURN_M10_INSTRUCTION)
        || embedded_sync != exchange.sync_in_program
        || exchange
            .program
            .words
            .iter()
            .skip(usize::from(embedded_sync))
            .any(|word| {
                matches!(
                    *word,
                    SANS_INACTIVE_INSTRUCTION | SYNC_SUPERVISOR_INSTRUCTION
                )
            })
    {
        return Err(invalid(
            "exchange phase has an invalid boundary or timed program",
        ));
    }
    if exchange.active != (exchange.program.words.len() > 1 + usize::from(embedded_sync)) {
        return Err(invalid(
            "exchange participation does not match timed program",
        ));
    }
    Ok(())
}

fn validate_address(address: TileAddress, repeat_pointer_count: Option<usize>) -> Result<()> {
    if let TileAddress::RepeatPointer { index, .. } = address
        && repeat_pointer_count.is_none_or(|count| usize::from(index) >= count)
    {
        return Err(invalid(
            "compute address refers to an unavailable repeat pointer",
        ));
    }
    Ok(())
}

fn active_exchange(step: &TileStep) -> bool {
    match step {
        TileStep::Exchange(exchange) => exchange.active,
        TileStep::Repeat(repeat) => repeat.body.iter().any(active_exchange),
        TileStep::Compute(_) | TileStep::Checkpoint(_) => false,
    }
}

fn emit_exchange_patches(
    code: &mut TileCode,
    exchange: &ExchangeStep,
    repeat_count: u32,
    symbols: &BTreeMap<String, u32>,
) -> Result<()> {
    let helper = symbol(symbols, PATCH_WORD_SYMBOL)?;
    for patch in &exchange.repeat_patches {
        let byte_offset = patch
            .word_offset
            .checked_mul(4)
            .ok_or_else(|| invalid("exchange patch offset overflow"))?;
        code.setzi(
            2,
            exchange
                .program
                .address
                .checked_add(byte_offset)
                .ok_or_else(|| invalid("exchange patch address overflow"))?,
        )?;
        code.setzi(3, patch.values.address)?;
        code.ld32(4, 11, 15, 0)?;
        code.setzi(5, repeat_count)?;
        code.call(helper, 9)?;
    }
    Ok(())
}

fn emit_exchange_setup_patch(
    code: &mut TileCode,
    exchange: &ExchangeStep,
    patch: &ExchangeSetupPatch,
    symbols: &BTreeMap<String, u32>,
) -> Result<()> {
    code.setzi(2, exchange.program.address)?;
    code.setzi(3, patch.offsets.address)?;
    code.setzi(4, patch.values.address)?;
    code.setzi(
        5,
        u32::try_from(patch.values.words.len())
            .map_err(|_| invalid("exchange setup patch is too large"))?,
    )?;
    code.call(symbol(symbols, PATCH_ROW_SYMBOL)?, 9)
}

fn emit_compute(
    code: &mut TileCode,
    tile: u16,
    compute: &ComputeStep,
    symbols: &BTreeMap<String, u32>,
    repeat_pointer_count: Option<usize>,
) -> Result<()> {
    let argument_base = FIRST_INPUT_REGISTER
        .checked_add(
            u8::try_from(compute.input_addresses.len())
                .map_err(|_| invalid("kernel input count exceeds u8"))?,
        )
        .ok_or_else(|| invalid("kernel input register overflow"))?;
    emit_address(code, 2, compute.output_address, repeat_pointer_count)?;
    for (index, &address) in compute.input_addresses.iter().enumerate() {
        emit_address(
            code,
            FIRST_INPUT_REGISTER
                + u8::try_from(index).map_err(|_| invalid("kernel input count exceeds u8"))?,
            address,
            repeat_pointer_count,
        )?;
    }
    for (index, &argument) in compute.arguments.iter().enumerate() {
        code.setzi(
            argument_base
                + u8::try_from(index).map_err(|_| invalid("kernel argument count exceeds u8"))?,
            argument,
        )?;
    }
    let kernel = symbols.get(&compute.symbol).copied().ok_or_else(|| {
        invalid(format!(
            "tile {tile} references missing kernel symbol {}",
            compute.symbol
        ))
    })?;
    code.call(kernel, 10)
}

fn emit_address(
    code: &mut TileCode,
    register: u8,
    address: TileAddress,
    repeat_pointer_count: Option<usize>,
) -> Result<()> {
    match address {
        TileAddress::Absolute(address) => code.setzi(register, address),
        TileAddress::RepeatPointer { index, offset } => {
            let count = repeat_pointer_count
                .ok_or_else(|| invalid("repeat pointer used outside repeat body"))?;
            if usize::from(index) >= count {
                return Err(invalid("repeat pointer index is out of range"));
            }
            code.ld32(register, 11, 15, index + 1)?;
            code.add_unsigned(register, offset)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_repeat(
    code: &mut TileCode,
    tile: u16,
    repeat: &RepeatStep,
    symbols: &BTreeMap<String, u32>,
    worker_barrier: Option<u32>,
    exchange_rows: &mut Vec<PlacedExchangeRow>,
    code_address: u32,
) -> Result<()> {
    let words = repeat
        .iterated_pointers
        .len()
        .checked_add(1)
        .ok_or_else(|| invalid("repeat frame size overflow"))?;
    let frame_bytes = i32::try_from((words * 4).next_multiple_of(8))
        .map_err(|_| invalid("repeat frame is too large"))?;
    code.add_immediate(11, 11, -frame_bytes)?;
    code.setzi(0, repeat.count)?;
    code.st32(0, 11, 15, 0)?;
    for (index, pointer) in repeat.iterated_pointers.iter().enumerate() {
        code.setzi(0, pointer.initial_address)?;
        code.st32(
            0,
            11,
            15,
            u16::try_from(index + 1).map_err(|_| invalid("too many repeat pointers"))?,
        )?;
    }
    let loop_start = code.address(code_address)?;
    emit_steps(
        code,
        tile,
        &repeat.body,
        symbols,
        worker_barrier,
        exchange_rows,
        Some(repeat.iterated_pointers.len()),
        Some(repeat.count),
        code_address,
    )?;
    for (index, pointer) in repeat.iterated_pointers.iter().enumerate() {
        let slot = u16::try_from(index + 1).map_err(|_| invalid("too many repeat pointers"))?;
        code.ld32(0, 11, 15, slot)?;
        code.add_unsigned(0, pointer.stride_bytes)?;
        code.st32(0, 11, 15, slot)?;
    }
    code.ld32(0, 11, 15, 0)?;
    code.add_immediate(0, 0, -1)?;
    code.st32(0, 11, 15, 0)?;
    let done_branch = code.words.len();
    code.brz(0, 0)?;
    code.jump(loop_start)?;
    let done = code.address(code_address)?;
    code.words[done_branch] = encode_brz_m_immediate(0, done)?;
    code.add_immediate(11, 11, frame_bytes)
}

fn emit_host_phases(
    code: &mut TileCode,
    symbols: &BTreeMap<String, u32>,
    phases: &[HostPhase],
) -> Result<()> {
    if phases.is_empty() {
        return Ok(());
    }
    let repeat_call = phases
        .iter()
        .any(|phase| !phase.active)
        .then(|| symbol(symbols, REPEAT_CALL_SYMBOL))
        .transpose()?;
    let host_run = phases
        .iter()
        .any(|phase| phase.active)
        .then(|| symbol(symbols, HOST_RUN_SYMBOL))
        .transpose()?;
    let mut index = 0;
    while index < phases.len() {
        let start = index;
        if phases[start].active {
            while index < phases.len()
                && phases[index].active
                && phases[index].address == phases[start].address
            {
                index += 1;
            }
            code.setzi(
                2,
                u32::try_from(index - start).map_err(|_| invalid("host run overflow"))?,
            )?;
            code.setzi(
                3,
                phases[start]
                    .run_table
                    .ok_or_else(|| invalid("active host phase has no run table"))?,
            )?;
            code.setzi(4, phases[start].address)?;
            code.call(host_run.expect("active host phase has host runner"), 9)?;
        } else {
            while index < phases.len() && !phases[index].active {
                index += 1;
            }
            code.setzi(
                2,
                u32::try_from(index - start).map_err(|_| invalid("host run overflow"))?,
            )?;
            code.setzi(3, phases[start].address)?;
            code.call(
                repeat_call.expect("inactive host phase has repeat helper"),
                9,
            )?;
        }
    }
    Ok(())
}

fn emit_cycle_sample(
    code: &mut TileCode,
    symbols: &BTreeMap<String, u32>,
    address: u32,
) -> Result<()> {
    code.setzi(2, address)?;
    code.call(symbol(symbols, SAMPLE_CYCLE_SYMBOL)?, 10)
}

fn symbol(symbols: &BTreeMap<String, u32>, name: &str) -> Result<u32> {
    symbols
        .get(name)
        .copied()
        .ok_or_else(|| invalid(format!("missing runtime symbol {name}")))
}

fn invalid(message: impl Into<String>) -> CodegenError {
    CodegenError::Invalid(message.into())
}

#[derive(Default)]
struct TileCode {
    words: Vec<u32>,
}

impl TileCode {
    fn address(&self, base: u32) -> Result<u32> {
        base.checked_add(
            u32::try_from(self.words.len())
                .map_err(|_| invalid("generated code exceeds u32"))?
                .checked_mul(4)
                .ok_or_else(|| invalid("generated code size overflow"))?,
        )
        .ok_or_else(|| invalid("generated code address overflow"))
    }

    fn setzi(&mut self, register: u8, immediate: u32) -> Result<()> {
        if immediate < 1 << 20 {
            self.words.push(encode_setzi_m(register, immediate)?);
        } else {
            self.words.push(encode_setzi_m(register, immediate >> 12)?);
            self.words
                .push(encode_shl_m_immediate(register, register, 12)?);
            self.words.push(encode_add_m_immediate(
                register,
                register,
                i32::from((immediate & 0xfff) as u16),
            )?);
        }
        Ok(())
    }

    fn instruction(&mut self, instruction: u32) {
        self.words.push(instruction);
    }

    fn ld32(&mut self, destination: u8, base: u8, delta: u8, offset: u16) -> Result<()> {
        self.words
            .push(encode_ld32_m_immediate(destination, base, delta, offset)?);
        Ok(())
    }

    fn st32(&mut self, source: u8, base: u8, delta: u8, offset: u16) -> Result<()> {
        self.words
            .push(encode_st32_m_immediate(source, base, delta, offset)?);
        Ok(())
    }

    fn add_immediate(&mut self, destination: u8, source: u8, immediate: i32) -> Result<()> {
        self.words
            .push(encode_add_m_immediate(destination, source, immediate)?);
        Ok(())
    }

    fn add_unsigned(&mut self, register: u8, mut immediate: u32) -> Result<()> {
        while immediate != 0 {
            let part = immediate.min(i16::MAX as u32);
            self.add_immediate(register, register, part as i32)?;
            immediate -= part;
        }
        Ok(())
    }

    fn put_special(&mut self, special: u8, register: u8) -> Result<()> {
        self.words.push(encode_put_special_m(special, register)?);
        Ok(())
    }

    fn call(&mut self, target: u32, return_register: u8) -> Result<()> {
        self.words
            .push(encode_call_m_immediate(return_register, target)?);
        Ok(())
    }

    fn brz(&mut self, register: u8, target: u32) -> Result<()> {
        self.words.push(encode_brz_m_immediate(register, target)?);
        Ok(())
    }

    fn jump(&mut self, target: u32) -> Result<()> {
        self.setzi(0, target)?;
        self.words.push(encode_br_m(0)?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symbols() -> BTreeMap<String, u32> {
        [
            (WORKER_BARRIER_SYMBOL.into(), 0x50000),
            (COMPLETE_SYMBOL.into(), 0x50004),
            (HOST_RUN_SYMBOL.into(), 0x50008),
            (REPEAT_CALL_SYMBOL.into(), 0x5000c),
            (SAMPLE_CYCLE_SYMBOL.into(), 0x50010),
            (PATCH_WORD_SYMBOL.into(), 0x50014),
            ("gemm".into(), 0x51000),
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn emits_resolved_exchange_and_compute_steps() {
        let program = TileProgram {
            tile: 7,
            steps: vec![
                TileStep::Exchange(ExchangeStep {
                    active: false,
                    incoming_base: 0,
                    preserve_base_registers: false,
                    incoming_mux: None,
                    incoming_format: 0,
                    incoming_mux_pair: None,
                    incoming_dcount: None,
                    sync_in_program: false,
                    program: PlacedExchangeRow {
                        address: 0x60000,
                        words: inactive_exchange_program(),
                    },
                    setup_patch: None,
                    repeat_patches: Vec::new(),
                    profile: StepProfile::default(),
                }),
                TileStep::Compute(ComputeStep {
                    symbol: "gemm".into(),
                    output_address: TileAddress::Absolute(0x70000),
                    input_addresses: vec![
                        TileAddress::Absolute(0x71000),
                        TileAddress::Absolute(0x72000),
                    ],
                    arguments: vec![64],
                    profile: StepProfile::default(),
                }),
            ],
        };
        let generated = emit(
            &program,
            &symbols(),
            &HostProgram::default(),
            &CodegenOptions {
                code_address: 0x52000,
                ..CodegenOptions::default()
            },
        )
        .unwrap();
        assert!(!generated.bytes.is_empty());
        assert_eq!(generated.exchange_rows.len(), 1);
        assert_eq!(generated.exchange_rows[0].address, 0x60000);
    }

    #[test]
    fn rejects_unresolved_or_malformed_inputs() {
        let program = TileProgram {
            tile: 0,
            steps: vec![TileStep::Exchange(ExchangeStep {
                active: false,
                incoming_base: 0,
                preserve_base_registers: false,
                incoming_mux: None,
                incoming_format: 0,
                incoming_mux_pair: None,
                incoming_dcount: None,
                sync_in_program: false,
                program: PlacedExchangeRow {
                    address: 3,
                    words: Vec::new(),
                },
                setup_patch: None,
                repeat_patches: Vec::new(),
                profile: StepProfile::default(),
            })],
        };
        assert!(matches!(
            emit(
                &program,
                &symbols(),
                &HostProgram::default(),
                &CodegenOptions::default()
            ),
            Err(CodegenError::Invalid(_))
        ));
    }

    #[test]
    fn randomized_repeat_patch_code_is_independent_of_iteration_count() {
        let mut random = fastrand::Rng::with_seed(0x7061_7463_685f_7265);
        let mut code_bytes = None;
        for _ in 0..64 {
            let count = random.u32(2..=128);
            let values = (0..count).map(|_| random.u32(..)).collect::<Vec<_>>();
            let program = TileProgram {
                tile: 0,
                steps: vec![TileStep::Repeat(RepeatStep {
                    count,
                    iterated_pointers: vec![RepeatPointer {
                        initial_address: 0x70000,
                        stride_bytes: 64,
                    }],
                    body: vec![TileStep::Exchange(ExchangeStep {
                        active: true,
                        incoming_base: 0x70000,
                        preserve_base_registers: false,
                        incoming_mux: None,
                        incoming_format: 0,
                        incoming_mux_pair: None,
                        incoming_dcount: None,
                        sync_in_program: false,
                        program: PlacedExchangeRow {
                            address: 0x60000,
                            words: vec![0, ipu_exchange::RETURN_M10_INSTRUCTION],
                        },
                        setup_patch: None,
                        repeat_patches: vec![ExchangePatch {
                            word_offset: 0,
                            values: PlacedExchangeRow {
                                address: 0x61000,
                                words: values.clone(),
                            },
                        }],
                        profile: StepProfile::default(),
                    })],
                    profile: StepProfile::default(),
                })],
            };
            let generated = emit(
                &program,
                &symbols(),
                &HostProgram::default(),
                &CodegenOptions {
                    code_address: 0x52000,
                    ..CodegenOptions::default()
                },
            )
            .unwrap();
            assert_eq!(generated.exchange_rows.len(), 2);
            assert_eq!(generated.exchange_rows[1].words, values);
            let emitted_words = generated
                .bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(
                emitted_words
                    .iter()
                    .filter(|word| **word == SYNC_SUPERVISOR_INSTRUCTION)
                    .count(),
                1
            );
            assert_eq!(
                *code_bytes.get_or_insert(generated.bytes.len()),
                generated.bytes.len()
            );
        }
    }
}
