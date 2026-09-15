//! Emit supervisor machine code from an address-resolved tile program.
//! Program validation is completed before instruction emission starts.
use crate::kernel::{FIRST_INPUT_REGISTER, OUTPUT_REGISTER, RETURN_REGISTER};
use ipu_target::ipu21::instruction::{
    SANS_INACTIVE_INSTRUCTION, SYNC_SUPERVISOR_INSTRUCTION, encode_add_m_immediate, encode_br_m,
    encode_brz_m_immediate, encode_call_m_immediate, encode_ld32_m_immediate, encode_put_special_m,
    encode_setzi_m, encode_shl_m_immediate, encode_st32_m_immediate,
};
use ipu_target::ipu21::registers::{
    INCOMING_BASE, INCOMING_DCOUNT, INCOMING_FORMAT, INCOMING_MUX, INCOMING_MUXPAIR, OUTGOING_BASE,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
mod validate;

// Recovered primitive PIC/XPIC plans arm A6 with one; their payload length is
// encoded in the timed instructions rather than this external-stream counter.
// Consolidated phases currently preserve that primitive-plan setting.
const INTERNAL_EXCHANGE_DCOUNT: u32 = 1;
const LAST_VALUE_REGISTER: u8 = 9;

pub const WORKER_BARRIER_SYMBOL: &str = "ipu_stack_static_worker_barrier";
pub const COMPLETE_SYMBOL: &str = "ipu_stack_static_complete";
pub const COMPLETED_SYMBOL: &str = "ipu_stack_static_completed";
pub const HOST_RUN_SYMBOL: &str = "ipu_stack_static_host_run";
pub const REPEAT_CALL_SYMBOL: &str = "ipu_stack_static_repeat_call";
pub const SAMPLE_CYCLE_SYMBOL: &str = "ipu_stack_static_sample_cycle";
pub const COPY_U16_SYMBOL: &str = "static_copy_u16";
pub const COPY_U32_SYMBOL: &str = "static_copy_u32";
pub const COPY_U64_SYMBOL: &str = "copy_u64";
pub const COPY_STRIDED_U32_SYMBOL: &str = "copy_strided_u32";
pub const COPY_STRIDED_U64_SYMBOL: &str = "copy_strided_u64";
pub const FILL_ZERO_U64_SYMBOL: &str = "fill_zero_u64";
pub const PATCH_REPEAT_TABLES_SYMBOL: &str = "static_patch_repeat_tables";
pub const PATCH_REPEAT_ARITHMETIC_SYMBOL: &str = "static_patch_repeat_arithmetic";
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
    #[error(transparent)]
    Instruction(#[from] ipu_target::ipu21::instruction::InstructionError),
    #[error("exchange encoding failed: {0}")]
    Exchange(#[from] crate::exchange::ExchangeError),
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
        offset: i32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExchangeStep {
    /// Whether this tile executes a timed send/receive program after the boundary.
    pub active: bool,
    /// Base address used by point-to-point receive rows.
    pub incoming_base: u32,
    /// Source base for a row encoded relative to a current Repeat parameter.
    #[serde(default)]
    pub outgoing_base: Option<TileAddress>,
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

impl ExchangeStep {
    /// Ordinary timed exchange; specialized receive controls and patches are opt-in.
    pub fn new(active: bool, incoming_base: u32, program: PlacedExchangeRow) -> Self {
        Self {
            active,
            incoming_base,
            program,
            outgoing_base: None,
            preserve_base_registers: false,
            incoming_mux: None,
            incoming_format: 0,
            incoming_mux_pair: None,
            incoming_dcount: None,
            sync_in_program: false,
            setup_patch: None,
            repeat_patches: Vec::new(),
            profile: StepProfile::default(),
        }
    }
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
    pub values: ExchangePatchValues,
}

/// Replacement instruction words, represented exactly rather than approximately.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExchangePatchValues {
    Table(PlacedExchangeRow),
    Arithmetic { initial: u32, step: u32 },
}

pub(crate) fn arithmetic_progression(words: &[u32]) -> Option<(u32, u32)> {
    if words.len() < 3 {
        return None;
    }
    let step = words[1].wrapping_sub(words[0]);
    words
        .windows(2)
        .all(|pair| pair[1].wrapping_sub(pair[0]) == step)
        .then_some((words[0], step))
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
    let exchange_rows = validate::program(program, host, options)?;

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
        emit_cycle_sample_at(&mut code, symbols, address, None, options.code_address)?;
    }

    let worker_barrier = program
        .steps
        .iter()
        .any(active_exchange)
        .then(|| symbol(symbols, WORKER_BARRIER_SYMBOL))
        .transpose()?;
    emit_steps(
        &mut code,
        program.tile,
        &program.steps,
        symbols,
        worker_barrier,
        None,
        None,
        options.code_address,
    )?;

    if let Some(address) = options.final_profile_address {
        emit_cycle_sample_at(&mut code, symbols, address, None, options.code_address)?;
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
    // Read-only descriptors follow the non-returning completion branch. Keeping
    // them with generated code avoids reserving writable exchange-row space.
    for (position, register, words) in std::mem::take(&mut code.literals) {
        code.words[position] = encode_setzi_m(register, code.address(options.code_address)?)?;
        code.words.extend(words);
    }

    Ok(GeneratedProgram {
        bytes: code.words.into_iter().flat_map(u32::to_le_bytes).collect(),
        exchange_rows,
    })
}

#[allow(clippy::too_many_arguments)]
fn emit_steps(
    code: &mut TileCode,
    tile: u16,
    steps: &[TileStep],
    symbols: &BTreeMap<String, u32>,
    worker_barrier: Option<u32>,
    repeat_count: Option<u32>,
    profile_enabled_slot: Option<u16>,
    code_address: u32,
) -> Result<()> {
    for step in steps {
        let profile = match step {
            TileStep::Exchange(step) => step.profile,
            TileStep::Compute(step) => step.profile,
            TileStep::Repeat(step) => step.profile,
            // Checkpoints have no cycle samples in the supervisor ABI.
            TileStep::Checkpoint(_) => StepProfile::default(),
        };
        if let Some(address) = profile.before {
            emit_cycle_sample_at(code, symbols, address, profile_enabled_slot, code_address)?;
        }
        match step {
            TileStep::Exchange(exchange) => {
                if let Some(patch) = &exchange.setup_patch {
                    emit_exchange_setup_patch(code, exchange, patch, symbols)?;
                }
                if !exchange.repeat_patches.is_empty() {
                    emit_exchange_patches(
                        code,
                        exchange,
                        repeat_count.expect("validated repeat patch"),
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
                    if let Some(base) = exchange.outgoing_base {
                        // Keep the moving base available to timed row sections;
                        // subsequent receive/down-count setup reuses m8.
                        emit_address(code, 6, base)?;
                        code.put_special(OUTGOING_BASE, 6)?;
                    } else {
                        code.put_special(OUTGOING_BASE, 15)?;
                    }
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
                } else if !exchange.active {
                    code.instruction(SANS_INACTIVE_INSTRUCTION);
                    code.instruction(ipu_target::ipu21::instruction::SYNC_ANS_INSTRUCTION);
                }
                code.call(exchange.program.address, 10)?;
            }
            TileStep::Compute(compute) => {
                emit_compute(code, tile, compute, symbols)?;
            }
            TileStep::Repeat(repeat) => {
                emit_repeat(code, tile, repeat, symbols, worker_barrier, code_address)?;
            }
            TileStep::Checkpoint(checkpoint) => {
                code.instruction(PATCHED_BREAKPOINT_TRAP_BASE | u32::from(checkpoint.breakpoint))
            }
        }
        if let Some(address) = profile.after {
            emit_cycle_sample_at(code, symbols, address, profile_enabled_slot, code_address)?;
        }
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
    code.ld32(4, 11, 15, 0)?;
    code.setzi(5, repeat_count)?;
    let mut tables = Vec::new();
    let mut arithmetic = Vec::new();
    for patch in &exchange.repeat_patches {
        let address = patch
            .word_offset
            .checked_mul(4)
            .and_then(|offset| exchange.program.address.checked_add(offset))
            .ok_or_else(|| invalid("exchange patch address overflow"))?;
        match &patch.values {
            ExchangePatchValues::Table(row) => tables.extend([address, row.address]),
            ExchangePatchValues::Arithmetic { initial, step } => {
                arithmetic.extend([address, *initial, *step]);
            }
        }
    }
    for (words, width, helper) in [
        (tables, 2, PATCH_REPEAT_TABLES_SYMBOL),
        (arithmetic, 3, PATCH_REPEAT_ARITHMETIC_SYMBOL),
    ] {
        if words.is_empty() {
            continue;
        }
        let count =
            u32::try_from(words.len() / width).map_err(|_| invalid("too many exchange patches"))?;
        code.literals.push((code.words.len(), 2, words));
        code.setzi(2, 0)?;
        code.setzi(3, count)?;
        code.call(symbol(symbols, helper)?, 9)?;
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
) -> Result<()> {
    let argument_base = FIRST_INPUT_REGISTER + compute.input_addresses.len() as u8;
    emit_address(code, OUTPUT_REGISTER, compute.output_address)?;
    for (index, &address) in compute.input_addresses.iter().enumerate() {
        emit_address(code, FIRST_INPUT_REGISTER + index as u8, address)?;
    }
    for (index, &argument) in compute.arguments.iter().enumerate() {
        code.setzi(argument_base + index as u8, argument)?;
    }
    let kernel = symbols.get(&compute.symbol).copied().ok_or_else(|| {
        invalid(format!(
            "tile {tile} references missing kernel symbol {}",
            compute.symbol
        ))
    })?;
    code.call(kernel, RETURN_REGISTER)
}

fn emit_address(code: &mut TileCode, register: u8, address: TileAddress) -> Result<()> {
    match address {
        TileAddress::Absolute(address) => code.setzi(register, address),
        TileAddress::RepeatPointer { index, offset } => {
            code.ld32(register, 11, 15, index + 1)?;
            code.add_offset(register, i64::from(offset))
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
    code_address: u32,
) -> Result<()> {
    let has_profile = repeat.body.iter().any(|step| {
        let profile = match step {
            TileStep::Compute(step) => step.profile,
            TileStep::Exchange(step) => step.profile,
            TileStep::Repeat(step) => step.profile,
            TileStep::Checkpoint(step) => step.profile,
        };
        profile.before.is_some() || profile.after.is_some()
    });
    let profile_slot = (repeat.iterated_pointers.len() + 1) as u16;
    let words = repeat.iterated_pointers.len() + 1 + usize::from(has_profile);
    let frame_bytes = (words * 4).next_multiple_of(8) as i32;
    code.add_immediate(11, 11, -frame_bytes)?;
    code.setzi(0, repeat.count)?;
    code.st32(0, 11, 15, 0)?;
    for (index, pointer) in repeat.iterated_pointers.iter().enumerate() {
        code.setzi(0, pointer.initial_address)?;
        code.st32(0, 11, 15, (index + 1) as u16)?;
    }
    if has_profile {
        code.setzi(0, 1)?;
        code.st32(0, 11, 15, profile_slot)?;
    }
    let loop_start = code.address(code_address)?;
    emit_steps(
        code,
        tile,
        &repeat.body,
        symbols,
        worker_barrier,
        Some(repeat.count),
        has_profile.then_some(profile_slot),
        code_address,
    )?;
    for (index, pointer) in repeat.iterated_pointers.iter().enumerate() {
        let slot = (index + 1) as u16;
        code.ld32(0, 11, 15, slot)?;
        code.add_offset(0, i64::from(pointer.stride_bytes))?;
        code.st32(0, 11, 15, slot)?;
    }
    if has_profile {
        // Preserve first-iteration samples; subsequent iterations skip sampling.
        code.setzi(0, 0)?;
        code.st32(0, 11, 15, profile_slot)?;
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
    for run in phases.chunk_by(|a, b| a.active == b.active && (!a.active || a.address == b.address))
    {
        let first = &run[0];
        code.setzi(
            2,
            u32::try_from(run.len()).map_err(|_| invalid("host run overflow"))?,
        )?;
        if first.active {
            code.setzi(3, first.run_table.expect("validated active host phase"))?;
            code.setzi(4, first.address)?;
            code.call(host_run.expect("active host phase has host runner"), 9)?;
        } else {
            code.setzi(3, first.address)?;
            code.call(
                repeat_call.expect("inactive host phase has repeat helper"),
                9,
            )?;
        }
    }
    Ok(())
}

fn emit_cycle_sample_at(
    code: &mut TileCode,
    symbols: &BTreeMap<String, u32>,
    address: u32,
    profile_enabled_slot: Option<u16>,
    code_address: u32,
) -> Result<()> {
    let skip = if let Some(slot) = profile_enabled_slot {
        code.ld32(2, 11, 15, slot)?;
        let branch = code.words.len();
        code.brz(2, 0)?;
        Some(branch)
    } else {
        None
    };
    code.setzi(2, address)?;
    code.call(symbol(symbols, SAMPLE_CYCLE_SYMBOL)?, 10)?;
    if let Some(branch) = skip {
        code.words[branch] = encode_brz_m_immediate(2, code.address(code_address)?)?;
    }
    Ok(())
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
    literals: Vec<(usize, u8, Vec<u32>)>,
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

    fn add_offset(&mut self, register: u8, mut immediate: i64) -> Result<()> {
        while immediate != 0 {
            let part = immediate.clamp(i64::from(i16::MIN), i64::from(i16::MAX));
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
mod tests;
