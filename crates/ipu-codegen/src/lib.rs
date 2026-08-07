use ipu_exchange::{
    PLAN_WORDS, SANS_INACTIVE_INSTRUCTION, SYNC_SUPERVISOR_INSTRUCTION, encode_add_m_immediate,
    encode_br_m, encode_brz_m_immediate, encode_call_m_immediate, encode_ld32_m_immediate,
    encode_put_special_m, encode_setzi_m, encode_shl_m_immediate, encode_st32_m_immediate,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub mod exchange;
pub mod graph;
pub mod kernel;
pub mod low;
pub mod mid;
mod package;
pub mod place;
pub mod storage;
pub use exchange::{ExchangeLoweringError, PhysicalExchangePhase, lower_exchanges};
pub use graph::{
    AddOptions, AttentionOptions, AttentionScale, BroadcastMode, ComputeGraph, GemmOptions,
    GraphError, GraphInput, GraphInputKind, GraphResult, Operation, OperationId, OperationKind,
    Region, RegionBuilder, Repeat, RepeatArguments, TensorShape, ValueId, ValueSequence,
    ValueSequenceId,
};
pub use kernel::{
    KernelAbi, KernelAbiError, KernelAvailability, KernelBuildPlan, KernelCompilation,
    KernelMaterializationError, KernelSymbols, PlannedKernelCall, ScalarArgument,
    materialize_kernel_run, tile_kernel_abi, validate_kernel_run,
};
pub use low::{
    ExchangePhase, ExchangePhaseId, KernelOperand, KernelRequirements, KernelRun, LogicalExchange,
    LowInput, LowLoweringError, LowLoweringResult, LowProgram, LowShard, LowShardId, LowValue,
    RepeatCarried, RepeatInvariant, RepeatIterated, RepeatRun, ShardDefinition, ShardExtent,
    ShardView, TileKernel, TileWork, TileWorkList, WorkProvenance, WorkReason, lower_to_tiles,
};
pub use mid::{
    AccumulationPrecision, AmpOrder, AxisTiling, ConversionDispatch, ConversionPlan, CostModel,
    ElementOrder, GemmKernelMode, HardwareTarget, Layout, LayoutError, LoweringError,
    LoweringResult, MemoryClass, MemoryOperand, MemoryRelation, MidGraph, MidInput, MidOperation,
    MidOperationKind, MidOperator, MidRegion, MidRepeat, MidValue, MidValueId, OperandRequirement,
    OperatorCandidate, OperatorDispatch, OperatorPlan, OperatorPlanError, OperatorRequirements,
    OutputAliasing, Padding, PipelineConfig, Precision, ProfilingConfig, SchedulingPolicy,
    TensorAxis, TensorFormat, TensorTiling, TensorType, TileKernelSpec, ToyCostModel, lower,
};
pub use package::{PackageBuildError, PackageBuildResult, PackageConfig, build_package};
pub use place::{IPU21_DATA_BASE, Placement, PlacementError, place};
pub use storage::{ByteSpan, StorageError, StorageResult, shard_storage_bytes, view_byte_spans};

const INCOMING_DBASE: u8 = 0xa4;
const INCOMING_DCOUNT: u8 = 0xa6;
const INCOMING_SBASE: u8 = 0xa7;
const FIRST_INPUT_REGISTER: u8 = 3;
const LAST_VALUE_REGISTER: u8 = 9;

pub const WORKER_BARRIER_SYMBOL: &str = "ipu_stack_static_worker_barrier";
pub const COMPLETE_SYMBOL: &str = "ipu_stack_static_complete";
pub const HOST_RUN_SYMBOL: &str = "ipu_stack_static_host_run";
pub const REPEAT_CALL_SYMBOL: &str = "ipu_stack_static_repeat_call";
pub const SAMPLE_CYCLE_SYMBOL: &str = "ipu_stack_static_sample_cycle";
pub const RUNTIME_ENTRY_SYMBOL: &str = "ipu_stack_static_start";
pub const PROGRAM_ADDRESS_SYMBOL: &str = "ipu_stack_static_program";
pub const WORKER_SYNC_CONTEXT_SYMBOL: &str = "ipu_stack_static_worker_sync_context";
pub const WORKER_STACK_BASE_SYMBOL: &str = "ipu_stack_static_worker_stack_base";
pub const PRNG_SEED_SYMBOL: &str = "ipu_stack_static_prng_seed";
pub const HOST_STAGING_SYMBOL: &str = "ipu_stack_static_host_staging";
pub const COMPLETION_ADDRESS_SYMBOL: &str = "ipu_stack_static_completion";

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Address at which `row` will be placed by the package builder.
    pub address: u32,
    /// Complete exchange row generated by `ipu-exchange`.
    pub row: Vec<u32>,
    #[serde(default)]
    pub profile: StepProfile,
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

#[derive(Clone, Debug, PartialEq, Eq)]
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

    if program.steps.iter().any(active_exchange) {
        code.put_special(INCOMING_SBASE, 15)?;
        code.put_special(INCOMING_DBASE, 15)?;
        code.setzi(8, 1)?;
        code.put_special(INCOMING_DCOUNT, 8)?;
    }
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
    exchange_rows: &mut Vec<PlacedExchangeRow>,
    repeat_pointer_count: Option<usize>,
    code_address: u32,
) -> Result<()> {
    for step in steps {
        match step {
            TileStep::Exchange(exchange) => {
                if let Some(address) = exchange.profile.before {
                    emit_cycle_sample(code, symbols, address)?;
                }
                code.instruction(SYNC_SUPERVISOR_INSTRUCTION);
                let active = exchange.row[0] != SANS_INACTIVE_INSTRUCTION;
                if active {
                    code.call(worker_barrier.expect("active exchange has barrier"), 7)?;
                }
                code.call(exchange.address, 10)?;
                if let Some(address) = exchange.profile.after {
                    emit_cycle_sample(code, symbols, address)?;
                }
                exchange_rows.push(PlacedExchangeRow {
                    address: exchange.address,
                    words: exchange.row.clone(),
                });
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
        }
    }
    Ok(())
}

fn validate(program: &TileProgram) -> Result<()> {
    validate_steps(&program.steps, None)
}

fn validate_steps(steps: &[TileStep], repeat_pointer_count: Option<usize>) -> Result<()> {
    for step in steps {
        match step {
            TileStep::Exchange(exchange) => {
                if exchange.address & 3 != 0 {
                    return Err(invalid("exchange row address is not word aligned"));
                }
                if exchange.row.len() != PLAN_WORDS {
                    return Err(invalid(format!(
                        "exchange row has {} words, expected {PLAN_WORDS}",
                        exchange.row.len()
                    )));
                }
            }
            TileStep::Compute(compute) => {
                if compute.symbol.is_empty() {
                    return Err(invalid("compute symbol is empty"));
                }
                let values = compute.input_addresses.len() + compute.arguments.len();
                let available = usize::from(LAST_VALUE_REGISTER - FIRST_INPUT_REGISTER + 1);
                if compute.input_addresses.is_empty() || values > available {
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
                validate_steps(&repeat.body, Some(repeat.iterated_pointers.len()))?;
            }
        }
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
        TileStep::Exchange(exchange) => exchange.row.first() != Some(&SANS_INACTIVE_INSTRUCTION),
        TileStep::Repeat(repeat) => repeat.body.iter().any(active_exchange),
        TileStep::Compute(_) => false,
    }
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
        code_address,
    )?;
    for (index, pointer) in repeat.iterated_pointers.iter().enumerate() {
        let offset = u16::try_from(index + 1).map_err(|_| invalid("too many repeat pointers"))?;
        code.ld32(0, 11, 15, offset)?;
        code.add_unsigned(0, pointer.stride_bytes)?;
        code.st32(0, 11, 15, offset)?;
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
            ("gemm".into(), 0x51000),
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn emits_resolved_exchange_and_compute_steps() {
        let mut row = vec![0; PLAN_WORDS];
        row[0] = SANS_INACTIVE_INSTRUCTION;
        let program = TileProgram {
            tile: 7,
            steps: vec![
                TileStep::Exchange(ExchangeStep {
                    address: 0x60000,
                    row,
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
                address: 3,
                row: Vec::new(),
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
    fn structured_repeat_code_size_is_independent_of_iteration_count() {
        let program = |count| TileProgram {
            tile: 0,
            steps: vec![TileStep::Repeat(RepeatStep {
                count,
                iterated_pointers: vec![RepeatPointer {
                    initial_address: 0x70000,
                    stride_bytes: 0x10000,
                }],
                body: vec![TileStep::Compute(ComputeStep {
                    symbol: "gemm".into(),
                    output_address: TileAddress::Absolute(0x60000),
                    input_addresses: vec![
                        TileAddress::Absolute(0x68000),
                        TileAddress::RepeatPointer {
                            index: 0,
                            offset: 32,
                        },
                    ],
                    arguments: Vec::new(),
                    profile: StepProfile::default(),
                })],
                profile: StepProfile::default(),
            })],
        };
        let short = emit(
            &program(2),
            &symbols(),
            &HostProgram::default(),
            &CodegenOptions::default(),
        )
        .unwrap();
        let long = emit(
            &program(10_000),
            &symbols(),
            &HostProgram::default(),
            &CodegenOptions::default(),
        )
        .unwrap();
        assert_eq!(short.bytes.len(), long.bytes.len());
        assert!(short.exchange_rows.is_empty());
    }
}
