use crate::Result;
use ipu_compiler::{LoweredTileProgram, LoweredTileStep};
use std::collections::BTreeMap;

const INCOMING_DBASE: u8 = 0xa4;
const INCOMING_DCOUNT: u8 = 0xa6;
const INCOMING_SBASE: u8 = 0xa7;
const KERNEL_ARGUMENT_BASE: u8 = 5;
const KERNEL_ARGUMENT_REGISTERS: usize = 5;

pub(crate) const WORKER_BARRIER: &str = "ipu_stack_static_worker_barrier";
pub(crate) const COMPLETE: &str = "ipu_stack_static_complete";
pub(crate) const HOST_RUN: &str = "ipu_stack_static_host_run";
pub(crate) const REPEAT_CALL: &str = "ipu_stack_static_repeat_call";
pub(crate) const SAMPLE_CYCLE: &str = "ipu_stack_static_sample_cycle";
pub(crate) const SAMPLE_CYCLE_NEXT: &str = "ipu_stack_static_sample_cycle_next";

#[derive(Clone, Copy)]
pub(crate) struct HostPhaseCall {
    pub address: u32,
    pub active: bool,
    pub run_table: Option<u32>,
}

pub(crate) struct HostCode<'a> {
    pub inputs: &'a [HostPhaseCall],
    pub outputs: &'a [HostPhaseCall],
    pub run_state: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct ProfileCode {
    pub initial: u32,
    pub after_sync: Vec<bool>,
    pub after_step: Vec<bool>,
    pub aggregate_end: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StepCodeSize {
    pub exchange: usize,
    pub compute: usize,
}

pub(crate) fn step_code_size(program: &LoweredTileProgram) -> StepCodeSize {
    let mut size = StepCodeSize::default();
    for step in &program.steps {
        match step {
            LoweredTileStep::Exchange { row, .. } => {
                let active = row.first() != Some(&ipu_exchange::SANS_INACTIVE_INSTRUCTION);
                size.exchange += (2 + usize::from(active)) * 4;
            }
            LoweredTileStep::Compute(command) => {
                size.compute += (4 + command.arguments.len()) * 4;
            }
            LoweredTileStep::IdleCompute { .. } => {}
        }
    }
    size
}

pub(crate) fn emit(
    program: &LoweredTileProgram,
    symbols: &BTreeMap<String, u32>,
    plan_addresses: &[u32],
    host: HostCode<'_>,
    profile: Option<&ProfileCode>,
) -> Result<Vec<u8>> {
    if let Some(profile) = profile
        && (profile.after_sync.len() != program.steps.len()
            || profile.after_step.len() != program.steps.len())
    {
        return Err("profile boundary count differs from tile step count".into());
    }
    let mut code = TileCode::new();
    let worker_barrier = symbol(symbols, WORKER_BARRIER)?;
    emit_host_phases(&mut code, symbols, host.inputs, host.run_state)?;
    if program.steps.iter().any(|step| {
        matches!(
            step,
            LoweredTileStep::Exchange { row, .. }
                if row.first() != Some(&ipu_exchange::SANS_INACTIVE_INSTRUCTION)
        )
    }) {
        code.put_special(INCOMING_SBASE, 15)?;
        code.put_special(INCOMING_DBASE, 15)?;
        code.setzi(8, 1)?;
        code.put_special(INCOMING_DCOUNT, 8)?;
    }
    if let Some(profile) = profile {
        emit_cycle_sample(&mut code, symbols, profile.initial)?;
    }
    let mut plan_index = 0usize;
    for (step_index, step) in program.steps.iter().enumerate() {
        match step {
            LoweredTileStep::Exchange { row, .. } => {
                code.instruction(ipu_exchange::SYNC_SUPERVISOR_INSTRUCTION);
                emit_next_cycle_sample(
                    &mut code,
                    symbols,
                    profile.and_then(|profile| profile.after_sync.get(step_index).copied()),
                )?;
                let active = row.first() != Some(&ipu_exchange::SANS_INACTIVE_INSTRUCTION);
                if active {
                    code.call(worker_barrier, 7)?;
                }
                let target = plan_addresses
                    .get(plan_index)
                    .copied()
                    .ok_or("missing exchange plan address")?;
                plan_index += 1;
                code.call(target, 10)?;
            }
            LoweredTileStep::Compute(command) => {
                if command.input_addresses.len() != 2 {
                    return Err(format!(
                        "kernel {} on tile {} has {} inputs; the current ABI requires two",
                        command.specialization.operation,
                        program.tile,
                        command.input_addresses.len()
                    )
                    .into());
                }
                let kernel = symbol(
                    symbols,
                    &format!("ipu_stack_{}", command.specialization.operation),
                )?;
                code.setzi(2, command.output_address)?;
                code.setzi(3, command.input_addresses[0])?;
                code.setzi(4, command.input_addresses[1])?;
                if command.arguments.len() > KERNEL_ARGUMENT_REGISTERS {
                    return Err("kernel scalar arguments exceed the register ABI".into());
                }
                for (index, &argument) in command.arguments.iter().enumerate() {
                    code.setzi(u8::try_from(index)? + KERNEL_ARGUMENT_BASE, argument)?;
                }
                code.call(kernel, 10)?;
            }
            LoweredTileStep::IdleCompute { .. } => {}
        }
        emit_next_cycle_sample(
            &mut code,
            symbols,
            profile.and_then(|profile| profile.after_step.get(step_index).copied()),
        )?;
    }
    if plan_index != plan_addresses.len() {
        return Err("unused exchange plan address".into());
    }
    if let Some(address) = profile.and_then(|profile| profile.aggregate_end) {
        emit_cycle_sample(&mut code, symbols, address)?;
    }
    emit_host_phases(&mut code, symbols, host.outputs, host.run_state)?;
    code.jump(symbol(symbols, COMPLETE)?)?;
    Ok(code.words.into_iter().flat_map(u32::to_le_bytes).collect())
}

fn emit_next_cycle_sample(
    code: &mut TileCode,
    symbols: &BTreeMap<String, u32>,
    enabled: Option<bool>,
) -> Result<()> {
    if enabled == Some(true) {
        code.call(symbol(symbols, SAMPLE_CYCLE_NEXT)?, 10)?;
    }
    Ok(())
}

fn emit_cycle_sample(
    code: &mut TileCode,
    symbols: &BTreeMap<String, u32>,
    address: u32,
) -> Result<()> {
    code.setzi(2, address)?;
    code.call(symbol(symbols, SAMPLE_CYCLE)?, 10)
}

fn emit_host_phases(
    code: &mut TileCode,
    symbols: &BTreeMap<String, u32>,
    phases: &[HostPhaseCall],
    host_run_state: u32,
) -> Result<()> {
    let repeat_call = symbol(symbols, REPEAT_CALL)?;
    let mut index = 0;
    while index < phases.len() {
        if phases[index].active {
            let start = index;
            while index < phases.len() && phases[index].active {
                index += 1;
            }
            code.setzi(2, u32::try_from(index - start)?)?;
            code.setzi(
                3,
                phases[start]
                    .run_table
                    .ok_or("active host run has no descriptor table")?,
            )?;
            code.setzi(4, host_run_state)?;
            code.call(symbol(symbols, HOST_RUN)?, 9)?;
            continue;
        }
        let start = index;
        while index < phases.len() && !phases[index].active {
            index += 1;
        }
        code.setzi(2, u32::try_from(index - start)?)?;
        code.setzi(3, phases[start].address)?;
        code.call(repeat_call, 9)?;
    }
    Ok(())
}

fn symbol(symbols: &BTreeMap<String, u32>, name: &str) -> Result<u32> {
    symbols
        .get(name)
        .copied()
        .ok_or_else(|| format!("static runtime has no {name} symbol").into())
}

struct TileCode {
    words: Vec<u32>,
}

impl TileCode {
    fn new() -> Self {
        Self { words: Vec::new() }
    }

    fn setzi(&mut self, register: u8, immediate: u32) -> Result<()> {
        self.words
            .push(ipu_exchange::encode_setzi_m(register, immediate)?);
        Ok(())
    }

    fn instruction(&mut self, instruction: u32) {
        self.words.push(instruction);
    }

    fn put_special(&mut self, special: u8, register: u8) -> Result<()> {
        self.words
            .push(ipu_exchange::encode_put_special_m(special, register)?);
        Ok(())
    }

    fn call(&mut self, target: u32, return_register: u8) -> Result<()> {
        self.words.push(ipu_exchange::encode_call_m_immediate(
            return_register,
            target,
        )?);
        Ok(())
    }

    fn jump(&mut self, target: u32) -> Result<()> {
        self.setzi(0, target)?;
        self.words.push(ipu_exchange::encode_br_m(0)?);
        Ok(())
    }
}
