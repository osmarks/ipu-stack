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
pub(crate) const EXCHANGE_COMPUTE_RUN: &str = "ipu_stack_static_exchange_compute_run";
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
    pub compute_calls: usize,
    pub compute_argument_words: usize,
    pub fused_run: usize,
    pub fused_compute_calls: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct ExchangeComputeRun {
    pub start_step: usize,
    pub iterations: usize,
    pub table_address: u32,
    pub table_entries: Vec<u32>,
}

pub(crate) fn plan_exchange_compute_runs(
    program: &LoweredTileProgram,
    plan_addresses: &[u32],
    mut cursor: u32,
    enabled: bool,
) -> Result<(Vec<ExchangeComputeRun>, u32)> {
    if !enabled {
        return Ok((Vec::new(), cursor));
    }
    let mut plan_by_step = vec![None; program.steps.len()];
    let mut plans = plan_addresses.iter().copied();
    for (step_index, step) in program.steps.iter().enumerate() {
        if matches!(step, LoweredTileStep::Exchange { .. }) {
            plan_by_step[step_index] = plans.next();
        }
    }
    if plans.next().is_some() {
        return Err("unused exchange plan while finding compact compute runs".into());
    }
    let mut runs = Vec::new();
    let mut step_index = 0;
    while step_index + 3 < program.steps.len() {
        let (LoweredTileStep::Exchange { .. }, LoweredTileStep::Compute(first)) =
            (&program.steps[step_index], &program.steps[step_index + 1])
        else {
            step_index += 1;
            continue;
        };
        if !first.arguments.is_empty() || first.input_addresses.len() != 2 {
            step_index += 1;
            continue;
        }
        let mut end = step_index + 2;
        while end + 1 < program.steps.len() {
            let (LoweredTileStep::Exchange { .. }, LoweredTileStep::Compute(command)) =
                (&program.steps[end], &program.steps[end + 1])
            else {
                break;
            };
            if !same_compute_abi(first, command) {
                break;
            }
            end += 2;
        }
        let iterations = (end - step_index) / 2;
        if iterations < 2 {
            step_index += 1;
            continue;
        }
        cursor = (cursor + 3) & !3;
        let table_address = cursor;
        let mut table_entries = Vec::with_capacity(iterations);
        for exchange_step in (step_index..end).step_by(2) {
            let LoweredTileStep::Exchange { row, .. } = &program.steps[exchange_step] else {
                unreachable!();
            };
            let active = row.first() != Some(&ipu_exchange::SANS_INACTIVE_INSTRUCTION);
            let address = plan_by_step[exchange_step]
                .ok_or("compact compute run exchange has no plan address")?;
            if address & 3 != 0 {
                return Err("exchange plan address is not word aligned".into());
            }
            table_entries.push(address | u32::from(active));
        }
        cursor = cursor
            .checked_add(
                u32::try_from(table_entries.len())?
                    .checked_mul(4)
                    .ok_or("compact exchange/compute run table size overflow")?,
            )
            .ok_or("compact exchange/compute run table address overflow")?;
        runs.push(ExchangeComputeRun {
            start_step: step_index,
            iterations,
            table_address,
            table_entries,
        });
        step_index = end;
    }
    Ok((runs, cursor))
}

fn same_compute_abi(
    left: &ipu_compiler::LoweredComputeCommand,
    right: &ipu_compiler::LoweredComputeCommand,
) -> bool {
    left.specialization.operation == right.specialization.operation
        && left.output_address == right.output_address
        && left.input_addresses == right.input_addresses
        && left.arguments == right.arguments
}

pub(crate) fn step_code_size(
    program: &LoweredTileProgram,
    runs: &[ExchangeComputeRun],
) -> StepCodeSize {
    let mut size = StepCodeSize::default();
    let mut step_index = 0;
    let mut run_index = 0;
    while step_index < program.steps.len() {
        if let Some(run) = runs.get(run_index)
            && run.start_step == step_index
        {
            size.fused_run += 7 * 4;
            size.fused_compute_calls += run.iterations;
            step_index += run.iterations * 2;
            run_index += 1;
            continue;
        }
        let step = &program.steps[step_index];
        match step {
            LoweredTileStep::Exchange { row, .. } => {
                let active = row.first() != Some(&ipu_exchange::SANS_INACTIVE_INSTRUCTION);
                size.exchange += (2 + usize::from(active)) * 4;
            }
            LoweredTileStep::Compute(command) => {
                size.compute += (4 + command.arguments.len()) * 4;
                size.compute_calls += 1;
                size.compute_argument_words += command.arguments.len();
            }
            LoweredTileStep::IdleCompute { .. } => {}
        }
        step_index += 1;
    }
    size
}

pub(crate) fn emit(
    program: &LoweredTileProgram,
    symbols: &BTreeMap<String, u32>,
    plan_addresses: &[u32],
    compute_runs: &[ExchangeComputeRun],
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
    let mut step_index = 0;
    let mut run_index = 0;
    while step_index < program.steps.len() {
        if let Some(run) = compute_runs.get(run_index)
            && run.start_step == step_index
        {
            let LoweredTileStep::Compute(command) = &program.steps[step_index + 1] else {
                unreachable!("compact exchange/compute run does not start with compute ABI");
            };
            code.setzi(2, u32::try_from(run.iterations)?)?;
            code.setzi(3, run.table_address)?;
            code.setzi(4, command.output_address)?;
            code.setzi(5, command.input_addresses[0])?;
            code.setzi(6, command.input_addresses[1])?;
            code.setzi(
                7,
                symbol(
                    symbols,
                    &format!("ipu_stack_{}", command.specialization.operation),
                )?,
            )?;
            code.call(symbol(symbols, EXCHANGE_COMPUTE_RUN)?, 9)?;
            plan_index += run.iterations;
            step_index += run.iterations * 2;
            run_index += 1;
            continue;
        }
        let step = &program.steps[step_index];
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
        step_index += 1;
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

#[cfg(test)]
mod tests {
    use super::*;
    use ipu_compiler::{
        LoweredComputeCommand, LoweredTileProgram, LoweredTileStep, OpId, SpecializationKey,
        TensorId,
    };
    use std::collections::BTreeMap;

    fn exchange(phase: usize, active: bool) -> LoweredTileStep {
        LoweredTileStep::Exchange {
            phase,
            epoch: 0,
            row: vec![if active {
                ipu_exchange::sans(1)
            } else {
                ipu_exchange::SANS_INACTIVE_INSTRUCTION
            }],
        }
    }

    fn compute(phase: usize, input_address: u32, arguments: Vec<u32>) -> LoweredTileStep {
        LoweredTileStep::Compute(LoweredComputeCommand {
            op: OpId(phase),
            phase,
            output: TensorId(3),
            inputs: vec![TensorId(1), TensorId(2)],
            output_address: 0x80000,
            input_addresses: vec![0x50000, input_address],
            arguments,
            specialization: SpecializationKey {
                operation: "gemm_f16_accumulate".into(),
                shape: vec![12, 64, 64],
                worker_count: 6,
                role: "inner-block".into(),
                alignment: 32,
            },
            metadata: BTreeMap::new(),
        })
    }

    #[test]
    fn plans_fixed_compute_abi_run_with_per_exchange_table() {
        let program = LoweredTileProgram {
            tile: 7,
            steps: vec![
                exchange(0, true),
                compute(1, 0x54000, Vec::new()),
                exchange(2, false),
                compute(3, 0x54000, Vec::new()),
                exchange(4, true),
                compute(5, 0x54000, Vec::new()),
            ],
        };
        let plans = [0x52000, 0x52020, 0x52040];
        let (runs, end) = plan_exchange_compute_runs(&program, &plans, 0x53002, true).unwrap();

        assert_eq!(runs.len(), 1);
        let run = &runs[0];
        assert_eq!(run.start_step, 0);
        assert_eq!(run.iterations, 3);
        assert_eq!(run.table_address & 3, 0);
        assert_eq!(run.table_entries.len(), run.iterations);
        for ((entry, plan), active) in run.table_entries.iter().zip(plans).zip([true, false, true])
        {
            assert_eq!(entry & !1, plan);
            assert_eq!(entry & 1 != 0, active);
        }
        assert_eq!(end - run.table_address, run.iterations as u32 * 4);
    }

    #[test]
    fn leaves_irregular_or_argument_taking_calls_unrolled() {
        let irregular = LoweredTileProgram {
            tile: 7,
            steps: vec![
                exchange(0, true),
                compute(1, 0x54000, Vec::new()),
                exchange(2, true),
                compute(3, 0x58000, Vec::new()),
                exchange(4, true),
                compute(5, 0x58000, vec![64]),
                exchange(6, true),
                compute(7, 0x58000, vec![64]),
            ],
        };
        let (runs, end) = plan_exchange_compute_runs(
            &irregular,
            &[0x52000, 0x52020, 0x52040, 0x52060],
            0x53000,
            true,
        )
        .unwrap();

        assert!(runs.is_empty());
        assert_eq!(end, 0x53000);
    }

    #[test]
    fn profiling_can_disable_compaction_without_changing_plan_extent() {
        let program = LoweredTileProgram {
            tile: 7,
            steps: vec![
                exchange(0, true),
                compute(1, 0x54000, Vec::new()),
                exchange(2, true),
                compute(3, 0x54000, Vec::new()),
            ],
        };
        let (runs, end) =
            plan_exchange_compute_runs(&program, &[0x52000, 0x52020], 0x53000, false).unwrap();

        assert!(runs.is_empty());
        assert_eq!(end, 0x53000);
    }
}
