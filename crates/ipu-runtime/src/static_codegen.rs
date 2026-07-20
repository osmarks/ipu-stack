use crate::Result;
use ipu_compiler::{LoweredTileProgram, LoweredTileStep};
use std::collections::BTreeMap;
use std::ops::Range;

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

#[derive(Clone, Debug)]
pub(crate) struct StaticTemplatePlan {
    pub name: String,
    pub instance_steps: Vec<Range<usize>>,
    pub record_addresses: Vec<u32>,
    pub records: Vec<Vec<StaticTemplateRecordWord>>,
    steps: Vec<StaticTemplateStep>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct StaticPlanPatch {
    pub word_address: u32,
    pub word_offset: u16,
    pub instruction: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum StaticTemplateRecordWord {
    Value(u32),
    Symbol(String),
}

#[derive(Clone, Debug)]
enum StaticTemplateStep {
    Exchange {
        sender_word_offset: Option<u16>,
        dynamic_sender_offset: bool,
        active: bool,
        dynamic_active: bool,
    },
    Compute {
        operation: String,
        operands: Vec<u32>,
        dynamic_operands: Vec<bool>,
        dynamic_kernel: bool,
        conditional: bool,
    },
    Idle,
}

pub(crate) fn plan_static_templates(
    program: &LoweredTileProgram,
    plan_addresses: &[u32],
    plan_patches: &[Option<StaticPlanPatch>],
    regions: &[crate::StaticTemplateRegion],
    mut cursor: u32,
) -> Result<(Vec<StaticTemplatePlan>, u32)> {
    let mut plan_by_step = vec![None; program.steps.len()];
    let mut patch_by_step = vec![None; program.steps.len()];
    let mut plans = plan_addresses.iter().copied();
    let mut patches = plan_patches.iter().copied();
    for (step_index, step) in program.steps.iter().enumerate() {
        if matches!(step, LoweredTileStep::Exchange { .. }) {
            plan_by_step[step_index] = plans.next();
            patch_by_step[step_index] = patches.next().flatten();
        }
    }
    if plans.next().is_some() {
        return Err("unused exchange plan while planning static templates".into());
    }
    if patches.next().is_some() {
        return Err("unused exchange plan patch while planning static templates".into());
    }

    let mut templates = Vec::with_capacity(regions.len());
    let mut previous_end = 0;
    for region in regions {
        if region.phase_instances.len() < 2 {
            return Err(format!("template {} requires at least two instances", region.name).into());
        }
        let instance_steps = region
            .phase_instances
            .iter()
            .map(|phases| phase_range_to_step_range(program, phases))
            .collect::<Result<Vec<_>>>()?;
        if instance_steps[0].start < previous_end
            || instance_steps
                .windows(2)
                .any(|pair| pair[0].end != pair[1].start)
        {
            return Err(format!(
                "template {} instances must be ordered, contiguous, and non-overlapping",
                region.name
            )
            .into());
        }
        let phase_count = region.phase_instances[0].len();
        if phase_count == 0
            || region
                .phase_instances
                .iter()
                .any(|phases| phases.len() != phase_count)
        {
            return Err(format!(
                "template {} instances have different phase counts",
                region.name
            )
            .into());
        }
        let mut records = vec![Vec::new(); instance_steps.len()];
        let mut template_steps = Vec::new();
        for relative_phase in 0..phase_count {
            let phase_steps = instance_steps
                .iter()
                .zip(&region.phase_instances)
                .map(|(steps, phases)| {
                    let phase = phases.start + relative_phase;
                    steps
                        .clone()
                        .filter(|&index| step_phase(&program.steps[index]) == phase)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let all_exchange = phase_steps.iter().all(|steps| {
                !steps.is_empty()
                    && steps.iter().all(|&index| {
                        matches!(program.steps[index], LoweredTileStep::Exchange { .. })
                    })
            });
            let all_compute = phase_steps.iter().all(|steps| {
                steps.iter().all(|&index| {
                    matches!(
                        program.steps[index],
                        LoweredTileStep::Compute(_) | LoweredTileStep::IdleCompute { .. }
                    )
                })
            });
            if all_exchange {
                let epoch_count = phase_steps[0].len();
                if phase_steps.iter().any(|steps| steps.len() != epoch_count) {
                    return Err(format!(
                        "template {} changes exchange epoch count in phase {relative_phase}",
                        region.name
                    )
                    .into());
                }
                for epoch in 0..epoch_count {
                    let actives = phase_steps
                        .iter()
                        .map(|steps| {
                            let LoweredTileStep::Exchange { row, .. } =
                                &program.steps[steps[epoch]]
                            else {
                                unreachable!();
                            };
                            row.first() != Some(&ipu_exchange::SANS_INACTIVE_INSTRUCTION)
                        })
                        .collect::<Vec<_>>();
                    let active = actives[0];
                    let dynamic_active = actives[1..].iter().any(|&value| value != active);
                    let sender_word_offsets = phase_steps
                        .iter()
                        .filter_map(|steps| {
                            patch_by_step[steps[epoch]].map(|patch| patch.word_offset)
                        })
                        .collect::<Vec<_>>();
                    let sender_word_offset = sender_word_offsets.first().copied();
                    let dynamic_sender_offset = sender_word_offsets.first().is_some_and(|first| {
                        sender_word_offsets.iter().any(|offset| offset != first)
                    });
                    for (instance, steps) in phase_steps.iter().enumerate() {
                        let step_index = steps[epoch];
                        let address = plan_by_step[step_index]
                            .ok_or("template exchange has no plan address")?;
                        if sender_word_offset.is_some() {
                            let patch = patch_by_step[step_index];
                            if dynamic_sender_offset {
                                records[instance].push(StaticTemplateRecordWord::Value(
                                    patch.map_or(0, |patch| patch.word_address),
                                ));
                            }
                            records[instance].push(StaticTemplateRecordWord::Value(
                                patch.map_or(0, |patch| patch.instruction),
                            ));
                        }
                        records[instance].push(StaticTemplateRecordWord::Value(address));
                        if dynamic_active {
                            records[instance].push(StaticTemplateRecordWord::Value(u32::from(
                                actives[instance],
                            )));
                        }
                    }
                    template_steps.push(StaticTemplateStep::Exchange {
                        sender_word_offset,
                        dynamic_sender_offset,
                        active,
                        dynamic_active,
                    });
                }
            } else if all_compute {
                let commands = phase_steps
                    .iter()
                    .map(|steps| {
                        steps
                            .iter()
                            .filter_map(|&index| {
                                let LoweredTileStep::Compute(command) = &program.steps[index]
                                else {
                                    return None;
                                };
                                Some(command)
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let command_count = commands.iter().map(Vec::len).max().unwrap_or(0);
                if command_count == 0 {
                    template_steps.push(StaticTemplateStep::Idle);
                } else {
                    for command_index in 0..command_count {
                        template_steps.push(plan_template_compute_step(
                            &commands,
                            command_index,
                            &mut records,
                            &region.name,
                            relative_phase,
                        )?);
                    }
                }
            } else {
                return Err(format!(
                    "template {} changes phase kind in phase {relative_phase}",
                    region.name
                )
                .into());
            }
        }
        let mut record_addresses = Vec::with_capacity(records.len());
        for record in &records {
            cursor = (cursor + 3) & !3;
            record_addresses.push(cursor);
            cursor = cursor
                .checked_add(
                    u32::try_from(record.len())?
                        .checked_mul(4)
                        .ok_or("static template record size overflow")?,
                )
                .ok_or("static template record address overflow")?;
        }
        previous_end = instance_steps.last().unwrap().end;
        templates.push(StaticTemplatePlan {
            name: region.name.clone(),
            instance_steps,
            record_addresses,
            records,
            steps: template_steps,
        });
    }
    Ok((templates, cursor))
}

fn plan_template_compute_step(
    commands: &[Vec<&ipu_compiler::LoweredComputeCommand>],
    command_index: usize,
    records: &mut [Vec<StaticTemplateRecordWord>],
    template_name: &str,
    relative_phase: usize,
) -> Result<StaticTemplateStep> {
    let active = commands
        .iter()
        .filter_map(|commands| commands.get(command_index).copied())
        .collect::<Vec<_>>();
    let first = active[0];
    if active.iter().any(|command| {
        command.input_addresses.len() != first.input_addresses.len()
            || command.arguments.len() != first.arguments.len()
    }) {
        return Err(format!(
            "template {template_name} changes compute ABI in phase {relative_phase} call {command_index}"
        )
        .into());
    }
    if first.input_addresses.len() != 2 || first.arguments.len() > KERNEL_ARGUMENT_REGISTERS {
        return Err(format!(
            "template {template_name} has unsupported compute ABI in phase {relative_phase} call {command_index}"
        )
        .into());
    }
    let operands = active
        .iter()
        .map(|command| {
            std::iter::once(command.output_address)
                .chain(command.input_addresses.iter().copied())
                .chain(command.arguments.iter().copied())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let dynamic_operands = (0..operands[0].len())
        .map(|operand| {
            operands[1..]
                .iter()
                .any(|values| values[operand] != operands[0][operand])
        })
        .collect::<Vec<_>>();
    let conditional = active.len() != commands.len();
    let dynamic_kernel = active
        .iter()
        .any(|command| command.specialization.operation != first.specialization.operation);
    for (record, instance_commands) in records.iter_mut().zip(commands) {
        let command = instance_commands.get(command_index).copied();
        if conditional {
            record.push(StaticTemplateRecordWord::Value(u32::from(
                command.is_some(),
            )));
        }
        if dynamic_kernel {
            match command {
                Some(command) => record.push(StaticTemplateRecordWord::Symbol(format!(
                    "ipu_stack_{}",
                    command.specialization.operation
                ))),
                None => record.push(StaticTemplateRecordWord::Value(0)),
            }
        }
        let values = command
            .map(|command| {
                std::iter::once(command.output_address)
                    .chain(command.input_addresses.iter().copied())
                    .chain(command.arguments.iter().copied())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| operands[0].clone());
        record.extend(
            values
                .iter()
                .zip(&dynamic_operands)
                .filter_map(|(&value, &dynamic)| {
                    dynamic.then_some(StaticTemplateRecordWord::Value(value))
                }),
        );
    }
    Ok(StaticTemplateStep::Compute {
        operation: first.specialization.operation.clone(),
        operands: operands[0].clone(),
        dynamic_operands,
        dynamic_kernel,
        conditional,
    })
}

fn phase_range_to_step_range(
    program: &LoweredTileProgram,
    phases: &Range<usize>,
) -> Result<Range<usize>> {
    if phases.start >= phases.end {
        return Err("static template phase range is empty".into());
    }
    let start = program
        .steps
        .iter()
        .position(|step| step_phase(step) >= phases.start)
        .unwrap_or(program.steps.len());
    let end = program
        .steps
        .iter()
        .position(|step| step_phase(step) >= phases.end)
        .unwrap_or(program.steps.len());
    if start == end
        || program.steps[start..end]
            .iter()
            .any(|step| !phases.contains(&step_phase(step)))
    {
        return Err("static template phase range does not map to contiguous tile steps".into());
    }
    Ok(start..end)
}

fn step_phase(step: &LoweredTileStep) -> usize {
    match step {
        LoweredTileStep::Exchange { phase, .. } | LoweredTileStep::IdleCompute { phase, .. } => {
            *phase
        }
        LoweredTileStep::Compute(command) => command.phase,
    }
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
    templates: &[StaticTemplatePlan],
    host: HostCode<'_>,
    profile: Option<&ProfileCode>,
    generated_base: u32,
) -> Result<Vec<u8>> {
    if let Some(profile) = profile
        && (profile.after_sync.len() != program.steps.len()
            || profile.after_step.len() != program.steps.len())
    {
        return Err("profile boundary count differs from tile step count".into());
    }
    if profile.is_some() && !templates.is_empty() {
        return Err("profiled code cannot use static templates".into());
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
    let mut template_index = 0;
    let mut template_calls = Vec::<(usize, usize)>::new();
    while step_index < program.steps.len() {
        if let Some(template) = templates.get(template_index)
            && template.instance_steps[0].start == step_index
        {
            for &record_address in &template.record_addresses {
                code.setzi(2, record_address)?;
                let call = code.words.len();
                code.call(0, 9)?;
                template_calls.push((call, template_index));
            }
            plan_index += template
                .instance_steps
                .iter()
                .flat_map(|range| &program.steps[range.clone()])
                .filter(|step| matches!(step, LoweredTileStep::Exchange { .. }))
                .count();
            step_index = template.instance_steps.last().unwrap().end;
            template_index += 1;
            continue;
        }
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
    let template_exchanges = if templates.is_empty() {
        None
    } else {
        let mut addresses = [0; 3];
        for (index, active) in [Some(false), Some(true), None].into_iter().enumerate() {
            addresses[index] = generated_address(generated_base, code.words.len())?;
            emit_static_template_exchange(&mut code, worker_barrier, generated_base, active)?;
        }
        Some(addresses)
    };
    let mut template_bodies = Vec::with_capacity(templates.len());
    for template in templates {
        template_bodies.push(code.words.len());
        emit_static_template_body(
            &mut code,
            template,
            symbols,
            template_exchanges.unwrap(),
            generated_base,
        )?;
    }
    for (call, template) in template_calls {
        let target = generated_address(generated_base, template_bodies[template])?;
        code.words[call] = ipu_exchange::encode_call_m_immediate(9, target)?;
    }
    Ok(code.words.into_iter().flat_map(u32::to_le_bytes).collect())
}

fn emit_static_template_exchange(
    code: &mut TileCode,
    worker_barrier: u32,
    generated_base: u32,
    active: Option<bool>,
) -> Result<()> {
    code.ld32(2, 11, 15, 0)?;
    code.ld32(8, 2, 15, 0)?;
    if active.is_none() {
        code.ld32(0, 2, 15, 1)?;
    } else {
        code.setzi(0, u32::from(active == Some(true)))?;
    }
    code.add_immediate(2, 2, if active.is_none() { 8 } else { 4 })?;
    code.st32(2, 11, 15, 0)?;
    code.instruction(ipu_exchange::SYNC_SUPERVISOR_INSTRUCTION);
    let skip_barrier = code.words.len();
    code.brz(0, 0)?;
    code.call(worker_barrier, 7)?;
    let after_barrier = generated_address(generated_base, code.words.len())?;
    code.words[skip_barrier] = ipu_exchange::encode_brz_m_immediate(0, after_barrier)?;
    let return_address = generated_address(generated_base, code.words.len() + 2)?;
    code.setzi(10, return_address)?;
    code.branch(8)?;
    code.branch(9)?;
    Ok(())
}

fn emit_static_template_body(
    code: &mut TileCode,
    template: &StaticTemplatePlan,
    symbols: &BTreeMap<String, u32>,
    template_exchanges: [u32; 3],
    generated_base: u32,
) -> Result<()> {
    code.add_immediate(11, 11, -16)?;
    code.st32(2, 11, 15, 0)?;
    code.st32(9, 11, 15, 1)?;
    for planned in &template.steps {
        match planned {
            StaticTemplateStep::Exchange {
                sender_word_offset,
                dynamic_sender_offset,
                active,
                dynamic_active,
            } => {
                if let Some(word_offset) = sender_word_offset {
                    code.ld32(1, 11, 15, 0)?;
                    if *dynamic_sender_offset {
                        code.ld32(8, 1, 15, 0)?;
                        code.ld32(3, 1, 15, 1)?;
                        code.add_immediate(1, 1, 8)?;
                    } else {
                        code.ld32(3, 1, 15, 0)?;
                        code.add_immediate(1, 1, 4)?;
                    }
                    code.st32(1, 11, 15, 0)?;
                    let skip_patch = code.words.len();
                    code.brz(3, 0)?;
                    if *dynamic_sender_offset {
                        code.st32(3, 8, 15, 0)?;
                    } else {
                        code.ld32(8, 1, 15, 0)?;
                        code.st32(3, 8, 15, *word_offset)?;
                    }
                    let after_patch = generated_address(generated_base, code.words.len())?;
                    code.words[skip_patch] = ipu_exchange::encode_brz_m_immediate(3, after_patch)?;
                }
                let helper = if *dynamic_active {
                    2
                } else {
                    usize::from(*active)
                };
                code.call(template_exchanges[helper], 9)?;
            }
            StaticTemplateStep::Compute {
                operation,
                operands,
                dynamic_operands,
                dynamic_kernel,
                conditional,
            } => {
                let dynamic_count = dynamic_operands.iter().filter(|&&dynamic| dynamic).count();
                let record_words =
                    dynamic_count + usize::from(*conditional) + usize::from(*dynamic_kernel);
                if record_words != 0 {
                    code.ld32(1, 11, 15, 0)?;
                }
                if *conditional {
                    code.ld32(0, 1, 15, 0)?;
                }
                let kernel_offset = u16::from(*conditional);
                if *dynamic_kernel {
                    code.ld32(8, 1, 15, kernel_offset)?;
                }
                let mut dynamic_offset = kernel_offset + u16::from(*dynamic_kernel);
                for (operand, (&value, &dynamic)) in
                    operands.iter().zip(dynamic_operands).enumerate()
                {
                    let register = u8::try_from(operand)? + 2;
                    if dynamic {
                        code.ld32(register, 1, 15, dynamic_offset)?;
                        dynamic_offset += 1;
                    } else {
                        code.setzi(register, value)?;
                    }
                }
                if record_words != 0 {
                    code.add_immediate(1, 1, i32::try_from(record_words * 4)?)?;
                    code.st32(1, 11, 15, 0)?;
                }
                let skip_call = if *conditional {
                    let branch = code.words.len();
                    code.brz(0, 0)?;
                    Some(branch)
                } else {
                    None
                };
                if *dynamic_kernel {
                    let return_address = generated_address(generated_base, code.words.len() + 2)?;
                    code.setzi(10, return_address)?;
                    code.branch(8)?;
                } else {
                    code.call(symbol(symbols, &format!("ipu_stack_{operation}"))?, 10)?;
                }
                if let Some(branch) = skip_call {
                    let after_call = generated_address(generated_base, code.words.len())?;
                    code.words[branch] = ipu_exchange::encode_brz_m_immediate(0, after_call)?;
                }
            }
            StaticTemplateStep::Idle => {}
        }
    }
    code.ld32(9, 11, 15, 1)?;
    code.add_immediate(11, 11, 16)?;
    code.branch(9)?;
    Ok(())
}

fn generated_address(base: u32, word: usize) -> Result<u32> {
    base.checked_add(
        u32::try_from(word)?
            .checked_mul(4)
            .ok_or("generated code offset overflow")?,
    )
    .ok_or_else(|| "generated code address overflow".into())
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

    fn ld32(&mut self, destination: u8, base: u8, delta: u8, offset: u16) -> Result<()> {
        self.words.push(ipu_exchange::encode_ld32_m_immediate(
            destination,
            base,
            delta,
            offset,
        )?);
        Ok(())
    }

    fn st32(&mut self, source: u8, base: u8, delta: u8, offset: u16) -> Result<()> {
        self.words.push(ipu_exchange::encode_st32_m_immediate(
            source, base, delta, offset,
        )?);
        Ok(())
    }

    fn add_immediate(&mut self, destination: u8, source: u8, immediate: i32) -> Result<()> {
        self.words.push(ipu_exchange::encode_add_m_immediate(
            destination,
            source,
            immediate,
        )?);
        Ok(())
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

    fn branch(&mut self, register: u8) -> Result<()> {
        self.words.push(ipu_exchange::encode_br_m(register)?);
        Ok(())
    }

    fn brz(&mut self, register: u8, target: u32) -> Result<()> {
        self.words
            .push(ipu_exchange::encode_brz_m_immediate(register, target)?);
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
        compute_with_operation(phase, input_address, arguments, "gemm_f16_accumulate")
    }

    fn compute_with_operation(
        phase: usize,
        input_address: u32,
        arguments: Vec<u32>,
        operation: &str,
    ) -> LoweredTileStep {
        LoweredTileStep::Compute(LoweredComputeCommand {
            op: OpId(phase),
            phase,
            output: TensorId(3),
            inputs: vec![TensorId(1), TensorId(2)],
            output_address: 0x80000,
            input_addresses: vec![0x50000, input_address],
            arguments,
            specialization: SpecializationKey {
                operation: operation.into(),
                shape: vec![12, 64, 64],
                worker_count: 6,
                role: "inner-block".into(),
                alignment: 32,
            },
            metadata: BTreeMap::new(),
        })
    }

    #[test]
    fn templates_align_rotated_tile_work_by_global_phase() {
        let program = LoweredTileProgram {
            tile: 7,
            steps: vec![
                exchange(0, true),
                compute_with_operation(1, 0x54000, Vec::new(), "gemm_f16_accumulate"),
                compute_with_operation(1, 0x58000, Vec::new(), "add_f16"),
                exchange(2, false),
                compute_with_operation(3, 0x5c000, Vec::new(), "add_f16"),
            ],
        };
        let regions = [crate::StaticTemplateRegion {
            name: "encoder_layer".into(),
            phase_instances: vec![0..2, 2..4],
        }];

        let patches = [
            Some(StaticPlanPatch {
                word_address: 0x52004,
                word_offset: 1,
                instruction: 0x7800_1238,
            }),
            None,
        ];
        let (templates, end) =
            plan_static_templates(&program, &[0x52000, 0x52020], &patches, &regions, 0x53002)
                .unwrap();

        assert_eq!(templates.len(), 1);
        let template = &templates[0];
        assert_eq!(template.name, "encoder_layer");
        assert_eq!(template.instance_steps, [0..3, 3..5]);
        assert_eq!(template.steps.len(), 3);
        assert!(matches!(
            template.steps[0],
            StaticTemplateStep::Exchange {
                sender_word_offset: Some(1),
                dynamic_sender_offset: false,
                active: true,
                dynamic_active: true,
            }
        ));
        assert!(matches!(
            template.steps[1],
            StaticTemplateStep::Compute {
                dynamic_kernel: true,
                conditional: false,
                ..
            }
        ));
        assert!(matches!(
            template.steps[2],
            StaticTemplateStep::Compute {
                conditional: true,
                ..
            }
        ));
        assert_eq!(template.records[0].len(), template.records[1].len());
        assert_eq!(
            end - template.record_addresses[0],
            2 * 4 * template.records[0].len() as u32
        );
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
