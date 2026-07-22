use crate::Result;
use ipu_compiler::{LoweredTileProgram, LoweredTileStep};
use rustc_hash::{FxHashMap as HashMap, FxHasher};
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::ops::Range;

const INCOMING_DBASE: u8 = 0xa4;
const INCOMING_DCOUNT: u8 = 0xa6;
const INCOMING_SBASE: u8 = 0xa7;
const KERNEL_FIRST_INPUT_REGISTER: u8 = 3;
const KERNEL_LAST_VALUE_REGISTER: u8 = 9;

pub(crate) const WORKER_BARRIER: &str = "ipu_stack_static_worker_barrier";
pub(crate) const COMPLETE: &str = "ipu_stack_static_complete";
pub(crate) const HOST_RUN: &str = "ipu_stack_static_host_run";
pub(crate) const REPEAT_CALL: &str = "ipu_stack_static_repeat_call";
pub(crate) const EXCHANGE_COMPUTE_RUN: &str = "ipu_stack_static_exchange_compute_run";
pub(crate) const TEMPLATE_PATCH: &str = "ipu_stack_static_template_patch";
pub(crate) const SAMPLE_CYCLE: &str = "ipu_stack_static_sample_cycle";
pub(crate) const SAMPLE_CYCLE_NEXT: &str = "ipu_stack_static_sample_cycle_next";

#[derive(Clone, Copy)]
pub(crate) struct HostPhaseCall {
    pub address: u32,
    pub active: bool,
    pub run_table: Option<u32>,
}

pub(crate) struct HostCode<'a> {
    pub weights: &'a [HostPhaseCall],
    pub inputs: &'a [HostPhaseCall],
    pub outputs: &'a [HostPhaseCall],
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
    pub record_secondary_addresses: Vec<u32>,
    pub record_split: u16,
    pub records: Vec<Vec<StaticTemplateRecordWord>>,
    pub patch_addresses: Vec<Vec<u32>>,
    pub patches: Vec<Vec<(u16, StaticTemplatePatchValue)>>,
    pub shared_address: u32,
    pub shared: Vec<StaticTemplateRecordWord>,
    pub exchange_step_count: usize,
    steps: Vec<StaticTemplateStep>,
}

pub(crate) fn compact_template_instances(
    program: &mut LoweredTileProgram,
    templates: &mut [StaticTemplatePlan],
) -> Result<()> {
    let mut removed = 0usize;
    for template in templates {
        let original_start = template.instance_steps[0].start;
        let original_end = template.instance_steps.last().unwrap().end;
        let start = original_start - removed;
        let end = original_end - removed;
        template.exchange_step_count = program.steps[start..end]
            .iter()
            .filter(|step| matches!(step, LoweredTileStep::Exchange { .. }))
            .count();
        let phase = step_phase(
            program
                .steps
                .get(start)
                .ok_or("template compaction starts outside the tile program")?,
        );
        program.steps.splice(
            start..end,
            [LoweredTileStep::IdleCompute {
                op: ipu_compiler::OpId(usize::MAX),
                phase,
            }],
        );
        removed += end - start - 1;
        template.instance_steps.clear();
        template.instance_steps.push(start..start + 1);
    }
    Ok(())
}

pub(crate) fn template_retained_symbols(template: &StaticTemplatePlan) -> Vec<String> {
    let mut symbols = template
        .steps
        .iter()
        .filter_map(|step| match step {
            StaticTemplateStep::Compute { operation, .. } => Some(format!("ipu_stack_{operation}")),
            _ => None,
        })
        .collect::<Vec<_>>();
    symbols.extend(
        template
            .records
            .iter()
            .flatten()
            .chain(&template.shared)
            .filter_map(|word| match word {
                StaticTemplateRecordWord::Symbol(symbol) => Some(symbol.clone()),
                StaticTemplateRecordWord::Value(_) => None,
            }),
    );
    symbols
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct StaticPlanPatch {
    pub word_address: u32,
    pub word_offset: u16,
    pub instruction: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum StaticTemplateRecordWord {
    Value(u32),
    Symbol(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum StaticTemplatePatchValue {
    Delta(i16),
    Word(StaticTemplateRecordWord),
}

pub(crate) fn template_patch_storage_words_range(
    slots: Range<usize>,
    patch: &[(u16, StaticTemplatePatchValue)],
) -> usize {
    let (narrow, wide) = patch
        .iter()
        .filter(|(slot, _)| slots.contains(&usize::from(*slot)))
        .fold((0usize, 0usize), |(narrow, wide), (_, value)| match value {
            StaticTemplatePatchValue::Delta(_) => (narrow + 1, wide),
            StaticTemplatePatchValue::Word(_) => (narrow, wide + 1),
        });
    let Some(span) = template_patch_group_span(slots, patch) else {
        return 0;
    };
    let changed = narrow + wide;
    1 + span.len().div_ceil(32) + changed.div_ceil(32) + narrow.div_ceil(2) + wide
}

pub(crate) fn template_patch_group_span(
    slots: Range<usize>,
    patch: &[(u16, StaticTemplatePatchValue)],
) -> Option<Range<usize>> {
    let mut changed = patch
        .iter()
        .map(|(slot, _)| usize::from(*slot))
        .filter(|slot| slots.contains(slot))
        .map(|slot| slot - slots.start);
    let first = changed.next()?;
    let (first, last) = changed.fold((first, first), |(first, last), slot| {
        (first.min(slot), last.max(slot))
    });
    Some((first / 32 * 32)..((last + 1).div_ceil(32) * 32).min(slots.len()))
}

pub(crate) fn template_patch_ranges(record_words: usize, split: usize) -> [Range<usize>; 4] {
    fn halves(range: Range<usize>) -> [Range<usize>; 2] {
        let local_midpoint = (range.len() / 2).div_ceil(32) * 32;
        let midpoint = range.start + local_midpoint.min(range.len());
        [range.start..midpoint, midpoint..range.end]
    }

    let [primary_first, primary_second] = halves(0..split);
    let [secondary_first, secondary_second] = halves(split..record_words);
    [
        primary_first,
        primary_second,
        secondary_first,
        secondary_second,
    ]
}

#[derive(Clone, Debug)]
enum StaticTemplateStep {
    Exchange {
        sender_word_offset: Option<u16>,
        sender_address: Option<TemplateValue>,
        sender_instruction: Option<TemplateValue>,
        plan_words: Vec<(u16, TemplateValue)>,
        plan_address: TemplateValue,
        active: TemplateValue,
    },
    Compute {
        operation: String,
        operands: Vec<TemplateValue>,
        kernel: Option<TemplateValue>,
        condition: Option<TemplateValue>,
    },
    Idle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TemplateValue {
    Constant(u32),
    Record(u16),
    RecordOffset { slot: u16, offset: i32 },
    Shared(u16),
}

struct TemplateRecords {
    rows: Vec<Vec<StaticTemplateRecordWord>>,
    columns: HashMap<u64, Vec<(Vec<StaticTemplateRecordWord>, u16)>>,
    affine_columns: HashMap<u64, Vec<(Vec<i64>, u16, u32)>>,
    shared: Vec<StaticTemplateRecordWord>,
    shared_values: HashMap<StaticTemplateRecordWord, u16>,
}

impl TemplateRecords {
    fn new(instances: usize) -> Self {
        Self {
            rows: vec![Vec::new(); instances],
            columns: HashMap::default(),
            affine_columns: HashMap::default(),
            shared: Vec::new(),
            shared_values: HashMap::default(),
        }
    }

    fn values(&mut self, values: impl IntoIterator<Item = u32>) -> Result<TemplateValue> {
        let values = values.into_iter().collect::<SmallVec<[u32; 32]>>();
        if values.windows(2).all(|pair| pair[0] == pair[1]) {
            return if values[0] < 1 << 20 {
                Ok(TemplateValue::Constant(values[0]))
            } else {
                self.shared(StaticTemplateRecordWord::Value(values[0]))
            };
        }
        let exact_hash = hash_value_words(&values);
        if let Some((_, slot)) = self.columns.get(&exact_hash).and_then(|entries| {
            entries.iter().find(|(key, _)| {
                key.len() == values.len()
                    && key
                        .iter()
                        .zip(&values)
                        .all(|(word, value)| word == &StaticTemplateRecordWord::Value(*value))
            })
        }) {
            return Ok(TemplateValue::Record(*slot));
        }
        let first = values[0];
        let normalized = || {
            values
                .iter()
                .map(|&value| i64::from(value) - i64::from(first))
        };
        let mut hasher = FxHasher::default();
        normalized().for_each(|value| value.hash(&mut hasher));
        let affine_hash = hasher.finish();
        if let Some((_, slot, previous_first)) =
            self.affine_columns.get(&affine_hash).and_then(|entries| {
                entries
                    .iter()
                    .find(|(key, _, _)| key.iter().copied().eq(normalized()))
            })
            && let Ok(offset) = i32::try_from(i64::from(first) - i64::from(*previous_first))
            && add_immediate_steps(offset) < self.rows.len()
        {
            return Ok(TemplateValue::RecordOffset {
                slot: *slot,
                offset,
            });
        }
        let words = values
            .iter()
            .copied()
            .map(StaticTemplateRecordWord::Value)
            .collect::<Vec<_>>();
        let slot = self.push_column(words.clone())?;
        self.columns
            .entry(exact_hash)
            .or_default()
            .push((words, slot));
        self.affine_columns.entry(affine_hash).or_default().push((
            normalized().collect(),
            slot,
            first,
        ));
        Ok(TemplateValue::Record(slot))
    }

    fn words(&mut self, words: Vec<StaticTemplateRecordWord>) -> Result<TemplateValue> {
        if words.windows(2).all(|pair| pair[0] == pair[1]) {
            return self.shared(words[0].clone());
        }
        let hash = hash_record_words(&words);
        if let Some((_, slot)) = self
            .columns
            .get(&hash)
            .and_then(|entries| entries.iter().find(|(key, _)| key == &words))
        {
            return Ok(TemplateValue::Record(*slot));
        }
        let slot = self.push_column(words.clone())?;
        self.columns.entry(hash).or_default().push((words, slot));
        Ok(TemplateValue::Record(slot))
    }

    fn shared(&mut self, word: StaticTemplateRecordWord) -> Result<TemplateValue> {
        if let Some(&slot) = self.shared_values.get(&word) {
            return Ok(TemplateValue::Shared(slot));
        }
        let slot = u16::try_from(self.shared.len())?;
        self.shared.push(word.clone());
        self.shared_values.insert(word, slot);
        Ok(TemplateValue::Shared(slot))
    }

    fn push_column(&mut self, words: Vec<StaticTemplateRecordWord>) -> Result<u16> {
        let slot = u16::try_from(self.rows.first().map_or(0, Vec::len))?;
        for (row, word) in self.rows.iter_mut().zip(words) {
            row.push(word);
        }
        Ok(slot)
    }
}

fn hash_value_words(values: &[u32]) -> u64 {
    let mut hasher = FxHasher::default();
    for value in values {
        0u8.hash(&mut hasher);
        value.hash(&mut hasher);
    }
    hasher.finish()
}

fn hash_record_words(words: &[StaticTemplateRecordWord]) -> u64 {
    let mut hasher = FxHasher::default();
    words.hash(&mut hasher);
    hasher.finish()
}

fn add_immediate_steps(offset: i32) -> usize {
    if offset >= 0 {
        usize::try_from(offset).unwrap().div_ceil(i16::MAX as usize)
    } else {
        usize::try_from(-i64::from(offset))
            .unwrap()
            .div_ceil(usize::try_from(-i32::from(i16::MIN)).unwrap())
    }
}

pub(crate) fn plan_static_templates(
    program: &LoweredTileProgram,
    plan_addresses: &[u32],
    plan_rows: &[Vec<u32>],
    plan_patches: &[Option<StaticPlanPatch>],
    regions: &[crate::StaticTemplateRegion],
    mut cursor: u32,
    cyclic: bool,
) -> Result<(Vec<StaticTemplatePlan>, u32)> {
    let mut plan_by_step = vec![None; program.steps.len()];
    let mut row_by_step = vec![None; program.steps.len()];
    let mut patch_by_step = vec![None; program.steps.len()];
    let mut plans = plan_addresses.iter().copied();
    let mut rows = plan_rows.iter().map(Vec::as_slice);
    let mut patches = plan_patches.iter().copied();
    for (step_index, step) in program.steps.iter().enumerate() {
        if matches!(step, LoweredTileStep::Exchange { .. }) {
            plan_by_step[step_index] = plans.next();
            row_by_step[step_index] = rows.next();
            patch_by_step[step_index] = patches.next().flatten();
        }
    }
    if plans.next().is_some() {
        return Err("unused exchange plan while planning static templates".into());
    }
    if rows.next().is_some() {
        return Err("unused exchange plan row while planning static templates".into());
    }
    if patches.next().is_some() {
        return Err("unused exchange plan patch while planning static templates".into());
    }

    let mut templates = Vec::with_capacity(regions.len());
    let mut previous_end = 0;
    for region in regions {
        if region.phase_instances.is_empty() {
            return Err(format!("template {} has no instances", region.name).into());
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
        let instance_phase_steps = instance_steps
            .iter()
            .zip(&region.phase_instances)
            .map(|(steps, phases)| phase_step_ranges(program, steps, phases))
            .collect::<Result<Vec<_>>>()?;
        let mut records = TemplateRecords::new(instance_steps.len());
        let mut template_steps = Vec::new();
        for relative_phase in 0..phase_count {
            let phase_steps = instance_phase_steps
                .iter()
                .map(|phases| phases[relative_phase].clone())
                .collect::<Vec<_>>();
            let all_exchange = phase_steps.iter().all(|steps| {
                !steps.is_empty()
                    && steps.clone().all(|index| {
                        matches!(program.steps[index], LoweredTileStep::Exchange { .. })
                    })
            });
            let all_compute = phase_steps.iter().all(|steps| {
                steps.clone().all(|index| {
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
                                &program.steps[steps.start + epoch]
                            else {
                                unreachable!();
                            };
                            row.first() != Some(&ipu_exchange::SANS_INACTIVE_INSTRUCTION)
                        })
                        .collect::<Vec<_>>();
                    let sender_patches = phase_steps
                        .iter()
                        .map(|steps| patch_by_step[steps.start + epoch])
                        .collect::<Vec<_>>();
                    let sender_word_offset = sender_patches
                        .iter()
                        .flatten()
                        .map(|patch| patch.word_offset)
                        .next();
                    let dynamic_sender_offset = sender_patches
                        .iter()
                        .flatten()
                        .any(|patch| Some(patch.word_offset) != sender_word_offset);
                    let sender_address = dynamic_sender_offset
                        .then(|| {
                            records.values(
                                sender_patches
                                    .iter()
                                    .map(|patch| patch.map_or(0, |patch| patch.word_address)),
                            )
                        })
                        .transpose()?;
                    let sender_instruction = sender_word_offset
                        .map(|_| {
                            records.values(
                                sender_patches
                                    .iter()
                                    .map(|patch| patch.map_or(0, |patch| patch.instruction)),
                            )
                        })
                        .transpose()?;
                    let instance_rows = phase_steps
                        .iter()
                        .map(|steps| {
                            row_by_step[steps.start + epoch]
                                .ok_or_else(|| "template exchange has no normalized row".into())
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let mut plan_words = Vec::new();
                    let plan_word_count = instance_rows.iter().map(|row| row.len()).max().unwrap();
                    for word in 0..plan_word_count {
                        let values = instance_rows
                            .iter()
                            .map(|row| row.get(word).copied().unwrap_or(0))
                            .collect::<SmallVec<[u32; 32]>>();
                        if values.windows(2).any(|pair| pair[0] != pair[1]) {
                            plan_words.push((u16::try_from(word)?, records.values(values)?));
                        }
                    }
                    let plan_address = records.values(
                        phase_steps
                            .iter()
                            .map(|steps| {
                                plan_by_step[steps.start + epoch]
                                    .ok_or_else(|| "template exchange has no plan address".into())
                            })
                            .collect::<Result<Vec<_>>>()?,
                    )?;
                    let active = records.values(actives.into_iter().map(u32::from))?;
                    template_steps.push(StaticTemplateStep::Exchange {
                        sender_word_offset,
                        sender_address,
                        sender_instruction,
                        plan_words,
                        plan_address,
                        active,
                    });
                }
            } else if all_compute {
                let commands = phase_steps
                    .iter()
                    .map(|steps| {
                        program.steps[steps.clone()]
                            .iter()
                            .filter_map(|step| {
                                let LoweredTileStep::Compute(command) = step else {
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
        previous_end = instance_steps.last().unwrap().end;
        let record_split = u16::try_from(
            records
                .rows
                .first()
                .map_or(0, |record| record.len().div_ceil(2)),
        )?;
        let mut patches = vec![Vec::new()];
        for pair in records.rows.windows(2) {
            patches.push(template_record_patch(&pair[0], &pair[1])?);
        }
        if cyclic {
            patches.push(template_record_patch(
                records.rows.last().unwrap(),
                records.rows.first().unwrap(),
            )?);
        }
        cursor = (cursor + 3) & !3;
        let primary_address = cursor;
        cursor = cursor
            .checked_add(u32::from(record_split) * 4)
            .ok_or("static template record address overflow")?;
        cursor = (cursor + 3) & !3;
        let secondary_address = cursor;
        let record_words = records.rows.first().map_or(0, Vec::len);
        cursor = cursor
            .checked_add(u32::try_from(record_words - usize::from(record_split))? * 4)
            .ok_or("static template record address overflow")?;
        let patch_ranges = template_patch_ranges(record_words, usize::from(record_split));
        let mut patch_addresses = Vec::with_capacity(patches.len());
        for patch in &patches {
            let mut addresses = Vec::with_capacity(patch_ranges.len());
            for slots in &patch_ranges {
                cursor = (cursor + 3) & !3;
                addresses.push(cursor);
                cursor = cursor
                    .checked_add(
                        u32::try_from(template_patch_storage_words_range(slots.clone(), patch))?
                            .checked_mul(4)
                            .ok_or("static template patch size overflow")?,
                    )
                    .ok_or("static template patch address overflow")?;
            }
            patch_addresses.push(addresses);
        }
        templates.push(StaticTemplatePlan {
            name: region.name.clone(),
            instance_steps,
            record_addresses: vec![primary_address; records.rows.len()],
            record_secondary_addresses: vec![secondary_address; records.rows.len()],
            record_split,
            records: records.rows,
            patch_addresses,
            patches,
            shared_address: 0,
            shared: records.shared,
            exchange_step_count: 0,
            steps: template_steps,
        });
    }
    Ok((templates, cursor))
}

fn template_record_patch(
    previous: &[StaticTemplateRecordWord],
    next: &[StaticTemplateRecordWord],
) -> Result<Vec<(u16, StaticTemplatePatchValue)>> {
    previous
        .iter()
        .zip(next)
        .enumerate()
        .filter(|(_, (previous, next))| previous != next)
        .map(|(slot, (previous, next))| {
            let value = match (previous, next) {
                (
                    StaticTemplateRecordWord::Value(previous),
                    StaticTemplateRecordWord::Value(next),
                ) if i16::try_from(i64::from(*next) - i64::from(*previous)).is_ok() => {
                    StaticTemplatePatchValue::Delta(
                        (i64::from(*next) - i64::from(*previous)) as i16,
                    )
                }
                _ => StaticTemplatePatchValue::Word(next.clone()),
            };
            Ok((u16::try_from(slot)?, value))
        })
        .collect()
}

fn plan_template_compute_step(
    commands: &[Vec<&ipu_compiler::LoweredComputeCommand>],
    command_index: usize,
    records: &mut TemplateRecords,
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
    let value_registers = first.input_addresses.len() + first.arguments.len();
    if first.input_addresses.is_empty()
        || value_registers
            > usize::from(KERNEL_LAST_VALUE_REGISTER - KERNEL_FIRST_INPUT_REGISTER + 1)
    {
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
                .collect::<SmallVec<[u32; 16]>>()
        })
        .collect::<Vec<_>>();
    let conditional = active.len() != commands.len();
    let dynamic_kernel = active
        .iter()
        .any(|command| command.specialization.operation != first.specialization.operation);
    let condition = conditional
        .then(|| {
            records.values(
                commands
                    .iter()
                    .map(|commands| u32::from(commands.get(command_index).is_some())),
            )
        })
        .transpose()?;
    let kernel = dynamic_kernel
        .then(|| {
            records.words(
                commands
                    .iter()
                    .map(|commands| match commands.get(command_index) {
                        Some(command) => StaticTemplateRecordWord::Symbol(format!(
                            "ipu_stack_{}",
                            command.specialization.operation
                        )),
                        None => StaticTemplateRecordWord::Value(0),
                    })
                    .collect(),
            )
        })
        .transpose()?;
    let operands = (0..operands[0].len())
        .map(|operand| {
            records.values(commands.iter().map(|commands| {
                commands
                    .get(command_index)
                    .map(|command| {
                        std::iter::once(command.output_address)
                            .chain(command.input_addresses.iter().copied())
                            .chain(command.arguments.iter().copied())
                            .nth(operand)
                            .unwrap()
                    })
                    .unwrap_or(operands[0][operand])
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StaticTemplateStep::Compute {
        operation: first.specialization.operation.to_string(),
        operands,
        kernel,
        condition,
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
        .partition_point(|step| step_phase(step) < phases.start);
    let end = program
        .steps
        .partition_point(|step| step_phase(step) < phases.end);
    if start == end
        || program.steps[start..end]
            .iter()
            .any(|step| !phases.contains(&step_phase(step)))
    {
        return Err("static template phase range does not map to contiguous tile steps".into());
    }
    Ok(start..end)
}

fn phase_step_ranges(
    program: &LoweredTileProgram,
    steps: &Range<usize>,
    phases: &Range<usize>,
) -> Result<Vec<Range<usize>>> {
    let mut ranges = Vec::with_capacity(phases.len());
    let mut cursor = steps.start;
    for phase in phases.clone() {
        let start = cursor;
        while cursor < steps.end && step_phase(&program.steps[cursor]) == phase {
            cursor += 1;
        }
        ranges.push(start..cursor);
    }
    if cursor != steps.end {
        return Err("static template steps are not ordered by phase".into());
    }
    Ok(ranges)
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
                size.compute += (2 + command.input_addresses.len() + command.arguments.len()) * 4;
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
    invocations: u32,
) -> Result<Vec<u8>> {
    if invocations == 0 {
        return Err("static graph invocation count must be nonzero".into());
    }
    if let Some(profile) = profile
        && (profile.after_sync.len() != program.steps.len()
            || profile.after_step.len() != program.steps.len())
    {
        return Err("profile boundary count differs from tile step count".into());
    }
    if let Some(profile) = profile {
        for template in templates {
            let first = &template.instance_steps[0];
            let after_sync = &profile.after_sync[first.clone()];
            let after_step = &profile.after_step[first.clone()];
            if after_sync.iter().any(|&sample| sample) {
                return Err("static templates cannot profile exchange sync boundaries".into());
            }
            for instance in &template.instance_steps[1..] {
                if profile.after_sync[instance.clone()] != *after_sync
                    || profile.after_step[instance.clone()] != *after_step
                {
                    return Err(
                        "static template instances have different profile boundaries".into(),
                    );
                }
            }
        }
    }
    let mut code = TileCode::new();
    let worker_barrier = symbol(symbols, WORKER_BARRIER)?;
    emit_host_phases(&mut code, symbols, host.weights)?;
    if invocations > 1 {
        code.add_immediate(11, 11, -8)?;
        code.setzi(0, invocations)?;
        code.st32(0, 11, 15, 0)?;
    }
    let invocation_start = generated_address(generated_base, code.words.len())?;
    emit_host_phases(&mut code, symbols, host.inputs)?;
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
            let record_words = template.records[0].len();
            let record_split = usize::from(template.record_split);
            code.add_immediate(11, 11, -16)?;
            code.setzi(4, template.record_addresses[0])?;
            code.setzi(5, template.record_secondary_addresses[0])?;
            code.setzi(6, u32::from(template.record_split))?;
            code.setzi(7, u32::try_from(record_words - record_split)?)?;
            code.st32(4, 11, 15, 0)?;
            code.st32(5, 11, 15, 1)?;
            code.st32(6, 11, 15, 2)?;
            code.st32(7, 11, 15, 3)?;
            for instance in 0..template.records.len() {
                let patch = &template.patches[instance];
                for (part, slots) in template_patch_ranges(record_words, record_split)
                    .into_iter()
                    .enumerate()
                {
                    if template_patch_storage_words_range(slots, patch) == 0 {
                        continue;
                    }
                    code.setzi(7, u32::from(part >= 2))?;
                    code.setzi(3, template.patch_addresses[instance][part])?;
                    code.call(symbol(symbols, TEMPLATE_PATCH)?, 9)?;
                }
                let call = code.words.len();
                code.call(0, 9)?;
                template_calls.push((call, template_index));
            }
            if template.patches.len() > template.records.len() {
                let reset = template.patches.len() - 1;
                for (part, slots) in template_patch_ranges(record_words, record_split)
                    .into_iter()
                    .enumerate()
                {
                    if template_patch_storage_words_range(slots, &template.patches[reset]) == 0 {
                        continue;
                    }
                    code.setzi(7, u32::from(part >= 2))?;
                    code.setzi(3, template.patch_addresses[reset][part])?;
                    code.call(symbol(symbols, TEMPLATE_PATCH)?, 9)?;
                }
            }
            code.add_immediate(11, 11, 16)?;
            plan_index += if template.exchange_step_count == 0 {
                template
                    .instance_steps
                    .iter()
                    .flat_map(|range| &program.steps[range.clone()])
                    .filter(|step| matches!(step, LoweredTileStep::Exchange { .. }))
                    .count()
            } else {
                template.exchange_step_count
            };
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
                let argument_base = KERNEL_FIRST_INPUT_REGISTER
                    .checked_add(u8::try_from(command.input_addresses.len())?)
                    .ok_or("kernel input register overflow")?;
                let value_registers = command.input_addresses.len() + command.arguments.len();
                if command.input_addresses.is_empty()
                    || value_registers
                        > usize::from(KERNEL_LAST_VALUE_REGISTER - KERNEL_FIRST_INPUT_REGISTER + 1)
                {
                    return Err(format!(
                        "kernel {} on tile {} needs {} input/argument registers; at most {} are available",
                        command.specialization.operation,
                        program.tile,
                        value_registers,
                        KERNEL_LAST_VALUE_REGISTER - KERNEL_FIRST_INPUT_REGISTER + 1,
                    )
                    .into());
                }
                let kernel = symbol(
                    symbols,
                    &format!("ipu_stack_{}", command.specialization.operation),
                )?;
                code.setzi(2, command.output_address)?;
                for (index, &address) in command.input_addresses.iter().enumerate() {
                    code.setzi(KERNEL_FIRST_INPUT_REGISTER + u8::try_from(index)?, address)?;
                }
                for (index, &argument) in command.arguments.iter().enumerate() {
                    code.setzi(argument_base + u8::try_from(index)?, argument)?;
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
    emit_host_phases(&mut code, symbols, host.outputs)?;
    if invocations > 1 {
        code.ld32(0, 11, 15, 0)?;
        code.add_immediate(0, 0, -1)?;
        code.st32(0, 11, 15, 0)?;
        let done_branch = code.words.len();
        code.brz(0, 0)?;
        code.jump(invocation_start)?;
        let done = generated_address(generated_base, code.words.len())?;
        code.words[done_branch] = ipu_exchange::encode_brz_m_immediate(0, done)?;
        code.add_immediate(11, 11, 8)?;
    }
    code.jump(symbol(symbols, COMPLETE)?)?;
    let template_exchanges = if templates.is_empty() {
        None
    } else {
        let address = generated_address(generated_base, code.words.len())?;
        emit_static_template_exchange(&mut code, worker_barrier, generated_base)?;
        Some(address)
    };
    let mut template_bodies = Vec::with_capacity(templates.len());
    for template in templates {
        template_bodies.push(code.words.len());
        let profile_after_step = profile.map(|profile| {
            let first = &template.instance_steps[0];
            &profile.after_step[first.clone()]
        });
        emit_static_template_body(
            &mut code,
            template,
            symbols,
            template_exchanges.unwrap(),
            generated_base,
            template
                .record_addresses
                .windows(2)
                .all(|pair| pair[0] == pair[1])
                && template
                    .record_secondary_addresses
                    .windows(2)
                    .all(|pair| pair[0] == pair[1]),
            profile_after_step,
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
) -> Result<()> {
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

fn emit_template_value(
    code: &mut TileCode,
    value: TemplateValue,
    register: u8,
    shared_address: u32,
    record_split: u16,
) -> Result<()> {
    match value {
        TemplateValue::Constant(value) => code.setzi(register, value),
        TemplateValue::Record(slot) => {
            let (base, offset) = if slot < record_split {
                (0, slot)
            } else {
                (1, slot - record_split)
            };
            code.ld32(1, 11, 15, base)?;
            code.ld32(register, 1, 15, offset)
        }
        TemplateValue::RecordOffset { slot, offset } => {
            let (base, slot) = if slot < record_split {
                (0, slot)
            } else {
                (1, slot - record_split)
            };
            code.ld32(1, 11, 15, base)?;
            code.ld32(register, 1, 15, slot)?;
            let mut remaining = offset;
            while remaining != 0 {
                let step = remaining.clamp(i32::from(i16::MIN), i32::from(i16::MAX));
                code.add_immediate(register, register, step)?;
                remaining -= step;
            }
            Ok(())
        }
        TemplateValue::Shared(slot) => {
            code.setzi(1, shared_address)?;
            code.ld32(register, 1, 15, slot)
        }
    }
}

fn emit_static_template_body(
    code: &mut TileCode,
    template: &StaticTemplatePlan,
    symbols: &BTreeMap<String, u32>,
    template_exchange: u32,
    generated_base: u32,
    record_addresses_in_parent_frame: bool,
    profile_after_step: Option<&[bool]>,
) -> Result<()> {
    code.add_immediate(11, 11, -16)?;
    if record_addresses_in_parent_frame {
        code.ld32(2, 11, 15, 4)?;
        code.ld32(3, 11, 15, 5)?;
    }
    code.st32(2, 11, 15, 0)?;
    code.st32(3, 11, 15, 1)?;
    code.st32(9, 11, 15, 2)?;
    for (step_index, planned) in template.steps.iter().enumerate() {
        match planned {
            StaticTemplateStep::Exchange {
                sender_word_offset,
                sender_address,
                sender_instruction,
                plan_words,
                plan_address,
                active,
            } => {
                if !plan_words.is_empty() {
                    emit_template_value(
                        code,
                        *plan_address,
                        8,
                        template.shared_address,
                        template.record_split,
                    )?;
                    for &(word, value) in plan_words {
                        emit_template_value(
                            code,
                            value,
                            3,
                            template.shared_address,
                            template.record_split,
                        )?;
                        code.st32(3, 8, 15, word)?;
                    }
                }
                if let Some(instruction) = sender_instruction {
                    emit_template_value(
                        code,
                        *instruction,
                        3,
                        template.shared_address,
                        template.record_split,
                    )?;
                    let skip_patch = code.words.len();
                    code.brz(3, 0)?;
                    if let Some(address) = sender_address {
                        emit_template_value(
                            code,
                            *address,
                            8,
                            template.shared_address,
                            template.record_split,
                        )?;
                        code.st32(3, 8, 15, 0)?;
                    } else {
                        emit_template_value(
                            code,
                            *plan_address,
                            8,
                            template.shared_address,
                            template.record_split,
                        )?;
                        code.st32(3, 8, 15, sender_word_offset.unwrap())?;
                    }
                    let after_patch = generated_address(generated_base, code.words.len())?;
                    code.words[skip_patch] = ipu_exchange::encode_brz_m_immediate(3, after_patch)?;
                }
                emit_template_value(
                    code,
                    *plan_address,
                    8,
                    template.shared_address,
                    template.record_split,
                )?;
                emit_template_value(
                    code,
                    *active,
                    0,
                    template.shared_address,
                    template.record_split,
                )?;
                code.call(template_exchange, 9)?;
            }
            StaticTemplateStep::Compute {
                operation,
                operands,
                kernel,
                condition,
            } => {
                if let Some(condition) = condition {
                    emit_template_value(
                        code,
                        *condition,
                        0,
                        template.shared_address,
                        template.record_split,
                    )?;
                }
                if let Some(kernel) = kernel {
                    emit_template_value(
                        code,
                        *kernel,
                        8,
                        template.shared_address,
                        template.record_split,
                    )?;
                }
                for (operand, &value) in operands.iter().enumerate() {
                    let register = u8::try_from(operand)? + 2;
                    emit_template_value(
                        code,
                        value,
                        register,
                        template.shared_address,
                        template.record_split,
                    )?;
                }
                let skip_call = if condition.is_some() {
                    let branch = code.words.len();
                    code.brz(0, 0)?;
                    Some(branch)
                } else {
                    None
                };
                if kernel.is_some() {
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
        emit_next_cycle_sample(
            code,
            symbols,
            profile_after_step.and_then(|samples| samples.get(step_index).copied()),
        )?;
    }
    code.ld32(9, 11, 15, 2)?;
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
) -> Result<()> {
    let repeat_call = symbol(symbols, REPEAT_CALL)?;
    let mut index = 0;
    while index < phases.len() {
        if phases[index].active {
            let start = index;
            while index < phases.len()
                && phases[index].active
                && phases[index].address == phases[start].address
            {
                index += 1;
            }
            code.setzi(2, u32::try_from(index - start)?)?;
            code.setzi(
                3,
                phases[start]
                    .run_table
                    .ok_or("active host run has no descriptor table")?,
            )?;
            code.setzi(4, phases[start].address)?;
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
    use std::sync::Arc;

    fn exchange(phase: usize, active: bool) -> LoweredTileStep {
        LoweredTileStep::Exchange {
            phase,
            epoch: 0,
            row: vec![if active {
                ipu_exchange::sans(1)
            } else {
                ipu_exchange::SANS_INACTIVE_INSTRUCTION
            }]
            .into(),
        }
    }

    fn compute(phase: usize, input_address: u32, arguments: Vec<u32>) -> LoweredTileStep {
        compute_with_operation(phase, input_address, arguments, "gemm_f16_accumulate")
    }

    fn compute_with_operation(
        phase: usize,
        input_address: u32,
        arguments: Vec<u32>,
        operation: &'static str,
    ) -> LoweredTileStep {
        LoweredTileStep::Compute(LoweredComputeCommand {
            op: OpId(phase),
            phase,
            command: Arc::new(ipu_compiler::KernelCommand {
                tile: 0,
                output: TensorId(3),
                inputs: vec![TensorId(1), TensorId(2)],
                arguments,
                specialization: SpecializationKey {
                    operation: operation.into(),
                    shape: vec![12, 64, 64],
                    worker_count: 6,
                    role: "inner-block".into(),
                    alignment: 32,
                },
                metadata: BTreeMap::new(),
            }),
            output_address: 0x80000,
            input_addresses: smallvec::smallvec![0x50000, input_address],
        })
    }

    #[test]
    fn template_records_intern_repeated_instance_columns() {
        let mut records = TemplateRecords::new(3);

        let first = records.values(vec![10, 20, 30]).unwrap();
        let repeated = records.values(vec![10, 20, 30]).unwrap();
        let second = records.values(vec![11, 21, 31]).unwrap();
        let wide_constant = records.values(vec![0x3f80_0000; 3]).unwrap();

        assert_eq!(first, TemplateValue::Record(0));
        assert_eq!(repeated, first);
        assert_eq!(second, TemplateValue::RecordOffset { slot: 0, offset: 1 });
        assert_eq!(wide_constant, TemplateValue::Shared(0));
        assert_eq!(
            records.rows,
            vec![
                vec![StaticTemplateRecordWord::Value(10)],
                vec![StaticTemplateRecordWord::Value(20)],
                vec![StaticTemplateRecordWord::Value(30)],
            ]
        );
        assert_eq!(
            records.shared,
            vec![StaticTemplateRecordWord::Value(0x3f80_0000)]
        );
    }

    #[test]
    fn template_affine_reuse_requires_net_sram_savings() {
        let mut short = TemplateRecords::new(2);
        short.values(vec![1, 2]).unwrap();
        assert_eq!(
            short.values(vec![0x20001, 0x20002]).unwrap(),
            TemplateValue::Record(1)
        );

        let mut long = TemplateRecords::new(27);
        long.values(1..=27).unwrap();
        assert_eq!(
            long.values((1..=27).map(|value| value + 0x20000)).unwrap(),
            TemplateValue::RecordOffset {
                slot: 0,
                offset: 0x20000,
            }
        );
    }

    #[test]
    fn template_compute_accepts_three_inputs_with_three_arguments() {
        let LoweredTileStep::Compute(mut first) = compute(3, 0x54000, vec![8, 64, 1024]) else {
            unreachable!()
        };
        Arc::make_mut(&mut first.command).inputs.push(TensorId(4));
        first.input_addresses.push(0x58000);
        let mut second = first.clone();
        second.output_address += 0x1000;
        let commands = vec![vec![&first], vec![&second]];
        let mut records = TemplateRecords::new(commands.len());

        let step = plan_template_compute_step(&commands, 0, &mut records, "test", 3).unwrap();

        let StaticTemplateStep::Compute { operands, .. } = step else {
            unreachable!()
        };
        assert_eq!(operands.len(), 7);
    }

    #[test]
    fn template_patch_span_omits_empty_segment_groups() {
        let patch = vec![
            (131, StaticTemplatePatchValue::Delta(4)),
            (
                134,
                StaticTemplatePatchValue::Word(StaticTemplateRecordWord::Value(9)),
            ),
        ];

        assert_eq!(template_patch_group_span(64..256, &patch), Some(64..96));
        assert_eq!(template_patch_storage_words_range(64..256, &patch), 5);
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
        let plan_rows = vec![vec![1, 2, 3], vec![1, 2, 4]];
        let (templates, end) = plan_static_templates(
            &program,
            &[0x52000, 0x52020],
            &plan_rows,
            &patches,
            &regions,
            0x53002,
            true,
        )
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
                sender_address: None,
                sender_instruction: Some(_),
                plan_words: ref words,
                plan_address: TemplateValue::Record(_),
                active: TemplateValue::Record(_),
            } if words.len() == 1
        ));
        assert!(matches!(
            template.steps[1],
            StaticTemplateStep::Compute {
                kernel: Some(_),
                condition: None,
                ..
            }
        ));
        assert!(matches!(
            template.steps[2],
            StaticTemplateStep::Compute {
                condition: Some(_),
                ..
            }
        ));
        assert_eq!(template.records[0].len(), template.records[1].len());
        assert!(end > template.record_secondary_addresses[0]);
        assert_eq!(template.patch_addresses.len(), 3);
        assert!(
            template
                .patch_addresses
                .iter()
                .all(|addresses| addresses.len() == 4)
        );
        assert!(template.patches[0].is_empty());
        for (patch, previous_record, expected_record) in [
            (
                &template.patches[1],
                &template.records[0],
                &template.records[1],
            ),
            (
                &template.patches[2],
                &template.records[1],
                &template.records[0],
            ),
        ] {
            for &(slot, ref value) in patch {
                let previous = &previous_record[usize::from(slot)];
                let expected = &expected_record[usize::from(slot)];
                match (value, previous, expected) {
                    (
                        StaticTemplatePatchValue::Delta(delta),
                        StaticTemplateRecordWord::Value(previous),
                        StaticTemplateRecordWord::Value(expected),
                    ) => assert_eq!(
                        i64::from(*previous) + i64::from(*delta),
                        i64::from(*expected)
                    ),
                    (StaticTemplatePatchValue::Word(value), _, expected) => {
                        assert_eq!(value, expected)
                    }
                    _ => panic!("invalid template patch encoding"),
                }
                assert_ne!(previous, expected);
            }
        }
        let original_exchanges = program
            .steps
            .iter()
            .filter(|step| matches!(step, LoweredTileStep::Exchange { .. }))
            .count();
        let mut compact_program = program.clone();
        let mut compact_templates = templates.clone();
        compact_template_instances(&mut compact_program, &mut compact_templates).unwrap();
        assert!(compact_program.steps.len() < program.steps.len());
        assert_eq!(compact_templates[0].exchange_step_count, original_exchanges);
        assert_eq!(compact_templates[0].records, templates[0].records);
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
