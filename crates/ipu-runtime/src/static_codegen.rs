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
    pub allocation: Option<ipu_compiler::TensorId>,
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
    pub patch_table_address: u32,
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum StaticTemplatePatchValue {
    Delta(i16),
    Delta32(u32),
    Difference {
        previous: StaticTemplateRecordWord,
        next: StaticTemplateRecordWord,
    },
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
            StaticTemplatePatchValue::Delta32(_) | StaticTemplatePatchValue::Difference { .. } => {
                (narrow, wide + 1)
            }
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

pub(crate) fn serialize_template_patch_range(
    slots: Range<usize>,
    patch: &[(u16, StaticTemplatePatchValue)],
    mut resolve: impl FnMut(&StaticTemplateRecordWord) -> Result<u32>,
) -> Result<Vec<u32>> {
    let Some(span) = template_patch_group_span(slots.clone(), patch) else {
        return Ok(vec![0]);
    };
    let group_count = span.len().div_ceil(32);
    let mut words = Vec::new();
    let mut narrow_bits = Vec::new();
    let mut narrow = Vec::new();
    let mut wide = Vec::new();
    words.push(u32::try_from(span.start)? | (u32::try_from(group_count)? << 16));
    for local_base in span.step_by(32) {
        let slot_base = slots.start + local_base;
        let slot_limit = (slot_base + 32).min(slots.end);
        let mut changed_mask = 0u32;
        for (slot, value) in patch
            .iter()
            .filter(|(slot, _)| (slot_base..slot_limit).contains(&usize::from(*slot)))
        {
            changed_mask |= 1 << (usize::from(*slot) - slot_base);
            match value {
                StaticTemplatePatchValue::Delta(delta) => {
                    narrow_bits.push(true);
                    narrow.push(*delta as u16);
                }
                StaticTemplatePatchValue::Delta32(delta) => {
                    narrow_bits.push(false);
                    wide.push(*delta);
                }
                StaticTemplatePatchValue::Difference { previous, next } => {
                    narrow_bits.push(false);
                    wide.push(resolve(next)?.wrapping_sub(resolve(previous)?));
                }
            }
        }
        words.push(changed_mask);
    }
    words.extend(narrow_bits.chunks(32).map(|bits| {
        bits.iter().enumerate().fold(0u32, |mask, (bit, narrow)| {
            mask | (u32::from(*narrow) << bit)
        })
    }));
    words.extend(
        narrow
            .chunks(2)
            .map(|pair| u32::from(pair[0]) | (u32::from(pair.get(1).copied().unwrap_or(0)) << 16)),
    );
    words.extend(wide);
    Ok(words)
}

/// Applies the packed transition format consumed by
/// `ipu_stack_static_template_patch` to one record segment.
pub(crate) fn apply_serialized_template_patch(record: &mut [u32], words: &[u32]) -> Result<usize> {
    let &header = words.first().ok_or("static template patch is empty")?;
    if header == 0 {
        return Ok(1);
    }
    let first = usize::try_from(header & 0xffff)?;
    let groups = usize::try_from(header >> 16)?;
    if groups == 0 || first.checked_add(groups * 32).is_none() {
        return Err("static template patch has an invalid span".into());
    }
    let changed_end = 1usize
        .checked_add(groups)
        .ok_or("static template patch mask size overflow")?;
    let changed_masks = words
        .get(1..changed_end)
        .ok_or("static template patch truncates changed masks")?;
    let changed = changed_masks
        .iter()
        .map(|mask| mask.count_ones() as usize)
        .sum::<usize>();
    let type_words = changed.div_ceil(32);
    let type_end = changed_end
        .checked_add(type_words)
        .ok_or("static template patch type size overflow")?;
    let types = words
        .get(changed_end..type_end)
        .ok_or("static template patch truncates type masks")?;
    let narrow = types
        .iter()
        .map(|mask| mask.count_ones() as usize)
        .sum::<usize>();
    let narrow_words = narrow.div_ceil(2);
    let narrow_end = type_end
        .checked_add(narrow_words)
        .ok_or("static template patch narrow size overflow")?;
    let narrow_values = words
        .get(type_end..narrow_end)
        .ok_or("static template patch truncates narrow values")?;
    let wide = changed - narrow;
    let end = narrow_end
        .checked_add(wide)
        .ok_or("static template patch wide size overflow")?;
    let wide_values = words
        .get(narrow_end..end)
        .ok_or("static template patch truncates wide values")?;

    let mut changed_index = 0usize;
    let mut narrow_index = 0usize;
    let mut wide_index = 0usize;
    for (group, &changed_mask) in changed_masks.iter().enumerate() {
        for bit in 0..32 {
            if changed_mask & (1 << bit) == 0 {
                continue;
            }
            let slot = first + group * 32 + bit;
            let value = record
                .get_mut(slot)
                .ok_or("static template patch writes beyond its record segment")?;
            let is_narrow = types[changed_index / 32] & (1 << (changed_index % 32)) != 0;
            if is_narrow {
                let packed = narrow_values[narrow_index / 2];
                let delta = if narrow_index % 2 == 0 {
                    packed as u16
                } else {
                    (packed >> 16) as u16
                } as i16;
                *value = value.wrapping_add_signed(i32::from(delta));
                narrow_index += 1;
            } else {
                *value = value.wrapping_add(wide_values[wide_index]);
                wide_index += 1;
            }
            changed_index += 1;
        }
    }
    debug_assert_eq!(changed_index, changed);
    debug_assert_eq!(narrow_index, narrow);
    debug_assert_eq!(wide_index, wide);
    Ok(end)
}

pub(crate) fn validate_template_transitions(
    template: &StaticTemplatePlan,
    mut resolve: impl FnMut(&StaticTemplateRecordWord) -> Result<u32>,
) -> Result<()> {
    let Some(first) = template.records.first() else {
        return Ok(());
    };
    let mut current = first.iter().map(&mut resolve).collect::<Result<Vec<_>>>()?;
    let split = usize::from(template.record_split);
    for instance in 1..template.records.len() {
        for slots in template_patch_ranges(current.len(), split) {
            let words = serialize_template_patch_range(
                slots.clone(),
                &template.patches[instance],
                &mut resolve,
            )?;
            let consumed = apply_serialized_template_patch(&mut current[slots], &words)?;
            if consumed != words.len() {
                return Err(format!(
                    "static template {} transition {instance} leaves {} trailing patch words",
                    template.name,
                    words.len() - consumed
                )
                .into());
            }
        }
        let expected = template.records[instance]
            .iter()
            .map(&mut resolve)
            .collect::<Result<Vec<_>>>()?;
        if let Some(slot) = current
            .iter()
            .zip(&expected)
            .position(|(actual, expected)| actual != expected)
        {
            return Err(format!(
                "static template {} transition {instance} reconstructs slot {slot} as 0x{:x}, expected 0x{:x}",
                template.name, current[slot], expected[slot]
            )
            .into());
        }
    }
    Ok(())
}

fn template_instance_word(
    template: &StaticTemplatePlan,
    instance: usize,
    value: TemplateValue,
) -> Result<StaticTemplateRecordWord> {
    match value {
        TemplateValue::Constant(value) => Ok(StaticTemplateRecordWord::Value(value)),
        TemplateValue::Record(slot) => template
            .records
            .get(instance)
            .and_then(|record| record.get(usize::from(slot)))
            .cloned()
            .ok_or_else(|| "static template value references a missing record slot".into()),
        TemplateValue::Shared(slot) => template
            .shared
            .get(usize::from(slot))
            .cloned()
            .ok_or_else(|| "static template value references a missing shared slot".into()),
    }
}

fn template_instance_value(
    template: &StaticTemplatePlan,
    instance: usize,
    value: TemplateValue,
) -> Result<u32> {
    let StaticTemplateRecordWord::Value(value) = template_instance_word(template, instance, value)?
    else {
        return Err("static template address resolves to a symbol".into());
    };
    Ok(value)
}

/// Validates the operands that the compact body will actually load. This is
/// intentionally separate from lowered-program validation: template command
/// alignment and record-column interning are part of code generation and can
/// otherwise change which values reach a kernel after the lowered body has
/// been discarded.
pub(crate) fn validate_template_kernel_operands(template: &StaticTemplatePlan) -> Result<()> {
    for (step_index, step) in template.steps.iter().enumerate() {
        let StaticTemplateStep::Compute {
            operation,
            abi,
            input_count,
            operands,
            kernel,
            condition,
        } = step
        else {
            continue;
        };
        for instance in 0..template.records.len() {
            if let Some(condition) = condition {
                if template_instance_value(template, instance, *condition)? == 0 {
                    continue;
                }
            }
            let operation = if let Some(kernel) = kernel {
                let StaticTemplateRecordWord::Symbol(symbol) =
                    template_instance_word(template, instance, *kernel)?
                else {
                    return Err(format!(
                        "static template {} instance {instance} step {step_index} has a nonsymbolic dynamic kernel",
                        template.name
                    )
                    .into());
                };
                symbol
                    .strip_prefix("ipu_stack_")
                    .ok_or("static template kernel symbol has no runtime prefix")?
                    .to_string()
            } else {
                operation.clone()
            };
            let specialization = ipu_compiler::SpecializationKey {
                operation: operation.clone().into(),
                shape: Vec::new(),
                worker_count: 0,
                role: "".into(),
                alignment: 1,
                abi: *abi,
            };
            let addresses = operands
                .iter()
                .take(input_count + 1)
                .map(|&value| template_instance_value(template, instance, value))
                .collect::<Result<Vec<_>>>()?;
            let address = |operand: ipu_compiler::KernelOperand| -> Result<u32> {
                match operand {
                    ipu_compiler::KernelOperand::Output => Ok(addresses[0]),
                    ipu_compiler::KernelOperand::Input(input) => addresses
                        .get(input + 1)
                        .copied()
                        .ok_or_else(|| "static template kernel constraint has no operand".into()),
                }
            };
            let span = |operand: ipu_compiler::KernelOperand| -> Result<(u32, u32)> {
                Ok((
                    address(operand)?,
                    specialization.operand_access_bytes(operand)?,
                ))
            };
            for constraint in specialization.memory_constraints() {
                match constraint {
                    ipu_compiler::KernelMemoryConstraint::InClass(operand, class) => {
                        let (value, bytes) = span(*operand)?;
                        let end = value
                            .checked_add(bytes)
                            .ok_or("static template kernel operand span overflows")?;
                        match class {
                            ipu_compiler::KernelMemoryClass::Ipu21Interleaved
                                if value < ipu_package::IPU21_INTERLEAVED_MEMORY_BASE
                                    || end > ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT =>
                            {
                                return Err(format!(
                                    "static template {} instance {instance} step {step_index} kernel {operation} resolves {operand:?} outside interleaved memory at 0x{value:x}..0x{end:x}",
                                    template.name
                                )
                                .into());
                            }
                            _ => {}
                        }
                    }
                    ipu_compiler::KernelMemoryConstraint::DistinctEffectiveElements(operands) => {
                        let mut elements =
                            BTreeMap::<u8, (ipu_compiler::KernelOperand, u32)>::new();
                        for &operand in *operands {
                            let (value, bytes) = span(operand)?;
                            let touched = ipu_package::ipu21_effective_memory_elements(value, bytes)
                                .ok_or_else(|| {
                                    format!(
                                        "static template {} instance {instance} step {step_index} kernel {operation} resolves {operand:?} outside tile SRAM at 0x{value:x} for {bytes} bytes",
                                        template.name
                                    )
                                })?;
                            for (element, _, _) in touched {
                                if let Some((other_operand, other_value)) =
                                    elements.insert(element, (operand, value))
                                {
                                    return Err(format!(
                                        "static template {} instance {instance} step {step_index} kernel {operation} maps {other_operand:?} at 0x{other_value:x} and {operand:?} at 0x{value:x} to memory element {element}",
                                        template.name
                                    )
                                    .into());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn template_patch_ranges(record_words: usize, split: usize) -> Vec<Range<usize>> {
    [0..split, split..record_words]
        .into_iter()
        .filter(|range| !range.is_empty())
        .collect()
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
        abi: ipu_compiler::KernelAbi,
        input_count: usize,
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
    Shared(u16),
}

struct TemplateRecords {
    rows: Vec<Vec<StaticTemplateRecordWord>>,
    columns: HashMap<u64, Vec<(Vec<StaticTemplateRecordWord>, u16)>>,
    shared: Vec<StaticTemplateRecordWord>,
    shared_values: HashMap<StaticTemplateRecordWord, u16>,
}

impl TemplateRecords {
    fn new(instances: usize) -> Self {
        Self {
            rows: vec![Vec::new(); instances],
            columns: HashMap::default(),
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
                let commands = align_template_commands(
                    phase_steps
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
                        .collect::<Vec<_>>(),
                );
                let command_count = commands.first().map_or(0, Vec::len);
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
        cursor = (cursor + 3) & !3;
        let patch_table_address = cursor;
        cursor = cursor
            .checked_add(
                u32::try_from(patches.len().saturating_sub(1))?
                    .checked_mul(u32::from(!patch_ranges.is_empty()) * 4)
                    .ok_or("static template patch table size overflow")?,
            )
            .ok_or("static template patch table address overflow")?;
        templates.push(StaticTemplatePlan {
            name: region.name.clone(),
            instance_steps,
            record_addresses: vec![primary_address; records.rows.len()],
            record_secondary_addresses: vec![secondary_address; records.rows.len()],
            record_split,
            records: records.rows,
            patch_table_address,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ComputeShape<'a> {
    phase_tile_command_index: usize,
    operation: &'a str,
    abi: ipu_compiler::KernelAbi,
    input_count: usize,
    argument_count: usize,
}

fn compute_shape(command: &ipu_compiler::LoweredComputeCommand) -> ComputeShape<'_> {
    ComputeShape {
        phase_tile_command_index: command.phase_tile_command_index,
        operation: &command.specialization.operation,
        abi: command.specialization.abi,
        input_count: command.input_addresses.len(),
        argument_count: command.arguments.len(),
    }
}

fn merge_compute_shapes<'a>(
    left: &[ComputeShape<'a>],
    right: &[ComputeShape<'a>],
) -> Vec<ComputeShape<'a>> {
    let columns = right.len() + 1;
    let mut common = vec![0usize; (left.len() + 1) * columns];
    for left_index in (0..left.len()).rev() {
        for right_index in (0..right.len()).rev() {
            common[left_index * columns + right_index] = if left[left_index] == right[right_index] {
                1 + common[(left_index + 1) * columns + right_index + 1]
            } else {
                common[(left_index + 1) * columns + right_index]
                    .max(common[left_index * columns + right_index + 1])
            };
        }
    }
    let mut merged = Vec::with_capacity(left.len() + right.len());
    let (mut left_index, mut right_index) = (0, 0);
    while left_index < left.len() && right_index < right.len() {
        if left[left_index] == right[right_index] {
            merged.push(left[left_index]);
            left_index += 1;
            right_index += 1;
        } else if common[(left_index + 1) * columns + right_index]
            >= common[left_index * columns + right_index + 1]
        {
            merged.push(left[left_index]);
            left_index += 1;
        } else {
            merged.push(right[right_index]);
            right_index += 1;
        }
    }
    merged.extend_from_slice(&left[left_index..]);
    merged.extend_from_slice(&right[right_index..]);
    merged
}

fn align_template_commands<'a>(
    commands: Vec<Vec<&'a ipu_compiler::LoweredComputeCommand>>,
) -> Vec<Vec<Option<&'a ipu_compiler::LoweredComputeCommand>>> {
    let merged = commands.iter().fold(Vec::new(), |merged, commands| {
        let shapes = commands
            .iter()
            .map(|command| compute_shape(command))
            .collect::<Vec<_>>();
        merge_compute_shapes(&merged, &shapes)
    });
    commands
        .into_iter()
        .map(|commands| {
            let mut commands = commands.into_iter().peekable();
            let aligned = merged
                .iter()
                .map(|&shape| {
                    commands
                        .peek()
                        .is_some_and(|command| compute_shape(command) == shape)
                        .then(|| commands.next().unwrap())
                })
                .collect::<Vec<_>>();
            debug_assert!(commands.next().is_none());
            aligned
        })
        .collect()
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
                (
                    StaticTemplateRecordWord::Value(previous),
                    StaticTemplateRecordWord::Value(next),
                ) => StaticTemplatePatchValue::Delta32(next.wrapping_sub(*previous)),
                _ => StaticTemplatePatchValue::Difference {
                    previous: previous.clone(),
                    next: next.clone(),
                },
            };
            Ok((u16::try_from(slot)?, value))
        })
        .collect()
}

fn plan_template_compute_step(
    commands: &[Vec<Option<&ipu_compiler::LoweredComputeCommand>>],
    command_index: usize,
    records: &mut TemplateRecords,
    template_name: &str,
    relative_phase: usize,
) -> Result<StaticTemplateStep> {
    let active = commands
        .iter()
        .filter_map(|commands| commands[command_index])
        .collect::<Vec<_>>();
    let first = active[0];
    if active.iter().any(|command| {
        command.input_addresses.len() != first.input_addresses.len()
            || command.arguments.len() != first.arguments.len()
            || command.specialization.abi != first.specialization.abi
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
                    .map(|commands| u32::from(commands[command_index].is_some())),
            )
        })
        .transpose()?;
    let kernel = dynamic_kernel
        .then(|| {
            records.words(
                commands
                    .iter()
                    .map(|commands| match commands[command_index] {
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
                commands[command_index]
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
        abi: first.specialization.abi,
        input_count: first.input_addresses.len(),
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
            if !after_sync
                .iter()
                .chain(after_step)
                .any(|boundary| *boundary)
            {
                continue;
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
            code.add_immediate(11, 11, -32)?;
            code.setzi(4, template.record_addresses[0])?;
            code.setzi(5, template.record_secondary_addresses[0])?;
            code.setzi(6, u32::from(template.record_split))?;
            code.setzi(7, u32::try_from(record_words - record_split)?)?;
            code.st32(4, 11, 15, 0)?;
            code.st32(5, 11, 15, 1)?;
            code.st32(6, 11, 15, 2)?;
            code.st32(7, 11, 15, 3)?;
            code.setzi(2, u32::try_from(template.records.len())?)?;
            code.st32(2, 11, 15, 4)?;
            code.setzi(2, template.patch_table_address)?;
            code.st32(2, 11, 15, 5)?;
            let loop_start = generated_address(generated_base, code.words.len())?;
            let call = code.words.len();
            code.call(0, 9)?;
            template_calls.push((call, template_index));
            code.ld32(2, 11, 15, 4)?;
            code.add_immediate(2, 2, -1)?;
            code.st32(2, 11, 15, 4)?;
            let loop_done = code.words.len();
            code.brz(2, 0)?;
            emit_template_patches(
                &mut code,
                symbols,
                template_patch_ranges(record_words, record_split),
                record_split,
            )?;
            code.jump(loop_start)?;
            let after_loop = generated_address(generated_base, code.words.len())?;
            code.words[loop_done] = ipu_exchange::encode_brz_m_immediate(2, after_loop)?;
            if template.patches.len() > template.records.len() {
                emit_template_patches(
                    &mut code,
                    symbols,
                    template_patch_ranges(record_words, record_split),
                    record_split,
                )?;
            }
            code.add_immediate(11, 11, 32)?;
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
        let profile_after_sync = profile.map(|profile| {
            let first = &template.instance_steps[0];
            &profile.after_sync[first.clone()]
        });
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
            profile_after_sync,
            profile_after_step,
        )?;
    }
    for (call, template) in template_calls {
        let target = generated_address(generated_base, template_bodies[template])?;
        code.words[call] = ipu_exchange::encode_call_m_immediate(9, target)?;
    }
    Ok(code.words.into_iter().flat_map(u32::to_le_bytes).collect())
}

fn emit_template_patches(
    code: &mut TileCode,
    symbols: &BTreeMap<String, u32>,
    ranges: Vec<Range<usize>>,
    record_split: usize,
) -> Result<()> {
    if ranges.is_empty() {
        return Ok(());
    }
    code.ld32(2, 11, 15, 5)?;
    code.ld32(3, 2, 15, 0)?;
    code.add_immediate(2, 2, 4)?;
    code.st32(2, 11, 15, 5)?;
    for range in ranges {
        code.setzi(7, u32::from(range.start >= record_split))?;
        code.call(symbol(symbols, TEMPLATE_PATCH)?, 9)?;
    }
    Ok(())
}

fn emit_static_template_exchange(
    code: &mut TileCode,
    worker_barrier: u32,
    generated_base: u32,
) -> Result<()> {
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
    profile_after_sync: Option<&[bool]>,
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
                code.instruction(ipu_exchange::SYNC_SUPERVISOR_INSTRUCTION);
                emit_next_cycle_sample(
                    code,
                    symbols,
                    profile_after_sync.and_then(|samples| samples.get(step_index).copied()),
                )?;
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
                abi: _,
                input_count: _,
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
            phase_tile_command_index: 0,
            command: Arc::new(ipu_compiler::KernelCommand {
                tile: 0,
                output: TensorId(3),
                inputs: vec![TensorId(1), TensorId(2)],
                arguments,
                specialization: Arc::new(SpecializationKey {
                    operation: operation.into(),
                    shape: vec![12, 64, 64],
                    worker_count: 6,
                    role: "inner-block".into(),
                    alignment: 32,
                    abi: ipu_compiler::KernelAbi::pace(
                        12 * 64 * 2,
                        12 * 64 * 2,
                        64 * 64 * 2,
                        false,
                    ),
                }),
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
        assert_eq!(second, TemplateValue::Record(1));
        assert_eq!(wide_constant, TemplateValue::Shared(0));
        assert_eq!(
            records.rows,
            vec![
                vec![
                    StaticTemplateRecordWord::Value(10),
                    StaticTemplateRecordWord::Value(11),
                ],
                vec![
                    StaticTemplateRecordWord::Value(20),
                    StaticTemplateRecordWord::Value(21),
                ],
                vec![
                    StaticTemplateRecordWord::Value(30),
                    StaticTemplateRecordWord::Value(31),
                ],
            ]
        );
        assert_eq!(
            records.shared,
            vec![StaticTemplateRecordWord::Value(0x3f80_0000)]
        );
    }

    #[test]
    fn template_offset_columns_remain_independent() {
        let mut records = TemplateRecords::new(27);
        assert_eq!(records.values(1..=27).unwrap(), TemplateValue::Record(0));
        assert_eq!(
            records
                .values((1..=27).map(|value| value + 0x20000))
                .unwrap(),
            TemplateValue::Record(1)
        );
        assert!(records.rows.iter().all(|row| row.len() == 2));
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
        let commands = align_template_commands(vec![vec![&first], vec![&second]]);
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
            (134, StaticTemplatePatchValue::Delta32(9)),
        ];

        assert_eq!(template_patch_group_span(64..256, &patch), Some(64..96));
        assert_eq!(template_patch_storage_words_range(64..256, &patch), 5);
    }

    #[test]
    fn serialized_template_patches_reconstruct_random_records() {
        let mut rng = fastrand::Rng::with_seed(0x7465_6d70_6c61_7465);
        for _ in 0..500 {
            let len = rng.usize(1..600);
            let previous = (0..len).map(|_| rng.u32(..)).collect::<Vec<_>>();
            let mut expected = previous.clone();
            for value in &mut expected {
                if rng.usize(..5) == 0 {
                    *value = match rng.usize(..3) {
                        0 => value.wrapping_add_signed(rng.i32(-32768..32768)),
                        1 => value.wrapping_add(rng.u32(..)),
                        _ => rng.u32(..),
                    };
                }
            }
            let previous_words = previous
                .iter()
                .copied()
                .map(StaticTemplateRecordWord::Value)
                .collect::<Vec<_>>();
            let expected_words = expected
                .iter()
                .copied()
                .map(StaticTemplateRecordWord::Value)
                .collect::<Vec<_>>();
            let patch = template_record_patch(&previous_words, &expected_words).unwrap();
            let split = len.div_ceil(2);
            let mut actual = previous;
            for slots in template_patch_ranges(len, split) {
                let words = serialize_template_patch_range(slots.clone(), &patch, |word| {
                    let StaticTemplateRecordWord::Value(value) = word else {
                        unreachable!()
                    };
                    Ok(*value)
                })
                .unwrap();
                let consumed = apply_serialized_template_patch(&mut actual[slots], &words).unwrap();
                assert_eq!(consumed, words.len());
            }
            assert_eq!(actual, expected);
        }
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
                kernel: None,
                condition: Some(_),
                ..
            }
        ));
        assert!(matches!(
            template.steps[2],
            StaticTemplateStep::Compute {
                kernel: None,
                condition: None,
                ..
            }
        ));
        assert_eq!(template.records[0].len(), template.records[1].len());
        assert!(end > template.record_secondary_addresses[0]);
        assert_eq!(template.patch_addresses.len(), 3);
        assert!(template.patch_addresses.iter().all(|addresses| {
            addresses.len()
                == template_patch_ranges(
                    template.records[0].len(),
                    usize::from(template.record_split),
                )
                .len()
        }));
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
                    (
                        StaticTemplatePatchValue::Delta32(delta),
                        StaticTemplateRecordWord::Value(previous),
                        StaticTemplateRecordWord::Value(expected),
                    ) => assert_eq!(previous.wrapping_add(*delta), *expected),
                    (
                        StaticTemplatePatchValue::Difference {
                            previous: encoded_previous,
                            next,
                        },
                        previous,
                        expected,
                    ) => {
                        assert_eq!(encoded_previous, previous);
                        assert_eq!(next, expected);
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

    #[test]
    fn repeated_templates_profile_sync_and_exchange_boundaries() {
        let program = LoweredTileProgram {
            tile: 7,
            steps: vec![
                exchange(0, true),
                compute(1, 0x54000, Vec::new()),
                exchange(2, true),
                compute(3, 0x54020, Vec::new()),
            ],
        };
        let regions = [crate::StaticTemplateRegion {
            name: "encoder_layer".into(),
            phase_instances: vec![0..2, 2..4],
        }];
        let plans = [0x52000, 0x52020];
        let rows = [vec![1, 2, 3], vec![1, 2, 4]];
        let (templates, _) = plan_static_templates(
            &program,
            &plans,
            &rows,
            &[None, None],
            &regions,
            0x60000,
            false,
        )
        .unwrap();
        let symbols = BTreeMap::from([
            (WORKER_BARRIER.into(), 0x50000),
            (COMPLETE.into(), 0x50020),
            (HOST_RUN.into(), 0x50040),
            (REPEAT_CALL.into(), 0x50060),
            (TEMPLATE_PATCH.into(), 0x50080),
            (SAMPLE_CYCLE.into(), 0x500a0),
            (SAMPLE_CYCLE_NEXT.into(), 0x500c0),
            ("ipu_stack_gemm_f16_accumulate".into(), 0x500e0),
        ]);
        let host = || HostCode {
            weights: &[],
            inputs: &[],
            outputs: &[],
        };
        let unprofiled = emit(
            &program,
            &symbols,
            &plans,
            &[],
            &templates,
            host(),
            None,
            0x70000,
            1,
        )
        .unwrap();
        let profiled = emit(
            &program,
            &symbols,
            &plans,
            &[],
            &templates,
            host(),
            Some(&ProfileCode {
                allocation: None,
                initial: 0x68000,
                after_sync: vec![true, false, true, false],
                after_step: vec![true; 4],
                aggregate_end: None,
            }),
            0x70000,
            1,
        )
        .unwrap();

        assert!(profiled.len() > unprofiled.len());
    }

    #[test]
    fn aggregate_profile_allows_different_template_instance_step_counts() {
        let program = LoweredTileProgram {
            tile: 7,
            steps: vec![
                compute(0, 0x54000, Vec::new()),
                compute(1, 0x54020, Vec::new()),
                compute(1, 0x54040, Vec::new()),
            ],
        };
        let regions = [crate::StaticTemplateRegion {
            name: "encoder_layer".into(),
            phase_instances: vec![0..1, 1..2],
        }];
        let (templates, _) =
            plan_static_templates(&program, &[], &[], &[], &regions, 0x60000, false).unwrap();
        let symbols = BTreeMap::from([
            (WORKER_BARRIER.into(), 0x50000),
            (COMPLETE.into(), 0x50020),
            (HOST_RUN.into(), 0x50040),
            (REPEAT_CALL.into(), 0x50060),
            (TEMPLATE_PATCH.into(), 0x50080),
            (SAMPLE_CYCLE.into(), 0x500a0),
            (SAMPLE_CYCLE_NEXT.into(), 0x500c0),
            ("ipu_stack_gemm_f16_accumulate".into(), 0x500e0),
        ]);

        emit(
            &program,
            &symbols,
            &[],
            &[],
            &templates,
            HostCode {
                weights: &[],
                inputs: &[],
                outputs: &[],
            },
            Some(&ProfileCode {
                allocation: None,
                initial: 0x68000,
                after_sync: vec![false; 3],
                after_step: vec![false; 3],
                aggregate_end: Some(0x68004),
            }),
            0x70000,
            1,
        )
        .unwrap();
    }

    #[test]
    fn repeated_template_executable_size_is_independent_of_instance_count() {
        fn emitted_size(instance_count: usize) -> usize {
            let mut steps = Vec::new();
            let mut phase_instances = Vec::new();
            let mut plan_addresses = Vec::new();
            let mut plan_rows = Vec::new();
            for instance in 0..instance_count {
                let phase = instance * 2;
                steps.push(exchange(phase, true));
                steps.push(compute(
                    phase + 1,
                    0x54000 + instance as u32 * 0x20,
                    Vec::new(),
                ));
                phase_instances.push(phase..phase + 2);
                plan_addresses.push(0x52000 + instance as u32 * 0x20);
                plan_rows.push(vec![1, 2, 3 + instance as u32]);
            }
            let mut program = LoweredTileProgram { tile: 7, steps };
            let regions = [crate::StaticTemplateRegion {
                name: "encoder_layer".into(),
                phase_instances,
            }];
            let (mut templates, _) = plan_static_templates(
                &program,
                &plan_addresses,
                &plan_rows,
                &vec![None; instance_count],
                &regions,
                0x60000,
                true,
            )
            .unwrap();
            compact_template_instances(&mut program, &mut templates).unwrap();
            let symbols = BTreeMap::from([
                (WORKER_BARRIER.into(), 0x50000),
                (COMPLETE.into(), 0x50020),
                (HOST_RUN.into(), 0x50040),
                (REPEAT_CALL.into(), 0x50060),
                (TEMPLATE_PATCH.into(), 0x50080),
                ("ipu_stack_gemm_f16_accumulate".into(), 0x500a0),
            ]);
            emit(
                &program,
                &symbols,
                &plan_addresses,
                &[],
                &templates,
                HostCode {
                    weights: &[],
                    inputs: &[],
                    outputs: &[],
                },
                None,
                0x70000,
                2,
            )
            .unwrap()
            .len()
        }

        assert_eq!(emitted_size(2), emitted_size(27));
    }
}
