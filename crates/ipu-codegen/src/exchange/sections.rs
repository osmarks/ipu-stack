//! Timed fixed/moving source sections within one synchronized exchange.
use super::*;
use ipu_exchange::{
    EXCHANGE_BASE_WRITE_CYCLES, encode_delay_m, encode_put_special_m, encode_setzi_m,
    patch_sender_instruction, sender_address_instruction_groups,
};
use reuse::ScheduleSection;

const PREFIX_CYCLES: u32 = 2 + EXCHANGE_BASE_WRITE_CYCLES + 1;
const SWITCH_CYCLES: u32 = 1 + 2 * EXCHANGE_BASE_WRITE_CYCLES + 1;

fn moving(transfer: &PendingTransfer) -> bool {
    transfer
        .source_addresses
        .iter()
        .any(|&a| a != transfer.source_address())
}

fn fixed_before_moving_is_valid(pending: &[PendingTransfer], tile_count: u16) -> bool {
    !memory_dependencies(pending, tile_count)
        .iter()
        .any(|&(before, after)| moving(&pending[before]) && !moving(&pending[after]))
}

/// Only mixed senders can benefit from switching between zero and a moving base.
pub(super) fn has_mixed_sources(pending: &[PendingTransfer], tile_count: u16) -> bool {
    let mut roles = vec![0u8; usize::from(tile_count)];
    for transfer in pending {
        roles[usize::from(transfer.source)] |= if moving(transfer) { 2 } else { 1 };
    }
    roles.contains(&3)
}

pub(super) fn try_separate(
    combined: &PhysicalExchangePhase,
    pending: Vec<PendingTransfer>,
    placement: &Placement,
    repeat_inputs: &BTreeMap<BlockValueId, Vec<u32>>,
    topology: &Topology,
    cache: &mut ExchangeScheduleCache,
) -> Result<Option<(PhysicalExchangePhase, ExchangeScheduleProblem)>, ExchangeLoweringError> {
    if combined.repeat_patches.iter().all(Vec::is_empty) {
        return Ok(None);
    }
    let tile_count = combined.programs.len() as u16;
    // Fixed -> moving is legal only if it does not reverse a forwarding,
    // overwrite, or read-before-write dependency, including later iterations.
    if !fixed_before_moving_is_valid(&pending, tile_count) {
        return Ok(None);
    }
    let (moving, fixed): (Vec<_>, Vec<_>) = pending.into_iter().partition(moving);
    let mut fixed = cache.select_section(
        combined.id,
        ScheduleSection::Fixed,
        topology,
        fixed,
        tile_count,
    )?;
    let moving = cache.select_section(
        combined.id,
        ScheduleSection::Moving,
        topology,
        moving,
        tile_count,
    )?;
    // Require a win even before counting patch savings. No speculative patch
    // timing model or extra search/retry path is needed to accept these schedules.
    if u64::from(fixed.optimized.schedule.horizon)
        + u64::from(moving.optimized.schedule.horizon)
        + u64::from(PREFIX_CYCLES + 2 + SWITCH_CYCLES)
        >= u64::from(combined.event_cycles)
    {
        return Ok(None);
    }
    let first = fixed
        .optimized
        .schedule
        .into_phase(combined.id, fixed.incoming_bases)?;
    let mut second = moving
        .optimized
        .schedule
        .into_phase(combined.id, moving.incoming_bases)?;
    relocate_repeat_rows(&mut second, &moving.pending, placement, repeat_inputs)?;
    let joined = join(first, second, fixed.pending.len() as u32)?;
    let patch_words =
        |phase: &PhysicalExchangePhase| phase.repeat_patches.iter().map(Vec::len).sum::<usize>();
    if patch_words(&joined) >= patch_words(combined) {
        return Ok(None);
    }
    tracing::info!(
        phase = combined.id.index(),
        combined_cycles = combined.event_cycles,
        section_cycles = joined.event_cycles,
        combined_patch_words = patch_words(combined),
        section_patch_words = patch_words(&joined),
        combined_max_row_bytes = combined
            .programs
            .iter()
            .map(|r| r.len() * 4)
            .max()
            .unwrap_or(0),
        section_max_row_bytes = joined
            .programs
            .iter()
            .map(|r| r.len() * 4)
            .max()
            .unwrap_or(0),
        "selected fixed/moving exchange sections under one sync"
    );
    fixed.pending.extend(moving.pending);
    Ok(Some((
        joined,
        schedule_problem(combined.id.index(), &fixed.pending),
    )))
}

/// The prologue loads the moving base into m6. Start with absolute sources,
/// drain the first section, then switch both address bases for the second.
/// Every participating tile executes the same scalar instructions at the
/// boundary; there is no barrier or additional worker/supervisor launch.
fn join(
    mut first: PhysicalExchangePhase,
    second: PhysicalExchangePhase,
    transfer_offset: u32,
) -> Result<PhysicalExchangePhase, ExchangeLoweringError> {
    let second_start = PREFIX_CYCLES
        .checked_add(first.event_cycles)
        .and_then(|v| v.checked_add(2 + SWITCH_CYCLES))
        .ok_or(ExchangeLoweringError::Overflow)?;
    for tile in 0..first.programs.len() {
        if !first.active[tile] && !second.active[tile] {
            continue;
        }
        let mut row = vec![
            encode_delay_m(1)?,
            encode_delay_m(1)?,
            encode_put_special_m(0xa7, 15)?,
            encode_delay_m(1)?,
        ];
        let mut body = std::mem::take(&mut first.programs[tile]);
        if body.pop() != Some(RETURN_M10_INSTRUCTION) {
            return Err(ExchangeLoweringError::Invariant(
                "exchange section has no return".into(),
            ));
        }
        row.extend(body);
        // Both horizons include receive teardown and SRAM hazards. Two extra
        // cycles allow either parity to reach an eight-byte-aligned boundary.
        let padding = first.event_cycles + 2 - first.tile_event_cycles[tile];
        if row.len() % 2 == 0 {
            row.push(encode_delay_m(1)?);
            row.push(encode_delay_m(padding - 1)?);
        } else {
            row.push(encode_delay_m(padding)?);
        }
        row.extend([
            encode_setzi_m(8, second.incoming_bases[tile])?,
            encode_put_special_m(0xa4, 8)?,
            encode_put_special_m(
                0xa7,
                if second.outgoing_bases[tile].is_some() {
                    6
                } else {
                    15
                },
            )?,
            encode_delay_m(1)?,
        ]);
        let word_offset = u32::try_from(row.len()).map_err(|_| ExchangeLoweringError::Overflow)?;
        row.extend_from_slice(&second.programs[tile]);
        first.programs[tile] = row;
        first.active[tile] = true;
        first.tile_event_cycles[tile] = second_start
            .checked_add(second.tile_event_cycles[tile])
            .ok_or(ExchangeLoweringError::Overflow)?;
        for activity in &mut first.activities[tile] {
            activity.start_cycle += PREFIX_CYCLES;
            activity.end_cycle += PREFIX_CYCLES;
            activity.memory_end_cycle += PREFIX_CYCLES;
        }
        first.activities[tile].extend(second.activities[tile].iter().map(|activity| {
            ExchangeActivity {
                transfer: activity.transfer + transfer_offset,
                start_cycle: activity.start_cycle + second_start,
                end_cycle: activity.end_cycle + second_start,
                memory_end_cycle: activity.memory_end_cycle + second_start,
                ..*activity
            }
        }));
        first.repeat_patches[tile] = second.repeat_patches[tile]
            .iter()
            .map(|patch| ExchangeRowPatch {
                word_offset: patch.word_offset + word_offset,
                values: patch.values.clone(),
            })
            .collect();
    }
    first.event_cycles = second_start
        .checked_add(second.event_cycles)
        .ok_or(ExchangeLoweringError::Overflow)?;
    first.outgoing_bases = second.outgoing_bases;
    Ok(first)
}

pub(super) fn relocate_repeat_rows(
    physical: &mut PhysicalExchangePhase,
    pending: &[PendingTransfer],
    placement: &Placement,
    repeat_inputs: &BTreeMap<BlockValueId, Vec<u32>>,
) -> Result<(), ExchangeLoweringError> {
    let address_groups = physical
        .programs
        .iter()
        .map(|program| sender_address_instruction_groups(program))
        .collect::<Result<Vec<_>, _>>()?;
    let mut patch_words = vec![0; pending.len()];
    for (groups, activity) in address_groups.iter().zip(&physical.activities) {
        let sends = activity
            .iter()
            .filter(|a| a.kind == ExchangeActivityKind::Send);
        if groups.len() != sends.clone().count() {
            return Err(ExchangeLoweringError::IncompatibleRepeatRows(
                "send instruction groups do not match scheduled messages",
            ));
        }
        for (group, send) in groups.iter().zip(sends) {
            patch_words[send.transfer as usize] = group.len();
        }
    }
    physical.outgoing_bases = repeat_outgoing_bases(
        pending,
        &patch_words,
        &placement.shard_addresses,
        physical.programs.len() as u16,
    );
    physical.repeat_patches = physical
        .programs
        .iter_mut()
        .enumerate()
        .zip(address_groups)
        .map(|((tile, program), address_groups)| {
            let sends = physical.activities[tile]
                .iter()
                .filter(|activity| activity.kind == ExchangeActivityKind::Send)
                .map(|activity| &pending[activity.transfer as usize])
                .collect::<Vec<_>>();
            let mut patches = Vec::new();
            let bases = physical.outgoing_bases[tile]
                .map(|(shard, offset)| {
                    repeat_inputs[&shard]
                        .iter()
                        .map(|address| {
                            address
                                .checked_add(offset)
                                .ok_or(ExchangeLoweringError::Overflow)
                        })
                        .collect::<Result<Vec<_>, ExchangeLoweringError>>()
                })
                .transpose()?;
            for (instructions, transfer) in address_groups.into_iter().zip(sends) {
                if bases.is_none()
                    && transfer
                        .source_addresses
                        .iter()
                        .all(|&a| a == transfer.source_address())
                {
                    continue;
                }
                let count = transfer
                    .source_addresses
                    .len()
                    .max(bases.as_ref().map_or(1, Vec::len));
                for (word_offset, byte_offset) in instructions {
                    let values = (0..count)
                        .map(|i| {
                            let base = bases
                                .as_ref()
                                .map_or(0, |b| b.get(i).copied().unwrap_or(b[0]));
                            let address = repeat_source_address(transfer, i)
                                .checked_sub(base)
                                .and_then(|a| a.checked_add(byte_offset))
                                .ok_or(ExchangeLoweringError::Overflow)?;
                            let mut instruction = program[word_offset];
                            patch_sender_instruction(&mut instruction, address)?;
                            Ok(instruction)
                        })
                        .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
                    if bases.is_none() && values[0] != program[word_offset] {
                        return Err(ExchangeLoweringError::IncompatibleRepeatRows(
                            "relocation changes the first iteration",
                        ));
                    }
                    program[word_offset] = values[0];
                    if values.iter().any(|&v| v != values[0]) {
                        patches.push(ExchangeRowPatch {
                            word_offset: u32::try_from(word_offset)
                                .map_err(|_| ExchangeLoweringError::Overflow)?,
                            values,
                        });
                    }
                }
            }
            Ok(patches)
        })
        .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transfer(
        source: u16,
        address: u32,
        stride: u32,
        destination: u16,
        target: u32,
        words: u32,
    ) -> PendingTransfer {
        PendingTransfer {
            source,
            source_shard: BlockValueId::from_index(u32::from(source)),
            source_offset: 0,
            source_addresses: vec![address, address + stride],
            source_elements: effective_memory_elements(address, words),
            destinations: vec![(destination, target)],
            words,
            width: ExchangeItemWidth::Word32,
            reserved_source: None,
        }
    }

    #[test]
    fn sections_preserve_forwarding_and_later_iteration_dependencies() {
        let producer = transfer(0, 0x10000, 0x1000, 1, 0x80000, 16);
        let forward = transfer(1, 0x80000, 0, 2, 0x84000, 16);
        assert!(!fixed_before_moving_is_valid(&[producer, forward], 4));
        // An absolute write aliases only the later moving source, not iteration zero.
        let reader = transfer(0, 0x10000, 0x1000, 1, 0x80000, 16);
        let overwrite = transfer(2, 0x20000, 0, 0, 0x11000, 16);
        assert!(!fixed_before_moving_is_valid(&[reader, overwrite], 4));
        let fixed = transfer(0, 0x10000, 0, 1, 0x80000, 16);
        let moving = transfer(0, 0x20000, 0x1000, 2, 0x84000, 16);
        assert!(has_mixed_sources(&[fixed.clone(), moving.clone()], 4));
        assert!(fixed_before_moving_is_valid(&[fixed, moving], 4));
    }

    #[test]
    fn joined_rows_match_decoded_timing_and_transfer_hazards() {
        let topology =
            Topology::new((0..8).map(ipu_exchange::c600_logical_to_physical).collect()).unwrap();
        for words in [1, 8, 63, 64, 65, 257] {
            let first = vec![
                transfer(0, 0x10000, 0, 2, 0x80000, words),
                transfer(1, 0x12000, 0, 3, 0x84000, words + 1),
            ];
            let second = vec![
                transfer(0, 0x14000, 0, 2, 0x88000, words + 2),
                transfer(4, 0x16000, 0, 5, 0x8c000, words + 3),
            ];
            let lower = |pending: &[PendingTransfer]| {
                let (counts, bases) = receive_configuration(pending, 8).unwrap();
                optimize_pending_schedule(&topology, pending, &bases, &counts, 8, None)
                    .unwrap()
                    .schedule
                    .into_phase(ExchangePhaseId::from_index(0), bases)
                    .unwrap()
            };
            let mut second_phase = lower(&second);
            // Exercise relocation-table offsets as well as raw instruction timing.
            let address_word =
                sender_address_instruction_groups(&second_phase.programs[0]).unwrap()[0][0].0;
            let values = vec![second_phase.programs[0][address_word]; 2];
            second_phase.repeat_patches[0].push(ExchangeRowPatch {
                word_offset: address_word as u32,
                values: values.clone(),
            });
            let mut joined = join(lower(&first), second_phase, first.len() as u32).unwrap();
            let patch = &joined.repeat_patches[0][0];
            assert_eq!(joined.programs[0][patch.word_offset as usize], values[0]);
            assert_eq!(patch.values, values);
            joined.repeat_patches[0].clear();
            let mut pending = first;
            pending.extend(second);
            validate_exchange_schedule(8, &schedule_problem(0, &pending), &joined).unwrap();
            for row in &joined.programs {
                let decoded = ipu_exchange::diagnostic::diagnose_plan_program(row, None).unwrap();
                assert!(!decoded.instructions.iter().any(|i| matches!(
                    i.operation,
                    ipu_exchange::diagnostic::PlanOperation::Sync(_)
                        | ipu_exchange::diagnostic::PlanOperation::Unknown(_)
                )));
            }
        }
    }
}
