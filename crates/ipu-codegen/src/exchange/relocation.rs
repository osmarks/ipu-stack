//! Relocate moving send groups against the Repeat base retained in m6.
use super::*;
use ipu_exchange::{
    encode_put_special_m, patch_sender_instruction, sender_address_instruction_groups,
};

pub(super) fn relocate_repeat_rows(
    physical: &mut PhysicalExchangePhase,
    pending: &[PendingTransfer],
    addresses: &BTreeMap<BlockValueId, u32>,
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
        addresses,
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
            if physical.outgoing_bases[tile].is_none() {
                // No common representable base: retain ordinary word patching.
                let moving = encode_put_special_m(0xa7, 6)?;
                if program.contains(&moving) {
                    for instruction in
                        ipu_exchange::diagnostic::diagnose_plan_program(program, None)?.instructions
                    {
                        if matches!(
                            instruction.operation,
                            ipu_exchange::diagnostic::PlanOperation::WriteBase {
                                incoming: false,
                                register: 6
                            }
                        ) {
                            program[instruction.word_offset as usize] =
                                encode_put_special_m(0xa7, 15)?;
                        }
                    }
                }
            }
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
                let bases = bases.as_ref().filter(|_| transfer.moving_source());
                if !transfer.moving_source() {
                    continue;
                }
                let count = transfer
                    .source_addresses
                    .len()
                    .max(bases.map_or(1, Vec::len));
                for (word_offset, byte_offset) in instructions {
                    let values = (0..count)
                        .map(|i| {
                            let base = bases.map_or(0, |b| b.get(i).copied().unwrap_or(b[0]));
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
    use ipu_exchange::diagnostic::{PlanOperation, diagnose_plan_program};

    #[test]
    fn base_changes_preserve_every_repeat_address_and_receive_control() {
        let topology = Topology::c600();
        for words in [1, 8, 63, 64, 65, 257] {
            let mut addresses = BTreeMap::new();
            let mut bindings = BTreeMap::new();
            let pending = (0..5)
                .map(|index| {
                    let id = BlockValueId::from_index(index);
                    let address = 0x10000 + index * 0x1000;
                    let moving = index % 2 == 1;
                    let source_addresses = (0..3)
                        .map(|i| address + if moving { i * 0x10000 } else { 0 })
                        .collect::<Vec<_>>();
                    addresses.insert(id, address);
                    if moving {
                        bindings.insert(id, source_addresses.clone());
                    }
                    let mut transfer = PendingTransfer {
                        source: if index == 4 { 1 } else { 0 },
                        source_shard: id,
                        source_offset: 0,
                        source_addresses,
                        source_elements: vec![],
                        destinations: vec![(
                            if index == 4 { 0 } else { 2 },
                            0x80000 + index * 0x1000,
                        )],
                        words,
                        width: ExchangeItemWidth::Word32,
                        reserved_source: None,
                    };
                    transfer.refresh_source_elements();
                    transfer
                })
                .collect::<Vec<_>>();
            let (counts, bases) = receive_configuration(&pending, 4).unwrap();
            let schedule = materialize_valid_schedule_order(
                &topology,
                &SchedulingProblem::new(&pending, 4),
                &bases,
                &counts,
                &[0, 1, 2, 3, 4],
            )
            .unwrap();
            let mut physical = schedule
                .into_phase(ExchangePhaseId::from_index(0), bases)
                .unwrap();
            validate_exchange_schedule(4, &schedule_problem(0, &pending), &physical).unwrap();
            // Crossing source sequences have no common nonnegative base.
            // Keep their word patches and never read an uninitialized m6.
            let mut fallback = physical.clone();
            let mut crossing = pending.clone();
            crossing[1].source_addresses[1] += 0x30000;
            relocate_repeat_rows(&mut fallback, &crossing, &addresses, &bindings).unwrap();
            assert!(fallback.outgoing_bases.iter().all(Option::is_none));
            assert!(!fallback.repeat_patches[0].is_empty());
            assert!(
                !diagnose_plan_program(&fallback.programs[0], None)
                    .unwrap()
                    .instructions
                    .iter()
                    .any(|i| matches!(
                        i.operation,
                        PlanOperation::WriteBase {
                            incoming: false,
                            register: 6
                        }
                    ))
            );
            for iteration in 0..3 {
                let mut row = fallback.programs[0].clone();
                for patch in &fallback.repeat_patches[0] {
                    row[patch.word_offset as usize] = patch.values[iteration];
                }
                for (group, send) in sender_address_instruction_groups(&row)
                    .unwrap()
                    .into_iter()
                    .zip(
                        fallback.activities[0]
                            .iter()
                            .filter(|a| a.kind == ExchangeActivityKind::Send),
                    )
                {
                    for (word, offset) in group {
                        let mut expected = row[word];
                        patch_sender_instruction(
                            &mut expected,
                            crossing[send.transfer as usize].source_addresses[iteration] + offset,
                        )
                        .unwrap();
                        assert_eq!(row[word], expected);
                    }
                }
            }
            relocate_repeat_rows(&mut physical, &pending, &addresses, &bindings).unwrap();
            assert!(physical.repeat_patches.iter().all(Vec::is_empty));
            let row = &physical.programs[0];
            let decoded = diagnose_plan_program(row, None).unwrap();
            let switches = decoded
                .instructions
                .iter()
                .filter(|i| matches!(i.operation, PlanOperation::WriteBase { .. }))
                .collect::<Vec<_>>();
            assert_eq!(switches.len(), 4);
            assert!(
                switches.iter().all(
                    |i| i.end_cycle - i.start_cycle == ipu_exchange::EXCHANGE_BASE_WRITE_CYCLES
                )
            );
            let groups = sender_address_instruction_groups(row).unwrap();
            let sends = physical.activities[0]
                .iter()
                .filter(|a| a.kind == ExchangeActivityKind::Send);
            let (shard, offset) = physical.outgoing_bases[0].unwrap();
            for (group, send) in groups.into_iter().zip(sends) {
                let transfer = &pending[send.transfer as usize];
                for (word, byte_offset) in group {
                    let register = switches
                        .iter()
                        .rev()
                        .find_map(|i| {
                            if i.word_offset as usize >= word {
                                return None;
                            }
                            match i.operation {
                                PlanOperation::WriteBase { register, .. } => Some(register),
                                _ => None,
                            }
                        })
                        .unwrap_or(6);
                    assert_eq!(register == 6, transfer.moving_source());
                    for iteration in 0..3 {
                        let base = if register == 6 {
                            bindings[&shard][iteration] + offset
                        } else {
                            0
                        };
                        let mut expected = row[word];
                        patch_sender_instruction(
                            &mut expected,
                            transfer.source_addresses[iteration] - base + byte_offset,
                        )
                        .unwrap();
                        assert_eq!(row[word], expected);
                    }
                }
            }
        }
    }
}
