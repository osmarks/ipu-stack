//! Relocate moving send sites against the Repeat base retained in m6.
use super::{
    ExchangeLoweringError, ExchangeRowPatch, PendingTransfer, PhysicalExchangePhase,
    repeat_outgoing_bases, repeat_source_address,
};
use crate::low::BlockValueId;
use std::collections::BTreeMap;

pub(super) fn relocate_repeat_rows(
    physical: &mut PhysicalExchangePhase,
    pending: &[PendingTransfer],
    addresses: &BTreeMap<BlockValueId, u32>,
    repeat_inputs: &BTreeMap<BlockValueId, Vec<u32>>,
) -> Result<(), ExchangeLoweringError> {
    let mut patch_words = vec![0; pending.len()];
    for row in &physical.programs {
        for site in row.send_addresses() {
            patch_words[site.message as usize] += 1;
        }
    }
    physical.outgoing_bases = repeat_outgoing_bases(
        pending,
        &patch_words,
        addresses,
        physical.programs.len() as u16,
    );
    let mut repeat_patches = Vec::with_capacity(physical.programs.len());
    for (tile, row) in physical.programs.iter_mut().enumerate() {
        if physical.outgoing_bases[tile].is_none() {
            // No common representable base: retain ordinary word patching.
            row.replace_outgoing_base_register(6, 15)?;
        }
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
        let mut patches = Vec::new();
        for index in 0..row.send_addresses().len() {
            let site = row.send_addresses()[index];
            let transfer = &pending[site.message as usize];
            if !transfer.moving_source() {
                continue;
            }
            let count = transfer
                .source_addresses
                .len()
                .max(bases.as_ref().map_or(1, Vec::len));
            let address = |iteration: usize| {
                let base = bases
                    .as_ref()
                    .map_or(0, |b| b.get(iteration).copied().unwrap_or(b[0]));
                repeat_source_address(transfer, iteration)
                    .checked_sub(base)
                    .and_then(|a| a.checked_add(site.byte_offset))
                    .ok_or(ExchangeLoweringError::Overflow)
            };
            let values = (0..count)
                .map(|iteration| Ok(row.relocated_send(index, address(iteration)?)?))
                .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
            if bases.is_none() && values[0] != row.words()[site.word_offset as usize] {
                return Err(ExchangeLoweringError::IncompatibleRepeatRows(
                    "relocation changes the first iteration",
                ));
            }
            row.set_send_address(index, address(0)?)?;
            if values.iter().any(|&value| value != values[0]) {
                patches.push(ExchangeRowPatch {
                    word_offset: site.word_offset,
                    values,
                });
            }
        }
        repeat_patches.push(patches);
    }
    physical.repeat_patches = repeat_patches;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exchange::{
        ExchangeActivityKind, ExchangeItemWidth, SchedulingProblem,
        materialize_valid_schedule_order, receive_configuration, schedule_problem,
        validate_exchange_schedule,
    };
    use crate::low::ExchangePhaseId;
    use ipu_exchange::diagnostic::sender_address_instruction_groups;
    use ipu_exchange::diagnostic::{PlanOperation, diagnose_plan_program};
    use ipu_exchange::patch_sender_instruction;
    use ipu_target::ipu21::fabric::Topology;

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
            let base_sites = fallback.programs[0].outgoing_base_writes();
            assert!(!base_sites.is_empty());
            for site in base_sites {
                assert_eq!(site.register, 15);
                assert_eq!(
                    fallback.programs[0].words()[site.word_offset as usize],
                    ipu_target::ipu21::instruction::encode_put_special_m(
                        ipu_target::ipu21::registers::OUTGOING_BASE,
                        site.register,
                    )
                    .unwrap(),
                );
            }
            assert!(
                !diagnose_plan_program(fallback.programs[0].words(), None)
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
                let mut row = fallback.programs[0].words().to_vec();
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
            let mut without_activities = physical.clone();
            without_activities
                .activities
                .iter_mut()
                .for_each(Vec::clear);
            relocate_repeat_rows(&mut without_activities, &pending, &addresses, &bindings).unwrap();
            relocate_repeat_rows(&mut physical, &pending, &addresses, &bindings).unwrap();
            assert_eq!(physical.programs, without_activities.programs);
            assert_eq!(physical.repeat_patches, without_activities.repeat_patches);
            assert_eq!(physical.outgoing_bases, without_activities.outgoing_bases);
            assert!(physical.repeat_patches.iter().all(Vec::is_empty));
            let row = physical.programs[0].words();
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
