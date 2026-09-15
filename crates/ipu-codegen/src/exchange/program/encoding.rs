//! Reuse an encoded prefix only while its sender, control, and lookahead inputs
//! remain unchanged. Speculative transfers still run the ordinary row encoder.
use crate::exchange::program::chunked::Chunked;
use crate::exchange::program::{
    EncodedRow, ExchangeError, OutgoingBaseWrite, ReceiveEvent, SendAddress, Sender,
    append_receive_events_record, append_sender_message, plan_event_cycles,
    validate_receive_events,
};
use ipu_target::ipu21::instruction::RETURN_M10_INSTRUCTION;

/// Standalone probe rows and complete phase rows share instruction emission.
/// The phase sink additionally retains relocation sites in shared chunks.
pub(super) trait RowSink: Extend<u32> {
    fn len(&self) -> usize;
    fn push(&mut self, word: u32);
    // Probe rows need only instructions. Phase rows retain relocation sites
    // at their final word positions.
    fn send_address(&mut self, _: u32, _: u32, _: u8) -> Result<(), ExchangeError> {
        Ok(())
    }
    fn receive_pointer(&mut self, _: usize) -> Result<(), ExchangeError> {
        Ok(())
    }
    fn outgoing_base(&mut self, _: u8) -> Result<(), ExchangeError> {
        Ok(())
    }
}
impl RowSink for Vec<u32> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn push(&mut self, word: u32) {
        Vec::push(self, word);
    }
}
#[derive(Clone, Debug, Default, PartialEq)]
struct RowBuffer {
    words: Chunked<u32>,
    sends: Chunked<SendAddress>,
    receive_pointers: Chunked<u32>,
    outgoing_bases: Chunked<OutgoingBaseWrite>,
}

impl RowBuffer {
    fn truncate(&mut self, words: usize) {
        self.words.truncate(words);
        self.sends.truncate(
            self.sends
                .partition_point(|site| (site.word_offset as usize) < words),
        );
        self.receive_pointers.truncate(
            self.receive_pointers
                .partition_point(|&offset| (offset as usize) < words),
        );
        self.outgoing_bases.truncate(
            self.outgoing_bases
                .partition_point(|site| (site.word_offset as usize) < words),
        );
    }

    fn finish(&self) -> EncodedRow {
        EncodedRow {
            words: self.words.to_vec(),
            sends: self.sends.to_vec(),
            receive_pointers: self.receive_pointers.to_vec(),
            outgoing_bases: self.outgoing_bases.to_vec(),
        }
    }
}

impl Extend<u32> for RowBuffer {
    fn extend<T: IntoIterator<Item = u32>>(&mut self, words: T) {
        self.words.extend(words);
    }
}

fn word_offset(offset: usize) -> Result<u32, ExchangeError> {
    u32::try_from(offset).map_err(|_| ExchangeError::Schedule("row word offset exceeds u32"))
}

impl RowSink for RowBuffer {
    fn len(&self) -> usize {
        self.words.len()
    }
    fn push(&mut self, word: u32) {
        self.words.push(word);
    }
    fn send_address(
        &mut self,
        message: u32,
        byte_offset: u32,
        item_shift: u8,
    ) -> Result<(), ExchangeError> {
        self.sends.push(SendAddress {
            word_offset: word_offset(self.len())?,
            message,
            byte_offset,
            item_shift,
        });
        Ok(())
    }
    fn receive_pointer(&mut self, offset: usize) -> Result<(), ExchangeError> {
        self.receive_pointers.push(word_offset(offset)?);
        Ok(())
    }
    fn outgoing_base(&mut self, register: u8) -> Result<(), ExchangeError> {
        self.outgoing_bases.push(OutgoingBaseWrite {
            word_offset: word_offset(self.len())?,
            register,
        });
        Ok(())
    }
}

#[derive(Debug)]
pub(super) struct EncodedSchedule {
    checkpoints: Chunked<Checkpoint>,
    row: RowBuffer,
    #[cfg(test)]
    resumed_words: usize,
}

impl EncodedSchedule {
    pub(super) fn finish(&self) -> EncodedRow {
        self.row.finish()
    }
    #[cfg(test)]
    pub(super) fn same_row(&self, other: &Self) -> bool {
        self.row == other.row
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Checkpoint {
    senders: usize,
    events: usize,
    words: usize,
    cycles: u32,
    // SENDPICP can advance towards the next receive control or gap boundary.
    // That lookahead is an input even though the next control isn't consumed.
    lookahead: Option<u32>,
}

impl Checkpoint {
    fn reusable(
        &self,
        senders: &Chunked<Sender>,
        events: &Chunked<ReceiveEvent>,
        horizon: u32,
    ) -> bool {
        let sender_start = senders
            .get(self.senders)
            .map_or(horizon, |sender| sender.timing.payload_start);
        let next_event = events.get(self.events);
        self.cycles <= sender_start
            && next_event.is_none_or(|event| event.cycles > self.cycles)
            && events
                .get(self.events.wrapping_sub(1))
                .is_none_or(|event| event.cycles < sender_start)
            && self.lookahead.is_none_or(|previous| {
                let next = next_event
                    .filter(|event| event.cycles < sender_start)
                    .map_or(sender_start, ReceiveEvent::issue_start);
                next == previous
            })
    }
}

pub(super) fn build_scheduled_program(
    senders: &Chunked<Sender>,
    receive_events: &Chunked<ReceiveEvent>,
    horizon_cycles: u32,
    prefix: Option<(&EncodedSchedule, usize, usize)>,
) -> Result<EncodedSchedule, ExchangeError> {
    debug_assert!(
        (1..senders.len())
            .all(|index| senders[index - 1].timing.payload_end
                <= senders[index].timing.payload_start)
    );
    let events = receive_events;
    // The prefix was validated already. Include the entire control group at
    // the edit boundary because controls sharing a cycle encode together.
    let mut changed = prefix.map_or(0, |(_, _, events)| events).min(events.len());
    if changed < events.len() {
        while changed > 0 && events[changed - 1].cycles == events[changed].cycles {
            changed -= 1;
        }
    }
    validate_receive_events(&events.slice(changed..events.len()))?;

    let mut words = RowBuffer::default();
    let mut checkpoints = Chunked::default();
    let mut resume = Checkpoint::default();
    if let Some((prefix, same_senders, same_events)) = prefix
        && let Some((index, checkpoint)) = (0..prefix.checkpoints.len())
            .rev()
            .map(|index| (index, prefix.checkpoints.get(index).unwrap()))
            .find(|(_, checkpoint)| {
                checkpoint.senders <= same_senders
                    && checkpoint.events <= same_events
                    && checkpoint.reusable(senders, events, horizon_cycles)
            })
    {
        resume = *checkpoint;
        words = prefix.row.clone();
        words.truncate(resume.words);
        checkpoints = prefix.checkpoints.clone();
        checkpoints.truncate(index + 1);
    }
    let mut event_cycles = resume.cycles;
    let mut event_index = resume.events;
    // The final boundary drains receive-only work through the phase horizon.
    for sender_index in resume.senders..=senders.len() {
        let sender = senders.get(sender_index);
        let boundary = sender.map_or(horizon_cycles, |sender| sender.timing.payload_start);
        let split = sender.map_or(events.len(), |_| {
            events.partition_point(|event| event.cycles <= boundary)
        });
        append_receive_events_record(
            &mut words,
            &mut event_cycles,
            &events.slice(event_index..split),
            boundary,
            |consumed, words, cycles, lookahead| {
                checkpoints.push(Checkpoint {
                    senders: sender_index,
                    events: event_index + consumed,
                    words,
                    cycles,
                    lookahead,
                })
            },
        )?;
        event_index = split;
        let Some(sender) = sender else { break };
        let split = events.partition_point(|event| event.cycles <= sender.timing.payload_end);
        append_sender_message(
            &mut words,
            &mut event_cycles,
            sender,
            &events.slice(event_index..split),
        )?;
        event_index = split;
        checkpoints.push(Checkpoint {
            senders: sender_index + 1,
            events: event_index,
            words: words.len(),
            cycles: event_cycles,
            lookahead: None,
        });
    }
    words.push(RETURN_M10_INSTRUCTION);
    debug_assert_eq!(plan_event_cycles(&words.words.to_vec())?, horizon_cycles);
    Ok(EncodedSchedule {
        checkpoints,
        row: words,
        #[cfg(test)]
        resumed_words: resume.words,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exchange::program::*;

    #[test]
    fn outgoing_base_uses_control_gaps_and_preserves_incremental_encoding() {
        let topology = Topology::c600();
        let mut builder = PhaseProgramBuilder::new(4);
        for round in 0..32 {
            let (source, receivers) = if round % 2 == 0 { (0, [2]) } else { (1, [0]) };
            let mut plan = multicast(&topology, source, &receivers, 64, 0).unwrap();
            patch_sender_address(&mut plan.sender, 0x10000 + round * 256).unwrap();
            patch_receiver_address(&mut plan.receivers[0], 0x50000 + round * 256).unwrap();
            plan.sender.message = round;
            let offset = builder
                .earliest_transfer_offset(source, &[], &receivers, &plan, 64, 0)
                .unwrap();
            builder
                .append_transfer_at(source, &[], &receivers, &plan, offset, 64)
                .unwrap();
            builder.finish().unwrap();
            if round % 2 == 1 {
                builder
                    .switch_outgoing_base(0, if round % 4 == 1 { 15 } else { 6 })
                    .unwrap();
            }
            for state in &builder.tile_states {
                let incremental = state.encoded().unwrap();
                let full = build_scheduled_program(
                    &state.senders,
                    &state.receive_events,
                    state.event_cycles,
                    None,
                )
                .unwrap();
                assert!(incremental.same_row(&full));
                state.finish().unwrap();
            }
        }
        let mut builder = PhaseProgramBuilder::new(4);
        let plan = multicast(&topology, 1, &[0], 256, 0).unwrap();
        let offset = builder
            .earliest_transfer_offset(1, &[], &[0], &plan, 256, 0)
            .unwrap();
        builder
            .append_transfer_at(1, &[], &[0], &plan, offset, 256)
            .unwrap();
        let receive_horizon = builder.tile_event_cycles(0).unwrap();
        builder.switch_outgoing_base(0, 15).unwrap();
        assert!(builder.tile_states[0].base_switches[0] < receive_horizon);
        assert_eq!(builder.tile_event_cycles(0).unwrap(), receive_horizon);
        builder.finish().unwrap();
    }

    #[test]
    fn validated_transfer_commits_its_encoded_trial_and_rejects_stale_trials() {
        let topology = Topology::c600();
        let plan = multicast(&topology, 0, &[2], 64, 0).unwrap();
        let mut builder = PhaseProgramBuilder::new(4);
        let offset = builder
            .earliest_transfer_offset(0, &[], &[2], &plan, 64, 0)
            .unwrap();
        let encoded = builder
            .staged
            .as_ref()
            .unwrap()
            .updates
            .iter()
            .map(|(tile, schedule)| (*tile, schedule.encoded().unwrap().clone()))
            .collect::<Vec<_>>();
        builder
            .append_transfer_at(0, &[], &[2], &plan, offset, 64)
            .unwrap();
        assert!(builder.staged.is_none());
        for (tile, row) in encoded {
            assert!(Arc::ptr_eq(
                &row,
                builder.tile_states[usize::from(tile)].encoded().unwrap()
            ));
        }
        let before = builder.finish().unwrap();
        let offset = builder
            .earliest_transfer_offset(0, &[], &[2], &plan, 64, 0)
            .unwrap();
        assert!(
            builder
                .append_transfer_at(0, &[2, 2], &[2], &plan, offset, 64)
                .is_err()
        );
        assert_eq!(builder.finish().unwrap(), before);
        // A different transfer must not accidentally commit the cached rows.
        builder
            .earliest_transfer_offset(0, &[], &[2], &plan, 64, 0)
            .unwrap();
        let other = multicast(&topology, 1, &[3], 32, 0).unwrap();
        let offset = builder
            .earliest_transfer_offset_deferred(1, &[], &[3], &other, 32, 0)
            .unwrap();
        let mut reference = builder.clone();
        reference.staged = None;
        reference
            .append_transfer_at(1, &[], &[3], &other, offset, 32)
            .unwrap();
        builder
            .append_transfer_at(1, &[], &[3], &other, offset, 32)
            .unwrap();
        assert_eq!(builder.finish().unwrap(), reference.finish().unwrap());
    }

    #[test]
    fn message_identity_survives_trial_replacement_and_insertion_before_a_cached_send() {
        let topology = Topology::c600();
        let mut late = multicast(&topology, 0, &[1], 32, 0).unwrap();
        let mut early = multicast(&topology, 0, &[2], 32, 0).unwrap();
        let mut builder = PhaseProgramBuilder::new(3);
        late.sender.message = 41;
        let offset = builder
            .earliest_transfer_offset(0, &[], &[1], &late, 32, 512)
            .unwrap();
        // Same words and timing, different caller identity: the speculative
        // row must not retain message 41 in its relocation metadata.
        late.sender.message = 7;
        builder
            .append_transfer_at(0, &[], &[1], &late, offset, 32)
            .unwrap();
        let row = builder.finish().unwrap().programs.remove(0).unwrap();
        assert_eq!(row.send_addresses()[0].message, 7);

        early.sender.message = 99;
        builder
            .append_transfer_at(0, &[], &[2], &early, 0, 32)
            .unwrap();
        let row = builder.finish().unwrap().programs.remove(0).unwrap();
        row.assert_relocation_sites([99, 7].into_iter());
    }

    #[test]
    fn rejected_receive_preserves_stream_and_encoding() {
        let row = multicast(&Topology::c600(), 0, &[2], 64, 0)
            .unwrap()
            .receivers[0]
            .clone();
        let mut schedule = TileProgramSchedule::default();
        schedule
            .append_receiver_at(&receive_row_timing_from_base(&row, 0).unwrap(), 0, 64)
            .unwrap();
        let encoded = schedule.encoded().unwrap().clone();
        let next = schedule
            .earliest_receiver_offset(&receive_row_timing_from_base(&row, 0).unwrap(), 64, 0)
            .unwrap();
        assert!(
            schedule
                .append_receiver_at(&receive_row_timing_from_base(&row, 0).unwrap(), 0, 64)
                .is_err()
        );
        assert!(Arc::ptr_eq(&encoded, schedule.encoded().unwrap()));
        assert_eq!(
            schedule
                .earliest_receiver_offset(&receive_row_timing_from_base(&row, 0).unwrap(), 64, 0)
                .unwrap(),
            next
        );
        schedule
            .append_receiver_at(&receive_row_timing_from_base(&row, 0).unwrap(), next, 64)
            .unwrap();
        schedule.finish().unwrap();
    }

    #[test]
    fn sender_insertion_checks_both_neighbors_without_changing_failed_history() {
        let row = multicast(&Topology::c600(), 0, &[2], 64, 0).unwrap().sender;
        let mut schedule = TileProgramSchedule::default();
        for offset in [3000, 1000, 2000] {
            schedule.append_sender_at(0, &row, offset).unwrap();
        }
        let before = schedule.finish().unwrap();
        for offset in [1999, 2000, 2001] {
            assert!(schedule.append_sender_at(0, &row, offset).is_err());
            assert_eq!(schedule.finish().unwrap(), before);
        }
        let mut ordered = TileProgramSchedule::default();
        for offset in [1000, 2000, 3000] {
            ordered.append_sender_at(0, &row, offset).unwrap();
        }
        assert_eq!(ordered.finish().unwrap(), before);
    }

    #[test]
    fn validation_budget_limits_effort_without_invalidating_the_schedule() {
        let topology = Topology::c600();
        let receivers = [2];
        let plan = multicast(&topology, 0, &receivers, 64, 0).unwrap();
        let mut builder = PhaseProgramBuilder::new(4).with_validation_budget(2);
        let offset = builder
            .earliest_transfer_offset(0, &[], &receivers, &plan, 64, 0)
            .unwrap();
        builder
            .append_transfer_at(0, &[], &receivers, &plan, offset, 64)
            .unwrap();
        assert_eq!(
            builder.earliest_transfer_offset(0, &[], &receivers, &plan, 64, 0),
            Err(ExchangeError::ValidationBudgetExceeded)
        );
        let offset = builder
            .earliest_transfer_offset_deferred(0, &[], &receivers, &plan, 64, 0)
            .unwrap();
        builder
            .append_transfer_at(0, &[], &receivers, &plan, offset, 64)
            .unwrap();
        builder.finish().unwrap();
    }

    #[test]
    #[ignore = "manual CPU benchmark"]
    fn benchmark_receive_validation() {
        let topology = Topology::c600();
        println!("transfers,elapsed_us,words,checksum");
        for count in [256, 1024, 4096] {
            let started = std::time::Instant::now();
            let mut schedule = TileProgramSchedule::default();
            let mut prefix = None;
            for index in 0..count {
                let plan = if index % 8 < 4 {
                    paired_multicast(&topology, 0, &[2, 3], 64).unwrap()
                } else {
                    multicast(&topology, 0, &[2], 64, 0).unwrap()
                };
                let mut row = plan.receivers[0].clone();
                patch_receiver_address(&mut row, 0x50000 + index % 128 * 512).unwrap();
                let offset = schedule
                    .earliest_receiver_offset(
                        &receive_row_timing_from_base(&row, 0).unwrap(),
                        64,
                        0,
                    )
                    .unwrap();
                schedule
                    .append_receiver_at(&receive_row_timing_from_base(&row, 0).unwrap(), offset, 64)
                    .unwrap();
                prefix = Some(schedule.encoded().unwrap().clone());
            }
            let encoded = prefix.unwrap();
            let checksum = encoded.row.words.iter().fold(0u64, |hash, &word| {
                hash.wrapping_mul(31).wrapping_add(u64::from(word))
            });
            println!(
                "{count},{},{},{checksum}",
                started.elapsed().as_micros(),
                encoded.row.words.len()
            );
        }
    }

    #[test]
    #[ignore = "manual CPU benchmark"]
    fn benchmark_incremental_encoding() {
        use std::{hint::black_box, time::Instant};
        let topology = Topology::c600();
        println!("transfers,full_ns,incremental_ns,speedup");
        for count in [32, 128, 512, 2048] {
            let mut schedule = TileProgramSchedule::default();
            let mut prefix = None;
            for index in 0..=count {
                if index == count {
                    prefix = Some(schedule.encoded().unwrap().clone());
                }
                let mut row = multicast(&topology, 0, &[2], 64, 0).unwrap().receivers[0].clone();
                patch_receiver_address(&mut row, 0x50000 + index % 256 * 256).unwrap();
                let offset = schedule
                    .earliest_receiver_offset(
                        &receive_row_timing_from_base(&row, 0).unwrap(),
                        64,
                        0,
                    )
                    .unwrap();
                schedule
                    .append_receiver_at(&receive_row_timing_from_base(&row, 0).unwrap(), offset, 64)
                    .unwrap();
            }
            let encode = |incremental| {
                build_scheduled_program(
                    &schedule.senders,
                    &schedule.receive_events,
                    schedule.event_cycles,
                    if incremental {
                        prefix
                            .as_deref()
                            .map(|prefix| (prefix, schedule.dirty_senders, schedule.dirty_events))
                    } else {
                        None
                    },
                )
                .unwrap()
            };
            assert_eq!(encode(false).row, encode(true).row);
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..5 {
                for choice in [round % 2, 1 - round % 2] {
                    let iterations = 100;
                    let start = Instant::now();
                    for _ in 0..iterations {
                        black_box(encode(choice == 1));
                    }
                    times[choice].push(start.elapsed().as_nanos() as f64 / iterations as f64);
                }
            }
            for time in &mut times {
                time.sort_by(f64::total_cmp);
            }
            println!(
                "{count},{:.0},{:.0},{:.2}",
                times[0][2],
                times[1][2],
                times[0][2] / times[1][2]
            );
        }
    }

    #[test]
    fn receive_only_rows_reuse_the_prefix_before_a_source_cutover() {
        let topology = Topology::c600();
        let mut schedule = TileProgramSchedule::default();
        for index in 0..128 {
            let mut row = multicast(&topology, 0, &[2], 64, 0).unwrap().receivers[0].clone();
            patch_receiver_address(&mut row, 0x50000 + index * 256).unwrap();
            let offset = schedule
                .earliest_receiver_offset(&receive_row_timing_from_base(&row, 0).unwrap(), 64, 0)
                .unwrap();
            schedule
                .append_receiver_at(&receive_row_timing_from_base(&row, 0).unwrap(), offset, 64)
                .unwrap();
            schedule.finish().unwrap();
        }
        let encoded = schedule.encoded().unwrap();
        assert!(encoded.resumed_words > encoded.row.words.len() / 2);
    }

    #[test]
    fn mixed_speculative_transfers_match_full_encoding() {
        let topology = Topology::c600();
        let mut random = fastrand::Rng::with_seed(0x656e_636f_6465);
        for _ in 0..4 {
            let mut builder = PhaseProgramBuilder::new(8);
            for index in 0..256 {
                let words = random.u32(64..=128);
                let (source, receivers, reserved, mut plan) = if index % 4 == 0 {
                    let source = random.u16(0..4) * 2;
                    let destination = (source + 2) % 8;
                    let receivers = vec![destination, destination + 1];
                    let plan = paired_multicast(&topology, source, &receivers, words).unwrap();
                    (source, receivers, vec![source ^ 1], plan)
                } else {
                    let source = random.u16(0..8);
                    let receivers = (0..8)
                        .filter(|&tile| (tile != source || index % 7 == 0) && random.bool())
                        .collect::<Vec<_>>();
                    if receivers.is_empty() || receivers == [source] {
                        continue;
                    }
                    let plan = multicast(&topology, source, &receivers, words, 0).unwrap();
                    (source, receivers, vec![], plan)
                };
                patch_sender_address(&mut plan.sender, random.u32(0..0x10000) * 8).unwrap();
                for receiver in &mut plan.receivers {
                    patch_receiver_address(receiver, random.u32(0..0x10000) * 8).unwrap();
                }
                plan.sender.message = index;
                let offset = builder
                    .earliest_transfer_offset(source, &reserved, &receivers, &plan, words, 0)
                    .unwrap();
                builder
                    .append_transfer_at(source, &reserved, &receivers, &plan, offset, words)
                    .unwrap();
                builder.finish().unwrap();
            }
        }
    }

    #[test]
    fn receive_control_at_send_start_is_issued_before_the_payload() {
        // A 28-word multicast loopback from the FP8 MLP has its mux teardown
        // at exactly the first outgoing word. The control is issued one cycle
        // earlier, so it belongs to the preceding receive-only interval.
        let mut schedule = TileProgramSchedule::default();
        schedule.senders.push(Sender {
            message: 0,
            instruction: 2070642691,
            words: 28,
            timing: ScheduledPayloadTiming {
                payload_start: 5354,
                payload_end: 5382,
                horizon: 5382,
            },
        });
        schedule.receive_events.push(ReceiveEvent {
            cycles: 5354,
            instruction: 1678165568,
            kind: ReceiveEventKind::OrdinaryNeutral,
        });
        schedule.event_cycles = 5382;
        let words = schedule.finish().unwrap().into_words();
        assert_eq!(plan_event_cycles(&words).unwrap(), 5382);
        diagnostic::validate_tile_program(0, &schedule, &words).unwrap();
    }
}
