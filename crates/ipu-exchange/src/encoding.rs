//! Reuse an encoded prefix only while its sender, control, and lookahead inputs
//! remain unchanged. Speculative transfers still run the ordinary row encoder.
use super::*;

#[derive(Debug)]
pub(super) struct EncodedSchedule {
    senders: Vec<ScheduledSenderRow>,
    events: Vec<ReceiveEvent>,
    checkpoints: Vec<Checkpoint>,
    pub(super) words: Vec<u32>,
    #[cfg(test)]
    resumed_words: usize,
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
        senders: &[ScheduledSenderRow],
        events: &[ReceiveEvent],
        horizon: u32,
    ) -> bool {
        let sender_start = senders
            .get(self.senders)
            .map_or(horizon, |sender| sender.start_cycles);
        let next_event = events.get(self.events);
        self.cycles <= sender_start
            && next_event.is_none_or(|event| event.cycles > self.cycles)
            && events
                .get(self.events.wrapping_sub(1))
                .is_none_or(|event| event.cycles < sender_start)
            && self.lookahead.is_none_or(|previous| {
                let next = next_event
                    .filter(|event| event.cycles < sender_start)
                    .map_or(sender_start, |event| event.cycles.saturating_sub(1));
                next == previous
            })
    }
}

pub(super) fn build_scheduled_program(
    senders: &[ScheduledSenderRow],
    receive_events: &[ReceiveEvent],
    horizon_cycles: u32,
    prefix: Option<&EncodedSchedule>,
) -> Result<EncodedSchedule, ExchangeError> {
    let mut senders = senders.to_vec();
    senders.sort_by_key(|sender| sender.start_cycles);
    if senders
        .windows(2)
        .any(|pair| pair[0].end_cycles > pair[1].start_cycles)
    {
        return Err(ExchangeError::Schedule("overlapping outgoing messages"));
    }
    let mut events = receive_events.to_vec();
    events.sort_by_key(|event| event.cycles);
    validate_receive_events(&events)?;

    let mut words = Vec::new();
    let mut checkpoints = Vec::new();
    let mut resume = Checkpoint::default();
    if let Some(prefix) = prefix {
        let same_senders = senders
            .iter()
            .zip(&prefix.senders)
            .take_while(|(a, b)| a == b)
            .count();
        let same_events = events
            .iter()
            .zip(&prefix.events)
            .take_while(|(a, b)| a == b)
            .count();
        if let Some((index, checkpoint)) =
            prefix
                .checkpoints
                .iter()
                .enumerate()
                .rev()
                .find(|(_, checkpoint)| {
                    checkpoint.senders <= same_senders
                        && checkpoint.events <= same_events
                        && checkpoint.reusable(&senders, &events, horizon_cycles)
                })
        {
            resume = *checkpoint;
            words.extend_from_slice(&prefix.words[..resume.words]);
            checkpoints.extend_from_slice(&prefix.checkpoints[..=index]);
        }
    }
    let mut event_cycles = resume.cycles;
    let mut event_index = resume.events;
    for (sender_index, sender) in senders.iter().enumerate().skip(resume.senders) {
        let split = event_index
            + events[event_index..].partition_point(|event| event.cycles <= sender.start_cycles);
        append_receive_events_record(
            &mut words,
            &mut event_cycles,
            &events[event_index..split],
            sender.start_cycles,
            true,
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
        let split = event_index
            + events[event_index..].partition_point(|event| event.cycles <= sender.end_cycles);
        append_sender_message(
            &mut words,
            &mut event_cycles,
            sender,
            &events[event_index..split],
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
    append_receive_events_record(
        &mut words,
        &mut event_cycles,
        &events[event_index..],
        horizon_cycles,
        true,
        |consumed, words, cycles, lookahead| {
            checkpoints.push(Checkpoint {
                senders: senders.len(),
                events: event_index + consumed,
                words,
                cycles,
                lookahead,
            })
        },
    )?;
    words.push(RETURN_M10_INSTRUCTION);
    debug_assert_eq!(plan_event_cycles(&words)?, horizon_cycles);
    Ok(EncodedSchedule {
        senders,
        events,
        checkpoints,
        words,
        #[cfg(test)]
        resumed_words: resume.words,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_receive_preserves_stream_and_encoding() {
        let row = Topology::c600().multicast(0, &[2], 64, 0).unwrap().receivers[0];
        let mut schedule = TileProgramSchedule::default();
        schedule.append_receiver_at(&row, 0, 64).unwrap();
        let encoded = schedule.encoded().unwrap().clone();
        let next = schedule.earliest_receiver_offset(&row, 64, 0).unwrap();
        assert!(schedule.append_receiver_at(&row, 0, 64).is_err());
        assert!(Arc::ptr_eq(&encoded, schedule.encoded().unwrap()));
        assert_eq!(schedule.earliest_receiver_offset(&row, 64, 0).unwrap(), next);
        schedule.append_receiver_at(&row, next, 64).unwrap();
        schedule.finish().unwrap();
    }

    #[test]
    fn sender_insertion_checks_both_neighbors_without_changing_failed_history() {
        let row = Topology::c600().multicast(0, &[2], 64, 0).unwrap().sender;
        let mut schedule = TileProgramSchedule::default();
        for offset in [3000, 1000, 2000] {
            schedule.append_sender_at(&row, offset).unwrap();
        }
        let before = schedule.finish().unwrap();
        for offset in [1999, 2000, 2001] {
            assert!(schedule.append_sender_at(&row, offset).is_err());
            assert_eq!(schedule.finish().unwrap(), before);
        }
        let mut ordered = TileProgramSchedule::default();
        for offset in [1000, 2000, 3000] {
            ordered.append_sender_at(&row, offset).unwrap();
        }
        assert_eq!(ordered.finish().unwrap(), before);
    }

    #[test]
    fn validation_budget_limits_effort_without_invalidating_the_schedule() {
        let topology = Topology::c600();
        let receivers = [2];
        let plan = topology.multicast(0, &receivers, 64, 0).unwrap();
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
                    topology.paired_multicast(0, &[2, 3], 64).unwrap()
                } else {
                    topology.multicast(0, &[2], 64, 0).unwrap()
                };
                let mut row = plan.receivers[0];
                patch_receiver_address(&mut row, 0x50000 + index % 128 * 512).unwrap();
                let offset = schedule.earliest_receiver_offset(&row, 64, 0).unwrap();
                schedule.append_receiver_at(&row, offset, 64).unwrap();
                prefix = Some(
                    build_scheduled_program(
                        &schedule.senders,
                        &schedule.receive_events,
                        schedule.event_cycles,
                        prefix.as_ref(),
                    )
                    .unwrap(),
                );
            }
            let encoded = prefix.unwrap();
            let checksum = encoded.words.iter().fold(0u64, |hash, &word| {
                hash.wrapping_mul(31).wrapping_add(u64::from(word))
            });
            println!(
                "{count},{},{},{checksum}",
                started.elapsed().as_micros(),
                encoded.words.len()
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
                let mut row = topology.multicast(0, &[2], 64, 0).unwrap().receivers[0];
                patch_receiver_address(&mut row, 0x50000 + index % 256 * 256).unwrap();
                let offset = schedule.earliest_receiver_offset(&row, 64, 0).unwrap();
                schedule.append_receiver_at(&row, offset, 64).unwrap();
            }
            let encode = |incremental| {
                build_scheduled_program(
                    &schedule.senders,
                    &schedule.receive_events,
                    schedule.event_cycles,
                    if incremental { prefix.as_deref() } else { None },
                )
                .unwrap()
            };
            assert_eq!(encode(false).words, encode(true).words);
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
            let mut row = topology.multicast(0, &[2], 64, 0).unwrap().receivers[0];
            patch_receiver_address(&mut row, 0x50000 + index * 256).unwrap();
            let offset = schedule.earliest_receiver_offset(&row, 64, 0).unwrap();
            schedule.append_receiver_at(&row, offset, 64).unwrap();
            schedule.finish().unwrap();
        }
        let encoded = schedule.encoded().unwrap();
        assert!(encoded.resumed_words > encoded.words.len() / 2);
    }

    #[test]
    fn mixed_speculative_transfers_match_full_encoding() {
        let topology = Topology::c600();
        let mut random = fastrand::Rng::with_seed(0x656e_636f_6465);
        for _ in 0..4 {
            let mut builder = PhaseProgramBuilder::new(8);
            for index in 0..256 {
                let words = random.u32(64..=128);
                let (source, receivers, reserved, plan) = if index % 4 == 0 {
                    let source = random.u16(0..4) * 2;
                    let destination = (source + 2) % 8;
                    let receivers = vec![destination, destination + 1];
                    let plan = topology
                        .paired_multicast(source, &receivers, words)
                        .unwrap();
                    (source, receivers, vec![source ^ 1], plan)
                } else {
                    let source = random.u16(0..8);
                    let receivers = (0..8)
                        .filter(|&tile| (tile != source || index % 7 == 0) && random.bool())
                        .collect::<Vec<_>>();
                    if receivers.is_empty() || receivers == [source] {
                        continue;
                    }
                    let plan = topology.multicast(source, &receivers, words, 0).unwrap();
                    (source, receivers, vec![], plan)
                };
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
        schedule.senders.push(ScheduledSenderRow {
            row: [
                1098907651, 1084227612, 2070642691, 1134559232, 0, 0, 0, 0, 0,
            ],
            start_cycles: 5354,
            end_cycles: 5382,
        });
        schedule.receive_events.push(ReceiveEvent {
            cycles: 5354,
            instruction: 1678165568,
            kind: ReceiveEventKind::OrdinaryNeutral,
        });
        schedule.event_cycles = 5382;
        let words = schedule.finish().unwrap();
        assert_eq!(plan_event_cycles(&words).unwrap(), 5382);
        diagnostic::validate_tile_program(0, &schedule, &words).unwrap();
    }
}
