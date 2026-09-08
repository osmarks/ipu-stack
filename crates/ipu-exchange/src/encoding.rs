//! Reuse an encoded prefix only while its sender, control, and lookahead inputs
//! remain unchanged. Speculative transfers still run the ordinary row encoder.
use super::*;

// Trials share immutable chunks and copy at most the final partial chunk.
// A flat chunk index keeps lookup/rollback bounded and avoids recursive drops.
const CHUNK_ITEMS: usize = 128;

#[derive(Clone, Debug)]
struct Chunks<T> {
    chunks: Vec<Arc<Vec<T>>>,
    len: usize,
}

impl<T> Default for Chunks<T> {
    fn default() -> Self {
        Self {
            chunks: Vec::new(),
            len: 0,
        }
    }
}

impl<T: Clone> Chunks<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn get(&self, index: usize) -> &T {
        &self.chunks[index / CHUNK_ITEMS][index % CHUNK_ITEMS]
    }
    fn iter(&self) -> impl DoubleEndedIterator<Item = &T> {
        self.chunks.iter().flat_map(|chunk| chunk.iter())
    }
    fn push(&mut self, value: T) {
        if self.len.is_multiple_of(CHUNK_ITEMS) {
            self.chunks.push(Arc::new(Vec::with_capacity(CHUNK_ITEMS)));
        }
        Arc::make_mut(self.chunks.last_mut().unwrap()).push(value);
        self.len += 1;
    }
    fn truncate(&mut self, len: usize) {
        assert!(len <= self.len);
        self.chunks.truncate(len.div_ceil(CHUNK_ITEMS));
        if !len.is_multiple_of(CHUNK_ITEMS) {
            Arc::make_mut(self.chunks.last_mut().unwrap()).truncate(len % CHUNK_ITEMS);
        }
        self.len = len;
    }
    fn to_vec(&self) -> Vec<T> {
        self.iter().cloned().collect()
    }
}

impl<T: Clone + PartialEq> PartialEq for Chunks<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().eq(other.iter())
    }
}
impl<T: Clone> Extend<T> for Chunks<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.push(value);
        }
    }
}

/// The row encoder needs only append and absolute word parity. Primitive rows
/// use a Vec; speculative phase rows share chunks through the same encoder.
pub(super) trait RowWords: Extend<u32> {
    fn len(&self) -> usize;
    fn push(&mut self, word: u32);
}
impl RowWords for Vec<u32> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn push(&mut self, word: u32) {
        Vec::push(self, word);
    }
}
impl RowWords for Chunks<u32> {
    fn len(&self) -> usize {
        self.len
    }
    fn push(&mut self, word: u32) {
        Chunks::push(self, word);
    }
}

#[derive(Debug)]
pub(super) struct EncodedSchedule {
    checkpoints: Chunks<Checkpoint>,
    words: Chunks<u32>,
    #[cfg(test)]
    resumed_words: usize,
}

impl EncodedSchedule {
    pub(super) fn words(&self) -> Vec<u32> {
        self.words.to_vec()
    }
    #[cfg(test)]
    pub(super) fn same_words(&self, other: &Self) -> bool {
        self.words == other.words
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
    prefix: Option<(&EncodedSchedule, usize, usize)>,
) -> Result<EncodedSchedule, ExchangeError> {
    debug_assert!(
        senders
            .windows(2)
            .all(|pair| pair[0].end_cycles <= pair[1].start_cycles)
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
    validate_receive_events(&events[changed..])?;

    let mut words = Chunks::default();
    let mut checkpoints = Chunks::default();
    let mut resume = Checkpoint::default();
    if let Some((prefix, same_senders, same_events)) = prefix
        && let Some((index, checkpoint)) = (0..prefix.checkpoints.len())
            .rev()
            .map(|index| (index, prefix.checkpoints.get(index)))
            .find(|(_, checkpoint)| {
                checkpoint.senders <= same_senders
                    && checkpoint.events <= same_events
                    && checkpoint.reusable(senders, events, horizon_cycles)
            })
    {
        resume = *checkpoint;
        words = prefix.words.clone();
        words.truncate(resume.words);
        checkpoints = prefix.checkpoints.clone();
        checkpoints.truncate(index + 1);
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
    debug_assert_eq!(plan_event_cycles(&words.to_vec())?, horizon_cycles);
    Ok(EncodedSchedule {
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
        let row = Topology::c600()
            .multicast(0, &[2], 64, 0)
            .unwrap()
            .receivers[0];
        let mut schedule = TileProgramSchedule::default();
        schedule.append_receiver_at(&row, 0, 64).unwrap();
        let encoded = schedule.encoded().unwrap().clone();
        let next = schedule.earliest_receiver_offset(&row, 64, 0).unwrap();
        assert!(schedule.append_receiver_at(&row, 0, 64).is_err());
        assert!(Arc::ptr_eq(&encoded, schedule.encoded().unwrap()));
        assert_eq!(
            schedule.earliest_receiver_offset(&row, 64, 0).unwrap(),
            next
        );
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
                prefix = Some(schedule.encoded().unwrap().clone());
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
