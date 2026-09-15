//! Dependency-ready list scheduling with lazy endpoint-pressure priorities.
//! Unlike static ordering policies, selection consumes the common materializer's
//! live endpoint availability and reports each transfer's actual completion.
//! Row encoding and SRAM legality remain in MaterializedSchedule::append.
use super::{
    ExchangeLoweringError, MaterializedSchedule, PendingTransfer, SchedulingProblem,
    TileAvailability, TilePredecessor, schedule_encoding_is_valid,
};
use ipu_target::ipu21::fabric::Topology;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ReadyTransfer {
    moving_source: bool,
    earliest_start: Reverse<u32>,
    endpoint_pressure: u64,
    fanout: u16,
    words: u32,
    source: Reverse<u16>,
    index: Reverse<usize>,
}

/// Incrementally list-schedules dependency-ready multicast hyperedges. Heap
/// keys are lower bounds on their start time and are refreshed lazily as
/// shared endpoints become busy.
struct TransferScheduler<'a> {
    transfers: &'a [PendingTransfer],
    word_pressure: Vec<u64>,
    dynamic_word_pressure: bool,
    directional_pressure: bool,
    dependents: &'a [Vec<usize>],
    indegrees: Vec<usize>,
    dependency_ready: Vec<u32>,
    ready: BinaryHeap<ReadyTransfer>,
    ready_groups: Vec<BinaryHeap<ReadyTransfer>>,
    transfer_group: Vec<usize>,
    completed: usize,
}

impl<'a> TransferScheduler<'a> {
    #[cfg(test)]
    fn new(problem: &'a SchedulingProblem<'_>) -> Self {
        Self::with_priority(problem, ExchangeSchedulingPriority::Combined)
    }

    fn with_priority(
        problem: &'a SchedulingProblem<'_>,
        priority: ExchangeSchedulingPriority,
    ) -> Self {
        let transfers = problem.transfers;
        let priority = match priority {
            ExchangeSchedulingPriority::Automatic => {
                // Point-to-point traffic benefits from draining the remaining
                // send/receive workloads independently. Multicast choices free
                // several receivers together and retain their combined pressure.
                if transfers
                    .iter()
                    .all(|transfer| transfer.destinations.len() == 1)
                {
                    ExchangeSchedulingPriority::RemainingDirectional
                } else {
                    ExchangeSchedulingPriority::Combined
                }
            }
            priority => priority,
        };

        let directional = matches!(priority, ExchangeSchedulingPriority::RemainingDirectional);
        let word_pressure = if directional {
            let mut pressure = vec![0; usize::from(problem.tile_count) * 2];
            for transfer in transfers {
                for resource in transfer.pressure_resources(true) {
                    pressure[resource] +=
                        u64::from(transfer.item_count().unwrap_or(transfer.words));
                }
            }
            pressure
        } else {
            problem.word_pressure.clone()
        };
        let mut scheduler = Self {
            transfers,
            word_pressure,
            directional_pressure: directional,
            // Both production policies update remaining work. Static pressure
            // remains available for controlled offline comparisons.
            dynamic_word_pressure: priority == ExchangeSchedulingPriority::RemainingDirectional
                || (priority == ExchangeSchedulingPriority::Combined
                    && transfers
                        .iter()
                        .any(|transfer| transfer.destinations.len() > 1)),
            dependents: &problem.dependents,
            indegrees: problem.indegrees(),
            dependency_ready: vec![0; transfers.len()],
            ready: BinaryHeap::new(),
            ready_groups: Vec::new(),
            transfer_group: Vec::new(),
            completed: 0,
        };
        for index in 0..transfers.len() {
            if scheduler.indegrees[index] == 0 {
                scheduler.push_ready(index, 0);
            }
        }
        // Initially ready transfers sharing the same endpoint roles have the
        // same changing readiness and pressure. Their relative word/index
        // priority is static, so only the best member needs a global entry.
        // Later dependency releases remain individual entries.
        let mut groups = BTreeMap::new();
        scheduler.transfer_group.resize(transfers.len(), usize::MAX);
        for candidate in std::mem::take(&mut scheduler.ready).into_vec() {
            let index = candidate.index.0;
            let transfer = &transfers[index];
            let receivers = transfer
                .destinations
                .iter()
                .map(|&(tile, _)| tile)
                .collect::<Vec<_>>();
            let group = *groups
                .entry((transfer.source, transfer.reserved_source, receivers))
                .or_insert_with(|| {
                    let group = scheduler.ready_groups.len();
                    scheduler.ready_groups.push(BinaryHeap::new());
                    group
                });
            scheduler.transfer_group[index] = group;
            scheduler.ready_groups[group].push(candidate);
        }
        scheduler.ready.extend(
            scheduler
                .ready_groups
                .iter()
                .filter_map(|queue| queue.peek().copied()),
        );
        scheduler
    }

    fn push_ready(&mut self, index: usize, earliest_start: u32) {
        let transfer = &self.transfers[index];
        let endpoint_pressure = transfer
            .pressure_resources(self.directional_pressure)
            // Bytes, rather than role count, approximate how long selecting
            // this hyperedge frees work on the phase's congested endpoints.
            .map(|tile| self.word_pressure[tile])
            .sum::<u64>();
        self.ready.push(ReadyTransfer {
            moving_source: transfer.moving_source(),
            earliest_start: Reverse(earliest_start),
            endpoint_pressure,
            fanout: u16::try_from(transfer.destinations.len()).expect("receivers fit tile count"),
            words: transfer.item_count().unwrap_or(transfer.words),
            source: Reverse(transfer.source),
            index: Reverse(index),
        });
    }

    fn refresh(
        &self,
        mut candidate: ReadyTransfer,
        availability: &[TileAvailability],
    ) -> ReadyTransfer {
        let index = candidate.index.0;
        let transfer = &self.transfers[index];
        candidate.earliest_start = Reverse(
            std::iter::once(self.dependency_ready[index])
                .chain(std::iter::once(
                    availability[usize::from(transfer.source)].send,
                ))
                .chain(
                    transfer
                        .reserved_source
                        .into_iter()
                        .map(|tile| availability[usize::from(tile)].send),
                )
                .chain(
                    transfer
                        .destinations
                        .iter()
                        .map(|&(tile, _)| availability[usize::from(tile)].receive),
                )
                .max()
                .unwrap_or(0),
        );
        if self.dynamic_word_pressure {
            candidate.endpoint_pressure = transfer
                .pressure_resources(self.directional_pressure)
                .map(|tile| self.word_pressure[tile])
                .sum();
        }
        candidate
    }

    fn next(&mut self, tile_availability: &[TileAvailability]) -> Option<(usize, u32)> {
        let mut repairs = 0;
        loop {
            let candidate = *self.ready.peek()?;
            let current = self.refresh(candidate, tile_availability);
            if candidate == current {
                let index = candidate.index.0;
                // Readiness ranks the queue; payload dependencies alone gate
                // the row builder, which pipelines source selection/delivery.
                if let Some(group) = self
                    .transfer_group
                    .get(index)
                    .copied()
                    .filter(|&group| group != usize::MAX)
                {
                    let queue = &mut self.ready_groups[group];
                    let head = queue.pop().expect("nonempty ready group");
                    debug_assert_eq!(head.index.0, index);
                    if let Some(mut next) = queue.peek().copied() {
                        next.earliest_start = current.earliest_start;
                        *self.ready.peek_mut().expect("ready head") = next;
                    } else {
                        self.ready.pop();
                    }
                } else {
                    self.ready.pop();
                }
                return Some((index, self.dependency_ready[index]));
            }
            repairs += 1;
            // A wave can invalidate most keys. Once logarithmic root repairs
            // cost a linear scan, refresh/reheapify once instead. Charge heap
            // traversal four times the contiguous scan: the large ViT captures
            // benefit from refreshing sooner than comparison counts suggest.
            // Both readiness
            // and pressure are monotone bounds, so this preserves eager priority.
            if self.ready.len() >= 128
                && 4 * repairs * self.ready.len().ilog2() as usize >= self.ready.len()
            {
                let mut entries = std::mem::take(&mut self.ready).into_vec();
                for entry in &mut entries {
                    *entry = self.refresh(*entry, tile_availability);
                }
                self.ready = BinaryHeap::from(entries);
            } else {
                *self.ready.peek_mut().expect("ready head") = current;
            }
        }
    }

    fn complete(&mut self, index: usize, completion: u32) {
        self.completed += 1;
        let transfer = &self.transfers[index];
        if self.dynamic_word_pressure {
            let items = u64::from(transfer.item_count().unwrap_or(transfer.words));
            for tile in transfer.pressure_resources(self.directional_pressure) {
                self.word_pressure[tile] = self.word_pressure[tile].saturating_sub(items);
            }
        }
        for &dependent in &self.dependents[index] {
            self.dependency_ready[dependent] = self.dependency_ready[dependent].max(completion);
            self.indegrees[dependent] -= 1;
            if self.indegrees[dependent] == 0 {
                self.push_ready(dependent, self.dependency_ready[dependent]);
            }
        }
    }

    fn is_complete(&self) -> bool {
        self.completed == self.transfers.len()
    }
}

pub(super) fn schedule(
    topology: &Topology,
    problem: &SchedulingProblem<'_>,
    incoming_bases: &[u32],
    receive_counts: &[usize],
    priority: ExchangeSchedulingPriority,
) -> Result<MaterializedSchedule, ExchangeLoweringError> {
    let pending = problem.transfers;
    let schedule = build(
        topology,
        problem,
        incoming_bases,
        receive_counts,
        false,
        priority,
    )?;
    if schedule_encoding_is_valid(&schedule)? {
        return Ok(schedule);
    }
    tracing::info!(
        transfers = pending.len(),
        "retrying exchange schedule with incremental instruction-alignment validation"
    );
    let started = std::time::Instant::now();
    let result = build(
        topology,
        problem,
        incoming_bases,
        receive_counts,
        true,
        priority,
    );
    tracing::info!(
        transfers = pending.len(),
        elapsed_ms = started.elapsed().as_millis(),
        success = result.is_ok(),
        "finished incremental instruction-alignment validation"
    );
    result
}

fn build(
    topology: &Topology,
    problem: &SchedulingProblem<'_>,
    incoming_bases: &[u32],
    receive_counts: &[usize],
    validate_encoding: bool,
    priority: ExchangeSchedulingPriority,
) -> Result<MaterializedSchedule, ExchangeLoweringError> {
    let pending = problem.transfers;
    let tile_count = problem.tile_count;
    let mut schedule = MaterializedSchedule::new(tile_count, pending);
    let mut scheduler = TransferScheduler::with_priority(problem, priority);
    let mut last_transfer = vec![TilePredecessor::default(); usize::from(tile_count)];
    while let Some((index, dependency_ready)) = scheduler.next(&schedule.tile_availability) {
        let completion = schedule.append(
            topology,
            pending,
            incoming_bases,
            receive_counts,
            index,
            dependency_ready,
            validate_encoding,
            &mut last_transfer,
        )?;
        scheduler.complete(index, completion);
    }
    debug_assert!(scheduler.is_complete());
    schedule.finish_horizon();
    Ok(schedule)
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    #[test]
    fn grouped_ready_queue_matches_eager_priority() {
        let mut random = fastrand::Rng::with_seed(0x7061697273);
        for case in 0..16 {
            let tiles = if case & 4 == 0 { 8 } else { 32 };
            let transfers = (0..1024)
                .map(|index| {
                    let source = random.u16(0..tiles);
                    let destination = (source + random.u16(1..tiles)) % tiles;
                    let words = random.u32(1..=512);
                    let source_address = if case % 2 == 0 {
                        0
                    } else {
                        0x80000 + random.u32(0..16) * 4096
                    };
                    let destination_address = if case % 2 == 0 {
                        0x80000 + index * 4096
                    } else {
                        0x80000 + random.u32(0..16) * 4096
                    };
                    let mut destinations = vec![(destination, destination_address)];
                    if case & 2 != 0 {
                        let other = (destination + 1) % tiles;
                        if other != source {
                            destinations.push((other, destination_address));
                        }
                    }
                    PendingTransfer {
                        source,
                        source_shard: BlockValueId::from_index(u32::from(source)),
                        source_offset: 0,
                        source_addresses: vec![source_address],
                        source_elements: effective_memory_elements(source_address, words),
                        destinations,
                        words,
                        width: ExchangeItemWidth::Word32,
                        reserved_source: None,
                    }
                })
                .collect::<Vec<_>>();
            let problem = SchedulingProblem::new(&transfers, tiles);
            for priority in [
                ExchangeSchedulingPriority::Automatic,
                ExchangeSchedulingPriority::Combined,
                ExchangeSchedulingPriority::RemainingDirectional,
            ] {
                let mut grouped = TransferScheduler::with_priority(&problem, priority);
                assert!(!grouped.ready_groups.is_empty());
                assert!(grouped.ready_groups.len() <= 2 * usize::from(tiles * (tiles - 1)));
                let mut reference = TransferScheduler::with_priority(&problem, priority);
                reference.ready = std::mem::take(&mut reference.ready_groups)
                    .into_iter()
                    .flatten()
                    .collect();
                reference.transfer_group.clear();
                let mut availability = vec![TileAvailability::default(); usize::from(tiles)];
                while let Some(actual) = grouped.next(&availability) {
                    let expected = reference
                        .ready
                        .iter()
                        .map(|&entry| reference.refresh(entry, &availability))
                        .max()
                        .unwrap();
                    assert_eq!(
                        actual,
                        (
                            expected.index.0,
                            reference.dependency_ready[expected.index.0]
                        )
                    );
                    reference
                        .ready
                        .retain(|entry| entry.index != expected.index);
                    let (index, dependency) = actual;
                    let transfer = &transfers[index];
                    let source = usize::from(transfer.source);
                    let completion = transfer
                        .destinations
                        .iter()
                        .map(|&(tile, _)| availability[usize::from(tile)].receive)
                        .chain([availability[source].send, dependency])
                        .max()
                        .unwrap()
                        + transfer.words;
                    availability[source].send = completion;
                    for &(tile, _) in &transfer.destinations {
                        availability[usize::from(tile)].receive = completion;
                    }
                    grouped.complete(index, completion);
                    reference.complete(index, completion);
                }
                assert!(grouped.is_complete());
                assert!(reference.is_complete());
            }
        }
    }

    #[test]
    fn randomized_transfer_schedules_preserve_hazards_without_same_role_overlap() {
        let mut random = fastrand::Rng::with_seed(0x736c_6f74);
        for _ in 0..64 {
            let tile_count = random.u16(2..=32);
            let transfer_count = random.usize(1..=256);
            let transfers = (0..transfer_count)
                .map(|_| {
                    let source = random.u16(0..tile_count);
                    let receiver_count = random.usize(1..=usize::from(tile_count.min(8) - 1));
                    let mut receivers = Vec::with_capacity(receiver_count);
                    while receivers.len() != receiver_count {
                        let tile = random.u16(0..tile_count);
                        if tile != source && !receivers.contains(&tile) {
                            receivers.push(tile);
                        }
                    }
                    let words = random.u32(1..=MAX_TRANSFER_WORDS);
                    PendingTransfer {
                        source,
                        source_shard: BlockValueId::from_index(u32::from(source)),
                        source_offset: 0,
                        destinations: receivers.into_iter().map(|tile| (tile, 0)).collect(),
                        source_addresses: vec![0],
                        source_elements: effective_memory_elements(0, words),
                        words,
                        width: ExchangeItemWidth::Word32,
                        reserved_source: None,
                    }
                })
                .collect::<Vec<_>>();
            let dependencies = memory_dependencies(&transfers, tile_count);
            let problem = SchedulingProblem::new(&transfers, tile_count);
            let mut scheduler = TransferScheduler::new(&problem);
            let mut availability = vec![TileAvailability::default(); usize::from(tile_count)];
            let mut occurrences = vec![0u8; transfers.len()];
            let mut intervals = vec![(0u32, 0u32); transfers.len()];
            while let Some((index, dependency_ready)) = scheduler.next(&availability) {
                occurrences[index] += 1;
                let transfer = &transfers[index];
                let start = std::iter::once(dependency_ready)
                    .chain(std::iter::once(
                        availability[usize::from(transfer.source)].send,
                    ))
                    .chain(
                        transfer
                            .destinations
                            .iter()
                            .map(|&(tile, _)| availability[usize::from(tile)].receive),
                    )
                    .max()
                    .unwrap_or(0);
                let end = start.saturating_add(transfers[index].words);
                intervals[index] = (start, end);
                availability[usize::from(transfer.source)].send = end;
                for &(tile, _) in &transfer.destinations {
                    availability[usize::from(tile)].receive = end;
                }
                scheduler.complete(index, end);
            }
            assert!(scheduler.is_complete());
            assert!(occurrences.into_iter().all(|count| count == 1));
            for &(before, after) in &dependencies {
                assert!(intervals[before].1 <= intervals[after].0);
            }
            for tile in 0..tile_count {
                let mut send_intervals = transfers
                    .iter()
                    .enumerate()
                    .filter(|(_, transfer)| transfer.source == tile)
                    .map(|(index, _)| intervals[index])
                    .collect::<Vec<_>>();
                send_intervals.sort_unstable();
                assert!(send_intervals.windows(2).all(|pair| pair[0].1 <= pair[1].0));
                let mut receive_intervals = transfers
                    .iter()
                    .enumerate()
                    .filter(|(_, transfer)| {
                        transfer
                            .destinations
                            .iter()
                            .any(|&(destination, _)| destination == tile)
                    })
                    .map(|(index, _)| intervals[index])
                    .collect::<Vec<_>>();
                receive_intervals.sort_unstable();
                assert!(
                    receive_intervals
                        .windows(2)
                        .all(|pair| pair[0].1 <= pair[1].0)
                );
            }

            let mut incumbent = MaterializedSchedule::new(tile_count, &transfers);
            incumbent.order.extend(0..transfers.len());
            let mut last_transfer = vec![None; usize::from(tile_count)];
            for (index, transfer) in transfers.iter().enumerate() {
                let predecessor = transfer
                    .tiles()
                    .filter_map(|tile| last_transfer[usize::from(tile)])
                    .max();
                for tile in transfer.tiles() {
                    last_transfer[usize::from(tile)] = Some(index);
                }
                incumbent.timings[index] = Some(MaterializedTiming {
                    end: index as u32 + 1,
                    predecessor,
                });
            }
            let repaired = repair::order(
                &SchedulingProblem::new(&transfers, tile_count),
                &incumbent,
                true,
            )
            .unwrap();
            let mut repaired_positions = vec![usize::MAX; transfers.len()];
            for (position, &index) in repaired.iter().enumerate() {
                assert_eq!(repaired_positions[index], usize::MAX);
                repaired_positions[index] = position;
            }
            assert!(
                repaired_positions
                    .iter()
                    .all(|position| *position != usize::MAX)
            );
            for &(before, after) in &dependencies {
                assert!(repaired_positions[before] < repaired_positions[after]);
            }
        }
    }
}

/// Endpoint-pressure policy for dependency-ready list scheduling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum ExchangeSchedulingPriority {
    #[default]
    Automatic,
    Combined,
    RemainingDirectional,
    #[cfg(test)]
    Streams(u32),
    #[cfg(test)]
    BalancedStreams(u32),
}
