//! Reorder an incumbent's critical neighborhood using endpoint pressure and
//! receive-address continuity. The common materializer evaluates the result.
use super::{MaterializedSchedule, PendingTransfer, SchedulingProblem, TileAvailability};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RepairReady {
    // Repair a bounded window of the incumbent before considering later work.
    // With time first, every endpoint advance repriced candidates throughout
    // the phase, turning a local improvement into another global search.
    window: Reverse<usize>,
    earliest_start: Reverse<u32>,
    epoch: Reverse<usize>,
    contiguous_receivers: usize,
    in_neighborhood: bool,
    endpoint_pressure: u64,
    incumbent_rank: Reverse<usize>,
    index: Reverse<usize>,
}

fn repair_ready(
    index: usize,
    pending: &[PendingTransfer],
    availability: &[TileAvailability],
    next_receive_address: &[Option<u32>],
    word_pressure: &[u64],
    rank: &[usize],
    epoch_width: usize,
    neighborhood: &[bool],
) -> RepairReady {
    let transfer = &pending[index];
    let earliest_start = std::iter::once(availability[usize::from(transfer.source)].send)
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
        .unwrap_or(0);
    let contiguous_receivers = transfer
        .destinations
        .iter()
        .filter(|(tile, address)| next_receive_address[usize::from(*tile)] == Some(*address))
        .count();
    let endpoint_pressure = transfer
        .tiles()
        .map(|tile| word_pressure[usize::from(tile)])
        .sum();
    RepairReady {
        window: Reverse(rank[index] / epoch_width),
        earliest_start: Reverse(earliest_start),
        epoch: Reverse(rank[index] / epoch_width),
        contiguous_receivers,
        in_neighborhood: neighborhood[index],
        endpoint_pressure,
        incumbent_rank: Reverse(rank[index]),
        index: Reverse(index),
    }
}

// Transfers with identical endpoint roles share availability and pressure.
// Keep one global heap entry per group, rather than refreshing every transfer
// whenever their common sender or receiver advances.
#[derive(Default)]
struct RepairGroup {
    ready: BTreeSet<RepairReady>,
    by_address: BTreeMap<(u16, u32), Vec<usize>>,
    revision: usize,
}

fn repair_rank(
    index: usize,
    rank: &[usize],
    epoch_width: usize,
    neighborhood: &[bool],
) -> RepairReady {
    RepairReady {
        window: Reverse(rank[index] / epoch_width),
        earliest_start: Reverse(0),
        epoch: Reverse(rank[index] / epoch_width),
        contiguous_receivers: 0,
        in_neighborhood: neighborhood[index],
        endpoint_pressure: 0,
        incumbent_rank: Reverse(rank[index]),
        index: Reverse(index),
    }
}

impl RepairGroup {
    fn best(
        &self,
        pending: &[PendingTransfer],
        availability: &[TileAvailability],
        next_receive_address: &[Option<u32>],
        word_pressure: &[u64],
        rank: &[usize],
        epoch_width: usize,
        neighborhood: &[bool],
        local: bool,
    ) -> Option<RepairReady> {
        let first = self.ready.last()?.index.0;
        let price = |index| {
            let mut candidate = repair_ready(
                index,
                pending,
                availability,
                next_receive_address,
                word_pressure,
                rank,
                epoch_width,
                neighborhood,
            );
            if !local {
                candidate.window = Reverse(0);
            }
            candidate
        };
        let mut best = price(first);
        // The static winner suffices unless another ready transfer continues
        // a receive stream. Index those exceptions by address instead of
        // rescanning all transfers in the group.
        for &(tile, _) in &pending[first].destinations {
            let Some(address) = next_receive_address[usize::from(tile)] else {
                continue;
            };
            if let Some(indices) = self.by_address.get(&(tile, address)) {
                for &index in indices {
                    let key = repair_rank(index, rank, epoch_width, neighborhood);
                    if key.epoch == best.epoch && self.ready.contains(&key) {
                        best = best.max(price(index));
                    }
                }
            }
        }
        Some(best)
    }
}

pub(super) fn order(
    problem: &SchedulingProblem<'_>,
    incumbent: &MaterializedSchedule,
    local: bool,
) -> Option<Vec<usize>> {
    let pending = problem.transfers;
    let tile_count = problem.tile_count;
    if pending.len() < 2 {
        return Some(incumbent.order.clone());
    }
    let mut critical = vec![false; pending.len()];
    let mut cursor = incumbent
        .timings
        .iter()
        .enumerate()
        .filter_map(|(index, timing)| timing.map(|timing| (index, timing.end)))
        .max_by_key(|entry| entry.1)
        .map(|entry| entry.0);
    while let Some(index) = cursor {
        if std::mem::replace(&mut critical[index], true) {
            break;
        }
        cursor = incumbent.timings[index].and_then(|timing| timing.predecessor);
    }

    let mut tile_orders = vec![Vec::new(); usize::from(tile_count)];
    for &index in &incumbent.order {
        for tile in pending[index].tiles() {
            tile_orders[usize::from(tile)].push(index);
        }
    }
    let mut neighborhood = critical.clone();
    for order in &tile_orders {
        if order.is_empty() {
            continue;
        }
        let radius = order.len().isqrt().max(1);
        for position in 0..order.len() {
            if !critical[order[position]] {
                continue;
            }
            let start = position.saturating_sub(radius);
            let end = (position + radius + 1).min(order.len());
            for &index in &order[start..end] {
                neighborhood[index] = true;
            }
        }
    }
    if neighborhood.iter().filter(|selected| **selected).count() < 2 {
        return Some(incumbent.order.clone());
    }

    let dependents = &problem.dependents;
    let mut indegrees = problem.indegrees();
    let mut rank = vec![0usize; pending.len()];
    for (position, &index) in incumbent.order.iter().enumerate() {
        rank[index] = position;
    }
    let epoch_width = pending.len().isqrt().max(1);
    let mut word_pressure = problem.word_pressure.clone();
    let mut availability = vec![TileAvailability::default(); usize::from(tile_count)];
    let mut next_receive_address = vec![None; usize::from(tile_count)];
    let mut group_ids = BTreeMap::new();
    let mut groups = Vec::<RepairGroup>::new();
    let mut transfer_groups = Vec::with_capacity(pending.len());
    for (index, transfer) in pending.iter().enumerate() {
        let mut receivers = transfer
            .destinations
            .iter()
            .map(|entry| entry.0)
            .collect::<Vec<_>>();
        receivers.sort_unstable();
        let next = groups.len();
        let group = *group_ids
            .entry((transfer.source, transfer.reserved_source, receivers))
            .or_insert(next);
        if group == next {
            groups.push(RepairGroup::default());
        }
        transfer_groups.push(group);
        for &destination in &transfer.destinations {
            groups[group]
                .by_address
                .entry(destination)
                .or_default()
                .push(index);
        }
        if indegrees[index] == 0 {
            groups[group]
                .ready
                .insert(repair_rank(index, &rank, epoch_width, &neighborhood));
        }
    }
    let mut ready = BinaryHeap::new();
    for (id, group) in groups.iter().enumerate() {
        if let Some(best) = group.best(
            pending,
            &availability,
            &next_receive_address,
            &word_pressure,
            &rank,
            epoch_width,
            &neighborhood,
            local,
        ) {
            ready.push((best, id, group.revision));
        }
    }
    let mut order = Vec::with_capacity(pending.len());
    // Bound queue work rather than rejecting phases by transfer count. A
    // broader search can recover cross-window opportunities in cheap cases.
    let budget = if local {
        usize::MAX
    } else {
        pending.len().saturating_mul(8).max(65_536)
    };
    let mut visits = 0usize;
    while let Some((candidate, group, revision)) = ready.pop() {
        visits += 1;
        if visits > budget {
            tracing::debug!(
                transfers = pending.len(),
                visits,
                "broader exchange repair reached its queue-work budget"
            );
            return None;
        }
        if revision != groups[group].revision {
            continue;
        }
        let Some(refreshed) = groups[group].best(
            pending,
            &availability,
            &next_receive_address,
            &word_pressure,
            &rank,
            epoch_width,
            &neighborhood,
            local,
        ) else {
            continue;
        };
        if refreshed != candidate {
            ready.push((refreshed, group, revision));
            continue;
        }
        let index = candidate.index.0;
        groups[group]
            .ready
            .remove(&repair_rank(index, &rank, epoch_width, &neighborhood));
        let transfer = &pending[index];
        let start = candidate.earliest_start.0;
        let items = transfer.item_count().unwrap_or(transfer.words);
        let end = match start.checked_add(items) {
            Some(end) => end,
            None => return Some(incumbent.order.clone()),
        };
        availability[usize::from(transfer.source)].send = end;
        if let Some(tile) = transfer.reserved_source {
            availability[usize::from(tile)].send = end;
        }
        let bytes = match transfer.words.checked_mul(4) {
            Some(bytes) => bytes,
            None => return Some(incumbent.order.clone()),
        };
        for &(tile, address) in &transfer.destinations {
            availability[usize::from(tile)].receive = end;
            next_receive_address[usize::from(tile)] = address.checked_add(bytes);
        }
        for tile in transfer.tiles() {
            word_pressure[usize::from(tile)] =
                word_pressure[usize::from(tile)].saturating_sub(u64::from(items));
        }
        order.push(index);
        let mut changed = vec![group];
        for &dependent in &dependents[index] {
            indegrees[dependent] -= 1;
            if indegrees[dependent] == 0 {
                let id = transfer_groups[dependent];
                groups[id]
                    .ready
                    .insert(repair_rank(dependent, &rank, epoch_width, &neighborhood));
                changed.push(id);
            }
        }
        changed.sort_unstable();
        changed.dedup();
        for id in changed {
            groups[id].revision += 1;
            if let Some(best) = groups[id].best(
                pending,
                &availability,
                &next_receive_address,
                &word_pressure,
                &rank,
                epoch_width,
                &neighborhood,
                local,
            ) {
                ready.push((best, id, groups[id].revision));
            }
        }
    }
    tracing::debug!(
        transfers = pending.len(),
        visits,
        local,
        "finished exchange repair ordering"
    );
    Some(if order.len() == pending.len() {
        order
    } else {
        incumbent.order.clone()
    })
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;

    fn transfers(count: usize) -> Vec<PendingTransfer> {
        (0..count)
            .map(|index| PendingTransfer {
                source: 0,
                reserved_source: None,
                source_shard: BlockValueId::from_index(0),
                source_offset: 0,
                source_addresses: vec![0x40000],
                source_elements: effective_memory_elements(0x40000, 1),
                destinations: vec![(1, 0x80000 + index as u32 * 4)],
                words: 1,
                width: ExchangeItemWidth::Word32,
            })
            .collect()
    }

    #[test]
    fn grouped_repair_priority_matches_exhaustive_ready_selection() {
        let mut random = fastrand::Rng::with_seed(0x6772_6f75_705f_7265);
        let mut pending = transfers(256);
        for (index, transfer) in pending.iter_mut().enumerate() {
            transfer.reserved_source = Some(3);
            transfer
                .destinations
                .push((2, 0x90000 + (255 - index as u32) * 8));
        }
        let rank = (0..pending.len()).collect::<Vec<_>>();
        for _ in 0..128 {
            let neighborhood = (0..pending.len())
                .map(|_| random.bool())
                .collect::<Vec<_>>();
            let mut group = RepairGroup::default();
            for (index, transfer) in pending.iter().enumerate() {
                for &address in &transfer.destinations {
                    group.by_address.entry(address).or_default().push(index);
                }
                if random.bool() {
                    group
                        .ready
                        .insert(repair_rank(index, &rank, 16, &neighborhood));
                }
            }
            let next = [
                None,
                Some(0x80000 + random.u32(0..256) * 4),
                Some(0x90000 + random.u32(0..256) * 8),
                None,
            ];
            let availability = vec![TileAvailability::default(); 4];
            let pressure = [13, 47, 31, 19];
            let expected = group
                .ready
                .iter()
                .map(|key| {
                    repair_ready(
                        key.index.0,
                        &pending,
                        &availability,
                        &next,
                        &pressure,
                        &rank,
                        16,
                        &neighborhood,
                    )
                })
                .max();
            assert_eq!(
                group.best(
                    &pending,
                    &availability,
                    &next,
                    &pressure,
                    &rank,
                    16,
                    &neighborhood,
                    true,
                ),
                expected
            );
        }
    }

    #[test]
    #[ignore = "manual repair-queue scaling benchmark"]
    fn repair_queue_scaling() {
        for count in [4096, 16384, 65536] {
            let pending = transfers(count);
            let mut incumbent = MaterializedSchedule::new(2, &pending);
            incumbent.order = (0..count).collect();
            for (index, timing) in incumbent.timings.iter_mut().enumerate() {
                *timing = Some(MaterializedTiming {
                    end: index as u32 + 1,
                    predecessor: index.checked_sub(1),
                });
            }
            let start = std::time::Instant::now();
            let result = order(&SchedulingProblem::new(&pending, 2), &incumbent, true).unwrap();
            eprintln!(
                "repair_queue transfers={count} elapsed={:?}",
                start.elapsed()
            );
            assert_eq!(result, incumbent.order);
        }
    }
}
