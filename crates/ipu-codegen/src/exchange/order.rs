//! Alternative dependency-respecting transfer orders, evaluated by the same scheduler.
use super::*;

pub(super) fn point_to_point_matching_wave_order(
    problem: &SchedulingProblem<'_>,
    incumbent_order: &[usize],
) -> Option<Vec<usize>> {
    let pending = problem.transfers;
    let tile_count = problem.tile_count;
    if pending.len() < 2
        || pending
            .iter()
            .any(|transfer| transfer.destinations.len() != 1)
    {
        return None;
    }

    let tile_count = usize::from(tile_count);
    let mut source_roles = vec![0usize; tile_count];
    let mut destination_roles = vec![0usize; tile_count];
    for transfer in pending {
        source_roles[usize::from(transfer.source)] += 1;
        destination_roles[usize::from(transfer.destinations[0].0)] += 1;
    }
    let active_sources = source_roles.iter().filter(|&&roles| roles != 0).count();
    let active_destinations = destination_roles
        .iter()
        .filter(|&&roles| roles != 0)
        .count();
    let maximum_roles = source_roles
        .iter()
        .chain(&destination_roles)
        .copied()
        .max()
        .unwrap_or(0);
    let mean_roles = pending
        .len()
        .div_ceil(active_sources.min(active_destinations).max(1));
    // Cardinality waves are a useful approximation only while endpoint
    // degrees are reasonably balanced. A highly skewed phase is governed by
    // unequal release times and keeps the ordinary exact list schedule.
    if maximum_roles > mean_roles.saturating_mul(2) {
        return None;
    }

    let mut incumbent_rank = vec![usize::MAX; pending.len()];
    for (rank, &index) in incumbent_order.iter().enumerate() {
        incumbent_rank[index] = rank;
    }
    let dependents = &problem.dependents;
    let mut indegrees = problem.indegrees();
    // Parallel edges to one destination cannot change a cardinality matching.
    // Keep their incumbent order once, and expose only the first ready edge.
    // Dependency releases and completed waves update this index incrementally.
    let mut ready = vec![BTreeMap::<u16, BTreeSet<(usize, usize)>>::new(); tile_count];
    for (index, transfer) in pending.iter().enumerate() {
        transfer.item_count().ok()?;
        if indegrees[index] == 0 {
            ready[usize::from(transfer.source)]
                .entry(transfer.destinations[0].0)
                .or_default()
                .insert((incumbent_rank[index], index));
        }
    }

    let mut order = Vec::with_capacity(pending.len());
    while order.len() != pending.len() {
        let adjacency = ready
            .iter()
            .map(|destinations| {
                let mut edges = destinations
                    .values()
                    .filter_map(|edges| edges.first().map(|&(_, index)| index))
                    .collect::<Vec<_>>();
                edges.sort_unstable_by_key(|&index| incumbent_rank[index]);
                edges
            })
            .collect::<Vec<_>>();
        let mut source_order = (0..tile_count)
            .filter(|&source| !adjacency[source].is_empty())
            .collect::<Vec<_>>();
        source_order.sort_unstable_by_key(|&source| incumbent_rank[adjacency[source][0]]);
        let mut wave = maximum_ready_matching(pending, &adjacency, &source_order, tile_count);
        if wave.is_empty() {
            return None;
        }
        wave.sort_unstable_by_key(|&index| incumbent_rank[index]);
        for index in wave {
            let transfer = &pending[index];
            let source = usize::from(transfer.source);
            let destination = transfer.destinations[0].0;
            let edges = ready[source].get_mut(&destination).unwrap();
            edges.remove(&(incumbent_rank[index], index));
            if edges.is_empty() {
                ready[source].remove(&destination);
            }
            order.push(index);
            for &dependent in &dependents[index] {
                indegrees[dependent] -= 1;
                if indegrees[dependent] == 0 {
                    let transfer = &pending[dependent];
                    ready[usize::from(transfer.source)]
                        .entry(transfer.destinations[0].0)
                        .or_default()
                        .insert((incumbent_rank[dependent], dependent));
                }
            }
        }
    }
    Some(order)
}

pub(super) fn maximum_ready_matching(
    pending: &[PendingTransfer],
    adjacency: &[Vec<usize>],
    source_order: &[usize],
    tile_count: usize,
) -> Vec<usize> {
    let mut source_edges = vec![None; tile_count];
    let mut destination_sources = vec![None; tile_count];
    let mut distances = vec![usize::MAX; tile_count];
    loop {
        let mut queue = VecDeque::new();
        for &source in source_order {
            if source_edges[source].is_none() {
                distances[source] = 0;
                queue.push_back(source);
            } else {
                distances[source] = usize::MAX;
            }
        }
        let mut reaches_free_destination = false;
        while let Some(source) = queue.pop_front() {
            for &index in &adjacency[source] {
                let destination = usize::from(pending[index].destinations[0].0);
                if let Some(next_source) = destination_sources[destination] {
                    if distances[next_source] == usize::MAX {
                        distances[next_source] = distances[source] + 1;
                        queue.push_back(next_source);
                    }
                } else {
                    reaches_free_destination = true;
                }
            }
        }
        if !reaches_free_destination {
            break;
        }
        let mut augmented = false;
        for &source in source_order {
            if source_edges[source].is_none()
                && augment_ready_matching(
                    source,
                    pending,
                    adjacency,
                    &mut source_edges,
                    &mut destination_sources,
                    &mut distances,
                )
            {
                augmented = true;
            }
        }
        if !augmented {
            break;
        }
    }
    source_edges.into_iter().flatten().collect()
}

fn augment_ready_matching(
    source: usize,
    pending: &[PendingTransfer],
    adjacency: &[Vec<usize>],
    source_edges: &mut [Option<usize>],
    destination_sources: &mut [Option<usize>],
    distances: &mut [usize],
) -> bool {
    for &index in &adjacency[source] {
        let destination = usize::from(pending[index].destinations[0].0);
        let paired_source = destination_sources[destination];
        if paired_source.is_none_or(|paired_source| {
            distances[paired_source] == distances[source] + 1
                && augment_ready_matching(
                    paired_source,
                    pending,
                    adjacency,
                    source_edges,
                    destination_sources,
                    distances,
                )
        }) {
            source_edges[source] = Some(index);
            destination_sources[destination] = Some(source);
            return true;
        }
    }
    distances[source] = usize::MAX;
    false
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RepairReady {
    // Repair a bounded window of the incumbent before considering later work.
    // With time first, every endpoint advance repriced candidates throughout
    // the phase, turning a local improvement into another global search.
    epoch: Reverse<usize>,
    earliest_start: Reverse<u32>,
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
    ) -> Option<RepairReady> {
        let first = self.ready.last()?.index.0;
        let price = |index| {
            repair_ready(
                index,
                pending,
                availability,
                next_receive_address,
                word_pressure,
                rank,
                epoch_width,
                neighborhood,
            )
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

pub(super) fn critical_neighborhood_order(
    problem: &SchedulingProblem<'_>,
    incumbent: &MaterializedSchedule,
) -> Vec<usize> {
    let pending = problem.transfers;
    let tile_count = problem.tile_count;
    if pending.len() < 2 {
        return incumbent.order.clone();
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
        return incumbent.order.clone();
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
        ) {
            ready.push((best, id, group.revision));
        }
    }
    let mut order = Vec::with_capacity(pending.len());
    while let Some((candidate, group, revision)) = ready.pop() {
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
            None => return incumbent.order.clone(),
        };
        availability[usize::from(transfer.source)].send = end;
        if let Some(tile) = transfer.reserved_source {
            availability[usize::from(tile)].send = end;
        }
        let bytes = match transfer.words.checked_mul(4) {
            Some(bytes) => bytes,
            None => return incumbent.order.clone(),
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
            ) {
                ready.push((best, id, groups[id].revision));
            }
        }
    }
    if order.len() == pending.len() {
        order
    } else {
        incumbent.order.clone()
    }
}

#[cfg(test)]
pub(super) fn reference_matching_wave_order(
    problem: &SchedulingProblem<'_>,
    incumbent_order: &[usize],
) -> Option<Vec<usize>> {
    let pending = problem.transfers;
    let tile_count = problem.tile_count;
    if pending.len() < 2
        || pending
            .iter()
            .any(|transfer| transfer.destinations.len() != 1)
    {
        return None;
    }

    let tile_count = usize::from(tile_count);
    let mut source_roles = vec![0usize; tile_count];
    let mut destination_roles = vec![0usize; tile_count];
    for transfer in pending {
        source_roles[usize::from(transfer.source)] += 1;
        destination_roles[usize::from(transfer.destinations[0].0)] += 1;
    }
    let active_sources = source_roles.iter().filter(|&&roles| roles != 0).count();
    let active_destinations = destination_roles
        .iter()
        .filter(|&&roles| roles != 0)
        .count();
    let maximum_roles = source_roles
        .iter()
        .chain(&destination_roles)
        .copied()
        .max()
        .unwrap_or(0);
    let mean_roles = pending
        .len()
        .div_ceil(active_sources.min(active_destinations).max(1));
    // Cardinality waves are a useful approximation only while endpoint
    // degrees are reasonably balanced. A highly skewed phase is governed by
    // unequal release times and keeps the ordinary exact list schedule.
    if maximum_roles > mean_roles.saturating_mul(2) {
        return None;
    }

    let mut incumbent_rank = vec![usize::MAX; pending.len()];
    for (rank, &index) in incumbent_order.iter().enumerate() {
        incumbent_rank[index] = rank;
    }
    let dependents = &problem.dependents;
    let mut indegrees = problem.indegrees();
    let mut remaining_send_words = vec![0u64; tile_count];
    let mut remaining_receive_words = vec![0u64; tile_count];
    for transfer in pending {
        let words = u64::from(transfer.item_count().ok()?);
        remaining_send_words[usize::from(transfer.source)] += words;
        remaining_receive_words[usize::from(transfer.destinations[0].0)] += words;
    }

    let mut scheduled = vec![false; pending.len()];
    let mut order = Vec::with_capacity(pending.len());
    while order.len() != pending.len() {
        let mut adjacency = vec![Vec::new(); tile_count];
        for index in 0..pending.len() {
            if !scheduled[index] && indegrees[index] == 0 {
                adjacency[usize::from(pending[index].source)].push(index);
            }
        }
        for edges in &mut adjacency {
            edges.sort_unstable_by_key(|&index| {
                let transfer = &pending[index];
                let destination = usize::from(transfer.destinations[0].0);
                (
                    incumbent_rank[index],
                    Reverse(remaining_receive_words[destination]),
                    Reverse(transfer.item_count().unwrap_or(transfer.words)),
                    index,
                )
            });
        }
        let mut source_order = (0..tile_count)
            .filter(|&source| !adjacency[source].is_empty())
            .collect::<Vec<_>>();
        source_order.sort_unstable_by_key(|&source| {
            (
                adjacency[source]
                    .iter()
                    .map(|&index| incumbent_rank[index])
                    .min()
                    .unwrap_or(usize::MAX),
                Reverse(remaining_send_words[source]),
                source,
            )
        });
        let mut wave = maximum_ready_matching(pending, &adjacency, &source_order, tile_count);
        if wave.is_empty() {
            return None;
        }
        wave.sort_unstable_by_key(|&index| {
            let transfer = &pending[index];
            (
                incumbent_rank[index],
                Reverse(transfer.item_count().unwrap_or(transfer.words)),
                transfer.source,
                transfer.destinations[0].0,
                index,
            )
        });
        for index in wave {
            let transfer = &pending[index];
            let words = u64::from(transfer.item_count().ok()?);
            scheduled[index] = true;
            order.push(index);
            let source = usize::from(transfer.source);
            remaining_send_words[source] = remaining_send_words[source].saturating_sub(words);
            let destination = usize::from(transfer.destinations[0].0);
            remaining_receive_words[destination] =
                remaining_receive_words[destination].saturating_sub(words);
            for &dependent in &dependents[index] {
                indegrees[dependent] -= 1;
            }
        }
    }
    Some(order)
}

#[cfg(test)]
mod tests {
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
                    &neighborhood
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
                    start: index as u32,
                    end: index as u32 + 1,
                    blocking_tile: 0,
                    predecessor: index.checked_sub(1),
                });
            }
            let start = std::time::Instant::now();
            let result =
                critical_neighborhood_order(&SchedulingProblem::new(&pending, 2), &incumbent);
            eprintln!(
                "repair_queue transfers={count} elapsed={:?}",
                start.elapsed()
            );
            assert_eq!(result, incumbent.order);
        }
    }
}
