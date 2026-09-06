//! Alternative dependency-respecting transfer orders, evaluated by the same scheduler.
use super::*;

pub(super) fn point_to_point_matching_wave_order(
    pending: &[PendingTransfer],
    tile_count: u16,
    incumbent_order: &[usize],
) -> Option<Vec<usize>> {
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
    let mut dependents = vec![Vec::new(); pending.len()];
    let mut indegrees = vec![0usize; pending.len()];
    for (before, after) in memory_dependencies(pending, tile_count as u16) {
        dependents[before].push(after);
        indegrees[after] += 1;
    }
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
            for dependent in std::mem::take(&mut dependents[index]) {
                indegrees[dependent] -= 1;
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
        earliest_start: Reverse(earliest_start),
        epoch: Reverse(rank[index] / epoch_width),
        contiguous_receivers,
        in_neighborhood: neighborhood[index],
        endpoint_pressure,
        incumbent_rank: Reverse(rank[index]),
        index: Reverse(index),
    }
}

pub(super) fn critical_neighborhood_order(
    pending: &[PendingTransfer],
    tile_count: u16,
    incumbent: &MaterializedSchedule,
) -> Vec<usize> {
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

    let mut dependents = vec![Vec::new(); pending.len()];
    let mut indegrees = vec![0usize; pending.len()];
    for (before, after) in memory_dependencies(pending, tile_count) {
        dependents[before].push(after);
        indegrees[after] += 1;
    }
    let mut rank = vec![0usize; pending.len()];
    for (position, &index) in incumbent.order.iter().enumerate() {
        rank[index] = position;
    }
    let epoch_width = pending.len().isqrt().max(1);
    let mut word_pressure = vec![0u64; usize::from(tile_count)];
    for transfer in pending {
        let items = u64::from(transfer.item_count().unwrap_or(transfer.words));
        for tile in transfer.tiles() {
            word_pressure[usize::from(tile)] += items;
        }
    }
    let mut availability = vec![TileAvailability::default(); usize::from(tile_count)];
    let mut next_receive_address = vec![None; usize::from(tile_count)];
    let mut ready = BinaryHeap::new();
    for (index, &indegree) in indegrees.iter().enumerate() {
        if indegree == 0 {
            ready.push(repair_ready(
                index,
                pending,
                &availability,
                &next_receive_address,
                &word_pressure,
                &rank,
                epoch_width,
                &neighborhood,
            ));
        }
    }
    let mut order = Vec::with_capacity(pending.len());
    while let Some(candidate) = ready.pop() {
        let index = candidate.index.0;
        let refreshed = repair_ready(
            index,
            pending,
            &availability,
            &next_receive_address,
            &word_pressure,
            &rank,
            epoch_width,
            &neighborhood,
        );
        if refreshed != candidate {
            ready.push(refreshed);
            continue;
        }
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
        for dependent in std::mem::take(&mut dependents[index]) {
            indegrees[dependent] -= 1;
            if indegrees[dependent] == 0 {
                ready.push(repair_ready(
                    dependent,
                    pending,
                    &availability,
                    &next_receive_address,
                    &word_pressure,
                    &rank,
                    epoch_width,
                    &neighborhood,
                ));
            }
        }
    }
    if order.len() == pending.len() {
        order
    } else {
        incumbent.order.clone()
    }
}
