//! Maximum-cardinality waves for balanced point-to-point traffic.
//! Input is the shared dependency problem plus an incumbent tie-breaking order;
//! output is a transfer permutation, never encoded rows or new timing rules.
use super::{PendingTransfer, SchedulingProblem};
#[cfg(test)]
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
pub(super) fn order(
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
