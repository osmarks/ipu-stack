//! Address-ordered streams in bounded payload waves, optionally balanced across
//! endpoints. Dependency depth keeps the returned permutation topological;
//! common materialization owns exact timing, hazards, and encoding retries.
use super::{
    ExchangeLoweringError, MaterializedSchedule, SchedulingProblem,
    materialize_valid_schedule_order,
};
use ipu_target::ipu21::fabric::Topology;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
/// Compact scheduling: visit address-ordered endpoint streams in bounded
/// payload waves. This amortizes receive-pointer/mux changes without imposing
/// barriers; the ordinary row builder still overlaps independent transfers.
pub(super) fn order(problem: &SchedulingProblem<'_>, words: u32, balanced: bool) -> Vec<usize> {
    let mut streams = BTreeMap::<_, Vec<usize>>::new();
    for (index, transfer) in problem.transfers.iter().enumerate() {
        streams
            .entry((
                !transfer.moving_source(),
                transfer.source,
                transfer.reserved_source,
                transfer
                    .destinations
                    .iter()
                    .map(|&(tile, _)| tile)
                    .collect::<Vec<_>>(),
            ))
            .or_default()
            .push(index);
    }
    let mut rank = vec![(0u64, Reverse(0u64), 0usize, 0usize); problem.transfers.len()];
    for (stream, indices) in streams.values_mut().enumerate() {
        indices.sort_unstable_by_key(|&index| {
            let transfer = &problem.transfers[index];
            (transfer.destinations[0].1, transfer.source_address(), index)
        });
        let total = indices
            .iter()
            .map(|&index| u64::from(problem.transfers[index].words))
            .sum::<u64>();
        let mut offset = 0u64;
        for (position, &index) in indices.iter().enumerate() {
            rank[index] = (offset / u64::from(words), Reverse(total), stream, position);
            offset += u64::from(problem.transfers[index].words);
        }
    }
    if balanced {
        balance_stream_chunks(problem, &mut rank);
    }
    // A ready forwarding transfer can still depend on a late payload. Giving
    // it an early stream rank reserves receiver rows far into the future and
    // delays input transfers for other forwarders. Finish each dependency
    // depth in the ordering first; the row builder still overlaps their actual
    // event times. Dependencies always point forward in original input order.
    let mut depth = vec![0usize; problem.transfers.len()];
    for (index, predecessors) in problem.predecessors.iter().enumerate() {
        depth[index] = predecessors
            .iter()
            .map(|&before| depth[before] + 1)
            .max()
            .unwrap_or(0);
    }
    // Depth strictly increases along every dependency edge, so this total
    // order is already topological; no separate ready queue is needed.
    let mut order = (0..rank.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|&index| {
        (
            depth[index],
            !problem.transfers[index].moving_source(),
            rank[index],
            index,
        )
    });
    order
}

/// Reorder whole chunks within a wave, preserving their address order. Start
/// independent endpoints early instead of walking sources in tile-number order.
/// This is only a ranking heuristic: exact timing and hazards remain in append.
fn balance_stream_chunks(
    problem: &SchedulingProblem<'_>,
    rank: &mut [(u64, Reverse<u64>, usize, usize)],
) {
    let mut chunks = BTreeMap::<_, Vec<usize>>::new();
    for (index, &(wave, _, stream, _)) in rank.iter().enumerate() {
        chunks.entry((wave, stream)).or_default().push(index);
    }
    // Even slots are sending endpoints, odd slots receiving endpoints. A tile
    // can send and receive together; its paired sender occupies a second TX.
    let mut loads = vec![(0u64, 0u64); 2 * usize::from(problem.tile_count)];
    let mut waves = BTreeMap::<_, Vec<_>>::new();
    for ((wave, stream), mut indices) in chunks {
        indices.sort_unstable_by_key(|&index| rank[index].3);
        let endpoints = problem.transfers[indices[0]]
            .pressure_resources(true)
            .collect::<Vec<_>>();
        let work = indices
            .iter()
            .map(|&index| {
                let transfer = &problem.transfers[index];
                u64::from(transfer.item_count().unwrap_or(transfer.words))
            })
            .sum::<u64>();
        for &endpoint in &endpoints {
            loads[endpoint].1 += work;
        }
        waves
            .entry(wave)
            .or_default()
            .push((stream, indices, endpoints, work));
    }
    for chunks in waves.into_values() {
        let score = |id: usize, loads: &[(u64, u64)]| {
            let (stream, _, endpoints, _) = &chunks[id];
            let ready = endpoints
                .iter()
                .map(|&endpoint| loads[endpoint].0)
                .max()
                .unwrap_or(0);
            let remaining = endpoints
                .iter()
                .map(|&endpoint| loads[endpoint].1)
                .max()
                .unwrap_or(0);
            (ready, Reverse(remaining), *stream)
        };
        let mut ready = (0..chunks.len())
            .map(|id| Reverse((score(id, &loads), id)))
            .collect::<BinaryHeap<_>>();
        let mut position = 0;
        while let Some(Reverse((old, id))) = ready.pop() {
            let current = score(id, &loads);
            // Availability only increases and remaining work only decreases.
            if current != old {
                ready.push(Reverse((current, id)));
                continue;
            }
            let (_, indices, endpoints, work) = &chunks[id];
            for &index in indices {
                rank[index].1 = Reverse(0);
                rank[index].2 = position;
            }
            position += 1;
            for &endpoint in endpoints {
                loads[endpoint].0 = current.0 + work;
                loads[endpoint].1 -= work;
            }
        }
    }
}

pub(super) fn schedule(
    topology: &Topology,
    problem: &SchedulingProblem<'_>,
    incoming_bases: &[u32],
    receive_counts: &[usize],
    words: u32,
    balanced: bool,
) -> Result<MaterializedSchedule, ExchangeLoweringError> {
    let order = order(problem, words, balanced);
    materialize_valid_schedule_order(topology, problem, incoming_bases, receive_counts, &order)
}
