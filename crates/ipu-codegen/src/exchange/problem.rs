use super::PendingTransfer;
use std::collections::BTreeSet;

/// Address/width-dependent facts shared by every trial for one physical phase.
pub(super) struct SchedulingProblem<'a> {
    pub(super) transfers: &'a [PendingTransfer],
    pub(super) tile_count: u16,
    pub(super) predecessors: Vec<Vec<usize>>,
    pub(super) dependents: Vec<Vec<usize>>,
    pub(super) word_pressure: Vec<u64>,
}

impl<'a> SchedulingProblem<'a> {
    pub(super) fn new(transfers: &'a [PendingTransfer], tile_count: u16) -> Self {
        let mut predecessors = vec![Vec::new(); transfers.len()];
        let mut dependents = vec![Vec::new(); transfers.len()];
        for (before, after) in memory_dependencies(transfers, tile_count) {
            predecessors[after].push(before);
            dependents[before].push(after);
        }
        let mut word_pressure = vec![0; usize::from(tile_count)];
        for transfer in transfers {
            let words = u64::from(transfer.item_count().unwrap_or(transfer.words));
            for tile in transfer.tiles() {
                word_pressure[usize::from(tile)] += words;
            }
        }
        Self {
            transfers,
            tile_count,
            predecessors,
            dependents,
            word_pressure,
        }
    }
    pub(super) fn indegrees(&self) -> Vec<usize> {
        self.predecessors.iter().map(Vec::len).collect()
    }
}

#[derive(Clone, Copy)]
struct TransferAccess {
    transfer: usize,
    start: u64,
    end: u64,
    write: bool,
}

#[derive(Clone, Default)]
struct AccessFrontier {
    last_write: Option<usize>,
    reads: Vec<usize>,
}

/// Preserves the original order of overlapping accesses while allowing the
/// scheduler to reorder transfers whose tile-memory effects are independent.
pub(super) fn memory_dependencies(
    transfers: &[PendingTransfer],
    tile_count: u16,
) -> BTreeSet<(usize, usize)> {
    let mut accesses = vec![Vec::new(); usize::from(tile_count)];
    for (index, transfer) in transfers.iter().enumerate() {
        let bytes = u64::from(transfer.words) * 4;
        for &address in &transfer.source_addresses {
            accesses[usize::from(transfer.source)].push(TransferAccess {
                transfer: index,
                start: u64::from(address),
                end: u64::from(address) + bytes,
                write: false,
            });
        }
        for &(tile, address) in &transfer.destinations {
            accesses[usize::from(tile)].push(TransferAccess {
                transfer: index,
                start: u64::from(address),
                end: u64::from(address) + bytes,
                write: true,
            });
        }
    }

    let mut dependencies = BTreeSet::new();
    for tile_accesses in &accesses {
        let mut boundaries = tile_accesses
            .iter()
            .flat_map(|access| [access.start, access.end])
            .collect::<Vec<_>>();
        boundaries.sort_unstable();
        boundaries.dedup();
        let mut frontier = vec![AccessFrontier::default(); boundaries.len().saturating_sub(1)];
        for access in tile_accesses {
            let start = boundaries
                .binary_search(&access.start)
                .expect("access start is a boundary");
            let end = boundaries
                .binary_search(&access.end)
                .expect("access end is a boundary");
            for state in &mut frontier[start..end] {
                if let Some(previous) = state.last_write
                    && previous != access.transfer
                {
                    dependencies.insert((previous, access.transfer));
                }
                if access.write {
                    for &previous in &state.reads {
                        if previous != access.transfer {
                            dependencies.insert((previous, access.transfer));
                        }
                    }
                    state.reads.clear();
                    state.last_write = Some(access.transfer);
                } else if state.reads.last() != Some(&access.transfer) {
                    state.reads.push(access.transfer);
                }
            }
        }
    }
    dependencies
}
