//! Order-sensitive row-storage estimate, without placement or scheduling.
//!
//! Count sender setup/send words, receive source/neutral controls and pointer
//! changes independently. Delays, bidi fusion and cross-phase row sharing remain
//! scheduler-dependent, so this is a ranking estimate, never a feasibility bound.

pub(crate) struct ExchangeStoragePhase {
    bytes: Vec<u64>,
    receive_end: Vec<Option<u64>>,
}

impl ExchangeStoragePhase {
    pub(crate) fn new(tiles: u16) -> Self {
        Self {
            bytes: vec![8; usize::from(tiles)], // entry and return
            receive_end: vec![None; usize::from(tiles)],
        }
    }

    pub(crate) fn send(&mut self, tile: u16, fragments: u64, long_fragments: u64) {
        self.bytes[usize::from(tile)] += fragments * 8 + long_fragments * 4;
    }

    // Before placement, the upper 32 bits identify the allocation; the lower
    // bits are its byte offset. Captures use absolute addresses instead.
    pub(crate) fn receive(&mut self, tile: u16, address: u64, bytes: u64) {
        let index = usize::from(tile);
        self.bytes[index] += 8 + 4 * u64::from(self.receive_end[index] != Some(address));
        self.receive_end[index] = Some(address + bytes);
    }

    pub(crate) fn finish(self) -> Vec<u64> {
        self.bytes
            .into_iter()
            .map(|bytes| bytes.next_multiple_of(8))
            .collect()
    }
}

/// Estimate each tile's row bytes for a captured, unscheduled phase. Uses the
/// same model as the planner; timings and encoded instructions are not inputs.
pub fn estimate_exchange_phase_storage(
    tile_count: u16,
    problem: &crate::exchange::ExchangeScheduleProblem,
) -> Vec<u64> {
    let mut phase = ExchangeStoragePhase::new(tile_count);
    for transfer in &problem.transfers {
        phase.send(transfer.source, 1, u64::from(transfer.words > 64));
        for destination in &transfer.destinations {
            phase.receive(
                destination.tile,
                u64::from(destination.address),
                u64::from(transfer.words) * 4,
            );
        }
    }
    phase.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_receives_share_pointer_setup_but_separate_allocations_do_not() {
        let mut contiguous = ExchangeStoragePhase::new(1);
        contiguous.receive(0, 0, 32);
        contiguous.receive(0, 32, 32);
        let mut separate = ExchangeStoragePhase::new(1);
        separate.receive(0, 0, 32);
        separate.receive(0, 1 << 32, 32);
        assert_eq!(contiguous.bytes[0] + 4, separate.bytes[0]);
    }
}
