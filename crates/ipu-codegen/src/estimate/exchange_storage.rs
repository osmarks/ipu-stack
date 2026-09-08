//! Order-sensitive row-storage estimate, without placement or scheduling.
//!
//! Count sender setup/send words, receive source/neutral controls and pointer
//! changes independently. Delays, bidi fusion and final row compatibility remain
//! scheduler-dependent, so this is a ranking estimate, never a feasibility bound.

use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
};

pub(crate) struct ExchangeStoragePhase {
    bytes: Vec<u64>,
    receive_end: Vec<Option<u64>>,
    signatures: Vec<DefaultHasher>,
    patches: Vec<u64>,
    shareable: Vec<bool>,
}

impl ExchangeStoragePhase {
    pub(crate) fn new(tiles: u16) -> Self {
        Self {
            bytes: vec![8; usize::from(tiles)], // entry and return
            receive_end: vec![None; usize::from(tiles)],
            signatures: vec![DefaultHasher::new(); usize::from(tiles)],
            patches: vec![0; usize::from(tiles)],
            shareable: vec![true; usize::from(tiles)],
        }
    }

    pub(crate) fn send(&mut self, tile: u16, fragments: u64, long_fragments: u64) {
        self.bytes[usize::from(tile)] += fragments * 8 + long_fragments * 4;
        self.patches[usize::from(tile)] += fragments;
    }

    // Before placement, the upper 32 bits identify the allocation; the lower
    // bits are its byte offset. Captures use absolute addresses instead.
    pub(crate) fn receive(&mut self, tile: u16, address: u64, bytes: u64) {
        let index = usize::from(tile);
        let reset = self.receive_end[index] != Some(address);
        self.bytes[index] += 8 + 4 * u64::from(reset);
        self.patches[index] += u64::from(reset);
        (2u8, reset).hash(&mut self.signatures[index]);
        self.receive_end[index] = Some(address + bytes);
    }

    pub(crate) fn connection(
        &mut self,
        source: u16,
        target: u16,
        bytes: u64,
        paired: bool,
        receivers: usize,
    ) {
        // Ordered routes and lengths determine structure; absolute addresses do
        // not. Receiver pointer-continuation decisions are hashed separately.
        (0u8, target, bytes, paired, receivers).hash(&mut self.signatures[usize::from(source)]);
        (1u8, source, bytes, paired, receivers).hash(&mut self.signatures[usize::from(target)]);
    }

    pub(crate) fn disable_sharing(&mut self, tile: u16) {
        self.shareable[usize::from(tile)] = false;
    }

    pub(crate) fn finish(self) -> Vec<u64> {
        self.bytes
            .into_iter()
            .map(|bytes| bytes.next_multiple_of(8))
            .collect()
    }
}

/// Accumulate estimated table storage across static phases, sharing compatible
/// address-independent transfer signatures. Hash matches predict sharing; only
/// final encoded rows establish that sharing is actually possible.
pub struct ExchangeStorageEstimator {
    bytes: Vec<u64>,
    rows: Vec<HashMap<(u64, u64, u64), bool>>,
}

impl ExchangeStorageEstimator {
    pub fn new(tiles: u16) -> Self {
        Self {
            bytes: vec![0; usize::from(tiles)],
            rows: vec![HashMap::new(); usize::from(tiles)],
        }
    }

    pub fn maximum_bytes(&self) -> u64 {
        self.bytes.iter().copied().max().unwrap_or(0)
    }

    /// Add a phase from a validated snapshot and return its unshared row sizes.
    pub fn add_phase(&mut self, problem: &crate::exchange::ExchangeScheduleProblem) -> Vec<u64> {
        let phase = captured_phase(self.bytes.len() as u16, problem);
        let sizes = phase
            .bytes
            .iter()
            .map(|bytes| bytes.next_multiple_of(8))
            .collect();
        self.add(phase);
        sizes
    }

    pub(crate) fn add(&mut self, phase: ExchangeStoragePhase) {
        for tile in 0..self.bytes.len() {
            let bytes = phase.bytes[tile].next_multiple_of(8);
            if !phase.shareable[tile] {
                self.bytes[tile] += bytes;
                continue;
            }
            let patches = phase.patches[tile];
            let key = (phase.signatures[tile].finish(), bytes, patches);
            let extra = match self.rows[tile].entry(key) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(false);
                    bytes
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    // On the second use allocate shared patch offsets and
                    // values for BOTH invocations. Later uses need only values.
                    let arrays = if *entry.get() { 1 } else { 3 };
                    entry.insert(true);
                    (patches * 4 * arrays).next_multiple_of(8)
                }
            };
            self.bytes[tile] += extra;
        }
    }
}

/// Estimate each tile's row bytes for a captured, unscheduled phase. Uses the
/// same model as the planner; timings and encoded instructions are not inputs.
pub fn estimate_exchange_phase_storage(
    tile_count: u16,
    problem: &crate::exchange::ExchangeScheduleProblem,
) -> Vec<u64> {
    captured_phase(tile_count, problem).finish()
}

fn captured_phase(
    tile_count: u16,
    problem: &crate::exchange::ExchangeScheduleProblem,
) -> ExchangeStoragePhase {
    let mut phase = ExchangeStoragePhase::new(tile_count);
    for transfer in &problem.transfers {
        phase.send(transfer.source, 1, u64::from(transfer.words > 64));
        if transfer.source_addresses.len() > 1 {
            phase.disable_sharing(transfer.source);
        }
        for destination in &transfer.destinations {
            phase.connection(
                transfer.source,
                destination.tile,
                u64::from(transfer.words) * 4,
                transfer.width == crate::exchange::ExchangeItemWidth::Paired64,
                transfer.destinations.len(),
            );
            phase.receive(
                destination.tile,
                u64::from(destination.address),
                u64::from(transfer.words) * 4,
            );
        }
    }
    phase
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
    fn sample(address: u64, bytes: u64, target: u16, iterated: bool) -> ExchangeStoragePhase {
        let mut phase = ExchangeStoragePhase::new(3);
        phase.connection(0, target, bytes, false, 1);
        phase.send(0, 1, u64::from(bytes > 256));
        phase.receive(target, address, bytes);
        if iterated {
            phase.disable_sharing(0);
        }
        phase
    }

    #[test]
    fn relocated_rows_share_code_but_keep_patch_storage() {
        let mut table = ExchangeStorageEstimator::new(3);
        table.add(sample(0x60000, 32, 1, false));
        assert_eq!(table.maximum_bytes(), 24);
        table.add(sample(0x70000, 32, 1, false));
        assert_eq!(table.maximum_bytes(), 40); // offsets and two value arrays
        table.add(sample(0x80000, 32, 1, false));
        assert_eq!(table.maximum_bytes(), 48); // one more value array
        assert_eq!(table.bytes[2], 8); // inactive rows share too
    }

    #[test]
    fn payload_routes_and_iterated_senders_prevent_incompatible_sharing() {
        for (changed, expected) in [
            (sample(0x70000, 64, 1, false), 48),
            (sample(0x70000, 32, 2, false), 32),
            (sample(0x70000, 32, 1, true), 40),
        ] {
            let mut table = ExchangeStorageEstimator::new(3);
            table.add(sample(0x60000, 32, 1, false));
            table.add(changed);
            assert_eq!(table.maximum_bytes(), expected);
        }
    }
}
