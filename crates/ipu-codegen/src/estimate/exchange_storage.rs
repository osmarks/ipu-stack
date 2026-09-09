//! Order-sensitive row-storage estimate, without placement or scheduling.
//!
//! Count sender setup/send words, receive source/neutral controls and pointer
//! changes independently. Delays, bidi fusion and final row compatibility remain
//! scheduler-dependent, so this is a ranking estimate, never a feasibility bound.

use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
};

/// An ordered sequence fingerprint with associative concatenation. Unlike
/// streaming a hasher, repeated subsequences can be composed in logarithmic time.
/// This predicts row sharing only; encoded rows remain the authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Signature {
    hash: u64,
    power: u64,
}
impl Default for Signature {
    fn default() -> Self {
        Self { hash: 0, power: 1 }
    }
}
impl Signature {
    fn event(value: impl Hash) -> Self {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        Self {
            hash: hasher.finish(),
            power: 0x9e3779b185ebca87,
        }
    }
    fn then(self, right: Self) -> Self {
        Self {
            hash: self.hash.wrapping_mul(right.power).wrapping_add(right.hash),
            power: self.power.wrapping_mul(right.power),
        }
    }
    fn repeat(mut self, mut count: u64) -> Self {
        let mut result = Self::default();
        while count != 0 {
            if count & 1 != 0 {
                result = result.then(self);
            }
            self = self.then(self);
            count >>= 1;
        }
        result
    }
    fn append(&mut self, value: impl Hash) {
        *self = self.then(Self::event(value));
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExchangeStoragePhase {
    bytes: Vec<u64>,
    receive_end: Vec<Option<u64>>,
    signatures: Vec<Signature>,
    patches: Vec<u64>,
    shareable: Vec<bool>,
}

impl ExchangeStoragePhase {
    pub(crate) fn new(tiles: u16) -> Self {
        Self {
            bytes: vec![8; usize::from(tiles)], // entry and return
            receive_end: vec![None; usize::from(tiles)],
            signatures: vec![Signature::default(); usize::from(tiles)],
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
        self.signatures[index].append((2u8, reset));
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
        self.signatures[usize::from(source)].append((0u8, target, bytes, paired, receivers));
        self.signatures[usize::from(target)].append((1u8, source, bytes, paired, receivers));
    }

    /// Append equal copy rows, splitting each at the hardware transfer limit.
    /// Counts and fingerprints equal connection()/receive() for every chunk,
    /// including loopback and pointer continuation across consecutive calls.
    pub(crate) fn connection_rows(
        &mut self,
        source: u16,
        target: u16,
        address: u64,
        bytes: u32,
        rows: u32,
        stride: u32,
        receivers: usize,
        limit: u32,
    ) {
        if bytes == 0 || rows == 0 {
            return;
        }
        if rows == 1 && bytes <= limit {
            self.connection(source, target, u64::from(bytes), false, receivers);
            self.receive(target, address, u64::from(bytes));
            return;
        }
        let source = usize::from(source);
        let target = usize::from(target);
        let chunks = u64::from(bytes.div_ceil(limit));
        let resets = u64::from(self.receive_end[target] != Some(address))
            + u64::from(rows - 1) * u64::from(stride != bytes);
        self.bytes[target] += u64::from(rows) * chunks * 8 + resets * 4;
        self.patches[target] += resets;
        let chunk = |bytes: u64, reset: bool| {
            let tx = Signature::event((0u8, target as u16, bytes, false, receivers));
            let rx = Signature::event((1u8, source as u16, bytes, false, receivers))
                .then(Signature::event((2u8, reset)));
            if source == target {
                (Signature::default(), tx.then(rx))
            } else {
                (tx, rx)
            }
        };
        let row = |reset| {
            let first_bytes = bytes.min(limit);
            let (mut tx, mut rx) = chunk(u64::from(first_bytes), reset);
            let remaining = bytes - first_bytes;
            let full = u64::from(remaining / limit);
            if full != 0 {
                let (a, b) = chunk(u64::from(limit), false);
                tx = tx.then(a.repeat(full));
                rx = rx.then(b.repeat(full));
            }
            if !remaining.is_multiple_of(limit) {
                let (a, b) = chunk(u64::from(remaining % limit), false);
                tx = tx.then(a);
                rx = rx.then(b);
            }
            (tx, rx)
        };
        let (tx, rx) = row(self.receive_end[target] != Some(address));
        let (more_tx, more_rx) = if rows > 1 {
            row(stride != bytes)
        } else {
            (Signature::default(), Signature::default())
        };
        self.signatures[source] =
            self.signatures[source].then(tx.then(more_tx.repeat(u64::from(rows - 1))));
        self.signatures[target] =
            self.signatures[target].then(rx.then(more_rx.repeat(u64::from(rows - 1))));
        self.receive_end[target] =
            Some(address + u64::from(rows - 1) * u64::from(stride) + u64::from(bytes));
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
            let key = (phase.signatures[tile].hash, bytes, patches);
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

#[cfg(test)]
mod symbolic_tests {
    use super::*;
    #[test]
    fn repeated_rows_match_chunk_events_and_pointer_continuation() {
        for source in [0, 1] {
            for bytes in [0u32, 1, 256, 257, 1024, 1025, 4097] {
                for rows in [0u32, 1, 2, 17, 1024] {
                    for gap in [0, 4, 256] {
                        let mut symbolic = ExchangeStoragePhase::new(2);
                        let mut enumerated = ExchangeStoragePhase::new(2);
                        for address in [
                            32u64,
                            32 + u64::from(rows) * u64::from(bytes + gap),
                            1 << 32,
                        ] {
                            symbolic.connection_rows(
                                source,
                                1,
                                address,
                                bytes,
                                rows,
                                bytes + gap,
                                3,
                                1024,
                            );
                            for row in 0..rows {
                                let mut offset = 0;
                                while offset < bytes {
                                    let chunk = (bytes - offset).min(1024);
                                    enumerated.connection(source, 1, u64::from(chunk), false, 3);
                                    enumerated.receive(
                                        1,
                                        address
                                            + u64::from(row) * u64::from(bytes + gap)
                                            + u64::from(offset),
                                        u64::from(chunk),
                                    );
                                    offset += chunk;
                                }
                            }
                            assert_eq!(symbolic, enumerated);
                        }
                    }
                }
            }
        }
    }
}
