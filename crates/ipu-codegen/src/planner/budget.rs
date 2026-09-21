//! Resident reservations coexist with every historical nonresident peak.
//! Keep tile identity: maxima on disjoint tiles must not be added together.

use crate::estimate::{MemoryObserver, MemoryPeaks, MemoryUsage};
use crate::{MemoryClass, ValueId};
use std::collections::BTreeMap;

#[derive(Clone, Default)]
pub(super) struct Memory {
    pub parameters: BTreeMap<ValueId, Vec<MemoryUsage>>,
    pub nonresident: Vec<MemoryPeaks>,
    pub nonresident_total: u64,
}

impl MemoryObserver for Memory {
    const SPLIT_RESIDENT: bool = true;
    fn resident(&mut self, origin: ValueId, class: MemoryClass, bytes: &[u64]) {
        self.parameters.insert(
            origin,
            bytes
                .iter()
                .map(|&bytes| {
                    let mut usage = MemoryUsage::default();
                    usage.add_class(class, bytes);
                    usage
                })
                .collect(),
        );
    }

    fn nonresident(&mut self, usage: &[MemoryUsage], maximum_standard: u64) {
        self.nonresident_total = self
            .nonresident_total
            .max(usage.iter().map(|usage| usage.total()).sum());
        self.nonresident.resize(usage.len(), MemoryPeaks::default());
        for (peak, &usage) in self.nonresident.iter_mut().zip(usage) {
            peak.observe(usage, maximum_standard);
        }
    }
}

impl Memory {
    pub fn peak(&self, include: impl Fn(ValueId) -> bool) -> MemoryPeaks {
        let mut result = MemoryPeaks::default();
        for (tile, temporary) in self.nonresident.iter().enumerate() {
            let mut resident = MemoryUsage::default();
            let mut maximum = temporary.maximum_standard_allocation;
            for (&id, bytes) in &self.parameters {
                if include(id) {
                    resident = resident.saturating_add(bytes[tile]);
                    maximum = maximum.max(bytes[tile].standard);
                }
            }
            result.include(MemoryPeaks {
                standard: temporary.standard.saturating_add(resident.standard),
                interleaved: temporary.interleaved.saturating_add(resident.interleaved),
                total: temporary.total.saturating_add(resident.total()),
                maximum_standard_allocation: maximum,
                exchange_rows: 0,
            });
        }
        result
    }

    pub fn dominates(&self, other: &Self) -> bool {
        self.nonresident_total <= other.nonresident_total
            && self
                .nonresident
                .iter()
                .zip(&other.nonresident)
                .all(|(a, b)| {
                    a.standard <= b.standard
                        && a.interleaved <= b.interleaved
                        && a.total <= b.total
                        && a.maximum_standard_allocation <= b.maximum_standard_allocation
                })
            && self.parameters.iter().all(|(id, a)| {
                a.iter()
                    .zip(&other.parameters[id])
                    .all(|(a, b)| a.standard <= b.standard && a.interleaved <= b.interleaved)
            })
    }
}

/// Applied after dominance pruning within identical live boundary states.
/// None retains the complete frontier; positive limits make search approximate.
/// These do not limit local candidate enumeration.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SearchLimits {
    pub states_per_boundary: Option<usize>,
    pub paths_per_state: Option<usize>,
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self {
            states_per_boundary: Some(128),
            paths_per_state: Some(16),
        }
    }
}
