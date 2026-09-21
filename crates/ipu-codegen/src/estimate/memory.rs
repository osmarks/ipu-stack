//! Allocation requirements and region liveness; no cycle pricing.

use super::*;
use ipu_target::Target;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct MemoryUsage {
    pub standard: u64,
    pub interleaved: u64,
}

impl MemoryUsage {
    pub const fn total(self) -> u64 {
        self.standard.saturating_add(self.interleaved)
    }

    pub(crate) fn add_class(&mut self, class: MemoryClass, bytes: u64) {
        let target = match class {
            MemoryClass::Ipu21Standard => &mut self.standard,
            MemoryClass::Ipu21Interleaved => &mut self.interleaved,
        };
        *target = target.saturating_add(bytes);
    }

    pub(crate) fn saturating_add(self, other: Self) -> Self {
        Self {
            standard: self.standard.saturating_add(other.standard),
            interleaved: self.interleaved.saturating_add(other.interleaved),
        }
    }
}

/// Tensor-only class maxima and maximum simultaneous live storage.
/// Separate peaks need not coexist. These are cheap
/// capacity screens, not a guarantee that aligned concrete placement succeeds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct MemoryPeaks {
    pub standard: u64,
    pub interleaved: u64,
    pub total: u64,
    /// Separate persistent estimate for generated exchange rows; ranking only.
    pub exchange_rows: u64,
    pub maximum_standard_allocation: u64,
}

impl MemoryPeaks {
    pub(crate) fn include(&mut self, other: Self) {
        self.standard = self.standard.max(other.standard);
        self.interleaved = self.interleaved.max(other.interleaved);
        self.total = self.total.max(other.total);
        self.maximum_standard_allocation = self
            .maximum_standard_allocation
            .max(other.maximum_standard_allocation);
        self.exchange_rows = self.exchange_rows.max(other.exchange_rows);
    }
    /// Shared Pareto dimensions for operator and region shortlists.
    pub(crate) fn objectives(self) -> [u64; 5] {
        [
            self.standard.saturating_add(self.exchange_rows),
            self.interleaved,
            self.total_with_exchange(),
            self.maximum_standard_allocation,
            self.exchange_rows,
        ]
    }

    pub(crate) fn total_with_exchange(self) -> u64 {
        self.total.saturating_add(self.exchange_rows)
    }

    pub(crate) fn observe(&mut self, usage: MemoryUsage, maximum_standard_allocation: u64) {
        self.standard = self.standard.max(usage.standard);
        self.interleaved = self.interleaved.max(usage.interleaved);
        self.total = self.total.max(usage.total());
        self.maximum_standard_allocation = self
            .maximum_standard_allocation
            .max(maximum_standard_allocation);
    }

    pub fn fits_with_budget(
        self,
        target: Target,
        reserved_standard_bytes: u64,
        tile_memory_budget_bytes: u64,
    ) -> bool {
        self.interleaved <= target.interleaved_data_bytes()
            // Row storage is a coarse ranking estimate: it sums independent
            // phase maxima and cannot prove that an allocation is impossible.
            // Exact encoded rows participate in package acceptance after scheduling.
            && self.total.saturating_add(reserved_standard_bytes)
                <= tile_memory_budget_bytes.min(target.planned_data_bytes())
            && self.standard_contiguous_overflow_with_reservation(target, reserved_standard_bytes)
                == 0
    }

    pub fn standard_contiguous_overflow_with_reservation(
        self,
        target: Target,
        reserved_standard_bytes: u64,
    ) -> u64 {
        self.maximum_standard_allocation
            .saturating_sub(target.maximum_standard_allocation(reserved_standard_bytes))
    }
}

pub(crate) fn tensor_memory(tensor: &TensorType) -> MemoryUsage {
    let mut usage = MemoryUsage::default();
    usage.add_class(
        tensor.format.layout.memory_class,
        maximum_shard_bytes(tensor),
    );
    usage
}
