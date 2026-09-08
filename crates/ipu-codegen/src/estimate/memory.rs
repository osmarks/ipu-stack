//! Allocation requirements and region liveness; no cycle pricing.

use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

/// Independent class maxima and maximum simultaneous live storage. Region 1
/// is shared by both classes; separate peaks need not coexist. These are cheap
/// capacity screens, not a guarantee that aligned concrete placement succeeds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryPeaks {
    pub standard: u64,
    pub interleaved: u64,
    pub total: u64,
    /// Persistent standard-memory estimate for generated exchange rows.
    pub exchange_rows: u64,
    pub maximum_standard_allocation: u64,
}

impl MemoryPeaks {
    pub(crate) const OBJECTIVE_COUNT: usize = 6;

    /// Shared Pareto dimensions for operator and region shortlists.
    pub(crate) fn objectives(self) -> [u64; Self::OBJECTIVE_COUNT] {
        [
            self.standard,
            self.interleaved,
            self.total,
            self.maximum_standard_allocation,
            self.standard_contiguous_overflow(),
            self.exchange_rows,
        ]
    }

    pub(crate) fn observe(&mut self, usage: MemoryUsage, maximum_standard_allocation: u64) {
        self.standard = self.standard.max(usage.standard);
        self.interleaved = self.interleaved.max(usage.interleaved);
        self.total = self.total.max(usage.total());
        self.maximum_standard_allocation = self
            .maximum_standard_allocation
            .max(maximum_standard_allocation);
    }

    pub fn fits_ipu21_with_budget(
        self,
        reserved_standard_bytes: u64,
        tile_memory_budget_bytes: u64,
    ) -> bool {
        self.interleaved <= u64::from(crate::memory::IPU21_INTERLEAVED_REGION_BYTES)
            // Row storage is a coarse ranking estimate: it sums independent
            // phase maxima and cannot prove that an allocation is impossible.
            // Exact encoded rows participate in package acceptance after scheduling.
            && self.total.saturating_sub(self.exchange_rows).saturating_add(reserved_standard_bytes)
                <= tile_memory_budget_bytes.min(u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES))
            && self.contiguous_overflow(reserved_standard_bytes) == 0
    }

    pub(crate) fn standard_contiguous_overflow(self) -> u64 {
        self.contiguous_overflow(0)
    }

    pub fn standard_contiguous_overflow_with_reservation(
        self,
        reserved_standard_bytes: u64,
    ) -> u64 {
        self.contiguous_overflow(reserved_standard_bytes.saturating_add(self.exchange_rows))
    }

    fn contiguous_overflow(self, reserved: u64) -> u64 {
        // A standard buffer can use all of region 1 when interleaved
        // temporaries are dead. Do not subtract an unrelated class peak.
        let upper_standard = u64::from(crate::memory::IPU21_INTERLEAVED_REGION_BYTES);
        let lower_standard =
            u64::from(crate::memory::IPU21_STANDARD_FIXED_BYTES).saturating_sub(reserved);
        self.maximum_standard_allocation
            .saturating_sub(lower_standard.max(upper_standard))
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
