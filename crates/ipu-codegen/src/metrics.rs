//! Shared cycle and per-tile memory metrics used by operator and region planning.

use crate::config::HardwareMemoryConstraints;
use crate::cost::CostEstimate;
use crate::layout::MemoryClass;

/// Maximum per-tile bytes attributed to each address/load class. The classes
/// share physical tile SRAM, so feasibility must check both the individual
/// interleaved-region limit and their combined size.
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
            MemoryClass::Standard => &mut self.standard,
            MemoryClass::Interleaved => &mut self.interleaved,
        };
        *target = target.saturating_add(bytes);
    }

    pub(crate) fn saturating_add(self, other: Self) -> Self {
        Self {
            standard: self.standard.saturating_add(other.standard),
            interleaved: self.interleaved.saturating_add(other.interleaved),
        }
    }

    pub fn fits(self, constraints: HardwareMemoryConstraints) -> bool {
        self.interleaved <= constraints.interleaved_bytes && self.total() <= constraints.total_bytes
    }
}

/// Independent class maxima and the maximum simultaneous total. The allocator
/// fixes the interleaved arena boundary for the whole program, so feasibility
/// uses the sum of the class maxima even when they occur in different phases.
/// `total` remains useful for ranking the actual peak live working set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryPeaks {
    pub standard: u64,
    pub interleaved: u64,
    pub total: u64,
    /// Persistent standard-memory estimate for generated exchange rows.
    pub exchange_rows: u64,
    pub maximum_standard_allocation: u64,
    /// Largest amount by which one standard-addressed allocation exceeded
    /// both contiguous ranges left around the interleaved region.
    pub standard_contiguous_overflow: u64,
}

impl MemoryPeaks {
    pub(crate) fn observe(
        &mut self,
        usage: MemoryUsage,
        maximum_standard_allocation: u64,
        constraints: HardwareMemoryConstraints,
    ) {
        self.standard = self.standard.max(usage.standard);
        self.interleaved = self.interleaved.max(usage.interleaved);
        self.total = self.total.max(usage.total());
        self.maximum_standard_allocation = self
            .maximum_standard_allocation
            .max(maximum_standard_allocation);
        let interleaved_boundary = self
            .interleaved
            .div_ceil(constraints.interleaved_element_bytes)
            * constraints.interleaved_element_bytes;
        let upper_standard = constraints
            .interleaved_bytes
            .saturating_sub(interleaved_boundary);
        let contiguous_capacity = constraints.standard_fixed_bytes.max(upper_standard);
        self.standard_contiguous_overflow = self
            .maximum_standard_allocation
            .saturating_sub(contiguous_capacity);
    }

    pub fn fits(self, constraints: HardwareMemoryConstraints) -> bool {
        self.fits_with_budget(constraints, 0, constraints.total_bytes)
    }

    pub fn fits_with_budget(
        self,
        constraints: HardwareMemoryConstraints,
        reserved_standard_bytes: u64,
        tile_memory_budget_bytes: u64,
    ) -> bool {
        let partitioned_bytes = self
            .standard
            .saturating_add(self.aligned_interleaved_bytes(constraints))
            .saturating_add(reserved_standard_bytes);
        self.interleaved <= constraints.interleaved_bytes
            && partitioned_bytes <= tile_memory_budget_bytes.min(constraints.total_bytes)
            && self
                .standard_contiguous_overflow_with_reservation(constraints, reserved_standard_bytes)
                == 0
    }

    fn aligned_interleaved_bytes(self, constraints: HardwareMemoryConstraints) -> u64 {
        self.interleaved
            .div_ceil(constraints.interleaved_element_bytes)
            .saturating_mul(constraints.interleaved_element_bytes)
    }

    pub fn standard_contiguous_overflow_with_reservation(
        self,
        constraints: HardwareMemoryConstraints,
        reserved_standard_bytes: u64,
    ) -> u64 {
        let interleaved_boundary = self
            .interleaved
            .div_ceil(constraints.interleaved_element_bytes)
            .saturating_mul(constraints.interleaved_element_bytes);
        let upper_standard = constraints
            .interleaved_bytes
            .saturating_sub(interleaved_boundary);
        let lower_standard = constraints
            .standard_fixed_bytes
            .saturating_sub(reserved_standard_bytes.saturating_add(self.exchange_rows));
        self.maximum_standard_allocation
            .saturating_sub(lower_standard.max(upper_standard))
    }

    pub(crate) fn conservative_tensor_usage(self) -> MemoryUsage {
        MemoryUsage {
            standard: self.standard.saturating_sub(self.exchange_rows),
            interleaved: self.interleaved,
        }
    }
}

/// Storage visible at an operator boundary plus phase-local scratch. Peak is
/// the simultaneous requirement used for candidate feasibility.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryEstimate {
    pub live: MemoryUsage,
    pub temporary: MemoryUsage,
    pub peak: MemoryUsage,
    /// Largest phase-local standard-addressed buffer which must fit in one
    /// contiguous standard-memory range.
    pub maximum_standard_temporary_allocation: u64,
}

impl MemoryEstimate {
    pub(crate) fn peaks(self, exchange_rows: u64) -> MemoryPeaks {
        MemoryPeaks {
            standard: self.peak.standard,
            interleaved: self.peak.interleaved,
            total: self.peak.total(),
            exchange_rows,
            maximum_standard_allocation: self.maximum_standard_temporary_allocation,
            standard_contiguous_overflow: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanMetrics<M = MemoryEstimate> {
    pub cost: CostEstimate,
    pub memory: M,
}

pub type OperationMetrics = PlanMetrics<MemoryEstimate>;
pub type RegionMetrics = PlanMetrics<MemoryPeaks>;

impl PlanMetrics<MemoryPeaks> {
    fn pareto_dimensions(self) -> [u64; 7] {
        [
            self.cost.cycles,
            self.memory.standard,
            self.memory.interleaved,
            self.memory.total,
            self.memory.maximum_standard_allocation,
            self.memory.standard_contiguous_overflow,
            self.memory.exchange_rows,
        ]
    }

    pub(crate) fn dominates(self, other: Self) -> bool {
        let left = self.pareto_dimensions();
        let right = other.pareto_dimensions();
        left.iter().zip(right).all(|(left, right)| *left <= right)
            && left.iter().zip(right).any(|(left, right)| *left < right)
    }
}
