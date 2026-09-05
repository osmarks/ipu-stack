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
    pub(crate) fn observe(&mut self, usage: MemoryUsage, maximum_standard_allocation: u64) {
        self.standard = self.standard.max(usage.standard);
        self.interleaved = self.interleaved.max(usage.interleaved);
        self.total = self.total.max(usage.total());
        self.maximum_standard_allocation = self
            .maximum_standard_allocation
            .max(maximum_standard_allocation);
        let interleaved_boundary = self
            .interleaved
            .div_ceil(u64::from(ipu_package::IPU21_INTERLEAVED_ELEMENT_SIZE))
            * u64::from(ipu_package::IPU21_INTERLEAVED_ELEMENT_SIZE);
        let upper_standard = u64::from(crate::memory::IPU21_INTERLEAVED_REGION_BYTES)
            .saturating_sub(interleaved_boundary);
        let contiguous_capacity =
            u64::from(crate::memory::IPU21_STANDARD_FIXED_BYTES).max(upper_standard);
        self.standard_contiguous_overflow = self
            .maximum_standard_allocation
            .saturating_sub(contiguous_capacity);
    }

    pub fn fits_ipu21_with_budget(
        self,
        reserved_standard_bytes: u64,
        tile_memory_budget_bytes: u64,
    ) -> bool {
        let partitioned_bytes = self
            .standard
            .saturating_add(self.aligned_interleaved_bytes())
            .saturating_add(reserved_standard_bytes);
        self.interleaved <= u64::from(crate::memory::IPU21_INTERLEAVED_REGION_BYTES)
            && partitioned_bytes
                <= tile_memory_budget_bytes.min(u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES))
            && self.standard_contiguous_overflow_with_reservation(reserved_standard_bytes) == 0
    }

    fn aligned_interleaved_bytes(self) -> u64 {
        self.interleaved
            .div_ceil(u64::from(ipu_package::IPU21_INTERLEAVED_ELEMENT_SIZE))
            .saturating_mul(u64::from(ipu_package::IPU21_INTERLEAVED_ELEMENT_SIZE))
    }

    pub fn standard_contiguous_overflow_with_reservation(
        self,
        reserved_standard_bytes: u64,
    ) -> u64 {
        let interleaved_boundary = self.aligned_interleaved_bytes();
        let upper_standard = u64::from(crate::memory::IPU21_INTERLEAVED_REGION_BYTES)
            .saturating_sub(interleaved_boundary);
        let lower_standard = u64::from(crate::memory::IPU21_STANDARD_FIXED_BYTES)
            .saturating_sub(reserved_standard_bytes.saturating_add(self.exchange_rows));
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

pub(crate) fn region_peak_memory(
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
) -> MemoryPeaks {
    region_peak_memory_with_multiplicity(initial, operations, outputs, values, &BTreeMap::new())
}

pub(crate) fn region_peak_memory_with_multiplicity(
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
    allocation_multiplicity: &BTreeMap<MidValueId, u32>,
) -> MemoryPeaks {
    region_estimate(
        initial,
        operations,
        outputs,
        values,
        allocation_multiplicity,
    )
    .map_or_else(unavailable_memory, |(_, peak)| peak)
}

pub(crate) fn unavailable_memory() -> MemoryPeaks {
    MemoryPeaks {
        standard: u64::MAX,
        interleaved: u64::MAX,
        total: u64::MAX,
        ..MemoryPeaks::default()
    }
}

pub(crate) fn region_estimate(
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
    allocation_multiplicity: &BTreeMap<MidValueId, u32>,
) -> Option<(std::sync::Arc<crate::TileGraph>, MemoryPeaks)> {
    let mut outputs = outputs.to_vec();
    // A pending view still needs its source storage at the region boundary.
    for operation in operations.iter().rev() {
        if let Some(offer) = operation
            .operator_plan()
            .and_then(|plan| plan.deferred_output)
            && operation
                .results
                .iter()
                .any(|result| outputs.contains(result))
        {
            outputs.push(operation.inputs[offer.source_input]);
        }
    }
    let candidate = crate::MidProgram {
        tile_count: values
            .iter()
            .map(|value| value.tensor_type.format.layout.tiling.tile_count)
            .max()
            .unwrap_or(1),
        inputs: initial
            .iter()
            .map(|&value| crate::MidInput {
                name: String::new(),
                kind: crate::GraphInputKind::Host,
                value,
            })
            .collect(),
        values: values.to_vec(),
        operations: operations.to_vec(),
        outputs,
        ..crate::MidProgram::default()
    };

    let program = crate::low::expand::expand_tiles(&candidate).ok()?;
    let low = crate::lower_to_tiles(&program, false);
    let mut peak =
        crate::place::program_memory_with_multiplicity(&low, allocation_multiplicity).ok()?;
    let rows = super::program::program_footprint(&program)
        .ok()?
        .estimated_row_bytes();
    peak.exchange_rows = rows;
    peak.standard = peak.standard.saturating_add(rows);
    peak.total = peak.total.saturating_add(rows);
    Some((program, peak))
}
