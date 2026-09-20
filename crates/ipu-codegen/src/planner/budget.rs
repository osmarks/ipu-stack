//! Per-tile memory allowances assigned by the outer planner to a high operation.
//! These are capacity limits, not physical arenas. Code and exchange rows are
//! outside this accounting for now.

/// The same quantities describe measured requirements and assigned limits.
/// Category peaks need not coincide: total is the maximum simultaneous sum,
/// not the sum of the two category maxima.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryRequirements {
    pub interleaved_required: u64,
    /// May occupy either standard or interleaved memory.
    pub flexible: u64,
    pub total: u64,
}

impl MemoryRequirements {
    pub fn observe(&mut self, interleaved_required: u64, flexible: u64) {
        self.interleaved_required = self.interleaved_required.max(interleaved_required);
        self.flexible = self.flexible.max(flexible);
        self.total = self
            .total
            .max(interleaved_required.saturating_add(flexible));
    }

    pub fn fits_within(self, allowance: Self) -> bool {
        self.interleaved_required <= allowance.interleaved_required
            && self.flexible <= allowance.flexible
            && self.total <= allowance.total
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OperationBudget {
    /// Resident allocations, including parameters, charged once across users.
    /// Fixed allowances accumulate across the model.
    pub fixed: MemoryRequirements,
    /// Boundary activations and private scratch during this operation. Boundary
    /// tensors can outlive the operation: graph-level liveness must reserve them
    /// across subsequent operations until their last use. Before assigning this
    /// allowance, the outer planner deducts all resident storage and unrelated
    /// live tensors (for example a residual held across a branch). Inputs and
    /// outputs counted here must not also be deducted as unrelated live tensors.
    pub temporary: MemoryRequirements,
}

impl OperationBudget {
    pub fn fits_within(self, allowance: Self) -> bool {
        self.fixed.fits_within(allowance.fixed) && self.temporary.fits_within(allowance.temporary)
    }
}
