//! Fusion rewrites selected mid operations before tile binding. Pass ordering
//! and Repeat traversal live here; individual passes only transform one region.
use super::{MidOperation, MidOperationKind, MidProgram, MidValue, MidValueId, rewrite};

mod elementwise;
mod output;
mod residual;

impl MidProgram {
    pub(crate) fn with_fusions(&self) -> Option<Self> {
        let mut result = self.clone();
        if !run(&mut result.operations, &mut result.values, &result.outputs) {
            return None;
        }
        result.refresh_estimates()?;
        Some(result)
    }
}

fn run(
    operations: &mut Vec<MidOperation>,
    values: &mut Vec<MidValue>,
    required: &[MidValueId],
) -> bool {
    let mut changed = false;
    for operation in &mut *operations {
        if let MidOperationKind::Repeat(repeat) = &mut operation.kind {
            changed |= run(&mut repeat.body.operations, values, &repeat.body.yields);
        }
    }
    changed |= residual::run(operations, values, required);
    changed |= elementwise::run(operations, values, required);
    changed |= output::run(operations, values, required);
    changed
}
