//! The local FP16 sum consumes one seed and a dense stack of contributors.
//! Distributed contributor grouping and staging belong to low reduction expansion.

use super::*;
use crate::mid::MidOperationKind;

pub(super) fn call(run: &KernelRun) -> Result<KernelCall, KernelAbiError> {
    run.check_arity(2, 1)?;
    let MidOperationKind::ReductionSum { partials } = run.kernel else {
        return Err(KernelAbiError::RequirementMismatch);
    };
    if run.requirements.outputs[0].format.precision != Precision::F16 {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let count = element_count(&run.outputs[0].extents)?;
    if partials < 2 || !count.is_multiple_of(8) {
        return Err(KernelAbiError::UnsupportedElementCount {
            symbol: "reduce_sum_f16",
            count,
            divisor: 8,
        });
    }
    Ok(KernelCall::exact(
        "reduce_sum_f16",
        vec![u32::from(partials - 1), count],
    ))
}
