//! The local FP16 sum consumes one seed and a dense stack of contributors.
//! Distributed contributor grouping and staging belong to planner construction.

use super::*;
use crate::mid::MidOperationKind;

pub(super) fn call(run: &KernelRun) -> Result<KernelCall, KernelAbiError> {
    let MidOperationKind::ReductionSum { partials } = run.kernel else {
        return Err(KernelAbiError::RequirementMismatch);
    };
    run.check_arity(if partials == 1 { 1 } else { 2 }, 1)?;
    if run.requirements.outputs[0].format.precision != Precision::F16 {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let count = element_count(&run.outputs[0].extents)?;
    if element_count(&run.inputs[0].extents)? != count {
        return Err(KernelAbiError::RequirementMismatch);
    }
    if partials == 1 {
        if !count.is_multiple_of(4) {
            return Err(KernelAbiError::RequirementMismatch);
        }
        return Ok(KernelCall::exact(
            crate::kernel::abi::COPY_U64_SYMBOL,
            vec![count / 4],
        ));
    }
    if element_count(&run.inputs[1].extents)?
        != count
            .checked_mul(u32::from(partials.saturating_sub(1)))
            .ok_or(KernelAbiError::RequirementMismatch)?
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
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
