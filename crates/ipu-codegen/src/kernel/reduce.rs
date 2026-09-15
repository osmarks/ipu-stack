//! The local FP16 sum consumes one seed and a dense stack of contributors.
//! Distributed contributor grouping and staging belong to planner construction.

use super::*;
use crate::mid::MidOperationKind;

pub(super) fn call(
    kernel: &MidOperationKind,
    inputs: &[TensorStorage<'_>],
    outputs: &[TensorStorage<'_>],
    build: Option<&mut KernelObjects>,
) -> Result<KernelCall, KernelError> {
    let MidOperationKind::ReductionSum { partials } = *kernel else {
        return Err(KernelError::RequirementMismatch);
    };
    check_arity(inputs, outputs, if partials == 1 { 1 } else { 2 }, 1)?;
    if outputs[0].format.precision != Precision::F16 {
        return Err(KernelError::RequirementMismatch);
    }
    let count = outputs[0].count()?;
    if inputs[0].count()? != count {
        return Err(KernelError::RequirementMismatch);
    }
    if partials == 1 {
        if !count.is_multiple_of(4) {
            return Err(KernelError::RequirementMismatch);
        }
        return Ok(KernelCall::copy_u64(count / 4));
    }
    if inputs[1].count()?
        != count
            .checked_mul(u32::from(partials.saturating_sub(1)))
            .ok_or(KernelError::RequirementMismatch)?
    {
        return Err(KernelError::RequirementMismatch);
    }
    if partials < 2 || !count.is_multiple_of(8) {
        return Err(KernelError::UnsupportedElementCount {
            symbol: "reduce_sum_f16",
            count,
            divisor: 8,
        });
    }
    if let Some(build) = build {
        build.add_compilation(KernelCompilation {
            source: "reduce_add_f16.S",
            name: "reduce_add_f16".into(),
            flags: Vec::new(),
        });
    }
    Ok(KernelCall::new(
        "reduce_sum_f16",
        vec![u32::from(partials - 1), count],
        f16_reduction_cycles(count.into(), partials.into()),
    ))
}

/// reduce_add_f16.S: six workers, eight elements per iteration, eleven issue
/// groups (including repeat alignment) plus two per remote partial. Setup includes supervisor rendezvous.
/// Identity reductions select the copy family and are priced there.
pub(crate) fn f16_reduction_cycles(elements: u64, partials: u64) -> u64 {
    if elements == 0 || partials < 2 {
        return 0;
    }
    282u64.saturating_add(
        elements
            .div_ceil(48)
            .saturating_mul(6)
            .saturating_mul(11u64.saturating_add(partials.saturating_sub(1).saturating_mul(2))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduction_matches_unmerged_hardware_samples() {
        // Independent IPU21 sweep measurements, including uneven worker loads.
        for (elements, partials, measured) in [
            (8, 2, 360),
            (56, 2, 438),
            (1472, 2, 2700),
            (2208, 4, 4974),
            (1472, 4, 3444),
            (576, 15, 3090),
            (2208, 28, 18222),
        ] {
            assert_eq!(f16_reduction_cycles(elements, partials), measured);
        }
        assert_eq!(f16_reduction_cycles(0, 4), 0);
        assert_eq!(f16_reduction_cycles(u64::MAX, u64::MAX), u64::MAX);
    }

    #[test]
    fn identity_reduction_covers_exactly_the_source_with_worker_copy_abi() {
        for elements in (4..=512).step_by(4) {
            let tensor =
                crate::TensorType::new([elements], Precision::F16, crate::Layout::row_sharded(1));
            let extents = [crate::ShardExtent {
                axis: 0,
                start: 0,
                logical_end: elements,
                physical_end: elements,
            }];
            let geometry = TensorStorage {
                format: &tensor.format,
                extents: &extents,
            };
            let call = KernelCall::select(
                &MidOperationKind::ReductionSum { partials: 1 },
                &[geometry],
                &[geometry],
                None,
            )
            .unwrap();
            let [whole, remainder] = call.arguments.as_slice() else {
                panic!("copy ABI needs both worker counts");
            };
            let mut covered = Vec::new();
            for worker in 0..ipu_target::ipu21::WORKER_CONTEXTS {
                // The assembly partitions contiguous words by quotient/remainder.
                let start = worker * whole + worker.min(*remainder);
                let count = whole + u32::from(worker < *remainder);
                covered.extend(start * 8..(start + count) * 8);
            }
            assert_eq!(covered, (0..elements * 2).collect::<Vec<_>>());
            assert_eq!(call.symbol, copy::COPY_U64_SYMBOL);
            assert!(call.cycles >= 246);
        }
    }
}
