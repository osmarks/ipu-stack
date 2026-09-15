//! Layer normalization and distributed statistics/apply call contracts.

use super::*;
use crate::mid::MidOperationKind;

pub(super) fn call(run: &KernelRun) -> Result<KernelCall, KernelAbiError> {
    let inputs = match run.kernel {
        MidOperationKind::LayerNorm => 3,
        MidOperationKind::AddLayerNorm | MidOperationKind::LayerNormApply { .. } => 4,
        MidOperationKind::LayerNormMoments => 1,
        MidOperationKind::AddLayerNormMoments => 2,
        _ => return Err(KernelAbiError::RequirementMismatch),
    };
    run.check_arity(
        inputs,
        1 + usize::from(run.kernel == MidOperationKind::AddLayerNormMoments),
    )?;
    if let Some(arguments) = output::fp8_arguments(run)? {
        return Ok(KernelCall::exact("layer_norm_f8", arguments));
    }
    if matches!(
        run.kernel,
        MidOperationKind::LayerNorm | MidOperationKind::AddLayerNorm
    ) {
        if run.requirements.outputs[0].format.precision != Precision::F16 {
            return Err(KernelAbiError::Unavailable(run.kernel.clone()));
        }
        let width = output::f16_row_width(run)?;
        if matrix_extent(&run.outputs[0], false, true)? != width
            || (run.kernel == MidOperationKind::AddLayerNorm
                && (run.inputs[0].extents != run.outputs[0].extents
                    || run.inputs[1].extents != run.outputs[0].extents))
        {
            return Err(KernelAbiError::RequirementMismatch);
        }
        return Ok(KernelCall::exact(
            if run.kernel == MidOperationKind::LayerNorm {
                "layer_norm_f16"
            } else {
                "add_layer_norm_f16"
            },
            vec![element_count(&run.outputs[0].extents)? / width, width],
        ));
    }
    if run.kernel == MidOperationKind::AddLayerNormMoments
        && (run.inputs[0].extents != run.inputs[1].extents
            || run.outputs[1].extents != run.inputs[0].extents
            || run.requirements.outputs[1].format != run.requirements.inputs[0].format
            || run.requirements.inputs[1].format != run.requirements.inputs[0].format)
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let extent = run.inputs[0]
        .extents
        .last()
        .ok_or(KernelAbiError::RequirementMismatch)?;
    let width = extent.logical_end - extent.start;
    let source_count = logical_elements(&run.inputs[0].extents)?;
    let rows = source_count
        .checked_div(width)
        .ok_or(KernelAbiError::RequirementMismatch)?;
    let statistics = rows
        .checked_mul(2)
        .ok_or(KernelAbiError::ElementCountOverflow)?;
    if !width.is_multiple_of(4)
        || extent.physical_end != extent.logical_end
        || run.requirements.inputs[0].format.precision != Precision::F16
        || run.requirements.inputs[0].format.layout.order != ElementOrder::RowMajor
        || run.requirements.outputs[0].format.layout.order != ElementOrder::RowMajor
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let (symbol, arguments) = match run.kernel {
        MidOperationKind::LayerNormMoments | MidOperationKind::AddLayerNormMoments => {
            if run.requirements.outputs[0].format.precision != Precision::F32
                || element_count(&run.outputs[0].extents)? != statistics
            {
                return Err(KernelAbiError::RequirementMismatch);
            }
            if run.kernel == MidOperationKind::LayerNormMoments {
                (
                    "layer_norm_moments",
                    vec![
                        element_count(&run.outputs[0].extents)?
                            / matrix_extent(&run.outputs[0], true, true)?,
                        width,
                    ],
                )
            } else {
                (
                    "add_layer_norm_moments",
                    vec![element_count(&run.inputs[0].extents)? / width, width],
                )
            }
        }
        MidOperationKind::LayerNormApply { parts } => {
            let statistics = statistics
                .checked_mul(u32::from(parts))
                .ok_or(KernelAbiError::ElementCountOverflow)?;
            if parts == 0
                || run.requirements.outputs[0].format.precision != Precision::F16
                || run.requirements.inputs[1..3]
                    .iter()
                    .any(|input| input.format.precision != Precision::F16)
                || run.requirements.inputs[3].format.precision != Precision::F32
                || logical_elements(&run.inputs[3].extents)? != statistics
            {
                return Err(KernelAbiError::RequirementMismatch);
            }
            let columns = matrix_extent(&run.outputs[0], true, true)?;
            (
                "layer_norm_apply",
                vec![
                    element_count(&run.outputs[0].extents)? / columns,
                    columns,
                    u32::from(parts),
                ],
            )
        }
        _ => unreachable!(),
    };
    Ok(KernelCall::exact(symbol, arguments))
}

fn logical_elements(extents: &[crate::ShardExtent]) -> Result<u32, KernelAbiError> {
    extents.iter().try_fold(1u32, |n, axis| {
        n.checked_mul(axis.logical_end - axis.start)
            .ok_or(KernelAbiError::ElementCountOverflow)
    })
}
