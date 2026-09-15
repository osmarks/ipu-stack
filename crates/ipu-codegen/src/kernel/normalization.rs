//! Layer normalization and distributed statistics/apply call contracts.

use super::*;
use crate::mid::MidOperationKind;

pub(super) const FP8_OUTPUT: OutputCapability = OutputCapability {
    complete_rows: true,
    input_order: ElementOrder::RowMajor,
    output_orders: &[ElementOrder::RowMajor, ElementOrder::Amp(AmpOrder::Left)],
    column_multiple: 4,
};

pub(super) fn call(
    kernel: &MidOperationKind,
    inputs: &[TensorStorage<'_>],
    outputs: &[TensorStorage<'_>],
    build: Option<&mut KernelObjects>,
) -> Result<KernelCall, KernelError> {
    let arity = match kernel {
        MidOperationKind::LayerNorm => 3,
        MidOperationKind::AddLayerNorm | MidOperationKind::LayerNormApply { .. } => 4,
        MidOperationKind::LayerNormMoments => 1,
        MidOperationKind::AddLayerNormMoments => 2,
        _ => return Err(KernelError::RequirementMismatch),
    };
    check_arity(
        inputs,
        outputs,
        arity,
        1 + usize::from(*kernel == MidOperationKind::AddLayerNormMoments),
    )?;
    let input = inputs[0];
    let output = outputs[0];
    if let Some(arguments) = fp8_arguments(kernel, inputs, output)? {
        if let Some(build) = build {
            build.add_vertex(
                "elementwise_f16.cpp",
                "layer_norm_f8",
                "LayerNormF16",
                vec![
                    "-O2".into(),
                    "-DVERTEX_LayerNormF16".into(),
                    "-DNORM_FP8".into(),
                ],
                &[],
                "layer_norm_f16.S",
                vec!["-DNORM_FP8".into()],
            );
        }
        let columns = input.matrix_extent(false, true)?;
        return Ok(KernelCall::new(
            "layer_norm_f8",
            arguments,
            fp8_layernorm_cycles(
                (input.count()? / columns).into(),
                columns.into(),
                output.format.layout.order == ElementOrder::Amp(AmpOrder::Left),
            ),
        ));
    }
    let output_elements = output.count()?;
    if matches!(
        kernel,
        MidOperationKind::LayerNorm | MidOperationKind::AddLayerNorm
    ) {
        let width = f16_row_width(kernel, inputs, output)?;
        if output.matrix_extent(false, true)? != width
            || (*kernel == MidOperationKind::AddLayerNorm
                && (input.extents != output.extents || inputs[1].extents != output.extents))
        {
            return Err(KernelError::RequirementMismatch);
        }
        let add = *kernel == MidOperationKind::AddLayerNorm;
        let symbol = if add {
            "add_layer_norm_f16"
        } else {
            "layer_norm_f16"
        };
        if let Some(build) = build {
            let flags: Vec<_> = add
                .then_some("-DNORM_WITH_ADD".into())
                .into_iter()
                .collect();
            let mut codelet = vec!["-O2".into(), "-DVERTEX_LayerNormF16".into()];
            codelet.extend(flags.iter().cloned());
            build.add_vertex(
                "elementwise_f16.cpp",
                symbol,
                "LayerNormF16",
                codelet,
                &[],
                "layer_norm_f16.S",
                flags,
            );
        }
        return Ok(KernelCall::new(
            symbol,
            vec![output_elements / width, width],
            layernorm_cycles((output_elements / width).into(), width.into(), add, true),
        ));
    }
    if *kernel == MidOperationKind::AddLayerNormMoments
        && (input.extents != inputs[1].extents
            || outputs[1].extents != input.extents
            || outputs[1].format != input.format
            || inputs[1].format != input.format)
    {
        return Err(KernelError::RequirementMismatch);
    }
    let width = input.matrix_extent(true, true)?;
    let source_count = input.logical_elements()?;
    let rows = source_count
        .checked_div(width)
        .ok_or(KernelError::RequirementMismatch)?;
    let statistics = rows
        .checked_mul(2)
        .ok_or(KernelError::ElementCountOverflow)?;
    if !width.is_multiple_of(4)
        || input.matrix_extent(false, true)? != width
        || input.format.precision != Precision::F16
        || input.format.layout.order != ElementOrder::RowMajor
        || output.format.layout.order != ElementOrder::RowMajor
    {
        return Err(KernelError::RequirementMismatch);
    }
    let call = match kernel {
        MidOperationKind::LayerNormMoments | MidOperationKind::AddLayerNormMoments => {
            if output.format.precision != Precision::F32
                || output_elements != statistics
                || output.matrix_extent(false, true)? != 2
                || (*kernel == MidOperationKind::AddLayerNormMoments
                    && input.count()? != source_count)
            {
                return Err(KernelError::RequirementMismatch);
            }
            let add = *kernel == MidOperationKind::AddLayerNormMoments;
            let symbol = if add {
                "add_layer_norm_moments"
            } else {
                "layer_norm_moments"
            };
            if let Some(build) = build {
                let flags: Vec<_> = add
                    .then_some("-DNORM_STORE_SUM".into())
                    .into_iter()
                    .collect();
                let mut codelet = vec!["-O2".into(), "-DVERTEX_LayerNormMoments".into()];
                codelet.extend(flags.iter().cloned());
                build.add_vertex(
                    "layer_norm_distributed.cpp",
                    symbol,
                    "LayerNormMoments",
                    codelet,
                    &[],
                    "layer_norm_moments.S",
                    flags,
                );
            }
            KernelCall::new(
                symbol,
                vec![rows, width],
                f16_layernorm_moments_cycles(rows.into(), width.into(), add),
            )
        }
        MidOperationKind::LayerNormApply { parts } => {
            let statistics = statistics
                .checked_mul(u32::from(*parts))
                .ok_or(KernelError::ElementCountOverflow)?;
            if *parts == 0
                || output.format.precision != Precision::F16
                || inputs[1..3]
                    .iter()
                    .any(|input| input.format.precision != Precision::F16)
                || inputs[3].format.precision != Precision::F32
                || inputs[3].logical_elements()? != statistics
                || output.matrix_extent(true, true)? != width
                || output_elements != source_count
            {
                return Err(KernelError::RequirementMismatch);
            }
            {
                if let Some(build) = build {
                    build.add_vertex(
                        "layer_norm_distributed.cpp",
                        "layer_norm_apply",
                        "LayerNormApply",
                        vec!["-O2".into(), "-DVERTEX_LayerNormApply".into()],
                        &[3, 4, 5, 6, 2, 7, 8, 9],
                        "worker_call.S",
                        Vec::new(),
                    );
                }
                KernelCall::new(
                    "layer_norm_apply",
                    vec![rows, width, u32::from(*parts)],
                    f16_layernorm_apply_cycles(rows.into(), width.into(), *parts),
                )
            }
        }
        _ => return Err(KernelError::RequirementMismatch),
    };
    Ok(call)
}

/// Mean and centered variance retain FP32 precision. Aligned full groups use
/// pipelined F16V8ACC (2 bundles / 8 values) and F32V4SQACC (5 / 4). The fused variant
/// adds three and two bundles respectively to read/add the residual operand.
fn norm_statistics_work(width: u64, add: bool) -> u64 {
    if width >= 96 && width.is_multiple_of(8) {
        width
            .div_ceil(48)
            .saturating_mul(if add { 30 } else { 12 })
            .saturating_add(width.div_ceil(24).saturating_mul(if add { 42 } else { 30 }))
            .saturating_add(24) // Prime/drain both passes.
    } else {
        width.div_ceil(12).saturating_mul(if add { 72 } else { 48 })
    }
}

/// Three worker launches per row; shared setup constants include partial
/// reductions and the scalar inverse standard deviation. FP16 output uses
/// twelve bundles per quad (fourteen with add); FP8 retains the pair path.
pub(super) fn layernorm_cycles(rows: u64, width: u64, add: bool, quad_affine: bool) -> u64 {
    let setup = if width >= 96 && width.is_multiple_of(8) {
        if add { 1386 } else { 1329 }
    } else if add {
        1290
    } else {
        1200
    };
    let apply = if quad_affine && width.is_multiple_of(4) {
        66u64.saturating_add(width.div_ceil(24).saturating_mul(if add { 84 } else { 72 }))
    } else {
        width.div_ceil(12).saturating_mul(if add { 72 } else { 60 })
    };
    let work = norm_statistics_work(width, add).saturating_add(apply);
    132u64.saturating_add(rows.saturating_mul(work.saturating_add(setup)))
}

pub(crate) fn f16_layernorm_moments_cycles(rows: u64, width: u64, add: bool) -> u64 {
    126u64
        .saturating_add(
            rows.saturating_mul(norm_statistics_work(width, false).saturating_add(1080)),
        )
        .saturating_add(if add {
            rows.saturating_mul(108 + width.div_ceil(48) * 24)
        } else {
            0
        })
}

/// Final moments merging is small but repeats in each worker. These costs
/// cover local application only; the mid copy prices the statistics exchange.
pub(crate) fn f16_layernorm_apply_cycles(rows: u64, width: u64, parts: u16) -> u64 {
    let (setup, row_setup, work) = if width.is_multiple_of(4) {
        (774u64, 402u64, width.div_ceil(24).saturating_mul(72))
    } else {
        (558, 324, width.div_ceil(12).saturating_mul(60))
    };
    setup.saturating_add(
        rows.saturating_mul((row_setup + u64::from(parts) * 54).saturating_add(work)),
    )
}

pub(crate) fn fp8_layernorm_cycles(rows: u64, width: u64, packed: bool) -> u64 {
    layernorm_cycles(rows, width, false, false).saturating_add(rows.saturating_mul(
        270u64.saturating_add(width.div_ceil(24).saturating_mul(if packed && rows > 1 {
            162
        } else {
            6
        })),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::pointwise::f16_add_cycles;

    #[test]
    fn elementwise_models_track_device_loops_and_row_setup() {
        // Independent direct-kernel measurements from elementwise_check.
        for (rows, width, norm, fused, moments, apply) in [
            (1, 144, 2196u64, 2454u64, 1464u64, 1620u64),
            (1, 576, 4140, 4992, 2112, 2916),
            (1, 1152, 6732, 8376, 2976, 4644),
            (3, 1152, 19938, 24864, 8676, 12468),
        ] {
            for (estimated, measured) in [
                (layernorm_cycles(rows, width, false, true), norm),
                (layernorm_cycles(rows, width, true, true), fused),
                (f16_layernorm_moments_cycles(rows, width, false), moments),
                (f16_layernorm_apply_cycles(rows, width, 1), apply),
            ] {
                assert!(
                    estimated.abs_diff(measured) < measured / 20 + 24,
                    "rows={rows} width={width}: estimate {estimated}, measured {measured}"
                );
            }
        }
        assert_eq!(f16_add_cycles(1728, 1728, 1728), 1800);
        assert_eq!(f16_add_cycles(3456, 3456, 1152), 3564);
        assert_eq!(layernorm_cycles(u64::MAX, u64::MAX, true, true), u64::MAX);
        assert_eq!(
            f16_layernorm_moments_cycles(u64::MAX, u64::MAX, false),
            u64::MAX
        );
        assert_eq!(f16_layernorm_apply_cycles(u64::MAX, u64::MAX, 2), u64::MAX);
    }
}
