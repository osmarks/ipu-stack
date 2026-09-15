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
) -> Result<KernelCall, KernelAbiError> {
    let arity = match kernel {
        MidOperationKind::LayerNorm => 3,
        MidOperationKind::AddLayerNorm | MidOperationKind::LayerNormApply { .. } => 4,
        MidOperationKind::LayerNormMoments => 1,
        MidOperationKind::AddLayerNormMoments => 2,
        _ => return Err(KernelAbiError::RequirementMismatch),
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
        return Ok(KernelCall::exact("layer_norm_f8", arguments));
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
            return Err(KernelAbiError::RequirementMismatch);
        }
        return Ok(KernelCall::exact(
            if *kernel == MidOperationKind::LayerNorm {
                "layer_norm_f16"
            } else {
                "add_layer_norm_f16"
            },
            vec![output_elements / width, width],
        ));
    }
    if *kernel == MidOperationKind::AddLayerNormMoments
        && (input.extents != inputs[1].extents
            || outputs[1].extents != input.extents
            || outputs[1].format != input.format
            || inputs[1].format != input.format)
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let width = input.matrix_extent(true, true)?;
    let source_count = input.logical_elements()?;
    let rows = source_count
        .checked_div(width)
        .ok_or(KernelAbiError::RequirementMismatch)?;
    let statistics = rows
        .checked_mul(2)
        .ok_or(KernelAbiError::ElementCountOverflow)?;
    if !width.is_multiple_of(4)
        || input.matrix_extent(false, true)? != width
        || input.format.precision != Precision::F16
        || input.format.layout.order != ElementOrder::RowMajor
        || output.format.layout.order != ElementOrder::RowMajor
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let (symbol, arguments) = match kernel {
        MidOperationKind::LayerNormMoments | MidOperationKind::AddLayerNormMoments => {
            if output.format.precision != Precision::F32
                || output_elements != statistics
                || output.matrix_extent(false, true)? != 2
                || (*kernel == MidOperationKind::AddLayerNormMoments
                    && input.count()? != source_count)
            {
                return Err(KernelAbiError::RequirementMismatch);
            }
            (
                if *kernel == MidOperationKind::LayerNormMoments {
                    "layer_norm_moments"
                } else {
                    "add_layer_norm_moments"
                },
                vec![rows, width],
            )
        }
        MidOperationKind::LayerNormApply { parts } => {
            let statistics = statistics
                .checked_mul(u32::from(*parts))
                .ok_or(KernelAbiError::ElementCountOverflow)?;
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
                return Err(KernelAbiError::RequirementMismatch);
            }
            ("layer_norm_apply", vec![rows, width, u32::from(*parts)])
        }
        _ => return Err(KernelAbiError::RequirementMismatch),
    };
    Ok(KernelCall::exact(symbol, arguments))
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

impl KernelBuildPlan {
    pub(super) fn add_normalization(&mut self, exact_symbols: &BTreeSet<&'static str>) {
        for (source, wrapper, vertex, variants) in [
            (
                "elementwise_f16.cpp",
                "layer_norm_f16.S",
                "LayerNormF16",
                &[
                    ("layer_norm_f16", None),
                    ("add_layer_norm_f16", Some("-DNORM_WITH_ADD")),
                    ("layer_norm_f8", Some("-DNORM_FP8")),
                ][..],
            ),
            (
                "layer_norm_distributed.cpp",
                "layer_norm_moments.S",
                "LayerNormMoments",
                &[
                    ("layer_norm_moments", None),
                    ("add_layer_norm_moments", Some("-DNORM_STORE_SUM")),
                ][..],
            ),
        ] {
            for &(symbol, extra) in variants {
                if !exact_symbols.contains(symbol) {
                    continue;
                }
                let flags: Vec<_> = extra.into_iter().map(str::to_owned).collect();
                let mut codelet_flags = vec!["-O2".into(), format!("-DVERTEX_{vertex}")];
                codelet_flags.extend(flags.iter().cloned());
                self.add_vertex(source, symbol, vertex, codelet_flags, &[], wrapper, flags);
            }
        }
        if exact_symbols.contains("layer_norm_apply") {
            self.add_vertex(
                "layer_norm_distributed.cpp",
                "layer_norm_apply",
                "LayerNormApply",
                vec!["-O2".into(), "-DVERTEX_LayerNormApply".into()],
                &[3, 4, 5, 6, 2, 7, 8, 9],
                "worker_call.S",
                Vec::new(),
            );
        }
    }
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
