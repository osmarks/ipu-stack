//! Useful issue-cycle equivalents, deliberately separate from latency costing.
//! No setup, address arithmetic, worker skew, or non-overlapped operand loads
//! enter arithmetic numerators. These are estimates, not instruction counters.
use super::*;
use crate::mid::MidOperationKind;

pub(super) fn work_estimate(run: &crate::KernelRun) -> Option<(f64, f64, &'static str)> {
    let logical: u64 = run.outputs[0]
        .extents
        .iter()
        .map(|e| u64::from(e.logical_end.saturating_sub(e.start)))
        .product();
    let physical: u64 = run.outputs[0]
        .extents
        .iter()
        .map(|e| u64::from(e.physical_end - e.start))
        .product();
    let precision = run.requirements.outputs[0].format.precision;
    // Usual allocation alignment permits the ACC/SQACC statistics loops.
    // Small/tail widths use the pair loops; addresses are not available here.
    let statistics_width = run
        .inputs
        .first()
        .and_then(|view| view.extents.last())
        .map_or(0, |extent| extent.physical_end - extent.start);
    let wide_statistics = statistics_width >= 96 && statistics_width.is_multiple_of(8);
    let statistics_issue_slots = if wide_statistics { 0.875 } else { 2.0 };
    let quad_affine = precision == Precision::F16 && statistics_width.is_multiple_of(4);
    let affine_issue_slots = if quad_affine { 1.5 } else { 2.0 };

    let (useful, issued, scale, basis) = match run.kernel {
        MidOperationKind::Gemm { multiply, .. } => {
            let [useful, physical] = run.product_flops?;
            let peak = match multiply {
                Precision::F16 => 128.0,
                Precision::F8F143 { .. } => 256.0,
                Precision::F32 => 32.0,
            };
            return Some((
                useful as f64 / peak,
                physical as f64 / peak,
                "AMP FLOPs / IPU21 peak",
            ));
        }
        // GELU_TWO_PAIRS: ten v4 arithmetic instructions and two v2 tanhs
        // per four values (including the finite-range clamp).
        MidOperationKind::Gelu => (
            logical,
            physical,
            if matches!(precision, Precision::F8F143 { .. }) {
                3.5
            } else {
                3.0
            },
            "GeLU arithmetic and optional FP8 conversion",
        ),
        MidOperationKind::LayerNormMoments | MidOperationKind::AddLayerNormMoments => {
            let elements: u64 = run.inputs[0]
                .extents
                .iter()
                .map(|e| u64::from(e.logical_end - e.start))
                .product();
            let slots = statistics_issue_slots
                + if run.kernel == MidOperationKind::AddLayerNormMoments {
                    0.25
                } else {
                    0.0
                };
            return Some((
                elements as f64 * slots,
                elements as f64 * slots,
                "layernorm: FP32 sum and centered squares, including vector accumulation",
            ));
        }
        MidOperationKind::LayerNormApply { .. } => (
            logical,
            physical,
            affine_issue_slots,
            "layernorm: normalization and affine arithmetic",
        ),
        // Vector path: bias add, clamp, square/cube, MIX, two tanhs and
        // three final arithmetic issues. GACC is data movement.
        MidOperationKind::BiasGelu => (
            logical,
            physical,
            if matches!(precision, Precision::F8F143 { .. }) {
                3.25
            } else {
                2.75
            },
            "bias add, MIX GeLU and optional FP8 conversion",
        ),
        MidOperationKind::AddLayerNorm => (
            logical,
            physical,
            statistics_issue_slots
                + if wide_statistics { 0.5 } else { 1.0 }
                + affine_issue_slots
                + if quad_affine { 0.25 } else { 0.5 },
            "residual add and layernorm arithmetic",
        ),
        MidOperationKind::LayerNorm => (
            logical,
            physical,
            statistics_issue_slots
                + affine_issue_slots
                + if matches!(precision, Precision::F8F143 { .. }) {
                    0.5
                } else {
                    0.0
                },
            "layernorm: FP32 statistics/normalization and affine arithmetic",
        ),
        MidOperationKind::Add => (
            logical,
            physical,
            match precision {
                Precision::F16 => 0.25,
                Precision::F32 => 0.5,
                _ => return None,
            },
            "vector add issue slots",
        ),
        MidOperationKind::ReductionSum { partials } => {
            let lanes = match precision {
                Precision::F16 => 4.0,
                Precision::F32 => 2.0,
                _ => return None,
            };
            (
                logical,
                physical,
                f64::from(partials.saturating_sub(1)) / lanes,
                "vector reduction add issue slots",
            )
        }
        MidOperationKind::Rearrange { .. } => (
            logical,
            physical,
            precision.bytes() as f64 / 4.0,
            "dense copy baseline: 8-byte load + store / 2 issue slots",
        ),
        MidOperationKind::Cast { from, to }
            if matches!(
                (from, to),
                (Precision::F16, Precision::F32) | (Precision::F32, Precision::F16)
            ) =>
        {
            (logical, physical, 0.5, "one vector conversion / 2 elements")
        }
        MidOperationKind::Cast {
            from: Precision::F16,
            to: Precision::F8F143 { .. },
        } => (
            logical,
            physical,
            0.125,
            "one vector conversion / 8 elements",
        ),
        MidOperationKind::Cast {
            from: Precision::F8F143 { .. },
            to: Precision::F16,
        } => (
            logical,
            physical,
            0.25,
            "one vector conversion / 4 elements",
        ),
        MidOperationKind::FillZero {
            padding_only: true, ..
        } => return Some((0.0, 0.0, "padding initialization: no useful tensor work")),
        MidOperationKind::AttentionSoftmax {
            key_columns,
            padded_key_columns,
            ..
        } => {
            // Full panels: five maximum reductions, four MIXes and one
            // accumulator readout, eight exps, eight FP16 additions, one
            // conversion and two FP32 additions: 29 arithmetic issue slots.
            // The narrow tail retains the original eight-slot pair sequence.
            let fp8 = matches!(precision, Precision::F8F143 { .. });
            let state = if fp8 { 64 } else { 16 };
            let rows = logical / u64::from(padded_key_columns + state);
            let physical_rows = physical / u64::from(padded_key_columns + state);
            let panel_slots = if fp8 { 31.0 } else { 29.0 };
            let row_reductions = if crate::kernel::cost::f16_softmax_split_rows(
                physical_rows,
                u64::from(key_columns),
                u64::from(padded_key_columns),
            ) {
                15.0
            } else {
                5.0
            };
            return Some((
                rows as f64
                    * (panel_slots * f64::from(key_columns / 16)
                        + 4.0 * f64::from(key_columns % 16)
                        + row_reductions),
                physical_rows as f64
                    * (panel_slots * f64::from(key_columns / 16)
                        + 8.0 * f64::from((key_columns % 16).div_ceil(2))
                        + row_reductions),
                "softmax: arithmetic/conversion issue slots, excluding loads and row addressing",
            ));
        }
        // Merge has scalar row-state work as well as vector arithmetic.
        // Leave unavailable until its arithmetic model is specified.
        _ => return None,
    };
    Some((useful as f64 * scale, issued as f64 * scale, basis))
}

pub(super) fn append_work_estimate(
    metadata: &mut Vec<ProfileMetadata>,
    program: &LowProgram,
    works: &[crate::BlockOperation<usize>],
) {
    let mut useful = 0.0;
    let mut physical = 0.0;
    let mut basis = "";
    for work in works {
        let crate::BlockOperation::Compute { run, .. } = work else {
            return;
        };
        let Some((u, p, b)) = work_estimate(&program.kernel_runs[run.0 as usize]) else {
            return;
        };
        useful += u;
        physical += p;
        basis = b;
    }
    metadata.extend([
        ProfileMetadata {
            name: "usefulCycles".into(),
            value: useful.to_string(),
        },
        ProfileMetadata {
            name: "physicalWorkCycles".into(),
            value: physical.to_string(),
        },
        ProfileMetadata {
            name: "workEstimateBasis".into(),
            value: basis.into(),
        },
    ]);
}
