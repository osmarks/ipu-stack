//! Useful issue-cycle equivalents, deliberately separate from latency costing.
//! No setup, address arithmetic, worker skew, or non-overlapped operand loads
//! enter arithmetic numerators. These are estimates, not instruction counters.
use super::*;
use crate::TileKernelSpec;

pub(super) fn work_estimate(work: crate::TileWorkRef<'_>) -> Option<(f64, f64, &'static str)> {
    let crate::TileWorkRef::Kernel(run) = work else {
        // Physical local copies may contain padding without a semantic view.
        return None;
    };
    let logical: u64 = run
        .output
        .extents
        .iter()
        .map(|e| u64::from(e.logical_end.saturating_sub(e.start)))
        .product();
    let physical: u64 = run
        .output
        .extents
        .iter()
        .map(|e| u64::from(e.physical_end - e.start))
        .product();
    let precision = run.requirements.output.format.precision;
    let (useful, issued, scale, basis) = match run.kernel {
        TileKernelSpec::Gemm { multiply, .. } => {
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
        TileKernelSpec::Gelu if precision == Precision::F16 => (
            logical,
            physical,
            3.0,
            "GeLU: 12 arithmetic issue slots / 4 elements",
        ),
        TileKernelSpec::Add => (
            logical,
            physical,
            match precision {
                Precision::F16 => 0.25,
                Precision::F32 => 0.5,
                _ => return None,
            },
            "vector add issue slots",
        ),
        TileKernelSpec::ReductionSum { partials } => {
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
        TileKernelSpec::Rearrange { .. } => (
            logical,
            physical,
            precision.bytes() as f64 / 4.0,
            "dense copy baseline: 8-byte load + store / 2 issue slots",
        ),
        TileKernelSpec::Cast { from, to }
            if matches!(
                (from, to),
                (Precision::F16, Precision::F32) | (Precision::F32, Precision::F16)
            ) =>
        {
            (logical, physical, 0.5, "one vector conversion / 2 elements")
        }
        TileKernelSpec::Cast {
            from: Precision::F16,
            to: Precision::F8F143 { .. },
        } => (
            logical,
            physical,
            0.125,
            "one vector conversion / 8 elements",
        ),
        TileKernelSpec::Cast {
            from: Precision::F8F143 { .. },
            to: Precision::F16,
        } => (
            logical,
            physical,
            0.25,
            "one vector conversion / 4 elements",
        ),
        TileKernelSpec::FillZero {
            padding_only: true, ..
        } => return Some((0.0, 0.0, "padding initialization: no useful tensor work")),
        TileKernelSpec::AttentionSoftmax {
            key_columns,
            padded_key_columns,
            ..
        } => {
            // Full panels: five maximum reductions, four MIXes and one
            // accumulator readout, eight exps, conversions, and sum updates.
            // The narrow tail retains the original eight-slot pair sequence.
            let rows = logical / u64::from(padded_key_columns + 16);
            let physical_rows = physical / u64::from(padded_key_columns + 16);
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
                    * (34.0 * f64::from(key_columns / 16)
                        + 4.0 * f64::from(key_columns % 16)
                        + row_reductions),
                physical_rows as f64
                    * (34.0 * f64::from(key_columns / 16)
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
    works: &[crate::TileWorkRef<'_>],
) {
    let mut useful = 0.0;
    let mut physical = 0.0;
    let mut basis = "";
    for &work in works {
        let Some((u, p, b)) = work_estimate(work) else {
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
