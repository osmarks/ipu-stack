//! Constant-time cycle models for the current IPU21 kernel loops.
//! Inputs are physical local extents, not logical tensor sizes. No codegen is
//! performed here; update these models when changing the corresponding loops.

/// Interleaved F16 AMP K16/C16 group: four issue cycles per row plus retained
/// worker/weight-feed overhead, calibrated against device/gemm_f16_amp.S.
pub(crate) fn f16_gemm_microgroup_cycles(rows: u64) -> u64 {
    rows.saturating_mul(4).saturating_add(160)
}

pub(crate) fn interleaved_f16_gemm_cycles(rows: u64, inner: u64, columns: u64) -> u64 {
    294u64.saturating_add(
        inner
            .div_ceil(16)
            .saturating_mul(columns.div_ceil(16))
            .saturating_mul(f16_gemm_microgroup_cycles(rows)),
    )
}

/// reduce_add_f16.S: six workers, eight elements per iteration, nine issue
/// groups plus six per remote partial. Setup includes supervisor rendezvous.
/// A one-partial sum is an identity and requires no reduction invocation.
pub(crate) fn f16_reduction_cycles(elements: u64, partials: u64) -> u64 {
    if elements == 0 || partials < 2 {
        return 0;
    }
    282u64.saturating_add(
        elements
            .div_ceil(48)
            .saturating_mul(6)
            .saturating_mul(9u64.saturating_add(partials.saturating_sub(1).saturating_mul(6))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduction_matches_unmerged_hardware_samples() {
        // Independent IPU21 sweep measurements, including uneven worker loads.
        for (elements, partials, measured) in [
            (1408, 2, 2982),
            (2208, 4, 7734),
            (1472, 4, 5304),
            (576, 15, 6978),
            (672, 15, 8094),
            (2304, 24, 42618),
        ] {
            assert_eq!(f16_reduction_cycles(elements, partials), measured);
        }
        assert_eq!(f16_reduction_cycles(48, 1), 0);
        assert_eq!(f16_reduction_cycles(0, 4), 0);
        assert_eq!(f16_reduction_cycles(u64::MAX, u64::MAX), u64::MAX);
    }

    #[test]
    fn gemm_tracks_hardware_across_row_and_group_counts() {
        for (rows, inner, columns, measured) in [
            (184, 288, 48, 48654u64),
            (182, 288, 48, 48180),
            (184, 288, 32, 32538),
            (146, 576, 32, 53718),
            (120, 80, 272, 55344),
        ] {
            let predicted = interleaved_f16_gemm_cycles(rows, inner, columns);
            assert!(predicted.abs_diff(measured) * 100 < measured * 2);
        }
        assert_eq!(
            interleaved_f16_gemm_cycles(u64::MAX, u64::MAX, u64::MAX),
            u64::MAX
        );
    }
}
