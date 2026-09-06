//! Constant-time cycle models for the current IPU21 kernel loops.
//! Inputs are physical local extents, not logical tensor sizes. No codegen is
//! performed here; update these models when changing the corresponding loops.

/// Word-pair transpose in unpack_transposed_amp_f16.S: six workers distribute
/// row pairs, with about fifteen issue slots per two columns. Physical geometry
/// does not retain logical tail masks, whose branches add some extra work.
pub(crate) fn f16_transposed_unpack_cycles(matrices: u64, rows: u64, columns: u64) -> u64 {
    300u64.saturating_add(
        matrices
            .saturating_mul(rows.div_ceil(12))
            .saturating_mul(100u64.saturating_add(columns.div_ceil(2).saturating_mul(90))),
    )
}

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

/// Packed stores balance rows across the six workers and split their ranges
/// at 16-row panel boundaries. Six scalar iterations model worker occupancy
/// without expanding tiles or emitting instructions.
pub(crate) fn f16_packed_gemm_cycles(
    rows: u64,
    inner: u64,
    columns: u64,
    interleaved: bool,
) -> u64 {
    let worker = (0..6)
        .map(|worker| {
            let count = rows / 6 + u64::from(worker < rows % 6);
            if count == 0 {
                return 0;
            }
            let start = worker * (rows / 6) + worker.min(rows % 6);
            let chunks = (start % 16 + count).div_ceil(16);
            count
                .saturating_mul(24)
                .saturating_add(chunks.saturating_mul(184))
        })
        .max()
        .unwrap_or(0);
    let group = worker.saturating_add(if interleaved { 170 } else { 202 });
    342u64.saturating_add(
        inner
            .div_ceil(16)
            .saturating_mul(columns.div_ceil(16))
            .saturating_mul(group),
    )
}

/// Row-wise softmax: packed maxima and fused exponent/store/sum take 71
/// issue groups per full 16-key panel. Masked pairs and zero padding use short
/// scalar loops; no tile program needs to be constructed to price them.
pub(crate) fn f16_softmax_cycles(rows: u64, keys: u64, padded_keys: u64) -> u64 {
    if rows == 0 {
        return 0;
    }
    let full_panels = keys / 16;
    let mut row = 31u64.saturating_add(full_panels.saturating_mul(71));
    let launch = if keys == padded_keys {
        216u64
    } else {
        let pairs = (keys % 16) / 2;
        let zero_pairs = 8 - (keys % 16).div_ceil(2);
        let zero_panels = (padded_keys / 16).saturating_sub(full_panels.saturating_add(1));
        row = row
            .saturating_add(21)
            .saturating_add(if full_panels != 0 { 4 } else { 2 })
            .saturating_add(2 * u64::from(pairs != 0))
            .saturating_add(12 * pairs + 12 * (keys % 2))
            .saturating_add(u64::from(zero_pairs != 0) + 2 * zero_pairs)
            .saturating_add(u64::from(zero_panels != 0))
            .saturating_add(zero_panels.saturating_mul(10));
        234
    };
    launch.saturating_add(rows.div_ceil(6).saturating_mul(6).saturating_mul(row))
}

/// Merge preserves FP32 state. Pair loops issue four groups for initialization
/// and six for updates; final normalization is folded into the row coefficients.
pub(crate) fn f16_attention_merge_cycles(
    rows: u64,
    values: u64,
    initial: bool,
    final_block: bool,
) -> u64 {
    if rows == 0 {
        return 0;
    }
    let panels = values / 16;
    let row = (if initial { 27u64 } else { 34u64 })
        .saturating_add(u64::from(panels != 0))
        .saturating_add(panels.saturating_mul(3))
        .saturating_add(
            values
                .div_ceil(2)
                .saturating_mul(if initial { 4 } else { 6 }),
        )
        .saturating_add(if final_block {
            if initial { 3 } else { 5 }
        } else {
            0
        });
    222u64.saturating_add(rows.div_ceil(6).saturating_mul(6).saturating_mul(row))
}

/// reduce_add_f16.S: six workers, eight elements per iteration, eleven issue
/// groups (including repeat alignment) plus two per remote partial. Setup includes supervisor rendezvous.
/// A one-partial sum is an identity and requires no reduction invocation.
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

/// gelu_f16.S has a fast path for whole 16-element blocks. Other even lengths
/// use 88-issue-group blocks and a 20-issue-group scalar-pair tail. Evaluate
/// at most six worker spans, independent of tensor size or tile count. The
/// tail setup conservatively covers the final worker's six-cycle exit skew.
pub(crate) fn f16_gelu_cycles(elements: u64) -> u64 {
    if elements == 0 {
        return 0;
    }
    if elements.is_multiple_of(16) {
        return 330u64.saturating_add(elements.div_ceil(96).saturating_mul(510));
    }
    (0..6)
        .map(|worker| {
            let pairs = elements.div_ceil(2).saturating_sub(worker * 8);
            if pairs == 0 {
                return 0;
            }
            let blocks = if pairs < 8 { 0 } else { (pairs - 8) / 48 + 1 };
            let tail = pairs.saturating_sub(blocks.saturating_mul(48));
            blocks
                .saturating_mul(528)
                .saturating_add(tail.saturating_mul(120))
                .saturating_add(if tail == 0 { 330 } else { 342 })
        })
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_row_models_match_hardware() {
        for rows in [7, 8] {
            assert_eq!(f16_softmax_cycles(rows, 64, 64), 3996);
            assert_eq!(f16_softmax_cycles(rows, 25, 64), 2838);
            assert_eq!(f16_attention_merge_cycles(rows, 72, true, false), 2430);
            assert_eq!(f16_attention_merge_cycles(rows, 72, false, false), 3378);
            assert_eq!(f16_attention_merge_cycles(rows, 72, false, true), 3438);
        }
        assert_eq!(f16_softmax_cycles(0, 64, 64), 0);
        assert_eq!(f16_softmax_cycles(u64::MAX, u64::MAX, u64::MAX), u64::MAX);
        assert_eq!(
            f16_attention_merge_cycles(u64::MAX, u64::MAX, false, true),
            u64::MAX
        );
    }

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
        assert_eq!(f16_reduction_cycles(48, 1), 0);
        assert_eq!(f16_reduction_cycles(0, 4), 0);
        assert_eq!(f16_reduction_cycles(u64::MAX, u64::MAX), u64::MAX);
    }

    #[test]
    fn gelu_tracks_hardware_for_blocks_and_pair_tails() {
        for (elements, measured) in [
            (2, 462),
            (6, 702),
            (14, 1182),
            (16, 840),
            (18, 858),
            (30, 1182),
            (94, 1176),
            (96, 840),
            (98, 990),
            (1408, 7980),
            (2208, 12060),
        ] {
            let predicted = f16_gelu_cycles(elements);
            assert!(predicted >= measured && predicted - measured <= 6);
        }
        assert_eq!(f16_gelu_cycles(0), 0);
        assert_eq!(f16_gelu_cycles(u64::MAX), u64::MAX);
    }

    #[test]
    fn packed_gemm_tracks_panel_boundaries_and_weight_memory() {
        // Device profiles: projected attention and GEMM with worker ranges
        // crossing multiple 16-row output panels.
        for (rows, inner, columns, interleaved, measured) in [
            (48, 240, 64, false, 35022u64),
            (48, 240, 64, true, 33102),
            (96, 64, 64, true, 12108),
            (128, 64, 64, true, 19944),
        ] {
            let predicted = f16_packed_gemm_cycles(rows, inner, columns, interleaved);
            assert!(predicted.abs_diff(measured) * 100 < measured);
        }
        assert_eq!(
            f16_packed_gemm_cycles(u64::MAX, u64::MAX, u64::MAX, true),
            u64::MAX
        );
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
