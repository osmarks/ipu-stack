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

/// Paired-row coefficient packing. Complete 16-column panels use an unrolled
/// transpose; other aligned widths retain column indexing and bounds checks.
/// Six worker contexts distribute pairs, including the physical zero-padding.
pub(crate) fn f16_coefficient_pack_cycles(matrices: u64, rows: u64, columns: u64) -> u64 {
    let pair = if columns == 16 {
        60
    } else {
        12 + columns.div_ceil(16) * 4 * 35
    };
    300u64.saturating_add(
        matrices
            .saturating_mul(rows.div_ceil(12))
            .saturating_mul(6)
            .saturating_mul(pair),
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

/// Row-wise softmax: four-wide maxima and pipelined MIX/exp/store/sum take 41
/// issue groups per full 16-key panel. Masked pairs and zero padding use short
/// scalar loops; no tile program needs to be constructed to price them.
fn f16_softmax_whole_rows(rows: u64, keys: u64, padded_keys: u64) -> u64 {
    if rows == 0 {
        return 0;
    }
    let full_panels = keys / 16;
    let mut row = 31u64
        .saturating_add(2 * u64::from(full_panels != 0))
        .saturating_add(full_panels.saturating_mul(41));
    let launch = if keys == padded_keys {
        row = row.saturating_add(7);
        216u64
    } else {
        row = row.saturating_add(5 + 6 * u64::from(full_panels != 0));
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

// Three local stages: partial maxima, exponentials/partial sums, final sums.
// A segment has ceil(padded_keys / 48) panels. 130 groups account for each
// segment's setup, row-state reductions, and address calculations; 408 cycles
// cover the launches and 21 groups per final worker wave reduce the sums.
fn f16_softmax_split_cycles(rows: u64, keys: u64, padded_keys: u64) -> u64 {
    if keys < 128 || rows == 0 {
        return u64::MAX;
    }
    let segment = padded_keys
        .div_ceil(48)
        .saturating_mul(41)
        .saturating_add(130);
    408u64
        .saturating_add(rows.div_ceil(2).saturating_mul(6).saturating_mul(segment))
        .saturating_add(rows.div_ceil(6).saturating_mul(126))
}

/// The ABI and the planner use the same choice; no tile program is built here.
pub(crate) fn f16_softmax_split_rows(rows: u64, keys: u64, padded_keys: u64) -> bool {
    f16_softmax_split_cycles(rows, keys, padded_keys)
        < f16_softmax_whole_rows(rows, keys, padded_keys)
}

pub(crate) fn f16_softmax_cycles(rows: u64, keys: u64, padded_keys: u64) -> u64 {
    f16_softmax_whole_rows(rows, keys, padded_keys).min(f16_softmax_split_cycles(
        rows,
        keys,
        padded_keys,
    ))
}

/// The FP8 epilogue keeps the same row schedule, with two eight-value casts
/// per score panel, half-panel addressing and a bounded masked-tail drain.
pub(crate) fn softmax_output_cycles(rows: u64, keys: u64, padded_keys: u64, fp8: bool) -> u64 {
    let base = f16_softmax_cycles(rows, keys, padded_keys);
    if !fp8 {
        return base;
    }
    let split = f16_softmax_split_rows(rows, keys, padded_keys);
    base.saturating_add(
        rows.div_ceil(if split { 2 } else { 6 })
            .saturating_mul(6)
            .saturating_mul(
                padded_keys
                    .div_ceil(if split { 48 } else { 16 })
                    .saturating_mul(7)
                    .saturating_add(8),
            ),
    )
}

/// Merge preserves FP32 state. Pair loops issue four groups for initialization
/// and six for updates; final normalization is folded into the row coefficients.
pub(crate) fn f16_attention_merge_cycles(
    rows: u64,
    values: u64,
    initial: bool,
    final_block: bool,
    output_f16: bool,
) -> u64 {
    if rows == 0 {
        return 0;
    }
    let panels = values / 16;
    let row = (if initial { 27u64 } else { 34u64 })
        .saturating_add(if output_f16 { 3 } else { 0 })
        .saturating_add(u64::from(panels != 0))
        .saturating_add(panels.saturating_mul(3))
        .saturating_add(
            values
                .div_ceil(2)
                .saturating_mul((if initial { 4 } else { 6 }) + u64::from(output_f16)),
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
/// use 80-issue-group blocks and a 20-issue-group scalar-pair tail. Evaluate
/// at most six worker spans, independent of tensor size or tile count. The
/// tail setup conservatively covers the final worker's six-cycle exit skew.
pub(crate) fn f16_gelu_cycles(elements: u64) -> u64 {
    gelu_row_cycles(elements, false)
}

pub(crate) fn f16_bias_gelu_cycles(rows: u64, width: u64) -> u64 {
    if rows == 0 || width == 0 {
        return 0;
    }
    // Worker setup repeats per row; supervisor entry and accumulator/TAS setup
    // occur once. MIX retains the coefficients, freeing ARF registers for x+b.
    // Allocation bases normally have eight-byte alignment. The kernel checks
    // offset views at runtime and falls back to the narrow loop when needed.
    let row = if width.is_multiple_of(4) {
        450u64.saturating_add(width.div_ceil(24).saturating_mul(90))
    } else {
        gelu_row_cycles(width, true).saturating_add(36)
    };
    row.saturating_sub(222)
        .saturating_mul(rows)
        .saturating_add(222)
}

fn gelu_row_cycles(elements: u64, bias: bool) -> u64 {
    let (aligned_block, block, pair, setup): (u64, u64, u64, u64) = if bias {
        (516, 534, 138, 426)
    } else {
        (462, 480, 120, 330)
    };
    if elements == 0 {
        return 0;
    }
    if elements.is_multiple_of(16) {
        return setup.saturating_add(elements.div_ceil(96).saturating_mul(aligned_block));
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
                .saturating_mul(block)
                .saturating_add(tail.saturating_mul(pair))
                .saturating_add(if tail == 0 {
                    setup
                } else {
                    setup + if bias { 36 } else { 12 }
                })
        })
        .max()
        .unwrap_or(0)
}

/// Four-half add: two loads and one store per six-worker wave; arithmetic
/// overlaps the next load. Setup includes priming and draining the pipeline. Repeated suffix broadcasts reset the pointers per row.
/// Allocation bases are eight-byte aligned; irregular widths use the pair loop.
pub(crate) fn f16_add_cycles(elements: u64, left: u64, right: u64) -> u64 {
    let width = left.min(right);
    let dense = left == elements && right == elements;
    let broadcast =
        width > 0 && (left == elements || right == elements) && elements.is_multiple_of(width);
    if elements.is_multiple_of(4)
        && left.is_multiple_of(4)
        && right.is_multiple_of(4)
        && (dense || broadcast)
    {
        let rows = if dense { 1 } else { elements / width };
        let columns = if dense { elements } else { width };
        504u64
            .saturating_add(rows.saturating_mul(columns.div_ceil(24).saturating_mul(18)))
            .saturating_add(rows.saturating_sub(1).saturating_mul(234))
    } else {
        450u64.saturating_add(elements.div_ceil(12).saturating_mul(54))
    }
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
fn layernorm_cycles(rows: u64, width: u64, add: bool, quad_affine: bool) -> u64 {
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

pub(crate) fn f16_layernorm_cycles(rows: u64, width: u64, add: bool) -> u64 {
    layernorm_cycles(rows, width, add, true)
}

pub(crate) fn f16_layernorm_moments_cycles(rows: u64, width: u64) -> u64 {
    126u64.saturating_add(
        rows.saturating_mul(norm_statistics_work(width, false).saturating_add(1080)),
    )
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

/// MIX bias-GeLU plus FP8 conversion, with worker-distributed 32-value panels.
pub(crate) fn fp8_bias_gelu_cycles(rows: u64, width: u64, packed: bool) -> u64 {
    330u64.saturating_add(
        rows.saturating_mul(
            198u64
                .saturating_add(width.div_ceil(192).saturating_mul(1104))
                .saturating_add(if packed { 12 } else { 0 }),
        ),
    )
}

/// Pair conversion stays in ARF and emits complete FP8 words. Packed LN
/// needs separate address calculations when a tile owns several rows.
pub(crate) fn fp8_elementwise_cycles(gelu: bool, rows: u64, width: u64, packed: bool) -> u64 {
    if gelu {
        138u64.saturating_add(
            rows.saturating_mul(372u64.saturating_add(width.div_ceil(192).saturating_mul(1002))),
        )
    } else {
        layernorm_cycles(rows, width, false, false).saturating_add(rows.saturating_mul(
            270u64.saturating_add(width.div_ceil(24).saturating_mul(if packed && rows > 1 {
                162
            } else {
                6
            })),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                (f16_layernorm_cycles(rows, width, false), norm),
                (f16_layernorm_cycles(rows, width, true), fused),
                (f16_layernorm_moments_cycles(rows, width), moments),
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
        assert_eq!(f16_layernorm_cycles(u64::MAX, u64::MAX, true), u64::MAX);
        assert_eq!(f16_layernorm_moments_cycles(u64::MAX, u64::MAX), u64::MAX);
        assert_eq!(f16_layernorm_apply_cycles(u64::MAX, u64::MAX, 2), u64::MAX);
    }

    #[test]
    fn attention_row_models_match_hardware() {
        for rows in [7, 8] {
            assert_eq!(f16_softmax_cycles(rows, 64, 64), 2664);
            assert_eq!(f16_softmax_cycles(rows, 25, 64), 2634);
            assert_eq!(
                f16_attention_merge_cycles(rows, 72, true, false, false),
                2430
            );
            assert_eq!(
                f16_attention_merge_cycles(rows, 72, false, false, false),
                3378
            );
            assert_eq!(
                f16_attention_merge_cycles(rows, 72, false, true, false),
                3438
            );
        }
        assert_eq!(f16_softmax_cycles(0, 64, 64), 0);
        assert_eq!(f16_softmax_cycles(u64::MAX, u64::MAX, u64::MAX), u64::MAX);
        assert_eq!(
            f16_attention_merge_cycles(u64::MAX, u64::MAX, false, true, false),
            u64::MAX
        );
    }

    #[test]
    fn softmax_segmentation_prices_launches_and_worker_rounding() {
        assert!(f16_softmax_split_rows(1, 128, 128));
        assert!(!f16_softmax_split_rows(7, 128, 128));
        assert!(!f16_softmax_split_rows(8, 128, 128));
        assert!(!f16_softmax_split_rows(6, 729, 768));
        for rows in [7, 8] {
            assert!(f16_softmax_split_rows(rows, 729, 768));
            // Uniform-input device measurements: 20,226 / 20,244 cycles.
            assert_eq!(f16_softmax_cycles(rows, 729, 768), 19_524);
        }
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
    fn bias_gelu_tracks_mix_kernel_and_row_tails() {
        for (rows, width, measured) in [
            (1, 16, 540),
            (1, 94, 1458),
            (1, 98, 1170),
            (1, 1408, 5760),
            (1, 2152, 8550),
            (1, 2208, 8730),
            (3, 94, 3942),
            (3, 2152, 25206),
            (3, 2208, 25746),
        ] {
            assert!(f16_bias_gelu_cycles(rows, width).abs_diff(measured) <= 6 * rows);
        }
        assert_eq!(f16_bias_gelu_cycles(0, 2152), 0);
        assert_eq!(f16_bias_gelu_cycles(1, 0), 0);
    }

    #[test]
    fn gelu_tracks_hardware_for_blocks_and_pair_tails() {
        for (elements, measured) in [
            (2, 462),
            (6, 702),
            (14, 1182),
            (16, 792),
            (18, 810),
            (30, 1182),
            (94, 1182),
            (96, 792),
            (98, 942),
            (1408, 7260),
            (2208, 10956),
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
            (128, 64, 64, true, 19872),
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
            // 2026-09-13 hardware: shared workers retain stride and bypass
            // empty-row checks where the specialization proves nonempty.
            // FP8 K extents are halved to count the same physical AMP groups.
            (244, 64, 64, 18576u64),
            (82, 96, 160, 29508),
            (122, 80, 128, 26232),
            (81, 80, 80, 12402),
            (146, 48, 48, 7122),
            (56, 64, 80, 8076),
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
