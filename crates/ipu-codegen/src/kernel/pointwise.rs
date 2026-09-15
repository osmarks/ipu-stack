//! Local Add and GeLU contracts. These kernels consume dense physical streams;
//! broadcast repetition and supported output epilogues are explicit here.

use super::*;
use crate::mid::MidOperationKind;

pub(super) const GELU_FP8_OUTPUT: OutputCapability = OutputCapability {
    complete_rows: false,
    input_order: ElementOrder::RowMajor,
    output_orders: &[ElementOrder::RowMajor, ElementOrder::Amp(AmpOrder::Left)],
    column_multiple: 4,
};

pub(super) fn call(
    kernel: &MidOperationKind,
    inputs: &[TensorStorage<'_>],
    outputs: &[TensorStorage<'_>],
) -> Result<KernelCall, KernelError> {
    check_arity(
        inputs,
        outputs,
        if *kernel == MidOperationKind::Gelu {
            1
        } else {
            2
        },
        1,
    )?;
    let output = outputs[0];
    if let Some(mut arguments) = fp8_arguments(kernel, inputs, output)? {
        let symbol = if *kernel == MidOperationKind::BiasGelu {
            if inputs[1].count()? != inputs[0].matrix_extent(true, true)? {
                return Err(KernelError::RequirementMismatch);
            }
            "bias_gelu_f8"
        } else {
            arguments.push(inputs[0].matrix_extent(false, true)?);
            "gelu_f8"
        };
        arguments.push(output.matrix_extent(false, true)?);
        return Ok(KernelCall::exact(symbol, arguments));
    }
    let count = output.count()?;
    match kernel {
        MidOperationKind::Gelu => {
            if inputs[0].format.precision != Precision::F16
                || output.format.precision != Precision::F16
                || inputs[0].format.layout != output.format.layout
            {
                return Err(KernelError::Unavailable(kernel.clone()));
            }
            let symbol = "gelu_tanh_approx_f16";
            if !count.is_multiple_of(2) {
                return Err(KernelError::UnsupportedElementCount {
                    symbol,
                    count,
                    divisor: 2,
                });
            }
            Ok(KernelCall::exact(symbol, vec![count]))
        }
        MidOperationKind::BiasGelu => {
            let width = f16_row_width(kernel, inputs, output)?;
            if inputs[0].extents != output.extents || inputs[1].count()? != width {
                return Err(KernelError::RequirementMismatch);
            }
            Ok(KernelCall::exact(
                "bias_gelu_f16",
                vec![count / width, width],
            ))
        }
        MidOperationKind::Add => {
            if output.format.precision != Precision::F16 {
                return Err(KernelError::Unavailable(kernel.clone()));
            }
            for &input in inputs {
                if input.extents.len() > output.extents.len() {
                    return Err(KernelError::RequirementMismatch);
                }
                let mut suffix = false;
                for (n, m) in input.widths().zip(
                    output
                        .widths()
                        .skip(output.extents.len() - input.extents.len()),
                ) {
                    suffix |= n != 1;
                    // The codelet repeats a contiguous suffix, not arbitrary strides.
                    if (suffix && n != m) || n == 0 {
                        return Err(KernelError::RequirementMismatch);
                    }
                }
            }
            Ok(KernelCall::exact(
                "add_f16",
                vec![count, inputs[0].count()?, inputs[1].count()?],
            ))
        }
        _ => Err(KernelError::RequirementMismatch),
    }
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

/// gelu_f16.S has a fast path for whole 16-element blocks. Other even lengths
/// use 80-issue-group blocks and a 20-issue-group scalar-pair tail. Evaluate
/// at most six worker spans, independent of tensor size or tile count. The
/// tail setup conservatively covers the final worker's six-cycle exit skew.
pub(super) fn gelu_row_cycles(elements: u64, bias: bool) -> u64 {
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
pub(crate) fn fp8_gelu_cycles(rows: u64, width: u64) -> u64 {
    138u64.saturating_add(
        rows.saturating_mul(372u64.saturating_add(width.div_ceil(192).saturating_mul(1002))),
    )
}

impl KernelBuildPlan {
    pub(super) fn add_pointwise(&mut self, exact_symbols: &BTreeSet<&'static str>) {
        if exact_symbols.contains("add_f16") {
            self.add_vertex(
                "elementwise_f16.cpp",
                "add_f16",
                "AddF16",
                vec!["-O2".into(), "-DVERTEX_AddF16".into()],
                &[3, 4, 2, 5, 6, 7],
                "worker_call.S",
                Vec::new(),
            );
        }
        for (source, symbol, extra) in [
            ("gelu_f16.S", "gelu_tanh_approx_f16", None),
            ("gelu_f16.S", "bias_gelu_f16", Some("-DGELU_WITH_BIAS")),
            ("gelu_f8.S", "gelu_f8", None),
            ("gelu_f8.S", "bias_gelu_f8", Some("-DGELU_WITH_BIAS")),
        ] {
            if exact_symbols.contains(symbol) {
                self.add_compilation(KernelCompilation {
                    source,
                    name: symbol.into(),
                    flags: extra.into_iter().map(str::to_owned).collect(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            let predicted = gelu_row_cycles(elements, false);
            assert!(predicted >= measured && predicted - measured <= 6);
        }
        assert_eq!(gelu_row_cycles(0, false), 0);
        assert_eq!(gelu_row_cycles(u64::MAX, false), u64::MAX);
    }
}
