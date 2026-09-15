//! GEMM assembly specializations, paired by physical row count.

use super::*;
use crate::mid::MidOperationKind;

use serde::{Deserialize, Serialize};

pub(super) fn cycles(implementation: &KernelImplementation) -> u64 {
    let KernelImplementation::Gemm(multiply, weights, inner_block, output_columns, _, rows, group) =
        *implementation
    else {
        return u64::MAX;
    };
    let rows = u64::from(rows);
    let columns = u64::from(output_columns);
    // Native FP8 uses the same instruction sequence for 32 K
    // elements that the F16 kernel uses for 16.
    let inner = u64::from(inner_block).div_ceil(if matches!(multiply, Precision::F8F143 { .. }) {
        2
    } else {
        1
    });
    let interleaved = weights == GemmWeightLoad::Interleaved;
    if multiply == Precision::F16 && group != 0 {
        return f16_packed_gemm_cycles(rows, inner, columns, interleaved);
    }
    if multiply != Precision::F32 && interleaved {
        return interleaved_f16_gemm_cycles(rows, inner, columns);
    }
    let (row_cycles, group_cycles) = match multiply {
        Precision::F16 => (rows, 1063),
        Precision::F32 => (rows.saturating_mul(4), 2126),
        Precision::F8F143 { .. } => (rows, 1063),
    };
    294u64.saturating_add(
        inner.div_ceil(16).saturating_mul(
            columns
                .saturating_mul(row_cycles)
                .div_ceil(4)
                .saturating_add(
                    columns
                        .div_ceil(16)
                        .saturating_mul(group_cycles)
                        .div_ceil(4),
                ),
        ),
    )
}

/// Matrix axes and logical arithmetic bounds for a selected GEMM invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmAxes {
    pub left_inner: crate::TensorAxis,
    pub right_inner: crate::TensorAxis,
    pub output_column: crate::TensorAxis,
    pub valid_inner: Option<u32>,
    pub valid_columns: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccumulationPrecision {
    F16,
    F32,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GemmKernelMode {
    Initialize,
    Accumulate,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GemmWeightLoad {
    Standard,
    Interleaved,
}

pub(super) const ALIGNMENT: u32 = 32;
pub(super) const READ_AHEAD_ELEMENTS: u32 = 8;
pub(super) const DISTINCT_OPERANDS: [MemoryOperand; 2] =
    [MemoryOperand::Output(0), MemoryOperand::Input(0)];

pub(super) fn access(
    multiply: Precision,
    operand: MemoryOperand,
) -> (crate::low::storage::StorageAccess, &'static [MemoryOperand]) {
    (
        crate::low::storage::StorageAccess {
            alignment: ALIGNMENT,
            access_tail_bytes: if operand == MemoryOperand::Input(0) {
                READ_AHEAD_ELEMENTS * multiply.bytes() as u32
            } else {
                0
            },
        },
        &DISTINCT_OPERANDS,
    )
}

pub(super) fn product_flops(run: &KernelRun) -> Result<[u64; 2], KernelAbiError> {
    let MidOperationKind::Gemm {
        axes, inner_block, ..
    } = run.kernel
    else {
        return Err(KernelAbiError::RequirementMismatch);
    };
    let axis = |a: crate::TensorAxis, rank| {
        a.resolve(rank)
            .map_err(|_| KernelAbiError::RequirementMismatch)
    };
    let li = axis(axes.left_inner, run.inputs[0].extents.len())?;
    let ri = axis(axes.right_inner, run.inputs[1].extents.len())?;
    let oc = axis(axes.output_column, run.outputs[0].extents.len())?;
    let rc = if ri + 1 == run.inputs[1].extents.len() {
        ri - 1
    } else {
        ri + 1
    };
    let size = |e: crate::ShardExtent, bound: Option<u32>| {
        u64::from(
            e.logical_end
                .min(bound.unwrap_or(u32::MAX))
                .saturating_sub(e.start),
        )
    };
    let rows: u64 = run.outputs[0]
        .extents
        .iter()
        .enumerate()
        .filter(|(a, _)| *a != oc)
        .map(|(_, &e)| size(e, None))
        .product();
    let cols = size(run.outputs[0].extents[oc], axes.valid_columns)
        .min(size(run.inputs[1].extents[rc], axes.valid_columns));
    let inner = size(run.inputs[0].extents[li], axes.valid_inner)
        .min(size(run.inputs[1].extents[ri], axes.valid_inner));
    let physical: u64 = run.outputs[0]
        .extents
        .iter()
        .map(|e| u64::from(e.physical_end - e.start))
        .product();
    Ok([
        2 * rows * cols * inner,
        2 * physical * u64::from(inner_block),
    ])
}

pub(super) fn call(
    kernel: &MidOperationKind,
    inputs: &[Geometry<'_>],
    outputs: &[Geometry<'_>],
) -> Result<KernelCall, KernelAbiError> {
    check_arity(inputs, outputs, 2, 1)?;
    let output = outputs[0];
    let MidOperationKind::Gemm {
        multiply,
        accumulate,
        weights,
        inner_block,
        output_columns,
        mode,
        ..
    } = *kernel
    else {
        return Err(KernelAbiError::RequirementMismatch);
    };
    if (weights == GemmWeightLoad::Interleaved && multiply == Precision::F32)
        || (matches!(multiply, Precision::F8F143 { .. })
            && (accumulate != AccumulationPrecision::F16
                || output.format().precision != Precision::F16))
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    // Scales affect call arguments, not the compiled AMP instruction stream.
    let (precision, arguments) = match multiply {
        Precision::F8F143 { scale_exponent } => (
            Precision::F8F143 { scale_exponent: 0 },
            vec![fp8_scale_argument(2 * i32::from(scale_exponent))?],
        ),
        _ => (multiply, Vec::new()),
    };
    Ok(KernelCall {
        implementation: KernelImplementation::Gemm(
            precision,
            weights,
            inner_block,
            output_columns,
            mode,
            gemm_rows(output)?,
            output
                .format()
                .layout
                .order
                .gemm_output_group()
                .unwrap_or(0),
        ),
        arguments,
    })
}

/// Packed stores use the leading address of one column group, with the row
/// permutation encoded by the GEMM. Other outputs require a contiguous view.
pub(super) fn packed_output(run: &KernelRun, shard: &BlockValue) -> Result<bool, KernelAbiError> {
    if !matches!(
        run.kernel,
        MidOperationKind::Gemm {
            multiply: Precision::F16,
            ..
        }
    ) {
        return Ok(false);
    }
    let order = run.requirements.outputs[0].format.layout.order;
    let Some(group) = order.gemm_output_group() else {
        return Ok(false);
    };
    let view = &run.outputs[0];
    let column = view
        .extents
        .len()
        .checked_sub(if order.gemm_output_transposed() { 2 } else { 1 })
        .ok_or(KernelAbiError::MissingGemmRows)?;
    let row = if column + 1 == view.extents.len() {
        column
            .checked_sub(1)
            .ok_or(KernelAbiError::MissingGemmRows)?
    } else {
        column + 1
    };
    let extent = view.extents[column];
    let start = extent.start - shard.extents[column].start;
    let end = extent.physical_end - shard.extents[column].start;
    if view.extents[row] != shard.extents[row]
        || !gemm_rows(run.geometry(MemoryOperand::Output(0)))?.is_multiple_of(16)
        || !start.is_multiple_of(16)
        || end <= start
        || start / group != (end - 1) / group
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    Ok(true)
}

pub(crate) fn gemm_rows(output: Geometry<'_>) -> Result<u32, KernelAbiError> {
    let column = output
        .rank()
        .checked_sub(if output.format().layout.order.gemm_output_transposed() {
            2
        } else {
            1
        })
        .ok_or(KernelAbiError::MissingGemmRows)?;
    output
        .widths()
        .enumerate()
        .filter(|(axis, _)| *axis != column)
        .try_fold(1u32, |rows, (_, width)| rows.checked_mul(width))
        .filter(|&rows| rows != 0)
        .ok_or(KernelAbiError::MissingGemmRows)
}

pub(super) fn specialized_gemm_symbol(
    prefix: &str,
    mode: GemmKernelMode,
    weight_suffix: &str,
    inner_block: u32,
    output_columns: u32,
    size: &str,
    small_rows: u32,
    large_rows: u32,
) -> String {
    let operation = match mode {
        GemmKernelMode::Initialize => "init",
        GemmKernelMode::Accumulate => "accumulate",
    };
    format!(
        "gemm_{prefix}_{operation}_{size}_rows{weight_suffix}_k{inner_block}_c{output_columns}_r{small_rows}_r{large_rows}"
    )
}

impl KernelBuildPlan {
    pub(super) fn add_gemms(&mut self, implementations: &BTreeSet<KernelImplementation>) {
        let mut rows = BTreeMap::<_, BTreeSet<_>>::new();
        for key in implementations {
            if let KernelImplementation::Gemm(
                precision,
                weights,
                inner,
                columns,
                mode,
                count,
                group,
            ) = key
            {
                rows.entry((*precision, *weights, *inner, *columns, *group))
                    .or_default()
                    .insert((*count, *mode));
            }
        }
        for ((precision, weights, inner_block, output_columns, output_group), used) in rows {
            let mut values = used.iter().map(|&(rows, _)| rows).collect::<Vec<_>>();
            values.dedup();
            let (source, prefix) = match precision {
                Precision::F16 => ("gemm_f16_amp.S", "f16"),
                Precision::F32 => ("gemm_f32_64_amp.S", "f32"),
                Precision::F8F143 { .. } => ("gemm_f16_amp.S", "f8"),
            };
            let weight_suffix = if weights == GemmWeightLoad::Interleaved {
                "_interleaved"
            } else {
                ""
            };
            let dispatch = format!("gemm_{prefix}{weight_suffix}_dispatch");
            if precision != Precision::F32 {
                let mut flags = vec![
                    "-DGEMM_DISPATCH_ONLY=1".into(),
                    format!("-DGEMM_DISPATCH_SYMBOL={dispatch}"),
                ];
                if matches!(precision, Precision::F8F143 { .. }) {
                    flags.push("-DGEMM_NATIVE_FP8=1".into());
                }
                if weights == GemmWeightLoad::Interleaved {
                    flags.push("-DGEMM_INTERLEAVED_WEIGHTS=1".into());
                }
                self.add_compilation(KernelCompilation {
                    source,
                    name: dispatch.clone(),
                    flags,
                });
            }
            let weight_suffix = if output_group == 0 {
                weight_suffix.to_owned()
            } else {
                format!("{weight_suffix}_packed{output_group}")
            };
            // The worker depends on precision and store permutation, not GEMM
            // extents, coefficient load mode, or the scale exponent.
            let worker = format!("gemm_{prefix}_packed{output_group}_worker");
            let worker_flags = vec![
                format!("-DGEMM_WORKER_SYMBOL={worker}"),
                format!("-DGEMM_WORKER_OUTPUT_SYMBOL={worker}_output"),
                format!("-DGEMM_WORKER_INNER_SYMBOL={worker}_inner"),
            ];
            if precision != Precision::F32 {
                let mut flags = worker_flags.clone();
                flags.extend([
                    "-DGEMM_WORKER_ONLY=1".into(),
                    format!("-DGEMM_OUTPUT_GROUP={output_group}"),
                    format!(
                        "-DGEMM_OUTPUT_GROUP_SHIFT={}",
                        output_group.max(16).ilog2() - 4
                    ),
                ]);
                if matches!(precision, Precision::F8F143 { .. }) {
                    flags.push("-DGEMM_NATIVE_FP8=1".into());
                }
                self.add_compilation(KernelCompilation {
                    source,
                    name: worker,
                    flags,
                });
            }
            for pair in values.chunks(2) {
                let small = pair[0];
                let large = *pair.last().expect("nonempty GEMM row pair");
                let variants = [
                    (GemmKernelMode::Initialize, "small", small),
                    (GemmKernelMode::Initialize, "large", large),
                    (GemmKernelMode::Accumulate, "small", small),
                    (GemmKernelMode::Accumulate, "large", large),
                ];
                let symbols = variants.map(|(mode, size, _)| {
                    specialized_gemm_symbol(
                        prefix,
                        mode,
                        &weight_suffix,
                        inner_block,
                        output_columns,
                        size,
                        small,
                        large,
                    )
                });
                let single_rows = pair.len() == 1;
                let mut flags = vec![
                    format!("-DGEMM_DISPATCH_SYMBOL={dispatch}"),
                    format!("-DGEMM_OUTPUT_GROUP={output_group}"),
                    format!(
                        "-DGEMM_OUTPUT_GROUP_SHIFT={}",
                        output_group.max(16).ilog2() - 4
                    ),
                    format!("-DGEMM_SMALL_ROWS={small}"),
                    format!("-DGEMM_LARGE_ROWS={large}"),
                    format!("-DGEMM_OUTPUT_COLUMNS={output_columns}"),
                    format!("-DGEMM_INNER_BLOCK_DIMENSION={inner_block}"),
                    format!("-DGEMM_INIT_SMALL_SYMBOL={}", symbols[0]),
                    format!("-DGEMM_INIT_LARGE_SYMBOL={}", symbols[1]),
                    format!("-DGEMM_ACCUMULATE_SMALL_SYMBOL={}", symbols[2]),
                    format!("-DGEMM_ACCUMULATE_LARGE_SYMBOL={}", symbols[3]),
                ];
                if matches!(precision, Precision::F8F143 { .. }) {
                    flags.push("-DGEMM_NATIVE_FP8=1".into());
                }
                flags.extend(worker_flags.iter().cloned());
                if single_rows {
                    flags.push("-DGEMM_SINGLE_ROWS=1".into());
                }
                if weights == GemmWeightLoad::Interleaved {
                    flags.push("-DGEMM_INTERLEAVED_WEIGHTS=1".into());
                }
                for (index, (symbol, (mode, _, rows))) in
                    symbols.into_iter().zip(variants).enumerate()
                {
                    if used.contains(&(rows, mode)) && (index % 2 == 0 || !single_rows) {
                        self.symbols.insert(
                            KernelImplementation::Gemm(
                                precision,
                                weights,
                                inner_block,
                                output_columns,
                                mode,
                                rows,
                                output_group,
                            ),
                            symbol,
                        );
                    }
                }
                self.add_compilation(KernelCompilation {
                source,
                name: format!(
                    "gemm_{prefix}{weight_suffix}_k{inner_block}_c{output_columns}_r{small}_r{large}"
                ),
                flags,
            });
            }
        }
    }
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

pub(super) fn input_padding(
    run: &KernelRun,
    inner: u32,
) -> Result<PaddingRequirement, KernelAbiError> {
    let input = &run.inputs[0];
    let MidOperationKind::Gemm { axes, .. } = run.kernel else {
        return Err(KernelAbiError::RequirementMismatch);
    };
    let mut region = input.clone();
    let mut zero = run.inputs[1].clone();
    let left = axes
        .left_inner
        .resolve(region.extents.len())
        .map_err(|_| KernelAbiError::RequirementMismatch)?;
    let right = axes
        .right_inner
        .resolve(zero.extents.len())
        .map_err(|_| KernelAbiError::RequirementMismatch)?;
    let l = region.extents[left];
    let r = zero.extents[right];
    if l.start != r.start {
        return Ok(PaddingRequirement::Required);
    }
    let start = l.logical_end.max(r.logical_end);
    let end = l.physical_end.min(r.physical_end).min(
        l.start
            .checked_add(inner)
            .ok_or(KernelAbiError::ElementCountOverflow)?,
    );
    if start >= end {
        return Ok(PaddingRequirement::Required);
    }
    // Discarded rows must remain zero; finite values there can
    // overflow even though the corresponding outputs are unused.
    for (axis, extent) in region.extents.iter_mut().enumerate() {
        if axis != left {
            extent.physical_end = extent.logical_end;
        }
    }
    for (view, axis) in [(&mut region, left), (&mut zero, right)] {
        view.extents[axis].start = start;
        view.extents[axis].logical_end = start;
        view.extents[axis].physical_end = end;
    }
    Ok(PaddingRequirement::FiniteIfZero { region, zero })
}

#[cfg(test)]
mod tests {
    use super::*;

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
