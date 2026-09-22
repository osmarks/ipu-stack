//! FP16/FP8 GEMM entry points and shared worker/dispatch objects.

use super::*;
use crate::mid::MidOperationKind;

use serde::{Deserialize, Serialize};

/// Traverse one selected resident product. Batch matrices occupy separate
/// panels; this changes kernel calls, not the distributed algorithm or storage.
/// Costing uses representative extents and lowering uses actual shard extents.
pub(crate) fn invocations(
    kernel: &MidOperationKind,
    extents: [&[crate::ShardExtent]; 3],
) -> Result<Vec<(MidOperationKind, [Vec<crate::ShardExtent>; 3])>, KernelError> {
    let MidOperationKind::Gemm {
        axes,
        multiply,
        mode,
        ..
    } = *kernel
    else {
        return Err(KernelError::RequirementMismatch);
    };
    let [left, right, output] = extents;
    let li = axes
        .left_inner
        .resolve(left.len())
        .map_err(|_| KernelError::RequirementMismatch)?;
    let ri = axes
        .right_inner
        .resolve(right.len())
        .map_err(|_| KernelError::RequirementMismatch)?;
    let oc = axes
        .output_column
        .resolve(output.len())
        .map_err(|_| KernelError::RequirementMismatch)?;
    let rc = 2 * right.len() - 3 - ri;
    let width = |e: crate::ShardExtent| e.physical_end - e.start;
    let batches = output[..output.len() - 2]
        .iter()
        .map(|&e| width(e))
        .product::<u32>();
    let k = width(left[li]);
    let n = width(output[oc]);
    if batches == 0 || k == 0 || n == 0 {
        return Ok(Vec::new());
    }
    let kb = if batches > 1 {
        if multiply == Precision::F16 { 16 } else { 32 }
    } else {
        k
    };
    let nb = if batches > 1 { 16 } else { n };
    let slice = |e: &mut crate::ShardExtent, start: u32, count: u32| {
        e.start += start;
        e.physical_end = (e.start + count).min(e.physical_end);
        e.logical_end = e.logical_end.min(e.physical_end).max(e.start);
    };
    let mut calls = Vec::new();
    for column in (0..n).step_by(nb as usize) {
        for inner in (0..k).step_by(kb as usize) {
            for batch in 0..batches {
                let mut regions = [left.to_vec(), right.to_vec(), output.to_vec()];
                let mut remaining = batch;
                for axis in (0..output.len() - 2).rev() {
                    let coordinate = remaining % width(output[axis]);
                    remaining /= width(output[axis]);
                    slice(&mut regions[2][axis], coordinate, 1);
                    for operand in 0..2 {
                        if let Some(input_axis) =
                            (axis + regions[operand].len()).checked_sub(output.len())
                        {
                            let extent = &mut regions[operand][input_axis];
                            slice(extent, if width(*extent) == 1 { 0 } else { coordinate }, 1);
                        }
                    }
                }
                slice(&mut regions[0][li], inner, kb);
                slice(&mut regions[1][ri], inner, kb);
                slice(&mut regions[1][rc], column, nb);
                slice(&mut regions[2][oc], column, nb);
                let mut call = kernel.clone();
                if let MidOperationKind::Gemm {
                    inner_block,
                    output_columns,
                    mode: call_mode,
                    ..
                } = &mut call
                {
                    *inner_block = width(regions[0][li]);
                    *output_columns = width(regions[2][oc]);
                    *call_mode = if inner == 0 {
                        mode
                    } else {
                        GemmKernelMode::Accumulate
                    };
                }
                calls.push((call, regions));
            }
        }
    }
    Ok(calls)
}

fn cycles(
    multiply: Precision,
    weights: GemmWeightLoad,
    inner_block: u32,
    output_columns: u32,
    rows: u32,
    group: u32,
) -> u64 {
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
    if interleaved {
        return interleaved_f16_gemm_cycles(rows, inner, columns);
    }
    let row_cycles = rows;
    let group_cycles = 1063;
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

/// Matrix axes and logical arithmetic bounds for a GEMM.
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

pub(super) fn product_flops(run: &KernelRun) -> Result<[u64; 2], KernelError> {
    let MidOperationKind::Gemm {
        axes, inner_block, ..
    } = run.kernel
    else {
        return Err(KernelError::RequirementMismatch);
    };
    let axis = |a: crate::TensorAxis, rank| {
        a.resolve(rank)
            .map_err(|_| KernelError::RequirementMismatch)
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
    inputs: &[TensorStorage<'_>],
    outputs: &[TensorStorage<'_>],
    build: Option<&mut KernelObjects>,
) -> Result<KernelCall, KernelError> {
    check_arity(inputs, outputs, 2, 1)?;
    let output = outputs[0];
    let MidOperationKind::Gemm {
        multiply,
        accumulate,
        weights,
        inner_block,
        output_columns,
        mode,
        axes,
    } = *kernel
    else {
        return Err(KernelError::RequirementMismatch);
    };
    // Accumulation reads C through ld128 and the second load port of
    // ld2xst64pace. Both require region 1; initialization alone only stores C.
    let grain = if multiply == Precision::F16 { 16 } else { 32 };
    if (mode == GemmKernelMode::Accumulate || inner_block > grain)
        && output.format.layout.memory_class != crate::MemoryClass::Ipu21Interleaved
    {
        return Err(KernelError::RequirementMismatch);
    }
    if multiply == Precision::F32
        || (matches!(multiply, Precision::F8F143 { .. })
            && (accumulate != AccumulationPrecision::F16
                || output.format.precision != Precision::F16))
    {
        return Err(KernelError::RequirementMismatch);
    }
    // Scales affect call arguments, not the compiled AMP instruction stream.
    let (precision, arguments) = match multiply {
        Precision::F8F143 { scale_exponent } => (
            Precision::F8F143 { scale_exponent: 0 },
            vec![fp8_scale_argument(2 * i32::from(scale_exponent))?],
        ),
        _ => (multiply, Vec::new()),
    };
    let rows = gemm_rows(output)?;
    let group = output.format.layout.order.gemm_output_group().unwrap_or(0);
    if let Some(build) = build {
        let source = "gemm_f16_amp.S";
        let prefix = if matches!(precision, Precision::F8F143 { .. }) {
            "f8"
        } else {
            "f16"
        };
        let weight_suffix = if weights == GemmWeightLoad::Interleaved {
            "_interleaved"
        } else {
            ""
        };
        let dispatch = format!("gemm_{prefix}{weight_suffix}_dispatch");
        let precision_flags = if matches!(precision, Precision::F8F143 { .. }) {
            vec!["-DGEMM_NATIVE_FP8=1".into()]
        } else {
            Vec::new()
        };
        let mut dispatch_flags = precision_flags.clone();
        dispatch_flags.extend([
            "-DGEMM_DISPATCH_ONLY=1".into(),
            format!("-DGEMM_DISPATCH_SYMBOL={dispatch}"),
        ]);
        if weights == GemmWeightLoad::Interleaved {
            dispatch_flags.push("-DGEMM_INTERLEAVED_WEIGHTS=1".into());
        }
        build.add_compilation(KernelCompilation {
            source,
            name: dispatch.clone(),
            flags: dispatch_flags,
        });

        // Workers depend on precision and output packing, not row count or weight loading.
        let worker = format!("gemm_{prefix}_packed{group}_worker");
        let worker_flags = [
            format!("-DGEMM_WORKER_SYMBOL={worker}"),
            format!("-DGEMM_WORKER_OUTPUT_SYMBOL={worker}_output"),
            format!("-DGEMM_WORKER_INNER_SYMBOL={worker}_inner"),
        ];
        let mut flags = precision_flags.clone();
        flags.extend(worker_flags.iter().cloned());
        flags.extend([
            "-DGEMM_WORKER_ONLY=1".into(),
            format!("-DGEMM_OUTPUT_GROUP={group}"),
            format!("-DGEMM_OUTPUT_GROUP_SHIFT={}", group.max(16).ilog2() - 4),
        ]);
        build.add_compilation(KernelCompilation {
            source,
            name: worker,
            flags,
        });

        let mut flags = precision_flags;
        flags.extend(worker_flags);
        flags.extend([
            format!("-DGEMM_DISPATCH_SYMBOL={dispatch}"),
            format!("-DGEMM_OUTPUT_GROUP={group}"),
            format!("-DGEMM_ROWS={rows}"),
            format!("-DGEMM_OUTPUT_COLUMNS={output_columns}"),
            format!("-DGEMM_INNER_BLOCK_DIMENSION={inner_block}"),
            format!(
                "-DGEMM_INIT_SYMBOL={}",
                specialized_gemm_symbol(
                    precision,
                    weights,
                    GemmKernelMode::Initialize,
                    inner_block,
                    output_columns,
                    rows,
                    group
                )
            ),
            format!(
                "-DGEMM_ACCUMULATE_SYMBOL={}",
                specialized_gemm_symbol(
                    precision,
                    weights,
                    GemmKernelMode::Accumulate,
                    inner_block,
                    output_columns,
                    rows,
                    group
                )
            ),
        ]);
        build.add_compilation(KernelCompilation {
            source,
            name: format!("gemm_{prefix}{weight_suffix}_packed{group}_k{inner_block}_c{output_columns}_r{rows}"),
            flags,
        });
    }
    let mut call = KernelCall::new(
        specialized_gemm_symbol(
            precision,
            weights,
            mode,
            inner_block,
            output_columns,
            rows,
            group,
        ),
        arguments,
        cycles(precision, weights, inner_block, output_columns, rows, group),
    );
    if precision == Precision::F16 {
        call.padding = input_padding(inputs[0], inputs[1], axes, inner_block)?;
    }
    Ok(call)
}

/// Packed stores use the leading address of one column group, with the row
/// permutation encoded by the GEMM. Other outputs require a contiguous view.
pub(super) fn packed_output(run: &KernelRun, shard: &BlockValue) -> Result<bool, KernelError> {
    if !matches!(
        run.kernel,
        MidOperationKind::Gemm {
            multiply: Precision::F16,
            ..
        }
    ) {
        return Ok(false);
    }
    let order = run.requirements.outputs[0].layout.order;
    let Some(group) = order.gemm_output_group() else {
        return Ok(false);
    };
    let view = &run.outputs[0];
    let column = view
        .extents
        .len()
        .checked_sub(if order.gemm_output_transposed() { 2 } else { 1 })
        .ok_or(KernelError::MissingGemmRows)?;
    let row = if column + 1 == view.extents.len() {
        column.checked_sub(1).ok_or(KernelError::MissingGemmRows)?
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
        return Err(KernelError::RequirementMismatch);
    }
    Ok(true)
}

pub(crate) fn gemm_rows(output: TensorStorage<'_>) -> Result<u32, KernelError> {
    let column = output
        .extents
        .len()
        .checked_sub(if output.format.layout.order.gemm_output_transposed() {
            2
        } else {
            1
        })
        .ok_or(KernelError::MissingGemmRows)?;
    output
        .widths()
        .enumerate()
        .filter(|(axis, _)| *axis != column)
        .try_fold(1u32, |rows, (_, width)| rows.checked_mul(width))
        .filter(|&rows| rows != 0)
        .ok_or(KernelError::MissingGemmRows)
}

pub(super) fn specialized_gemm_symbol(
    precision: Precision,
    weights: GemmWeightLoad,
    mode: GemmKernelMode,
    inner: u32,
    columns: u32,
    rows: u32,
    group: u32,
) -> String {
    let prefix = match precision {
        Precision::F16 => "f16",
        Precision::F32 => unreachable!("FP32 GEMM is unsupported"),
        Precision::F8F143 { .. } => "f8",
    };
    let weights = if weights == GemmWeightLoad::Interleaved {
        "_interleaved"
    } else {
        ""
    };
    let packed = if group == 0 {
        String::new()
    } else {
        format!("_packed{group}")
    };
    let operation = match mode {
        GemmKernelMode::Initialize => "init",
        GemmKernelMode::Accumulate => "accumulate",
    };
    format!("gemm_{prefix}_{operation}_rows{weights}{packed}_k{inner}_c{columns}_r{rows}")
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

fn input_padding(
    input: TensorStorage<'_>,
    weights: TensorStorage<'_>,
    axes: GemmAxes,
    inner: u32,
) -> Result<PaddingRequirement, KernelError> {
    let left = axes
        .left_inner
        .resolve(input.extents.len())
        .map_err(|_| KernelError::RequirementMismatch)?;
    let right = axes
        .right_inner
        .resolve(weights.extents.len())
        .map_err(|_| KernelError::RequirementMismatch)?;
    let l = input.extents[left];
    let r = weights.extents[right];
    if l.start != r.start {
        return Ok(PaddingRequirement::Required);
    }
    let start = l.logical_end.max(r.logical_end);
    let end = l.physical_end.min(r.physical_end).min(
        l.start
            .checked_add(inner)
            .ok_or(KernelError::ElementCountOverflow)?,
    );
    if start >= end {
        return Ok(PaddingRequirement::Required);
    }
    let mut region = input.extents.to_vec();
    let mut zero = weights.extents.to_vec();
    // Discarded rows must remain zero; finite values there can
    // overflow even though the corresponding outputs are unused.
    for (axis, extent) in region.iter_mut().enumerate() {
        if axis != left {
            extent.physical_end = extent.logical_end;
        }
    }
    for (view, axis) in [(&mut region, left), (&mut zero, right)] {
        view[axis].start = start;
        view[axis].logical_end = start;
        view[axis].physical_end = end;
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
