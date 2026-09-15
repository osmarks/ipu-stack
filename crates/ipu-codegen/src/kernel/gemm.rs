//! GEMM assembly specializations, paired by physical row count.

use super::*;
use crate::mid::MidOperationKind;

pub(super) fn call(run: &KernelRun) -> Result<KernelCall, KernelAbiError> {
    run.check_arity(2, 1)?;
    let MidOperationKind::Gemm {
        multiply,
        accumulate,
        weights,
        inner_block,
        output_columns,
        mode,
        ..
    } = run.kernel
    else {
        return Err(KernelAbiError::RequirementMismatch);
    };
    let output = run.requirements.outputs[0].format.precision;
    if (weights == GemmWeightLoad::Interleaved && multiply == Precision::F32)
        || (matches!(multiply, Precision::F8F143 { .. })
            && (accumulate != crate::AccumulationPrecision::F16 || output != Precision::F16))
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    // Scales change the call argument, not the compiled AMP instruction stream.
    let (precision, arguments) = if let Precision::F8F143 { scale_exponent } = multiply {
        (
            Precision::F8F143 { scale_exponent: 0 },
            vec![fp8_scale_argument(2 * i32::from(scale_exponent))?],
        )
    } else {
        (multiply, Vec::new())
    };
    Ok(KernelCall {
        implementation: KernelImplementation::Gemm(
            precision,
            weights,
            inner_block,
            output_columns,
            mode,
            gemm_rows(run)?,
            run.requirements.outputs[0]
                .format
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
        || !gemm_rows(run)?.is_multiple_of(16)
        || !start.is_multiple_of(16)
        || end <= start
        || start / group != (end - 1) / group
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    Ok(true)
}

pub(crate) fn gemm_rows(run: &KernelRun) -> Result<u32, KernelAbiError> {
    let rank = run.outputs[0].extents.len();
    let output_order = &run.requirements.outputs[0].format.layout.order;
    let matrix_column_axis = rank
        .checked_sub(if output_order.gemm_output_transposed() {
            2
        } else {
            1
        })
        .ok_or(KernelAbiError::MissingGemmRows)?;
    run.outputs[0]
        .extents
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != matrix_column_axis)
        .try_fold(1u32, |rows, extent| {
            rows.checked_mul(extent.1.physical_end - extent.1.start)
        })
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
    pub(super) fn add_gemm(
        &mut self,
        (precision, weights, inner_block, output_columns, output_group): (
            Precision,
            GemmWeightLoad,
            u32,
            u32,
            u32,
        ),
        used: BTreeSet<(u32, GemmKernelMode)>,
    ) {
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
        if precision != Precision::F32
            && !self.compilations.iter().any(|unit| unit.name == dispatch)
        {
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
            self.compilations.push(KernelCompilation {
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
        if precision != Precision::F32 && !self.compilations.iter().any(|unit| unit.name == worker)
        {
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
            self.compilations.push(KernelCompilation {
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
            for (index, (symbol, (mode, _, rows))) in symbols.into_iter().zip(variants).enumerate()
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
            self.compilations.push(KernelCompilation {
                source,
                name: format!(
                    "gemm_{prefix}{weight_suffix}_k{inner_block}_c{output_columns}_r{small}_r{large}"
                ),
                flags,
            });
        }
    }
}
