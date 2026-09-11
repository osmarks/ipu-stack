//! GEMM assembly specializations, paired by physical row count.

use super::*;

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
        let values = used
            .iter()
            .map(|&(rows, _)| rows)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
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
                retained_symbols: vec![],
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
                retained_symbols: vec![],
            });
        }
        for pair in values.chunks(2) {
            let small = pair[0];
            let large = *pair.last().expect("nonempty GEMM row pair");
            let symbols = [
                (GemmKernelMode::Initialize, "small", small),
                (GemmKernelMode::Initialize, "large", large),
                (GemmKernelMode::Accumulate, "small", small),
                (GemmKernelMode::Accumulate, "large", large),
            ]
            .map(|(mode, size, _)| {
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
            for (mode, row_index) in [
                (GemmKernelMode::Initialize, 0usize),
                (GemmKernelMode::Accumulate, 2usize),
            ] {
                if used.contains(&(small, mode)) {
                    self.symbols.insert(
                        KernelSpecialization::Gemm(
                            precision,
                            weights,
                            inner_block,
                            output_columns,
                            mode,
                            small,
                            output_group,
                        ),
                        symbols[row_index].clone(),
                    );
                }
                if pair.len() == 2 && used.contains(&(large, mode)) {
                    self.symbols.insert(
                        KernelSpecialization::Gemm(
                            precision,
                            weights,
                            inner_block,
                            output_columns,
                            mode,
                            large,
                            output_group,
                        ),
                        symbols[row_index + 1].clone(),
                    );
                }
            }
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
            flags.extend(worker_flags.clone());
            if single_rows {
                flags.push("-DGEMM_SINGLE_ROWS=1".into());
            }
            if weights == GemmWeightLoad::Interleaved {
                flags.push("-DGEMM_INTERLEAVED_WEIGHTS=1".into());
            }
            let retained_symbols = symbols
                .into_iter()
                .enumerate()
                .filter_map(|(index, symbol)| {
                    let mode = if index < 2 {
                        GemmKernelMode::Initialize
                    } else {
                        GemmKernelMode::Accumulate
                    };
                    let rows = if index % 2 == 0 { small } else { large };
                    (used.contains(&(rows, mode)) && (index % 2 == 0 || !single_rows))
                        .then_some(symbol)
                })
                .collect();
            self.compilations.push(KernelCompilation {
                source,
                name: format!(
                    "gemm_{prefix}{weight_suffix}_k{inner_block}_c{output_columns}_r{small}_r{large}"
                ),
                flags,
                retained_symbols,
            });
        }
    }
}
