//! Device object recipes for the selected kernel specializations.

use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelCompilation {
    pub source: &'static str,
    pub name: String,
    pub flags: Vec<String>,
    pub retained_symbols: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelBuildPlan {
    pub compilations: Vec<KernelCompilation>,
    pub(super) symbols: BTreeMap<KernelSpecialization, String>,
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
        "ipu_stack_gemm_{prefix}_{operation}_{size}_rows{weight_suffix}_k{inner_block}_c{output_columns}_r{small_rows}_r{large_rows}"
    )
}

impl KernelBuildPlan {
    /// Derives device objects from the finalized schedule, so row variants are
    /// compiler specializations rather than a fixed collection of binaries.
    pub fn from_program(program: &LowProgram) -> Result<Self, KernelAbiError> {
        let mut inventory = KernelInventory::default();
        for tile in &program.tiles {
            inventory.collect(program, tile)?;
        }
        let KernelInventory {
            rows,
            gelu,
            reduction_add,
            rearrangements,
            unpacks,
            attention,
            attention_stages,
        } = inventory;
        let mut plan = Self::default();
        for ((precision, weights, inner_block, output_columns), values) in rows {
            let values = values.into_iter().collect::<Vec<_>>();
            let (source, prefix) = match precision {
                Precision::F16 => ("gemm_f16_amp.S", "f16"),
                Precision::F32 => ("gemm_f32_64_amp.S", "f32"),
                Precision::F8F143 { .. } => continue,
            };
            let weight_suffix = if weights == GemmWeightLoad::Interleaved {
                "_interleaved"
            } else {
                ""
            };
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
                        weight_suffix,
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
                    plan.symbols.insert(
                        KernelSpecialization::Gemm(
                            precision,
                            weights,
                            inner_block,
                            output_columns,
                            mode,
                            small,
                        ),
                        symbols[row_index].clone(),
                    );
                    if pair.len() == 2 {
                        plan.symbols.insert(
                            KernelSpecialization::Gemm(
                                precision,
                                weights,
                                inner_block,
                                output_columns,
                                mode,
                                large,
                            ),
                            symbols[row_index + 1].clone(),
                        );
                    }
                }
                let single_rows = pair.len() == 1;
                let mut flags = vec![
                    format!("-DGEMM_SMALL_ROWS={small}"),
                    format!("-DGEMM_LARGE_ROWS={large}"),
                    format!("-DGEMM_OUTPUT_COLUMNS={output_columns}"),
                    format!("-DGEMM_INNER_BLOCK_DIMENSION={inner_block}"),
                    format!("-DGEMM_INIT_SMALL_SYMBOL={}", symbols[0]),
                    format!("-DGEMM_INIT_LARGE_SYMBOL={}", symbols[1]),
                    format!("-DGEMM_ACCUMULATE_SMALL_SYMBOL={}", symbols[2]),
                    format!("-DGEMM_ACCUMULATE_LARGE_SYMBOL={}", symbols[3]),
                ];
                if single_rows {
                    flags.push("-DGEMM_SINGLE_ROWS=1".into());
                }
                if weights == GemmWeightLoad::Interleaved {
                    flags.push("-DGEMM_INTERLEAVED_WEIGHTS=1".into());
                }
                let retained_symbols = if single_rows {
                    vec![symbols[0].clone(), symbols[2].clone()]
                } else {
                    symbols.into_iter().collect()
                };
                plan.compilations.push(KernelCompilation {
                    source,
                    name: format!(
                        "gemm_{prefix}{weight_suffix}_k{inner_block}_c{output_columns}_r{small}_r{large}"
                    ),
                    flags,
                    retained_symbols,
                });
            }
        }
        if gelu {
            plan.compilations.push(KernelCompilation {
                source: "gelu_f16.S",
                name: "gelu_f16".into(),
                flags: Vec::new(),
                retained_symbols: vec!["ipu_stack_gelu_tanh_approx_f16".into()],
            });
        }
        if reduction_add {
            plan.compilations.push(KernelCompilation {
                source: "reduce_add_f16.S",
                name: "reduce_add_f16".into(),
                flags: Vec::new(),
                retained_symbols: vec!["ipu_stack_reduce_sum_f16".into()],
            });
        }
        let has_worker_codelets = !rearrangements.is_empty() || !unpacks.is_empty();
        for (order, logical_rows, physical_rows, logical_columns, physical_columns) in unpacks {
            let order_index = order.codelet_index();
            let suffix = format!(
                "o{order_index}_r{logical_rows}_p{physical_rows}_c{logical_columns}_p{physical_columns}"
            );
            let vertex = format!("UnpackAmpToRowMajorF16_{suffix}");
            let codelet = format!("__runCodelet_{vertex}");
            let call = format!("ipu_stack_unpack_amp_to_row_major_f16_{suffix}");
            plan.compilations.push(KernelCompilation {
                source: "unpack_amp_f16.cpp",
                name: format!("unpack_amp_f16_codelet_{suffix}"),
                flags: vec![
                    "-O2".into(),
                    format!("-DUNPACK_SOURCE_ORDER={order_index}"),
                    format!("-DUNPACK_LOGICAL_ROWS={logical_rows}"),
                    format!("-DUNPACK_PHYSICAL_ROWS={physical_rows}"),
                    format!("-DUNPACK_LOGICAL_COLUMNS={logical_columns}"),
                    format!("-DUNPACK_PHYSICAL_COLUMNS={physical_columns}"),
                    format!("-DUNPACK_VERTEX_NAME={vertex}"),
                ],
                retained_symbols: Vec::new(),
            });
            plan.compilations.push(KernelCompilation {
                source: "rearrange_f16.S",
                name: format!("unpack_amp_f16_wrapper_{suffix}"),
                flags: vec![
                    format!("-DREARRANGE_CALL_SYMBOL={call}"),
                    format!("-DREARRANGE_CODELET_SYMBOL={codelet}"),
                ],
                retained_symbols: vec![call.clone()],
            });
            plan.symbols.insert(
                KernelSpecialization::Unpack((
                    order,
                    logical_rows,
                    physical_rows,
                    logical_columns,
                    physical_columns,
                )),
                call,
            );
        }
        for (order, logical_rows, physical_rows, logical_columns, physical_columns) in
            rearrangements
        {
            let order_index = order.codelet_index();
            let (row_block, column_block) = match order {
                RearrangeTarget::BlockMajor {
                    row_block,
                    column_block,
                } => (row_block, column_block),
                _ => (AMP_INNER_BLOCK as u16, AMP_COLUMN_MICRO as u16),
            };
            let suffix = format!(
                "o{order_index}_r{logical_rows}_p{physical_rows}_c{logical_columns}_p{physical_columns}"
            );
            let vertex = format!("RearrangeRowMajorToAmpF16_{suffix}");
            let codelet = format!("__runCodelet_{vertex}");
            let call = format!("ipu_stack_rearrange_row_major_to_amp_f16_{suffix}");
            plan.symbols.insert(
                KernelSpecialization::Rearrange((
                    order,
                    logical_rows,
                    physical_rows,
                    logical_columns,
                    physical_columns,
                )),
                call.clone(),
            );
            if order == RearrangeTarget::AmpLeft
                && logical_columns.is_multiple_of(2)
                && physical_columns.is_multiple_of(AMP_COLUMN_MICRO)
            {
                plan.compilations.push(KernelCompilation {
                    source: "rearrange_amp_left_f16.S",
                    name: format!("rearrange_amp_left_f16_{suffix}"),
                    flags: vec![
                        format!("-DREARRANGE_CALL_SYMBOL={call}"),
                        format!("-DREARRANGE_LOGICAL_ROWS={logical_rows}"),
                        format!("-DREARRANGE_PHYSICAL_ROWS={physical_rows}"),
                        format!("-DREARRANGE_LOGICAL_COLUMNS={logical_columns}"),
                        format!("-DREARRANGE_PHYSICAL_COLUMNS={physical_columns}"),
                    ],
                    retained_symbols: vec![call.clone()],
                });
                continue;
            }
            if order
                == (RearrangeTarget::BlockMajor {
                    row_block: AMP_INNER_BLOCK as u16,
                    column_block: AMP_COLUMN_MICRO as u16,
                })
                && physical_rows == AMP_INNER_BLOCK
                && logical_columns.is_multiple_of(4)
                && physical_columns.is_multiple_of(AMP_COLUMN_MICRO)
            {
                plan.compilations.push(KernelCompilation {
                    source: "rearrange_block_major_f16.S",
                    name: format!("rearrange_block_major_f16_{suffix}"),
                    flags: vec![
                        format!("-DREARRANGE_CALL_SYMBOL={call}"),
                        format!("-DREARRANGE_PHYSICAL_COLUMNS={physical_columns}"),
                    ],
                    retained_symbols: vec![call.clone()],
                });
                continue;
            }
            if order == RearrangeTarget::AmpTransposedRight
                && logical_rows == 64
                && physical_rows == 64
                && logical_columns == 16
                && physical_columns == 16
            {
                plan.compilations.push(KernelCompilation {
                    source: "rearrange_transposed_right_f16.S",
                    name: format!("rearrange_transposed_right_f16_{suffix}"),
                    flags: vec![format!("-DREARRANGE_CALL_SYMBOL={call}")],
                    retained_symbols: vec![call.clone()],
                });
                continue;
            }
            plan.compilations.push(KernelCompilation {
                source: "rearrange_f16.cpp",
                name: format!("rearrange_f16_codelet_{suffix}"),
                flags: vec![
                    "-O2".into(),
                    format!("-DREARRANGE_TARGET_ORDER={order_index}"),
                    format!("-DREARRANGE_LOGICAL_ROWS={logical_rows}"),
                    format!("-DREARRANGE_PHYSICAL_ROWS={physical_rows}"),
                    format!("-DREARRANGE_LOGICAL_COLUMNS={logical_columns}"),
                    format!("-DREARRANGE_PHYSICAL_COLUMNS={physical_columns}"),
                    format!("-DREARRANGE_INNER_DIMENSION={AMP_COLUMN_MICRO}"),
                    format!("-DREARRANGE_ROW_BLOCK={row_block}"),
                    format!("-DREARRANGE_COLUMN_BLOCK={column_block}"),
                    format!("-DREARRANGE_VERTEX_NAME={vertex}"),
                ],
                retained_symbols: Vec::new(),
            });
            plan.compilations.push(KernelCompilation {
                source: "rearrange_f16.S",
                name: format!("rearrange_f16_wrapper_{suffix}"),
                flags: vec![
                    format!("-DREARRANGE_CALL_SYMBOL={call}"),
                    format!("-DREARRANGE_CODELET_SYMBOL={codelet}"),
                ],
                retained_symbols: vec![call.clone()],
            });
        }
        if has_worker_codelets || !attention.is_empty() || !attention_stages.is_empty() {
            plan.compilations.push(KernelCompilation {
                source: "worker_support.S",
                name: "worker_support".into(),
                flags: Vec::new(),
                retained_symbols: Vec::new(),
            });
        }
        for shape in attention {
            let suffix = format!(
                "m{}_q{}_k{}_d{}_v{}_{:08x}",
                shape.matrices,
                shape.query_rows,
                shape.key_rows,
                shape.query_dimension,
                shape.value_dimension,
                shape.scale_bits,
            );
            let call_symbol = format!("ipu_stack_flash_attention_online_f16_{suffix}");
            let vertex = format!("FlashAttentionOnlineF16_{suffix}");
            let codelet = format!("__runCodelet_{vertex}");
            let common_flags = vec![
                format!("-DATTENTION_MATRICES={}", shape.matrices),
                format!("-DATTENTION_QUERY_ROWS={}", shape.query_rows),
                format!("-DATTENTION_KEY_ROWS={}", shape.key_rows),
                format!("-DATTENTION_QUERY_DIMENSION={}", shape.query_dimension),
                format!("-DATTENTION_VALUE_DIMENSION={}", shape.value_dimension),
                format!("-DATTENTION_SCALE={}", f32::from_bits(shape.scale_bits)),
            ];
            let mut codelet_flags = common_flags;
            codelet_flags.push(format!("-DATTENTION_VERTEX_NAME={vertex}"));
            plan.compilations.push(KernelCompilation {
                source: "flash_attention_online_f16.cpp",
                name: format!("flash_attention_codelet_{suffix}"),
                flags: codelet_flags,
                retained_symbols: Vec::new(),
            });
            plan.compilations.push(KernelCompilation {
                source: "flash_attention_online_f16.S",
                name: format!("flash_attention_wrapper_{suffix}"),
                flags: vec![
                    format!("-DATTENTION_CALL_SYMBOL={call_symbol}"),
                    format!("-DATTENTION_CODELET_SYMBOL={codelet}"),
                ],
                retained_symbols: vec![call_symbol.clone()],
            });
            plan.symbols
                .insert(KernelSpecialization::Attention(shape), call_symbol);
        }
        plan.add_attention_stages(attention_stages)?;
        Ok(plan)
    }

    pub fn call(&self, run: &KernelRun) -> Result<PlannedKernelCall, KernelAbiError> {
        let abi = validate_kernel_run(run)?;
        let TileKernel::Planned(kernel) = &run.kernel;
        if abi.availability != KernelAvailability::Implemented {
            return Err(KernelAbiError::Unavailable(kernel.clone()));
        }
        let symbol = match abi.symbols {
            KernelSymbols::Exact(symbol) => symbol.to_owned(),
            _ => self
                .symbols
                .get(&KernelSpecialization::from_run(run)?)
                .cloned()
                .ok_or(KernelAbiError::RequirementMismatch)?,
        };
        Ok(PlannedKernelCall {
            symbol,
            arguments: scalar_values(run, &abi)?,
        })
    }

    pub fn retained_symbols(&self) -> impl Iterator<Item = &str> {
        self.compilations
            .iter()
            .flat_map(|compilation| compilation.retained_symbols.iter().map(String::as_str))
    }
}
