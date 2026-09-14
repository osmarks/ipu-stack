//! Attention stage build recipes. Assembly workers take block sizes at runtime;
//! every stage shares its worker code across query-row counts.

use super::*;
use crate::ShardView;

pub(super) fn call(run: &KernelRun) -> Result<KernelCall, KernelAbiError> {
    if run.outputs.len() != 1 {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let output = run.requirements.outputs[0].format.precision;
    let (implementation, arguments) = match run.kernel {
        TileKernelSpec::FlashAttention { .. } => {
            run.check_arity(3, 1)?;
            if output != Precision::F32
                || run
                    .requirements
                    .inputs
                    .iter()
                    .any(|input| input.format.precision != Precision::F16)
            {
                return Err(KernelAbiError::Unavailable(run.kernel.clone()));
            }
            (
                KernelImplementation::Attention(attention_shape(run)?),
                Vec::new(),
            )
        }
        TileKernelSpec::AttentionSoftmax {
            head_dimension,
            key_columns,
            padded_key_columns,
        } => {
            run.check_arity(1, 1)?;
            let rows = gemm_rows(run)?;
            (
                KernelImplementation::Softmax(
                    head_dimension,
                    key_columns,
                    padded_key_columns,
                    output,
                ),
                vec![
                    rows,
                    key_columns,
                    u32::from(cost::f16_softmax_split_rows(
                        u64::from(rows),
                        u64::from(key_columns),
                        u64::from(padded_key_columns),
                    )),
                ],
            )
        }
        TileKernelSpec::AttentionMerge {
            value_dimension,
            padded_value_dimension,
            key_block_columns,
            initial,
            final_block,
        } => {
            if output != Precision::F32 && !(output == Precision::F16 && final_block) {
                return Err(KernelAbiError::Unavailable(run.kernel.clone()));
            }
            run.check_arity(if output == Precision::F16 { 3 } else { 2 }, 1)?;
            (
                KernelImplementation::Merge(
                    value_dimension,
                    padded_value_dimension,
                    key_block_columns,
                    output,
                    run.requirements.inputs[1].format.precision,
                ),
                vec![u32::from(initial), u32::from(final_block), gemm_rows(run)?],
            )
        }
        _ => return Err(KernelAbiError::RequirementMismatch),
    };
    Ok(KernelCall {
        implementation,
        arguments,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct AttentionKernelShape {
    pub(crate) matrices: u32,
    pub(crate) query_rows: u32,
    pub(crate) key_rows: u32,
    pub(crate) query_dimension: u32,
    pub(crate) value_dimension: u32,
    pub(crate) scale_bits: u32,
}

pub(crate) fn attention_shape(run: &KernelRun) -> Result<AttentionKernelShape, KernelAbiError> {
    let TileKernelSpec::FlashAttention {
        options,
        accumulate,
    } = &run.kernel
    else {
        return Err(KernelAbiError::RequirementMismatch);
    };
    if options.causal || *accumulate != crate::AccumulationPrecision::F32 {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let [query, key, value] = run.inputs.as_slice() else {
        return Err(KernelAbiError::RequirementMismatch);
    };
    let extents = |view: &ShardView| {
        view.extents
            .iter()
            .map(|extent| extent.physical_end - extent.start)
            .collect::<Vec<_>>()
    };
    let query = extents(query);
    let key = extents(key);
    let value = extents(value);
    if query.len() < 2 || query.len() != key.len() || query.len() != value.len() {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let rank = query.len();
    if query[..rank - 2] != key[..rank - 2]
        || query[..rank - 2] != value[..rank - 2]
        || query[rank - 1] != key[rank - 1]
        || key[rank - 2] != value[rank - 2]
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let matrices = query[..rank - 2]
        .iter()
        .try_fold(1u32, |product, &extent| product.checked_mul(extent))
        .ok_or(KernelAbiError::ElementCountOverflow)?;
    let scale = options
        .scale
        .as_value()
        .unwrap_or_else(|| 1.0 / (query[rank - 1] as f32).sqrt());
    Ok(AttentionKernelShape {
        matrices,
        query_rows: query[rank - 2],
        key_rows: key[rank - 2],
        query_dimension: query[rank - 1],
        value_dimension: value[rank - 1],
        scale_bits: scale.to_bits(),
    })
}

impl KernelBuildPlan {
    pub(super) fn add_attention(&mut self, shape: AttentionKernelShape) {
        let suffix = format!(
            "m{}_q{}_k{}_d{}_v{}_{:08x}",
            shape.matrices,
            shape.query_rows,
            shape.key_rows,
            shape.query_dimension,
            shape.value_dimension,
            shape.scale_bits,
        );
        let call_symbol = format!("flash_attention_online_f16_{suffix}");
        let vertex = format!("FlashAttentionOnlineF16_{suffix}");
        let flags = vec![
            format!("-DATTENTION_MATRICES={}", shape.matrices),
            format!("-DATTENTION_QUERY_ROWS={}", shape.query_rows),
            format!("-DATTENTION_KEY_ROWS={}", shape.key_rows),
            format!("-DATTENTION_QUERY_DIMENSION={}", shape.query_dimension),
            format!("-DATTENTION_VALUE_DIMENSION={}", shape.value_dimension),
            format!("-DATTENTION_SCALE={}", f32::from_bits(shape.scale_bits)),
            format!("-DATTENTION_VERTEX_NAME={vertex}"),
        ];
        self.add_vertex(
            "flash_attention_online_f16.cpp",
            &call_symbol,
            &vertex,
            flags,
            &[3, 4, 5, 2],
        );
        self.symbols
            .insert(KernelImplementation::Attention(shape), call_symbol);
    }

    pub(super) fn add_attention_stages(
        &mut self,
        stages: BTreeSet<KernelImplementation>,
    ) -> Result<(), KernelAbiError> {
        let mut compiled = BTreeSet::new();
        for key in stages {
            let (name, symbol, source, flags) = match key {
                KernelImplementation::Softmax(head, keys, padded, precision) => {
                    let full = keys == padded;
                    let name = format!(
                        "attention_softmax_d{head}_p{padded}_{}",
                        if full { "full" } else { "tail" }
                    );
                    let symbol = match precision {
                        Precision::F16 => format!("{name}_f16"),
                        Precision::F8F143 { scale_exponent } => {
                            format!("{name}_f8_s{scale_exponent}")
                        }
                        _ => return Err(KernelAbiError::RequirementMismatch),
                    };
                    let name = symbol.replace('-', "m");
                    let symbol = name.clone();
                    let scale_bits = (1.0_f32 / (head as f32).sqrt()).to_bits();
                    let mut flags = vec![
                        format!("-DATTENTION_HEAD_DIMENSION={head}"),
                        format!("-DATTENTION_FULL_BLOCK={}", u8::from(full)),
                        format!("-DATTENTION_KEY_BLOCK_COLUMNS={padded}"),
                        format!("-DATTENTION_SCALE_BITS=0x{scale_bits:08x}"),
                        format!("-DATTENTION_SOFTMAX_SYMBOL={symbol}"),
                    ];
                    if let Precision::F8F143 { scale_exponent } = precision {
                        if !padded.is_multiple_of(32) {
                            return Err(KernelAbiError::RequirementMismatch);
                        }
                        flags.extend([
                            "-DATTENTION_OUTPUT_F8".into(),
                            format!("-DATTENTION_OUTPUT_SCALE={scale_exponent}"),
                        ]);
                    }
                    (name, symbol, "attention_softmax_f16.S", flags)
                }
                KernelImplementation::Merge(values, padded, keys, output, weights) => {
                    let suffix = match output {
                        Precision::F16 => "out16",
                        Precision::F32 => "out32",
                        _ => return Err(KernelAbiError::RequirementMismatch),
                    };
                    let name = format!(
                        "attention_merge_v{values}_p{padded}_k{keys}_{suffix}{}",
                        if matches!(weights, Precision::F8F143 { .. }) {
                            "_p8"
                        } else {
                            ""
                        }
                    );
                    let symbol = format!("{name}_f16");
                    let flags = vec![
                        format!("-DATTENTION_VALUE_DIMENSION={values}"),
                        format!("-DATTENTION_WEIGHT_BYTES={}", weights.bytes()),
                        format!("-DATTENTION_PADDED_VALUE_DIMENSION={padded}"),
                        format!("-DATTENTION_KEY_BLOCK_COLUMNS={keys}"),
                        format!("-DATTENTION_MERGE_SYMBOL={symbol}"),
                        format!(
                            "-DATTENTION_MERGE_OUTPUT_F16={}",
                            u8::from(output == Precision::F16)
                        ),
                    ];
                    (name, symbol, "attention_merge_f16.S", flags)
                }
                _ => return Err(KernelAbiError::RequirementMismatch),
            };
            self.symbols.insert(key, symbol.clone());
            if compiled.insert(symbol.clone()) {
                self.compilations.push(KernelCompilation {
                    source,
                    name,
                    flags,
                    retained_symbols: vec![symbol],
                });
            }
        }
        Ok(())
    }
}
