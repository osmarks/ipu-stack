//! Attention stage build recipes. Assembly workers take block sizes at runtime;
//! every stage shares its worker code across query-row counts.

use super::*;

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
        let call_symbol = format!("ipu_stack_flash_attention_online_f16_{suffix}");
        let vertex = format!("FlashAttentionOnlineF16_{suffix}");
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
        self.compilations.push(KernelCompilation {
            source: "flash_attention_online_f16.cpp",
            name: format!("flash_attention_codelet_{suffix}"),
            flags: codelet_flags,
            retained_symbols: Vec::new(),
        });
        self.add_worker_wrapper(
            format!("flash_attention_wrapper_{suffix}"),
            &call_symbol,
            &vertex,
            &[3, 4, 5, 2],
        );
        self.symbols
            .insert(KernelSpecialization::Attention(shape), call_symbol);
    }

    pub(super) fn add_attention_stages(
        &mut self,
        stages: BTreeSet<KernelSpecialization>,
    ) -> Result<(), KernelAbiError> {
        let mut compiled = BTreeSet::new();
        for key in stages {
            let (name, symbol, flags) = match key {
                KernelSpecialization::Softmax(head, keys, padded, _) => {
                    let full = keys == padded;
                    let name = format!(
                        "attention_softmax_d{head}_p{padded}_{}",
                        if full { "full" } else { "tail" }
                    );
                    let symbol = format!("ipu_stack_{name}_f16");
                    let scale_bits = (1.0_f32 / (head as f32).sqrt()).to_bits();
                    let flags = vec![
                        format!("-DATTENTION_HEAD_DIMENSION={head}"),
                        format!("-DATTENTION_KEY_BLOCK_COLUMNS={padded}"),
                        format!("-DATTENTION_SCALE_BITS=0x{scale_bits:08x}"),
                        format!("-DATTENTION_SOFTMAX_SYMBOL={symbol}"),
                    ];
                    self.symbols.insert(key, symbol.clone());
                    if compiled.insert(symbol.clone()) {
                        self.compilations.push(KernelCompilation {
                            source: if full {
                                "attention_softmax_f16.S"
                            } else {
                                "attention_stages_f16.S"
                            },
                            name,
                            flags,
                            retained_symbols: vec![symbol],
                        });
                    }
                    continue;
                }
                KernelSpecialization::Merge(values, padded, keys, _) => {
                    let name = format!("attention_merge_v{values}_p{padded}_k{keys}");
                    let symbol = format!("ipu_stack_{name}_f16");
                    let flags = vec![
                        format!("-DATTENTION_VALUE_DIMENSION={values}"),
                        format!("-DATTENTION_PADDED_VALUE_DIMENSION={padded}"),
                        format!("-DATTENTION_KEY_BLOCK_COLUMNS={keys}"),
                        format!("-DATTENTION_MERGE_SYMBOL={symbol}"),
                    ];
                    (name, symbol, flags)
                }
                _ => return Err(KernelAbiError::RequirementMismatch),
            };
            self.symbols.insert(key, symbol.clone());
            if compiled.insert(symbol.clone()) {
                self.compilations.push(KernelCompilation {
                    source: "attention_stages_f16.S",
                    name,
                    flags,
                    retained_symbols: vec![symbol],
                });
            }
        }
        Ok(())
    }
}
