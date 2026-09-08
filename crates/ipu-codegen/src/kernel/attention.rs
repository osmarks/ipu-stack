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
            .insert(KernelSpecialization::Attention(shape), call_symbol);
    }

    pub(super) fn add_attention_stages(
        &mut self,
        stages: BTreeSet<KernelSpecialization>,
    ) -> Result<(), KernelAbiError> {
        let mut compiled = BTreeSet::new();
        for key in stages {
            let (name, symbol, source, flags) = match key {
                KernelSpecialization::Softmax(head, keys, padded) => {
                    let full = keys == padded;
                    let name = format!(
                        "attention_softmax_d{head}_p{padded}_{}",
                        if full { "full" } else { "tail" }
                    );
                    let symbol = format!("{name}_f16");
                    let scale_bits = (1.0_f32 / (head as f32).sqrt()).to_bits();
                    let flags = vec![
                        format!("-DATTENTION_HEAD_DIMENSION={head}"),
                        format!("-DATTENTION_FULL_BLOCK={}", u8::from(full)),
                        format!("-DATTENTION_KEY_BLOCK_COLUMNS={padded}"),
                        format!("-DATTENTION_SCALE_BITS=0x{scale_bits:08x}"),
                        format!("-DATTENTION_SOFTMAX_SYMBOL={symbol}"),
                    ];
                    (name, symbol, "attention_softmax_f16.S", flags)
                }
                KernelSpecialization::Merge(values, padded, keys) => {
                    let name = format!("attention_merge_v{values}_p{padded}_k{keys}");
                    let symbol = format!("{name}_f16");
                    let flags = vec![
                        format!("-DATTENTION_VALUE_DIMENSION={values}"),
                        format!("-DATTENTION_PADDED_VALUE_DIMENSION={padded}"),
                        format!("-DATTENTION_KEY_BLOCK_COLUMNS={keys}"),
                        format!("-DATTENTION_MERGE_SYMBOL={symbol}"),
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
