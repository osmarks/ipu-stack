//! Attention stage build recipes. Assembly workers take block sizes at runtime;
//! only full-block C++ softmax specializes query rows in its worker code.

use super::*;

impl KernelBuildPlan {
    pub(super) fn add_attention_stages(
        &mut self,
        stages: Vec<(TileKernelSpec, u32)>,
    ) -> Result<(), KernelAbiError> {
        let mut compiled = BTreeSet::new();
        for (kernel, rows) in stages {
            let key = KernelSpecialization::stage(&kernel, rows)?;
            let (name, symbol, flags) = match key {
                KernelSpecialization::Softmax(head, keys, padded, rows) if keys == padded => {
                    let name = format!("attention_softmax_q{rows}_k{keys}_d{head}");
                    let symbol = format!("ipu_stack_{name}_f16");
                    self.symbols.insert(key, symbol.clone());
                    if !compiled.insert(symbol.clone()) {
                        continue;
                    }
                    let vertex = format!("AttentionSoftmax_q{rows}_k{keys}_d{head}");
                    self.compilations.push(KernelCompilation {
                        source: "attention_softmax_f16.cpp",
                        name: format!("{name}_codelet"),
                        flags: vec![
                            "-Os".into(),
                            format!("-DATTENTION_HEAD_DIMENSION={head}"),
                            format!("-DATTENTION_KEY_BLOCK_COLUMNS={padded}"),
                            format!("-DATTENTION_QUERY_ROWS={rows}"),
                            format!("-DATTENTION_KEY_ROWS={keys}"),
                            format!("-DATTENTION_VERTEX_NAME={vertex}"),
                        ],
                        retained_symbols: Vec::new(),
                    });
                    self.compilations.push(KernelCompilation {
                        source: "attention_softmax_f16_wrapper.S",
                        name: format!("{name}_wrapper"),
                        flags: vec![
                            format!("-DATTENTION_CALL_SYMBOL={symbol}"),
                            format!("-DATTENTION_CODELET_SYMBOL=__runCodelet_{vertex}"),
                        ],
                        retained_symbols: vec![symbol],
                    });
                    continue;
                }
                KernelSpecialization::Softmax(head, _, padded, _) => {
                    let name = format!("attention_softmax_d{head}_p{padded}");
                    let symbol = format!("ipu_stack_{name}_f16");
                    let scale_bits = (1.0_f32 / (head as f32).sqrt()).to_bits();
                    let flags = vec![
                        format!("-DATTENTION_HEAD_DIMENSION={head}"),
                        format!("-DATTENTION_KEY_BLOCK_COLUMNS={padded}"),
                        format!("-DATTENTION_SCALE_BITS=0x{scale_bits:08x}"),
                        format!("-DATTENTION_SOFTMAX_SYMBOL={symbol}"),
                    ];
                    (name, symbol, flags)
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
