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

impl KernelBuildPlan {
    /// Derives device objects from the finalized schedule, so row variants are
    /// compiler specializations rather than a fixed collection of binaries.
    pub fn from_program(program: &LowProgram) -> Result<Self, KernelAbiError> {
        let mut inventory = KernelInventory::default();
        for tile in &program.tiles {
            inventory.collect(program, tile)?;
        }
        Self::from_inventory(inventory)
    }

    pub(super) fn from_inventory(inventory: KernelInventory) -> Result<Self, KernelAbiError> {
        let KernelInventory {
            rows,
            gelu,
            normalization,
            add,
            cast_f32_f16,
            fp8_casts,
            reduction_add,
            rearrangements,
            unpacks,
            attention,
            attention_stages,
        } = inventory;
        let mut plan = Self::default();
        for (configuration, rows) in rows {
            plan.add_gemm(configuration, rows);
        }
        for (enabled, symbol, vertex, registers) in [
            (
                normalization,
                "layer_norm_f16",
                "LayerNormF16",
                &[3, 4, 5, 2, 6, 7][..],
            ),
            (add, "add_f16", "AddF16", &[3, 4, 2, 5, 6, 7][..]),
        ] {
            if enabled {
                plan.compilations.push(KernelCompilation {
                    source: "elementwise_f16.cpp",
                    name: format!("{symbol}_codelet"),
                    flags: vec!["-O2".into(), format!("-DVERTEX_{vertex}")],
                    retained_symbols: vec![],
                });
                if symbol == "layer_norm_f16" {
                    plan.compilations.push(KernelCompilation {
                        source: "layer_norm_f16.S",
                        name: "layer_norm_f16_wrapper".into(),
                        flags: vec![],
                        retained_symbols: vec![symbol.into()],
                    });
                } else {
                    plan.add_worker_wrapper(format!("{symbol}_wrapper"), symbol, vertex, registers);
                }
            }
        }
        if gelu {
            plan.compilations.push(KernelCompilation {
                source: "gelu_f16.S",
                name: "gelu_f16".into(),
                flags: Vec::new(),
                retained_symbols: vec!["gelu_tanh_approx_f16".into()],
            });
        }
        if reduction_add {
            plan.compilations.push(KernelCompilation {
                source: "reduce_add_f16.S",
                name: "reduce_add_f16".into(),
                flags: Vec::new(),
                retained_symbols: vec!["reduce_sum_f16".into()],
            });
        }
        if cast_f32_f16 {
            plan.compilations.push(KernelCompilation {
                source: "cast_f32_f16.cpp",
                name: "cast_f32_f16_codelet".into(),
                flags: vec!["-O2".into()],
                retained_symbols: Vec::new(),
            });
            plan.add_worker_wrapper(
                "cast_f32_f16_wrapper".into(),
                "cast_f32_f16",
                "CastF32ToF16",
                &[3, 2, 4],
            );
        }
        let has_worker_codelets = normalization
            || add
            || cast_f32_f16
            || !fp8_casts.is_empty()
            || !rearrangements.is_empty()
            || !unpacks.is_empty();
        for (from, to) in fp8_casts {
            let name = |bytes| match bytes {
                1 => "f8",
                2 => "f16",
                _ => "f32",
            };
            let symbol = format!("cast_{}_{}", name(from), name(to));
            let vertex = format!("Cast{from}To{to}");
            plan.compilations.push(KernelCompilation {
                source: "cast_f8.cpp",
                name: format!("{symbol}_codelet"),
                flags: vec![
                    "-O2".into(),
                    format!("-DINPUT_BYTES={from}"),
                    format!("-DOUTPUT_BYTES={to}"),
                    format!("-DCAST_VERTEX={vertex}"),
                ],
                retained_symbols: Vec::new(),
            });
            plan.add_worker_wrapper(
                format!("{symbol}_wrapper"),
                &symbol,
                &vertex,
                &[3, 2, 4, 5, 6, 7, 8],
            );
        }
        for shape in unpacks {
            plan.add_unpack(shape);
        }
        for shape in rearrangements {
            plan.add_rearrangement(shape);
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
            plan.add_attention(shape);
        }
        plan.add_attention_stages(attention_stages)?;
        Ok(plan)
    }

    /// Marshal supervisor registers into the C++ vertex argument block.
    /// The register order follows the vertex fields, independently of call ABI.
    pub(super) fn add_worker_wrapper(
        &mut self,
        name: String,
        symbol: &str,
        vertex: &str,
        registers: &[u8],
    ) {
        let arguments = registers
            .iter()
            .map(|register| format!("$m{register}"))
            .collect::<Vec<_>>()
            .join(",");
        let frame_bytes = (registers.len() * 4).next_multiple_of(16);
        self.compilations.push(KernelCompilation {
            source: "worker_call.S",
            name,
            flags: vec![
                format!("-DWORKER_CALL_SYMBOL={symbol}"),
                format!("-DWORKER_CODELET_SYMBOL=__runCodelet_{vertex}"),
                format!("-DWORKER_ARGUMENTS={arguments}"),
                format!("-DWORKER_FRAME_BYTES={frame_bytes}"),
            ],
            retained_symbols: vec![symbol.to_owned()],
        });
    }

    pub fn call(&self, run: &KernelRun) -> Result<PlannedKernelCall, KernelAbiError> {
        let abi = validate_kernel_run(run)?;
        let kernel = &run.kernel;
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
