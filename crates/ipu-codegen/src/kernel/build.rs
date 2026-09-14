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
            exact_symbols,
            rows,
            rearrangements,
            unpacks,
            attention,
            attention_stages,
        } = inventory;
        let mut plan = Self::default();
        for (configuration, rows) in rows {
            plan.add_gemm(configuration, rows);
        }
        for (source, wrapper, vertex, variants) in [
            (
                "elementwise_f16.cpp",
                "layer_norm_f16.S",
                "LayerNormF16",
                &[
                    ("layer_norm_f16", None),
                    ("add_layer_norm_f16", Some("-DNORM_WITH_ADD")),
                    ("layer_norm_f8", Some("-DNORM_FP8")),
                ][..],
            ),
            (
                "layer_norm_distributed.cpp",
                "layer_norm_moments.S",
                "LayerNormMoments",
                &[
                    ("layer_norm_moments", None),
                    ("add_layer_norm_moments", Some("-DNORM_STORE_SUM")),
                ][..],
            ),
        ] {
            for &(symbol, extra) in variants {
                if !exact_symbols.contains(symbol) {
                    continue;
                }
                let flags: Vec<_> = extra.into_iter().map(str::to_owned).collect();
                let mut codelet_flags = vec!["-O2".into(), format!("-DVERTEX_{vertex}")];
                codelet_flags.extend(flags.clone());
                plan.compilations.extend([
                    KernelCompilation {
                        source,
                        name: format!("{symbol}_codelet"),
                        flags: codelet_flags,
                        retained_symbols: vec![],
                    },
                    KernelCompilation {
                        source: wrapper,
                        name: format!("{symbol}_wrapper"),
                        flags,
                        retained_symbols: vec![symbol.into()],
                    },
                ]);
            }
        }
        if exact_symbols.contains("add_f16") {
            plan.add_vertex(
                "elementwise_f16.cpp",
                "add_f16",
                "AddF16",
                vec!["-O2".into(), "-DVERTEX_AddF16".into()],
                &[3, 4, 2, 5, 6, 7],
            );
        }
        for (source, symbol, extra) in [
            ("gelu_f16.S", "gelu_tanh_approx_f16", None),
            ("gelu_f16.S", "bias_gelu_f16", Some("-DGELU_WITH_BIAS")),
            ("gelu_f8.S", "gelu_f8", None),
            ("gelu_f8.S", "bias_gelu_f8", Some("-DGELU_WITH_BIAS")),
        ] {
            if exact_symbols.contains(symbol) {
                plan.compilations.push(KernelCompilation {
                    source,
                    name: symbol.into(),
                    flags: extra.into_iter().map(str::to_owned).collect(),
                    retained_symbols: vec![symbol.into()],
                });
            }
        }
        if exact_symbols.contains("reduce_sum_f16") {
            plan.compilations.push(KernelCompilation {
                source: "reduce_add_f16.S",
                name: "reduce_add_f16".into(),
                flags: Vec::new(),
                retained_symbols: vec!["reduce_sum_f16".into()],
            });
        }
        if exact_symbols.contains("cast_f32_f16") {
            plan.add_vertex(
                "cast_f32_f16.cpp",
                "cast_f32_f16",
                "CastF32ToF16",
                vec!["-O2".into()],
                &[3, 2, 4],
            );
        }
        if exact_symbols.contains("layer_norm_apply") {
            plan.add_vertex(
                "layer_norm_distributed.cpp",
                "layer_norm_apply",
                "LayerNormApply",
                vec!["-O2".into(), "-DVERTEX_LayerNormApply".into()],
                &[3, 4, 5, 6, 2, 7, 8, 9],
            );
        }
        // Scales are call arguments, so all FP8 scales share these recipes.
        let f8 = Precision::F8F143 { scale_exponent: 0 };
        for (from, to) in [
            (f8, f8),
            (f8, Precision::F16),
            (f8, Precision::F32),
            (Precision::F16, f8),
            (Precision::F32, f8),
        ] {
            let symbol = cast_symbol(from, to);
            if !exact_symbols.contains(symbol) {
                continue;
            }
            let (from, to) = (from.bytes(), to.bytes());
            let vertex = format!("Cast{from}To{to}");
            let wrapper = plan.add_vertex(
                "cast_f8.cpp",
                symbol,
                &vertex,
                vec![
                    "-O2".into(),
                    format!("-DINPUT_BYTES={from}"),
                    format!("-DOUTPUT_BYTES={to}"),
                    format!("-DCAST_VERTEX={vertex}"),
                ],
                &[3, 2, 4, 5, 6, 7, 8, 9],
            );
            if (from, to) == (2, 1) {
                wrapper.source = "cast_f8_call.S";
            }
        }
        for shape in unpacks {
            plan.add_unpack(shape);
        }
        for shape in rearrangements {
            plan.add_rearrangement(shape);
        }
        for shape in attention {
            plan.add_attention(shape);
        }
        plan.add_attention_stages(attention_stages)?;
        // Only compiler-generated C++ codelets need the worker stack symbols.
        // Derive this from the selected recipes, including assembly fast paths.
        if plan
            .compilations
            .iter()
            .any(|unit| unit.source.ends_with(".cpp"))
        {
            plan.compilations.push(KernelCompilation {
                source: "worker_support.S",
                name: "worker_support".into(),
                flags: Vec::new(),
                retained_symbols: Vec::new(),
            });
        }
        Ok(plan)
    }

    /// Compile a C++ vertex and marshal supervisor registers into its argument block.
    /// The register order follows the vertex fields, independently of call ABI.
    pub(super) fn add_vertex(
        &mut self,
        source: &'static str,
        symbol: &str,
        vertex: &str,
        flags: Vec<String>,
        registers: &[u8],
    ) -> &mut KernelCompilation {
        self.compilations.push(KernelCompilation {
            source,
            name: format!("{symbol}_codelet"),
            flags,
            retained_symbols: Vec::new(),
        });
        let arguments = registers
            .iter()
            .map(|register| format!("$m{register}"))
            .collect::<Vec<_>>()
            .join(",");
        let frame_bytes = (registers.len() * 4).next_multiple_of(16);
        self.compilations.push(KernelCompilation {
            source: "worker_call.S",
            name: format!("{symbol}_wrapper"),
            flags: vec![
                format!("-DWORKER_CALL_SYMBOL={symbol}"),
                format!("-DWORKER_CODELET_SYMBOL=__runCodelet_{vertex}"),
                format!("-DWORKER_ARGUMENTS={arguments}"),
                format!("-DWORKER_FRAME_BYTES={frame_bytes}"),
            ],
            retained_symbols: vec![symbol.to_owned()],
        });
        self.compilations.last_mut().unwrap()
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
