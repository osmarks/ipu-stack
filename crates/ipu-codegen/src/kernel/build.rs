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
            fp8_casts,
            rearrangements,
            unpacks,
            attention,
            attention_stages,
        } = inventory;
        let mut plan = Self::default();
        for (configuration, rows) in rows {
            plan.add_gemm(configuration, rows);
        }
        for (symbol, extra) in [
            ("layer_norm_f16", None),
            ("add_layer_norm_f16", Some("-DNORM_WITH_ADD")),
        ] {
            if !exact_symbols.contains(symbol) {
                continue;
            }
            let flags = extra.into_iter().map(str::to_owned).collect::<Vec<_>>();
            let mut codelet_flags = vec!["-O2".into(), "-DVERTEX_LayerNormF16".into()];
            codelet_flags.extend(flags.clone());
            plan.compilations.extend([
                KernelCompilation {
                    source: "elementwise_f16.cpp",
                    name: format!("{symbol}_codelet"),
                    flags: codelet_flags,
                    retained_symbols: vec![],
                },
                KernelCompilation {
                    source: "layer_norm_f16.S",
                    name: format!("{symbol}_wrapper"),
                    flags,
                    retained_symbols: vec![symbol.into()],
                },
            ]);
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
        for (symbol, extra) in [
            ("gelu_tanh_approx_f16", None),
            ("bias_gelu_f16", Some("-DGELU_WITH_BIAS")),
        ] {
            if exact_symbols.contains(symbol) {
                plan.compilations.push(KernelCompilation {
                    source: "gelu_f16.S",
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
        if exact_symbols.contains("layer_norm_moments") {
            plan.compilations.push(KernelCompilation {
                source: "layer_norm_distributed.cpp",
                name: "layer_norm_moments_codelet".into(),
                flags: vec!["-O2".into(), "-DVERTEX_LayerNormMoments".into()],
                retained_symbols: Vec::new(),
            });
            plan.compilations.push(KernelCompilation {
                source: "layer_norm_moments.S",
                name: "layer_norm_moments_wrapper".into(),
                flags: Vec::new(),
                retained_symbols: vec!["layer_norm_moments".into()],
            });
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
        let has_worker_codelets = [
            "layer_norm_f16",
            "add_layer_norm_f16",
            "layer_norm_moments",
            "layer_norm_apply",
            "add_f16",
            "cast_f32_f16",
        ]
        .iter()
        .any(|symbol| exact_symbols.contains(symbol))
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
            plan.add_vertex(
                "cast_f8.cpp",
                &symbol,
                &vertex,
                vec![
                    "-O2".into(),
                    format!("-DINPUT_BYTES={from}"),
                    format!("-DOUTPUT_BYTES={to}"),
                    format!("-DCAST_VERTEX={vertex}"),
                ],
                &[3, 2, 4, 5, 6, 7, 8, 9],
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

    /// Compile a C++ vertex and marshal supervisor registers into its argument block.
    /// The register order follows the vertex fields, independently of call ABI.
    pub(super) fn add_vertex(
        &mut self,
        source: &'static str,
        symbol: &str,
        vertex: &str,
        flags: Vec<String>,
        registers: &[u8],
    ) {
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
