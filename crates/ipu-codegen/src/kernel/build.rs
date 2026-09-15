//! Device object recipes for the selected kernel specializations.

use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelCompilation {
    pub source: &'static str,
    pub name: String,
    pub flags: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelBuildPlan {
    pub compilations: Vec<KernelCompilation>,
    /// All callable entry points, including kernels supplied by the runtime object.
    pub(super) symbols: BTreeMap<KernelImplementation, String>,
}

impl KernelBuildPlan {
    /// Collect resolved implementation keys; only GEMM needs additional grouping
    /// to share worker objects across row-count variants.
    pub fn from_program(program: &crate::TileGraph) -> Result<Self, KernelAbiError> {
        let implementations = program
            .body
            .walk()
            .filter_map(|work| match work {
                BlockOperation::Compute { run, .. } => Some(
                    program.kernel_runs[run.0 as usize]
                        .call()
                        .map(|call| call.implementation),
                ),
                BlockOperation::Copy { copy, .. } => Some(Ok(KernelImplementation::Exact(
                    program.local_copies[copy.0 as usize].symbol(),
                ))),
                _ => None,
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        Self::from_implementations(implementations)
    }

    pub(super) fn from_implementations(
        implementations: BTreeSet<KernelImplementation>,
    ) -> Result<Self, KernelAbiError> {
        let exact_symbols = implementations
            .iter()
            .filter_map(|key| match key {
                KernelImplementation::Exact(symbol) => Some(*symbol),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let mut plan = Self {
            compilations: Vec::new(),
            symbols: exact_symbols
                .iter()
                .map(|&symbol| (KernelImplementation::Exact(symbol), symbol.to_owned()))
                .collect(),
        };
        plan.add_gemms(&implementations);
        plan.add_normalization(&exact_symbols);
        plan.add_pointwise(&exact_symbols);
        plan.add_reduction(&exact_symbols);
        plan.add_casts(&exact_symbols);
        for key in &implementations {
            if matches!(
                key,
                KernelImplementation::Unpack(_) | KernelImplementation::Rearrange(_)
            ) {
                plan.add_rearrangement(key);
            }
        }
        plan.add_attention(&implementations)?;
        // Only compiler-generated C++ codelets need the worker stack symbols.
        // Derive this from the selected recipes, including assembly fast paths.
        if plan
            .compilations
            .iter()
            .any(|unit| unit.source.ends_with(".cpp"))
        {
            plan.add_compilation(KernelCompilation {
                source: "worker_support.S",
                name: "worker_support".into(),
                flags: Vec::new(),
            });
        }
        Ok(plan)
    }

    /// Register each named object once. Conflicting definitions are a compiler bug.
    pub(super) fn add_compilation(&mut self, unit: KernelCompilation) {
        if let Some(existing) = self.compilations.iter().find(|old| old.name == unit.name) {
            assert_eq!(existing, &unit, "conflicting kernel object definitions");
        } else {
            self.compilations.push(unit);
        }
    }

    /// Compile a C++ vertex with its assembly entry point. Nonempty registers
    /// request generic argument marshalling in vertex-field order; custom wrappers
    /// supply their own marshalling and flags.
    pub(super) fn add_vertex(
        &mut self,
        source: &'static str,
        symbol: &str,
        vertex: &str,
        flags: Vec<String>,
        registers: &[u8],
        wrapper: &'static str,
        mut wrapper_flags: Vec<String>,
    ) {
        self.add_compilation(KernelCompilation {
            source,
            name: format!("{symbol}_codelet"),
            flags,
        });
        if !registers.is_empty() {
            let arguments = registers
                .iter()
                .map(|register| format!("$m{register}"))
                .collect::<Vec<_>>()
                .join(",");
            let frame_bytes = (registers.len() * 4).next_multiple_of(16);
            wrapper_flags.extend([
                format!("-DWORKER_CALL_SYMBOL={symbol}"),
                format!("-DWORKER_CODELET_SYMBOL=__runCodelet_{vertex}"),
                format!("-DWORKER_ARGUMENTS={arguments}"),
                format!("-DWORKER_FRAME_BYTES={frame_bytes}"),
            ]);
        }
        self.add_compilation(KernelCompilation {
            source: wrapper,
            name: format!("{symbol}_wrapper"),
            flags: wrapper_flags,
        });
    }

    pub(super) fn symbol<'a>(
        &'a self,
        implementation: &KernelImplementation,
    ) -> Result<&'a str, KernelAbiError> {
        match implementation {
            KernelImplementation::Exact(symbol) => Ok(symbol),
            _ => self
                .symbols
                .get(implementation)
                .map(String::as_str)
                .ok_or(KernelAbiError::RequirementMismatch),
        }
    }

    pub fn retained_symbols(&self) -> impl Iterator<Item = &str> {
        self.symbols.values().map(String::as_str)
    }
}
