//! Device object collection and shared compilation/wrapper construction.

use super::*;
use ipu_target::Target;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelCompilation {
    pub source: &'static str,
    pub name: String,
    pub flags: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelObjects {
    pub compilations: Vec<KernelCompilation>,
    /// All callable entry points, including kernels supplied by the runtime object.
    pub(super) symbols: BTreeSet<String>,
}

impl KernelObjects {
    /// Family construction registers objects directly; this collector only deduplicates them.
    pub fn from_program(target: Target, program: &crate::TileGraph) -> Result<Self, KernelError> {
        let mut plan = Self::default();
        for work in program.body.walk() {
            match work {
                BlockOperation::Compute { run, .. } => {
                    program.kernel_runs[run.0 as usize].call(target, Some(&mut plan))?;
                }
                BlockOperation::Copy { copy, .. } => {
                    plan.symbols
                        .insert(program.local_copies[copy.0 as usize].symbol().into());
                }
                _ => {}
            }
        }
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

    pub fn retained_symbols(&self) -> impl Iterator<Item = &str> {
        self.symbols.iter().map(String::as_str)
    }
}
