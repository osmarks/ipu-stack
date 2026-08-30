#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelCompilation {
    pub source: &'static str,
    pub name: String,
    pub optimization: Option<KernelOptimization>,
    pub definitions: Vec<(&'static str, String)>,
    pub retained_symbols: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelOptimization {
    Size,
    Speed,
}

impl KernelCompilation {
    /// Produces textual options only at the toolchain boundary.
    pub fn compiler_flags(&self) -> Vec<String> {
        self.optimization
            .map(|optimization| match optimization {
                KernelOptimization::Size => "-Os".to_owned(),
                KernelOptimization::Speed => "-O2".to_owned(),
            })
            .into_iter()
            .chain(
                self.definitions
                    .iter()
                    .map(|(name, value)| format!("-D{name}={value}")),
            )
            .collect()
    }
}
