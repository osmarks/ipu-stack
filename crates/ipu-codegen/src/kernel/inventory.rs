//! Collect complete call identities and group compatible GEMM row variants.

use super::*;

#[derive(Default)]
pub(super) struct KernelInventory {
    pub(super) exact_symbols: BTreeSet<&'static str>,
    pub(super) rows:
        BTreeMap<(Precision, GemmWeightLoad, u32, u32, u32), BTreeSet<(u32, GemmKernelMode)>>,
    pub(super) rearrangements: BTreeSet<(RearrangeTarget, u32, u32, u32, u32)>,
    pub(super) unpacks: BTreeSet<(UnpackSource, u32, u32, u32, u32)>,
    pub(super) attention: BTreeSet<AttentionKernelShape>,
    pub(super) attention_stages: BTreeSet<KernelImplementation>,
}

impl KernelInventory {
    pub(super) fn collect(&mut self, program: &crate::TileGraph) -> Result<(), KernelAbiError> {
        for work in program.body.walk() {
            match work {
                BlockOperation::Compute { run, .. } => {
                    match program.kernel_runs[run.0 as usize].call()?.implementation {
                        KernelImplementation::Exact(symbol) => {
                            self.exact_symbols.insert(symbol);
                        }
                        KernelImplementation::Gemm(
                            precision,
                            weights,
                            inner,
                            columns,
                            mode,
                            rows,
                            output_group,
                        ) => {
                            self.rows
                                .entry((precision, weights, inner, columns, output_group))
                                .or_default()
                                .insert((rows, mode));
                        }
                        KernelImplementation::Attention(shape) => {
                            self.attention.insert(shape);
                        }
                        KernelImplementation::Rearrange(shape) => {
                            self.rearrangements.insert(shape);
                        }
                        KernelImplementation::Unpack(shape) => {
                            self.unpacks.insert(shape);
                        }
                        stage @ (KernelImplementation::Softmax(..)
                        | KernelImplementation::Merge(..)) => {
                            self.attention_stages.insert(stage);
                        }
                    }
                }
                BlockOperation::Copy { copy, .. } => {
                    self.exact_symbols
                        .insert(program.local_copies[copy.0 as usize].symbol());
                }
                BlockOperation::Exchange(_)
                | BlockOperation::Checkpoint(..)
                | BlockOperation::Repeat(_) => {}
            }
        }
        Ok(())
    }
}
