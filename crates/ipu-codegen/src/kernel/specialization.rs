//! Shared specialization keys and collection from scheduled calls.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum RearrangeTarget {
    AmpLeft,
    AmpTransposedRight,
    BlockMajor { row_block: u16, column_block: u16 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum UnpackSource {
    AmpOutput,
    AmpTransposedLeft,
}

impl UnpackSource {
    pub(super) fn from_order(order: ElementOrder) -> Option<Self> {
        match order {
            ElementOrder::Amp(AmpOrder::Output) => Some(Self::AmpOutput),
            ElementOrder::Amp(AmpOrder::TransposedLeft) => Some(Self::AmpTransposedLeft),
            _ => None,
        }
    }

    pub(super) const fn codelet_index(self) -> u32 {
        match self {
            Self::AmpOutput => 0,
            Self::AmpTransposedLeft => 1,
        }
    }
}

impl RearrangeTarget {
    pub(super) fn from_order(order: ElementOrder) -> Option<Self> {
        match order {
            ElementOrder::Amp(AmpOrder::Left) => Some(Self::AmpLeft),
            ElementOrder::Amp(AmpOrder::TransposedRight) => Some(Self::AmpTransposedRight),
            ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                row_block,
                column_block,
            }) => Some(Self::BlockMajor {
                row_block,
                column_block,
            }),
            _ => None,
        }
    }

    pub(super) const fn codelet_index(self) -> u32 {
        match self {
            Self::AmpLeft => 0,
            Self::AmpTransposedRight => 1,
            Self::BlockMajor { .. } => 2,
        }
    }
}

/// The same key selects a build recipe and resolves its eventual call.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum KernelSpecialization {
    Gemm(Precision, GemmWeightLoad, u32, u32, GemmKernelMode, u32),
    Attention(AttentionKernelShape),
    Softmax(u32, u32, u32, u32),
    Merge(u32, u32, u32, u32),
    Rearrange((RearrangeTarget, u32, u32, u32, u32)),
    Unpack((UnpackSource, u32, u32, u32, u32)),
}

impl KernelSpecialization {
    pub(super) fn stage(kernel: &TileKernelSpec, rows: u32) -> Result<Self, KernelAbiError> {
        Ok(match kernel {
            TileKernelSpec::AttentionSoftmax {
                head_dimension,
                key_columns,
                padded_key_columns,
            } => Self::Softmax(*head_dimension, *key_columns, *padded_key_columns, rows),
            TileKernelSpec::AttentionMerge {
                value_dimension,
                padded_value_dimension,
                key_block_columns,
                ..
            } => Self::Merge(
                *value_dimension,
                *padded_value_dimension,
                *key_block_columns,
                rows,
            ),
            _ => return Err(KernelAbiError::RequirementMismatch),
        })
    }

    pub(super) fn from_run(run: &KernelRun) -> Result<Self, KernelAbiError> {
        let kernel = &run.kernel;
        Ok(match kernel {
            TileKernelSpec::Gemm {
                multiply,
                weights,
                inner_block,
                output_columns,
                mode,
                ..
            } => Self::Gemm(
                *multiply,
                *weights,
                *inner_block,
                *output_columns,
                *mode,
                gemm_rows(run)?,
            ),
            TileKernelSpec::FlashAttention { .. } => Self::Attention(attention_shape(run)?),
            TileKernelSpec::AttentionSoftmax { .. } | TileKernelSpec::AttentionMerge { .. } => {
                Self::stage(kernel, gemm_rows(run)?)?
            }
            TileKernelSpec::Rearrange { from, to } if from.order == ElementOrder::RowMajor => {
                Self::Rearrange(rearrangement_specialization(
                    RearrangeTarget::from_order(to.order)
                        .ok_or(KernelAbiError::RequirementMismatch)?,
                    matrix_extent(run, true, false)?,
                    matrix_extent(run, false, false)?,
                    matrix_extent(run, true, true)?,
                    matrix_extent(run, false, true)?,
                ))
            }
            TileKernelSpec::Rearrange { from, to } if to.order == ElementOrder::RowMajor => {
                Self::Unpack((
                    UnpackSource::from_order(from.order)
                        .ok_or(KernelAbiError::RequirementMismatch)?,
                    input_matrix_extent(run, true, false)?,
                    input_matrix_extent(run, false, false)?,
                    input_matrix_extent(run, true, true)?,
                    input_matrix_extent(run, false, true)?,
                ))
            }
            _ => return Err(KernelAbiError::RequirementMismatch),
        })
    }
}

#[derive(Default)]
pub(super) struct KernelInventory {
    pub(super) rows: BTreeMap<(Precision, GemmWeightLoad, u32, u32), BTreeSet<u32>>,
    pub(super) gelu: bool,
    pub(super) reduction_add: bool,
    pub(super) rearrangements: BTreeSet<(RearrangeTarget, u32, u32, u32, u32)>,
    pub(super) unpacks: BTreeSet<(UnpackSource, u32, u32, u32, u32)>,
    pub(super) attention: BTreeSet<AttentionKernelShape>,
    pub(super) attention_stages: BTreeSet<KernelSpecialization>,
}

impl KernelInventory {
    pub(super) fn collect(
        &mut self,
        program: &LowProgram,
        tile: &TileWorkList,
    ) -> Result<(), KernelAbiError> {
        for work in program.work(tile) {
            match work {
                TileWorkRef::Kernel(run) => {
                    let abi = validate_kernel_run(run)?;
                    let kernel = &run.kernel;
                    if abi.availability != KernelAvailability::Implemented {
                        return Err(KernelAbiError::Unavailable(kernel.clone()));
                    }
                    if matches!(abi.symbols, KernelSymbols::Exact(_)) {
                        self.gelu |= matches!(kernel, TileKernelSpec::Gelu);
                        self.reduction_add |= matches!(kernel, TileKernelSpec::ReductionSum { .. });
                        continue;
                    }
                    match KernelSpecialization::from_run(run)? {
                        KernelSpecialization::Gemm(precision, weights, inner, columns, _, rows) => {
                            self.rows
                                .entry((precision, weights, inner, columns))
                                .or_default()
                                .insert(rows);
                        }
                        KernelSpecialization::Attention(shape) => {
                            self.attention.insert(shape);
                        }
                        KernelSpecialization::Rearrange(shape) => {
                            self.rearrangements.insert(shape);
                        }
                        KernelSpecialization::Unpack(shape) => {
                            self.unpacks.insert(shape);
                        }
                        stage @ (KernelSpecialization::Softmax(..)
                        | KernelSpecialization::Merge(..)) => {
                            self.attention_stages.insert(stage);
                        }
                    }
                }
                TileWorkRef::Repeat(repeat) => self.collect(program, &repeat.body)?,
                TileWorkRef::Exchange(_)
                | TileWorkRef::LocalCopy(_)
                | TileWorkRef::Checkpoint(..) => {}
            }
        }
        Ok(())
    }
}

pub(super) fn rearrangement_specialization(
    order: RearrangeTarget,
    logical_rows: u32,
    physical_rows: u32,
    logical_columns: u32,
    physical_columns: u32,
) -> (RearrangeTarget, u32, u32, u32, u32) {
    if physical_rows == AMP_INNER_BLOCK
        && logical_rows < physical_rows
        && matches!(
            order,
            RearrangeTarget::AmpTransposedRight | RearrangeTarget::BlockMajor { .. }
        )
    {
        (order, 0, physical_rows, 0, physical_columns)
    } else {
        (
            order,
            logical_rows,
            physical_rows,
            logical_columns,
            physical_columns,
        )
    }
}
