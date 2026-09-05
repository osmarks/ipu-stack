//! Tile-call contracts and typed scalar arguments.

use super::*;

pub const OUTPUT_REGISTER: u8 = 2;

pub const FIRST_INPUT_REGISTER: u8 = 3;

pub const RETURN_REGISTER: u8 = 10;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelSymbols {
    Exact(&'static str),
    GemmSpecialized,
    AttentionSpecialized,
    AttentionStageSpecialized,
    RearrangeSpecialized,
    UnpackSpecialized,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelAvailability {
    Implemented,
    Required,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarValue {
    ElementCount,
    QueryRows,
    KeyRows,
    NumPartials,
    ScaleExponent,
    InitialBlock,
    FinalBlock,
    WordsPerWorker,
    RemainderWorkers,
    LogicalRows,
    PhysicalRows,
    Matrices,
    LogicalColumns,
    PhysicalColumns,
    TargetOrder,
    LeftBroadcastStride,
    RightBroadcastStride,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelAbi {
    pub symbols: KernelSymbols,
    pub availability: KernelAvailability,
    pub inputs: usize,
    pub scalar_arguments: &'static [ScalarValue],
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum KernelAbiError {
    #[error("kernel requirements do not match the tile-kernel family")]
    RequirementMismatch,
    #[error("kernel run has {actual} pointer operands, ABI requires {expected}")]
    PointerArity { expected: usize, actual: usize },
    #[error("kernel operand {0} is fragmented into multiple views")]
    FragmentedOperand(usize),
    #[error("kernel {0:?} has no device implementation")]
    Unavailable(TileKernelSpec),
    #[error("GEMM output view does not have a matrix row axis")]
    MissingGemmRows,
    #[error("kernel element count overflowed")]
    ElementCountOverflow,
    #[error("kernel {symbol} requires an element count divisible by {divisor}, got {count}")]
    UnsupportedElementCount {
        symbol: &'static str,
        count: u32,
        divisor: u32,
    },
}

pub(super) fn scalar_values(run: &KernelRun, abi: &KernelAbi) -> Result<Vec<u32>, KernelAbiError> {
    let count = element_count(run)?;
    abi.scalar_arguments
        .iter()
        .map(|argument| match argument {
            ScalarValue::ElementCount => Ok(count),
            ScalarValue::QueryRows => gemm_rows(run),
            ScalarValue::KeyRows => match &run.kernel {
                TileKernelSpec::AttentionSoftmax { key_columns, .. } => Ok(*key_columns),
                _ => Err(KernelAbiError::RequirementMismatch),
            },
            ScalarValue::NumPartials => match &run.kernel {
                TileKernelSpec::ReductionSum { partials } => Ok(u32::from(*partials - 1)),
                _ => Err(KernelAbiError::RequirementMismatch),
            },
            ScalarValue::ScaleExponent => match &run.kernel {
                TileKernelSpec::Gemm {
                    multiply: Precision::F8F143 { scale_exponent },
                    ..
                } => Ok(u32::from_ne_bytes(i32::from(*scale_exponent).to_ne_bytes())),
                _ => Err(KernelAbiError::RequirementMismatch),
            },
            ScalarValue::InitialBlock => match &run.kernel {
                TileKernelSpec::AttentionMerge { initial, .. } => Ok(u32::from(*initial)),
                _ => Err(KernelAbiError::RequirementMismatch),
            },
            ScalarValue::FinalBlock => match &run.kernel {
                TileKernelSpec::AttentionMerge { final_block, .. } => Ok(u32::from(*final_block)),
                _ => Err(KernelAbiError::RequirementMismatch),
            },
            ScalarValue::WordsPerWorker => output_byte_count(run).map(|bytes| bytes / 8 / 6),
            ScalarValue::RemainderWorkers => output_byte_count(run).map(|bytes| bytes / 8 % 6),
            ScalarValue::LogicalRows => matrix_extent(run, true, false),
            ScalarValue::PhysicalRows => matrix_extent(run, false, false),
            ScalarValue::Matrices => matrix_count(run),
            ScalarValue::LogicalColumns => matrix_extent(run, true, true),
            ScalarValue::PhysicalColumns => matrix_extent(run, false, true),
            ScalarValue::TargetOrder => match &run.kernel {
                TileKernelSpec::Rearrange {
                    to: crate::Layout { order, .. },
                    ..
                } => RearrangeTarget::from_order(*order)
                    .map(RearrangeTarget::codelet_index)
                    .ok_or(KernelAbiError::RequirementMismatch),
                _ => Err(KernelAbiError::RequirementMismatch),
            },
            ScalarValue::LeftBroadcastStride | ScalarValue::RightBroadcastStride => {
                Err(KernelAbiError::RequirementMismatch)
            }
        })
        .collect()
}

pub(super) fn element_count(run: &KernelRun) -> Result<u32, KernelAbiError> {
    run.output.extents.iter().try_fold(1u32, |product, extent| {
        product
            .checked_mul(extent.physical_end - extent.start)
            .ok_or(KernelAbiError::ElementCountOverflow)
    })
}

pub(super) fn output_byte_count(run: &KernelRun) -> Result<u32, KernelAbiError> {
    let precision = match &run.requirements {
        KernelRequirements::Operator(requirements) => requirements.output.format.precision,
        KernelRequirements::Conversion { output, .. } => output.format.precision,
    };
    element_count(run)?
        .checked_mul(
            u32::try_from(precision.bytes()).map_err(|_| KernelAbiError::ElementCountOverflow)?,
        )
        .ok_or(KernelAbiError::ElementCountOverflow)
}

pub fn tile_kernel_abi(
    kernel: &TileKernelSpec,
    requirements: &KernelRequirements,
) -> Result<KernelAbi, KernelAbiError> {
    let precision = match requirements {
        KernelRequirements::Operator(requirements) => requirements.output.format.precision,
        KernelRequirements::Conversion { output, .. } => output.format.precision,
    };
    let (symbols, availability, inputs, scalars): (_, _, usize, &'static [ScalarValue]) =
        match kernel {
            TileKernelSpec::FillZero => (
                KernelSymbols::Exact(crate::FILL_ZERO_U64_SYMBOL),
                KernelAvailability::Implemented,
                0,
                &[ScalarValue::WordsPerWorker, ScalarValue::RemainderWorkers],
            ),
            TileKernelSpec::Gemm {
                multiply,
                mode,
                weights,
                ..
            } => {
                if !matches!(requirements, KernelRequirements::Operator(_)) {
                    return Err(KernelAbiError::RequirementMismatch);
                }
                if *weights == GemmWeightLoad::Interleaved && *multiply != Precision::F16 {
                    return Err(KernelAbiError::RequirementMismatch);
                }
                let symbols = gemm_symbols(*multiply, *mode);
                let scalars: &'static [ScalarValue] =
                    if matches!(multiply, Precision::F8F143 { .. }) {
                        &[ScalarValue::ScaleExponent]
                    } else {
                        &[]
                    };
                (symbols.0, symbols.1, 2, scalars)
            }
            TileKernelSpec::Gelu => {
                let symbol = gelu_symbol(requirements).unwrap_or("ipu_stack_unsupported_gelu");
                (
                    KernelSymbols::Exact(symbol),
                    if symbol == "ipu_stack_unsupported_gelu" {
                        KernelAvailability::Required
                    } else {
                        KernelAvailability::Implemented
                    },
                    1,
                    &[ScalarValue::ElementCount],
                )
            }
            TileKernelSpec::ReductionSum { .. } => {
                if precision != Precision::F16 {
                    return Err(KernelAbiError::RequirementMismatch);
                }
                (
                    KernelSymbols::Exact("ipu_stack_reduce_sum_f16"),
                    KernelAvailability::Implemented,
                    2,
                    &[ScalarValue::NumPartials, ScalarValue::ElementCount],
                )
            }
            TileKernelSpec::Add => (
                exact_symbol(precision, "ipu_stack_add_f16", "ipu_stack_add_f32"),
                KernelAvailability::Required,
                2,
                &[
                    ScalarValue::ElementCount,
                    ScalarValue::LeftBroadcastStride,
                    ScalarValue::RightBroadcastStride,
                ],
            ),
            TileKernelSpec::FlashAttention { .. } => (
                KernelSymbols::AttentionSpecialized,
                if matches!(requirements, KernelRequirements::Operator(requirements)
                if requirements.output.format.precision == Precision::F32
                    && requirements.inputs.iter().all(|input| input.format.precision == Precision::F16))
                {
                    KernelAvailability::Implemented
                } else {
                    KernelAvailability::Required
                },
                3,
                &[],
            ),
            TileKernelSpec::AttentionSoftmax {
                key_columns,
                padded_key_columns,
                ..
            } => (
                KernelSymbols::AttentionStageSpecialized,
                KernelAvailability::Implemented,
                1,
                if key_columns != padded_key_columns {
                    &[ScalarValue::QueryRows, ScalarValue::KeyRows]
                } else {
                    &[]
                },
            ),
            TileKernelSpec::AttentionMerge { .. } => (
                KernelSymbols::AttentionStageSpecialized,
                KernelAvailability::Implemented,
                2,
                &[
                    ScalarValue::InitialBlock,
                    ScalarValue::FinalBlock,
                    ScalarValue::QueryRows,
                ],
            ),
            TileKernelSpec::Cast { from, to } => (
                KernelSymbols::Exact(cast_symbol(*from, *to)),
                KernelAvailability::Required,
                1,
                &[ScalarValue::ElementCount],
            ),
            TileKernelSpec::Rearrange { from, to }
                if precision == Precision::F16
                    && UnpackSource::from_order(from.order).is_some()
                    && to.order == ElementOrder::RowMajor =>
            {
                (
                    KernelSymbols::UnpackSpecialized,
                    KernelAvailability::Implemented,
                    1,
                    &[
                        ScalarValue::Matrices,
                        ScalarValue::LogicalRows,
                        ScalarValue::PhysicalRows,
                        ScalarValue::LogicalColumns,
                        ScalarValue::PhysicalColumns,
                    ],
                )
            }
            TileKernelSpec::Rearrange { from, to }
                if precision == Precision::F16
                    && from.order == ElementOrder::RowMajor
                    && matches!(
                        to.order,
                        ElementOrder::Amp(AmpOrder::Left | AmpOrder::TransposedRight)
                            | ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. })
                    ) =>
            {
                (
                    KernelSymbols::RearrangeSpecialized,
                    KernelAvailability::Implemented,
                    1,
                    &[
                        ScalarValue::LogicalRows,
                        ScalarValue::PhysicalRows,
                        ScalarValue::TargetOrder,
                        ScalarValue::LogicalColumns,
                        ScalarValue::PhysicalColumns,
                    ],
                )
            }
            TileKernelSpec::Rearrange { .. } => (
                KernelSymbols::Exact("ipu_stack_rearrange"),
                KernelAvailability::Required,
                1,
                &[],
            ),
        };
    Ok(KernelAbi {
        symbols,
        availability,
        inputs,
        scalar_arguments: scalars,
    })
}

pub fn validate_kernel_run(run: &KernelRun) -> Result<KernelAbi, KernelAbiError> {
    let kernel = &run.kernel;
    let abi = tile_kernel_abi(kernel, &run.requirements)?;
    if run.inputs.len() != abi.inputs {
        return Err(KernelAbiError::PointerArity {
            expected: abi.inputs,
            actual: run.inputs.len(),
        });
    }
    if let Some(index) = run
        .inputs
        .iter()
        .position(|operand| operand.views.len() != 1)
    {
        return Err(KernelAbiError::FragmentedOperand(index));
    }
    if matches!(kernel, TileKernelSpec::Gelu) {
        let KernelSymbols::Exact(symbol) = abi.symbols else {
            return Err(KernelAbiError::RequirementMismatch);
        };
        let divisor = 2;
        let count = element_count(run)?;
        if !count.is_multiple_of(divisor) {
            return Err(KernelAbiError::UnsupportedElementCount {
                symbol,
                count,
                divisor,
            });
        }
    }
    if let TileKernelSpec::ReductionSum { partials } = kernel {
        let count = element_count(run)?;
        if *partials < 2 || !count.is_multiple_of(8) {
            return Err(KernelAbiError::UnsupportedElementCount {
                symbol: "ipu_stack_reduce_sum_f16",
                count,
                divisor: 8,
            });
        }
    }
    if matches!(kernel, TileKernelSpec::FillZero) {
        let bytes = output_byte_count(run)?;
        if !bytes.is_multiple_of(8) {
            return Err(KernelAbiError::UnsupportedElementCount {
                symbol: crate::FILL_ZERO_U64_SYMBOL,
                count: bytes,
                divisor: 8,
            });
        }
    }
    Ok(abi)
}

pub(super) fn gelu_symbol(requirements: &KernelRequirements) -> Option<&'static str> {
    let KernelRequirements::Operator(requirements) = requirements else {
        return None;
    };
    let [input] = requirements.inputs.as_slice() else {
        return None;
    };
    if input.format.precision != Precision::F16
        || requirements.output.format.precision != Precision::F16
    {
        return None;
    }
    let input_layout = &input.format.layout;
    let output_layout = &requirements.output.format.layout;
    (input_layout == output_layout).then_some("ipu_stack_gelu_tanh_approx_f16")
}

pub(super) fn gemm_symbols(
    precision: Precision,
    mode: GemmKernelMode,
) -> (KernelSymbols, KernelAvailability) {
    if matches!(precision, Precision::F8F143 { .. }) {
        (
            KernelSymbols::Exact(match mode {
                GemmKernelMode::Initialize => "ipu_stack_gemm_f8_init",
                GemmKernelMode::Accumulate => "ipu_stack_gemm_f8_accumulate",
            }),
            KernelAvailability::Required,
        )
    } else {
        (
            KernelSymbols::GemmSpecialized,
            KernelAvailability::Implemented,
        )
    }
}

pub(super) fn exact_symbol(
    precision: Precision,
    f16_symbol: &'static str,
    f32_symbol: &'static str,
) -> KernelSymbols {
    KernelSymbols::Exact(match precision {
        Precision::F16 => f16_symbol,
        Precision::F32 => f32_symbol,
        Precision::F8F143 { .. } => "ipu_stack_unsupported_f8_kernel",
    })
}

pub(super) fn cast_symbol(from: Precision, to: Precision) -> &'static str {
    match (from, to) {
        (Precision::F16, Precision::F32) => "ipu_stack_cast_f16_f32",
        (Precision::F32, Precision::F16) => "ipu_stack_cast_f32_f16",
        (Precision::F8F143 { .. }, Precision::F16) => "ipu_stack_cast_f8_f16",
        (Precision::F8F143 { .. }, Precision::F32) => "ipu_stack_cast_f8_f32",
        (Precision::F16, Precision::F8F143 { .. }) => "ipu_stack_cast_f16_f8",
        (Precision::F32, Precision::F8F143 { .. }) => "ipu_stack_cast_f32_f8",
        _ => "ipu_stack_cast_identity",
    }
}
