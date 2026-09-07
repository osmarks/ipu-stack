//! Tile-call contracts and typed scalar arguments.

use super::*;

pub const OUTPUT_REGISTER: u8 = 2;

pub const FIRST_INPUT_REGISTER: u8 = 3;

pub const RETURN_REGISTER: u8 = 10;

fn fp8_scale_argument(scale: i32) -> Result<u32, KernelAbiError> {
    if (-32..=31).contains(&scale) {
        Ok(u32::from_ne_bytes(scale.to_ne_bytes()))
    } else {
        Err(KernelAbiError::Fp8Scale(scale))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelSymbols {
    Exact(&'static str),
    Specialized,
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
    SplitSoftmaxRows,
    NumPartials,
    ScaleExponent,
    CastSourceScale,
    CastDestinationScale,
    CastPanelRows,
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
            ScalarValue::SplitSoftmaxRows => match &run.kernel {
                TileKernelSpec::AttentionSoftmax {
                    key_columns,
                    padded_key_columns,
                    ..
                } => Ok(u32::from(cost::f16_softmax_split_rows(
                    u64::from(gemm_rows(run)?),
                    u64::from(*key_columns),
                    u64::from(*padded_key_columns),
                ))),
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
                } => fp8_scale_argument(2 * i32::from(*scale_exponent)),
                _ => Err(KernelAbiError::RequirementMismatch),
            },
            ScalarValue::CastSourceScale | ScalarValue::CastDestinationScale => match run.kernel {
                TileKernelSpec::Cast { from, to } => {
                    let precision = if *argument == ScalarValue::CastSourceScale {
                        from
                    } else {
                        to
                    };
                    let scale = match precision {
                        Precision::F8F143 { scale_exponent } => i32::from(scale_exponent),
                        _ => 0,
                    };
                    fp8_scale_argument(scale)
                }
                _ => Err(KernelAbiError::RequirementMismatch),
            },
            ScalarValue::CastPanelRows => {
                if matches!(
                    run.kernel,
                    TileKernelSpec::Cast {
                        from: Precision::F16,
                        to: Precision::F8F143 { .. }
                    }
                ) {
                    let order = run.requirements.output.format.layout.order;
                    if order.fp8_cast_panel_rows(1, 1) == 0 {
                        return Ok(0);
                    }
                    let columns = matrix_extent(run, false, true)?;
                    let rows = element_count(run)? / columns;
                    u32::try_from(
                        run.requirements
                            .output
                            .format
                            .layout
                            .order
                            .fp8_cast_panel_rows(u64::from(rows), u64::from(columns)),
                    )
                    .map_err(|_| KernelAbiError::RequirementMismatch)
                } else {
                    Ok(0)
                }
            }
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
    if let TileKernelSpec::FillZero { bytes, .. } = run.kernel {
        return Ok(bytes);
    }
    let precision = run.requirements.output.format.precision;
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
    let precision = requirements.output.format.precision;
    let (symbols, availability, inputs, scalars): (_, _, usize, &'static [ScalarValue]) =
        match kernel {
            TileKernelSpec::FillZero { .. } => (
                KernelSymbols::Exact(crate::FILL_ZERO_U64_SYMBOL),
                KernelAvailability::Implemented,
                0,
                &[ScalarValue::WordsPerWorker, ScalarValue::RemainderWorkers],
            ),
            TileKernelSpec::Gemm {
                multiply,
                accumulate,
                weights,
                ..
            } => {
                if requirements.inputs.len() != 2 {
                    return Err(KernelAbiError::RequirementMismatch);
                }
                if *weights == GemmWeightLoad::Interleaved && *multiply == Precision::F32 {
                    return Err(KernelAbiError::RequirementMismatch);
                }
                if matches!(multiply, Precision::F8F143 { .. })
                    && (*accumulate != crate::AccumulationPrecision::F16
                        || precision != Precision::F16)
                {
                    return Err(KernelAbiError::RequirementMismatch);
                }
                let scalars: &'static [ScalarValue] =
                    if matches!(multiply, Precision::F8F143 { .. }) {
                        &[ScalarValue::ScaleExponent]
                    } else {
                        &[]
                    };
                (
                    KernelSymbols::Specialized,
                    KernelAvailability::Implemented,
                    2,
                    scalars,
                )
            }
            TileKernelSpec::Gelu => {
                let symbol = gelu_symbol(requirements).unwrap_or("unsupported_gelu");
                (
                    KernelSymbols::Exact(symbol),
                    if symbol == "unsupported_gelu" {
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
                    KernelSymbols::Exact("reduce_sum_f16"),
                    KernelAvailability::Implemented,
                    2,
                    &[ScalarValue::NumPartials, ScalarValue::ElementCount],
                )
            }
            TileKernelSpec::Add => (
                exact_symbol(precision, "add_f16", "add_f32"),
                KernelAvailability::Required,
                2,
                &[
                    ScalarValue::ElementCount,
                    ScalarValue::LeftBroadcastStride,
                    ScalarValue::RightBroadcastStride,
                ],
            ),
            TileKernelSpec::FlashAttention { .. } => (
                KernelSymbols::Specialized,
                if requirements.output.format.precision == Precision::F32
                    && requirements
                        .inputs
                        .iter()
                        .all(|input| input.format.precision == Precision::F16)
                {
                    KernelAvailability::Implemented
                } else {
                    KernelAvailability::Required
                },
                3,
                &[],
            ),
            TileKernelSpec::AttentionSoftmax { .. } => (
                KernelSymbols::Specialized,
                KernelAvailability::Implemented,
                1,
                &[
                    ScalarValue::QueryRows,
                    ScalarValue::KeyRows,
                    ScalarValue::SplitSoftmaxRows,
                ],
            ),
            TileKernelSpec::AttentionMerge { .. } => (
                KernelSymbols::Specialized,
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
                if (*from, *to) == (Precision::F32, Precision::F16)
                    || matches!(from, Precision::F8F143 { .. })
                    || matches!(to, Precision::F8F143 { .. })
                {
                    KernelAvailability::Implemented
                } else {
                    KernelAvailability::Required
                },
                1,
                if matches!(from, Precision::F8F143 { .. })
                    || matches!(to, Precision::F8F143 { .. })
                {
                    &[
                        ScalarValue::ElementCount,
                        ScalarValue::CastSourceScale,
                        ScalarValue::CastDestinationScale,
                        ScalarValue::CastPanelRows,
                    ]
                } else {
                    &[ScalarValue::ElementCount]
                },
            ),
            TileKernelSpec::Rearrange { from, to }
                if precision == Precision::F16
                    && UnpackSource::from_order(from.order).is_some()
                    && to.order == ElementOrder::RowMajor =>
            {
                (
                    KernelSymbols::Specialized,
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
                    KernelSymbols::Specialized,
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
                KernelSymbols::Exact("rearrange"),
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
    if matches!(
        kernel,
        TileKernelSpec::Cast {
            from: Precision::F16,
            to: Precision::F8F143 { .. }
        }
    ) {
        let panel_rows = scalar_values(run, &abi)?[3];
        let input = &run.inputs[0].views[0];
        if run.requirements.inputs[0].format.layout.order
            != run.requirements.output.format.layout.order
            || input.extents.len() != run.output.extents.len()
            || input
                .extents
                .iter()
                .zip(&run.output.extents)
                .any(|(from, to)| from.physical_end - from.start != to.physical_end - to.start)
            || (panel_rows != 0
                && !element_count(run)?.is_multiple_of(
                    panel_rows
                        .checked_mul(32)
                        .ok_or(KernelAbiError::ElementCountOverflow)?,
                ))
        {
            return Err(KernelAbiError::RequirementMismatch);
        }
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
                symbol: "reduce_sum_f16",
                count,
                divisor: 8,
            });
        }
    }
    if matches!(kernel, TileKernelSpec::FillZero { .. }) {
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
    (input_layout == output_layout).then_some("gelu_tanh_approx_f16")
}

pub(super) fn exact_symbol(
    precision: Precision,
    f16_symbol: &'static str,
    f32_symbol: &'static str,
) -> KernelSymbols {
    KernelSymbols::Exact(match precision {
        Precision::F16 => f16_symbol,
        Precision::F32 => f32_symbol,
        Precision::F8F143 { .. } => "unsupported_f8_kernel",
    })
}

pub(super) fn cast_symbol(from: Precision, to: Precision) -> &'static str {
    match (from, to) {
        (Precision::F16, Precision::F32) => "cast_f16_f32",
        (Precision::F32, Precision::F16) => "cast_f32_f16",
        (Precision::F8F143 { .. }, Precision::F16) => "cast_f8_f16",
        (Precision::F8F143 { .. }, Precision::F32) => "cast_f8_f32",
        (Precision::F16, Precision::F8F143 { .. }) => "cast_f16_f8",
        (Precision::F32, Precision::F8F143 { .. }) => "cast_f32_f8",
        (Precision::F8F143 { .. }, Precision::F8F143 { .. }) => "cast_f8_f8",
        _ => "cast_identity",
    }
}
