//! Address-independent kernel access contracts and call geometry.

use super::*;

/// Access contract of an actual kernel buffer, without candidate planning policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelAccess {
    pub format: TensorFormat,
    pub alignment: u32,
    pub access_tail_bytes: u32,
}

impl KernelAccess {
    pub fn new(format: TensorFormat, alignment: u32) -> Self {
        Self {
            format,
            alignment,
            access_tail_bytes: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelRequirements {
    pub inputs: Vec<KernelAccess>,
    pub output: KernelAccess,
    pub distinct_elements: Vec<Vec<MemoryOperand>>,
}

impl KernelRequirements {
    pub fn new(
        kernel: &TileKernelSpec,
        inputs: impl IntoIterator<Item = TensorFormat>,
        output: TensorFormat,
    ) -> Self {
        let alignment = match kernel {
            TileKernelSpec::Gemm { .. } => 32,
            TileKernelSpec::Rearrange { .. } => 2,
            _ => 8,
        };
        let mut requirements = Self {
            inputs: inputs
                .into_iter()
                .map(|format| KernelAccess::new(format, alignment))
                .collect(),
            output: KernelAccess::new(output, alignment),
            distinct_elements: Vec::new(),
        };
        if let TileKernelSpec::Gemm { multiply, .. } = kernel
            && let Some(left) = requirements.inputs.first_mut()
        {
            left.access_tail_bytes = 8 * multiply.bytes() as u32;
            requirements
                .distinct_elements
                .push(vec![MemoryOperand::Output, MemoryOperand::Input(0)]);
        }
        requirements
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct AttentionKernelShape {
    pub(crate) matrices: u32,
    pub(crate) query_rows: u32,
    pub(crate) key_rows: u32,
    pub(crate) query_dimension: u32,
    pub(crate) value_dimension: u32,
    pub(crate) scale_bits: u32,
}

pub(crate) fn attention_shape(run: &KernelRun) -> Result<AttentionKernelShape, KernelAbiError> {
    let TileKernelSpec::FlashAttention {
        options,
        accumulate,
    } = &run.kernel
    else {
        return Err(KernelAbiError::RequirementMismatch);
    };
    if options.causal || *accumulate != crate::AccumulationPrecision::F32 {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let [query, key, value] = run.inputs.as_slice() else {
        return Err(KernelAbiError::RequirementMismatch);
    };
    let extents = |operand: &crate::KernelOperand| {
        let [view] = operand.views.as_slice() else {
            return None;
        };
        Some(
            view.extents
                .iter()
                .map(|extent| extent.physical_end - extent.start)
                .collect::<Vec<_>>(),
        )
    };
    let query = extents(query).ok_or(KernelAbiError::RequirementMismatch)?;
    let key = extents(key).ok_or(KernelAbiError::RequirementMismatch)?;
    let value = extents(value).ok_or(KernelAbiError::RequirementMismatch)?;
    if query.len() < 2 || query.len() != key.len() || query.len() != value.len() {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let rank = query.len();
    if query[..rank - 2] != key[..rank - 2]
        || query[..rank - 2] != value[..rank - 2]
        || query[rank - 1] != key[rank - 1]
        || key[rank - 2] != value[rank - 2]
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let matrices = query[..rank - 2]
        .iter()
        .try_fold(1u32, |product, &extent| product.checked_mul(extent))
        .ok_or(KernelAbiError::ElementCountOverflow)?;
    let scale = options
        .scale
        .as_value()
        .unwrap_or_else(|| 1.0 / (query[rank - 1] as f32).sqrt());
    Ok(AttentionKernelShape {
        matrices,
        query_rows: query[rank - 2],
        key_rows: key[rank - 2],
        query_dimension: query[rank - 1],
        value_dimension: value[rank - 1],
        scale_bits: scale.to_bits(),
    })
}

pub(crate) fn gemm_rows(run: &KernelRun) -> Result<u32, KernelAbiError> {
    let rank = run.output.extents.len();
    let output_order = &run.requirements.output.format.layout.order;
    let matrix_column_axis = rank
        .checked_sub(if output_order.gemm_output_transposed() {
            2
        } else {
            1
        })
        .ok_or(KernelAbiError::MissingGemmRows)?;
    run.output
        .extents
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != matrix_column_axis)
        .try_fold(1u32, |rows, extent| {
            rows.checked_mul(extent.1.physical_end - extent.1.start)
        })
        .filter(|&rows| rows != 0)
        .ok_or(KernelAbiError::MissingGemmRows)
}

pub(crate) fn matrix_extent(
    run: &KernelRun,
    logical: bool,
    columns: bool,
) -> Result<u32, KernelAbiError> {
    let rank = run.output.extents.len();
    let axis = rank
        .checked_sub(if columns { 1 } else { 2 })
        .ok_or(KernelAbiError::RequirementMismatch)?;
    let extent = &run.output.extents[axis];
    Ok(if logical {
        extent.logical_end - extent.start
    } else {
        extent.physical_end - extent.start
    })
}

pub(crate) fn input_matrix_extent(
    run: &KernelRun,
    logical: bool,
    columns: bool,
) -> Result<u32, KernelAbiError> {
    let view = run
        .inputs
        .first()
        .and_then(|operand| operand.views.first())
        .ok_or(KernelAbiError::RequirementMismatch)?;
    let rank = view.extents.len();
    let axis = rank
        .checked_sub(if columns { 1 } else { 2 })
        .ok_or(KernelAbiError::RequirementMismatch)?;
    let extent = &view.extents[axis];
    Ok(if logical {
        extent.logical_end - extent.start
    } else {
        extent.physical_end - extent.start
    })
}

pub(crate) fn matrix_count(run: &KernelRun) -> Result<u32, KernelAbiError> {
    let view = run
        .inputs
        .first()
        .and_then(|operand| operand.views.first())
        .ok_or(KernelAbiError::RequirementMismatch)?;
    view.extents[..view.extents.len().saturating_sub(2)]
        .iter()
        .try_fold(1u32, |product, extent| {
            product
                .checked_mul(extent.physical_end - extent.start)
                .ok_or(KernelAbiError::ElementCountOverflow)
        })
}
