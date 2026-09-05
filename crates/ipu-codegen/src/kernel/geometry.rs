//! Shape and extent extraction for kernel specialization.

use super::*;

pub(super) fn attention_shape(run: &KernelRun) -> Result<AttentionKernelShape, KernelAbiError> {
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

pub(super) fn gemm_rows(run: &KernelRun) -> Result<u32, KernelAbiError> {
    let rank = run.output.extents.len();
    let output_order = &run.requirements.output.format.layout.order;
    let matrix_column_axis = rank
        .checked_sub(
            if matches!(
                output_order,
                ElementOrder::Amp(AmpOrder::TransposedOutput | AmpOrder::TransposedLeft)
            ) {
                2
            } else {
                1
            },
        )
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

pub(super) fn matrix_extent(
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

pub(super) fn input_matrix_extent(
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

pub(super) fn matrix_count(run: &KernelRun) -> Result<u32, KernelAbiError> {
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
