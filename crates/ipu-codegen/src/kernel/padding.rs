//! Padding exceptions of resolved calls. ABI bounds determine skipped reads;
//! graph provenance and the decision to omit initialization belong to low.

use super::*;
use crate::low::ShardView;
use crate::mid::MidOperationKind;

pub(crate) enum PaddingRequirement {
    Required,
    Unread(Vec<ShardView>),
    FiniteIfZero { region: ShardView, zero: ShardView },
}

impl KernelCall {
    pub(crate) fn input_padding(
        &self,
        run: &KernelRun,
        operand: usize,
    ) -> Result<PaddingRequirement, KernelAbiError> {
        let input = run
            .inputs
            .get(operand)
            .ok_or(KernelAbiError::RequirementMismatch)?;
        if operand != 0 {
            return Ok(PaddingRequirement::Required);
        }
        match &self.implementation {
            KernelImplementation::Exact("cast_f16_f8") => {
                let [_, row_bounds, _, panel_rows, columns, stride] = self.arguments.as_slice()
                else {
                    return Err(KernelAbiError::RequirementMismatch);
                };
                if *stride == 0 || *panel_rows == 0 {
                    return Ok(PaddingRequirement::Required);
                }
                let rank = input.extents.len();
                if rank < 2 {
                    return Err(KernelAbiError::RequirementMismatch);
                }
                // These are the actual bounds decoded by cast_f8.cpp. A zero
                // descriptor reads physical rows, including on the overflow fallback.
                let rows = if *row_bounds == 0 {
                    input_matrix_extent(run, false, false)?
                } else {
                    row_bounds & 0xffff
                };
                let mut regions = Vec::new();
                for (axis, count) in [(rank - 2, rows), (rank - 1, *columns)] {
                    let mut region = input.clone();
                    // Outer padding is not covered by the matrix descriptor.
                    for extent in &mut region.extents[..rank - 2] {
                        extent.physical_end = extent.logical_end;
                    }
                    let extent = &mut region.extents[axis];
                    extent.start = extent
                        .start
                        .checked_add(count)
                        .ok_or(KernelAbiError::ElementCountOverflow)?;
                    extent.logical_end = extent.logical_end.max(extent.start);
                    if extent.start < extent.physical_end {
                        regions.push(region);
                    }
                }
                Ok(PaddingRequirement::Unread(regions))
            }
            KernelImplementation::Gemm(Precision::F16, _, inner, _, _, _, _) => {
                let MidOperationKind::Gemm { axes, .. } = run.kernel else {
                    return Err(KernelAbiError::RequirementMismatch);
                };
                let mut region = input.clone();
                let mut zero = run.inputs[1].clone();
                let left = axes
                    .left_inner
                    .resolve(region.extents.len())
                    .map_err(|_| KernelAbiError::RequirementMismatch)?;
                let right = axes
                    .right_inner
                    .resolve(zero.extents.len())
                    .map_err(|_| KernelAbiError::RequirementMismatch)?;
                let l = region.extents[left];
                let r = zero.extents[right];
                if l.start != r.start {
                    return Ok(PaddingRequirement::Required);
                }
                let start = l.logical_end.max(r.logical_end);
                let end = l.physical_end.min(r.physical_end).min(
                    l.start
                        .checked_add(*inner)
                        .ok_or(KernelAbiError::ElementCountOverflow)?,
                );
                if start >= end {
                    return Ok(PaddingRequirement::Required);
                }
                // Discarded rows must remain zero; finite values there can
                // overflow even though the corresponding outputs are unused.
                for (axis, extent) in region.extents.iter_mut().enumerate() {
                    if axis != left {
                        extent.physical_end = extent.logical_end;
                    }
                }
                for (view, axis) in [(&mut region, left), (&mut zero, right)] {
                    view.extents[axis].start = start;
                    view.extents[axis].logical_end = start;
                    view.extents[axis].physical_end = end;
                }
                Ok(PaddingRequirement::FiniteIfZero { region, zero })
            }
            // The current FP8 GeLU ABI consumes every physical input row and
            // requires logical width == physical width (output::fp8_arguments).
            // It therefore has no unread input padding to report.
            _ => Ok(PaddingRequirement::Required),
        }
    }
}
