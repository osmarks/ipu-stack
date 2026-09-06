//! Whole-device primitives. These describe tensor work, never tile identities.

use super::*;

/// A rectangular operand window in global tensor coordinates. Omitted axes
/// retain their full extent. Windows do not allocate temporary tensors.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OperandWindow(pub Vec<(u16, u32, u32)>);

/// Matrix axes used by a local product after the distributed operands have
/// been materialized. Blocking is in the selected GEMM kernel specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProductAxes {
    pub left_inner: TensorAxis,
    pub right_inner: TensorAxis,
    pub output_column: TensorAxis,
}

/// Map output coordinates back to the source: first add the window offsets,
/// then apply the optional factor-axis view. Layout/storage order is separate.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CoordinateMapping {
    pub offsets: Vec<u32>,
    pub view: Option<AxisFactorView>,
}

impl From<AxisFactorView> for CoordinateMapping {
    fn from(view: AxisFactorView) -> Self {
        Self {
            offsets: Vec::new(),
            view: Some(view),
        }
    }
}

impl CoordinateMapping {
    /// Compose output -> intermediate -> source without materializing the
    /// intermediate. Return None when the result needs more than one factor
    /// view, or when an intermediate supplies logical zero padding.
    pub(crate) fn compose(
        &self,
        next: &Self,
        source: &TensorShape,
        intermediate: &TensorShape,
        output: &TensorShape,
    ) -> Option<Self> {
        fn fits(
            mapping: &CoordinateMapping,
            source: &TensorShape,
            output: &TensorShape,
        ) -> Option<()> {
            let shape = mapping
                .view
                .map_or_else(|| Some(source.clone()), |v| v.output_shape(source))?;
            if shape.0.len() != output.0.len() || mapping.offsets.len() > shape.0.len() {
                return None;
            }
            output
                .0
                .iter()
                .zip(&shape.0)
                .enumerate()
                .all(|(axis, (&size, &bound))| {
                    mapping
                        .offsets
                        .get(axis)
                        .copied()
                        .unwrap_or(0)
                        .checked_add(size)
                        .is_some_and(|end| end <= bound)
                })
                .then_some(())
        }
        fits(self, source, intermediate)?;
        let next_shape = next.view.map_or_else(
            || Some(intermediate.clone()),
            |v| v.output_shape(intermediate),
        )?;
        if output.0.len() != next_shape.0.len()
            || next.offsets.len() > output.0.len()
            || output.0.iter().enumerate().any(|(axis, &size)| {
                next.offsets
                    .get(axis)
                    .copied()
                    .unwrap_or(0)
                    .checked_add(size)
                    .is_none()
            })
        {
            return None;
        }
        if fits(next, intermediate, output).is_none() {
            // The last copy may add padding if the removed copy did not crop
            // the source. Otherwise bypassing it could expose cropped values.
            let full = self
                .view
                .map_or_else(|| Some(source.clone()), |v| v.output_shape(source))?;
            if intermediate != &full
                || self.offsets.iter().any(|&offset| offset != 0)
                || output.0.len() != intermediate.0.len()
                || next.offsets.len() > output.0.len()
            {
                return None;
            }
        }
        let mut offsets = vec![0u32; output.0.len()];
        for (axis, offset) in offsets.iter_mut().enumerate() {
            *offset = self.offsets.get(axis).copied().unwrap_or(0);
        }
        let view = match (self.view, next.view) {
            (view, None) => view,
            (None, Some(view)) => {
                // A window before a view must retain the view's factor width.
                if source.0[view.split_axis] != intermediate.0[view.split_axis] {
                    return None;
                }
                offsets[view.merge_axis] = offsets[view.merge_axis].checked_mul(view.factor)?;
                Some(view)
            }
            (Some(first), Some(second)) => {
                if first.split_axis != second.split_axis
                    || first.merge_axis != second.merge_axis
                    || offsets[first.split_axis] != 0
                    || offsets[first.merge_axis] != 0
                    || first.output_shape(source)?.0[first.split_axis]
                        != intermediate.0[first.split_axis]
                {
                    return None;
                }
                Some(AxisFactorView::new(
                    first.split_axis,
                    first.merge_axis,
                    first.factor.checked_mul(second.factor)?,
                ))
            }
        };
        for (axis, offset) in offsets.iter_mut().enumerate() {
            *offset = offset.checked_add(next.offsets.get(axis).copied().unwrap_or(0))?;
        }
        while offsets.last() == Some(&0) {
            offsets.pop();
        }
        let result = Self { offsets, view };
        Some(result)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Primitive {
    /// Populate a distributed output using a logical coordinate mapping.
    /// Ownership, storage-order changes and padding follow the tensor layouts.
    Copy {
        mapping: CoordinateMapping,
        /// Reuse compatible resident storage when lowering can prove it safe.
        reuse_local: bool,
    },
    /// Invoke the selected kernel over the output distribution. GEMM blocking
    /// enumerates local calls later; it does not choose distribution or staging.
    Compute {
        kernel: TileKernelSpec,
        operands: Vec<OperandWindow>,
        product: Option<ProductAxes>,
        /// Input whose allocation holds the new output version. It may be an
        /// additional dependency beyond the kernel's explicit operands.
        reuse_input: Option<usize>,
    },
    /// Independent partials occupy an explicit tensor axis. The selected
    /// reduction policy determines whether remote contributors arrive together
    /// or in successive bounded stages.
    Sum {
        axis: u16,
        staging: ReductionStaging,
    },
}
