//! Output epilogues implemented by kernel families. Mid rewrites use these
//! contracts to move computation and its broadcast operands together.
use super::*;

pub(crate) struct OutputCapability {
    /// Operand zero follows the output coordinates; the remaining operands
    /// are row parameters placed with the ordinary broadcast tiling rules.
    pub operands: usize,
    pub complete_rows: bool,
    pub input_order: ElementOrder,
    pub output_orders: &'static [ElementOrder],
    pub column_multiple: u32,
}

impl TileKernelSpec {
    pub(crate) fn output_capability(&self, precision: Precision) -> Option<OutputCapability> {
        if !matches!(precision, Precision::F8F143 { .. }) {
            return None;
        }
        let (operands, complete_rows) = match self {
            Self::Gelu => (1, false),
            Self::BiasGelu => (2, false),
            Self::LayerNorm => (3, true),
            _ => return None,
        };
        Some(OutputCapability {
            operands,
            complete_rows,
            input_order: ElementOrder::RowMajor,
            output_orders: &[ElementOrder::RowMajor, ElementOrder::Amp(AmpOrder::Left)],
            column_multiple: 4,
        })
    }
}

pub(super) fn fp8_arguments(run: &KernelRun) -> Result<Option<Vec<u32>>, KernelAbiError> {
    let Some(capability) = run
        .kernel
        .output_capability(run.requirements.outputs[0].format.precision)
    else {
        return Ok(None);
    };
    let input = &run.inputs[0];
    let width = input_matrix_extent(run, false, true)?;
    let columns = matrix_extent(&run.outputs[0], false, true)?;
    let packed =
        run.requirements.outputs[0].format.layout.order == ElementOrder::Amp(AmpOrder::Left);
    if width == 0
        || !width.is_multiple_of(capability.column_multiple)
        || run.inputs.len() != capability.operands
        || input_matrix_extent(run, true, true)? != width
        || !capability
            .output_orders
            .contains(&run.requirements.outputs[0].format.layout.order)
        || columns
            != if packed {
                width.next_multiple_of(32)
            } else {
                width
            }
        || input.extents.len() != run.outputs[0].extents.len()
        || input.extents[..input.extents.len() - 1]
            != run.outputs[0].extents[..run.outputs[0].extents.len() - 1]
        || run.requirements.inputs.iter().any(|r| {
            r.format.precision != Precision::F16 || r.format.layout.order != capability.input_order
        })
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let Precision::F8F143 { scale_exponent } = run.requirements.outputs[0].format.precision else {
        unreachable!();
    };
    Ok(Some(vec![
        element_count(&run.inputs[0].extents)? / width,
        width,
        fp8_scale_argument(i32::from(scale_exponent))?,
        u32::from(packed),
    ]))
}

/// Shared row contract of the bias and normalization codelets. Families add
/// their own operand correspondence and column-padding restrictions.
pub(super) fn f16_row_width(run: &KernelRun) -> Result<u32, KernelAbiError> {
    let width = matrix_extent(&run.outputs[0], true, true)?;
    if width == 0
        || !width.is_multiple_of(2)
        || run.requirements.outputs[0].format.precision != Precision::F16
        || run.requirements.outputs[0].format.layout.order != ElementOrder::RowMajor
        || run
            .requirements
            .inputs
            .iter()
            .any(|r| r.format.precision != Precision::F16)
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    Ok(width)
}
