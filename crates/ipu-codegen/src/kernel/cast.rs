//! Cast format support, packed call geometry and shifted FP16-to-F143 access.
//! The shifted output prefix and chunks separate input/output memory elements;
//! whether the extra prefix saves storage belongs to the mid donation rewrite.

use super::*;

/// Cast format support is shared with optimistic planning. Packed FP16-to-FP8
/// calls also carry the readable prefix and physical row bounds used by the
/// assembly loops, without a separate scalar getter recipe.
pub(super) fn call(run: &KernelRun) -> Result<KernelCall, KernelAbiError> {
    run.check_arity(1, 1)?;
    let TileKernelSpec::Cast { from, to } = run.kernel else {
        return Err(KernelAbiError::RequirementMismatch);
    };
    let symbol = symbol(from, to).ok_or_else(|| KernelAbiError::Unavailable(run.kernel.clone()))?;
    let count = element_count(&run.outputs[0].extents)?;
    if !matches!(from, Precision::F8F143 { .. }) && !matches!(to, Precision::F8F143 { .. }) {
        return Ok(KernelCall::exact(symbol, vec![count]));
    }
    let order = run.requirements.outputs[0].format.layout.order;
    let row_pack = run.requirements.inputs[0].format.layout.order == ElementOrder::RowMajor
        && order == ElementOrder::Amp(AmpOrder::Left);
    let f16_to_f8 = from == Precision::F16 && matches!(to, Precision::F8F143 { .. });
    let panel_rows = if f16_to_f8 { panel_rows(run)? } else { 0 };
    if f16_to_f8 {
        let input = &run.inputs[0];
        if (run.requirements.inputs[0].format.layout.order
            != run.requirements.outputs[0].format.layout.order
            && !row_pack)
            || input.extents.len() != run.outputs[0].extents.len()
            || input
                .extents
                .iter()
                .zip(&run.outputs[0].extents)
                .enumerate()
                .any(|(axis, (from, to))| {
                    let width = from.physical_end - from.start;
                    let target_width = to.physical_end - to.start;
                    width != target_width
                        && !(run.requirements.outputs[0].format.layout.order
                            == ElementOrder::Amp(AmpOrder::Left)
                            && axis + 1 == input.extents.len()
                            && if row_pack {
                                width.is_multiple_of(4) && target_width >= width
                            } else {
                                width.is_multiple_of(16)
                                    && target_width == width.next_multiple_of(32)
                            })
                })
            || (row_pack
                && (!matrix_extent(&run.outputs[0], false, true)?.is_multiple_of(32)
                    || !input_matrix_extent(run, false, true)?.is_multiple_of(4)))
            || (panel_rows != 0
                && !element_count(&run.outputs[0].extents)?.is_multiple_of(
                    panel_rows
                        .checked_mul(32)
                        .ok_or(KernelAbiError::ElementCountOverflow)?,
                ))
        {
            return Err(KernelAbiError::RequirementMismatch);
        }
    }

    let scale = |precision| {
        fp8_scale_argument(match precision {
            Precision::F8F143 { scale_exponent } => i32::from(scale_exponent),
            _ => 0,
        })
    };
    let source_metadata = if from == Precision::F16 && row_pack {
        let physical = input_matrix_extent(run, false, false)?;
        let logical = input_matrix_extent(run, true, false)?;
        // The FP16 source-scale word carries two row bounds. Zero retains
        // initialization when the compact descriptor cannot represent them.
        if logical < physical && physical <= u16::MAX.into() {
            (physical << 16) | logical
        } else {
            0
        }
    } else {
        scale(from)?
    };
    let source_extent = if row_pack {
        let logical = input_matrix_extent(run, true, true)?;
        if logical.is_multiple_of(4) {
            logical
        } else {
            input_matrix_extent(run, false, true)?
        }
    } else {
        element_count(&run.inputs[0].extents)?
    };
    let source_columns = if row_pack {
        input_matrix_extent(run, false, true)?
    } else {
        0
    };
    Ok(KernelCall::exact(
        symbol,
        vec![
            count,
            source_metadata,
            scale(to)?,
            panel_rows,
            source_extent,
            source_columns,
        ],
    ))
}

fn panel_rows(run: &KernelRun) -> Result<u32, KernelAbiError> {
    let order = run.requirements.outputs[0].format.layout.order;
    if order.fp8_cast_panel_rows(1, 1) == 0 {
        return Ok(0);
    }
    let columns = matrix_extent(&run.outputs[0], false, true)?;
    let count = element_count(&run.outputs[0].extents)?;
    let rows = count
        .checked_div(columns)
        .ok_or(KernelAbiError::RequirementMismatch)?;
    if order == ElementOrder::Amp(AmpOrder::Left)
        && rows == 1
        && element_count(&run.inputs[0].extents)? == count
        && input_matrix_extent(run, true, true)? == input_matrix_extent(run, false, true)?
        && input_matrix_extent(run, true, false)? == input_matrix_extent(run, false, false)?
    {
        return Ok(0);
    }
    u32::try_from(order.fp8_cast_panel_rows(u64::from(rows), u64::from(columns)))
        .map_err(|_| KernelAbiError::RequirementMismatch)
}

pub(crate) fn symbol(from: Precision, to: Precision) -> Option<&'static str> {
    Some(match (from, to) {
        (Precision::F32, Precision::F16) => "cast_f32_f16",
        (Precision::F8F143 { .. }, Precision::F16) => "cast_f8_f16",
        (Precision::F8F143 { .. }, Precision::F32) => "cast_f8_f32",
        (Precision::F16, Precision::F8F143 { .. }) => "cast_f16_f8",
        (Precision::F32, Precision::F8F143 { .. }) => "cast_f32_f8",
        (Precision::F8F143 { .. }, Precision::F8F143 { .. }) => "cast_f8_f8",
        _ => return None,
    })
}

/// Keep the output in earlier memory elements than each input chunk.
pub(crate) const CAST_PREFIX_BYTES: u32 = 32768;

pub(crate) struct CastChunks {
    pub axis: usize,
    pub ranges: Vec<(u32, u32)>,
}

impl CastChunks {
    pub fn new(order: ElementOrder, dimensions: &[u32]) -> Option<Self> {
        let (axis, grain, atom) = match (order, dimensions) {
            (ElementOrder::Amp(AmpOrder::Left), [rows, columns]) if columns.is_multiple_of(32) => {
                (1, 32, rows.checked_mul(64)?)
            }
            (ElementOrder::RowMajor, [elements]) if elements.is_multiple_of(8) => (0, 8, 16),
            (ElementOrder::RowMajor, [_, columns]) if columns.is_multiple_of(8) => {
                (0, 1, columns.checked_mul(2)?)
            }
            _ => return None,
        };
        if atom == 0 {
            return None;
        }
        let atoms = dimensions[axis] / grain;
        atoms.checked_mul(atom)?;
        if atoms == 0 || atom > CAST_PREFIX_BYTES {
            return None;
        }
        let mut ranges = Vec::new();
        let mut consumed = 0u32;
        while consumed < atoms {
            let input_start = CAST_PREFIX_BYTES.checked_add(consumed.checked_mul(atom)?)?;
            let output_start = consumed.checked_mul(atom)? / 2;
            // End output before the input's first 32 KiB group. Reads may
            // span several groups; none is touched by this chunk's stores.
            let output_limit = input_start / CAST_PREFIX_BYTES * CAST_PREFIX_BYTES;
            let count =
                ((output_limit - output_start).checked_mul(2)? / atom).min(atoms - consumed);
            if count == 0 {
                return None;
            }
            ranges.push((consumed * grain, (consumed + count) * grain));
            consumed += count;
        }
        Some(Self { axis, ranges })
    }
}
