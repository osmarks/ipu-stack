//! Cast format support, packed call geometry and shifted FP16-to-F143 access.
//! The shifted output prefix and chunks separate input/output memory elements;
//! Whether the extra prefix saves storage belongs to the low donation pass.

use super::*;
use crate::mid::MidOperationKind;
use ipu_target::ipu21::costs::COSTS;

pub(super) fn access(
    from: Precision,
    to: Precision,
    output: &BlockValue,
) -> (crate::low::storage::StorageAccess, &'static [MemoryOperand]) {
    let alignment = if from == Precision::F16
        && matches!(to, Precision::F8F143 { .. })
        && matches!(
            output.definition,
            crate::ShardDefinition::ShiftedAlias { .. }
        ) {
        CAST_PREFIX_BYTES
    } else {
        8
    };
    (
        crate::low::storage::StorageAccess {
            alignment,
            access_tail_bytes: 0,
        },
        &[],
    )
}

pub(super) fn f16_fp8_cycles(elements: u64, panel_rows: u64) -> u64 {
    if panel_rows == 0 {
        return 330 + elements.div_ceil(48).saturating_mul(24);
    }
    // Bank separation is unknown before placement; price the fallback.
    let panels = elements.div_ceil(panel_rows * 32);
    if panel_rows <= 32 && panels >= 6 {
        330 + panels.div_ceil(6).saturating_mul(120 + panel_rows * 72)
    } else {
        330 + panels.saturating_mul(120 + panel_rows.div_ceil(6) * 72)
    }
}

pub(super) fn stream_cycles(elements: u64, input_bytes: u64, output_bytes: u64) -> u64 {
    if (input_bytes, output_bytes) == (4, 2) {
        return 330 + elements.div_ceil(12).saturating_mul(48);
    }
    COSTS.kernel_launch_cycles
        + elements
            .saturating_mul(input_bytes + output_bytes)
            .div_ceil(COSTS.local_copy_bytes_per_cycle)
}

/// Packed FP16-to-FP8 calls carry the readable prefix and physical row bounds used by the
/// assembly loops, without a separate scalar getter recipe.
pub(super) fn call(
    kernel: &MidOperationKind,
    inputs: &[TensorStorage<'_>],
    outputs: &[TensorStorage<'_>],
    build: Option<&mut KernelObjects>,
) -> Result<KernelCall, KernelError> {
    check_arity(inputs, outputs, 1, 1)?;
    let MidOperationKind::Cast { from, to } = *kernel else {
        return Err(KernelError::RequirementMismatch);
    };
    let symbol = symbol(from, to).ok_or_else(|| KernelError::Unavailable(kernel.clone()))?;
    let count = outputs[0].count()?;
    if !matches!(from, Precision::F8F143 { .. }) && !matches!(to, Precision::F8F143 { .. }) {
        if let Some(build) = build {
            build.add_vertex(
                "cast_f32_f16.cpp",
                symbol,
                "CastF32ToF16",
                vec!["-O2".into()],
                &[3, 2, 4],
                "worker_call.S",
                Vec::new(),
            );
        }
        return Ok(KernelCall::new(
            symbol,
            vec![count],
            stream_cycles(count.into(), from.bytes().into(), to.bytes().into()),
        ));
    }
    let order = outputs[0].format.layout.order;
    let row_pack = inputs[0].format.layout.order == ElementOrder::RowMajor
        && order == ElementOrder::Amp(AmpOrder::Left);
    let f16_to_f8 = from == Precision::F16 && matches!(to, Precision::F8F143 { .. });
    let panel_rows = if f16_to_f8 {
        panel_rows(inputs[0], outputs[0])?
    } else {
        0
    };

    let scale = |precision| {
        fp8_scale_argument(match precision {
            Precision::F8F143 { scale_exponent } => i32::from(scale_exponent),
            _ => 0,
        })
    };
    let (source_metadata, readable_rows) = if from == Precision::F16 && row_pack {
        let physical = inputs[0].matrix_extent(false, false)?;
        let logical = inputs[0].matrix_extent(true, false)?;
        // The FP16 source-scale word carries two row bounds. Zero retains
        // initialization when the compact descriptor cannot represent them.
        if logical < physical && physical <= u16::MAX.into() {
            ((physical << 16) | logical, logical)
        } else {
            (0, physical)
        }
    } else {
        (scale(from)?, 0)
    };
    let source_extent = if row_pack {
        let logical = inputs[0].matrix_extent(true, true)?;
        if logical.is_multiple_of(4) {
            logical
        } else {
            inputs[0].matrix_extent(false, true)?
        }
    } else {
        inputs[0].count()?
    };
    let source_columns = if row_pack {
        inputs[0].matrix_extent(false, true)?
    } else {
        0
    };
    if let Some(build) = build {
        let (from, to) = (from.bytes(), to.bytes());
        let vertex = format!("Cast{from}To{to}");
        build.add_vertex(
            "cast_f8.cpp",
            symbol,
            &vertex,
            vec![
                "-O2".into(),
                format!("-DINPUT_BYTES={from}"),
                format!("-DOUTPUT_BYTES={to}"),
                format!("-DCAST_VERTEX={vertex}"),
            ],
            &[3, 2, 4, 5, 6, 7, 8, 9],
            if (from, to) == (2, 1) {
                "cast_f8_call.S"
            } else {
                "worker_call.S"
            },
            Vec::new(),
        );
    }
    let cycles = if f16_to_f8 {
        f16_fp8_cycles(count.into(), panel_rows.into())
    } else {
        stream_cycles(count.into(), from.bytes().into(), to.bytes().into())
    };
    let mut call = KernelCall::new(
        symbol,
        vec![
            count,
            source_metadata,
            scale(to)?,
            panel_rows,
            source_extent,
            source_columns,
        ],
        cycles,
    );
    if f16_to_f8 && panel_rows != 0 && source_columns != 0 {
        // These are the actual selected read bounds. Descriptor overflow uses
        // physical rows in both the device ABI and the padding proof.
        call.padding = input_padding(inputs[0], readable_rows, source_extent)?;
    }
    Ok(call)
}

/// Select and validate the FP16-to-FP8 traversal once for both calls and costs.
fn panel_rows(input: TensorStorage<'_>, output: TensorStorage<'_>) -> Result<u32, KernelError> {
    let order = output.format.layout.order;
    let row_pack = input.format.layout.order == ElementOrder::RowMajor
        && order == ElementOrder::Amp(AmpOrder::Left);
    if (input.format.layout.order != order && !row_pack)
        || input.extents.len() != output.extents.len()
        || (0..input.extents.len()).any(|axis| {
            let width = input.dimension(axis);
            let target = output.dimension(axis);
            width != target
                && !(order == ElementOrder::Amp(AmpOrder::Left)
                    && axis + 1 == input.extents.len()
                    && if row_pack {
                        width.is_multiple_of(4) && target >= width
                    } else {
                        width.is_multiple_of(16) && target == width.next_multiple_of(32)
                    })
        })
    {
        return Err(KernelError::RequirementMismatch);
    }
    if order.fp8_cast_panel_rows(1, 1) == 0 {
        return Ok(0);
    }
    if input.extents.len() < 2 {
        return Err(KernelError::RequirementMismatch);
    }
    let columns = u64::from(output.matrix_extent(false, true)?);
    if columns == 0
        || (row_pack
            && (!columns.is_multiple_of(32)
                || !input.dimension(input.extents.len() - 1).is_multiple_of(4)))
    {
        return Err(KernelError::RequirementMismatch);
    }
    let count = output.elements();
    let rows = count / columns;
    if order == ElementOrder::Amp(AmpOrder::Left)
        && rows == 1
        && input.elements() == count
        && input.extents[input.extents.len() - 2..]
            .iter()
            .all(|extent| extent.logical_end == extent.physical_end)
    {
        return Ok(0);
    }
    let panels = order.fp8_cast_panel_rows(rows, columns);
    if panels != 0
        && !count.is_multiple_of(
            panels
                .checked_mul(32)
                .ok_or(KernelError::ElementCountOverflow)?,
        )
    {
        return Err(KernelError::RequirementMismatch);
    }
    u32::try_from(panels).map_err(|_| KernelError::ElementCountOverflow)
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

fn input_padding(
    input: TensorStorage<'_>,
    rows: u32,
    columns: u32,
) -> Result<PaddingRequirement, KernelError> {
    let rank = input.extents.len();
    let mut regions = Vec::new();
    for (axis, count) in [(rank - 2, rows), (rank - 1, columns)] {
        let mut region = input.extents.to_vec();
        // Outer padding is not covered by the matrix descriptor.
        for extent in &mut region[..rank - 2] {
            extent.physical_end = extent.logical_end;
        }
        let extent = &mut region[axis];
        extent.start = extent
            .start
            .checked_add(count)
            .ok_or(KernelError::ElementCountOverflow)?;
        extent.logical_end = extent.logical_end.max(extent.start);
        if extent.start < extent.physical_end {
            regions.push(region);
        }
    }
    Ok(PaddingRequirement::Unread(regions))
}
