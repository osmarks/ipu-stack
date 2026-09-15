//! Cast format support, packed call geometry and shifted FP16-to-F143 access.
//! The shifted output prefix and chunks separate input/output memory elements;
//! Whether the extra prefix saves storage belongs to the low donation pass.

use super::*;
use crate::estimate::IPU21_TARGET_COSTS;
use crate::mid::MidOperationKind;

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
    IPU21_TARGET_COSTS.kernel_launch_cycles
        + elements
            .saturating_mul(input_bytes + output_bytes)
            .div_ceil(IPU21_TARGET_COSTS.local_copy_bytes_per_cycle)
}

/// Packed FP16-to-FP8 calls carry the readable prefix and physical row bounds used by the
/// assembly loops, without a separate scalar getter recipe.
pub(super) fn call(
    kernel: &MidOperationKind,
    inputs: &[TensorStorage<'_>],
    outputs: &[TensorStorage<'_>],
) -> Result<KernelCall, KernelAbiError> {
    check_arity(inputs, outputs, 1, 1)?;
    let MidOperationKind::Cast { from, to } = *kernel else {
        return Err(KernelAbiError::RequirementMismatch);
    };
    let symbol = symbol(from, to).ok_or_else(|| KernelAbiError::Unavailable(kernel.clone()))?;
    let count = outputs[0].count()?;
    if !matches!(from, Precision::F8F143 { .. }) && !matches!(to, Precision::F8F143 { .. }) {
        return Ok(KernelCall::exact(symbol, vec![count]));
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
    let source_metadata = if from == Precision::F16 && row_pack {
        let physical = inputs[0].matrix_extent(false, false)?;
        let logical = inputs[0].matrix_extent(true, false)?;
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

/// Select and validate the FP16-to-FP8 traversal once for both calls and costs.
fn panel_rows(input: TensorStorage<'_>, output: TensorStorage<'_>) -> Result<u32, KernelAbiError> {
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
        return Err(KernelAbiError::RequirementMismatch);
    }
    if order.fp8_cast_panel_rows(1, 1) == 0 {
        return Ok(0);
    }
    if input.extents.len() < 2 {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let columns = u64::from(output.trailing_dimension(0).unwrap());
    if columns == 0
        || (row_pack
            && (!columns.is_multiple_of(32)
                || !input.dimension(input.extents.len() - 1).is_multiple_of(4)))
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let count = output.elements();
    let rows = count / columns;
    if order == ElementOrder::Amp(AmpOrder::Left)
        && rows == 1
        && input.elements() == count
        && (input.extents.len() - 2..input.extents.len())
            .all(|axis| input.logical_dimension(axis) == input.dimension(axis))
    {
        return Ok(0);
    }
    let panels = order.fp8_cast_panel_rows(rows, columns);
    if panels != 0
        && !count.is_multiple_of(
            panels
                .checked_mul(32)
                .ok_or(KernelAbiError::ElementCountOverflow)?,
        )
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    u32::try_from(panels).map_err(|_| KernelAbiError::ElementCountOverflow)
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

pub(super) fn input_padding(
    call: &KernelCall,
    run: &KernelRun,
) -> Result<PaddingRequirement, KernelAbiError> {
    let input = &run.inputs[0];
    let [_, row_bounds, _, panel_rows, columns, stride] = call.arguments.as_slice() else {
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
        run.geometry(MemoryOperand::Input(0))
            .matrix_extent(false, false)?
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

impl KernelBuildPlan {
    pub(super) fn add_casts(&mut self, exact_symbols: &BTreeSet<&'static str>) {
        if exact_symbols.contains("cast_f32_f16") {
            self.add_vertex(
                "cast_f32_f16.cpp",
                "cast_f32_f16",
                "CastF32ToF16",
                vec!["-O2".into()],
                &[3, 2, 4],
                "worker_call.S",
                Vec::new(),
            );
        }
        // Scales are call arguments, so all FP8 scales share these recipes.
        let f8 = Precision::F8F143 { scale_exponent: 0 };
        for (from, to) in [
            (f8, f8),
            (f8, Precision::F16),
            (f8, Precision::F32),
            (Precision::F16, f8),
            (Precision::F32, f8),
        ] {
            let symbol = cast::symbol(from, to).expect("FP8 cast implementation");
            if !exact_symbols.contains(symbol) {
                continue;
            }
            let (from, to) = (from.bytes(), to.bytes());
            let vertex = format!("Cast{from}To{to}");
            self.add_vertex(
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
    }
}
