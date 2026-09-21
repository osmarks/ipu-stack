//! Layout-specific rearrangement workers and assembly fast paths.

use super::*;
use crate::mid::MidOperationKind;
use ipu_target::Target;

// These indices are the device workers' ABI, not another layout representation.
fn order_index(order: ElementOrder, unpack: bool) -> Option<u32> {
    match (unpack, order) {
        (true, ElementOrder::Amp(AmpOrder::Output))
        | (false, ElementOrder::Amp(AmpOrder::Left)) => Some(0),
        (true, ElementOrder::Amp(AmpOrder::TransposedLeft))
        | (false, ElementOrder::Amp(AmpOrder::TransposedRight)) => Some(1),
        (_, ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. })) => Some(2),
        (true, ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix { .. })) => Some(3),
        _ => None,
    }
}

pub(crate) fn supported(from: ElementOrder, to: ElementOrder, precision: Precision) -> bool {
    precision == Precision::F16
        && ((order_index(from, true).is_some() && to == ElementOrder::RowMajor)
            || (from == ElementOrder::RowMajor && order_index(to, false).is_some()))
}

pub(super) fn call(
    kernel: &MidOperationKind,
    inputs: &[TensorStorage<'_>],
    outputs: &[TensorStorage<'_>],
    build: Option<&mut KernelObjects>,
) -> Result<KernelCall, KernelError> {
    check_arity(inputs, outputs, 1, 1)?;
    let MidOperationKind::Rearrange { from, to } = kernel else {
        return Err(KernelError::RequirementMismatch);
    };
    if !supported(from.order, to.order, outputs[0].format.precision) {
        return Err(KernelError::Unavailable(kernel.clone()));
    }
    let rows = outputs[0].matrix_extent(true, false)?;
    let physical_rows = outputs[0].matrix_extent(false, false)?;
    let columns = outputs[0].matrix_extent(true, true)?;
    let physical_columns = outputs[0].matrix_extent(false, true)?;
    let matrices = inputs[0]
        .widths()
        .take(inputs[0].extents.len().saturating_sub(2))
        .try_fold(1u32, |count, width| count.checked_mul(width))
        .ok_or(KernelError::ElementCountOverflow)?;
    let unpack = from.order != ElementOrder::RowMajor;
    let geometry = if unpack { inputs[0] } else { outputs[0] };
    let shape = (
        if unpack { from.order } else { to.order },
        geometry.matrix_extent(true, false)?,
        geometry.matrix_extent(false, false)?,
        geometry.matrix_extent(true, true)?,
        geometry.matrix_extent(false, true)?,
    );
    let arguments = if from.order == ElementOrder::RowMajor {
        vec![
            rows,
            physical_rows,
            order_index(to.order, false).ok_or(KernelError::RequirementMismatch)?,
            columns,
            physical_columns,
            matrices,
        ]
    } else {
        vec![matrices, rows, physical_rows, columns, physical_columns]
    };
    let (order, rows, physical_rows, columns, physical_columns) = if unpack {
        shape
    } else {
        rearrangement_specialization(shape.0, shape.1, shape.2, shape.3, shape.4)
    };
    let index = order_index(order, unpack).expect("supported rearrangement order");
    let (row_block, column_block) = match order {
        ElementOrder::BlockMajor(
            BlockMajorOrder::Matrix {
                row_block,
                column_block,
            }
            | BlockMajorOrder::TransposedMatrix {
                row_block,
                column_block,
            },
        ) => (row_block, column_block),
        _ => (AMP_INNER_BLOCK as u16, AMP_COLUMN_MICRO as u16),
    };
    let mut suffix = format!("o{index}_r{rows}_p{physical_rows}_c{columns}_p{physical_columns}");
    if matches!(order, ElementOrder::BlockMajor(_)) {
        let separator = if unpack { "_" } else { "x" };
        suffix.push_str(&format!("_b{row_block}{separator}{column_block}"));
    }
    let (prefix, vertex, call, source) = if unpack {
        (
            "UNPACK",
            "UnpackAmpToRowMajorF16",
            "unpack_amp_to_row_major_f16",
            "unpack_amp_f16.cpp",
        )
    } else {
        (
            "REARRANGE",
            "RearrangeRowMajorToAmpF16",
            "rearrange_row_major_to_amp_f16",
            "rearrange_f16.cpp",
        )
    };
    let call = format!("{call}_{suffix}");
    let selected = assembly(
        order,
        unpack,
        rows,
        physical_rows,
        columns,
        physical_columns,
        matrices.into(),
    );
    if let Some(build) = build {
        let vertex = format!("{vertex}_{suffix}");
        let mut flags = vec![
            format!("-D{prefix}_LOGICAL_ROWS={rows}"),
            format!("-D{prefix}_PHYSICAL_ROWS={physical_rows}"),
            format!("-D{prefix}_LOGICAL_COLUMNS={columns}"),
            format!("-D{prefix}_PHYSICAL_COLUMNS={physical_columns}"),
        ];
        if let Some((source, _)) = selected {
            flags.push(format!("-D{prefix}_CALL_SYMBOL={call}"));
            let name = if unpack {
                "unpack_transposed_amp_f16"
            } else {
                "rearrange_f16_codelet"
            };
            build.add_compilation(KernelCompilation {
                source,
                name: format!("{name}_{suffix}"),
                flags,
            });
        } else {
            let direction = if unpack { "SOURCE" } else { "TARGET" };
            flags.extend([
                "-O2".into(),
                format!("-D{prefix}_{direction}_ORDER={index}"),
            ]);
            if !unpack {
                flags.push(format!("-DREARRANGE_INNER_DIMENSION={AMP_COLUMN_MICRO}"));
            }
            flags.extend([
                format!("-D{prefix}_ROW_BLOCK={row_block}"),
                format!("-D{prefix}_COLUMN_BLOCK={column_block}"),
                format!("-D{prefix}_VERTEX_NAME={vertex}"),
            ]);
            build.add_vertex(
                source,
                &call,
                &vertex,
                flags,
                &[3, 2, 4, 5, 6, 7, 8, 9],
                "worker_call.S",
                Vec::new(),
            );
        }
    }
    let matrices = u64::from(matrices);
    let elements = matrices
        .saturating_mul(u64::from(physical_rows))
        .saturating_mul(u64::from(physical_columns));
    if physical_rows == 0 || physical_columns == 0 {
        return Ok(KernelCall::new(call, arguments, 0));
    }
    let cycles = selected.map_or_else(
        || {
            let per_element = if order == ElementOrder::Amp(AmpOrder::TransposedRight) && !unpack {
                3
            } else {
                10
            };
            elements
                .saturating_mul(per_element)
                .saturating_add(ipu_target::ipu21::costs::COSTS.kernel_launch_cycles)
        },
        |(_, cycles)| cycles,
    );
    Ok(KernelCall::new(call, arguments, cycles))
}

pub(crate) fn supports_row_major_population(order: ElementOrder) -> bool {
    order == ElementOrder::RowMajor || order_index(order, false).is_some()
}

pub(crate) fn estimate(
    target: Target,
    from: ElementOrder,
    input: TensorStorage<'_>,
    output: TensorStorage<'_>,
) -> u64 {
    if from == output.format.layout.order {
        return 0;
    }
    let mut source = input.format.layout.clone();
    source.order = from;
    let kernel = MidOperationKind::Rearrange {
        from: source,
        to: output.format.layout.clone(),
    };
    KernelCall::select(target, &kernel, &[input], &[output], None)
        .map_or(u64::MAX, |call| call.cycles)
}

/// Assembly selection is shared by build construction and geometry costing.
fn assembly(
    order: ElementOrder,
    unpack: bool,
    rows: u32,
    physical_rows: u32,
    columns: u32,
    physical_columns: u32,
    matrices: u64,
) -> Option<(&'static str, u64)> {
    let elements = matrices
        .saturating_mul(physical_rows.into())
        .saturating_mul(physical_columns.into());
    let launch = ipu_target::ipu21::costs::COSTS.kernel_launch_cycles;
    match (unpack, order) {
        (true, ElementOrder::Amp(AmpOrder::TransposedLeft)) => Some((
            "unpack_transposed_amp_f16.S",
            f16_transposed_unpack_cycles(matrices, physical_rows.into(), physical_columns.into()),
        )),
        (false, ElementOrder::Amp(AmpOrder::Left))
            if columns.is_multiple_of(2) && physical_columns.is_multiple_of(AMP_COLUMN_MICRO) =>
        {
            Some(("rearrange_amp_left_f16.S", elements.saturating_add(launch)))
        }
        (
            false,
            ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                row_block,
                column_block,
            }),
        ) if u32::from(row_block) == physical_rows
            && physical_rows.is_multiple_of(AMP_COLUMN_MICRO)
            && u32::from(column_block) == AMP_COLUMN_MICRO
            && columns.is_multiple_of(4)
            && physical_columns.is_multiple_of(AMP_COLUMN_MICRO) =>
        {
            Some((
                "rearrange_block_major_f16.S",
                f16_coefficient_pack_cycles(matrices, physical_rows.into(), columns.into()),
            ))
        }
        (false, ElementOrder::Amp(AmpOrder::TransposedRight))
            if (rows, physical_rows, columns, physical_columns) == (64, 64, 16, 16) =>
        {
            Some((
                "rearrange_transposed_right_f16.S",
                elements.saturating_mul(3).saturating_add(launch),
            ))
        }
        _ => None,
    }
}

pub(super) fn rearrangement_specialization(
    order: ElementOrder,
    mut logical_rows: u32,
    physical_rows: u32,
    logical_columns: u32,
    physical_columns: u32,
) -> (ElementOrder, u32, u32, u32, u32) {
    if physical_rows == AMP_INNER_BLOCK
        && logical_rows < physical_rows
        && matches!(
            order,
            ElementOrder::Amp(AmpOrder::TransposedRight)
                | ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. })
        )
    {
        // Row tails share one worker, but column alignment selects wide loads.
        // Erasing it could incorrectly admit a two-halfword tail to ld64.
        logical_rows = 0;
    }
    (
        order,
        logical_rows,
        physical_rows,
        logical_columns,
        physical_columns,
    )
}

/// Word-pair transpose in unpack_transposed_amp_f16.S: six workers distribute
/// row pairs, with about fifteen issue slots per two columns. Physical geometry
/// does not retain logical tail masks, whose branches add some extra work.
pub(crate) fn f16_transposed_unpack_cycles(matrices: u64, rows: u64, columns: u64) -> u64 {
    300u64.saturating_add(
        matrices
            .saturating_mul(rows.div_ceil(12))
            .saturating_mul(100u64.saturating_add(columns.div_ceil(2).saturating_mul(90))),
    )
}

/// Paired-row coefficient packing. Complete 16-column panels use an unrolled
/// transpose; other aligned widths retain column indexing and bounds checks.
/// Six worker contexts distribute pairs, including the physical zero-padding.
pub(crate) fn f16_coefficient_pack_cycles(matrices: u64, rows: u64, columns: u64) -> u64 {
    let pair = if columns == 16 {
        60
    } else {
        12 + columns.div_ceil(16) * 4 * 35
    };
    300u64.saturating_add(
        matrices
            .saturating_mul(rows.div_ceil(12))
            .saturating_mul(6)
            .saturating_mul(pair),
    )
}
