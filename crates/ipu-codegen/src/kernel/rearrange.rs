//! Layout-specific rearrangement workers and assembly fast paths.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum RearrangeTarget {
    AmpLeft,
    AmpTransposedRight,
    BlockMajor { row_block: u16, column_block: u16 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum UnpackSource {
    AmpOutput,
    AmpTransposedLeft,
    Blocked(BlockMajorOrder),
}

impl UnpackSource {
    pub(super) fn from_order(order: ElementOrder) -> Option<Self> {
        match order {
            ElementOrder::Amp(AmpOrder::Output) => Some(Self::AmpOutput),
            ElementOrder::Amp(AmpOrder::TransposedLeft) => Some(Self::AmpTransposedLeft),
            ElementOrder::BlockMajor(order) => Some(Self::Blocked(order)),
            _ => None,
        }
    }

    pub(super) const fn codelet_index(self) -> u32 {
        match self {
            Self::AmpOutput => 0,
            Self::AmpTransposedLeft => 1,
            Self::Blocked(BlockMajorOrder::Matrix { .. }) => 2,
            Self::Blocked(BlockMajorOrder::TransposedMatrix { .. }) => 3,
        }
    }
}

impl RearrangeTarget {
    pub(super) fn from_order(order: ElementOrder) -> Option<Self> {
        match order {
            ElementOrder::Amp(AmpOrder::Left) => Some(Self::AmpLeft),
            ElementOrder::Amp(AmpOrder::TransposedRight) => Some(Self::AmpTransposedRight),
            ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                row_block,
                column_block,
            }) => Some(Self::BlockMajor {
                row_block,
                column_block,
            }),
            _ => None,
        }
    }

    pub(super) const fn codelet_index(self) -> u32 {
        match self {
            Self::AmpLeft => 0,
            Self::AmpTransposedRight => 1,
            Self::BlockMajor { .. } => 2,
        }
    }
}

pub(crate) fn supported(from: ElementOrder, to: ElementOrder, precision: Precision) -> bool {
    precision == Precision::F16
        && ((UnpackSource::from_order(from).is_some() && to == ElementOrder::RowMajor)
            || (from == ElementOrder::RowMajor && RearrangeTarget::from_order(to).is_some()))
}

pub(super) fn call(run: &KernelRun) -> Result<KernelCall, KernelAbiError> {
    run.check_arity(1, 1)?;
    let TileKernelSpec::Rearrange { from, to } = &run.kernel else {
        return Err(KernelAbiError::RequirementMismatch);
    };
    if !supported(
        from.order,
        to.order,
        run.requirements.outputs[0].format.precision,
    ) {
        return Err(KernelAbiError::Unavailable(run.kernel.clone()));
    }
    let rows = matrix_extent(&run.outputs[0], true, false)?;
    let physical_rows = matrix_extent(&run.outputs[0], false, false)?;
    let columns = matrix_extent(&run.outputs[0], true, true)?;
    let physical_columns = matrix_extent(&run.outputs[0], false, true)?;
    let matrices = matrix_count(run)?;
    let (implementation, arguments) = if from.order == ElementOrder::RowMajor {
        let target =
            RearrangeTarget::from_order(to.order).ok_or(KernelAbiError::RequirementMismatch)?;
        (
            KernelImplementation::Rearrange(rearrangement_specialization(
                target,
                rows,
                physical_rows,
                columns,
                physical_columns,
            )),
            vec![
                rows,
                physical_rows,
                target.codelet_index(),
                columns,
                physical_columns,
                matrices,
            ],
        )
    } else {
        (
            KernelImplementation::Unpack((
                UnpackSource::from_order(from.order).ok_or(KernelAbiError::RequirementMismatch)?,
                input_matrix_extent(run, true, false)?,
                input_matrix_extent(run, false, false)?,
                input_matrix_extent(run, true, true)?,
                input_matrix_extent(run, false, true)?,
            )),
            vec![matrices, rows, physical_rows, columns, physical_columns],
        )
    };
    Ok(KernelCall {
        implementation,
        arguments,
    })
}

pub(crate) fn supports_row_major_population(order: ElementOrder) -> bool {
    order == ElementOrder::RowMajor || RearrangeTarget::from_order(order).is_some()
}

impl KernelBuildPlan {
    pub(super) fn add_unpack(&mut self, shape: (UnpackSource, u32, u32, u32, u32)) {
        let (order, logical_rows, physical_rows, logical_columns, physical_columns) = shape;
        let order_index = order.codelet_index();
        let (row_block, column_block) = match order {
            UnpackSource::Blocked(
                BlockMajorOrder::Matrix {
                    row_block,
                    column_block,
                }
                | BlockMajorOrder::TransposedMatrix {
                    row_block,
                    column_block,
                },
            ) => (row_block, column_block),
            _ => (16, 16),
        };
        let suffix = format!(
            "o{order_index}_r{logical_rows}_p{physical_rows}_c{logical_columns}_p{physical_columns}"
        );
        let suffix = if order_index >= 2 {
            format!("{suffix}_b{row_block}_{column_block}")
        } else {
            suffix
        };
        let vertex = format!("UnpackAmpToRowMajorF16_{suffix}");
        let call = format!("unpack_amp_to_row_major_f16_{suffix}");
        self.symbols
            .insert(KernelImplementation::Unpack(shape), call.clone());
        let mut flags = vec![
            format!("-DUNPACK_LOGICAL_ROWS={logical_rows}"),
            format!("-DUNPACK_PHYSICAL_ROWS={physical_rows}"),
            format!("-DUNPACK_LOGICAL_COLUMNS={logical_columns}"),
            format!("-DUNPACK_PHYSICAL_COLUMNS={physical_columns}"),
        ];
        if order_index == 1 {
            flags.push(format!("-DUNPACK_CALL_SYMBOL={call}"));
            self.compilations.push(KernelCompilation {
                source: "unpack_transposed_amp_f16.S",
                name: format!("unpack_transposed_amp_f16_{suffix}"),
                flags,
            });
        } else {
            flags.extend([
                "-O2".into(),
                format!("-DUNPACK_SOURCE_ORDER={order_index}"),
                format!("-DUNPACK_ROW_BLOCK={row_block}"),
                format!("-DUNPACK_COLUMN_BLOCK={column_block}"),
                format!("-DUNPACK_VERTEX_NAME={vertex}"),
            ]);
            self.add_vertex(
                "unpack_amp_f16.cpp",
                &call,
                &vertex,
                flags,
                &[3, 2, 4, 5, 6, 7, 8, 9],
            );
        }
    }

    pub(super) fn add_rearrangement(&mut self, shape: (RearrangeTarget, u32, u32, u32, u32)) {
        let (order, logical_rows, physical_rows, logical_columns, physical_columns) = shape;
        let order_index = order.codelet_index();
        let (row_block, column_block) = match order {
            RearrangeTarget::BlockMajor {
                row_block,
                column_block,
            } => (row_block, column_block),
            _ => (AMP_INNER_BLOCK as u16, AMP_COLUMN_MICRO as u16),
        };
        let suffix = format!(
            "o{order_index}_r{logical_rows}_p{physical_rows}_c{logical_columns}_p{physical_columns}"
        );
        let suffix = if matches!(order, RearrangeTarget::BlockMajor { .. }) {
            format!("{suffix}_b{row_block}x{column_block}")
        } else {
            suffix
        };
        let vertex = format!("RearrangeRowMajorToAmpF16_{suffix}");
        let call = format!("rearrange_row_major_to_amp_f16_{suffix}");
        self.symbols
            .insert(KernelImplementation::Rearrange(shape), call.clone());
        let assembly = if order == RearrangeTarget::AmpLeft
            && logical_columns.is_multiple_of(2)
            && physical_columns.is_multiple_of(AMP_COLUMN_MICRO)
        {
            Some("rearrange_amp_left_f16.S")
        } else if matches!(order, RearrangeTarget::BlockMajor { .. })
            && u32::from(row_block) == physical_rows
            && physical_rows.is_multiple_of(AMP_COLUMN_MICRO)
            && u32::from(column_block) == AMP_COLUMN_MICRO
            && logical_columns.is_multiple_of(4)
            && physical_columns.is_multiple_of(AMP_COLUMN_MICRO)
        {
            Some("rearrange_block_major_f16.S")
        } else if order == RearrangeTarget::AmpTransposedRight
            && logical_rows == 64
            && physical_rows == 64
            && logical_columns == 16
            && physical_columns == 16
        {
            Some("rearrange_transposed_right_f16.S")
        } else {
            None
        };
        let mut flags = vec![
            format!("-DREARRANGE_LOGICAL_ROWS={logical_rows}"),
            format!("-DREARRANGE_PHYSICAL_ROWS={physical_rows}"),
            format!("-DREARRANGE_LOGICAL_COLUMNS={logical_columns}"),
            format!("-DREARRANGE_PHYSICAL_COLUMNS={physical_columns}"),
        ];
        if let Some(source) = assembly {
            flags.push(format!("-DREARRANGE_CALL_SYMBOL={call}"));
            self.compilations.push(KernelCompilation {
                source,
                name: format!("rearrange_f16_codelet_{suffix}"),
                flags,
            });
        } else {
            flags.extend([
                "-O2".into(),
                format!("-DREARRANGE_TARGET_ORDER={order_index}"),
                format!("-DREARRANGE_INNER_DIMENSION={AMP_COLUMN_MICRO}"),
                format!("-DREARRANGE_ROW_BLOCK={row_block}"),
                format!("-DREARRANGE_COLUMN_BLOCK={column_block}"),
                format!("-DREARRANGE_VERTEX_NAME={vertex}"),
            ]);
            self.add_vertex(
                "rearrange_f16.cpp",
                &call,
                &vertex,
                flags,
                &[3, 2, 4, 5, 6, 7, 8, 9],
            );
        }
    }
}

pub(super) fn rearrangement_specialization(
    order: RearrangeTarget,
    mut logical_rows: u32,
    physical_rows: u32,
    logical_columns: u32,
    physical_columns: u32,
) -> (RearrangeTarget, u32, u32, u32, u32) {
    if physical_rows == AMP_INNER_BLOCK
        && logical_rows < physical_rows
        && matches!(
            order,
            RearrangeTarget::AmpTransposedRight | RearrangeTarget::BlockMajor { .. }
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
