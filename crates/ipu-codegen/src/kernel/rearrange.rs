//! Layout-specific rearrangement workers and assembly fast paths.

use super::*;

impl KernelBuildPlan {
    pub(super) fn add_unpack(
        &mut self,
        (order, logical_rows, physical_rows, logical_columns, physical_columns): (
            UnpackSource,
            u32,
            u32,
            u32,
            u32,
        ),
    ) {
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
        let call = format!("ipu_stack_unpack_amp_to_row_major_f16_{suffix}");
        self.compilations.push(KernelCompilation {
            source: "unpack_amp_f16.cpp",
            name: format!("unpack_amp_f16_codelet_{suffix}"),
            flags: vec![
                "-O2".into(),
                format!("-DUNPACK_SOURCE_ORDER={order_index}"),
                format!("-DUNPACK_ROW_BLOCK={row_block}"),
                format!("-DUNPACK_COLUMN_BLOCK={column_block}"),
                format!("-DUNPACK_LOGICAL_ROWS={logical_rows}"),
                format!("-DUNPACK_PHYSICAL_ROWS={physical_rows}"),
                format!("-DUNPACK_LOGICAL_COLUMNS={logical_columns}"),
                format!("-DUNPACK_PHYSICAL_COLUMNS={physical_columns}"),
                format!("-DUNPACK_VERTEX_NAME={vertex}"),
            ],
            retained_symbols: Vec::new(),
        });
        self.add_worker_wrapper(
            format!("unpack_amp_f16_wrapper_{suffix}"),
            &call,
            &vertex,
            &[3, 2, 4, 5, 6, 7, 8],
        );
        self.symbols.insert(
            KernelSpecialization::Unpack((
                order,
                logical_rows,
                physical_rows,
                logical_columns,
                physical_columns,
            )),
            call,
        );
    }

    pub(super) fn add_rearrangement(
        &mut self,
        (order, logical_rows, physical_rows, logical_columns, physical_columns): (
            RearrangeTarget,
            u32,
            u32,
            u32,
            u32,
        ),
    ) {
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
        let suffix = match order {
            RearrangeTarget::BlockMajor {
                row_block,
                column_block,
            } => format!("{suffix}_b{row_block}x{column_block}"),
            _ => suffix,
        };
        let vertex = format!("RearrangeRowMajorToAmpF16_{suffix}");
        let call = format!("ipu_stack_rearrange_row_major_to_amp_f16_{suffix}");
        self.symbols.insert(
            KernelSpecialization::Rearrange((
                order,
                logical_rows,
                physical_rows,
                logical_columns,
                physical_columns,
            )),
            call.clone(),
        );
        if order == RearrangeTarget::AmpLeft
            && logical_columns.is_multiple_of(2)
            && physical_columns.is_multiple_of(AMP_COLUMN_MICRO)
        {
            self.compilations.push(KernelCompilation {
                source: "rearrange_amp_left_f16.S",
                name: format!("rearrange_amp_left_f16_{suffix}"),
                flags: vec![
                    format!("-DREARRANGE_CALL_SYMBOL={call}"),
                    format!("-DREARRANGE_LOGICAL_ROWS={logical_rows}"),
                    format!("-DREARRANGE_PHYSICAL_ROWS={physical_rows}"),
                    format!("-DREARRANGE_LOGICAL_COLUMNS={logical_columns}"),
                    format!("-DREARRANGE_PHYSICAL_COLUMNS={physical_columns}"),
                ],
                retained_symbols: vec![call.clone()],
            });
            return;
        }
        if order
            == (RearrangeTarget::BlockMajor {
                row_block: AMP_INNER_BLOCK as u16,
                column_block: AMP_COLUMN_MICRO as u16,
            })
            && physical_rows == AMP_INNER_BLOCK
            && logical_columns.is_multiple_of(4)
            && physical_columns.is_multiple_of(AMP_COLUMN_MICRO)
        {
            self.compilations.push(KernelCompilation {
                source: "rearrange_block_major_f16.S",
                name: format!("rearrange_block_major_f16_{suffix}"),
                flags: vec![
                    format!("-DREARRANGE_CALL_SYMBOL={call}"),
                    format!("-DREARRANGE_PHYSICAL_COLUMNS={physical_columns}"),
                ],
                retained_symbols: vec![call.clone()],
            });
            return;
        }
        if order == RearrangeTarget::AmpTransposedRight
            && logical_rows == 64
            && physical_rows == 64
            && logical_columns == 16
            && physical_columns == 16
        {
            self.compilations.push(KernelCompilation {
                source: "rearrange_transposed_right_f16.S",
                name: format!("rearrange_transposed_right_f16_{suffix}"),
                flags: vec![format!("-DREARRANGE_CALL_SYMBOL={call}")],
                retained_symbols: vec![call.clone()],
            });
            return;
        }
        self.compilations.push(KernelCompilation {
            source: "rearrange_f16.cpp",
            name: format!("rearrange_f16_codelet_{suffix}"),
            flags: vec![
                "-O2".into(),
                format!("-DREARRANGE_TARGET_ORDER={order_index}"),
                format!("-DREARRANGE_LOGICAL_ROWS={logical_rows}"),
                format!("-DREARRANGE_PHYSICAL_ROWS={physical_rows}"),
                format!("-DREARRANGE_LOGICAL_COLUMNS={logical_columns}"),
                format!("-DREARRANGE_PHYSICAL_COLUMNS={physical_columns}"),
                format!("-DREARRANGE_INNER_DIMENSION={AMP_COLUMN_MICRO}"),
                format!("-DREARRANGE_ROW_BLOCK={row_block}"),
                format!("-DREARRANGE_COLUMN_BLOCK={column_block}"),
                format!("-DREARRANGE_VERTEX_NAME={vertex}"),
            ],
            retained_symbols: Vec::new(),
        });
        self.add_worker_wrapper(
            format!("rearrange_f16_wrapper_{suffix}"),
            &call,
            &vertex,
            &[3, 2, 4, 5, 6, 7, 8],
        );
    }
}
