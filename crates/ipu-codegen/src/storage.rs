//! Conversion from logical shard views to physical byte ranges.

mod traversal;
pub(crate) use traversal::{ByteTraversal, byte_traversal, panel_byte_traversal};

use crate::{
    AMP_COLUMN_MICRO, AmpOrder, BlockMajorOrder, ElementOrder, Precision, ShardExtent, TensorFormat,
};

/// Relative storage geometry, independent of low IR identities and placement.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TensorStorage<'a> {
    pub format: &'a TensorFormat,
    pub extents: &'a [ShardExtent],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteSpan {
    /// Byte offset relative to the beginning of the shard allocation.
    pub offset: u32,
    pub bytes: u32,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum StorageError {
    #[error("view does not refer to the supplied shard")]
    WrongShard,
    #[error("view extents are not a valid subset of the shard")]
    InvalidView,
    #[error("matrix storage requires a tensor of rank at least two")]
    MatrixRank,
    #[error("AMP {role:?} storage dimensions are incompatible with its block shape")]
    AmpBlock { role: AmpOrder },
    #[error("block-major {order:?} storage dimensions are incompatible with its block shape")]
    BlockMajorBlock { order: BlockMajorOrder },
    #[error("tensor storage size overflowed")]
    Overflow,
}

pub type StorageResult<T> = Result<T, StorageError>;

#[cfg(test)]
mod sort_bench;

/// Allocation-relative view identity. Storage traversal depends on precision,
/// element order and these extents, not tile identity, memory class or tensor shape.
#[derive(PartialEq, Eq, Hash)]
pub(crate) struct ViewGeometry {
    precision: Precision,
    order: ElementOrder,
    allocation: Vec<ShardExtent>,
    view: Vec<ShardExtent>,
}
impl ViewGeometry {
    pub(crate) fn new(shard: TensorStorage<'_>, view: &[ShardExtent]) -> StorageResult<Self> {
        if shard.extents.len() != view.len() {
            return Err(StorageError::InvalidView);
        }
        let mut allocation = shard.extents.to_vec();
        let mut view = view.to_vec();
        for (a, v) in allocation.iter_mut().zip(&mut view) {
            let start = a.start;
            a.start = 0;
            a.logical_end -= start;
            a.physical_end -= start;
            v.start = v
                .start
                .checked_sub(start)
                .ok_or(StorageError::InvalidView)?;
            v.logical_end = v
                .logical_end
                .checked_sub(start)
                .ok_or(StorageError::InvalidView)?;
            v.physical_end = v
                .physical_end
                .checked_sub(start)
                .ok_or(StorageError::InvalidView)?;
        }
        Ok(Self {
            precision: shard.format.precision,
            order: shard.format.layout.order,
            allocation,
            view,
        })
    }
}

/// Returns the physical allocation size of a shard.
pub(crate) fn storage_bytes(shard: TensorStorage<'_>) -> StorageResult<u32> {
    let elements = shard.extents.iter().try_fold(1u64, |product, extent| {
        product
            .checked_mul(u64::from(extent.physical_end - extent.start))
            .ok_or(StorageError::Overflow)
    })?;
    u32::try_from(
        elements
            .checked_mul(shard.format.precision.bytes())
            .ok_or(StorageError::Overflow)?,
    )
    .map_err(|_| StorageError::Overflow)
}

/// Converts a rectangular semantic view into coalesced physical byte spans.
///
/// Block-major matrices and transposed AMP operands are packed independently
/// for each outer matrix. AMP left and output operands flatten their outer
/// axes into the row dimension, which lets one tile-kernel invocation process
/// activation batches with shared weights.
#[cfg(test)]
pub(crate) fn physical_byte_spans(
    shard: TensorStorage<'_>,
    view: &[ShardExtent],
) -> StorageResult<Vec<ByteSpan>> {
    Ok(byte_traversal(shard, view, true)?.spans().collect())
}

/// Converts a semantic view into physical spans ordered by canonical logical
/// coordinates. Zipping spans from two layouts therefore describes a layout
/// conversion without materializing an intermediate packed buffer.
pub(crate) fn logical_byte_spans(
    shard: TensorStorage<'_>,
    view: &[ShardExtent],
) -> StorageResult<Vec<ByteSpan>> {
    Ok(byte_traversal(shard, view, false)?.spans().collect())
}

// Walk contiguous storage lanes instead of visiting each element. Physical
// traversal may exchange the two matrix axes; semantic traversal must retain
// canonical coordinate order for layout conversions.
#[cfg(test)]
fn byte_spans(
    shard: TensorStorage<'_>,
    view: &[ShardExtent],
    physical_order: bool,
) -> StorageResult<Vec<ByteSpan>> {
    validate_view(shard, view)?;
    let rank = view.len();
    let (row_fast, lane) = match shard.format.layout.order {
        ElementOrder::RowMajor => (false, u32::MAX),
        ElementOrder::Amp(AmpOrder::Output) => (false, 2),
        ElementOrder::Amp(AmpOrder::TransposedOutput) => (true, 2),
        ElementOrder::Amp(AmpOrder::TransposedLeft)
        | ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. }) => {
            (true, amp_micro_dimension(shard.format.precision))
        }
        _ => (false, amp_micro_dimension(shard.format.precision)),
    };
    let swap_axes = physical_order && row_fast && rank >= 2;
    let lane = if row_fast && !swap_axes { 1 } else { lane };
    let shard_widths = shard
        .extents
        .iter()
        .map(|extent| extent.physical_end - extent.start)
        .collect::<Vec<_>>();
    let mut view_widths = view
        .iter()
        .map(|extent| extent.physical_end - extent.start)
        .collect::<Vec<_>>();
    if swap_axes {
        view_widths.swap(rank - 2, rank - 1);
    }
    let elements = view_widths.iter().try_fold(1_u64, |elements, &width| {
        elements
            .checked_mul(u64::from(width))
            .ok_or(StorageError::Overflow)
    })?;
    let element_bytes =
        u32::try_from(shard.format.precision.bytes()).map_err(|_| StorageError::Overflow)?;
    let mut view_coordinates = vec![0; view_widths.len()];
    let mut shard_coordinates = vec![0; shard_widths.len()];
    let mut spans = Vec::<ByteSpan>::new();
    let mut logical = 0;
    while logical < elements {
        decode_row_major(&view_widths, logical, &mut view_coordinates);
        let remaining = view_widths
            .last()
            .zip(view_coordinates.last())
            .map_or(1, |(&width, &coordinate)| width - coordinate);
        if swap_axes {
            view_coordinates.swap(rank - 2, rank - 1);
        }
        for ((shard_coordinate, view_coordinate), (shard_extent, view_extent)) in shard_coordinates
            .iter_mut()
            .zip(&view_coordinates)
            .zip(shard.extents.iter().zip(view))
        {
            *shard_coordinate = view_extent.start - shard_extent.start + view_coordinate;
        }
        let physical = physical_index(shard, &shard_widths, &shard_coordinates)?;
        let coordinate = if swap_axes {
            shard_coordinates[rank - 2]
        } else {
            shard_coordinates.last().copied().unwrap_or(0)
        };
        let run = remaining.min(lane - coordinate % lane);
        let bytes = run
            .checked_mul(element_bytes)
            .ok_or(StorageError::Overflow)?;
        logical += u64::from(run);
        let offset = u32::try_from(physical)
            .ok()
            .and_then(|index| index.checked_mul(element_bytes))
            .ok_or(StorageError::Overflow)?;
        match spans.last_mut() {
            Some(previous) if previous.offset.checked_add(previous.bytes) == Some(offset) => {
                previous.bytes = previous
                    .bytes
                    .checked_add(bytes)
                    .ok_or(StorageError::Overflow)?;
            }
            _ => spans.push(ByteSpan { offset, bytes }),
        }
    }
    Ok(spans)
}

#[cfg(test)]
fn physical_index(
    shard: TensorStorage<'_>,
    widths: &[u32],
    coordinates: &[u32],
) -> StorageResult<u64> {
    let rank = widths.len();
    match shard.format.layout.order {
        ElementOrder::RowMajor => encode_row_major(widths, coordinates),
        ElementOrder::BlockMajor(order) => {
            if rank < 2 {
                return Err(StorageError::MatrixRank);
            }
            let rows = widths[rank - 2];
            let columns = widths[rank - 1];
            let outer = encode_row_major(&widths[..rank - 2], &coordinates[..rank - 2])?;
            let row = coordinates[rank - 2];
            let column = coordinates[rank - 1];
            let matrix_elements = u64::from(rows) * u64::from(columns);
            let within = block_major_matrix_index(
                order,
                shard.format.precision,
                rows,
                columns,
                row,
                column,
            )?;
            outer
                .checked_mul(matrix_elements)
                .and_then(|base| base.checked_add(u64::from(within)))
                .ok_or(StorageError::Overflow)
        }
        ElementOrder::Amp(role) => {
            if rank < 2 {
                return Err(StorageError::MatrixRank);
            }
            let rows = widths[rank - 2];
            let columns = widths[rank - 1];
            let outer = encode_row_major(&widths[..rank - 2], &coordinates[..rank - 2])?;
            let row = coordinates[rank - 2];
            let column = coordinates[rank - 1];
            if role == AmpOrder::TransposedRight {
                let matrix_elements = u64::from(rows) * u64::from(columns);
                let within =
                    right_matrix_index(shard.format.precision, columns, rows, column, row)?;
                outer
                    .checked_mul(matrix_elements)
                    .and_then(|base| base.checked_add(u64::from(within)))
                    .ok_or(StorageError::Overflow)
            } else if matches!(role, AmpOrder::Left | AmpOrder::Output) {
                let flat_rows = widths[..rank - 2].iter().try_fold(rows, |rows, &extent| {
                    rows.checked_mul(extent).ok_or(StorageError::Overflow)
                })?;
                amp_matrix_index(
                    role,
                    shard.format.precision,
                    flat_rows,
                    columns,
                    u32::try_from(outer)
                        .ok()
                        .and_then(|outer| outer.checked_mul(rows))
                        .and_then(|row_base| row_base.checked_add(row))
                        .ok_or(StorageError::Overflow)?,
                    column,
                )
                .map(u64::from)
            } else {
                let matrix_elements = u64::from(rows) * u64::from(columns);
                let within =
                    amp_matrix_index(role, shard.format.precision, rows, columns, row, column)?;
                outer
                    .checked_mul(matrix_elements)
                    .and_then(|base| base.checked_add(u64::from(within)))
                    .ok_or(StorageError::Overflow)
            }
        }
    }
}

#[cfg(test)]
fn encode_row_major(widths: &[u32], coordinates: &[u32]) -> StorageResult<u64> {
    widths
        .iter()
        .zip(coordinates)
        .try_fold(0_u64, |linear, (&width, &coordinate)| {
            if coordinate >= width {
                return Err(StorageError::InvalidView);
            }
            linear
                .checked_mul(u64::from(width))
                .and_then(|linear| linear.checked_add(u64::from(coordinate)))
                .ok_or(StorageError::Overflow)
        })
}

#[cfg(test)]
fn amp_matrix_index(
    role: AmpOrder,
    precision: Precision,
    rows: u32,
    columns: u32,
    row: u32,
    column: u32,
) -> StorageResult<u32> {
    const COLUMN_MICRO: u32 = AMP_COLUMN_MICRO;
    if row >= rows || column >= columns {
        return Err(StorageError::InvalidView);
    }
    match role {
        AmpOrder::Left => {
            let inner = amp_micro_dimension(precision);
            if !columns.is_multiple_of(inner) {
                return Err(StorageError::AmpBlock { role });
            }
            (column / inner)
                .checked_mul(rows)
                .and_then(|panel| panel.checked_mul(inner))
                .and_then(|base| base.checked_add(row * inner + column % inner))
                .ok_or(StorageError::Overflow)
        }
        AmpOrder::TransposedRight => right_matrix_index(precision, columns, rows, column, row),
        AmpOrder::TransposedLeft => {
            amp_matrix_index(AmpOrder::Left, precision, columns, rows, column, row)
        }
        AmpOrder::Output => {
            if !columns.is_multiple_of(COLUMN_MICRO) {
                return Err(StorageError::AmpBlock { role });
            }
            let logical_pair = column % COLUMN_MICRO / 2;
            let physical_pair = logical_pair % 4 * 2 + logical_pair / 4;
            let physical_column = physical_pair * 2 + column % 2;
            (column / COLUMN_MICRO)
                .checked_mul(rows * COLUMN_MICRO)
                .and_then(|base| base.checked_add(row * COLUMN_MICRO + physical_column))
                .ok_or(StorageError::Overflow)
        }
        AmpOrder::TransposedOutput => {
            amp_matrix_index(AmpOrder::Output, precision, columns, rows, column, row)
        }
    }
}

#[cfg(test)]
fn right_matrix_index(
    precision: Precision,
    rows: u32,
    columns: u32,
    row: u32,
    column: u32,
) -> StorageResult<u32> {
    let inner = amp_micro_dimension(precision);
    if row >= rows
        || column >= columns
        || !rows.is_multiple_of(inner)
        || !columns.is_multiple_of(AMP_COLUMN_MICRO)
    {
        return Err(StorageError::InvalidView);
    }
    let panel = (column / AMP_COLUMN_MICRO)
        .checked_mul(rows / inner)
        .and_then(|base| base.checked_add(row / inner))
        .ok_or(StorageError::Overflow)?;
    panel
        .checked_mul(inner * AMP_COLUMN_MICRO)
        .and_then(|base| base.checked_add(column % AMP_COLUMN_MICRO * inner + row % inner))
        .ok_or(StorageError::Overflow)
}

#[cfg(test)]
fn block_major_matrix_index(
    order: BlockMajorOrder,
    precision: Precision,
    rows: u32,
    columns: u32,
    row: u32,
    column: u32,
) -> StorageResult<u32> {
    let (row_block, column_block, transposed) = match order {
        BlockMajorOrder::Matrix {
            row_block,
            column_block,
        } => (u32::from(row_block), u32::from(column_block), false),
        BlockMajorOrder::TransposedMatrix {
            row_block,
            column_block,
        } => (u32::from(row_block), u32::from(column_block), true),
    };
    if transposed {
        return block_major_matrix_index(
            BlockMajorOrder::Matrix {
                row_block: row_block as u16,
                column_block: column_block as u16,
            },
            precision,
            columns,
            rows,
            column,
            row,
        );
    }
    let row_micro = amp_micro_dimension(precision);
    if row >= rows
        || column >= columns
        || row_block == 0
        || column_block == 0
        || !rows.is_multiple_of(row_block)
        || !columns.is_multiple_of(column_block)
        || !row_block.is_multiple_of(row_micro)
    {
        return Err(StorageError::BlockMajorBlock { order });
    }
    let row_groups = row_block / row_micro;
    let panel = (row / row_block)
        .checked_mul(columns / column_block)
        .and_then(|block| block.checked_mul(row_groups))
        .and_then(|base| {
            base.checked_add(column / column_block * row_groups + row % row_block / row_micro)
        })
        .ok_or(StorageError::Overflow)?;
    panel
        .checked_mul(row_micro * column_block)
        .and_then(|base| base.checked_add(column % column_block * row_micro + row % row_micro))
        .ok_or(StorageError::Overflow)
}

fn validate_view(shard: TensorStorage<'_>, view: &[ShardExtent]) -> StorageResult<()> {
    if view.len() != shard.extents.len()
        || view.iter().zip(shard.extents).any(|(view, shard)| {
            view.axis != shard.axis
                || view.start < shard.start
                || view.start > view.logical_end
                || view.logical_end > view.physical_end
                || view.physical_end > shard.physical_end
        })
    {
        tracing::error!(
            layout = ?shard.format.layout,
            shard_extents = ?shard.extents,
            view_extents = ?view,
            "tensor view falls outside its storage shard"
        );
        return Err(StorageError::InvalidView);
    }
    Ok(())
}

#[cfg(test)]
fn physical_coordinates(
    shard: TensorStorage<'_>,
    widths: &[u32],
    physical: u64,
) -> StorageResult<Vec<u32>> {
    let rank = widths.len();
    let mut coordinates = vec![0; rank];
    match shard.format.layout.order {
        ElementOrder::RowMajor => decode_row_major(widths, physical, &mut coordinates),
        ElementOrder::BlockMajor(order) => {
            if rank < 2 {
                return Err(StorageError::MatrixRank);
            }
            let rows = widths[rank - 2];
            let columns = widths[rank - 1];
            let matrix_elements = u64::from(rows) * u64::from(columns);
            let outer = physical / matrix_elements;
            decode_row_major(&widths[..rank - 2], outer, &mut coordinates[..rank - 2]);
            let linear =
                u32::try_from(physical % matrix_elements).map_err(|_| StorageError::Overflow)?;
            let (row, column) = block_major_matrix_coordinates(
                order,
                shard.format.precision,
                rows,
                columns,
                linear,
            )?;
            coordinates[rank - 2] = row;
            coordinates[rank - 1] = column;
        }
        ElementOrder::Amp(role) => {
            if rank < 2 {
                return Err(StorageError::MatrixRank);
            }
            let rows = widths[rank - 2];
            let columns = widths[rank - 1];
            if matches!(role, AmpOrder::Left | AmpOrder::Output) {
                let outer_rows = widths[..rank - 2]
                    .iter()
                    .try_fold(rows, |product, &extent| {
                        product.checked_mul(extent).ok_or(StorageError::Overflow)
                    })?;
                let linear = u32::try_from(physical).map_err(|_| StorageError::Overflow)?;
                let (flat_row, column) = amp_matrix_coordinates(
                    role,
                    shard.format.precision,
                    outer_rows,
                    columns,
                    linear,
                )?;
                let outer = flat_row / rows;
                decode_row_major(
                    &widths[..rank - 2],
                    u64::from(outer),
                    &mut coordinates[..rank - 2],
                );
                coordinates[rank - 2] = flat_row % rows;
                coordinates[rank - 1] = column;
            } else {
                let matrix_elements = u64::from(rows) * u64::from(columns);
                let outer = physical / matrix_elements;
                decode_row_major(&widths[..rank - 2], outer, &mut coordinates[..rank - 2]);
                let linear = u32::try_from(physical % matrix_elements)
                    .map_err(|_| StorageError::Overflow)?;
                let (row, column) =
                    amp_matrix_coordinates(role, shard.format.precision, rows, columns, linear)?;
                coordinates[rank - 2] = row;
                coordinates[rank - 1] = column;
            }
        }
    }
    Ok(coordinates)
}

#[cfg(test)]
fn decode_row_major(widths: &[u32], mut linear: u64, output: &mut [u32]) {
    for (coordinate, &width) in output.iter_mut().zip(widths).rev() {
        *coordinate = (linear % u64::from(width)) as u32;
        linear /= u64::from(width);
    }
}

/// Maps a physical linear element in one AMP-packed matrix to its logical
/// row and column. This is shared by package-data producers and placement.
pub fn amp_matrix_coordinates(
    role: AmpOrder,
    precision: Precision,
    rows: u32,
    columns: u32,
    linear: u32,
) -> StorageResult<(u32, u32)> {
    const COLUMN_MICRO: u32 = AMP_COLUMN_MICRO;
    match role {
        AmpOrder::Left => {
            let inner = amp_micro_dimension(precision);
            if !columns.is_multiple_of(inner) {
                return Err(StorageError::AmpBlock { role });
            }
            let panel_elements = rows.checked_mul(inner).ok_or(StorageError::Overflow)?;
            let panel = linear / panel_elements;
            let offset = linear % panel_elements;
            Ok((offset / inner, panel * inner + offset % inner))
        }
        AmpOrder::TransposedRight => right_matrix_coordinates(precision, columns, rows, linear)
            .map(|(column, row)| (row, column)),
        AmpOrder::TransposedLeft => {
            amp_matrix_coordinates(AmpOrder::Left, precision, columns, rows, linear)
                .map(|(column, row)| (row, column))
        }
        AmpOrder::Output => {
            if !columns.is_multiple_of(COLUMN_MICRO) {
                return Err(StorageError::AmpBlock { role });
            }
            let panel_elements = rows
                .checked_mul(COLUMN_MICRO)
                .ok_or(StorageError::Overflow)?;
            let panel = linear / panel_elements;
            let offset = linear % panel_elements;
            let physical_column = offset % COLUMN_MICRO;
            let physical_pair = physical_column / 2;
            let logical_pair = (physical_pair % 2) * 4 + physical_pair / 2;
            Ok((
                offset / COLUMN_MICRO,
                panel * COLUMN_MICRO + logical_pair * 2 + physical_column % 2,
            ))
        }
        AmpOrder::TransposedOutput => {
            amp_matrix_coordinates(AmpOrder::Output, precision, columns, rows, linear)
                .map(|(column, row)| (row, column))
        }
    }
}

fn right_matrix_coordinates(
    precision: Precision,
    rows: u32,
    columns: u32,
    linear: u32,
) -> StorageResult<(u32, u32)> {
    let inner = amp_micro_dimension(precision);
    if !rows.is_multiple_of(inner) || !columns.is_multiple_of(AMP_COLUMN_MICRO) {
        return Err(StorageError::InvalidView);
    }
    let panel_elements = inner * AMP_COLUMN_MICRO;
    let panel = linear / panel_elements;
    let offset = linear % panel_elements;
    let inner_groups = rows / inner;
    let column_group = panel / inner_groups;
    let inner_group = panel % inner_groups;
    Ok((
        inner_group * inner + offset % inner,
        column_group * AMP_COLUMN_MICRO + offset / inner,
    ))
}

/// Maps a physical linear element in block-major storage to its logical
/// matrix coordinate.
pub fn block_major_matrix_coordinates(
    order: BlockMajorOrder,
    precision: Precision,
    rows: u32,
    columns: u32,
    linear: u32,
) -> StorageResult<(u32, u32)> {
    let (row_block, column_block, transposed) = match order {
        BlockMajorOrder::Matrix {
            row_block,
            column_block,
        } => (u32::from(row_block), u32::from(column_block), false),
        BlockMajorOrder::TransposedMatrix {
            row_block,
            column_block,
        } => (u32::from(row_block), u32::from(column_block), true),
    };
    if transposed {
        return block_major_matrix_coordinates(
            BlockMajorOrder::Matrix {
                row_block: row_block as u16,
                column_block: column_block as u16,
            },
            precision,
            columns,
            rows,
            linear,
        )
        .map(|(column, row)| (row, column));
    }
    let row_micro = amp_micro_dimension(precision);
    if row_block == 0
        || column_block == 0
        || !rows.is_multiple_of(row_block)
        || !columns.is_multiple_of(column_block)
        || !row_block.is_multiple_of(row_micro)
    {
        return Err(StorageError::BlockMajorBlock { order });
    }
    let panel_elements = row_micro * column_block;
    let panel = linear / panel_elements;
    let offset = linear % panel_elements;
    let row_groups = row_block / row_micro;
    let column_groups = columns / column_block;
    let panels_per_row_block = column_groups * row_groups;
    let row_block_index = panel / panels_per_row_block;
    let within_block = panel % panels_per_row_block;
    let column_group = within_block / row_groups;
    let row_group = within_block % row_groups;
    Ok((
        row_block_index * row_block + row_group * row_micro + offset % row_micro,
        column_group * column_block + offset / row_micro,
    ))
}

fn amp_micro_dimension(precision: Precision) -> u32 {
    match precision {
        Precision::F8F143 { .. } => 32,
        Precision::F16 => 16,
        Precision::F32 => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mid::ShardExtent;
    use crate::{
        AMP_COLUMN_MICRO, AMP_INNER_BLOCK, AmpOrder, BlockMajorOrder, ElementOrder, Layout,
        MemoryClass, Precision, TensorTiling, TensorType,
    };
    use crate::{
        BlockValue, BlockValueId, ShardDefinition, ShardView, logical_view_byte_spans,
        shard_storage_bytes, view_byte_spans,
    };

    pub(super) fn shard(layout: Layout, dimensions: &[u32]) -> BlockValue {
        BlockValue {
            id: BlockValueId::from_index(0),
            tile: 0,
            tensor_type: TensorType::new(dimensions.iter().copied(), Precision::F16, layout),
            extents: dimensions
                .iter()
                .enumerate()
                .map(|(axis, &end)| ShardExtent {
                    axis: axis as u16,
                    start: 0,
                    logical_end: end,
                    physical_end: end,
                })
                .collect(),
            definition: ShardDefinition::Value(crate::MidValueId::from_index(0)),
        }
    }

    #[test]
    fn fp8_panel_regrouping_preserves_logical_coordinates() {
        // Independently compare the device's two-pointer regrouping with the
        // storage codecs, including multiple panels and rectangular matrices.
        for (rows, columns) in [(32, 64), (96, 128), (128, 32)] {
            for order in [
                ElementOrder::Amp(AmpOrder::Left),
                ElementOrder::Amp(AmpOrder::TransposedLeft),
                ElementOrder::Amp(AmpOrder::TransposedRight),
                ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                    row_block: 32,
                    column_block: 16,
                }),
                ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix {
                    row_block: 32,
                    column_block: 16,
                }),
            ] {
                let group_rows = match order {
                    ElementOrder::Amp(AmpOrder::Left) => rows,
                    ElementOrder::Amp(AmpOrder::TransposedLeft) => columns,
                    _ => 16,
                };
                let coordinates = |precision, linear| match order {
                    ElementOrder::Amp(role) => {
                        amp_matrix_coordinates(role, precision, rows, columns, linear).unwrap()
                    }
                    ElementOrder::BlockMajor(role) => {
                        block_major_matrix_coordinates(role, precision, rows, columns, linear)
                            .unwrap()
                    }
                    _ => unreachable!(),
                };
                for destination in 0..rows * columns {
                    let panel = destination / (group_rows * 32) * (group_rows * 32);
                    let offset = destination - panel;
                    let source = panel
                        + (offset % 32 / 16) * group_rows * 16
                        + (offset / 32) * 16
                        + offset % 16;
                    assert_eq!(
                        coordinates(Precision::F16, source),
                        coordinates(Precision::F8F143 { scale_exponent: -4 }, destination),
                        "{order:?} {rows}x{columns} element {destination}"
                    );
                }
            }
        }
    }

    #[test]
    fn partial_span_traversals_preserve_coordinates_in_every_storage_order() {
        let mut random = fastrand::Rng::with_seed(0x7370_616e);
        for precision in [
            Precision::F8F143 { scale_exponent: 0 },
            Precision::F16,
            Precision::F32,
        ] {
            for order in [
                ElementOrder::RowMajor,
                ElementOrder::Amp(AmpOrder::Left),
                ElementOrder::Amp(AmpOrder::Output),
                ElementOrder::Amp(AmpOrder::TransposedLeft),
                ElementOrder::Amp(AmpOrder::TransposedRight),
                ElementOrder::Amp(AmpOrder::TransposedOutput),
                ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                    row_block: 64,
                    column_block: 64,
                }),
                ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix {
                    row_block: 64,
                    column_block: 64,
                }),
            ] {
                let mut block = shard(
                    Layout {
                        order,
                        tiling: TensorTiling::replicated(1),
                        memory_class: MemoryClass::Ipu21Standard,
                    },
                    &[3, 64, 128],
                );
                block.tensor_type.format.precision = precision;
                for extent in &mut block.extents {
                    extent.start += 7;
                    extent.logical_end += 7;
                    extent.physical_end += 7;
                }
                for _ in 0..16 {
                    let view = block
                        .extents
                        .iter()
                        .map(|extent| {
                            let start = random.u32(extent.start..extent.physical_end);
                            let end = random.u32(start + 1..=extent.physical_end);
                            ShardExtent {
                                start,
                                logical_end: end,
                                physical_end: end,
                                ..*extent
                            }
                        })
                        .collect::<Vec<_>>();
                    let mut expected = Vec::new();
                    for batch in view[0].start..view[0].physical_end {
                        for row in view[1].start..view[1].physical_end {
                            for column in view[2].start..view[2].physical_end {
                                let index = physical_index(
                                    block.storage(),
                                    &[3, 64, 128],
                                    &[batch - 7, row - 7, column - 7],
                                )
                                .unwrap();
                                let offset = index as u32 * precision.bytes() as u32;
                                expected.extend(offset..offset + precision.bytes() as u32);
                            }
                        }
                    }
                    let expand = |spans: Vec<ByteSpan>| {
                        spans
                            .into_iter()
                            .flat_map(|span| span.offset..span.offset + span.bytes)
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(
                        expand(logical_byte_spans(block.storage(), &view).unwrap()),
                        expected,
                        "{precision:?} {order:?}"
                    );
                    expected.sort_unstable();
                    assert_eq!(
                        expand(physical_byte_spans(block.storage(), &view).unwrap()),
                        expected,
                        "{precision:?} {order:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn randomized_amp_mappings_are_bijections_and_views_preserve_element_count() {
        let mut state = 0x9e37_79b9u32;
        for _ in 0..128 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let rows = 1 + (state % 19);
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let batches = 1 + (state % 4);
            for (layout, physical_rows, columns) in [
                (Layout::amp_left(64, 1), rows, 64),
                (Layout::block_major_matrix(64, 1), 64, 64),
                (
                    Layout::block_major_matrix_storage(
                        64,
                        64,
                        1,
                        1,
                        1,
                        MemoryClass::Ipu21Interleaved,
                    ),
                    64,
                    64,
                ),
                (Layout::amp_output(1), rows, 64),
            ] {
                let shard = shard(layout, &[batches, physical_rows, columns]);
                let full = ShardView {
                    shard: shard.id,
                    extents: shard.extents.clone(),
                };
                assert_eq!(
                    view_byte_spans(&shard, &full).unwrap(),
                    vec![ByteSpan {
                        offset: 0,
                        bytes: shard_storage_bytes(&shard).unwrap(),
                    }]
                );
                let widths = shard
                    .extents
                    .iter()
                    .map(|extent| extent.physical_end - extent.start)
                    .collect::<Vec<_>>();
                let ordered = logical_view_byte_spans(&shard, &full).unwrap();
                let mut expected = vec![0; widths.len()];
                let mut logical = 0_u64;
                for span in ordered {
                    for offset in (span.offset..span.offset + span.bytes).step_by(2) {
                        decode_row_major(&widths, logical, &mut expected);
                        assert_eq!(
                            physical_coordinates(shard.storage(), &widths, u64::from(offset / 2))
                                .unwrap(),
                            expected
                        );
                        logical += 1;
                    }
                }
                assert_eq!(
                    logical,
                    widths
                        .iter()
                        .map(|&width| u64::from(width))
                        .product::<u64>()
                );

                let mut view = full;
                view.extents[1].start = physical_rows / 3;
                view.extents[1].logical_end = physical_rows - physical_rows / 4;
                view.extents[1].physical_end = view.extents[1].logical_end;
                let selected = u64::from(batches)
                    * u64::from(view.extents[1].physical_end - view.extents[1].start)
                    * u64::from(columns)
                    * 2;
                assert_eq!(
                    view_byte_spans(&shard, &view)
                        .unwrap()
                        .iter()
                        .map(|span| u64::from(span.bytes))
                        .sum::<u64>(),
                    selected
                );
            }
        }
    }

    #[test]
    fn randomized_word_aligned_logical_transfers_populate_packed_storage() {
        let mut random = fastrand::Rng::with_seed(0x776f_7264_5f70_6163);
        for case in 0..128 {
            let batches = random.u32(1..=3);
            let rows = AMP_INNER_BLOCK;
            let columns = random.u32(2..=48) * 2;
            let physical_columns = columns.div_ceil(AMP_COLUMN_MICRO) * AMP_COLUMN_MICRO;
            let source = shard(
                Layout::row_major(TensorTiling::replicated(1)),
                &[batches, rows, columns],
            );
            let source_view = ShardView {
                shard: source.id,
                extents: source.extents.clone(),
            };
            for order in [
                ElementOrder::Amp(AmpOrder::Left),
                ElementOrder::Amp(AmpOrder::TransposedRight),
            ] {
                let mut destination = shard(
                    Layout {
                        order,
                        tiling: TensorTiling::replicated(1),
                        memory_class: MemoryClass::Ipu21Standard,
                    },
                    &[batches, rows, columns],
                );
                destination.extents[2].physical_end = physical_columns;
                let mut destination_view = ShardView {
                    shard: destination.id,
                    extents: destination.extents.clone(),
                };
                destination_view.extents[2].physical_end = columns;
                let source_spans = logical_view_byte_spans(&source, &source_view).unwrap();
                let destination_spans =
                    logical_view_byte_spans(&destination, &destination_view).unwrap();
                assert!(
                    source_spans
                        .iter()
                        .chain(&destination_spans)
                        .all(|span| span.offset & 0b11 == 0 && span.bytes & 0b11 == 0),
                    "case {case}, order {order:?}"
                );

                let logical_elements = usize::try_from(batches * rows * columns).unwrap();
                let source_data = (0..logical_elements)
                    .map(|index| u16::try_from(index).unwrap())
                    .collect::<Vec<_>>();
                let mut destination_data =
                    vec![0u16; usize::try_from(batches * rows * physical_columns).unwrap()];
                let mut source_position = (0usize, 0u32);
                let mut destination_position = (0usize, 0u32);
                while source_position.0 < source_spans.len() {
                    let source_span = source_spans[source_position.0];
                    let destination_span = destination_spans[destination_position.0];
                    let bytes = (source_span.bytes - source_position.1)
                        .min(destination_span.bytes - destination_position.1);
                    let source_start =
                        usize::try_from((source_span.offset + source_position.1) / 2).unwrap();
                    let destination_start =
                        usize::try_from((destination_span.offset + destination_position.1) / 2)
                            .unwrap();
                    let elements = usize::try_from(bytes / 2).unwrap();
                    destination_data[destination_start..destination_start + elements]
                        .copy_from_slice(&source_data[source_start..source_start + elements]);
                    source_position.1 += bytes;
                    destination_position.1 += bytes;
                    if source_position.1 == source_span.bytes {
                        source_position = (source_position.0 + 1, 0);
                    }
                    if destination_position.1 == destination_span.bytes {
                        destination_position = (destination_position.0 + 1, 0);
                    }
                }
                let unpacked = destination_spans
                    .iter()
                    .flat_map(|span| {
                        (span.offset / 2..(span.offset + span.bytes) / 2)
                            .map(|index| destination_data[index as usize])
                    })
                    .collect::<Vec<_>>();
                assert_eq!(unpacked, source_data, "case {case}, order {order:?}");
            }
        }
    }

    #[test]
    fn randomized_block_major_panels_are_single_contiguous_spans() {
        let mut random = fastrand::Rng::with_seed(0x6b36_3472);
        for _ in 0..128 {
            let output_column_block = [32, 64, 128][random.usize(0..3)];
            let inner_blocks = random.u32(1..=8);
            let column_blocks = random.u32(1..=4);
            let rows = inner_blocks * 64;
            let columns = column_blocks * output_column_block;
            let batches = random.u32(1..=4);
            let shard = shard(
                Layout::block_major_matrix_storage(
                    64,
                    output_column_block,
                    1,
                    1,
                    1,
                    MemoryClass::Ipu21Interleaved,
                ),
                &[batches, rows, columns],
            );
            let inner = random.u32(0..inner_blocks) * 64;
            let column_block = random.u32(0..column_blocks);
            let view_column_blocks = random.u32(1..=column_blocks - column_block);
            let column = column_block * output_column_block;
            let view_columns = view_column_blocks * output_column_block;
            let view = ShardView {
                shard: shard.id,
                extents: vec![
                    ShardExtent {
                        axis: 0,
                        start: 0,
                        logical_end: batches,
                        physical_end: batches,
                    },
                    ShardExtent {
                        axis: 1,
                        start: inner,
                        logical_end: inner + 64,
                        physical_end: inner + 64,
                    },
                    ShardExtent {
                        axis: 2,
                        start: column,
                        logical_end: column + view_columns,
                        physical_end: column + view_columns,
                    },
                ],
            };
            let spans = view_byte_spans(&shard, &view).unwrap();
            assert_eq!(
                spans.iter().map(|span| span.bytes).sum::<u32>(),
                batches * 64 * view_columns * 2
            );
            assert_eq!(
                spans[0].offset,
                ((inner / 64) * column_blocks + column / output_column_block)
                    * 64
                    * output_column_block
                    * 2
            );
        }
    }

    #[test]
    fn randomized_attention_micro_panels_are_directly_transferable() {
        let mut random = fastrand::Rng::with_seed(0x6d69_6372_6f21);
        for _ in 0..128 {
            let panels = random.u32(1..=8);
            let selected = random.u32(0..panels);
            for order in [
                ElementOrder::Amp(AmpOrder::TransposedRight),
                ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                    row_block: 64,
                    column_block: AMP_COLUMN_MICRO as u16,
                }),
            ] {
                let layout = Layout {
                    order,
                    tiling: crate::TensorTiling::replicated(1),
                    memory_class: MemoryClass::Ipu21Standard,
                };
                let source = shard(layout.clone(), &[AMP_INNER_BLOCK, AMP_COLUMN_MICRO]);
                let destination = shard(layout, &[AMP_INNER_BLOCK, panels * AMP_COLUMN_MICRO]);
                let source_view = ShardView {
                    shard: source.id,
                    extents: source.extents.clone(),
                };
                let destination_view = ShardView {
                    shard: destination.id,
                    extents: vec![
                        destination.extents[0],
                        ShardExtent {
                            axis: 1,
                            start: selected * AMP_COLUMN_MICRO,
                            logical_end: (selected + 1) * AMP_COLUMN_MICRO,
                            physical_end: (selected + 1) * AMP_COLUMN_MICRO,
                        },
                    ],
                };
                let source_spans = view_byte_spans(&source, &source_view).unwrap();
                let destination_spans = view_byte_spans(&destination, &destination_view).unwrap();
                assert_eq!(
                    source_spans.iter().map(|span| span.bytes).sum::<u32>(),
                    destination_spans.iter().map(|span| span.bytes).sum::<u32>()
                );
                let source_offsets = source_spans
                    .iter()
                    .flat_map(|span| (span.offset..span.offset + span.bytes).step_by(2));
                let destination_offsets = destination_spans
                    .iter()
                    .flat_map(|span| (span.offset..span.offset + span.bytes).step_by(2));
                for (source_offset, destination_offset) in source_offsets.zip(destination_offsets) {
                    let source_coordinates = physical_coordinates(
                        source.storage(),
                        &[AMP_INNER_BLOCK, AMP_COLUMN_MICRO],
                        u64::from(source_offset / 2),
                    )
                    .unwrap();
                    let destination_coordinates = physical_coordinates(
                        destination.storage(),
                        &[AMP_INNER_BLOCK, panels * AMP_COLUMN_MICRO],
                        u64::from(destination_offset / 2),
                    )
                    .unwrap();
                    assert_eq!(source_coordinates[0], destination_coordinates[0]);
                    assert_eq!(
                        source_coordinates[1] + selected * AMP_COLUMN_MICRO,
                        destination_coordinates[1]
                    );
                }
            }
        }
    }

    #[test]
    fn randomized_compatible_micro_panel_orders_preserve_coordinates() {
        let mut random = fastrand::Rng::with_seed(0x7061_6e65_6c21);
        for case in 0..128 {
            let precision = if case % 2 == 0 {
                Precision::F16
            } else {
                Precision::F8F143 { scale_exponent: -4 }
            };
            let bytes = precision.bytes() as u32;
            let panels = random.u32(1..=8) * 2;
            let selected = random.u32(0..panels);
            for (source_order, destination_order) in [
                (
                    ElementOrder::Amp(AmpOrder::Left),
                    ElementOrder::Amp(AmpOrder::TransposedRight),
                ),
                (
                    ElementOrder::Amp(AmpOrder::TransposedLeft),
                    ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                        row_block: 64,
                        column_block: AMP_COLUMN_MICRO as u16,
                    }),
                ),
            ] {
                let layout = |order| Layout {
                    order,
                    tiling: crate::TensorTiling::replicated(1),
                    memory_class: MemoryClass::Ipu21Standard,
                };
                let mut source = shard(
                    layout(source_order),
                    &[AMP_INNER_BLOCK, panels * AMP_COLUMN_MICRO],
                );
                let mut destination = shard(layout(destination_order), &[AMP_INNER_BLOCK, 32]);
                source.tensor_type.format.precision = precision;
                destination.tensor_type.format.precision = precision;
                let source_view = ShardView {
                    shard: source.id,
                    extents: vec![
                        source.extents[0],
                        ShardExtent {
                            axis: 1,
                            start: selected * AMP_COLUMN_MICRO,
                            logical_end: (selected + 1) * AMP_COLUMN_MICRO,
                            physical_end: (selected + 1) * AMP_COLUMN_MICRO,
                        },
                    ],
                };
                let destination_view = ShardView {
                    shard: destination.id,
                    extents: vec![
                        destination.extents[0],
                        ShardExtent {
                            axis: 1,
                            start: 0,
                            logical_end: AMP_COLUMN_MICRO,
                            physical_end: AMP_COLUMN_MICRO,
                        },
                    ],
                };
                let source_offsets = view_byte_spans(&source, &source_view)
                    .unwrap()
                    .into_iter()
                    .flat_map(|span| {
                        (span.offset..span.offset + span.bytes).step_by(bytes as usize)
                    });
                let destination_offsets = view_byte_spans(&destination, &destination_view)
                    .unwrap()
                    .into_iter()
                    .flat_map(|span| {
                        (span.offset..span.offset + span.bytes).step_by(bytes as usize)
                    });
                let pairs = source_offsets.zip(destination_offsets).collect::<Vec<_>>();
                assert_eq!(
                    pairs.len(),
                    (AMP_INNER_BLOCK * AMP_COLUMN_MICRO) as usize,
                    "case {case}"
                );
                for (source_offset, destination_offset) in pairs {
                    let source_coordinates = physical_coordinates(
                        source.storage(),
                        &[AMP_INNER_BLOCK, panels * AMP_COLUMN_MICRO],
                        u64::from(source_offset / bytes),
                    )
                    .unwrap();
                    let destination_coordinates = physical_coordinates(
                        destination.storage(),
                        &[AMP_INNER_BLOCK, 32],
                        u64::from(destination_offset / bytes),
                    )
                    .unwrap();
                    assert_eq!(source_coordinates[0], destination_coordinates[0]);
                    assert_eq!(
                        source_coordinates[1],
                        destination_coordinates[1] + selected * AMP_COLUMN_MICRO
                    );
                }
            }
        }
    }
}
