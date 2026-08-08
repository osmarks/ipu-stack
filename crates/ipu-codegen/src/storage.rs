//! Conversion from logical shard views to physical byte ranges.

use crate::low::{LowShard, ShardView};
use crate::mid::{
    AMP_COLUMN_MICRO, AMP_INNER_BLOCK, AMP_OUTPUT_COLUMN_BLOCK, AmpOrder, ElementOrder, Precision,
};

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
    #[error("AMP storage requires a tensor of rank at least two")]
    AmpRank,
    #[error("AMP {role:?} storage dimensions are incompatible with its block shape")]
    AmpBlock { role: AmpOrder },
    #[error("tensor storage size overflowed")]
    Overflow,
}

pub type StorageResult<T> = Result<T, StorageError>;

/// Returns the physical allocation size of a shard.
pub fn shard_storage_bytes(shard: &LowShard) -> StorageResult<u32> {
    let elements = shard.extents.iter().try_fold(1u64, |product, extent| {
        product
            .checked_mul(u64::from(extent.physical_end - extent.start))
            .ok_or(StorageError::Overflow)
    })?;
    u32::try_from(
        elements
            .checked_mul(shard.tensor_type.format.precision.bytes())
            .ok_or(StorageError::Overflow)?,
    )
    .map_err(|_| StorageError::Overflow)
}

/// Converts a rectangular semantic view into coalesced physical byte spans.
///
/// AMP permutations are applied independently to the final two axes. Earlier
/// axes are conventional row-major outer dimensions, so batched tensors need
/// no special case in exchange or placement code.
pub fn view_byte_spans(shard: &LowShard, view: &ShardView) -> StorageResult<Vec<ByteSpan>> {
    validate_view(shard, view)?;
    if let Some(spans) = right_k64_panel_spans(shard, view)? {
        return Ok(spans);
    }
    let widths = shard
        .extents
        .iter()
        .map(|extent| extent.physical_end - extent.start)
        .collect::<Vec<_>>();
    let elements = widths.iter().try_fold(1u64, |product, &width| {
        product
            .checked_mul(u64::from(width))
            .ok_or(StorageError::Overflow)
    })?;
    let element_bytes = u32::try_from(shard.tensor_type.format.precision.bytes())
        .map_err(|_| StorageError::Overflow)?;
    let mut spans = Vec::<ByteSpan>::new();
    for physical in 0..elements {
        let local = physical_coordinates(shard, &widths, physical)?;
        let selected = local.iter().zip(&shard.extents).zip(&view.extents).all(
            |((&coordinate, shard_extent), view_extent)| {
                let global = shard_extent.start + coordinate;
                global >= view_extent.start && global < view_extent.physical_end
            },
        );
        if !selected {
            continue;
        }
        let offset = u32::try_from(physical)
            .ok()
            .and_then(|index| index.checked_mul(element_bytes))
            .ok_or(StorageError::Overflow)?;
        match spans.last_mut() {
            Some(last) if last.offset.checked_add(last.bytes) == Some(offset) => {
                last.bytes = last
                    .bytes
                    .checked_add(element_bytes)
                    .ok_or(StorageError::Overflow)?;
            }
            _ => spans.push(ByteSpan {
                offset,
                bytes: element_bytes,
            }),
        }
    }
    Ok(spans)
}

fn right_k64_panel_spans(
    shard: &LowShard,
    view: &ShardView,
) -> StorageResult<Option<Vec<ByteSpan>>> {
    if shard.extents.len() < 2
        || shard.tensor_type.format.layout.order != ElementOrder::Amp(AmpOrder::RightK64)
    {
        return Ok(None);
    }
    let rank = shard.extents.len();
    if shard.extents[..rank - 2] != view.extents[..rank - 2] {
        return Ok(None);
    }
    let rows = shard.extents[rank - 2].physical_end - shard.extents[rank - 2].start;
    let columns = shard.extents[rank - 1].physical_end - shard.extents[rank - 1].start;
    let inner_start = view.extents[rank - 2].start - shard.extents[rank - 2].start;
    let column_start = view.extents[rank - 1].start - shard.extents[rank - 1].start;
    let inner_width = view.extents[rank - 2].physical_end - view.extents[rank - 2].start;
    let column_width = view.extents[rank - 1].physical_end - view.extents[rank - 1].start;
    if rows.is_multiple_of(AMP_INNER_BLOCK)
        && columns.is_multiple_of(AMP_OUTPUT_COLUMN_BLOCK)
        && inner_width == AMP_INNER_BLOCK
        && column_width == AMP_OUTPUT_COLUMN_BLOCK
        && inner_start.is_multiple_of(AMP_INNER_BLOCK)
        && column_start.is_multiple_of(AMP_OUTPUT_COLUMN_BLOCK)
    {
        let panel = inner_start
            .checked_div(AMP_INNER_BLOCK)
            .and_then(|inner| inner.checked_mul(columns / AMP_OUTPUT_COLUMN_BLOCK))
            .and_then(|panel| panel.checked_add(column_start / AMP_OUTPUT_COLUMN_BLOCK))
            .ok_or(StorageError::Overflow)?;
        let bytes = AMP_INNER_BLOCK
            .checked_mul(AMP_OUTPUT_COLUMN_BLOCK)
            .and_then(|elements| {
                elements.checked_mul(shard.tensor_type.format.precision.bytes() as u32)
            })
            .ok_or(StorageError::Overflow)?;
        let matrix_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| {
                elements.checked_mul(shard.tensor_type.format.precision.bytes() as u32)
            })
            .ok_or(StorageError::Overflow)?;
        let outer = shard.extents[..rank - 2]
            .iter()
            .try_fold(1u32, |product, extent| {
                product
                    .checked_mul(extent.physical_end - extent.start)
                    .ok_or(StorageError::Overflow)
            })?;
        let panel_offset = panel.checked_mul(bytes).ok_or(StorageError::Overflow)?;
        let mut spans = Vec::<ByteSpan>::with_capacity(outer as usize);
        for matrix in 0..outer {
            let offset = matrix
                .checked_mul(matrix_bytes)
                .and_then(|offset| offset.checked_add(panel_offset))
                .ok_or(StorageError::Overflow)?;
            match spans.last_mut() {
                Some(previous) if previous.offset + previous.bytes == offset => {
                    previous.bytes = previous
                        .bytes
                        .checked_add(bytes)
                        .ok_or(StorageError::Overflow)?;
                }
                _ => spans.push(ByteSpan { offset, bytes }),
            }
        }
        return Ok(Some(spans));
    }
    Ok(None)
}

fn validate_view(shard: &LowShard, view: &ShardView) -> StorageResult<()> {
    if view.shard != shard.id {
        return Err(StorageError::WrongShard);
    }
    if view.extents.len() != shard.extents.len()
        || view
            .extents
            .iter()
            .zip(&shard.extents)
            .any(|(view, shard)| {
                view.axis != shard.axis
                    || view.start < shard.start
                    || view.start > view.logical_end
                    || view.logical_end > view.physical_end
                    || view.physical_end > shard.physical_end
            })
    {
        return Err(StorageError::InvalidView);
    }
    Ok(())
}

fn physical_coordinates(
    shard: &LowShard,
    widths: &[u32],
    physical: u64,
) -> StorageResult<Vec<u32>> {
    let rank = widths.len();
    let mut coordinates = vec![0; rank];
    match shard.tensor_type.format.layout.order {
        ElementOrder::RowMajor => decode_row_major(widths, physical, &mut coordinates),
        ElementOrder::Amp(role) => {
            if rank < 2 {
                return Err(StorageError::AmpRank);
            }
            let rows = widths[rank - 2];
            let columns = widths[rank - 1];
            let matrix_elements = u64::from(rows) * u64::from(columns);
            let outer = physical / matrix_elements;
            decode_row_major(&widths[..rank - 2], outer, &mut coordinates[..rank - 2]);
            let linear =
                u32::try_from(physical % matrix_elements).map_err(|_| StorageError::Overflow)?;
            let (row, column) = amp_matrix_coordinates(
                role,
                shard.tensor_type.format.precision,
                rows,
                columns,
                linear,
            )?;
            coordinates[rank - 2] = row;
            coordinates[rank - 1] = column;
        }
    }
    Ok(coordinates)
}

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
        AmpOrder::Right => {
            let inner = amp_micro_dimension(precision);
            if !rows.is_multiple_of(inner) || !columns.is_multiple_of(COLUMN_MICRO) {
                return Err(StorageError::AmpBlock { role });
            }
            let panel_elements = inner * COLUMN_MICRO;
            let panel = linear / panel_elements;
            let offset = linear % panel_elements;
            let inner_groups = rows / inner;
            let column_group = panel / inner_groups;
            let inner_group = panel % inner_groups;
            let load_channel = offset / inner;
            let load_pair = load_channel / 2;
            let logical_pair = (load_pair % 2) * 4 + load_pair / 2;
            Ok((
                inner_group * inner + offset % inner,
                column_group * COLUMN_MICRO + logical_pair * 2 + load_channel % 2,
            ))
        }
        AmpOrder::RightK64 => {
            const INNER_BLOCK: u32 = AMP_INNER_BLOCK;
            let inner = amp_micro_dimension(precision);
            if !rows.is_multiple_of(INNER_BLOCK)
                || !columns.is_multiple_of(COLUMN_MICRO)
                || !INNER_BLOCK.is_multiple_of(inner)
            {
                return Err(StorageError::AmpBlock { role });
            }
            let panel_elements = inner * COLUMN_MICRO;
            let panel = linear / panel_elements;
            let offset = linear % panel_elements;
            let inner_groups_per_block = INNER_BLOCK / inner;
            let column_groups = columns / COLUMN_MICRO;
            let panels_per_inner_block = column_groups * inner_groups_per_block;
            let inner_block = panel / panels_per_inner_block;
            let within_block = panel % panels_per_inner_block;
            let column_group = within_block / inner_groups_per_block;
            let inner_group = within_block % inner_groups_per_block;
            let load_channel = offset / inner;
            let load_pair = load_channel / 2;
            let logical_pair = (load_pair % 2) * 4 + load_pair / 2;
            Ok((
                inner_block * INNER_BLOCK + inner_group * inner + offset % inner,
                column_group * COLUMN_MICRO + logical_pair * 2 + load_channel % 2,
            ))
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
    }
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
    use crate::low::{LowShardId, ShardDefinition, ShardExtent};
    use crate::mid::{Layout, Precision, TensorType};

    fn shard(layout: Layout, dimensions: &[u32]) -> LowShard {
        LowShard {
            id: LowShardId::from_index(0),
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
    fn randomized_amp_mappings_are_bijections_and_views_preserve_element_count() {
        let mut state = 0x9e37_79b9u32;
        for _ in 0..128 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let rows = 1 + (state % 19);
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let batches = 1 + (state % 4);
            for (layout, physical_rows, columns) in [
                (Layout::amp_left(64, 1), rows, 64),
                (Layout::amp_right(64, 1), 32, 64),
                (Layout::amp_right_k64_interleaved_grid(1, 1, 1), 64, 64),
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
    fn randomized_k64_right_panels_are_single_contiguous_spans() {
        let mut random = fastrand::Rng::with_seed(0x6b36_3472);
        for _ in 0..128 {
            let inner_blocks = random.u32(1..=8);
            let column_blocks = random.u32(1..=4);
            let rows = inner_blocks * 64;
            let columns = column_blocks * 64;
            let batches = random.u32(1..=4);
            let shard = shard(
                Layout::amp_right_k64_interleaved_grid(1, 1, 1),
                &[batches, rows, columns],
            );
            let inner = random.u32(0..inner_blocks) * 64;
            let column = random.u32(0..column_blocks) * 64;
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
                        logical_end: column + 64,
                        physical_end: column + 64,
                    },
                ],
            };
            let spans = view_byte_spans(&shard, &view).unwrap();
            assert_eq!(
                spans.iter().map(|span| span.bytes).sum::<u32>(),
                batches * 64 * 64 * 2
            );
            assert_eq!(
                spans[0].offset,
                ((inner / 64) * column_blocks + column / 64) * 64 * 64 * 2
            );
        }
    }
}
