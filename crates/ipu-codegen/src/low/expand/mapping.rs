//! Intersections and physical micro-panel geometry for block views.

use super::*;

pub(super) fn intersect_extents(
    left: &[ShardExtent],
    right: &[ShardExtent],
) -> Option<Vec<ShardExtent>> {
    if left.len() != right.len() {
        return None;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let start = left.start.max(right.start);
            let end = left.logical_end.min(right.logical_end);
            (start < end).then_some(ShardExtent {
                axis: left.axis,
                start,
                logical_end: end,
                physical_end: end,
            })
        })
        .collect()
}

pub(super) fn intersect_extents_with_shared_padding(
    left: &[ShardExtent],
    right: &[ShardExtent],
) -> Option<Vec<ShardExtent>> {
    if left.len() != right.len() {
        return None;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let start = left.start.max(right.start);
            let logical_end = left.logical_end.min(right.logical_end);
            (start < logical_end).then(|| {
                let shared_tail =
                    if logical_end == left.logical_end && logical_end == right.logical_end {
                        left.physical_end
                            .saturating_sub(left.logical_end)
                            .min(right.physical_end.saturating_sub(right.logical_end))
                    } else {
                        0
                    };
                ShardExtent {
                    axis: left.axis,
                    start,
                    logical_end,
                    physical_end: logical_end + shared_tail,
                }
            })
        })
        .collect()
}

pub(super) fn split_mapping_at_panel_boundaries(
    source_shard: &BlockValue,
    mut source: ShardView,
    destination_shard: &BlockValue,
    mut destination: ShardView,
) -> ExpansionResult<Vec<(ShardView, ShardView)>> {
    let source_rank = source.extents.len();
    let destination_rank = destination.extents.len();
    let outer_elements = |extents: &[ShardExtent]| {
        extents[..extents.len().saturating_sub(2)]
            .iter()
            .try_fold(1_u32, |elements, extent| {
                elements.checked_mul(extent.logical_end - extent.start)
            })
    };
    if source_rank < 2
        || destination_rank < 2
        || source_shard.extents.len() != source_rank
        || destination_shard.extents.len() != destination_rank
        || outer_elements(&source.extents) != Some(1)
        || outer_elements(&destination.extents) != Some(1)
    {
        return Err(ExpansionError::InvalidOperatorPlan);
    }

    let aligned_ranges = |source: ShardExtent,
                          source_shard: ShardExtent,
                          destination: ShardExtent,
                          destination_shard: ShardExtent|
     -> ExpansionResult<Vec<(ShardExtent, ShardExtent)>> {
        let logical_width = source.logical_end - source.start;
        if logical_width != destination.logical_end - destination.start {
            return Err(ExpansionError::InvalidOperatorPlan);
        }
        let width = source.physical_end - source.start;
        if width != destination.physical_end - destination.start {
            return Err(ExpansionError::InvalidOperatorPlan);
        }
        let mut ranges = Vec::new();
        let mut offset = 0;
        while offset < width {
            let source_position = source
                .start
                .checked_sub(source_shard.start)
                .and_then(|start| start.checked_add(offset))
                .ok_or(ExpansionError::InvalidOperatorPlan)?;
            let destination_position = destination
                .start
                .checked_sub(destination_shard.start)
                .and_then(|start| start.checked_add(offset))
                .ok_or(ExpansionError::InvalidOperatorPlan)?;
            let source_remaining = AMP_COLUMN_MICRO - source_position % AMP_COLUMN_MICRO;
            let destination_remaining = AMP_COLUMN_MICRO - destination_position % AMP_COLUMN_MICRO;
            let length = (width - offset)
                .min(source_remaining)
                .min(destination_remaining);
            let source_start = source.start + offset;
            let destination_start = destination.start + offset;
            ranges.push((
                ShardExtent {
                    axis: source.axis,
                    start: source_start,
                    logical_end: source
                        .logical_end
                        .min(source_start + length)
                        .max(source_start),
                    physical_end: source_start + length,
                },
                ShardExtent {
                    axis: destination.axis,
                    start: destination_start,
                    logical_end: destination
                        .logical_end
                        .min(destination_start + length)
                        .max(destination_start),
                    physical_end: destination_start + length,
                },
            ));
            offset += length;
        }
        Ok(ranges)
    };

    let source_row_axis = source_rank - 2;
    let source_column_axis = source_rank - 1;
    let destination_row_axis = destination_rank - 2;
    let destination_column_axis = destination_rank - 1;

    // The global row tail can finish part-way through a micro-panel while
    // both allocations contain padding through the same panel boundary.
    // Carry that padding with the useful values so the direct physical
    // exchange remains word-aligned. A split head's column tail is not
    // extended because the following source columns may belong to another
    // head rather than padding.
    let source_rows = source.extents[source_row_axis];
    let destination_rows = destination.extents[destination_row_axis];
    if source_rows.logical_end == source_shard.tensor_type.shape.0[source_row_axis]
        && destination_rows.logical_end
            == destination_shard.tensor_type.shape.0[destination_row_axis]
    {
        let source_panel_tail = (AMP_COLUMN_MICRO
            - (source_rows.logical_end - source_shard.extents[source_row_axis].start)
                % AMP_COLUMN_MICRO)
            % AMP_COLUMN_MICRO;
        let destination_panel_tail = (AMP_COLUMN_MICRO
            - (destination_rows.logical_end
                - destination_shard.extents[destination_row_axis].start)
                % AMP_COLUMN_MICRO)
            % AMP_COLUMN_MICRO;
        let padding = source_panel_tail
            .min(destination_panel_tail)
            .min(source_shard.extents[source_row_axis].physical_end - source_rows.logical_end)
            .min(
                destination_shard.extents[destination_row_axis].physical_end
                    - destination_rows.logical_end,
            );
        source.extents[source_row_axis].physical_end = source_rows
            .physical_end
            .max(source_rows.logical_end + padding);
        destination.extents[destination_row_axis].physical_end = destination_rows
            .physical_end
            .max(destination_rows.logical_end + padding);
    }

    let rows = aligned_ranges(
        source.extents[source_row_axis],
        source_shard.extents[source_row_axis],
        destination.extents[destination_row_axis],
        destination_shard.extents[destination_row_axis],
    )?;
    let columns = aligned_ranges(
        source.extents[source_column_axis],
        source_shard.extents[source_column_axis],
        destination.extents[destination_column_axis],
        destination_shard.extents[destination_column_axis],
    )?;
    let mut pieces = Vec::with_capacity(rows.len().saturating_mul(columns.len()));
    for (source_row, destination_row) in rows {
        for &(source_column, destination_column) in &columns {
            let mut source_extents = source.extents.clone();
            let mut destination_extents = destination.extents.clone();
            source_extents[source_row_axis] = source_row;
            source_extents[source_column_axis] = source_column;
            destination_extents[destination_row_axis] = destination_row;
            destination_extents[destination_column_axis] = destination_column;
            pieces.push((
                ShardView {
                    shard: source.shard,
                    extents: source_extents,
                },
                ShardView {
                    shard: destination.shard,
                    extents: destination_extents,
                },
            ));
        }
    }
    Ok(pieces)
}
