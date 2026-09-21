//! Intersections and physical micro-panel geometry for block views.

use super::*;
use crate::TensorShape;

/// Adapt logical view geometry to low-level shard regions.
pub(in crate::low::expand) fn view_source_extents(
    view: AxisFactorView,
    source: &TensorShape,
    output: &TensorShape,
    ranges: &[(u32, u32)],
) -> Option<Vec<ShardExtent>> {
    view.map_slice(source, output, ranges)?
        .into_iter()
        .enumerate()
        .map(|(axis, (start, end))| {
            Some(ShardExtent {
                axis: u16::try_from(axis).ok()?,
                start,
                logical_end: end,
                physical_end: end,
            })
        })
        .collect()
}

pub(in crate::low::expand) fn intersect_extents(
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

pub(in crate::low::expand) fn intersect_extents_with_shared_padding(
    left: &[ShardExtent],
    right: &[ShardExtent],
) -> Option<Vec<ShardExtent>> {
    let mut extents = intersect_extents(left, right)?;
    for ((extent, left), right) in extents.iter_mut().zip(left).zip(right) {
        if extent.logical_end == left.logical_end && extent.logical_end == right.logical_end {
            extent.physical_end += left
                .physical_end
                .saturating_sub(left.logical_end)
                .min(right.physical_end.saturating_sub(right.logical_end));
        }
    }
    Some(extents)
}

pub(in crate::low::expand) fn split_mapping_at_panel_boundaries(
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
        || outer_elements(&source.extents).is_none()
        || outer_elements(&source.extents) != outer_elements(&destination.extents)
    {
        return Err(ExpansionError::InvalidOperatorPlan);
    }

    let aligned_ranges = |source: ShardExtent,
                          source_shard: ShardExtent,
                          destination: ShardExtent,
                          destination_shard: ShardExtent,
                          source_panel: u32,
                          destination_panel: u32|
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
            let source_remaining = source_panel - source_position % source_panel;
            let destination_remaining =
                destination_panel - destination_position % destination_panel;
            let length = (width - offset)
                .min(source_remaining)
                .min(destination_remaining);
            let piece = |extent: ShardExtent| {
                let start = extent.start + offset;
                ShardExtent {
                    axis: extent.axis,
                    start,
                    logical_end: extent.logical_end.min(start + length).max(start),
                    physical_end: start + length,
                }
            };
            ranges.push((piece(source), piece(destination)));
            offset += length;
        }
        Ok(ranges)
    };

    let source_row_axis = source_rank - 2;
    let source_column_axis = source_rank - 1;
    let destination_row_axis = destination_rank - 2;
    let destination_column_axis = destination_rank - 1;

    extend_panel_row_padding(
        source_shard,
        &mut source,
        destination_shard,
        &mut destination,
    );

    let source_panel = source_shard
        .tensor_type
        .format
        .exchange_panel_shape()
        .ok_or(ExpansionError::InvalidOperatorPlan)?;
    let destination_panel = destination_shard
        .tensor_type
        .format
        .exchange_panel_shape()
        .ok_or(ExpansionError::InvalidOperatorPlan)?;

    let rows = aligned_ranges(
        source.extents[source_row_axis],
        source_shard.extents[source_row_axis],
        destination.extents[destination_row_axis],
        destination_shard.extents[destination_row_axis],
        source_panel[0],
        destination_panel[0],
    )?;
    let columns = aligned_ranges(
        source.extents[source_column_axis],
        source_shard.extents[source_column_axis],
        destination.extents[destination_column_axis],
        destination_shard.extents[destination_column_axis],
        source_panel[1],
        destination_panel[1],
    )?;
    let mut pieces = Vec::with_capacity(rows.len().saturating_mul(columns.len()));
    for batch in 0..outer_elements(&source.extents).unwrap() {
        for &(source_row, destination_row) in &rows {
            for &(source_column, destination_column) in &columns {
                let mut source_extents = source.extents.clone();
                let mut destination_extents = destination.extents.clone();
                for extents in [&mut source_extents, &mut destination_extents] {
                    let rank = extents.len();
                    let mut index = batch;
                    for extent in extents[..rank - 2].iter_mut().rev() {
                        let width = extent.logical_end - extent.start;
                        extent.start += index % width;
                        index /= width;
                        extent.logical_end = extent.start + 1;
                        extent.physical_end = extent.logical_end;
                    }
                }
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
    }
    Ok(pieces)
}

/// Preserve known padding when a mapped row reaches both tensors' global tails.
pub(in crate::low::expand) fn extend_panel_row_padding(
    source_shard: &BlockValue,
    source: &mut ShardView,
    destination_shard: &BlockValue,
    destination: &mut ShardView,
) {
    if source.extents.len() < 2 || destination.extents.len() < 2 {
        return;
    }
    let (Some(source_panel), Some(destination_panel)) = (
        source_shard.tensor_type.format.exchange_panel_shape(),
        destination_shard.tensor_type.format.exchange_panel_shape(),
    ) else {
        return;
    };
    let source_row_axis = source.extents.len() - 2;
    let destination_row_axis = destination.extents.len() - 2;
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
        let source_panel_tail = (source_panel[0]
            - (source_rows.logical_end - source_shard.extents[source_row_axis].start)
                % source_panel[0])
            % source_panel[0];
        let destination_panel_tail = (destination_panel[0]
            - (destination_rows.logical_end
                - destination_shard.extents[destination_row_axis].start)
                % destination_panel[0])
            % destination_panel[0];
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
}

impl TileGraphBuilder {
    pub(super) fn offset_copy_mappings(
        &mut self,
        inputs: &[ShardView],
        outputs: &[BlockValueId],
        offsets: &[u32],
    ) -> ExpansionResult<Vec<(ShardView, ShardView)>> {
        let mut regions = CopyRegions::new(&self.program.shards, inputs);
        let mut mappings = Vec::new();
        for &output in outputs {
            let destination = &self.program.shards[output.index() as usize];
            let tile = destination.tile;
            let mut source_region = destination.extents.clone();
            offset_extents(&mut source_region, offsets, u32::checked_add)?;
            let intersections = regions
                .intersections(&source_region, tile)
                .into_iter()
                .map(|(_, source)| {
                    (
                        intersect_extents_with_shared_padding(
                            &inputs[source].extents,
                            &source_region,
                        )
                        .expect("selected intersection remains nonempty"),
                        source,
                    )
                })
                .collect::<Vec<_>>();
            for (source_extents, source) in intersections {
                let mut destination_extents = source_extents.clone();
                offset_extents(&mut destination_extents, offsets, u32::checked_sub)?;
                mappings.push((
                    ShardView {
                        shard: inputs[source].shard,
                        extents: source_extents,
                    },
                    ShardView {
                        shard: output,
                        extents: destination_extents,
                    },
                ));
            }
        }
        Ok(mappings)
    }

    pub(in crate::low::expand) fn window_view_mappings(
        &self,
        sources: &[ShardView],
        source_shape: &crate::TensorShape,
        output_shards: &[BlockValueId],
        view: AxisFactorView,
        offsets: &[u32],
    ) -> ExpansionResult<Vec<(ShardView, ShardView)>> {
        let mut regions = CopyRegions::new(&self.program.shards, sources);
        let mut mappings = Vec::new();
        for &output in output_shards {
            let mut output_extents = self.program.shards[output.index() as usize].extents.clone();
            offset_extents(&mut output_extents, offsets, u32::checked_add)?;
            let tile = self.program.shards[output.index() as usize].tile;
            let output_shape = view
                .output_shape(source_shape)
                .ok_or(ExpansionError::InvalidOperatorPlan)?;
            for (extent, &size) in output_extents.iter_mut().zip(&output_shape.0) {
                extent.logical_end = extent.logical_end.min(size);
            }
            if output_extents
                .iter()
                .any(|extent| extent.start >= extent.logical_end)
            {
                continue;
            }
            let split = view.split_axis;
            let merge = view.merge_axis;
            if view.reversed {
                // A joined head is contiguous. Divide only at head boundaries,
                // not at individual output columns as in the forward move.
                let width = source_shape.0[split];
                let split_extent = output_extents[split];
                for stream in output_extents[merge].start..output_extents[merge].logical_end {
                    let mut start = split_extent.start;
                    while start < split_extent.logical_end {
                        let base = start / width * width;
                        let end = (base + width).min(split_extent.logical_end);
                        let mut ranges = output_extents
                            .iter()
                            .map(|extent| (extent.start, extent.logical_end))
                            .collect::<Vec<_>>();
                        ranges[merge] = (stream, stream + 1);
                        ranges[split] = (start, end);
                        let target =
                            view_source_extents(view, source_shape, &output_shape, &ranges)
                                .ok_or(ExpansionError::InvalidOperatorPlan)?;
                        for (source_extents, source) in regions.intersections(&target, tile) {
                            let mut destination_extents = source_extents.clone();
                            destination_extents[merge].start = stream;
                            destination_extents[merge].logical_end = stream + 1;
                            destination_extents[merge].physical_end = stream + 1;
                            destination_extents[split].start += base;
                            destination_extents[split].logical_end += base;
                            destination_extents[split].physical_end += base;
                            offset_extents(&mut destination_extents, offsets, u32::checked_sub)?;
                            mappings.push((
                                ShardView {
                                    shard: sources[source].shard,
                                    extents: source_extents,
                                },
                                ShardView {
                                    shard: output,
                                    extents: destination_extents,
                                },
                            ));
                        }
                        start = end;
                    }
                }
                continue;
            }
            let part_width = output_shape.0[split];
            for stream in output_extents[merge].start..output_extents[merge].logical_end {
                let mut stream_extents = output_extents.clone();
                stream_extents[merge].start = stream;
                stream_extents[merge].logical_end = stream + 1;
                stream_extents[merge].physical_end = stream + 1;
                let ranges = stream_extents
                    .iter()
                    .map(|extent| (extent.start, extent.logical_end))
                    .collect::<Vec<_>>();
                let target = view_source_extents(view, source_shape, &output_shape, &ranges)
                    .ok_or(ExpansionError::InvalidOperatorPlan)?;
                let column_base = target[split]
                    .start
                    .checked_sub(stream_extents[split].start)
                    .ok_or(ExpansionError::InvalidOperatorPlan)?;
                for (mut source_extents, source) in regions.intersections(&target, tile) {
                    let mut destination_extents = source_extents.clone();
                    destination_extents[merge] = stream_extents[merge];
                    destination_extents[split].start -= column_base;
                    destination_extents[split].logical_end -= column_base;
                    destination_extents[split].physical_end -= column_base;
                    let source_shard = &sources[source];
                    let complete_part = source_extents[split].start == column_base
                        && source_extents[split].logical_end == column_base + part_width
                        && source_shard.extents[split].start == column_base
                        && source_shard.extents[split].logical_end == column_base + part_width;
                    if complete_part {
                        let source_padding = source_shard.extents[split]
                            .physical_end
                            .saturating_sub(source_extents[split].logical_end);
                        let destination_padding = output_extents[split]
                            .physical_end
                            .saturating_sub(output_extents[split].logical_end);
                        let padding = source_padding.min(destination_padding);
                        source_extents[split].physical_end += padding;
                        destination_extents[split].physical_end += padding;
                    }
                    offset_extents(&mut destination_extents, offsets, u32::checked_sub)?;
                    let source_view = ShardView {
                        shard: sources[source].shard,
                        extents: source_extents,
                    };
                    let destination_view = ShardView {
                        shard: output,
                        extents: destination_extents,
                    };
                    mappings.push((source_view, destination_view));
                }
            }
        }
        Ok(mappings)
    }

    /// Match packed grids using their precision-specific panel shape. Irregular
    /// boundaries use smaller clipped fragments with identical traversal, even
    /// when the outer panel sequence and tile ownership differ.
    pub(in crate::low::expand) fn micro_panel_mappings(
        &self,
        mut mappings: Vec<(ShardView, ShardView)>,
    ) -> ExpansionResult<Option<(Vec<(ShardView, ShardView)>, CopyOrder)>> {
        for (source, destination) in &mut mappings {
            extend_panel_row_padding(
                &self.program.shards[source.shard.index() as usize],
                source,
                &self.program.shards[destination.shard.index() as usize],
                destination,
            );
        }
        // Complete grids have one shared traversal specification. Irregular
        // boundaries retain the existing clipped-rectangle fallback below.
        let regular = !mappings.is_empty()
            && mappings.iter().all(|(source, destination)| {
                let a = &self.program.shards[source.shard.index() as usize];
                let b = &self.program.shards[destination.shard.index() as usize];
                let rank = source.extents.len();
                let other_rank = destination.extents.len();
                rank >= 2
                    && other_rank >= 2
                    && a.tensor_type
                        .format
                        .supports_micro_panel_exchange(&b.tensor_type.format)
                    && source.extents[rank - 2..]
                        .iter()
                        .zip(&destination.extents[other_rank - 2..])
                        .all(|(x, y)| {
                            x.physical_end - x.start == y.physical_end - y.start
                                && x.logical_end - x.start == y.logical_end - y.start
                        })
                    && [(a, source), (b, destination)]
                        .into_iter()
                        .all(|(shard, view)| {
                            crate::storage::panel_byte_traversal(shard.storage(), &view.extents)
                                .is_ok_and(|traversal| traversal.word_aligned())
                        })
            });
        if regular {
            return Ok(Some((mappings, CopyOrder::Panels)));
        }
        let mut split = Vec::new();
        for (source, destination) in mappings {
            let source_shard = &self.program.shards[source.shard.index() as usize];
            let destination_shard = &self.program.shards[destination.shard.index() as usize];
            if !source_shard
                .tensor_type
                .format
                .supports_micro_panel_exchange(&destination_shard.tensor_type.format)
            {
                return Ok(None);
            }
            let pieces = split_mapping_at_panel_boundaries(
                source_shard,
                source,
                destination_shard,
                destination,
            )?;
            for (source, destination) in pieces {
                let source_spans = source
                    .bind(&self.program.shards)?
                    .traversal(CopyOrder::Physical)?;
                let destination_spans = destination
                    .bind(&self.program.shards)?
                    .traversal(CopyOrder::Physical)?;
                if !source_spans.word_aligned()
                    || !destination_spans.word_aligned()
                    || source_spans.byte_len() != destination_spans.byte_len()
                {
                    return Ok(None);
                }
                split.push((source, destination));
            }
        }
        Ok(Some((split, CopyOrder::Physical)))
    }
}

/// Translate a window to or from the source coordinate system.
fn offset_extents(
    extents: &mut [ShardExtent],
    offsets: &[u32],
    shift: fn(u32, u32) -> Option<u32>,
) -> ExpansionResult<()> {
    for (extent, &offset) in extents.iter_mut().zip(offsets) {
        for coordinate in [
            &mut extent.start,
            &mut extent.logical_end,
            &mut extent.physical_end,
        ] {
            *coordinate = shift(*coordinate, offset).ok_or(ExpansionError::IdOverflow)?;
        }
    }
    Ok(())
}
