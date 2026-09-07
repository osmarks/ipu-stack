//! Materialize coordinate mappings through shared local-copy and exchange machinery.

use super::*;

impl TileGraphBuilder {
    pub(super) fn copy_tensor(
        &mut self,
        operation: &MidOperation,
        mapping: &crate::CoordinateMapping,
        reuse_local: bool,
        body: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let mut batch = conversion::MaterializationBatch::default();
        self.prepare_copy_tensor(operation, mapping, reuse_local, &mut batch, body)?;
        self.append_materialization(batch, operation_provenance(operation), body)
    }

    pub(super) fn prepare_copy_tensor(
        &mut self,
        operation: &MidOperation,
        mapping: &crate::CoordinateMapping,
        reuse_local: bool,
        batch: &mut conversion::MaterializationBatch,
        body: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let ([input], [output]) = (operation.inputs.as_slice(), operation.results.as_slice())
        else {
            return Err(ExpansionError::ResultArity);
        };
        let inputs = self.value_shards(*input)?.to_vec();
        let outputs = self.value_shards(*output)?.to_vec();
        let source_order = self.shards[inputs
            .first()
            .ok_or(ExpansionError::InvalidOperatorPlan)?
            .index() as usize]
            .tensor_type
            .format
            .layout
            .order;
        let output_order = self.shards[outputs
            .first()
            .ok_or(ExpansionError::InvalidOperatorPlan)?
            .index() as usize]
            .tensor_type
            .format
            .layout
            .order;
        let (mappings, order) = if let Some(view) = mapping.view {
            let mappings = self.window_view_mappings(&inputs, &outputs, view, &mapping.offsets)?;
            if let Some(physical) = self.micro_panel_mappings(mappings.clone())? {
                (physical, CopyOrder::Physical)
            } else if matches!(
                source_order,
                ElementOrder::Amp(AmpOrder::Output | AmpOrder::TransposedLeft)
                    | ElementOrder::BlockMajor(_)
            ) {
                let mut provenance = operation_provenance(operation);
                provenance.reason = WorkReason::OperatorKernel;
                let unpacked = self
                    .unpack_amp_to_row_major(*input, provenance, body)?
                    .ok_or(ExpansionError::InvalidOperatorPlan)?;
                (
                    self.window_view_mappings(&unpacked, &outputs, view, &mapping.offsets)?,
                    CopyOrder::Semantic,
                )
            } else {
                (mappings, CopyOrder::Semantic)
            }
        } else {
            let mappings = self.offset_copy_mappings(
                operation,
                &inputs,
                &outputs,
                &mapping.offsets,
                reuse_local,
                body,
            )?;
            (
                mappings,
                if source_order == output_order {
                    CopyOrder::Physical
                } else {
                    CopyOrder::Semantic
                },
            )
        };
        self.prepare_mapped_views(
            mappings,
            order,
            order,
            operation_provenance(operation),
            batch,
            body,
        )
    }

    fn offset_copy_mappings(
        &mut self,
        operation: &MidOperation,
        inputs: &[BlockValueId],
        outputs: &[BlockValueId],
        offsets: &[u32],
        reuse_local: bool,
        body: &mut BlockRegion,
    ) -> ExpansionResult<Vec<(ShardView, ShardView)>> {
        let mut mappings = Vec::new();
        for &output in outputs {
            let destination = &self.shards[output.index() as usize];
            let tile = destination.tile;
            let mut source_region = destination.extents.clone();
            for (axis, extent) in source_region.iter_mut().enumerate() {
                let offset = offsets.get(axis).copied().unwrap_or(0);
                extent.start = extent
                    .start
                    .checked_add(offset)
                    .ok_or(ExpansionError::IdOverflow)?;
                extent.logical_end = extent
                    .logical_end
                    .checked_add(offset)
                    .ok_or(ExpansionError::IdOverflow)?;
                extent.physical_end = extent
                    .physical_end
                    .checked_add(offset)
                    .ok_or(ExpansionError::IdOverflow)?;
            }
            let intersections = self
                .intersecting_shard_set(inputs, &source_region, tile)
                .into_iter()
                .map(|(_, source)| {
                    (
                        intersect_extents_with_shared_padding(
                            &self.shards[source.index() as usize].extents,
                            &source_region,
                        )
                        .expect("selected intersection remains nonempty"),
                        source,
                    )
                })
                .collect::<Vec<_>>();
            if reuse_local
                && offsets.iter().all(|&offset| offset == 0)
                && let [(extents, source)] = intersections.as_slice()
                && *extents == self.shards[output.index() as usize].extents
                && self.shards[source.index() as usize].tile == tile
                && self.shards[source.index() as usize]
                    .tensor_type
                    .format
                    .precision
                    == self.shards[output.index() as usize]
                        .tensor_type
                        .format
                        .precision
                && self.shards[source.index() as usize]
                    .tensor_type
                    .format
                    .layout
                    .order
                    == self.shards[output.index() as usize]
                        .tensor_type
                        .format
                        .layout
                        .order
            {
                let mut view = self.full_view(*source);
                view.extents = extents.clone();
                self.materialized_views.insert(output, view);
                self.shards[output.index() as usize].definition = ShardDefinition::Unmaterialized;
                continue;
            }
            if intersections.is_empty() {
                self.append_fill_zero(body, output, operation_provenance(operation))?;
            }
            for (source_extents, source) in intersections {
                let mut destination_extents = source_extents.clone();
                for (axis, extent) in destination_extents.iter_mut().enumerate() {
                    let offset = offsets.get(axis).copied().unwrap_or(0);
                    extent.start -= offset;
                    extent.logical_end -= offset;
                    extent.physical_end -= offset;
                }
                mappings.push((
                    ShardView {
                        shard: self.full_view(source).shard,
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

    pub(super) fn window_view_mappings(
        &self,
        source_shards: &[BlockValueId],
        output_shards: &[BlockValueId],
        view: AxisFactorView,
        offsets: &[u32],
    ) -> ExpansionResult<Vec<(ShardView, ShardView)>> {
        let mut mappings = Vec::new();
        for &output in output_shards {
            let mut output_extents = self.shards[output.index() as usize].extents.clone();
            for (axis, extent) in output_extents.iter_mut().enumerate() {
                let offset = offsets.get(axis).copied().unwrap_or(0);
                extent.start = extent
                    .start
                    .checked_add(offset)
                    .ok_or(ExpansionError::IdOverflow)?;
                extent.logical_end = extent
                    .logical_end
                    .checked_add(offset)
                    .ok_or(ExpansionError::IdOverflow)?;
                extent.physical_end = extent
                    .physical_end
                    .checked_add(offset)
                    .ok_or(ExpansionError::IdOverflow)?;
            }
            let tile = self.shards[output.index() as usize].tile;
            let source_shape = &self.shards[source_shards[0].index() as usize]
                .tensor_type
                .shape;
            let output_shape = view
                .output_shape(source_shape)
                .ok_or(ExpansionError::InvalidOperatorPlan)?;
            for (extent, &size) in output_extents.iter_mut().zip(&output_shape.0) {
                extent.logical_end = extent.logical_end.min(size);
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
                        let target = view
                            .map_slice(source_shape, &output_shape, &ranges)
                            .ok_or(ExpansionError::InvalidOperatorPlan)?
                            .into_iter()
                            .enumerate()
                            .map(|(axis, (start, end))| ShardExtent {
                                axis: axis as u16,
                                start,
                                logical_end: end,
                                physical_end: end,
                            })
                            .collect::<Vec<_>>();
                        for (source_extents, source) in
                            self.intersecting_shard_set(source_shards, &target, tile)
                        {
                            let mut destination_extents = source_extents.clone();
                            destination_extents[merge].start = stream;
                            destination_extents[merge].logical_end = stream + 1;
                            destination_extents[merge].physical_end = stream + 1;
                            destination_extents[split].start += base;
                            destination_extents[split].logical_end += base;
                            destination_extents[split].physical_end += base;
                            for (axis, extent) in destination_extents.iter_mut().enumerate() {
                                let offset = offsets.get(axis).copied().unwrap_or(0);
                                extent.start -= offset;
                                extent.logical_end -= offset;
                                extent.physical_end -= offset;
                            }
                            mappings.push((
                                ShardView {
                                    shard: self.full_view(source).shard,
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
                let (target, column_base) = view
                    .source_extents(source_shape, &output_shape, &stream_extents)
                    .ok_or(ExpansionError::InvalidOperatorPlan)?;
                for (mut source_extents, source) in
                    self.intersecting_shard_set(source_shards, &target, tile)
                {
                    let mut destination_extents = source_extents.clone();
                    destination_extents[merge] = stream_extents[merge];
                    destination_extents[split].start -= column_base;
                    destination_extents[split].logical_end -= column_base;
                    destination_extents[split].physical_end -= column_base;
                    let source_shard = &self.shards[source.index() as usize];
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
                    for (axis, extent) in destination_extents.iter_mut().enumerate() {
                        let offset = offsets.get(axis).copied().unwrap_or(0);
                        extent.start -= offset;
                        extent.logical_end -= offset;
                        extent.physical_end -= offset;
                    }
                    let source_view = ShardView {
                        shard: self.full_view(source).shard,
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

    /// Splits corresponding views at each allocation's 16-element micro-panel
    /// boundaries. Within every resulting rectangle the source and
    /// destination have identical physical traversal, even when their outer
    /// panel sequence and tile ownership differ.
    pub(super) fn micro_panel_mappings(
        &self,
        mappings: Vec<(ShardView, ShardView)>,
    ) -> ExpansionResult<Option<Vec<(ShardView, ShardView)>>> {
        let mut split = Vec::new();
        for (source, destination) in mappings {
            let source_shard = &self.shards[source.shard.index() as usize];
            let destination_shard = &self.shards[destination.shard.index() as usize];
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
                let source_spans = view_byte_spans(source_shard, &source)?;
                let destination_spans = view_byte_spans(destination_shard, &destination)?;
                let valid_spans = source_spans
                    .iter()
                    .chain(&destination_spans)
                    .all(|span| span.offset & 0b11 == 0 && span.bytes & 0b11 == 0);
                let source_bytes = source_spans.iter().map(|span| span.bytes).sum::<u32>();
                let destination_bytes =
                    destination_spans.iter().map(|span| span.bytes).sum::<u32>();
                if !valid_spans || source_bytes != destination_bytes {
                    return Ok(None);
                }
                split.push((source, destination));
            }
        }
        Ok(Some(split))
    }
}
