//! Lower a whole-device Copy through coordinate mappings, physical realization
//! and one batch of pre-exchange work, recipients and post-exchange work.
//! Geometry helpers describe coverage; explicit policies constrain realization.

use super::*;
use crate::tensor::{BlockMajorOrder, TensorFormat};

/// Work needed to populate ordinary destination blocks at one exchange boundary.
#[derive(Default)]
pub(super) struct MaterializationBatch {
    transfers: BTreeMap<CopyOrder, BTreeMap<ShardView, Vec<ShardView>>>,
    before: Vec<(u16, LocalCopy)>,
    pub(super) after: Vec<(u16, LocalCopy)>,
    kernels: Vec<(u16, KernelRun)>,
    loopback_candidates: Vec<(ShardView, ShardView, CopyOrder)>,
}

impl TileGraphBuilder {
    pub(super) fn prepare_copy_tensor(
        &mut self,
        operation: &MidOperation,
        batch: &mut MaterializationBatch,
        body: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let MidOperationKind::Copy {
            ref mapping,
            reuse_local,
            policy,
            packing,
        } = operation.kind
        else {
            return Err(ExpansionError::InvalidCopyPlan);
        };
        let ([input], [output]) = (operation.inputs.as_slice(), operation.results.as_slice())
        else {
            return Err(ExpansionError::ResultArity);
        };
        let inputs = self.value_views(*input)?.to_vec();
        let outputs = self.allocation_shards(*output)?;
        let source_shape = self.logical_values[input.index() as usize]
            .tensor_type
            .shape
            .clone();
        // Whole-buffer bindings require canonical storage with its own strides.
        let reuse_local = reuse_local
            && packing == PackingPolicy::Automatic
            && !self.required_storage.contains(output);
        let output_order = self.shards[outputs
            .first()
            .ok_or(ExpansionError::InvalidOperatorPlan)?
            .index() as usize]
            .tensor_type
            .format
            .layout
            .order;
        if policy == CopyPolicy::LocalKernel {
            if !mapping.is_identity() || packing != PackingPolicy::Automatic {
                return Err(ExpansionError::InvalidCopyPlan);
            }
            return self.local_rearrangement(operation, &inputs, &outputs, body);
        }
        let inputs = if policy == CopyPolicy::StageLogicalThenTransform
            && output_order == ElementOrder::RowMajor
        {
            self.unpack_amp_to_row_major(*input, operation_provenance(operation), body)?
                .unwrap_or(inputs)
        } else {
            inputs
        };
        let source_order = self.shards[inputs
            .first()
            .ok_or(ExpansionError::InvalidOperatorPlan)?
            .shard
            .index() as usize]
            .tensor_type
            .format
            .layout
            .order;
        let (mappings, order) = if let Some(view) = mapping.view {
            let mappings = self.window_view_mappings(
                &inputs,
                &source_shape,
                &outputs,
                view,
                &mapping.offsets,
            )?;
            if packing != PackingPolicy::Staged
                && let Some(physical) = self.micro_panel_mappings(mappings.clone())?
            {
                physical
            } else if policy == CopyPolicy::DirectRetile {
                return Err(ExpansionError::InvalidCopyPlan);
            } else if matches!(
                source_order,
                ElementOrder::Amp(AmpOrder::Output | AmpOrder::TransposedLeft)
                    | ElementOrder::BlockMajor(_)
            ) {
                let mut provenance = operation_provenance(operation);
                provenance.reason = WorkReason::LayoutRearrangement;
                let unpacked = self
                    .unpack_amp_to_row_major(*input, provenance, body)?
                    .ok_or(ExpansionError::InvalidOperatorPlan)?;
                (
                    self.window_view_mappings(
                        &unpacked,
                        &source_shape,
                        &outputs,
                        view,
                        &mapping.offsets,
                    )?,
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
            )?;
            (
                mappings,
                match policy {
                    CopyPolicy::DirectRetile => CopyOrder::Physical,
                    CopyPolicy::StageLogicalThenTransform => CopyOrder::Semantic,
                    CopyPolicy::Automatic if source_order == output_order => CopyOrder::Physical,
                    CopyPolicy::Automatic => CopyOrder::Semantic,
                    CopyPolicy::LocalKernel => unreachable!(),
                },
            )
        };
        let covered = mappings
            .iter()
            .map(|(_, destination)| destination.shard)
            .collect::<BTreeSet<_>>();
        for output in outputs {
            if !covered.contains(&output)
                && matches!(
                    self.shards[output.index() as usize].definition,
                    ShardDefinition::Value(_)
                )
            {
                self.append_fill_zero(body, output, operation_provenance(operation))?;
            }
        }
        self.prepare_mapped_views(
            mappings,
            order,
            order,
            packing,
            operation_provenance(operation),
            batch,
            body,
        )
    }

    pub(super) fn unpack_amp_to_row_major(
        &mut self,
        source: MidValueId,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<Option<Vec<ShardView>>> {
        let sources = self.value_views(source)?.to_vec();
        for source_view in &sources {
            let source = &self.shards[source_view.shard.index() as usize];
            let compatible = source_view.extents.len() >= 2
                && source.tensor_type.format.precision == Precision::F16
                && match source.tensor_type.format.layout.order {
                    ElementOrder::Amp(AmpOrder::Output) => {
                        let columns = source_view.extents[source_view.extents.len() - 1];
                        (columns.physical_end - columns.start).is_multiple_of(AMP_COLUMN_MICRO)
                    }
                    ElementOrder::Amp(AmpOrder::TransposedLeft) => {
                        let rows = source_view.extents[source_view.extents.len() - 2];
                        (rows.physical_end - rows.start).is_multiple_of(AMP_COLUMN_MICRO)
                    }
                    ElementOrder::BlockMajor(_) => true,
                    _ => false,
                };
            if !compatible {
                tracing::debug!(
                    shard = source_view.shard.index(),
                    rank = source_view.extents.len(),
                    precision = ?source.tensor_type.format.precision,
                    order = ?source.tensor_type.format.layout.order,
                    extents = ?source_view.extents,
                    "cannot unpack source storage into row-major order"
                );
                return Ok(None);
            }
        }

        let mut staging_views = Vec::with_capacity(sources.len());
        for source_view in sources {
            let block = &self.shards[source_view.shard.index() as usize];
            let tile = block.tile;
            let mut staging_type = self.logical_values[source.index() as usize]
                .tensor_type
                .clone();
            let to = Layout::row_major(TensorTiling::replicated(1));
            let from = std::mem::replace(&mut staging_type.format.layout, to.clone());
            let staging = self.push_shard(BlockValue {
                id: BlockValueId(0),
                tile,
                tensor_type: staging_type,
                extents: source_view.extents.clone(),
                definition: ShardDefinition::Staging,
            })?;
            let run = self.bind_kernel(
                provenance,
                TileKernelSpec::Rearrange { from, to },
                vec![source_view],
                vec![self.full_view(staging)],
            )?;
            self.append_kernel(tiles, tile, run)?;
            staging_views.push(self.full_view(staging));
        }
        Ok(Some(staging_views))
    }

    pub(super) fn local_rearrangement(
        &mut self,
        operation: &MidOperation,
        inputs: &[ShardView],
        outputs: &[BlockValueId],
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        if inputs.len() != outputs.len() {
            return Err(ExpansionError::InvalidCopyPlan);
        }
        let shards = inputs
            .iter()
            .cloned()
            .zip(outputs.iter().copied())
            .collect::<Vec<_>>();
        for (input, output) in shards {
            let source = &self.shards[input.shard.index() as usize];
            let destination = &self.shards[output.index() as usize];
            if source.tile != destination.tile
                || input.extents.len() != destination.extents.len()
                || input
                    .extents
                    .iter()
                    .zip(&destination.extents)
                    .any(|(a, b)| a.start != b.start || a.logical_end != b.logical_end)
            {
                return Err(ExpansionError::InvalidCopyPlan);
            }
            let tile = destination.tile;
            let run = self.bind_kernel(
                operation_provenance(operation),
                TileKernelSpec::Rearrange {
                    from: source.tensor_type.format.layout.clone(),
                    to: destination.tensor_type.format.layout.clone(),
                },
                vec![input],
                vec![self.full_view(output)],
            )?;
            self.append_kernel(tiles, tile, run)?;
        }
        Ok(())
    }

    fn offset_copy_mappings(
        &mut self,
        operation: &MidOperation,
        inputs: &[ShardView],
        outputs: &[BlockValueId],
        offsets: &[u32],
        reuse_local: bool,
    ) -> ExpansionResult<Vec<(ShardView, ShardView)>> {
        let mut regions = CopyRegions::new(&self.shards, inputs);
        let mut mappings = Vec::new();
        for (result_index, &output) in outputs.iter().enumerate() {
            let destination = &self.shards[output.index() as usize];
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
            if reuse_local
                && offsets.iter().all(|&offset| offset == 0)
                && let [(extents, source)] = intersections.as_slice()
                && *extents == self.shards[output.index() as usize].extents
                && self.shards[inputs[*source].shard.index() as usize].tile == tile
                && self.shards[inputs[*source].shard.index() as usize]
                    .tensor_type
                    .format
                    .precision
                    == self.shards[output.index() as usize]
                        .tensor_type
                        .format
                        .precision
                && self.shards[inputs[*source].shard.index() as usize]
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
                let mut view = inputs[*source].clone();
                view.extents = extents.clone();
                if !self.exported_values.contains(&operation.results[0]) {
                    self.bindings[operation.results[0].index() as usize][result_index] = view;
                    self.shards[output.index() as usize].definition =
                        ShardDefinition::Unmaterialized;
                    continue;
                }
                // Host bindings export the canonical shard's complete storage,
                // not a ShardView. Preserve its shape and address with an alias
                // only when the reused allocation has exactly these bounds.
                if view.extents == self.shards[view.shard.index() as usize].extents {
                    self.alias_shard(output, view.shard);
                    continue;
                }
            }
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

    pub(super) fn window_view_mappings(
        &self,
        sources: &[ShardView],
        source_shape: &crate::TensorShape,
        output_shards: &[BlockValueId],
        view: AxisFactorView,
        offsets: &[u32],
    ) -> ExpansionResult<Vec<(ShardView, ShardView)>> {
        let mut regions = CopyRegions::new(&self.shards, sources);
        let mut mappings = Vec::new();
        for &output in output_shards {
            let mut output_extents = self.shards[output.index() as usize].extents.clone();
            offset_extents(&mut output_extents, offsets, u32::checked_add)?;
            let tile = self.shards[output.index() as usize].tile;
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
    pub(super) fn micro_panel_mappings(
        &self,
        mut mappings: Vec<(ShardView, ShardView)>,
    ) -> ExpansionResult<Option<(Vec<(ShardView, ShardView)>, CopyOrder)>> {
        for (source, destination) in &mut mappings {
            extend_panel_row_padding(
                &self.shards[source.shard.index() as usize],
                source,
                &self.shards[destination.shard.index() as usize],
                destination,
            );
        }
        // Complete grids have one shared traversal specification. Irregular
        // boundaries retain the existing clipped-rectangle fallback below.
        let regular = !mappings.is_empty()
            && mappings.iter().all(|(source, destination)| {
                let a = &self.shards[source.shard.index() as usize];
                let b = &self.shards[destination.shard.index() as usize];
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
                let source_spans = source.bind(&self.shards)?.traversal(CopyOrder::Physical)?;
                let destination_spans = destination
                    .bind(&self.shards)?
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

impl TileGraphBuilder {
    pub(super) fn prepare_mapped_views(
        &mut self,
        mappings: Vec<(ShardView, ShardView)>,
        copy_order: CopyOrder,
        exchange_order: CopyOrder,
        packing: PackingPolicy,
        provenance: WorkProvenance,
        batch: &mut MaterializationBatch,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let mut grouped = BTreeMap::<BlockValueId, Vec<(ShardView, ShardView)>>::new();
        for mapping in mappings {
            grouped.entry(mapping.1.shard).or_default().push(mapping);
        }
        let mut packed_sources = BTreeMap::new();
        for (destination_shard, mappings) in grouped {
            // A clipped boundary on one destination must not expand complete
            // panel grids on every other destination into tiny rectangles.
            let physical = if copy_order == CopyOrder::Semantic && packing != PackingPolicy::Staged
            {
                self.micro_panel_mappings(mappings.clone())?
            } else {
                None
            };
            let (mappings, copy_order, exchange_order) = physical.map_or(
                (mappings, copy_order, exchange_order),
                |(mappings, order)| (mappings, order, order),
            );
            let transfers = batch.transfers.entry(exchange_order).or_default();
            let destination = &self.shards[destination_shard.index() as usize];
            let geometric_mappings = mappings
                .iter()
                .map(|(source, target)| {
                    let source = source.bind(&self.shards)?;
                    target.bind(&self.shards)?;
                    if target.shard != destination_shard {
                        return Err(ExpansionError::InvalidCopyPlan);
                    }
                    Ok(crate::storage::CopyMapping {
                        source: source.shard.storage(),
                        source_extents: source.extents,
                        destination_extents: &target.extents,
                    })
                })
                .collect::<ExpansionResult<Vec<_>>>()?;
            let geometry = self.cache.destination(
                destination.storage(),
                &geometric_mappings,
                copy_order,
                crate::exchange::MAX_TRANSFER_WORDS * 4,
            )?;
            let preparation = select_destination_packing(
                &destination.tensor_type,
                &destination.extents,
                &geometry,
                packing,
            )?;
            // A pack kernel writes the complete physical output, including padding.
            if preparation
                .as_ref()
                .is_none_or(|staging| staging.kernel.is_none())
            {
                self.append_copy_clears(
                    tiles,
                    destination_shard,
                    geometry.uncovered()?,
                    provenance,
                )?;
            }
            let staging = if let Some(staging) = &preparation {
                Some(self.push_shard(BlockValue {
                    id: BlockValueId(0),
                    tile: self.shards[destination_shard.index() as usize].tile,
                    tensor_type: staging.tensor_type.clone(),
                    extents: staging.extents.clone(),
                    definition: ShardDefinition::Staging,
                })?)
            } else {
                None
            };
            if let Some(staging) = staging {
                let block = &self.shards[staging.index() as usize];
                let ranges = crate::storage::uncovered_bytes(
                    block.storage(),
                    mappings
                        .iter()
                        .map(|(_, destination)| destination.extents.as_slice()),
                    CopyOrder::Semantic,
                )?;
                self.append_copy_clears(tiles, staging, &ranges, provenance)?;
            }
            for (mut source, mut destination) in mappings {
                if let Some(staging) = staging {
                    destination.shard = staging;
                    for extent in source.extents.iter_mut().chain(&mut destination.extents) {
                        extent.physical_end = extent.logical_end;
                    }
                }
                let source_tile = self.shards[source.shard.index() as usize].tile;
                let destination_tile = self.shards[destination.shard.index() as usize].tile;
                if source_tile == destination_tile {
                    if staging.is_none()
                        && copy_order == exchange_order
                        && self.shards[source.shard.index() as usize]
                            .tensor_type
                            .format
                            .layout
                            .memory_class
                            == MemoryClass::Ipu21Standard
                    {
                        batch
                            .loopback_candidates
                            .push((source, destination, copy_order));
                        continue;
                    }
                    let copies = if staging.is_some() {
                        &mut batch.before
                    } else {
                        &mut batch.after
                    };
                    append_span_copies(
                        &self.cache,
                        &self.shards,
                        &source,
                        &destination,
                        destination_tile,
                        copies,
                        copy_order,
                    )?;
                } else {
                    if exchange_order != CopyOrder::Panels
                        && !source
                            .bind(&self.shards)?
                            .traversal(exchange_order)?
                            .word_aligned()
                    {
                        source = self.pack_exchange_source(
                            source,
                            exchange_order,
                            &mut packed_sources,
                            &mut batch.before,
                        )?;
                    }
                    transfers.entry(source).or_default().push(destination);
                }
            }
            if let Some(staging) = staging {
                let staging = self.full_view(staging);
                let tile = self.shards[destination_shard.index() as usize].tile;
                if let Some(kernel) = preparation
                    .as_ref()
                    .and_then(|staging| staging.kernel.as_ref())
                {
                    batch.kernels.push((
                        tile,
                        self.bind_kernel(
                            provenance,
                            kernel.clone(),
                            vec![staging],
                            vec![self.full_view(destination_shard)],
                        )?,
                    ));
                } else {
                    append_span_copies(
                        &self.cache,
                        &self.shards,
                        &staging,
                        &self.logical_view(destination_shard),
                        tile,
                        &mut batch.after,
                        CopyOrder::Semantic,
                    )?;
                }
            }
        }
        Ok(())
    }

    // Destination packing cannot repair halfword reads from a source panel.
    // Gather that logical slice locally before sending it as whole words.
    fn pack_exchange_source(
        &mut self,
        source: ShardView,
        order: CopyOrder,
        packed: &mut BTreeMap<(CopyOrder, ShardView), ShardView>,
        copies: &mut Vec<(u16, LocalCopy)>,
    ) -> ExpansionResult<ShardView> {
        let key = (order, source.clone());
        if let Some(view) = packed.get(&key) {
            return Ok(view.clone());
        }
        let shard = &self.shards[source.shard.index() as usize];
        let tile = shard.tile;
        let bytes = source.bind(&self.shards)?.traversal(order)?.byte_len();
        if bytes == 0 || !bytes.is_multiple_of(4) {
            return Err(ExpansionError::InvalidCopyPlan);
        }
        let precision = shard.tensor_type.format.precision;
        let staging = self.push_packed_buffer(
            tile,
            u32::try_from(bytes / precision.bytes()).map_err(|_| ExpansionError::IdOverflow)?,
            precision,
            ShardDefinition::Staging,
        )?;
        let view = self.full_view(staging);
        append_span_copies(
            &self.cache,
            &self.shards,
            &source,
            &view,
            tile,
            copies,
            order,
        )?;
        packed.insert(key, view.clone());
        Ok(view)
    }

    pub(super) fn append_materialization(
        &mut self,
        mut batch: MaterializationBatch,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        for (source, destination, order) in batch.loopback_candidates {
            let transfers = batch.transfers.entry(order).or_default();
            // Keep the existing multicast send; only add its local receiver.
            // Placement separates same-class source/receiver SRAM elements.
            let source_spans = source.bind(&self.shards)?.traversal(order)?;
            let destination_spans = destination.bind(&self.shards)?.traversal(order)?;
            let aligned = source_spans.word_aligned() && destination_spans.word_aligned();
            if let Some(destinations) = transfers.get_mut(&source)
                && destinations.len() >= 2
                && aligned
                // The local receiver must not split existing messages further.
                && destinations.iter().any(|view| {
                    view.bind(&self.shards).and_then(|bound| bound.traversal(order)).is_ok_and(|remote|
                        remote.spans().map(|span| span.bytes)
                            .eq(destination_spans.spans().map(|span| span.bytes)))
                })
                && destinations.iter().all(|view| {
                    self.shards[view.shard.index() as usize].tile
                        != self.shards[source.shard.index() as usize].tile
                })
            {
                destinations.push(destination);
            } else {
                append_span_copies(
                    &self.cache,
                    &self.shards,
                    &source,
                    &destination,
                    self.shards[source.shard.index() as usize].tile,
                    &mut batch.after,
                    order,
                )?;
            }
        }
        for (tile, copy) in batch.before {
            self.append_local_copy(
                tiles,
                tile,
                crate::kernel::CopyRun::bind(copy, &self.shards)?,
            )?;
        }
        self.append_mixed_phase(batch.transfers, provenance, tiles)?;
        for (tile, copy) in batch.after {
            self.append_local_copy(
                tiles,
                tile,
                crate::kernel::CopyRun::bind(copy, &self.shards)?,
            )?;
        }
        for (tile, run) in batch.kernels {
            self.append_kernel(tiles, tile, run)?;
        }
        Ok(())
    }

    pub(super) fn logical_view(&self, shard: BlockValueId) -> ShardView {
        let mut view = self.full_view(shard);
        for extent in &mut view.extents {
            extent.physical_end = extent.logical_end;
        }
        view
    }
}

struct CopyStaging {
    tensor_type: TensorType,
    extents: Vec<ShardExtent>,
    kernel: Option<TileKernelSpec>,
}

/// Select destination work from measured geometry and the copy's requested
/// policy. Geometry caches remain reusable when this policy changes.
fn select_destination_packing(
    destination: &TensorType,
    extents: &[ShardExtent],
    geometry: &crate::storage::DestinationGeometry,
    policy: PackingPolicy,
) -> ExpansionResult<Option<CopyStaging>> {
    let transform = geometry.semantic
        && (!geometry.destination_word_aligned
            || (destination.format.layout.order != ElementOrder::RowMajor
                && !geometry.same_element_order));
    if !transform {
        return match policy {
            PackingPolicy::Staged => Err(ExpansionError::InvalidCopyPlan),
            _ => Ok(None),
        };
    }
    let stage = match policy {
        PackingPolicy::Automatic => {
            let bytes = u64::from(geometry.bytes);
            let clear_cycles = if geometry.padding {
                crate::estimate::IPU21_TARGET_COSTS
                    .kernel_launch_cycles
                    .saturating_add(bytes.div_ceil(8 * 6))
            } else {
                0
            };
            let direct = geometry.fragments.is_some_and(|fragments| {
                crate::estimate::exchange_work_cycles(
                    bytes,
                    fragments.saturating_mul(crate::estimate::EXCHANGE_FRAGMENT_CONTROLS),
                )
                .saturating_add(clear_cycles)
                    < crate::estimate::row_major_pack_cycles(
                        destination,
                        bytes.div_ceil(destination.format.precision.bytes().max(1)),
                    )
            });
            !direct
        }
        PackingPolicy::Direct if geometry.fragments.is_some() => false,
        PackingPolicy::Staged => true,
        _ => return Err(ExpansionError::InvalidCopyPlan),
    };
    if !stage {
        return Ok(None);
    }
    let mut extents = extents.to_vec();
    for extent in &mut extents {
        extent.physical_end = extent.logical_end;
    }
    let tensor_type = TensorType {
        shape: destination.shape.clone(),
        format: TensorFormat {
            precision: destination.format.precision,
            layout: Layout::row_major(TensorTiling::replicated(1)),
        },
    };
    let kernel = (destination.format.precision == Precision::F16
        && matches!(
            destination.format.layout.order,
            ElementOrder::Amp(AmpOrder::Left | AmpOrder::TransposedRight)
                | ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. })
        ))
    .then(|| TileKernelSpec::Rearrange {
        from: tensor_type.format.layout.clone(),
        to: destination.format.layout.clone(),
    });
    Ok(Some(CopyStaging {
        tensor_type,
        extents,
        kernel,
    }))
}
