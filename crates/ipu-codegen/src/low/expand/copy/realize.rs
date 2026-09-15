//! Realize mapped copy regions as preparation, exchange and destination work.

use super::*;
use crate::mid::MidOperationKind;
use crate::tensor::{BlockMajorOrder, TensorFormat};

pub(in crate::low::expand) fn append_span_copies(
    cache: &crate::storage::GeometryCache,
    shards: &[BlockValue],
    source: &ShardView,
    destination: &ShardView,
    tile: u16,
    copies: &mut Vec<(u16, LocalCopy)>,
    order: CopyOrder,
) -> ExpansionResult<()> {
    let a = source.bind(shards)?;
    let b = destination.bind(shards)?;
    let source_geometry = a.geometry(cache, order)?;
    let target_geometry = b.geometry(cache, order)?;
    let pair = cache.pair(&source_geometry, &target_geometry)?;
    copies.extend(
        LocalCopy::from_pair(
            source.shard,
            destination.shard,
            a.backing.0 == b.backing.0,
            &pair,
        )?
        .into_iter()
        .map(|copy| (tile, copy)),
    );
    Ok(())
}

/// Work needed to populate ordinary destination blocks at one exchange boundary.
#[derive(Default)]
pub(in crate::low::expand) struct MaterializationBatch {
    transfers: BTreeMap<CopyOrder, BTreeMap<ShardView, Vec<ShardView>>>,
    before: Vec<(u16, LocalCopy)>,
    after: Vec<(u16, LocalCopy)>,
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
        let source_shape = self.program.logical_values[input.index() as usize]
            .tensor_type
            .shape
            .clone();
        let output_order = self.program.shards[outputs
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
        let source_order = self.program.shards[inputs
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
            let mappings = self.offset_copy_mappings(&inputs, &outputs, &mapping.offsets)?;
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
                    self.program.shards[output.index() as usize].definition,
                    ShardDefinition::Value(_)
                )
            {
                let bytes = shard_storage_bytes(&self.program.shards[output.index() as usize])?;
                self.append_zero_range(
                    body,
                    output,
                    crate::ByteSpan { offset: 0, bytes },
                    false,
                    operation_provenance(operation),
                )?;
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

    fn unpack_amp_to_row_major(
        &mut self,
        source: MidValueId,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<Option<Vec<ShardView>>> {
        let sources = self.value_views(source)?.to_vec();
        for source_view in &sources {
            let source = &self.program.shards[source_view.shard.index() as usize];
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
            let block = &self.program.shards[source_view.shard.index() as usize];
            let tile = block.tile;
            let mut staging_type = self.program.logical_values[source.index() as usize]
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
                MidOperationKind::Rearrange { from, to },
                vec![source_view],
                vec![self.full_view(staging)],
            )?;
            self.append_kernel(tiles, tile, run)?;
            staging_views.push(self.full_view(staging));
        }
        Ok(Some(staging_views))
    }

    fn local_rearrangement(
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
            let source = &self.program.shards[input.shard.index() as usize];
            let destination = &self.program.shards[output.index() as usize];
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
                MidOperationKind::Rearrange {
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
}

impl TileGraphBuilder {
    pub(in crate::low::expand) fn prepare_mapped_views(
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
            let destination = &self.program.shards[destination_shard.index() as usize];
            let geometric_mappings = mappings
                .iter()
                .map(|(source, target)| {
                    let source = source.bind(&self.program.shards)?;
                    target.bind(&self.program.shards)?;
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
                    tile: self.program.shards[destination_shard.index() as usize].tile,
                    tensor_type: staging.tensor_type.clone(),
                    extents: staging.extents.clone(),
                    definition: ShardDefinition::Staging,
                })?)
            } else {
                None
            };
            if let Some(staging) = staging {
                let block = &self.program.shards[staging.index() as usize];
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
                let source_tile = self.program.shards[source.shard.index() as usize].tile;
                let destination_tile = self.program.shards[destination.shard.index() as usize].tile;
                if source_tile == destination_tile {
                    if staging.is_none()
                        && copy_order == exchange_order
                        && self.program.shards[source.shard.index() as usize]
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
                        &self.program.shards,
                        &source,
                        &destination,
                        destination_tile,
                        copies,
                        copy_order,
                    )?;
                } else {
                    if exchange_order != CopyOrder::Panels
                        && !source
                            .bind(&self.program.shards)?
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
                let tile = self.program.shards[destination_shard.index() as usize].tile;
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
                        &self.program.shards,
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
    /// All these clears precede the copies. Overwriting a covered gap is safe
    /// and cheaper than another launch when the gap is sufficiently small.
    fn append_copy_clears(
        &mut self,
        tiles: &mut BlockRegion,
        shard: BlockValueId,
        ranges: &[crate::ByteSpan],
        provenance: WorkProvenance,
    ) -> ExpansionResult<()> {
        let block = &self.program.shards[shard.index() as usize];
        let padding = crate::storage::uncovered_bytes(
            block.storage(),
            [block.extents.as_slice()],
            CopyOrder::Semantic,
        )?;
        let padding_only = !padding.is_empty() && ranges == padding;
        let bytes = shard_storage_bytes(block)?;
        let mut ranges = ranges
            .iter()
            .map(|range| {
                let start = range.offset / 8 * 8;
                let end = (u64::from(range.offset) + u64::from(range.bytes)).div_ceil(8) * 8;
                crate::ByteSpan {
                    offset: start,
                    bytes: end.min(u64::from(bytes)) as u32 - start,
                }
            })
            .peekable();
        let launch_bytes = crate::estimate::IPU21_TARGET_COSTS.kernel_launch_cycles * 48;
        while let Some(mut range) = ranges.next() {
            while let Some(next) = ranges.peek()
                && u64::from(next.offset.saturating_sub(range.offset + range.bytes)) <= launch_bytes
            {
                range.bytes = next.offset + next.bytes - range.offset;
                ranges.next();
            }
            self.append_zero_range(tiles, shard, range, padding_only, provenance)?;
        }
        Ok(())
    }

    fn append_zero_range(
        &mut self,
        tiles: &mut BlockRegion,
        shard: BlockValueId,
        range: crate::ByteSpan,
        padding_only: bool,
        provenance: WorkProvenance,
    ) -> ExpansionResult<()> {
        let tile = self.program.shards[shard.index() as usize].tile;
        {
            let run = self.bind_kernel(
                provenance,
                MidOperationKind::FillZero {
                    offset: range.offset,
                    bytes: range.bytes,
                    padding_only,
                },
                Vec::new(),
                vec![self.full_view(shard)],
            )?;
            self.append_kernel(tiles, tile, run)
        }
    }

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
        let shard = &self.program.shards[source.shard.index() as usize];
        let tile = shard.tile;
        let bytes = source
            .bind(&self.program.shards)?
            .traversal(order)?
            .byte_len();
        if bytes == 0 || !bytes.is_multiple_of(4) {
            return Err(ExpansionError::InvalidCopyPlan);
        }
        let precision = shard.tensor_type.format.precision;
        let elements =
            u32::try_from(bytes / precision.bytes()).map_err(|_| ExpansionError::IdOverflow)?;
        let staging = self.push_shard(BlockValue {
            id: BlockValueId(0),
            tile,
            tensor_type: TensorType::new(
                [elements],
                precision,
                Layout::row_major(TensorTiling::replicated(1)),
            ),
            extents: vec![ShardExtent {
                axis: 0,
                start: 0,
                logical_end: elements,
                physical_end: elements,
            }],
            definition: ShardDefinition::Staging,
        })?;
        let view = self.full_view(staging);
        append_span_copies(
            &self.cache,
            &self.program.shards,
            &source,
            &view,
            tile,
            copies,
            order,
        )?;
        packed.insert(key, view.clone());
        Ok(view)
    }

    pub(in crate::low::expand) fn append_materialization(
        &mut self,
        mut batch: MaterializationBatch,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        for (source, destination, order) in batch.loopback_candidates {
            let transfers = batch.transfers.entry(order).or_default();
            // Keep the existing multicast send; only add its local receiver.
            // Placement separates same-class source/receiver SRAM elements.
            let source_spans = source.bind(&self.program.shards)?.traversal(order)?;
            let destination_spans = destination.bind(&self.program.shards)?.traversal(order)?;
            let aligned = source_spans.word_aligned() && destination_spans.word_aligned();
            if let Some(destinations) = transfers.get_mut(&source)
                && destinations.len() >= 2
                && aligned
                // The local receiver must not split existing messages further.
                && destinations.iter().any(|view| {
                    view.bind(&self.program.shards).and_then(|bound| bound.traversal(order)).is_ok_and(|remote|
                        remote.spans().map(|span| span.bytes)
                            .eq(destination_spans.spans().map(|span| span.bytes)))
                })
                && destinations.iter().all(|view| {
                    self.program.shards[view.shard.index() as usize].tile
                        != self.program.shards[source.shard.index() as usize].tile
                })
            {
                destinations.push(destination);
            } else {
                append_span_copies(
                    &self.cache,
                    &self.program.shards,
                    &source,
                    &destination,
                    self.program.shards[source.shard.index() as usize].tile,
                    &mut batch.after,
                    order,
                )?;
            }
        }
        for (tile, copy) in batch.before {
            self.append_local_copy(
                tiles,
                tile,
                crate::kernel::CopyRun::bind(copy, &self.program.shards)?,
            )?;
        }
        let mut transfers = Vec::new();
        for (order, mappings) in batch.transfers {
            transfers.extend(mappings.into_iter().map(|(source, mut destinations)| {
                destinations.sort_unstable();
                destinations.dedup();
                LogicalExchange {
                    source,
                    destinations,
                    order,
                }
            }));
        }
        self.append_exchange_phase(transfers, provenance, tiles)?;
        for (tile, copy) in batch.after {
            self.append_local_copy(
                tiles,
                tile,
                crate::kernel::CopyRun::bind(copy, &self.program.shards)?,
            )?;
        }
        for (tile, run) in batch.kernels {
            self.append_kernel(tiles, tile, run)?;
        }
        Ok(())
    }

    fn logical_view(&self, shard: BlockValueId) -> ShardView {
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
    kernel: Option<MidOperationKind>,
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
                    < crate::kernel::rearrange::estimate(
                        ElementOrder::RowMajor,
                        crate::storage::TensorStorage {
                            format: &destination.format,
                            extents,
                        },
                        crate::storage::TensorStorage {
                            format: &destination.format,
                            extents,
                        },
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
    .then(|| MidOperationKind::Rearrange {
        from: tensor_type.format.layout.clone(),
        to: destination.format.layout.clone(),
    });
    Ok(Some(CopyStaging {
        tensor_type,
        extents,
        kernel,
    }))
}
