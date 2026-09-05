//! Attention panel preparation and distribution.

use super::*;

impl BlockBuilder {
    pub(super) fn materialize_attention_queries(
        &mut self,
        query: MidValueId,
        tasks: &[AttentionTask],
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> BlockBuildResult<()> {
        if !self.has_deferred_value(query) {
            return Ok(());
        }
        let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
        let mut local_copies = Vec::new();
        let physical = tasks.iter().all(|task| task.query_receive.is_none());
        let physical = physical
            && tasks
                .iter()
                .all(|task| self.deferred_supports_physical_exchange(query, task.query));
        let direct = tasks.iter().all(|task| task.query_receive.is_none());
        if direct && !physical {
            for task in tasks {
                if self.shard_has_padding(task.query) {
                    self.append_fill_zero(tiles, task.query, provenance)?;
                }
            }
        }
        for task in tasks {
            self.gather_deferred_panel(
                query,
                task.head,
                task.query_row_start,
                task.query_rows,
                0,
                task.query_dimension,
                task.query_receive.unwrap_or(task.query),
                if physical {
                    CopyOrder::Physical
                } else {
                    CopyOrder::Semantic
                },
                &mut transfers,
                &mut local_copies,
            )?;
        }
        for (tile, copy) in local_copies {
            self.append_local_copy(tiles, tile, copy)?;
        }
        self.append_ordered_phase(
            transfers,
            provenance,
            if physical {
                CopyOrder::Physical
            } else {
                CopyOrder::Semantic
            },
            tiles,
        )?;
        if !physical {
            for task in tasks {
                self.append_attention_rearrange(
                    tiles,
                    task.tile,
                    task.query_receive
                        .ok_or(BlockBuildError::InvalidOperatorPlan)?,
                    task.query,
                    WorkProvenance {
                        operation: provenance.operation,
                        value: provenance.value,
                        reason: WorkReason::OperatorKernel,
                    },
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_distributed_attention_panels(
        &mut self,
        value: MidValueId,
        destinations: &BTreeMap<u32, Vec<BlockValueId>>,
        block_start: u32,
        valid_rows: u32,
        logical_columns: u32,
        physical_columns: u32,
        order: ElementOrder,
        owner_offset: u32,
        semantic_gathers: &mut BTreeMap<ShardView, Vec<ShardView>>,
        physical_gathers: &mut BTreeMap<ShardView, Vec<ShardView>>,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> BlockBuildResult<Vec<PreparedDistributedPanel>> {
        let panels = physical_columns.div_ceil(AMP_COLUMN_MICRO);
        if panels == 0 || valid_rows == 0 {
            return Err(BlockBuildError::InvalidOperatorPlan);
        }
        let mut packed_panels = Vec::new();
        for (&stream, stream_destinations) in destinations {
            for panel in 0..panels {
                let column_start = panel * AMP_COLUMN_MICRO;
                let panel_columns = logical_columns
                    .saturating_sub(column_start)
                    .min(AMP_COLUMN_MICRO);
                if panel_columns == 0 {
                    continue;
                }
                let owner = usize::try_from(owner_offset.saturating_add(panel))
                    .map_err(|_| BlockBuildError::IdOverflow)?
                    % stream_destinations.len();
                let tile = stream_destinations[owner];
                let tile = self.shards[tile.index() as usize].tile;
                let packed = self.push_attention_buffer(
                    tile,
                    valid_rows,
                    AMP_INNER_BLOCK,
                    panel_columns,
                    AMP_COLUMN_MICRO,
                    order,
                )?;
                let physical = self.deferred_supports_physical_exchange(value, packed);
                let word_exchange = !physical
                    && self.deferred_panel_benefits_from_word_exchange(
                        value,
                        stream,
                        block_start,
                        valid_rows,
                        column_start,
                        panel_columns,
                        packed,
                    )?;
                if word_exchange && self.shard_has_padding(packed) {
                    self.append_fill_zero(tiles, packed, provenance)?;
                }
                let row_major = if physical || word_exchange {
                    None
                } else {
                    Some(self.push_attention_buffer(
                        tile,
                        valid_rows,
                        valid_rows,
                        panel_columns,
                        panel_columns,
                        ElementOrder::RowMajor,
                    )?)
                };
                let gather_destination = row_major.unwrap_or(packed);
                let mut local_copies = Vec::new();
                self.gather_deferred_panel(
                    value,
                    stream,
                    block_start,
                    valid_rows,
                    column_start,
                    panel_columns,
                    gather_destination,
                    if physical {
                        CopyOrder::Physical
                    } else {
                        CopyOrder::Semantic
                    },
                    if physical {
                        physical_gathers
                    } else {
                        semantic_gathers
                    },
                    &mut local_copies,
                )?;
                for (tile, copy) in local_copies {
                    self.append_local_copy(tiles, tile, copy)?;
                }
                packed_panels.push(PreparedDistributedPanel {
                    panel,
                    row_major,
                    packed,
                    tile,
                    destinations: stream_destinations.clone(),
                });
            }
        }
        Ok(packed_panels)
    }

    pub(super) fn append_prepared_panel_broadcasts(
        &mut self,
        panels: &[PreparedDistributedPanel],
        destination_row_start: u32,
        broadcasts: &mut BTreeMap<ShardView, Vec<ShardView>>,
        tiles: &mut BlockRegion,
    ) -> BlockBuildResult<()> {
        for panel in panels {
            let source = self.full_view(panel.packed);
            let source_rows = source.extents[0].physical_end - source.extents[0].start;
            let column_start = panel.panel * AMP_COLUMN_MICRO;
            for &destination in &panel.destinations {
                let destination_tile = self.shards[destination.index() as usize].tile;
                let destination_view = self.narrow_view(
                    destination,
                    &[
                        (
                            0,
                            destination_row_start,
                            destination_row_start + source_rows,
                        ),
                        (1, column_start, column_start + AMP_COLUMN_MICRO),
                    ],
                )?;
                if panel.tile == destination_tile {
                    let mut copies = Vec::new();
                    append_span_copies(
                        &self.shards,
                        &source,
                        &destination_view,
                        panel.tile,
                        &mut copies,
                        CopyOrder::Physical,
                    )?;
                    for (tile, copy) in copies {
                        self.append_local_copy(tiles, tile, copy)?;
                    }
                } else {
                    broadcasts
                        .entry(source.clone())
                        .or_default()
                        .push(destination_view);
                }
            }
        }
        Ok(())
    }

    pub(super) fn append_attention_rearrange(
        &mut self,
        tiles: &mut BlockRegion,
        tile: u16,
        source: BlockValueId,
        destination: BlockValueId,
        provenance: WorkProvenance,
    ) -> BlockBuildResult<()> {
        let input = self.shards[source.index() as usize]
            .tensor_type
            .format
            .clone();
        let output = self.shards[destination.index() as usize]
            .tensor_type
            .format
            .clone();
        self.append_kernel(
            tiles,
            tile,
            self.kernel_run(
                provenance,
                TileKernelSpec::Rearrange {
                    from: input.layout.clone(),
                    to: output.layout.clone(),
                },
                vec![KernelOperand {
                    views: vec![self.full_view(source)],
                }],
                self.full_view(destination),
            )?,
        )
    }
}
