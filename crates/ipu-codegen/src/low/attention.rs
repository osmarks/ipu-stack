//! Blocked and materialized attention tile phases.

use super::*;

pub(super) struct PreparedDistributedPanel {
    pub(super) panel: u32,
    pub(super) row_major: Option<LowShardId>,
    pub(super) packed: LowShardId,
    pub(super) tile: u16,
    pub(super) destinations: Vec<LowShardId>,
}

pub(super) struct PreparedAttentionBlock {
    pub(super) row_start: u32,
    pub(super) valid_rows: u32,
    pub(super) key_panels: Vec<PreparedDistributedPanel>,
    pub(super) value_panels: Vec<PreparedDistributedPanel>,
}

pub(super) struct AttentionTask {
    pub(super) tile: u16,
    pub(super) head: u32,
    pub(super) query_row_start: u32,
    pub(super) query_rows: u32,
    pub(super) query_dimension: u32,
    pub(super) value_dimension: u32,
    pub(super) query: LowShardId,
    pub(super) query_receive: Option<LowShardId>,
    pub(super) output: LowShardId,
    pub(super) scratch: LowShardId,
    pub(super) weights: LowShardId,
    pub(super) key_staging: LowShardId,
    pub(super) value_staging: LowShardId,
}

#[derive(Clone, Copy)]
pub(super) struct AttentionBufferShape {
    pub(super) query_block_rows: u32,
    pub(super) logical_staging_rows: u32,
    pub(super) physical_staging_rows: u32,
    pub(super) scratch_columns: u32,
    pub(super) state_columns: u32,
    pub(super) padded_query_dimension: u32,
    pub(super) padded_value_dimension: u32,
    pub(super) reuse_key_staging_for_state: bool,
}

#[derive(Clone, Copy)]
pub(super) enum AttentionOperand {
    Key,
    Value,
}

impl LoweringState {
    pub(super) fn build_attention_tasks(
        &mut self,
        query: MidValueId,
        result: MidValueId,
        shape: AttentionBufferShape,
    ) -> LowLoweringResult<Vec<AttentionTask>> {
        let value_row_block = u16::try_from(shape.physical_staging_rows)
            .map_err(|_| LowLoweringError::InvalidOperatorPlan)?;
        let outputs = self.value_shards(result)?.to_vec();
        let mut tasks = Vec::with_capacity(outputs.len());
        for output in outputs {
            let tile = self.shards[output.index() as usize].tile;
            let rank = self.shards[output.index() as usize].extents.len();
            if rank != 3 {
                return Err(LowLoweringError::InvalidOperatorPlan);
            }
            let rows = self.shards[output.index() as usize].extents[rank - 2].physical_end
                - self.shards[output.index() as usize].extents[rank - 2].start;
            let value_dimension = *self.shards[output.index() as usize]
                .tensor_type
                .shape
                .0
                .last()
                .ok_or(LowLoweringError::InvalidOperatorPlan)?;
            if rows == 0 || rows > shape.query_block_rows {
                return Err(LowLoweringError::InvalidOperatorPlan);
            }
            let canonical_query = self.local_shard(query, tile)?;
            let query_dimension = *self.shards[canonical_query.index() as usize]
                .tensor_type
                .shape
                .0
                .last()
                .ok_or(LowLoweringError::InvalidOperatorPlan)?;
            let deferred_query = self.has_deferred_value(query);
            let query_shard = if deferred_query {
                self.push_attention_buffer(
                    tile,
                    rows,
                    rows,
                    query_dimension,
                    shape.padded_query_dimension,
                    ElementOrder::Amp(AmpOrder::Left),
                )?
            } else {
                canonical_query
            };
            let direct_query = deferred_query
                && (self.deferred_supports_physical_exchange(query, query_shard)
                    || self.deferred_panel_benefits_from_word_exchange(
                        query,
                        self.shards[output.index() as usize].extents[rank - 3].start,
                        self.shards[output.index() as usize].extents[rank - 2].start,
                        rows,
                        0,
                        query_dimension,
                        query_shard,
                    )?);
            let query_receive = (deferred_query && !direct_query)
                .then(|| {
                    self.push_attention_buffer(
                        tile,
                        rows,
                        rows,
                        query_dimension,
                        query_dimension,
                        ElementOrder::RowMajor,
                    )
                })
                .transpose()?;
            let scratch = self.push_attention_scratch(
                tile,
                rows,
                shape.scratch_columns,
                Precision::F16,
                ElementOrder::Amp(AmpOrder::Left),
                MemoryClass::Ipu21Interleaved,
            )?;
            let key_staging = self.push_attention_buffer(
                tile,
                shape.logical_staging_rows,
                shape.physical_staging_rows,
                query_dimension,
                shape.padded_query_dimension,
                ElementOrder::Amp(AmpOrder::TransposedRight),
            )?;
            self.shards[key_staging.index() as usize].definition = ShardDefinition::ExchangeStaging;
            let weights = self.push_attention_scratch(
                tile,
                rows,
                shape.state_columns,
                Precision::F16,
                ElementOrder::Amp(AmpOrder::Left),
                MemoryClass::Ipu21Standard,
            )?;
            if shape.reuse_key_staging_for_state
                && crate::shard_storage_bytes(&self.shards[weights.index() as usize])?
                    <= crate::shard_storage_bytes(&self.shards[key_staging.index() as usize])?
            {
                // Materialized QK consumes the packed K matrix before softmax
                // starts. Reinterpret that now-dead standard-memory allocation
                // as probabilities plus row state so the PV kernel retains its
                // proven standard-load path without a second large buffer.
                self.shards[weights.index() as usize].definition =
                    ShardDefinition::Alias(key_staging);
            }
            let value_staging = self.push_attention_buffer(
                tile,
                shape.logical_staging_rows,
                shape.physical_staging_rows,
                value_dimension,
                shape.padded_value_dimension,
                ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                    row_block: value_row_block,
                    column_block: AMP_COLUMN_MICRO as u16,
                }),
            )?;
            self.shards[value_staging.index() as usize].definition =
                ShardDefinition::ExchangeStaging;
            tasks.push(AttentionTask {
                tile,
                head: self.shards[output.index() as usize].extents[rank - 3].start,
                query_row_start: self.shards[output.index() as usize].extents[rank - 2].start,
                query_rows: rows,
                query_dimension,
                value_dimension,
                query: query_shard,
                query_receive,
                output,
                scratch,
                weights,
                key_staging,
                value_staging,
            });
        }
        Ok(tasks)
    }

    pub(super) fn materialize_attention_queries(
        &mut self,
        query: MidValueId,
        tasks: &[AttentionTask],
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
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
                        .ok_or(LowLoweringError::InvalidOperatorPlan)?,
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
    pub(super) fn prepare_attention_blocks(
        &mut self,
        key: MidValueId,
        value: MidValueId,
        tasks: &[AttentionTask],
        key_rows: u32,
        block_rows: u32,
        padded_query_dimension: u32,
        padded_value_dimension: u32,
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<Vec<PreparedAttentionBlock>> {
        let blocks = key_rows.div_ceil(block_rows);
        let key_destinations = tasks.iter().fold(
            BTreeMap::<u32, Vec<LowShardId>>::new(),
            |mut destinations, task| {
                destinations
                    .entry(task.head)
                    .or_default()
                    .push(task.key_staging);
                destinations
            },
        );
        let value_destinations = tasks.iter().fold(
            BTreeMap::<u32, Vec<LowShardId>>::new(),
            |mut destinations, task| {
                destinations
                    .entry(task.head)
                    .or_default()
                    .push(task.value_staging);
                destinations
            },
        );
        let key_panel_count = padded_query_dimension.div_ceil(AMP_COLUMN_MICRO);
        let value_panel_count = padded_value_dimension.div_ceil(AMP_COLUMN_MICRO);
        let mut semantic_gathers = BTreeMap::<ShardView, Vec<ShardView>>::new();
        let mut physical_gathers = BTreeMap::<ShardView, Vec<ShardView>>::new();
        let mut prepared = Vec::new();
        for block in 0..blocks {
            let row_start = block * block_rows;
            let valid_rows = key_rows.saturating_sub(row_start).min(block_rows);
            let owner_offset = block.saturating_mul(key_panel_count + value_panel_count);
            let key_panels = self.prepare_distributed_attention_panels(
                key,
                &key_destinations,
                row_start,
                valid_rows,
                tasks[0].query_dimension,
                padded_query_dimension,
                ElementOrder::Amp(AmpOrder::TransposedRight),
                owner_offset,
                &mut semantic_gathers,
                &mut physical_gathers,
                provenance,
                tiles,
            )?;
            let row_block =
                u16::try_from(block_rows).map_err(|_| LowLoweringError::InvalidOperatorPlan)?;
            let value_panels = self.prepare_distributed_attention_panels(
                value,
                &value_destinations,
                row_start,
                valid_rows,
                tasks[0].value_dimension,
                padded_value_dimension,
                ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                    row_block,
                    column_block: AMP_COLUMN_MICRO as u16,
                }),
                owner_offset + key_panel_count,
                &mut semantic_gathers,
                &mut physical_gathers,
                provenance,
                tiles,
            )?;
            prepared.push(PreparedAttentionBlock {
                row_start,
                valid_rows,
                key_panels,
                value_panels,
            });
        }
        self.append_mixed_phase(semantic_gathers, physical_gathers, provenance, tiles)?;
        for block in &prepared {
            for panel in block.key_panels.iter().chain(&block.value_panels) {
                if let Some(row_major) = panel.row_major {
                    self.append_attention_rearrange(
                        tiles,
                        panel.tile,
                        row_major,
                        panel.packed,
                        WorkProvenance {
                            operation: provenance.operation,
                            value: provenance.value,
                            reason: WorkReason::OperatorKernel,
                        },
                    )?;
                }
            }
        }
        Ok(prepared)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower_blocked_attention(
        &mut self,
        operation: &MidOperation,
        query_key: TileKernelSpec,
        probability_value: TileKernelSpec,
        query_block_rows: u32,
        key_block_rows: u32,
        padded_query_dimension: u32,
        padded_value_dimension: u32,
        requirements: &StorageRequirements,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [query, key, value] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidOperatorPlan);
        };
        let [result] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        if key_block_rows != AMP_INNER_BLOCK || query_block_rows == 0 {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let key_shards = self.value_shards(*key)?.to_vec();
        let value_shards = self.value_shards(*value)?.to_vec();
        if key_shards.len() != value_shards.len() {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        if self.has_deferred_value(*key) != self.has_deferred_value(*value) {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let tasks = self.build_attention_tasks(
            *query,
            *result,
            AttentionBufferShape {
                query_block_rows,
                logical_staging_rows: key_block_rows,
                physical_staging_rows: key_block_rows,
                scratch_columns: padded_value_dimension.max(key_block_rows),
                state_columns: key_block_rows + 16,
                padded_query_dimension,
                padded_value_dimension,
                reuse_key_staging_for_state: false,
            },
        )?;
        let key_rows = self.shards[self.value_shards(*key)?[0].index() as usize]
            .tensor_type
            .shape
            .0[1];
        let blocks = usize::try_from(key_rows.div_ceil(key_block_rows))
            .map_err(|_| LowLoweringError::IdOverflow)?;
        if blocks == 0 {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let exchange_provenance = WorkProvenance {
            operation: operation.source,
            value: Some(*key),
            reason: WorkReason::OperatorInputs,
        };
        let kernel_provenance = WorkProvenance {
            operation: operation.source,
            value: Some(*result),
            reason: WorkReason::OperatorKernel,
        };
        self.materialize_attention_queries(*query, &tasks, exchange_provenance, tiles)?;
        let deferred_key_value = self.has_deferred_value(*key);
        let prepared_blocks = if deferred_key_value {
            self.prepare_attention_blocks(
                *key,
                *value,
                &tasks,
                key_rows,
                key_block_rows,
                padded_query_dimension,
                padded_value_dimension,
                exchange_provenance,
                tiles,
            )?
        } else {
            Vec::new()
        };
        for block in 0..blocks {
            let block_start =
                u32::try_from(block).map_err(|_| LowLoweringError::IdOverflow)? * key_block_rows;
            let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
            let mut task_sources = Vec::with_capacity(tasks.len());
            if deferred_key_value {
                let prepared = prepared_blocks
                    .get(block)
                    .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                self.append_prepared_panel_broadcasts(
                    &prepared.key_panels,
                    0,
                    &mut transfers,
                    tiles,
                )?;
                self.append_prepared_panel_broadcasts(
                    &prepared.value_panels,
                    0,
                    &mut transfers,
                    tiles,
                )?;
                self.append_phase(transfers, exchange_provenance, tiles)?;
                task_sources.extend(
                    tasks
                        .iter()
                        .map(|task| (task.key_staging, task.value_staging, prepared.valid_rows)),
                );
            } else {
                for task in &tasks {
                    let source_matches = |candidate: &&LowShardId| {
                        let shard = &self.shards[candidate.index() as usize];
                        shard.extents[0].start == task.head && shard.extents[1].start == block_start
                    };
                    let key_source = *key_shards
                        .iter()
                        .find(source_matches)
                        .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                    let value_source = *value_shards
                        .iter()
                        .find(source_matches)
                        .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                    let valid_key_rows = self.shards[key_source.index() as usize].extents[1]
                        .logical_end
                        .saturating_sub(block_start);
                    let mut operands = Vec::with_capacity(2);
                    for (source, destination) in [
                        (key_source, task.key_staging),
                        (value_source, task.value_staging),
                    ] {
                        if self.shards[source.index() as usize].tile == task.tile {
                            operands.push(source);
                        } else {
                            transfers
                                .entry(self.full_view(source))
                                .or_default()
                                .push(self.full_view(destination));
                            operands.push(destination);
                        }
                    }
                    task_sources.push((operands[0], operands[1], valid_key_rows));
                }
                self.append_phase(transfers, exchange_provenance, tiles)?;
            }
            for (task, (key_operand, value_operand, valid_key_rows)) in
                tasks.iter().zip(task_sources)
            {
                let score_view = self.narrow_view(task.scratch, &[(1, 0, key_block_rows)])?;
                self.append_kernel(
                    tiles,
                    task.tile,
                    KernelRun::new(
                        kernel_provenance,
                        query_key.clone(),
                        vec![
                            KernelOperand {
                                views: vec![self.full_view(task.query)],
                            },
                            KernelOperand {
                                views: vec![self.full_view(key_operand)],
                            },
                        ],
                        score_view.clone(),
                        requirements.clone(),
                    ),
                )?;
                self.append_kernel(
                    tiles,
                    task.tile,
                    KernelRun::new(
                        kernel_provenance,
                        TileKernelSpec::AttentionSoftmax {
                            head_dimension: task.query_dimension,
                            key_columns: valid_key_rows,
                            padded_key_columns: key_block_rows,
                        },
                        vec![KernelOperand {
                            views: vec![score_view],
                        }],
                        self.full_view(task.weights),
                        requirements.clone(),
                    ),
                )?;
                let probability_view = self.narrow_view(task.weights, &[(1, 0, key_block_rows)])?;
                let block_value_view =
                    self.narrow_view(task.scratch, &[(1, 0, padded_value_dimension)])?;
                self.append_kernel(
                    tiles,
                    task.tile,
                    KernelRun::new(
                        kernel_provenance,
                        probability_value.clone(),
                        vec![
                            KernelOperand {
                                views: vec![probability_view],
                            },
                            KernelOperand {
                                views: vec![self.full_view(value_operand)],
                            },
                        ],
                        block_value_view.clone(),
                        requirements.clone(),
                    ),
                )?;
                self.append_kernel(
                    tiles,
                    task.tile,
                    KernelRun::new(
                        kernel_provenance,
                        TileKernelSpec::AttentionMerge {
                            value_dimension: task.value_dimension,
                            padded_value_dimension,
                            key_block_columns: key_block_rows,
                            initial: block == 0,
                            final_block: block + 1 == blocks,
                        },
                        vec![
                            KernelOperand {
                                views: vec![block_value_view],
                            },
                            KernelOperand {
                                views: vec![self.full_view(task.weights)],
                            },
                        ],
                        self.full_view(task.output),
                        requirements.clone(),
                    ),
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn append_materialized_attention_input(
        &mut self,
        operand: AttentionOperand,
        sources: &[LowShardId],
        prepared: &[PreparedAttentionBlock],
        tasks: &[AttentionTask],
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
        if prepared.is_empty() {
            for task in tasks {
                let destination = match operand {
                    AttentionOperand::Key => task.key_staging,
                    AttentionOperand::Value => task.value_staging,
                };
                let matching_sources = sources
                    .iter()
                    .copied()
                    .filter(|source| {
                        self.shards[source.index() as usize].extents[0].start == task.head
                    })
                    .collect::<Vec<_>>();
                for source in matching_sources {
                    let source_view = self.full_view(source);
                    let row_extent = self.shards[source.index() as usize].extents[1];
                    let destination_view = self.narrow_view(
                        destination,
                        &[(0, row_extent.start, row_extent.physical_end)],
                    )?;
                    let source_tile = self.shards[source.index() as usize].tile;
                    if source_tile == task.tile {
                        let mut copies = Vec::new();
                        append_span_copies(
                            &self.shards,
                            &source_view,
                            &destination_view,
                            task.tile,
                            &mut copies,
                            CopyOrder::Physical,
                        )?;
                        for (tile, copy) in copies {
                            self.append_local_copy(tiles, tile, copy)?;
                        }
                    } else {
                        transfers
                            .entry(source_view)
                            .or_default()
                            .push(destination_view);
                    }
                }
            }
        } else {
            for block in prepared {
                let panels = match operand {
                    AttentionOperand::Key => &block.key_panels,
                    AttentionOperand::Value => &block.value_panels,
                };
                self.append_prepared_panel_broadcasts(
                    panels,
                    block.row_start,
                    &mut transfers,
                    tiles,
                )?;
            }
        }
        self.append_physical_phase(transfers, provenance, tiles)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower_materialized_attention(
        &mut self,
        operation: &MidOperation,
        query_key: TileKernelSpec,
        probability_value: TileKernelSpec,
        query_block_rows: u32,
        padded_key_rows: u32,
        padded_query_dimension: u32,
        padded_value_dimension: u32,
        requirements: &StorageRequirements,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [query, key, value] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidOperatorPlan);
        };
        let [result] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        if query_block_rows == 0
            || padded_key_rows == 0
            || !padded_key_rows.is_multiple_of(AMP_INNER_BLOCK)
            || self.has_deferred_value(*key) != self.has_deferred_value(*value)
        {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let key_shards = self.value_shards(*key)?.to_vec();
        let value_shards = self.value_shards(*value)?.to_vec();
        if key_shards.len() != value_shards.len() {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let key_rows = self.shards[key_shards[0].index() as usize]
            .tensor_type
            .shape
            .0[1];
        if key_rows == 0 || key_rows > padded_key_rows {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let tasks = self.build_attention_tasks(
            *query,
            *result,
            AttentionBufferShape {
                query_block_rows,
                logical_staging_rows: key_rows,
                physical_staging_rows: padded_key_rows,
                scratch_columns: padded_key_rows.max(padded_value_dimension),
                state_columns: padded_key_rows + AMP_COLUMN_MICRO,
                padded_query_dimension,
                padded_value_dimension,
                reuse_key_staging_for_state: true,
            },
        )?;
        if tasks.is_empty() {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let exchange_provenance = WorkProvenance {
            operation: operation.source,
            value: Some(*key),
            reason: WorkReason::OperatorInputs,
        };
        let kernel_provenance = WorkProvenance {
            operation: operation.source,
            value: Some(*result),
            reason: WorkReason::OperatorKernel,
        };
        self.materialize_attention_queries(*query, &tasks, exchange_provenance, tiles)?;
        let prepared = if self.has_deferred_value(*key) {
            self.prepare_attention_blocks(
                *key,
                *value,
                &tasks,
                key_rows,
                AMP_INNER_BLOCK,
                padded_query_dimension,
                padded_value_dimension,
                exchange_provenance,
                tiles,
            )?
        } else {
            Vec::new()
        };
        self.append_materialized_attention_input(
            AttentionOperand::Key,
            &key_shards,
            &prepared,
            &tasks,
            exchange_provenance,
            tiles,
        )?;
        for task in &tasks {
            let scores = self.narrow_view(task.scratch, &[(1, 0, padded_key_rows)])?;
            self.append_kernel(
                tiles,
                task.tile,
                KernelRun::new(
                    kernel_provenance,
                    query_key.clone(),
                    vec![
                        KernelOperand {
                            views: vec![self.full_view(task.query)],
                        },
                        KernelOperand {
                            views: vec![self.full_view(task.key_staging)],
                        },
                    ],
                    scores.clone(),
                    requirements.clone(),
                ),
            )?;
            self.append_kernel(
                tiles,
                task.tile,
                KernelRun::new(
                    kernel_provenance,
                    TileKernelSpec::AttentionSoftmax {
                        head_dimension: task.query_dimension,
                        key_columns: key_rows,
                        padded_key_columns: padded_key_rows,
                    },
                    vec![KernelOperand {
                        views: vec![scores],
                    }],
                    self.full_view(task.weights),
                    requirements.clone(),
                ),
            )?;
        }
        self.append_materialized_attention_input(
            AttentionOperand::Value,
            &value_shards,
            &prepared,
            &tasks,
            exchange_provenance,
            tiles,
        )?;
        for task in &tasks {
            let probabilities = self.narrow_view(task.weights, &[(1, 0, padded_key_rows)])?;
            let block_value = self.narrow_view(task.scratch, &[(1, 0, padded_value_dimension)])?;
            self.append_kernel(
                tiles,
                task.tile,
                KernelRun::new(
                    kernel_provenance,
                    probability_value.clone(),
                    vec![
                        KernelOperand {
                            views: vec![probabilities],
                        },
                        KernelOperand {
                            views: vec![self.full_view(task.value_staging)],
                        },
                    ],
                    block_value.clone(),
                    requirements.clone(),
                ),
            )?;
            self.append_kernel(
                tiles,
                task.tile,
                KernelRun::new(
                    kernel_provenance,
                    TileKernelSpec::AttentionMerge {
                        value_dimension: task.value_dimension,
                        padded_value_dimension,
                        key_block_columns: padded_key_rows,
                        initial: true,
                        final_block: true,
                    },
                    vec![
                        KernelOperand {
                            views: vec![block_value],
                        },
                        KernelOperand {
                            views: vec![self.full_view(task.weights)],
                        },
                    ],
                    self.full_view(task.output),
                    requirements.clone(),
                ),
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_distributed_attention_panels(
        &mut self,
        value: MidValueId,
        destinations: &BTreeMap<u32, Vec<LowShardId>>,
        block_start: u32,
        valid_rows: u32,
        logical_columns: u32,
        physical_columns: u32,
        order: ElementOrder,
        owner_offset: u32,
        semantic_gathers: &mut BTreeMap<ShardView, Vec<ShardView>>,
        physical_gathers: &mut BTreeMap<ShardView, Vec<ShardView>>,
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<Vec<PreparedDistributedPanel>> {
        let panels = physical_columns.div_ceil(AMP_COLUMN_MICRO);
        if panels == 0 || valid_rows == 0 {
            return Err(LowLoweringError::InvalidOperatorPlan);
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
                    .map_err(|_| LowLoweringError::IdOverflow)?
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
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
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
        tiles: &mut [TileWorkList],
        tile: u16,
        source: LowShardId,
        destination: LowShardId,
        provenance: WorkProvenance,
    ) -> LowLoweringResult<()> {
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
            KernelRun::new(
                provenance,
                TileKernelSpec::Rearrange {
                    from: input.layout.clone(),
                    to: output.layout.clone(),
                },
                vec![KernelOperand {
                    views: vec![self.full_view(source)],
                }],
                self.full_view(destination),
                StorageRequirements {
                    inputs: vec![OperandRequirement::new(input, 2)],
                    output_aliasing: OutputAliasing::Fresh,
                    output: OperandRequirement::new(output, 2),
                    distinct_elements: Vec::new(),
                },
            ),
        )
    }

    pub(super) fn push_attention_scratch(
        &mut self,
        tile: u16,
        rows: u32,
        columns: u32,
        precision: Precision,
        order: ElementOrder,
        memory_class: MemoryClass,
    ) -> LowLoweringResult<LowShardId> {
        self.push_shard(LowShard {
            id: LowShardId(0),
            tile,
            tensor_type: TensorType::new(
                [rows, columns],
                precision,
                Layout {
                    order,
                    tiling: TensorTiling::replicated(1),
                    memory_class,
                },
            ),
            extents: vec![
                ShardExtent {
                    axis: 0,
                    start: 0,
                    logical_end: rows,
                    physical_end: rows,
                },
                ShardExtent {
                    axis: 1,
                    start: 0,
                    logical_end: columns,
                    physical_end: columns,
                },
            ],
            definition: ShardDefinition::Staging,
        })
    }

    pub(super) fn push_packed_buffer(
        &mut self,
        tile: u16,
        elements: u32,
        precision: Precision,
        definition: ShardDefinition,
    ) -> LowLoweringResult<LowShardId> {
        self.push_shard(LowShard {
            id: LowShardId(0),
            tile,
            tensor_type: TensorType::new(
                [elements],
                precision,
                Layout {
                    order: ElementOrder::RowMajor,
                    tiling: TensorTiling::replicated(1),
                    memory_class: MemoryClass::Ipu21Standard,
                },
            ),
            extents: vec![ShardExtent {
                axis: 0,
                start: 0,
                logical_end: elements,
                physical_end: elements,
            }],
            definition,
        })
    }

    pub(super) fn push_attention_buffer(
        &mut self,
        tile: u16,
        logical_rows: u32,
        physical_rows: u32,
        logical_columns: u32,
        physical_columns: u32,
        order: ElementOrder,
    ) -> LowLoweringResult<LowShardId> {
        self.push_shard(LowShard {
            id: LowShardId(0),
            tile,
            tensor_type: TensorType::new(
                [logical_rows, logical_columns],
                Precision::F16,
                Layout {
                    order,
                    tiling: TensorTiling::replicated(1),
                    memory_class: MemoryClass::Ipu21Standard,
                },
            ),
            extents: vec![
                ShardExtent {
                    axis: 0,
                    start: 0,
                    logical_end: logical_rows,
                    physical_end: physical_rows,
                },
                ShardExtent {
                    axis: 1,
                    start: 0,
                    logical_end: logical_columns,
                    physical_end: physical_columns,
                },
            ],
            definition: ShardDefinition::Staging,
        })
    }
}
