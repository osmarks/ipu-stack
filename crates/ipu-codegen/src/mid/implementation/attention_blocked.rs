//! Blocked attention iteration and stage construction.

use super::*;

impl BlockBuilder {
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
        tiles: &mut BlockRegion,
    ) -> BlockBuildResult<Vec<PreparedAttentionBlock>> {
        let blocks = key_rows.div_ceil(block_rows);
        let key_destinations = tasks.iter().fold(
            BTreeMap::<u32, Vec<BlockValueId>>::new(),
            |mut destinations, task| {
                destinations
                    .entry(task.head)
                    .or_default()
                    .push(task.key_staging);
                destinations
            },
        );
        let value_destinations = tasks.iter().fold(
            BTreeMap::<u32, Vec<BlockValueId>>::new(),
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
                u16::try_from(block_rows).map_err(|_| BlockBuildError::InvalidOperatorPlan)?;
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
    pub(super) fn build_blocked_attention(
        &mut self,
        operation: &MidOperation,
        query_key: TileKernelSpec,
        probability_value: TileKernelSpec,
        query_block_rows: u32,
        key_block_rows: u32,
        padded_query_dimension: u32,
        padded_value_dimension: u32,
        tiles: &mut BlockRegion,
    ) -> BlockBuildResult<()> {
        let [query, key, value] = operation.inputs.as_slice() else {
            return Err(BlockBuildError::InvalidOperatorPlan);
        };
        let [result] = operation.results.as_slice() else {
            return Err(BlockBuildError::ResultArity);
        };
        if key_block_rows != AMP_INNER_BLOCK || query_block_rows == 0 {
            return Err(BlockBuildError::InvalidOperatorPlan);
        }
        let key_shards = self.value_shards(*key)?.to_vec();
        let value_shards = self.value_shards(*value)?.to_vec();
        if key_shards.len() != value_shards.len() {
            return Err(BlockBuildError::InvalidOperatorPlan);
        }
        if self.has_deferred_value(*key) != self.has_deferred_value(*value) {
            return Err(BlockBuildError::InvalidOperatorPlan);
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
            .map_err(|_| BlockBuildError::IdOverflow)?;
        if blocks == 0 {
            return Err(BlockBuildError::InvalidOperatorPlan);
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
                u32::try_from(block).map_err(|_| BlockBuildError::IdOverflow)? * key_block_rows;
            let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
            let mut task_sources = Vec::with_capacity(tasks.len());
            if deferred_key_value {
                let prepared = prepared_blocks
                    .get(block)
                    .ok_or(BlockBuildError::InvalidOperatorPlan)?;
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
                    let source_matches = |candidate: &&BlockValueId| {
                        let shard = &self.shards[candidate.index() as usize];
                        shard.extents[0].start == task.head && shard.extents[1].start == block_start
                    };
                    let key_source = *key_shards
                        .iter()
                        .find(source_matches)
                        .ok_or(BlockBuildError::InvalidOperatorPlan)?;
                    let value_source = *value_shards
                        .iter()
                        .find(source_matches)
                        .ok_or(BlockBuildError::InvalidOperatorPlan)?;
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
                    self.kernel_run(
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
                    )?,
                )?;
                self.append_kernel(
                    tiles,
                    task.tile,
                    self.kernel_run(
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
                    )?,
                )?;
                let probability_view = self.narrow_view(task.weights, &[(1, 0, key_block_rows)])?;
                let block_value_view =
                    self.narrow_view(task.scratch, &[(1, 0, padded_value_dimension)])?;
                self.append_kernel(
                    tiles,
                    task.tile,
                    self.kernel_run(
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
                    )?,
                )?;
                self.append_kernel(
                    tiles,
                    task.tile,
                    self.kernel_run(
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
                    )?,
                )?;
            }
        }
        Ok(())
    }
}
