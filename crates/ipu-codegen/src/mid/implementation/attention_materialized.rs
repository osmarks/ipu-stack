//! Materialized score/probability attention construction.

use super::*;

impl BlockBuilder {
    pub(super) fn append_materialized_attention_input(
        &mut self,
        operand: AttentionOperand,
        sources: &[BlockValueId],
        prepared: &[PreparedAttentionBlock],
        tasks: &[AttentionTask],
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> BlockBuildResult<()> {
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
    pub(super) fn build_materialized_attention(
        &mut self,
        operation: &MidOperation,
        query_key: TileKernelSpec,
        probability_value: TileKernelSpec,
        query_block_rows: u32,
        padded_key_rows: u32,
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
        if query_block_rows == 0
            || padded_key_rows == 0
            || !padded_key_rows.is_multiple_of(AMP_INNER_BLOCK)
            || self.has_deferred_value(*key) != self.has_deferred_value(*value)
        {
            return Err(BlockBuildError::InvalidOperatorPlan);
        }
        let key_shards = self.value_shards(*key)?.to_vec();
        let value_shards = self.value_shards(*value)?.to_vec();
        if key_shards.len() != value_shards.len() {
            return Err(BlockBuildError::InvalidOperatorPlan);
        }
        let key_rows = self.shards[key_shards[0].index() as usize]
            .tensor_type
            .shape
            .0[1];
        if key_rows == 0 || key_rows > padded_key_rows {
            return Err(BlockBuildError::InvalidOperatorPlan);
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
                self.kernel_run(
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
                )?,
            )?;
            self.append_kernel(
                tiles,
                task.tile,
                self.kernel_run(
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
                )?,
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
                self.kernel_run(
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
                )?,
            )?;
        }
        Ok(())
    }
}
