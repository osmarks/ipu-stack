//! Blocked and materialized attention tile phases.

use super::*;

pub(super) struct PreparedDistributedPanel {
    pub(super) panel: u32,
    pub(super) packed: BlockValueId,
    pub(super) tile: u16,
    pub(super) destinations: Vec<BlockValueId>,
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
    pub(super) query: BlockValueId,
    pub(super) output: BlockValueId,
    pub(super) scratch: BlockValueId,
    pub(super) weights: BlockValueId,
    pub(super) key_staging: BlockValueId,
    pub(super) value_staging: BlockValueId,
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

impl TileGraphBuilder {
    pub(super) fn build_attention_tasks(
        &mut self,
        query: MidValueId,
        result: MidValueId,
        shape: AttentionBufferShape,
    ) -> ExpansionResult<Vec<AttentionTask>> {
        let value_row_block = u16::try_from(shape.physical_staging_rows)
            .map_err(|_| ExpansionError::InvalidOperatorPlan)?;
        let outputs = self.value_shards(result)?.to_vec();
        let mut tasks = Vec::with_capacity(outputs.len());
        for output in outputs {
            let tile = self.shards[output.index() as usize].tile;
            let rank = self.shards[output.index() as usize].extents.len();
            if rank != 3 {
                return Err(ExpansionError::InvalidOperatorPlan);
            }
            let rows = self.shards[output.index() as usize].extents[rank - 2].physical_end
                - self.shards[output.index() as usize].extents[rank - 2].start;
            let value_dimension = *self.shards[output.index() as usize]
                .tensor_type
                .shape
                .0
                .last()
                .ok_or(ExpansionError::InvalidOperatorPlan)?;
            if rows == 0 || rows > shape.query_block_rows {
                return Err(ExpansionError::InvalidOperatorPlan);
            }
            let canonical_query = self.local_shard(query, tile)?;
            let query_dimension = *self.shards[canonical_query.index() as usize]
                .tensor_type
                .shape
                .0
                .last()
                .ok_or(ExpansionError::InvalidOperatorPlan)?;
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
                output,
                scratch,
                weights,
                key_staging,
                value_staging,
            });
        }
        Ok(tasks)
    }

    pub(super) fn push_attention_scratch(
        &mut self,
        tile: u16,
        rows: u32,
        columns: u32,
        precision: Precision,
        order: ElementOrder,
        memory_class: MemoryClass,
    ) -> ExpansionResult<BlockValueId> {
        self.push_shard(BlockValue {
            id: BlockValueId(0),
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

    pub(super) fn push_attention_buffer(
        &mut self,
        tile: u16,
        logical_rows: u32,
        physical_rows: u32,
        logical_columns: u32,
        physical_columns: u32,
        order: ElementOrder,
    ) -> ExpansionResult<BlockValueId> {
        self.push_shard(BlockValue {
            id: BlockValueId(0),
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
