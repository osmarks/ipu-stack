//! Consumer-sized materialization of deferred conversions and views.

use super::*;

pub(super) struct DeferredValue {
    pub(super) transform: AxisFactorView,
    pub(super) shards: Vec<BlockValueId>,
}

impl TileGraphBuilder {
    pub(super) fn deferred_root(&self, mut value: MidValueId) -> Option<MidValueId> {
        let mut remaining = self.deferred_conversions.len().saturating_add(1);
        while !self.deferred_values.contains_key(&value) {
            value = *self.deferred_conversions.get(&value)?;
            remaining = remaining.checked_sub(1)?;
        }
        Some(value)
    }

    pub(super) fn has_deferred_value(&self, value: MidValueId) -> bool {
        self.deferred_root(value).is_some()
    }

    pub(super) fn deferred_supports_physical_exchange(
        &self,
        value: MidValueId,
        destination: BlockValueId,
    ) -> bool {
        let Some(root) = self.deferred_root(value) else {
            return false;
        };
        let Some(source) = self
            .deferred_values
            .get(&root)
            .and_then(|deferred| deferred.shards.first())
        else {
            return false;
        };
        self.value_shards(value)
            .ok()
            .and_then(|shards| shards.first())
            .and_then(|shard| {
                self.shards[shard.index() as usize]
                    .tensor_type
                    .shape
                    .0
                    .last()
            })
            .is_some_and(|width| width.is_multiple_of(2))
            && self.shards[source.index() as usize]
                .tensor_type
                .format
                .supports_f16_micro_panel_exchange(
                    &self.shards[destination.index() as usize].tensor_type.format,
                )
    }

    pub(super) fn build_deferred_output(
        &mut self,
        operation: &MidOperation,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<bool> {
        let Some(offered) = operation
            .operator_plan()
            .and_then(|plan| plan.deferred_output)
        else {
            return Ok(false);
        };
        let [result] = operation.results.as_slice() else {
            return Ok(false);
        };
        let Some(source) = operation.inputs.get(offered.source_input) else {
            return Ok(false);
        };
        let source_type = &self.shards[self.value_shards(*source)?[0].index() as usize].tensor_type;
        let result_type = &self.shards[self.value_shards(*result)?[0].index() as usize].tensor_type;
        if offered.transform.output_shape(&source_type.shape).as_ref() != Some(&result_type.shape) {
            return Err(ExpansionError::InvalidOperatorPlan);
        }
        let source_shards = self.value_shards(*source)?.to_vec();
        let source_format = &self.shards[source_shards[0].index() as usize]
            .tensor_type
            .format;
        let result_format = &self.shards[self.value_shards(*result)?[0].index() as usize]
            .tensor_type
            .format;
        let direct_panel_exchange = source_format.supports_f16_micro_panel_exchange(result_format);
        tracing::debug!(
            source = ?operation.source.map(OperationId::index),
            source_order = ?source_format.layout.order,
            result_order = ?result_format.layout.order,
            direct_panel_exchange,
            "selected deferred-output storage"
        );
        let staging_shards = if direct_panel_exchange
            || !matches!(
                source_format.layout.order,
                ElementOrder::Amp(AmpOrder::Output | AmpOrder::TransposedLeft)
            ) {
            source_shards
        } else {
            let Some(staging) = self.unpack_amp_to_row_major(
                *source,
                WorkProvenance {
                    operation: operation.source,
                    value: Some(*result),
                    reason: WorkReason::OperatorKernel,
                },
                tiles,
            )?
            else {
                return Ok(false);
            };
            staging
        };
        self.deferred_values.insert(
            *result,
            DeferredValue {
                transform: offered.transform,
                shards: staging_shards,
            },
        );
        for shard in self.value_shards(*result)?.to_vec() {
            self.shards[shard.index() as usize].definition = ShardDefinition::Unmaterialized;
        }
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn deferred_panel_mappings(
        &self,
        value: MidValueId,
        stream: u32,
        row_start: u32,
        rows: u32,
        column_start: u32,
        columns: u32,
        destination: BlockValueId,
    ) -> ExpansionResult<Vec<(ShardView, ShardView)>> {
        let deferred_root = self
            .deferred_root(value)
            .ok_or(ExpansionError::InvalidOperatorPlan)?;
        let deferred = self
            .deferred_values
            .get(&deferred_root)
            .ok_or(ExpansionError::InvalidOperatorPlan)?;
        let deferred_shards = deferred.shards.clone();
        let logical_type = &self.shards[self.value_shards(value)?[0].index() as usize].tensor_type;
        let source_type = &self.shards[deferred.shards[0].index() as usize].tensor_type;
        let logical_target = [
            (stream, stream + 1),
            (row_start, row_start + rows),
            (column_start, column_start + columns),
        ];
        let mapping = deferred
            .transform
            .map_slice(&source_type.shape, &logical_type.shape, &logical_target)
            .ok_or(ExpansionError::InvalidOperatorPlan)?;
        let target = mapping
            .iter()
            .copied()
            .enumerate()
            .map(|(axis, (start, end))| {
                Ok(ShardExtent {
                    axis: u16::try_from(axis).map_err(|_| ExpansionError::IdOverflow)?,
                    start,
                    logical_end: end,
                    physical_end: end,
                })
            })
            .collect::<ExpansionResult<Vec<_>>>()?;
        let destination_tile = self.shards[destination.index() as usize].tile;
        let mut covered = 0u64;
        let mut mappings = Vec::new();
        for (source_extents, source) in
            self.intersecting_shard_set(&deferred_shards, &target, destination_tile)
        {
            let destination_extents = (0..mapping.len())
                .filter(|&axis| axis != deferred.transform.merge_axis)
                .enumerate()
                .map(|(destination_axis, source_axis)| {
                    let source = source_extents
                        .get(source_axis)
                        .ok_or(ExpansionError::InvalidOperatorPlan)?;
                    let base = target
                        .get(source_axis)
                        .ok_or(ExpansionError::InvalidOperatorPlan)?
                        .start;
                    Ok(ShardExtent {
                        axis: u16::try_from(destination_axis)
                            .map_err(|_| ExpansionError::IdOverflow)?,
                        start: source.start - base,
                        logical_end: source.logical_end - base,
                        physical_end: source.logical_end - base,
                    })
                })
                .collect::<ExpansionResult<Vec<_>>>()?;
            covered = covered.saturating_add(
                u64::from(source_extents[1].logical_end - source_extents[1].start)
                    * u64::from(source_extents[2].logical_end - source_extents[2].start),
            );
            let source_view = ShardView {
                shard: source,
                extents: source_extents,
            };
            let destination_view = ShardView {
                shard: destination,
                extents: destination_extents,
            };
            mappings.push((source_view, destination_view));
        }
        if covered != u64::from(rows) * u64::from(columns) {
            return Err(ExpansionError::InvalidOperatorPlan);
        }
        Ok(mappings)
    }

    pub(super) fn prepare_deferred_panel(
        &mut self,
        value: MidValueId,
        stream: u32,
        row_start: u32,
        rows: u32,
        column_start: u32,
        columns: u32,
        destination: BlockValueId,
        provenance: WorkProvenance,
        batch: &mut MaterializationBatch,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let mut mappings = self.deferred_panel_mappings(
            value,
            stream,
            row_start,
            rows,
            column_start,
            columns,
            destination,
        )?;
        let order = if self.deferred_supports_physical_exchange(value, destination) {
            mappings = self
                .f16_micro_panel_mappings(mappings)?
                .ok_or(ExpansionError::InvalidOperatorPlan)?;
            CopyOrder::Physical
        } else {
            CopyOrder::Semantic
        };
        self.prepare_mapped_views(mappings, order, order, provenance, batch, tiles)
    }
}
