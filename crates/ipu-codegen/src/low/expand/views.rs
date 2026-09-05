//! Logical view mapping and physical micro-panel decomposition.

use super::*;

impl TileGraphBuilder {
    pub(super) fn build_view(
        &mut self,
        operation: &MidOperation,
        operator: &crate::MidOperator,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let crate::MidOperator::View(view) = operator else {
            return Err(ExpansionError::InvalidOperatorPlan);
        };
        let [input] = operation.inputs.as_slice() else {
            return Err(ExpansionError::InvalidOperatorPlan);
        };
        let [result] = operation.results.as_slice() else {
            return Err(ExpansionError::ResultArity);
        };
        let input_type = self
            .value_shards(*input)?
            .first()
            .map(|shard| self.shards[shard.index() as usize].tensor_type.clone())
            .ok_or(ExpansionError::InvalidOperatorPlan)?;
        let output_shards = self.value_shards(*result)?.to_vec();
        let output_type = output_shards
            .first()
            .map(|shard| self.shards[shard.index() as usize].tensor_type.clone())
            .ok_or(ExpansionError::InvalidOperatorPlan)?;
        let original_sources = self.value_shards(*input)?.to_vec();
        let direct_panel_exchange = input_type
            .format
            .supports_f16_micro_panel_exchange(&output_type.format);
        tracing::debug!(
            source = ?operation.source.map(OperationId::index),
            source_order = ?input_type.format.layout.order,
            destination_order = ?output_type.format.layout.order,
            direct_panel_exchange,
            "selected view exchange strategy"
        );
        if direct_panel_exchange {
            let mappings = self.view_mappings(&original_sources, &output_shards, *view)?;
            if let Some(mappings) = self.f16_micro_panel_mappings(mappings)? {
                tracing::info!(
                    source = ?operation.source.map(OperationId::index),
                    mappings = mappings.len(),
                    "lowering view as physical micro-panel exchange"
                );
                return self.build_mapped_views(
                    mappings,
                    CopyOrder::Physical,
                    CopyOrder::Physical,
                    WorkProvenance {
                        operation: operation.source,
                        value: Some(*result),
                        reason: WorkReason::OperatorInputs,
                    },
                    tiles,
                );
            }
        }

        let source_shards = if matches!(
            input_type.format.layout.order,
            ElementOrder::Amp(AmpOrder::Output | AmpOrder::TransposedLeft)
        ) {
            self.unpack_amp_to_row_major(
                *input,
                WorkProvenance {
                    operation: operation.source,
                    value: Some(*result),
                    reason: WorkReason::OperatorKernel,
                },
                tiles,
            )?
            .ok_or(ExpansionError::InvalidOperatorPlan)?
        } else {
            self.value_shards(*input)?.to_vec()
        };

        let mappings = self.view_mappings(&source_shards, &output_shards, *view)?;
        self.build_mapped_views(
            mappings,
            CopyOrder::Semantic,
            CopyOrder::Semantic,
            WorkProvenance {
                operation: operation.source,
                value: Some(*result),
                reason: WorkReason::OperatorInputs,
            },
            tiles,
        )
    }

    pub(super) fn view_mappings(
        &self,
        source_shards: &[BlockValueId],
        output_shards: &[BlockValueId],
        view: AxisFactorView,
    ) -> ExpansionResult<Vec<(ShardView, ShardView)>> {
        let mut mappings = Vec::new();
        for &output in output_shards {
            let output_extents = self.shards[output.index() as usize].extents.clone();
            let tile = self.shards[output.index() as usize].tile;
            let source_shape = &self.shards[source_shards[0].index() as usize]
                .tensor_type
                .shape;
            let output_shape = &self.shards[output.index() as usize].tensor_type.shape;
            if view.output_shape(source_shape).as_ref() != Some(output_shape) {
                return Err(ExpansionError::InvalidOperatorPlan);
            }
            let split = view.split_axis;
            let merge = view.merge_axis;
            let part_width = output_shape.0[split];
            for stream in output_extents[merge].start..output_extents[merge].logical_end {
                let mut stream_extents = output_extents.clone();
                stream_extents[merge].start = stream;
                stream_extents[merge].logical_end = stream + 1;
                stream_extents[merge].physical_end = stream + 1;
                let (target, column_base) = view
                    .source_extents(source_shape, output_shape, &stream_extents)
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
                    let source_view = ShardView {
                        shard: source,
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

    /// Splits corresponding views at each allocation's F16 micro-panel
    /// boundaries. Within every resulting rectangle the source and
    /// destination have identical physical traversal, even when their outer
    /// panel sequence and tile ownership differ.
    pub(super) fn f16_micro_panel_mappings(
        &self,
        mappings: Vec<(ShardView, ShardView)>,
    ) -> ExpansionResult<Option<Vec<(ShardView, ShardView)>>> {
        let mut split = Vec::new();
        for (source, destination) in mappings {
            let source_shard = &self.shards[source.shard.index() as usize];
            let destination_shard = &self.shards[destination.shard.index() as usize];
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
