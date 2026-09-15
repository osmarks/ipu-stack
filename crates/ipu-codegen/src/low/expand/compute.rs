//! Bind executable mid computations to resident tensor regions. Algorithm
//! construction and iteration are complete before this stage.

use super::*;
use crate::OperandIndexing;
use crate::mid::MidOperationKind;
use crate::tensor::Broadcast;

impl TileGraphBuilder {
    pub(super) fn build_compute(
        &mut self,
        operation: &MidOperation,
        body: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let kernel = &operation.kind;

        let operands = &operation.operands;
        let output_aliases = &operation.output_aliases;
        let output = *operation
            .results
            .first()
            .ok_or(ExpansionError::ResultArity)?;
        let outputs = self.allocation_shards(output)?;
        let inputs_by_tile = self.views_by_tile(&operation.inputs)?;
        let results_by_tile = self.views_by_tile(&operation.results)?;
        // Results share the invocation distribution, even when their
        // tensor ranks differ (for example residuals and row statistics).
        if results_by_tile.iter().any(|result| {
            result
                .iter()
                .zip(&results_by_tile[0])
                .any(|(a, b)| a.len() != b.len())
        }) {
            return Err(ExpansionError::ResultArity);
        }
        let mut next_fragment = vec![0; usize::from(self.tile_count)];
        for output in outputs {
            let block = &self.shards[output.index() as usize];
            let tile = block.tile;
            let ordinal = next_fragment[usize::from(tile)];
            next_fragment[usize::from(tile)] += 1;
            if block
                .extents
                .iter()
                .any(|axis| axis.physical_end == axis.start)
            {
                continue;
            }
            let results = results_by_tile
                .iter()
                .map(|tiles| tiles[usize::from(tile)][ordinal].shard)
                .collect::<Vec<_>>();
            self.bind_compute_aliases(&results, output_aliases, &inputs_by_tile)?;
            let inputs = inputs_by_tile
                .iter()
                .zip(operands)
                .zip(&operation.inputs)
                .map(|((tiles, indexing), value)| {
                    let resident = &tiles[usize::from(tile)];
                    match indexing {
                        OperandIndexing::Elementwise { result } => {
                            let output =
                                *results.get(*result).ok_or(ExpansionError::ResultArity)?;
                            resident
                                .iter()
                                .find_map(|source| {
                                    self.elementwise_view(
                                        source,
                                        &self.logical_values[value.index() as usize]
                                            .tensor_type
                                            .shape,
                                        output,
                                    )
                                })
                                .ok_or(ExpansionError::InvalidOperatorPlan)
                        }
                        OperandIndexing::Local(window) | OperandIndexing::Fragment(window) => {
                            let index = if resident.len() == 1 { 0 } else { ordinal };
                            if resident.len() != 1
                                && resident.len() != results_by_tile[0][usize::from(tile)].len()
                            {
                                return Err(ExpansionError::InvalidOperatorPlan);
                            }
                            let source = resident
                                .get(index)
                                .ok_or(ExpansionError::InvalidOperatorPlan)?;
                            Ok(ShardView {
                                shard: source.shard,
                                extents: window
                                    .select(
                                        &source.extents,
                                        matches!(indexing, OperandIndexing::Fragment(_)),
                                    )
                                    .ok_or(ExpansionError::InvalidOperatorPlan)?,
                            })
                        }
                    }
                })
                .collect::<ExpansionResult<Vec<_>>>()?;
            let results = results
                .into_iter()
                .enumerate()
                .map(|(index, shard)| {
                    let mut view = self.full_view(shard);
                    if let Some(window) = operation.output_windows.get(index) {
                        view.extents = window
                            .select(&view.extents, true)
                            .ok_or(ExpansionError::InvalidOperatorPlan)?;
                    }
                    Ok(view)
                })
                .collect::<ExpansionResult<Vec<_>>>()?;
            if inputs
                .iter()
                .chain(&results)
                .any(|v| v.extents.iter().any(|e| e.start == e.physical_end))
            {
                continue;
            }
            let mut kind = kernel.clone();
            if let MidOperationKind::Gemm {
                axes,
                inner_block,
                output_columns,
                ..
            } = &mut kind
            {
                let li = axes.left_inner.resolve(inputs[0].extents.len())?;
                let oc = axes.output_column.resolve(results[0].extents.len())?;
                *inner_block = inputs[0].extents[li].physical_end - inputs[0].extents[li].start;
                *output_columns =
                    results[0].extents[oc].physical_end - results[0].extents[oc].start;
            }
            let run = self.bind_kernel(operation_provenance(operation), kind, inputs, results)?;
            self.append_kernel(body, tile, run)?;
        }
        Ok(())
    }

    pub(super) fn elementwise_view(
        &self,
        source: &ShardView,
        shape: &crate::TensorShape,
        output: BlockValueId,
    ) -> Option<ShardView> {
        let output = &self.shards[output.index() as usize];
        let indexing = Broadcast::new(&shape.0, &output.tensor_type.shape.0)?;
        let mut view = source.clone();
        for (axis, extent) in view.extents.iter_mut().enumerate() {
            if indexing.is_broadcast(axis) {
                if extent.start != 0 || extent.logical_end == 0 {
                    return None;
                }
                extent.logical_end = 1;
                extent.physical_end = 1;
            } else {
                let required = output.extents[indexing.output_axis(axis)];
                if extent.start > required.start || extent.logical_end < required.logical_end {
                    return None;
                }
                // An identity selection retains the complete physical panel.
                // A proper logical subregion must instead keep the backing
                // strides and let family binding check whether it is usable.
                if extent.start != required.start || extent.logical_end != required.logical_end {
                    extent.start = required.start;
                    extent.logical_end = required.logical_end;
                    extent.physical_end = required.logical_end;
                }
            }
        }
        Some(view)
    }

    /// Preserve per-tile fragment order without rescanning all shards for each
    /// output. Lifetime-only dependencies follow the explicit operands.
    pub(super) fn views_by_tile(
        &self,
        inputs: &[MidValueId],
    ) -> ExpansionResult<Vec<Vec<Vec<ShardView>>>> {
        inputs
            .iter()
            .map(|&value| {
                let mut tiles = vec![Vec::new(); usize::from(self.tile_count)];
                for source in self.value_views(value)? {
                    tiles[usize::from(self.shards[source.shard.index() as usize].tile)]
                        .push(source.clone());
                }
                Ok(tiles)
            })
            .collect()
    }

    pub(super) fn bind_compute_aliases(
        &mut self,
        outputs: &[BlockValueId],
        aliases: &[(usize, usize)],
        inputs_by_tile: &[Vec<Vec<ShardView>>],
    ) -> ExpansionResult<()> {
        for &(result, input) in aliases {
            let target = *outputs.get(result).ok_or(ExpansionError::ResultArity)?;
            let tile = self.shards[target.index() as usize].tile;
            let previous = inputs_by_tile[input][usize::from(tile)]
                .iter()
                .find(|source| {
                    source.extents == self.shards[target.index() as usize].extents
                        && source.extents == self.shards[source.shard.index() as usize].extents
                })
                .map(|source| source.shard)
                .ok_or(ExpansionError::InvalidOperatorPlan)?;
            self.shards[target.index() as usize].definition =
                ShardDefinition::WritableAlias(previous);
        }
        Ok(())
    }
}
