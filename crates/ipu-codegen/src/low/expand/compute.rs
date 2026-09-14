//! Dispatch distributed compute to its family. Elementwise/local kernels bind
//! resident operands here; products and sums own their local construction.

use super::*;
use crate::tensor::Broadcast;
use crate::{Compute, OperandIndexing, OperandWindow};

impl TileGraphBuilder {
    pub(super) fn build_compute(
        &mut self,
        operation: &MidOperation,
        compute: &Compute,
        body: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        match compute {
            Compute::Product(product) => self.build_product(operation, product, body),
            Compute::Sum { axis, staging } => {
                let mut batch = reduce::SumBatch::default();
                self.prepare_sum(operation, usize::from(*axis), *staging, &mut batch)?;
                self.append_sum_batch(batch, operation_provenance(operation), body)
            }
            Compute::Kernel {
                kernel,
                operands,
                output_aliases,
            } => {
                let donate_cast = matches!(
                    kernel,
                    TileKernelSpec::Cast {
                        from: Precision::F16,
                        to: Precision::F8F143 { .. }
                    }
                ) && output_aliases == &[(0, 0)];
                let output = *operation
                    .results
                    .first()
                    .ok_or(ExpansionError::ResultArity)?;
                let outputs = self.value_shards(output)?.to_vec();
                let inputs_by_tile = self.shards_by_tile(&operation.inputs)?;
                let results_by_tile = self.shards_by_tile(&operation.results)?;
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
                        .map(|tiles| tiles[usize::from(tile)][ordinal])
                        .collect::<Vec<_>>();
                    self.bind_compute_aliases(
                        &results,
                        output_aliases,
                        &inputs_by_tile,
                        if donate_cast {
                            -(crate::kernel::cast::CAST_PREFIX_BYTES as i32)
                        } else {
                            0
                        },
                    )?;
                    let inputs = inputs_by_tile
                        .iter()
                        .zip(operands)
                        .map(|(tiles, indexing)| {
                            let resident = &tiles[usize::from(tile)];
                            match indexing {
                                OperandIndexing::Elementwise { result } => {
                                    let output =
                                        *results.get(*result).ok_or(ExpansionError::ResultArity)?;
                                    resident
                                        .iter()
                                        .find_map(|&source| self.elementwise_view(source, output))
                                        .ok_or(ExpansionError::InvalidOperatorPlan)
                                }
                                OperandIndexing::Local(window) => {
                                    let index = if resident.len() == 1 { 0 } else { ordinal };
                                    if resident.len() != 1
                                        && resident.len()
                                            != results_by_tile[0][usize::from(tile)].len()
                                    {
                                        return Err(ExpansionError::InvalidOperatorPlan);
                                    }
                                    let source = *resident
                                        .get(index)
                                        .ok_or(ExpansionError::InvalidOperatorPlan)?;
                                    self.window(source, window)
                                }
                            }
                        })
                        .collect::<ExpansionResult<Vec<_>>>()?;
                    let results = results
                        .into_iter()
                        .map(|shard| self.full_view(shard))
                        .collect();
                    let run = self.kernel_run(
                        operation_provenance(operation),
                        kernel.clone(),
                        inputs,
                        results,
                    )?;
                    if donate_cast {
                        self.append_in_place_cast(body, tile, run)?;
                    } else {
                        self.append_kernel(body, tile, run)?;
                    }
                }
                Ok(())
            }
        }
    }

    pub(super) fn elementwise_view(
        &self,
        source: BlockValueId,
        output: BlockValueId,
    ) -> Option<ShardView> {
        let input = &self.shards[source.index() as usize];
        let output = &self.shards[output.index() as usize];
        let indexing = Broadcast::new(&input.tensor_type.shape.0, &output.tensor_type.shape.0)?;
        let mut view = self.full_view(source);
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

    pub(super) fn window(
        &self,
        source: BlockValueId,
        window: &OperandWindow,
    ) -> ExpansionResult<ShardView> {
        let ranges = window
            .0
            .iter()
            .map(|&(axis, start, end)| (usize::from(axis), start, end))
            .collect::<Vec<_>>();
        self.narrow_view(source, &ranges)
    }

    /// Preserve per-tile fragment order without rescanning all shards for each
    /// output. Lifetime-only dependencies follow the explicit operands.
    pub(super) fn shards_by_tile(
        &self,
        inputs: &[MidValueId],
    ) -> ExpansionResult<Vec<Vec<Vec<BlockValueId>>>> {
        inputs
            .iter()
            .map(|&value| {
                let mut tiles = vec![Vec::new(); usize::from(self.tile_count)];
                for &source in self.value_shards(value)? {
                    tiles[usize::from(self.shards[source.index() as usize].tile)].push(source);
                }
                Ok(tiles)
            })
            .collect()
    }

    pub(super) fn bind_compute_aliases(
        &mut self,
        outputs: &[BlockValueId],
        aliases: &[(usize, usize)],
        inputs_by_tile: &[Vec<Vec<BlockValueId>>],
        offset: i32,
    ) -> ExpansionResult<()> {
        for &(result, input) in aliases {
            let target = *outputs.get(result).ok_or(ExpansionError::ResultArity)?;
            let tile = self.shards[target.index() as usize].tile;
            let previous = inputs_by_tile[input][usize::from(tile)]
                .iter()
                .copied()
                .find(|&source| {
                    self.shards[source.index() as usize].extents
                        == self.shards[target.index() as usize].extents
                })
                .ok_or(ExpansionError::InvalidOperatorPlan)?;
            self.shards[target.index() as usize].definition = if offset == 0 {
                ShardDefinition::WritableAlias(previous)
            } else {
                ShardDefinition::ShiftedAlias {
                    source: previous,
                    offset,
                }
            };
        }
        Ok(())
    }
}
