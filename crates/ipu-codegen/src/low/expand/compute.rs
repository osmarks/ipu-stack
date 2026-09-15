//! Bind executable mid computations to resident tensor regions. Algorithm
//! construction and iteration are complete before this stage.

use super::*;
use crate::mid::MidOperationKind;
use crate::tensor::Broadcast;
use crate::{OperandIndexing, OperandWindow};

/// Writable aliases need complete allocations, rather than borrowed slices.
pub(super) fn allocation_inputs<'a>(
    operation: &'a MidOperation,
) -> impl Iterator<Item = &'a MidValueId> {
    operation
        .output_aliases
        .iter()
        .map(|&(_, input)| input)
        .filter_map(|index| operation.inputs.get(index))
}

impl TileGraphBuilder {
    pub(super) fn build_compute(
        &mut self,
        operation: &MidOperation,
        body: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let kernel = &operation.kind;

        let operands = &operation.operands;
        let output_aliases = &operation.output_aliases;
        let donate_cast = matches!(
            kernel,
            MidOperationKind::Cast {
                from: Precision::F16,
                to: Precision::F8F143 { .. }
            }
        ) && output_aliases == &[(0, 0)];
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
                            if matches!(indexing, OperandIndexing::Fragment(_)) {
                                self.fragment_window(source, window)
                            } else {
                                self.window(source, window)
                            }
                        }
                    }
                })
                .collect::<ExpansionResult<Vec<_>>>()?;
            let results = results
                .into_iter()
                .enumerate()
                .map(|(index, shard)| {
                    let view = self.full_view(shard);
                    operation
                        .output_windows
                        .get(index)
                        .map_or(Ok(view.clone()), |window| {
                            self.fragment_window(&view, window)
                        })
                })
                .collect::<ExpansionResult<Vec<_>>>()?;
            if inputs
                .iter()
                .chain(&results)
                .any(|v| v.extents.iter().any(|e| e.start == e.physical_end))
            {
                continue;
            }
            if donate_cast {
                let [input]: [ShardView; 1] = inputs
                    .try_into()
                    .map_err(|_| ExpansionError::InvalidOperatorPlan)?;
                let [output]: [ShardView; 1] = results
                    .try_into()
                    .map_err(|_| ExpansionError::ResultArity)?;
                self.build_shifted_cast(
                    body,
                    tile,
                    operation_provenance(operation),
                    kernel.clone(),
                    input,
                    output,
                )?;
            } else {
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
                let run =
                    self.bind_kernel(operation_provenance(operation), kind, inputs, results)?;
                self.append_kernel(body, tile, run)?;
            }
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

    pub(super) fn window(
        &self,
        source: &ShardView,
        window: &OperandWindow,
    ) -> ExpansionResult<ShardView> {
        let ranges = window
            .0
            .iter()
            .map(|&(axis, start, end)| (usize::from(axis), start, end))
            .collect::<Vec<_>>();
        self.narrow_view(source, &ranges)
    }

    fn fragment_window(
        &self,
        source: &ShardView,
        window: &OperandWindow,
    ) -> ExpansionResult<ShardView> {
        let mut result = source.clone();
        for &(axis, start, end) in &window.0 {
            let e = result
                .extents
                .get_mut(axis as usize)
                .ok_or(ExpansionError::InvalidOperatorPlan)?;
            let origin = e.start;
            e.start = origin.saturating_add(start).min(e.physical_end);
            e.physical_end = origin.saturating_add(end).min(e.physical_end).max(e.start);
            e.logical_end = e.logical_end.min(e.physical_end).max(e.start);
        }
        Ok(result)
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
        offset: i32,
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
