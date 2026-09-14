//! Dispatch distributed compute to its family. Elementwise/local kernels bind
//! resident operands here; products and sums own their local construction.

use super::*;
use crate::{Compute, OperandWindow};

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
                let inputs_by_tile = self.inputs_by_tile(&operation.inputs)?;
                for output in outputs {
                    let block = &self.shards[output.index() as usize];
                    if block
                        .extents
                        .iter()
                        .any(|axis| axis.physical_end == axis.start)
                    {
                        continue;
                    }
                    let tile = block.tile;
                    self.bind_compute_aliases(
                        operation,
                        output,
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
                        .map(|(tiles, window)| {
                            let source = tiles[usize::from(tile)]
                                .iter()
                                .copied()
                                .find(|&source| {
                                    if donate_cast {
                                        return self.shards[source.index() as usize].extents
                                            == self.shards[output.index() as usize].extents;
                                    }
                                    // A tile can own several row fragments. Equal-shape
                                    // pointwise operands must match this output's coordinates.
                                    let same_shape = self.shards[source.index() as usize]
                                        .tensor_type
                                        .shape
                                        == self.shards[output.index() as usize].tensor_type.shape;
                                    !(same_shape
                                        || matches!(
                                            kernel,
                                            TileKernelSpec::Gelu
                                                | TileKernelSpec::Add
                                                | TileKernelSpec::BiasGelu
                                                | TileKernelSpec::AddLayerNorm
                                        ))
                                        || self.broadcast_view(source, output).is_some()
                                })
                                .ok_or(ExpansionError::InvalidOperatorPlan)?;
                            if matches!(
                                kernel,
                                TileKernelSpec::Add
                                    | TileKernelSpec::BiasGelu
                                    | TileKernelSpec::AddLayerNorm
                            ) && window.0.is_empty()
                            {
                                self.broadcast_view(source, output)
                                    .ok_or(ExpansionError::InvalidOperatorPlan)
                            } else {
                                self.window(source, window)
                            }
                        })
                        .collect::<ExpansionResult<Vec<_>>>()?;
                    let mut run = self.kernel_run(
                        operation_provenance(operation),
                        kernel.clone(),
                        inputs
                            .into_iter()
                            .map(|view| KernelOperand { views: vec![view] })
                            .collect(),
                        self.full_view(output),
                    )?;
                    for &value in operation.results.iter().skip(1) {
                        let shard = self.local_shard(value, tile)?;
                        let view = self.full_view(shard);
                        let format = self.shards[shard.index() as usize]
                            .tensor_type
                            .format
                            .clone();
                        Arc::make_mut(&mut run.metadata)
                            .requirements
                            .additional_outputs
                            .push(crate::KernelAccess::new(format, 8));
                        run.additional_outputs.push(view);
                    }
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
    pub(super) fn inputs_by_tile(
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
        operation: &MidOperation,
        output: BlockValueId,
        aliases: &[(usize, usize)],
        inputs_by_tile: &[Vec<Vec<BlockValueId>>],
        offset: i32,
    ) -> ExpansionResult<()> {
        let tile = self.shards[output.index() as usize].tile;
        for &(result, input) in aliases {
            let target = if result == 0 {
                output
            } else {
                self.local_shard(operation.results[result], tile)?
            };
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
