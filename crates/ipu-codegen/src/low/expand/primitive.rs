//! Realize selected whole-device primitives. Distribution and intermediate
//! tensor storage have already been chosen in mid.

use super::*;
use crate::{OperandWindow, Primitive, ProductAxes};

impl TileGraphBuilder {
    pub(super) fn build_primitive(
        &mut self,
        operation: &MidOperation,
        primitive: &Primitive,
        body: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        match primitive {
            Primitive::Copy {
                mapping,
                reuse_local,
            } => self.copy_tensor(operation, mapping, *reuse_local, body),
            Primitive::Sum { axis, staging } => {
                let mut batch = reduce::SumBatch::default();
                self.prepare_sum(operation, usize::from(*axis), *staging, &mut batch)?;
                self.append_sum_batch(batch, operation_provenance(operation), body)
            }
            Primitive::Compute {
                kernel,
                operands,
                product,
                reuse_input,
            } => {
                let output = *operation
                    .results
                    .first()
                    .ok_or(ExpansionError::ResultArity)?;
                let outputs = self.value_shards(output)?.to_vec();
                // Preserve shard order within each tile, but avoid searching all tiles
                // again for every output shard (quadratic for whole-device operations).
                let inputs_by_tile = operation
                    .inputs
                    .iter()
                    .map(|&value| {
                        let mut tiles = vec![Vec::new(); usize::from(self.tile_count)];
                        for &source in self.value_shards(value)? {
                            tiles[usize::from(self.shards[source.index() as usize].tile)]
                                .push(source);
                        }
                        Ok(tiles)
                    })
                    .collect::<ExpansionResult<Vec<_>>>()?;
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
                    if let Some(index) = reuse_input {
                        let previous = inputs_by_tile[*index][usize::from(tile)]
                            .iter()
                            .copied()
                            .find(|&candidate| {
                                let candidate = &self.shards[candidate.index() as usize];
                                candidate.extents == block.extents
                            })
                            .ok_or(ExpansionError::InvalidOperatorPlan)?;
                        self.shards[output.index() as usize].definition =
                            ShardDefinition::WritableAlias(previous);
                    }
                    let inputs = inputs_by_tile
                        .iter()
                        .zip(operands)
                        .map(|(tiles, window)| {
                            let source = tiles[usize::from(tile)]
                                .iter()
                                .copied()
                                .find(|&source| {
                                    !matches!(
                                        kernel,
                                        TileKernelSpec::Gelu
                                            | TileKernelSpec::Add
                                            | TileKernelSpec::BiasGelu
                                            | TileKernelSpec::AddLayerNorm
                                    ) || self.broadcast_view(source, output).is_some()
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
                    if let Some(axes) = product {
                        self.cached_product_calls(
                            operation_provenance(operation),
                            tile,
                            kernel,
                            &inputs,
                            output,
                            *axes,
                            body,
                        )?;
                    } else {
                        let run = self.kernel_run(
                            operation_provenance(operation),
                            kernel.clone(),
                            inputs
                                .into_iter()
                                .map(|view| KernelOperand { views: vec![view] })
                                .collect(),
                            self.full_view(output),
                        )?;
                        self.append_kernel(body, tile, run)?;
                    }
                }
                Ok(())
            }
        }
    }

    fn window(&self, source: BlockValueId, window: &OperandWindow) -> ExpansionResult<ShardView> {
        let ranges = window
            .0
            .iter()
            .map(|&(axis, start, end)| (usize::from(axis), start, end))
            .collect::<Vec<_>>();
        self.narrow_view(source, &ranges)
    }

    pub(super) fn product_calls(
        &mut self,
        provenance: WorkProvenance,
        tile: u16,
        kernel: &TileKernelSpec,
        inputs: &[ShardView],
        output: BlockValueId,
        axes: ProductAxes,
        body: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let [left, right] = inputs else {
            return Err(ExpansionError::InvalidOperatorPlan);
        };
        let TileKernelSpec::Gemm {
            inner_block,
            output_columns,
            mode,
            ..
        } = kernel
        else {
            return Err(ExpansionError::InvalidOperatorPlan);
        };
        let left_inner = axes.left_inner.resolve(left.extents.len())?;
        let right_inner = axes.right_inner.resolve(right.extents.len())?;
        let output_column = axes
            .output_column
            .resolve(self.shards[output.index() as usize].extents.len())?;
        let right_column = if right_inner == right.extents.len() - 1 {
            right_inner - 1
        } else {
            right_inner + 1
        };
        let left_row = if left_inner == left.extents.len() - 1 {
            left_inner - 1
        } else {
            left_inner + 1
        };
        let output_row = if output_column == self.shards[output.index() as usize].extents.len() - 1
        {
            output_column - 1
        } else {
            output_column + 1
        };
        let bounds = |extent: ShardExtent| (extent.start, extent.physical_end);
        if bounds(left.extents[left_row])
            != bounds(self.shards[output.index() as usize].extents[output_row])
            || bounds(right.extents[right_column])
                != bounds(self.shards[output.index() as usize].extents[output_column])
        {
            return Err(ExpansionError::InvalidOperatorPlan);
        }
        let inner = left.extents[left_inner].physical_end - left.extents[left_inner].start;
        if inner != right.extents[right_inner].physical_end - right.extents[right_inner].start
            || *inner_block == 0
            || *output_columns == 0
        {
            return Err(ExpansionError::InvalidOperatorPlan);
        }
        let columns = self.shards[output.index() as usize].extents[output_column];
        let group = self.shards[output.index() as usize]
            .tensor_type
            .format
            .layout
            .order
            .gemm_output_group();
        let column_step = group.map_or(*output_columns, |group| (*output_columns).min(group));
        for column in (columns.start..columns.physical_end).step_by(column_step as usize) {
            let column_end = (column + column_step).min(columns.physical_end);
            for k in (0..inner).step_by(*inner_block as usize) {
                let width = (inner - k).min(*inner_block);
                let l = self.narrow_view(
                    left.shard,
                    &[(
                        left_inner,
                        left.extents[left_inner].start + k,
                        left.extents[left_inner].start + k + width,
                    )],
                )?;
                let r = self.narrow_view(
                    right.shard,
                    &[
                        (
                            right_inner,
                            right.extents[right_inner].start + k,
                            right.extents[right_inner].start + k + width,
                        ),
                        (right_column, column, column_end),
                    ],
                )?;
                let destination =
                    self.narrow_view(output, &[(output_column, column, column_end)])?;
                let mut kernel = kernel.clone();
                if let TileKernelSpec::Gemm {
                    mode: call_mode,
                    inner_block,
                    output_columns,
                    weights,
                    ..
                } = &mut kernel
                {
                    *call_mode = if k == 0 {
                        *mode
                    } else {
                        crate::GemmKernelMode::Accumulate
                    };
                    *inner_block = width;
                    *output_columns = column_end - column;
                    *weights = if self.shards[right.shard.index() as usize]
                        .tensor_type
                        .format
                        .layout
                        .memory_class
                        == MemoryClass::Ipu21Interleaved
                    {
                        crate::GemmWeightLoad::Interleaved
                    } else {
                        crate::GemmWeightLoad::Standard
                    };
                }
                let mut run = self.kernel_run(
                    provenance,
                    kernel,
                    vec![
                        KernelOperand { views: vec![l] },
                        KernelOperand { views: vec![r] },
                    ],
                    destination,
                )?;
                let size = |e: ShardExtent, bound: Option<u32>| {
                    u64::from(
                        e.logical_end
                            .min(bound.unwrap_or(u32::MAX))
                            .saturating_sub(e.start),
                    )
                };
                let rows: u64 = run
                    .output
                    .extents
                    .iter()
                    .enumerate()
                    .filter(|(axis, _)| *axis != output_column)
                    .map(|(_, &e)| size(e, None))
                    .product();
                let cols = size(run.output.extents[output_column], axes.valid_columns).min(size(
                    run.inputs[1].views[0].extents[right_column],
                    axes.valid_columns,
                ));
                let inner =
                    size(run.inputs[0].views[0].extents[left_inner], axes.valid_inner).min(size(
                        run.inputs[1].views[0].extents[right_inner],
                        axes.valid_inner,
                    ));
                let physical: u64 = run
                    .output
                    .extents
                    .iter()
                    .map(|e| u64::from(e.physical_end - e.start))
                    .product();
                run.product_flops =
                    Some([2 * rows * cols * inner, 2 * physical * u64::from(width)]);
                self.append_kernel(body, tile, run)?;
            }
        }
        Ok(())
    }

    pub(super) fn prepare_sum(
        &mut self,
        operation: &MidOperation,
        axis: usize,
        staging: crate::ReductionStaging,
        batch: &mut reduce::SumBatch,
    ) -> ExpansionResult<()> {
        let ([input], [output]) = (operation.inputs.as_slice(), operation.results.as_slice())
        else {
            return Err(ExpansionError::ResultArity);
        };
        let sources = self.value_shards(*input)?.to_vec();
        let outputs = self.value_shards(*output)?.to_vec();
        let mut groups = BTreeMap::<Vec<ShardExtent>, Vec<ShardView>>::new();
        for source in sources {
            let mut block = self.shards[source.index() as usize].clone();
            if axis + 2 >= block.extents.len()
                || block.extents[axis].physical_end - block.extents[axis].start != 1
            {
                return Err(ExpansionError::InvalidOperatorPlan);
            }
            block.extents.remove(axis);
            for (axis, extent) in block.extents.iter_mut().enumerate() {
                extent.axis = axis as u16;
            }
            block.tensor_type.shape.0.remove(axis);
            block.tensor_type.format.layout.tiling = TensorTiling::replicated(1);
            block.definition = ShardDefinition::Alias(source);
            let extents = block.extents.clone();
            let alias = self.push_shard(block)?;
            groups
                .entry(extents)
                .or_default()
                .push(self.full_view(alias));
        }
        self.prepare_sum_partials(
            groups.into_values(),
            &outputs,
            staging,
            operation_provenance(operation),
            batch,
        )
    }
}
