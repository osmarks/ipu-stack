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
                offsets,
                reuse_local,
            } => self.copy_tensor(operation, offsets, *reuse_local, body),
            Primitive::MappedCopy { view, offsets } => {
                let inputs = self.value_shards(operation.inputs[0])?.to_vec();
                let outputs = self.value_shards(operation.results[0])?.to_vec();
                let mappings = self.window_view_mappings(&inputs, &outputs, *view, offsets)?;
                let physical = self.f16_micro_panel_mappings(mappings.clone())?;
                let (mappings, order) = physical.map_or((mappings, CopyOrder::Semantic), |m| {
                    (m, CopyOrder::Physical)
                });
                self.build_mapped_views(
                    mappings,
                    order,
                    order,
                    operation_provenance(operation),
                    body,
                )
            }
            Primitive::View(view) => self.build_view(operation, *view, body),
            Primitive::Sum { axis, staging } => {
                self.sum_tensor(operation, usize::from(*axis), *staging, body)
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
                        let previous = self
                            .find_local_shard(operation.inputs[*index], tile)?
                            .ok_or(ExpansionError::InvalidOperatorPlan)?;
                        if self.shards[previous.index() as usize].extents
                            != self.shards[output.index() as usize].extents
                        {
                            return Err(ExpansionError::InvalidOperatorPlan);
                        }
                        self.shards[output.index() as usize].definition =
                            ShardDefinition::WritableAlias(previous);
                    }
                    let inputs = operation
                        .inputs
                        .iter()
                        .zip(operands)
                        .map(|(&value, window)| {
                            let source = self
                                .value_shards(value)?
                                .iter()
                                .copied()
                                .find(|&source| {
                                    self.shards[source.index() as usize].tile == tile
                                        && (!matches!(
                                            kernel,
                                            TileKernelSpec::Gelu | TileKernelSpec::Add
                                        ) || self.broadcast_view(source, output).is_some())
                                })
                                .ok_or(ExpansionError::InvalidOperatorPlan)?;
                            self.window(source, window)
                        })
                        .collect::<ExpansionResult<Vec<_>>>()?;
                    if let Some(axes) = product {
                        self.product_calls(
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

    fn copy_tensor(
        &mut self,
        operation: &MidOperation,
        offsets: &[u32],
        reuse_local: bool,
        body: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let ([input], [output]) = (operation.inputs.as_slice(), operation.results.as_slice())
        else {
            return Err(ExpansionError::ResultArity);
        };
        let inputs = self.value_shards(*input)?.to_vec();
        let outputs = self.value_shards(*output)?.to_vec();
        let mut mappings = Vec::new();
        for output in outputs {
            let destination = &self.shards[output.index() as usize];
            let tile = destination.tile;
            let mut source_region = destination.extents.clone();
            for (axis, extent) in source_region.iter_mut().enumerate() {
                let offset = offsets.get(axis).copied().unwrap_or(0);
                extent.start = extent
                    .start
                    .checked_add(offset)
                    .ok_or(ExpansionError::IdOverflow)?;
                extent.logical_end = extent
                    .logical_end
                    .checked_add(offset)
                    .ok_or(ExpansionError::IdOverflow)?;
                extent.physical_end = extent
                    .physical_end
                    .checked_add(offset)
                    .ok_or(ExpansionError::IdOverflow)?;
            }
            let intersections = self
                .intersecting_shard_set(&inputs, &source_region, tile)
                .into_iter()
                .map(|(_, source)| {
                    (
                        intersect_extents_with_shared_padding(
                            &self.shards[source.index() as usize].extents,
                            &source_region,
                        )
                        .expect("selected intersection remains nonempty"),
                        source,
                    )
                })
                .collect::<Vec<_>>();
            if reuse_local
                && offsets.iter().all(|&offset| offset == 0)
                && let [(extents, source)] = intersections.as_slice()
                && *extents == self.shards[output.index() as usize].extents
                && self.shards[source.index() as usize].tile == tile
                && self.shards[source.index() as usize]
                    .tensor_type
                    .format
                    .precision
                    == self.shards[output.index() as usize]
                        .tensor_type
                        .format
                        .precision
                && self.shards[source.index() as usize]
                    .tensor_type
                    .format
                    .layout
                    .order
                    == self.shards[output.index() as usize]
                        .tensor_type
                        .format
                        .layout
                        .order
            {
                let mut view = self.full_view(*source);
                view.extents = extents.clone();
                self.materialized_views.insert(output, view);
                self.shards[output.index() as usize].definition = ShardDefinition::Unmaterialized;
                continue;
            }
            if intersections.is_empty() {
                self.append_fill_zero(body, output, operation_provenance(operation))?;
            }
            for (source_extents, source) in intersections {
                let mut destination_extents = source_extents.clone();
                for (axis, extent) in destination_extents.iter_mut().enumerate() {
                    let offset = offsets.get(axis).copied().unwrap_or(0);
                    extent.start -= offset;
                    extent.logical_end -= offset;
                    extent.physical_end -= offset;
                }
                mappings.push((
                    ShardView {
                        shard: self.full_view(source).shard,
                        extents: source_extents,
                    },
                    ShardView {
                        shard: output,
                        extents: destination_extents,
                    },
                ));
            }
        }
        let order = if self.shards[inputs[0].index() as usize]
            .tensor_type
            .format
            .layout
            .order
            == self.shards[self.value_shards(*output)?[0].index() as usize]
                .tensor_type
                .format
                .layout
                .order
        {
            CopyOrder::Physical
        } else {
            CopyOrder::Semantic
        };
        self.build_mapped_views(
            mappings,
            order,
            order,
            operation_provenance(operation),
            body,
        )
    }

    fn product_calls(
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
        for column in (columns.start..columns.physical_end).step_by(*output_columns as usize) {
            let column_end = (column + output_columns).min(columns.physical_end);
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
                let run = self.kernel_run(
                    provenance,
                    kernel,
                    vec![
                        KernelOperand { views: vec![l] },
                        KernelOperand { views: vec![r] },
                    ],
                    destination,
                )?;
                self.append_kernel(body, tile, run)?;
            }
        }
        Ok(())
    }

    fn sum_tensor(
        &mut self,
        operation: &MidOperation,
        axis: usize,
        staging: crate::ReductionStaging,
        body: &mut BlockRegion,
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
        self.append_sum_partials(
            groups.into_values(),
            &outputs,
            staging,
            operation_provenance(operation),
            body,
        )
    }
}
