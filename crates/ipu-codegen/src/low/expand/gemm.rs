//! Instantiate a selected distributed product: resident operand windows, local
//! contraction/column blocks, batch matrices and bound GEMM calls.

use super::*;

impl TileGraphBuilder {
    pub(super) fn build_product(
        &mut self,
        operation: &MidOperation,
        product: &crate::Product,
        body: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let [result] = operation.results.as_slice() else {
            return Err(ExpansionError::ResultArity);
        };
        let outputs = self.value_shards(*result)?.to_vec();
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
                &product.output_aliases,
                &inputs_by_tile,
                0,
            )?;
            let inputs = inputs_by_tile
                .iter()
                .zip(&product.operands)
                .map(|(tiles, window)| {
                    let source = *tiles[usize::from(tile)]
                        .first()
                        .ok_or(ExpansionError::InvalidOperatorPlan)?;
                    self.window(source, window)
                })
                .collect::<ExpansionResult<Vec<_>>>()?;
            self.product_calls(
                operation_provenance(operation),
                tile,
                &inputs,
                output,
                product,
                body,
            )?;
        }
        Ok(())
    }
    fn product_calls(
        &mut self,
        provenance: WorkProvenance,
        tile: u16,
        inputs: &[ShardView],
        output: BlockValueId,
        product: &crate::Product,
        body: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let [left, right] = inputs else {
            return Err(ExpansionError::InvalidOperatorPlan);
        };
        let crate::Product {
            inner_block,
            output_columns,
            mode,
            axes,
            ..
        } = product;
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
                let kernel = TileKernelSpec::Gemm {
                    multiply: product.multiply,
                    accumulate: product.accumulate,
                    mode: if k == 0 {
                        *mode
                    } else {
                        crate::GemmKernelMode::Accumulate
                    },
                    inner_block: width,
                    output_columns: column_end - column,
                    weights: if self.shards[right.shard.index() as usize]
                        .tensor_type
                        .format
                        .layout
                        .memory_class
                        == MemoryClass::Ipu21Interleaved
                    {
                        crate::GemmWeightLoad::Interleaved
                    } else {
                        crate::GemmWeightLoad::Standard
                    },
                };
                let run = self.kernel_run(
                    provenance,
                    kernel,
                    vec![
                        KernelOperand { views: vec![l] },
                        KernelOperand { views: vec![r] },
                    ],
                    vec![destination],
                )?;
                let flattens_outer_rows = matches!(
                    run.requirements.outputs[0].format.layout.order,
                    ElementOrder::Amp(AmpOrder::Left | AmpOrder::Output)
                );
                // Count after batch splitting: each local call owns only its
                // selected matrix, including its own logical padding bounds.
                let mut append = |mut run: KernelRun| -> ExpansionResult<()> {
                    let size = |e: ShardExtent, bound: Option<u32>| {
                        u64::from(
                            e.logical_end
                                .min(bound.unwrap_or(u32::MAX))
                                .saturating_sub(e.start),
                        )
                    };
                    let rows: u64 = run.outputs[0]
                        .extents
                        .iter()
                        .enumerate()
                        .filter(|(axis, _)| *axis != output_column)
                        .map(|(_, &e)| size(e, None))
                        .product();
                    let cols =
                        size(run.outputs[0].extents[output_column], axes.valid_columns).min(size(
                            run.inputs[1].views[0].extents[right_column],
                            axes.valid_columns,
                        ));
                    let inner = size(run.inputs[0].views[0].extents[left_inner], axes.valid_inner)
                        .min(size(
                            run.inputs[1].views[0].extents[right_inner],
                            axes.valid_inner,
                        ));
                    let physical: u64 = run.outputs[0]
                        .extents
                        .iter()
                        .map(|e| u64::from(e.physical_end - e.start))
                        .product();
                    run.product_flops =
                        Some([2 * rows * cols * inner, 2 * physical * u64::from(width)]);
                    self.append_kernel(body, tile, run)
                };
                if run.outputs[0].extents.len() > 2 && !flattens_outer_rows {
                    let mut coordinates = vec![0; run.outputs[0].extents.len() - 2];
                    let mut matrices = Vec::new();
                    split_gemm_matrices(&run, 0, &mut coordinates, &mut matrices)?;
                    if matrices.len() > 1 {
                        for matrix in matrices {
                            append(matrix)?;
                        }
                        continue;
                    }
                }
                append(run)?;
            }
        }
        Ok(())
    }
}

fn split_gemm_matrices(
    run: &KernelRun,
    axis: usize,
    coordinates: &mut [u32],
    runs: &mut Vec<KernelRun>,
) -> ExpansionResult<()> {
    if axis < coordinates.len() {
        let extent = run.outputs[0]
            .extents
            .get(axis)
            .ok_or(ExpansionError::InvalidOperatorPlan)?;
        if extent.logical_end != extent.physical_end {
            return Err(ExpansionError::InvalidOperatorPlan);
        }
        for coordinate in extent.start..extent.physical_end {
            coordinates[axis] = coordinate;
            split_gemm_matrices(run, axis + 1, coordinates, runs)?;
        }
        return Ok(());
    }

    let mut matrix = run.clone();
    narrow_gemm_matrix_view(&mut matrix.outputs[0], coordinates)?;
    for operand in &mut matrix.inputs {
        for view in &mut operand.views {
            narrow_gemm_matrix_view(view, coordinates)?;
        }
    }
    runs.push(matrix);
    Ok(())
}

fn narrow_gemm_matrix_view(
    view: &mut ShardView,
    output_coordinates: &[u32],
) -> ExpansionResult<()> {
    let input_axes = view.extents.len().saturating_sub(2);
    if input_axes > output_coordinates.len() {
        return Err(ExpansionError::InvalidOperatorPlan);
    }
    let output_axis_offset = output_coordinates.len() - input_axes;
    for (axis, extent) in view.extents[..input_axes].iter_mut().enumerate() {
        if extent.physical_end - extent.start == 1 {
            continue;
        }
        let coordinate = output_coordinates[output_axis_offset + axis];
        if coordinate < extent.start || coordinate >= extent.physical_end {
            return Err(ExpansionError::InvalidOperatorPlan);
        }
        extent.start = coordinate;
        extent.logical_end = coordinate + 1;
        extent.physical_end = coordinate + 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AccumulationPrecision, Compute, GraphInputKind, MidInput, MidValue, Product, ProductAxes,
        TensorAxis, ValueId,
    };

    #[test]
    fn batched_product_calls_count_only_their_own_arithmetic() {
        for batches in 1..=4 {
            let mut left = Layout::row_major(TensorTiling::replicated(1));
            left.order = ElementOrder::Amp(AmpOrder::Left);
            let mut packed = left.clone();
            packed.order = ElementOrder::BlockMajor(crate::BlockMajorOrder::Matrix {
                row_block: 16,
                column_block: 16,
            });
            let values = [
                TensorType::new([batches, 16, 16], Precision::F16, left),
                TensorType::new([1, 16, 16], Precision::F16, packed.clone()),
                TensorType::new([batches, 16, 16], Precision::F16, packed),
            ]
            .into_iter()
            .enumerate()
            .map(|(i, tensor_type)| {
                let id = MidValueId::from_index(i as u32);
                MidValue {
                    id,
                    origin: ValueId::from_index(i as u32),
                    tensor_type,
                    tile_offset: 0,
                    storage_group: id,
                }
            })
            .collect();
            let mid = MidProgram {
                tile_count: 1,
                values,
                inputs: (0..2)
                    .map(|i| MidInput {
                        name: format!("input{i}"),
                        kind: GraphInputKind::Host,
                        value: MidValueId::from_index(i),
                    })
                    .collect(),
                outputs: vec![MidValueId::from_index(2)],
                operations: vec![MidOperation {
                    source: None,
                    inputs: vec![MidValueId::from_index(0), MidValueId::from_index(1)],
                    results: vec![MidValueId::from_index(2)],
                    kind: MidOperationKind::Compute(Compute::Product(Product {
                        multiply: Precision::F16,
                        accumulate: AccumulationPrecision::F32,
                        mode: crate::GemmKernelMode::Initialize,
                        inner_block: 16,
                        output_columns: 16,
                        axes: ProductAxes {
                            left_inner: TensorAxis::FromEnd(1),
                            right_inner: TensorAxis::FromEnd(2),
                            output_column: TensorAxis::FromEnd(1),
                            valid_inner: None,
                            valid_columns: None,
                        },
                        operands: Default::default(),
                        output_aliases: Vec::new(),
                    })),
                    estimated_cycles: 0,
                    estimated_exchange_cycles: 0,
                }],
                ..MidProgram::default()
            };
            mid.validate().unwrap();
            let low = expand_tiles(&mid, false).unwrap();
            let mut flops = [0; 2];
            for run in low.kernel_calls() {
                crate::validate_kernel_run(run).unwrap();
                let counted = run.product_flops.unwrap();
                flops[0] += counted[0];
                flops[1] += counted[1];
                assert!(counted[0] <= counted[1]);
            }
            let expected = 2 * u64::from(batches) * 16 * 16 * 16;
            assert_eq!(flops, [expected, expected]);
        }
    }
}
