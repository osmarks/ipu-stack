//! Distributed GEMM decomposition and explicit independent partials.

use super::*;

impl Builder {
    pub(super) fn gemm(
        &mut self,
        plan: &OperatorPlan,
        output: &TensorType,
        inner_block: u32,
        column_block: u32,
        orientation: GemmOrientation,
        distribution: GemmDistribution,
    ) -> Option<MidValueId> {
        let MidOperator::Gemm {
            multiply,
            accumulate,
            ..
        } = plan.operator
        else {
            return None;
        };
        let (left, right) = orientation.operand_indices();
        let left = MidValueId(left as u32);
        let right = MidValueId(right as u32);
        let left_source_type = self.tensor(left).clone();
        let right_source_type = self.tensor(right).clone();
        let mut left_type = left_source_type.clone();
        let mut right_type = right_source_type.clone();
        left_type.format = plan.requirements.inputs[left.index() as usize]
            .format
            .clone();
        right_type.format = plan.requirements.inputs[right.index() as usize]
            .format
            .clone();
        let (left_row, left_inner) = orientation.matrix_axes(left_type.shape.0.len());
        let (right_inner, right_column) = orientation.matrix_axes(right_type.shape.0.len());
        let (output_row, output_column) = orientation.matrix_axes(output.shape.0.len());
        let axes = ProductAxes {
            left_inner: TensorAxis::FromStart(left_inner as u16),
            right_inner: TensorAxis::FromStart(right_inner as u16),
            output_column: TensorAxis::FromEnd(if orientation == GemmOrientation::Normal {
                1
            } else {
                2
            }),
        };
        let kernel = |mode| TileKernelSpec::Gemm {
            multiply,
            accumulate,
            mode,
            weights: GemmWeightLoad::Standard,
            inner_block,
            output_columns: column_block,
        };
        match distribution {
            GemmDistribution::ParallelReduction {
                row_partitions,
                column_partitions,
                inner_partitions,
                reduction_staging,
                ..
            } => {
                let left = self.copy(left, left_type.clone(), vec![]);
                let mut partial = plan.dispatch.gemm_partial_tensor(output);
                // Partials follow compute rows, including padding/group boundaries;
                // the final result may partition those rows differently.
                let mut rows = *axis(&left_type, left_row)?;
                rows.axis = TensorAxis::FromStart(output_row as u16);
                let target = partial
                    .format
                    .layout
                    .tiling
                    .axes
                    .iter_mut()
                    .find(|dim| dim.axis.resolve(output.shape.0.len()) == Ok(output_row))?;
                *target = rows;
                let mut right_staging = right_type.clone();
                let mut inner = *axis(&left_type, left_inner)?;
                inner.axis = TensorAxis::FromStart(right_inner as u16);
                inner.tile_stride = Some(column_partitions);
                let mut column = *axis(&partial, output_column)?;
                column.axis = TensorAxis::FromStart(right_column as u16);
                column.tile_stride = Some(1);
                right_staging.format.layout.tiling = TensorTiling {
                    tile_count: row_partitions
                        .checked_mul(column_partitions)?
                        .checked_mul(inner_partitions)?,
                    replicas: row_partitions,
                    axes: vec![column, inner],
                };
                if multiply == Precision::F16 {
                    right_staging.format.layout.memory_class = MemoryClass::Ipu21Interleaved;
                }
                let weights = self.materialize(
                    right,
                    right_staging,
                    vec![],
                    plan.requirements.inputs[right.index() as usize].local_staging
                        == LocalOperandStaging::Direct,
                );
                let mut partials = partial;
                partials.shape.0.insert(0, u32::from(inner_partitions));
                let rank = partials.shape.0.len();
                for dim in &mut partials.format.layout.tiling.axes {
                    let old = dim.axis.resolve(rank - 1).ok()?;
                    dim.axis = TensorAxis::FromStart((old + 1) as u16);
                    dim.tile_stride = Some(if old == output_row {
                        column_partitions.checked_mul(inner_partitions)?
                    } else {
                        1
                    });
                }
                partials.format.layout.tiling.axes.push(
                    AxisTiling::new(
                        TensorAxis::FromStart(0),
                        inner_partitions,
                        1,
                        Padding::Reject,
                    )
                    .with_tile_stride(column_partitions),
                );
                partials.format.layout.tiling.tile_count = row_partitions
                    .checked_mul(column_partitions)?
                    .checked_mul(inner_partitions)?;
                let products = self.compute(
                    vec![left, weights],
                    partials,
                    kernel(GemmKernelMode::Initialize),
                    Some(axes),
                    None,
                    vec![],
                );
                Some(self.emit(
                    vec![products],
                    output.clone(),
                    Primitive::Sum {
                        axis: 0,
                        staging: reduction_staging,
                    },
                ))
            }
            GemmDistribution::OutputStationary => {
                // Bounded K panels are separate tensor values in mid. Their
                // lifetime ends after each product; tile expansion can reuse
                // physical storage without inferring a GEMM staging strategy.
                let mut left_panel = left_type.clone();
                left_panel.format.layout.tiling =
                    project_grid(output, &left_type, output_row, left_row)?;
                let mut right_panel = right_type.clone();
                right_panel.format.layout.tiling =
                    project_grid(output, &right_type, output_column, right_column)?;
                // Residency is a whole-device geometry decision. A compatible
                // existing distribution needs no materialization or new panels.
                for (tensor, input, inner_axis) in [
                    (&mut left_panel, &left_type, left_inner),
                    (&mut right_panel, &right_type, right_inner),
                ] {
                    if let Some(axis) = axis(input, inner_axis) {
                        tensor.format.layout.tiling.axes.push(*axis);
                    }
                }
                if same_distribution(&left_panel, &left_source_type)
                    && same_distribution(&right_panel, &right_source_type)
                    && left_panel.format.layout.order == left_source_type.format.layout.order
                    && right_panel.format.layout.order == right_source_type.format.layout.order
                {
                    return Some(self.compute(
                        vec![left, right],
                        output.clone(),
                        kernel(GemmKernelMode::Initialize),
                        Some(axes),
                        None,
                        vec![],
                    ));
                }
                for (tensor, axis) in [
                    (&mut left_panel, left_inner),
                    (&mut right_panel, right_inner),
                ] {
                    tensor.shape.0[axis] = inner_block;
                    tensor
                        .format
                        .layout
                        .tiling
                        .axes
                        .retain(|dim| dim.axis.resolve(tensor.shape.0.len()) != Ok(axis));
                    tensor.format.layout.tiling.axes.push(AxisTiling::new(
                        TensorAxis::FromStart(axis as u16),
                        1,
                        inner_block,
                        Padding::Zero,
                    ));
                }
                if multiply == Precision::F16 {
                    right_panel.format.layout.memory_class = MemoryClass::Ipu21Interleaved;
                }
                let inner = left_type.shape.0[left_inner];
                let mut result = None;
                for start in (0..inner).step_by(inner_block as usize) {
                    let mut l = left_panel.clone();
                    let mut r = right_panel.clone();
                    l.shape.0[left_inner] = inner_block.min(inner - start);
                    r.shape.0[right_inner] = inner_block.min(inner - start);
                    let mut lo = vec![0; l.shape.0.len()];
                    lo[left_inner] = start;
                    let mut ro = vec![0; r.shape.0.len()];
                    ro[right_inner] = start;
                    let l = self.copy(left, l, lo);
                    let r = self.copy(right, r, ro);
                    result = Some(self.compute(
                        vec![l, r],
                        output.clone(),
                        kernel(if result.is_none() {
                            GemmKernelMode::Initialize
                        } else {
                            GemmKernelMode::Accumulate
                        }),
                        Some(axes),
                        result,
                        vec![],
                    ));
                }
                result
            }
        }
    }
}
