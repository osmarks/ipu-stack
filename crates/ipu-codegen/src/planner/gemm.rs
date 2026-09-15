//! Distributed GEMM decomposition and explicit independent partials.

use super::fragments::{FragmentBuilder, project_grid};
use crate::kernel::{AccumulationPrecision, GemmKernelMode};
use crate::mid::MidOperationKind;
use crate::mid::{MidValueId, OperandIndexing, OperandWindow};
use crate::planner::operator::{
    GemmDistribution, GemmOrientation, OperatorFamily, OperatorPlan, ProductGrid,
};
use crate::tensor::{
    AmpOrder, AxisTiling, BlockMajorOrder, ElementOrder, MemoryClass, Padding, Precision,
    TensorAxis, TensorTiling, TensorType, axis_tiling, same_distribution,
};
use crate::{GemmAxes, ReductionStaging};

/// Construction parameters, consumed immediately to emit executable mid work.
#[derive(Clone, Debug)]
pub(super) struct Product {
    pub multiply: Precision,
    pub accumulate: AccumulationPrecision,
    pub mode: GemmKernelMode,
    pub inner_block: u32,
    pub output_columns: u32,
    pub axes: crate::GemmAxes,
}

impl FragmentBuilder {
    pub(super) fn product(
        &mut self,
        inputs: Vec<MidValueId>,
        (output, reuse): (TensorType, Option<MidValueId>),
        product: Product,
        operands: Vec<OperandIndexing>,
    ) -> Option<MidValueId> {
        let [left, right] = inputs.as_slice() else {
            return None;
        };
        let tensors = [
            self.tensor(*left).clone(),
            self.tensor(*right).clone(),
            output.clone(),
        ];
        let mut shapes = Vec::new();
        let mut windows = [OperandWindow::default(), OperandWindow::default()];
        for (i, operand) in operands.iter().enumerate() {
            let OperandIndexing::Local(window) = operand else {
                return None;
            };
            *windows.get_mut(i)? = window.clone();
        }
        for (i, tensor) in tensors.iter().enumerate() {
            tensor.format.layout.resolve(&tensor.shape).ok()?.axes()?;
            shapes.push(
                windows
                    .get(i)
                    .unwrap_or(&OperandWindow::default())
                    .local_tensor(tensor, false)?
                    .shape
                    .0,
            );
        }
        let axes = product.axes;
        let li = axes.left_inner.resolve(shapes[0].len()).ok()?;
        let ri = axes.right_inner.resolve(shapes[1].len()).ok()?;
        let oc = axes.output_column.resolve(shapes[2].len()).ok()?;
        let rc = if ri + 1 == shapes[1].len() {
            ri - 1
        } else {
            ri + 1
        };
        let inner = shapes[0][li];
        if product.inner_block == 0 || product.output_columns == 0 || inner != shapes[1][ri] {
            return None;
        }
        let columns = output
            .format
            .layout
            .order
            .gemm_output_group()
            .map_or(product.output_columns, |g| g.min(product.output_columns));
        let flatten = matches!(
            output.format.layout.order,
            ElementOrder::Amp(AmpOrder::Left | AmpOrder::Output)
        );
        let batch_axes = if flatten { 0 } else { shapes[2].len() - 2 };
        let batches = shapes[2][..batch_axes]
            .iter()
            .try_fold(1u32, |n, &d| n.checked_mul(d))?;
        let indexing = if batch_axes == 0 {
            Vec::new()
        } else {
            tensors
                .iter()
                .map(|t| {
                    crate::tensor::Broadcast::new(
                        &t.shape.0[..t.shape.0.len() - 2],
                        &output.shape.0[..batch_axes],
                    )
                })
                .collect::<Option<Vec<_>>>()?
        };
        let mut result = reuse;
        for column in (0..shapes[2][oc]).step_by(columns as usize) {
            for k in (0..inner).step_by(product.inner_block as usize) {
                let width = product.inner_block.min(inner - k);
                let column_end = (column + columns).min(shapes[2][oc]);
                for batch in 0..batches {
                    let mut ranges = [
                        windows[0].clone(),
                        windows[1].clone(),
                        OperandWindow::default(),
                    ];
                    for (operand, axis, start, end) in [
                        (0, li, k, k + width),
                        (1, ri, k, k + width),
                        (1, rc, column, column_end),
                        (2, oc, column, column_end),
                    ] {
                        let base = ranges[operand]
                            .0
                            .iter()
                            .find(|r| r.0 as usize == axis)
                            .map_or(0, |r| r.1);
                        ranges[operand].0.retain(|r| r.0 as usize != axis);
                        ranges[operand]
                            .0
                            .push((axis as u16, base + start, base + end));
                    }
                    let mut coordinates = vec![0; batch_axes];
                    let mut remainder = batch;
                    for axis in (0..batch_axes).rev() {
                        coordinates[axis] = remainder % shapes[2][axis];
                        remainder /= shapes[2][axis];
                    }
                    for (i, indexing) in indexing.iter().enumerate() {
                        for axis in 0..shapes[i].len() - 2 {
                            let coordinate = if indexing.is_broadcast(axis) {
                                0
                            } else {
                                coordinates[indexing.output_axis(axis)]
                            };
                            ranges[i].0.push((axis as u16, coordinate, coordinate + 1));
                        }
                    }
                    result = Some(
                        self.compute(
                            inputs.clone(),
                            [(output.clone(), result, ranges[2].clone())],
                            MidOperationKind::Gemm {
                                axes,
                                multiply: product.multiply,
                                accumulate: product.accumulate,
                                mode: if k == 0 {
                                    product.mode
                                } else {
                                    GemmKernelMode::Accumulate
                                },
                                weights: if tensors[1].format.layout.memory_class
                                    == MemoryClass::Ipu21Interleaved
                                {
                                    crate::GemmWeightLoad::Interleaved
                                } else {
                                    crate::GemmWeightLoad::Standard
                                },
                                inner_block: width,
                                output_columns: column_end - column,
                            },
                            vec![
                                OperandIndexing::Fragment(ranges[0].clone()),
                                OperandIndexing::Fragment(ranges[1].clone()),
                            ],
                        )[0],
                    );
                }
            }
        }
        result
    }

    pub(super) fn gemm(
        &mut self,
        plan: &OperatorPlan,
        output: &TensorType,
        inner_block: u32,
        column_block: u32,
        orientation: GemmOrientation,
        distribution: GemmDistribution,
    ) -> Option<MidValueId> {
        let OperatorFamily::Gemm {
            multiply,
            accumulate,
            ..
        } = plan.operator
        else {
            return None;
        };
        let (left, right) = orientation.operand_indices();
        let left = MidValueId::from_index(left as u32);
        let right = MidValueId::from_index(right as u32);
        let left_source_type = self.tensor(left).clone();
        let right_source_type = self.tensor(right).clone();
        let mut left_type = left_source_type.clone();
        let mut right_type = right_source_type.clone();
        left_type.format = plan.inputs[left.index() as usize].format.clone();
        right_type.format = plan.inputs[right.index() as usize].format.clone();
        let (left_row, left_inner) = orientation.matrix_axes(left_type.shape.0.len());
        let (right_inner, right_column) = orientation.matrix_axes(right_type.shape.0.len());
        let (output_row, output_column) = orientation.matrix_axes(output.shape.0.len());
        let axes = GemmAxes {
            valid_inner: None,
            valid_columns: None,
            left_inner: TensorAxis::FromStart(left_inner as u16),
            right_inner: TensorAxis::FromStart(right_inner as u16),
            output_column: TensorAxis::FromEnd(if orientation == GemmOrientation::Normal {
                1
            } else {
                2
            }),
        };
        let product = |mode| Product {
            multiply,
            accumulate,
            mode,
            inner_block,
            output_columns: column_block,
            axes,
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
                let mut rows = *axis_tiling(&left_type, left_row)?;
                rows.axis = TensorAxis::FromStart(output_row as u16);
                let target = partial
                    .format
                    .layout
                    .tiling
                    .axes
                    .iter_mut()
                    .find(|dim| dim.axis.resolve(output.shape.0.len()) == Ok(output_row))?;
                *target = rows;
                let mut right_staging = right_type;
                let mut inner = *axis_tiling(&left_type, left_inner)?;
                inner.axis = TensorAxis::FromStart(right_inner as u16);
                inner.tile_stride = Some(column_partitions);
                let mut column = *axis_tiling(&partial, output_column)?;
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
                let weights = self.copy(right, right_staging, vec![]);
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
                let products = self.product(
                    vec![left, weights],
                    (partials, None),
                    product(GemmKernelMode::Initialize),
                    vec![crate::OperandIndexing::local(); 2],
                )?;
                Some(self.sum(products, &output.clone(), 0, reduction_staging)?)
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
                    if let Some(axis) = axis_tiling(input, inner_axis) {
                        tensor.format.layout.tiling.axes.push(*axis);
                    }
                }
                if same_distribution(&left_panel, &left_source_type)
                    && same_distribution(&right_panel, &right_source_type)
                    && left_panel.format.layout.order == left_source_type.format.layout.order
                    && right_panel.format.layout.order == right_source_type.format.layout.order
                {
                    return Some(self.product(
                        vec![left, right],
                        (output.clone(), None),
                        product(GemmKernelMode::Initialize),
                        vec![crate::OperandIndexing::local(); 2],
                    )?);
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
                    result = Some(self.product(
                        vec![l, r],
                        (output.clone(), result),
                        product(if result.is_none() {
                            GemmKernelMode::Initialize
                        } else {
                            GemmKernelMode::Accumulate
                        }),
                        vec![crate::OperandIndexing::local(); 2],
                    )?);
                }
                result
            }
        }
    }
}

impl FragmentBuilder {
    /// A batched product with independent row/column/K ownership, followed by
    /// redistribution (or a sum of explicit partials) into its consumer layout.
    pub(super) fn distributed_product(
        &mut self,

        left: MidValueId,
        right: MidValueId,
        output: &TensorType,
        axes: GemmAxes,
        grid: ProductGrid,
        fp8_scale: Option<i8>,
    ) -> Option<MidValueId> {
        let mut l = self.tensor(left).clone();
        let mut r = self.tensor(right).clone();
        if l.shape.0.len() != 3
            || r.shape.0.len() != 3
            || output.shape.0.len() != 3
            || output.format.precision != Precision::F16
        {
            return None;
        }
        let li = axes.left_inner.resolve(3).ok()?;
        let ri = axes.right_inner.resolve(3).ok()?;
        let rc = if ri == 2 { 1 } else { 2 };
        if li != 2
            || axes.output_column.resolve(3).ok()? != 2
            || grid.rows == 0
            || grid.columns == 0
            || grid.inner == 0
        {
            return None;
        }
        let heads = u16::try_from(output.shape.0[0]).ok()?;
        let col_stride = heads;
        let inner_stride = heads.checked_mul(grid.columns)?;
        let row_stride = inner_stride.checked_mul(grid.inner)?;
        let tiles = row_stride.checked_mul(grid.rows)?;
        let dim = |axis, partitions, grain, stride| {
            AxisTiling::new(
                TensorAxis::FromStart(axis),
                partitions,
                grain,
                Padding::Zero,
            )
            .with_tile_stride(stride)
        };
        let head = dim(0, heads, 1, 1);
        let rows = dim(1, grid.rows, 1, row_stride);
        let inner = axes
            .valid_inner
            .unwrap_or(l.shape.0[li])
            .min(l.shape.0[li])
            .min(r.shape.0[ri]);
        let columns = axes.valid_columns.unwrap_or(output.shape.0[2]);

        r.shape.0[ri] = inner;
        r.shape.0[rc] = columns;
        let multiply = fp8_scale.map_or(Precision::F16, |scale_exponent| Precision::F8F143 {
            scale_exponent,
        });
        let grain = if fp8_scale.is_some() { 32 } else { 16 };
        let inner_width = inner.div_ceil(grain).div_ceil(u32::from(grid.inner)) * grain;
        let column_width = columns.div_ceil(16).div_ceil(u32::from(grid.columns)) * 16;
        // The last K group must use the same coefficient block shape as the
        // others. Keep the left padding addressable (softmax stores zero weights
        // there), while the right operand retains the true logical K bound.
        l.shape.0[li] = inner_width.checked_mul(u32::from(grid.inner))?;
        // A fused producer may already provide the selected operand precision.
        l.format.precision = self.tensor(left).format.precision;
        r.format.precision = self.tensor(right).format.precision;
        l.format.layout.order = ElementOrder::Amp(AmpOrder::Left);
        l.format.layout.tiling = TensorTiling {
            tile_count: tiles,
            replicas: grid.columns,
            axes: vec![
                head,
                rows,
                dim(li as u16, grid.inner, inner_width, inner_stride),
            ],
        };
        r.format.layout.tiling = TensorTiling {
            tile_count: tiles,
            replicas: grid.rows,
            axes: vec![
                head,
                dim(ri as u16, grid.inner, inner_width, inner_stride),
                dim(rc as u16, grid.columns, 16, col_stride),
            ],
        };
        r.format.layout.order = if ri == 2 {
            ElementOrder::Amp(AmpOrder::TransposedRight)
        } else {
            ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                row_block: u16::try_from(inner_width).ok()?,
                column_block: 16,
            })
        };
        r.format.layout.memory_class = MemoryClass::Ipu21Interleaved;
        let l = self.copy(left, l, vec![]);
        let r = self.copy(right, r, vec![]);
        let l = self.cast(l, multiply);
        let r = self.cast(r, multiply);
        let mut product = output.clone();
        product.shape.0[2] = columns;
        product.format.layout.order = ElementOrder::Amp(AmpOrder::Left);
        product.format.layout.tiling = TensorTiling {
            tile_count: tiles,
            replicas: 1,
            axes: vec![head, rows, dim(2, grid.columns, 16, col_stride)],
        };
        if grid.inner > 1 {
            product.shape.0.insert(0, u32::from(grid.inner));
            for axis in &mut product.format.layout.tiling.axes {
                axis.axis = TensorAxis::FromStart(axis.axis.resolve(3).ok()? as u16 + 1);
            }
            product
                .format
                .layout
                .tiling
                .axes
                .push(dim(0, grid.inner, 1, inner_stride));
        }
        let result = self.product(
            vec![l, r],
            (product, None),
            Product {
                multiply,
                accumulate: if fp8_scale.is_some() {
                    AccumulationPrecision::F16
                } else {
                    AccumulationPrecision::F32
                },
                mode: GemmKernelMode::Initialize,
                inner_block: inner_width,
                output_columns: column_width,
                axes,
            },
            vec![crate::OperandIndexing::local(); 2],
        )?;
        if grid.inner > 1 {
            let mut reduced = output.clone();
            reduced.shape.0[2] = columns;
            let sum = self.sum(result, &reduced, 0, ReductionStaging::Complete)?;
            Some(self.copy(sum, output.clone(), vec![]))
        } else {
            Some(self.copy(result, output.clone(), vec![]))
        }
    }
}
