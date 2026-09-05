//! Decompose selected algorithms into compact, whole-device tensor operations.
//! No shard enumeration, tile calls, byte spans, or physical allocation occurs here.

use super::*;
use std::sync::Arc;

pub(crate) fn implement(
    plan: &OperatorPlan,
    inputs: &[TensorType],
    output: &TensorType,
) -> Option<Arc<MidProgram>> {
    let mut b = Builder::new(inputs);
    let result = match &plan.dispatch {
        OperatorDispatch::Pointwise { kernel, .. } => {
            let mut operands = Vec::new();
            for (index, input) in inputs.iter().enumerate() {
                let mut resident = input.clone();
                resident.format.layout.tiling = if input.shape == output.shape {
                    output.format.layout.tiling.clone()
                } else {
                    TensorTiling::replicated(output.format.layout.tiling.tile_count)
                };
                operands.push(b.copy(MidValueId(index as u32), resident, vec![]));
            }
            let reuse = match &plan.requirements.output_aliasing {
                OutputAliasing::MayAliasInputs(indices) => indices.iter().find_map(|&index| {
                    let input = *operands.get(usize::from(index))?;
                    (b.tensor(input) == output).then_some(input)
                }),
                OutputAliasing::Fresh => None,
            };
            b.compute(
                operands,
                output.clone(),
                kernel.clone(),
                None,
                reuse,
                vec![],
            )
        }
        OperatorDispatch::View => {
            let MidOperator::View(view) = plan.operator else {
                return None;
            };
            b.emit(vec![MidValueId(0)], output.clone(), Primitive::View(view))
        }
        OperatorDispatch::BlockedGemm {
            inner_block,
            output_column_block,
            orientation,
            distribution,
        } => b.gemm(
            plan,
            output,
            *inner_block,
            *output_column_block,
            *orientation,
            *distribution,
        )?,
        OperatorDispatch::BlockedAttention {
            query_key,
            probability_value,
            query_block_rows: _,
            key_block_rows,
            padded_query_dimension,
            padded_value_dimension,
        } => b.attention(
            output,
            query_key,
            probability_value,
            *key_block_rows,
            *padded_query_dimension,
            *padded_value_dimension,
            false,
        )?,
        OperatorDispatch::MaterializedAttention {
            query_key,
            probability_value,
            query_block_rows: _,
            padded_key_rows,
            padded_query_dimension,
            padded_value_dimension,
        } => b.attention(
            output,
            query_key,
            probability_value,
            *padded_key_rows,
            *padded_query_dimension,
            *padded_value_dimension,
            true,
        )?,
    };
    b.program.outputs = vec![result];
    b.program.tile_count = b
        .program
        .values
        .iter()
        .map(|v| v.tensor_type.format.layout.tiling.tile_count)
        .max()?;
    let (cycles, peak) = crate::estimate::analyze_mid(&b.program, &BTreeMap::new())?;
    b.program.estimated_cycles = cycles.total;
    b.program.estimated_exchange_cycles = cycles.exchange;
    b.program.peak_memory = peak;
    Some(Arc::new(b.program))
}

struct Builder {
    program: MidProgram,
}

impl Builder {
    fn new(inputs: &[TensorType]) -> Self {
        let mut b = Self {
            program: MidProgram::default(),
        };
        for tensor in inputs {
            let value = b.value(tensor.clone());
            b.program.inputs.push(MidInput {
                name: String::new(),
                kind: GraphInputKind::Host,
                value,
            });
        }
        b
    }

    fn value(&mut self, tensor_type: TensorType) -> MidValueId {
        let id = MidValueId(self.program.values.len() as u32);
        self.program.values.push(MidValue {
            id,
            tile_offset: 0,
            tensor_type,
            origin: ValueId::from_index(0),
            storage_group: id,
        });
        id
    }

    fn tensor(&self, value: MidValueId) -> &TensorType {
        &self.program.values[value.index() as usize].tensor_type
    }

    fn emit(
        &mut self,
        inputs: Vec<MidValueId>,
        output: TensorType,
        primitive: Primitive,
    ) -> MidValueId {
        let result = self.value(output);
        self.program.operations.push(MidOperation {
            source: None,
            inputs,
            results: vec![result],
            kind: MidOperationKind::Primitive(primitive),
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        });
        result
    }

    fn copy(&mut self, input: MidValueId, output: TensorType, offsets: Vec<u32>) -> MidValueId {
        let source = self.tensor(input);
        if offsets.iter().all(|&offset| offset == 0)
            && (source == &output
                || (source.format.precision == output.format.precision
                    && source.format.layout.order == output.format.layout.order
                    && source.format.layout.memory_class == output.format.layout.memory_class
                    && same_distribution(source, &output)))
        {
            return input;
        }
        self.emit(
            vec![input],
            output,
            Primitive::Copy {
                offsets,
                reuse_local: true,
            },
        )
    }

    fn compute(
        &mut self,
        mut inputs: Vec<MidValueId>,
        output: TensorType,
        kernel: TileKernelSpec,
        product: Option<ProductAxes>,
        reuse: Option<MidValueId>,
        mut operands: Vec<OperandWindow>,
    ) -> MidValueId {
        if operands.is_empty() {
            operands.resize(inputs.len(), OperandWindow::default());
        }
        let reuse_input = reuse.map(|value| {
            let index = inputs.len();
            inputs.push(value);
            index
        });
        self.emit(
            inputs,
            output,
            Primitive::Compute {
                kernel,
                operands,
                product,
                reuse_input,
            },
        )
    }

    fn gemm(
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
                let partial = plan.dispatch.gemm_partial_tensor(output);
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

    fn attention(
        &mut self,
        output: &TensorType,
        query_key: &TileKernelSpec,
        probability_value: &TileKernelSpec,
        key_block: u32,
        query_width: u32,
        value_width: u32,
        materialized: bool,
    ) -> Option<MidValueId> {
        let query = self.tensor(MidValueId(0)).clone();
        let key = self.tensor(MidValueId(1)).clone();
        let value = self.tensor(MidValueId(2)).clone();
        let rank = output.shape.0.len();
        if rank != 3 || key_block == 0 {
            return None;
        }
        let key_rows = key.shape.0[1];
        let mut query_type = query.clone();
        query_type.format.layout = output.format.layout.clone();
        query_type.format.layout.order = ElementOrder::Amp(AmpOrder::Left);
        query_type.shape.0[2] = query_width;
        let query_buffer = self.copy(MidValueId(0), query_type, vec![]);
        let mut scores_type = output.clone();
        scores_type.shape.0[2] = key_block;
        scores_type.format.layout.order = ElementOrder::Amp(AmpOrder::Left);
        scores_type.format.layout.memory_class = MemoryClass::Ipu21Interleaved;
        let mut weights_type = scores_type.clone();
        weights_type.shape.0[2] = key_block + AMP_COLUMN_MICRO;
        weights_type.format.layout.memory_class = MemoryClass::Ipu21Standard;
        let mut product_type = scores_type.clone();
        product_type.shape.0[2] = value_width;
        let qk_axes = ProductAxes {
            left_inner: TensorAxis::FromEnd(1),
            right_inner: TensorAxis::FromEnd(1),
            output_column: TensorAxis::FromEnd(1),
        };
        let pv_axes = ProductAxes {
            left_inner: TensorAxis::FromEnd(1),
            right_inner: TensorAxis::FromEnd(2),
            output_column: TensorAxis::FromEnd(1),
        };
        let mut weights = None;
        let mut result = None;
        for start in (0..key_rows).step_by(key_block as usize) {
            let valid = key_block.min(key_rows - start);
            let mut packed_key = key.clone();
            let mut packed_value = value.clone();
            for (tensor, width) in [
                (&mut packed_key, query_width),
                (&mut packed_value, value_width),
            ] {
                tensor.shape.0[1] = valid;
                tensor.shape.0[2] = width;
                tensor.format.layout.tiling = project_grid(output, tensor, 0, 0)?;
                tensor.format.layout.tiling.axes.push(AxisTiling::new(
                    TensorAxis::FromStart(1),
                    1,
                    key_block,
                    Padding::Zero,
                ));
                tensor.format.layout.tiling.axes.push(AxisTiling::new(
                    TensorAxis::FromStart(2),
                    1,
                    AMP_COLUMN_MICRO,
                    Padding::Zero,
                ));
                tensor.format.layout.memory_class = MemoryClass::Ipu21Standard;
            }
            packed_key.format.layout.order = ElementOrder::Amp(AmpOrder::TransposedRight);
            packed_value.format.layout.order = ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                row_block: AMP_INNER_BLOCK as u16,
                column_block: AMP_COLUMN_MICRO as u16,
            });
            let k = self.copy(MidValueId(1), packed_key, vec![0, start, 0]);
            let scores = self.compute(
                vec![query_buffer, k],
                scores_type.clone(),
                query_key.clone(),
                Some(qk_axes),
                None,
                vec![],
            );
            weights = Some(self.compute(
                vec![scores],
                weights_type.clone(),
                TileKernelSpec::AttentionSoftmax {
                    head_dimension: query.shape.0[2],
                    key_columns: valid,
                    padded_key_columns: key_block,
                },
                None,
                weights,
                vec![],
            ));
            let weights_id = weights?;
            let v = self.copy(MidValueId(2), packed_value, vec![0, start, 0]);
            let product = self.compute(
                vec![weights_id, v],
                product_type.clone(),
                probability_value.clone(),
                Some(pv_axes),
                None,
                vec![
                    OperandWindow(vec![(2, 0, key_block)]),
                    OperandWindow::default(),
                ],
            );
            result = Some(self.compute(
                vec![product, weights_id],
                output.clone(),
                TileKernelSpec::AttentionMerge {
                    value_dimension: output.shape.0[2],
                    padded_value_dimension: value_width,
                    key_block_columns: key_block,
                    initial: start == 0,
                    final_block: materialized || start + key_block >= key_rows,
                },
                None,
                result,
                vec![],
            ));
        }
        result
    }
}

fn axis(tensor: &TensorType, axis: usize) -> Option<&AxisTiling> {
    tensor
        .format
        .layout
        .tiling
        .axes
        .iter()
        .find(|dim| dim.axis.resolve(tensor.shape.0.len()) == Ok(axis))
}

/// Project an output grid onto an operand's matching axis. Other distributed
/// coordinates become replicas while their physical strides remain unchanged.
fn project_grid(
    output: &TensorType,
    operand: &TensorType,
    output_axis: usize,
    operand_axis: usize,
) -> Option<TensorTiling> {
    let tiling = &output.format.layout.tiling;
    let strides = tiling.axis_strides().ok()?;
    let mut replicas = tiling.replicas;
    let mut axes = Vec::new();
    for (dim, stride) in tiling.axes.iter().zip(strides) {
        let index = dim.axis.resolve(output.shape.0.len()).ok()?;
        let mapped = if index == output_axis {
            Some(operand_axis)
        } else if index + 2 < output.shape.0.len() && output_axis + 2 >= output.shape.0.len() {
            operand
                .shape
                .0
                .len()
                .checked_sub(output.shape.0.len() - index)
        } else {
            None
        };
        if let Some(mapped) = mapped {
            let mut dim = *dim;
            dim.axis = TensorAxis::FromStart(mapped as u16);
            dim.tile_stride = Some(u16::try_from(stride).ok()?);
            axes.push(dim);
        } else {
            replicas = replicas.checked_mul(dim.partitions)?;
        }
    }
    Some(TensorTiling {
        tile_count: tiling.tile_count,
        replicas,
        axes,
    })
}

/// Inline compact operator implementations into the same whole-device IR.
/// This is algorithm decomposition, not tile expansion or another search.
pub(crate) fn resolve(program: &MidProgram) -> Option<MidProgram> {
    let mut result = program.clone();
    result.operations = resolve_region(&program.operations, &mut result.values, &program.outputs)?;
    Some(result)
}

fn resolve_region(
    operations: &[MidOperation],
    values: &mut Vec<MidValue>,
    required: &[MidValueId],
) -> Option<Vec<MidOperation>> {
    let mut result = Vec::new();
    let mut deferred = BTreeMap::<MidValueId, MidValueId>::new();
    let mut views = BTreeMap::<MidValueId, (MidValueId, AxisFactorView)>::new();
    for operation in operations {
        match &operation.kind {
            MidOperationKind::Operator {
                plan,
                implementation,
                ..
            } => {
                if let Some(offer) = plan.deferred_output
                    && !operation.results.iter().any(|id| required.contains(id))
                {
                    views.insert(
                        operation.results[0],
                        (operation.inputs[offer.source_input], offer.transform),
                    );
                    continue;
                }
                let input_ids = operation
                    .inputs
                    .iter()
                    .map(|id| deferred.get(id).copied().unwrap_or(*id))
                    .collect::<Vec<_>>();
                let inputs = input_ids
                    .iter()
                    .map(|id| values[id.index() as usize].tensor_type.clone())
                    .collect::<Vec<_>>();
                let output_id = *operation.results.first()?;
                let output = values[output_id.index() as usize].clone();
                let fragment = if input_ids == operation.inputs {
                    implementation.clone()
                } else {
                    None
                }
                .or_else(|| implement(plan, &inputs, &output.tensor_type))?;
                let anchor = match plan.dispatch {
                    OperatorDispatch::BlockedGemm { orientation, .. } => {
                        values[input_ids[orientation.operand_indices().0].index() as usize]
                            .tile_offset
                    }
                    _ => output.tile_offset,
                };
                let mut ids = vec![None; fragment.values.len()];
                for (input, &value) in fragment.inputs.iter().zip(&input_ids) {
                    ids[input.value.index() as usize] = Some(value);
                }
                for (&source, &target) in fragment.outputs.iter().zip(&operation.results) {
                    ids[source.index() as usize] = Some(target);
                }
                for value in &fragment.values {
                    if ids[value.id.index() as usize].is_none() {
                        let id = MidValueId(values.len() as u32);
                        ids[value.id.index() as usize] = Some(id);
                        values.push(MidValue {
                            id,
                            tile_offset: anchor,
                            tensor_type: value.tensor_type.clone(),
                            origin: output.origin,
                            storage_group: id,
                        });
                    }
                }
                for step in &fragment.operations {
                    let mut step = step.clone();
                    step.source = operation.source;
                    for value in step.inputs.iter_mut().chain(&mut step.results) {
                        *value = ids[value.index() as usize]?;
                    }
                    if let MidOperationKind::Primitive(Primitive::Copy { offsets, .. }) = &step.kind
                        && let Some(&(source, view)) = views.get(&step.inputs[0])
                    {
                        step.kind = MidOperationKind::Primitive(Primitive::MappedCopy {
                            view,
                            offsets: offsets.clone(),
                        });
                        step.inputs[0] = source;
                    }
                    if let MidOperationKind::Primitive(Primitive::Compute { operands, .. }) =
                        &step.kind
                    {
                        let offset = values[step.results[0].index() as usize].tile_offset;
                        for input in step.inputs.iter_mut().take(operands.len()) {
                            let mapping = views.get(input).copied();
                            if values[input.index() as usize].tile_offset != offset
                                || mapping.is_some()
                            {
                                let mut value = values[input.index() as usize].clone();
                                value.id = MidValueId(values.len() as u32);
                                value.tile_offset = offset;
                                value.storage_group = value.id;
                                let id = value.id;
                                values.push(value);
                                result.push(MidOperation {
                                    source: operation.source,
                                    inputs: vec![mapping.map_or(*input, |(source, _)| source)],
                                    results: vec![id],
                                    kind: MidOperationKind::Primitive(mapping.map_or(
                                        Primitive::Copy {
                                            offsets: vec![],
                                            reuse_local: true,
                                        },
                                        |(_, view)| Primitive::MappedCopy {
                                            view,
                                            offsets: vec![],
                                        },
                                    )),
                                    estimated_cycles: 0,
                                    estimated_exchange_cycles: 0,
                                });
                                *input = id;
                            }
                        }
                    }
                    result.push(step);
                }
            }
            MidOperationKind::Repeat(repeat) => {
                let mut repeat = repeat.clone();
                repeat.body.operations =
                    resolve_region(&repeat.body.operations, values, &repeat.body.yields)?;
                let mut operation = operation.clone();
                operation.kind = MidOperationKind::Repeat(repeat);
                result.push(operation);
            }
            MidOperationKind::Convert(plan) => {
                if plan.output.materialization == OperandMaterialization::DispatchSlices
                    && !operation.results.iter().any(|id| required.contains(id))
                {
                    let source = deferred
                        .get(&operation.inputs[0])
                        .copied()
                        .unwrap_or(operation.inputs[0]);
                    deferred.insert(operation.results[0], source);
                    continue;
                }
                let mut operation = operation.clone();
                let mut plan = plan.clone();
                plan.output.materialization = OperandMaterialization::Complete;
                operation.kind = MidOperationKind::Convert(plan);
                result.push(operation);
            }
            MidOperationKind::Primitive(_) => result.push(operation.clone()),
        }
    }
    Some(result)
}

pub(crate) fn same_distribution(a: &TensorType, b: &TensorType) -> bool {
    a.shape == b.shape
        && a.format.layout.tiling.tile_count == b.format.layout.tiling.tile_count
        && a.format
            .layout
            .resolve(&a.shape)
            .ok()
            .zip(b.format.layout.resolve(&b.shape).ok())
            .is_some_and(|(a, b)| {
                a.padded_shape == b.padded_shape
                    && a.axes().zip(b.axes()).is_some_and(|(a, b)| {
                        a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.same_partitioning(b))
                    })
            })
}
