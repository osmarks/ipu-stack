//! Decompose selected algorithms into compact, whole-device tensor operations.
//! No shard enumeration, tile calls, byte spans, or physical allocation occurs here.

mod attention;
mod gemm;

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
            let mut kernel_output = output.clone();
            if matches!(kernel, TileKernelSpec::FlashAttention { .. }) {
                kernel_output.format.precision = Precision::F32;
            }
            let result = b.compute(operands, kernel_output, kernel.clone(), None, reuse, vec![]);
            b.cast(result, output.format.precision)
        }
        OperatorDispatch::View => {
            let mapping = match plan.operator {
                MidOperator::View(view) => view.into(),
                MidOperator::Slice(slice) => {
                    let mut offsets = vec![0; inputs[0].shape.0.len()];
                    offsets[slice.axis] = slice.start;
                    CoordinateMapping {
                        offsets,
                        view: None,
                    }
                }
                _ => return None,
            };
            b.emit(
                vec![MidValueId(0)],
                output.clone(),
                Primitive::Copy {
                    mapping,
                    reuse_local: false,
                },
            )
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
        OperatorDispatch::Attention {
            query_key,
            probability_value,
            materialized,
            key_block_rows,
            padded_query_dimension,
            padded_value_dimension,
        } => b.attention(
            output,
            *key_block_rows,
            *padded_query_dimension,
            *padded_value_dimension,
            *materialized,
            *query_key,
            *probability_value,
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

    fn cast(&mut self, input: MidValueId, precision: Precision) -> MidValueId {
        let mut output = self.tensor(input).clone();
        let from = output.format.precision;
        if from == precision {
            return input;
        }
        output.format.precision = precision;
        self.compute(
            vec![input],
            output,
            TileKernelSpec::Cast {
                from,
                to: precision,
            },
            None,
            None,
            vec![],
        )
    }

    fn copy(&mut self, input: MidValueId, output: TensorType, offsets: Vec<u32>) -> MidValueId {
        self.materialize(input, output, offsets, true)
    }

    fn materialize(
        &mut self,
        input: MidValueId,
        output: TensorType,
        offsets: Vec<u32>,
        reuse_local: bool,
    ) -> MidValueId {
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
                mapping: CoordinateMapping {
                    offsets,
                    view: None,
                },
                reuse_local,
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
pub(crate) fn resolve(mut program: MidProgram) -> Option<MidProgram> {
    program.operations =
        resolve_region(&program.operations, &mut program.values, &program.outputs)?;
    Some(program)
}

fn resolve_region(
    operations: &[MidOperation],
    values: &mut Vec<MidValue>,
    required: &[MidValueId],
) -> Option<Vec<MidOperation>> {
    let mut result = Vec::new();
    let mut deferred = BTreeMap::<MidValueId, MidValueId>::new();
    for operation in operations {
        match &operation.kind {
            MidOperationKind::Operator {
                plan,
                implementation,
                ..
            } => {
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
                    if let MidOperationKind::Primitive(Primitive::Compute { operands, .. }) =
                        &step.kind
                    {
                        let offset = values[step.results[0].index() as usize].tile_offset;
                        for input in step.inputs.iter_mut().take(operands.len()) {
                            if values[input.index() as usize].tile_offset != offset {
                                let mut value = values[input.index() as usize].clone();
                                value.id = MidValueId(values.len() as u32);
                                value.tile_offset = offset;
                                value.storage_group = value.id;
                                let id = value.id;
                                values.push(value);
                                result.push(MidOperation {
                                    source: operation.source,
                                    inputs: vec![*input],
                                    results: vec![id],
                                    kind: MidOperationKind::Primitive(Primitive::Copy {
                                        mapping: CoordinateMapping::default(),
                                        reuse_local: true,
                                    }),
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
    super::copy::compose(&mut result, values, required);
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
