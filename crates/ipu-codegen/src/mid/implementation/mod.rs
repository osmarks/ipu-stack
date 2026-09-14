//! Decompose selected algorithms into compact, whole-device tensor operations.
//! No shard enumeration, tile calls, byte spans, or physical allocation occurs here.

mod attention;
mod gemm;
mod layernorm;

use super::*;
use std::sync::Arc;

pub(crate) fn implement(
    plan: &OperatorPlan,
    inputs: &[TensorType],
    output: &TensorType,
) -> Option<Arc<MidProgram>> {
    let mut b = Builder::new(inputs);
    let result = match &plan.dispatch {
        OperatorDispatch::LayerNorm { parts } => b.layernorm(output, *parts)?,
        OperatorDispatch::Pointwise { kernel, .. } => {
            let mut operands = Vec::new();
            for (index, input) in inputs.iter().enumerate() {
                let mut resident = input.clone();
                resident.format.layout.tiling = pointwise_input_tiling(input, output)?;
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
            fp8_scales,
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
            *fp8_scales,
        )?,
    };
    b.program.outputs = vec![result];
    b.program.tile_count = b
        .program
        .values
        .iter()
        .map(|v| v.tensor_type.format.layout.tiling.tile_count)
        .max()?;
    b.program.refresh_estimates()?;
    Some(Arc::new(b.program))
}

/// Project the output's ownership onto non-broadcast operand dimensions.
/// Replicating a whole multi-row parameter and then selecting its columns
/// leaves strided views; partition it before dispatch instead.
pub(super) fn pointwise_input_tiling(
    input: &TensorType,
    output: &TensorType,
) -> Option<TensorTiling> {
    let tiling = &output.format.layout.tiling;
    if input.shape == output.shape {
        return Some(tiling.clone());
    }
    let offset = output.shape.0.len().checked_sub(input.shape.0.len())?;
    project_tiling(output, |index| {
        index
            .checked_sub(offset)
            .filter(|&axis| input.shape.0[axis] != 1)
    })
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
                output_aliases: reuse_input.map(|input| (0, input)).into_iter().collect(),
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
    project_tiling(output, |index| {
        if index == output_axis {
            Some(operand_axis)
        } else if index + 2 < output.shape.0.len() && output_axis + 2 >= output.shape.0.len() {
            operand
                .shape
                .0
                .len()
                .checked_sub(output.shape.0.len() - index)
        } else {
            None
        }
    })
}

/// Keep physical tile strides while omitted distributed axes become replicas.
fn project_tiling(
    output: &TensorType,
    map_axis: impl Fn(usize) -> Option<usize>,
) -> Option<TensorTiling> {
    let tiling = &output.format.layout.tiling;
    let mut replicas = tiling.replicas;
    let mut axes = Vec::new();
    for (dim, stride) in tiling.axes.iter().zip(tiling.axis_strides().ok()?) {
        if let Some(mapped) = map_axis(dim.axis.resolve(output.shape.0.len()).ok()?) {
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

/// Bind an executable fragment into a region. The family has already chosen
/// every operation and declared the input contract; this only remaps values.
/// Temporary ownership is relative to the selected result's owners.
pub(crate) fn append_fragment(
    fragment: &MidProgram,
    inputs: &[MidValueId],
    outputs: &[MidValueId],
    source: Option<OperationId>,
    values: &mut Vec<MidValue>,
    operations: &mut Vec<MidOperation>,
) -> Option<()> {
    if fragment.inputs.len() != inputs.len() || fragment.outputs.len() != outputs.len() {
        return None;
    }
    let output = values[outputs.first()?.index() as usize].clone();
    let mut ids = vec![None; fragment.values.len()];
    for (input, &value) in fragment.inputs.iter().zip(inputs) {
        if fragment.values[input.value.index() as usize].tensor_type
            != values[value.index() as usize].tensor_type
        {
            return None;
        }
        ids[input.value.index() as usize] = Some(value);
    }
    for (&from, &to) in fragment.outputs.iter().zip(outputs) {
        ids[from.index() as usize] = Some(to);
    }
    for value in &fragment.values {
        if ids[value.id.index() as usize].is_none() {
            let id = MidValueId(values.len() as u32);
            ids[value.id.index() as usize] = Some(id);
            values.push(MidValue {
                id,
                tile_offset: output.tile_offset,
                tensor_type: value.tensor_type.clone(),
                origin: output.origin,
                storage_group: id,
            });
        }
    }
    for step in &fragment.operations {
        let mut step = step.clone();
        step.source = source;
        for value in step.inputs.iter_mut().chain(&mut step.results) {
            *value = ids[value.index() as usize]?;
        }
        operations.push(step);
    }
    Some(())
}

pub(crate) fn same_distribution(a: &TensorType, b: &TensorType) -> bool {
    a.shape == b.shape
        && a.format.layout.tiling.tile_count == b.format.layout.tiling.tile_count
        && a.format
            .layout
            .resolve(&a.shape)
            .ok()
            .zip(b.format.layout.resolve(&b.shape).ok())
            .is_some_and(|(left, right)| {
                left.padded_shape == right.padded_shape
                    && (a.format.layout.tiling == b.format.layout.tiling
                        || left.axes().zip(right.axes()).is_some_and(|(a, b)| {
                            a.len() == b.len()
                                && a.iter().zip(b).all(|(a, b)| a.same_partitioning(b))
                        }))
            })
}
