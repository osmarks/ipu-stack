//! Direct executable mid construction for selected operation families.
//! Family algorithms share this value/operation builder; binding and costing
//! consume the resulting MidProgram without another template expansion.
use crate::graph::{GraphInputKind, ValueId};
use crate::kernel::TileKernelSpec;
use crate::mid::{
    Compute, CoordinateMapping, LocalSite, MidInput, MidOperation, MidOperationKind, MidProgram,
    MidValue, MidValueId, OperandIndexing,
};
use crate::planner::operator::{OperatorDispatch, OperatorFamily, OperatorPlan};
use crate::tensor::{
    Precision, TensorTiling, TensorType, broadcast_operand_tiling, project_tiling,
    same_distribution,
};
use std::sync::Arc;

pub(crate) fn build_fragment(
    plan: &OperatorPlan,
    inputs: &[TensorType],
    output: &TensorType,
) -> Option<Arc<MidProgram>> {
    let mut b = FragmentBuilder::new(inputs);
    let result = match &plan.dispatch {
        OperatorDispatch::LayerNorm { parts } => b.layernorm(output, *parts)?,
        OperatorDispatch::LocalKernel => {
            let kernel = plan.operator.local_kernel()?;
            // The fused attention callable consumes already-selected local
            // Q/K/V panels. Its row domains differ from the result's domain.
            let indexing = if matches!(kernel, TileKernelSpec::FlashAttention { .. }) {
                OperandIndexing::local()
            } else {
                OperandIndexing::Elementwise { result: 0 }
            };
            let mut operands = Vec::new();
            for (index, input) in inputs.iter().enumerate() {
                let mut resident = input.clone();
                if matches!(indexing, OperandIndexing::Elementwise { .. }) {
                    resident.format.layout.tiling = broadcast_operand_tiling(input, output)?;
                }
                operands.push(b.copy(
                    LocalSite::from("operand").at(index as u32),
                    MidValueId::from_index(index as u32),
                    resident,
                    vec![],
                ));
            }
            let reuse = plan.reuse_inputs.as_ref().and_then(|indices| {
                indices.iter().find_map(|&index| {
                    let input = *operands.get(usize::from(index))?;
                    (b.tensor(input) == output).then_some(input)
                })
            });
            let mut kernel_output = output.clone();
            if matches!(kernel, TileKernelSpec::FlashAttention { .. }) {
                kernel_output.format.precision = Precision::F32;
            }
            let indexing = vec![indexing; operands.len()];
            let result = b.kernel("compute", operands, kernel_output, kernel, reuse, indexing);
            b.cast("output.cast", result, output.format.precision)
        }
        OperatorDispatch::View => {
            let mapping = match plan.operator {
                OperatorFamily::View(view) => view.into(),
                OperatorFamily::Slice(slice) => {
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
                "view",
                vec![MidValueId::from_index(0)],
                [output.clone()],
                MidOperationKind::Copy {
                    policy: crate::CopyPolicy::Automatic,
                    packing: crate::PackingPolicy::Automatic,
                    mapping,
                    reuse_local: false,
                },
            )[0]
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

pub(super) struct FragmentBuilder {
    pub(super) program: MidProgram,
}

impl FragmentBuilder {
    pub(super) fn new(inputs: &[TensorType]) -> Self {
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

    pub(super) fn value(&mut self, tensor_type: TensorType) -> MidValueId {
        let id = MidValueId::from_index(self.program.values.len() as u32);
        self.program.values.push(MidValue {
            id,
            owners: crate::tensor::OwnerMap::default(),
            tensor_type,
            origin: ValueId::from_index(0),
            storage_group: id,
        });
        id
    }

    pub(super) fn tensor(&self, value: MidValueId) -> &TensorType {
        &self.program.values[value.index() as usize].tensor_type
    }

    pub(super) fn emit(
        &mut self,
        site: impl Into<LocalSite>,
        inputs: Vec<MidValueId>,
        outputs: impl IntoIterator<Item = TensorType>,
        kind: MidOperationKind,
    ) -> Vec<MidValueId> {
        let results = outputs
            .into_iter()
            .map(|output| self.value(output))
            .collect::<Vec<_>>();
        self.program.operations.push(MidOperation {
            site: Some(site.into()),
            source: None,
            inputs,
            results: results.clone(),
            kind,
        });
        results
    }

    pub(super) fn cast(
        &mut self,
        site: impl Into<LocalSite>,
        input: MidValueId,
        precision: Precision,
    ) -> MidValueId {
        let mut output = self.tensor(input).clone();
        let from = output.format.precision;
        if from == precision {
            return input;
        }
        output.format.precision = precision;
        self.kernel(
            site,
            vec![input],
            output,
            TileKernelSpec::Cast {
                from,
                to: precision,
            },
            None,
            vec![OperandIndexing::Elementwise { result: 0 }],
        )
    }

    pub(super) fn copy(
        &mut self,
        site: impl Into<LocalSite>,
        input: MidValueId,
        output: TensorType,
        offsets: Vec<u32>,
    ) -> MidValueId {
        self.materialize(site, input, output, offsets, true)
    }

    pub(super) fn materialize(
        &mut self,
        site: impl Into<LocalSite>,
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
            site,
            vec![input],
            [output],
            MidOperationKind::Copy {
                policy: crate::CopyPolicy::Automatic,
                packing: crate::PackingPolicy::Automatic,
                mapping: CoordinateMapping {
                    offsets,
                    view: None,
                },
                reuse_local,
            },
        )[0]
    }

    pub(super) fn kernel(
        &mut self,
        site: impl Into<LocalSite>,
        inputs: Vec<MidValueId>,
        output: TensorType,
        kernel: TileKernelSpec,
        reuse: Option<MidValueId>,
        operands: Vec<OperandIndexing>,
    ) -> MidValueId {
        self.compute(
            site,
            inputs,
            [(output, reuse)],
            Compute::Kernel {
                kernel,
                operands,
                output_aliases: Vec::new(),
            },
        )[0]
    }

    pub(super) fn compute(
        &mut self,
        site: impl Into<LocalSite>,
        mut inputs: Vec<MidValueId>,
        outputs: impl IntoIterator<Item = (TensorType, Option<MidValueId>)>,
        mut compute: Compute,
    ) -> Vec<MidValueId> {
        let mut types = Vec::new();
        for (output, reuse) in outputs {
            if let Some(value) = reuse {
                let aliases = match &mut compute {
                    Compute::Product(product) => &mut product.output_aliases,
                    Compute::Kernel { output_aliases, .. } => output_aliases,
                    Compute::Sum { .. } => unreachable!("sum output ownership is independent"),
                };
                aliases.push((types.len(), inputs.len()));
                inputs.push(value);
            }
            types.push(output);
        }
        self.emit(site, inputs, types, MidOperationKind::Compute(compute))
    }
}

/// Project an output grid onto an operand's matching axis. Other distributed
/// coordinates become replicas while their physical strides remain unchanged.
pub(super) fn project_grid(
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
