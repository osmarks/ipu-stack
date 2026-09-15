//! Attention stage build recipes. Assembly workers take block sizes at runtime;
//! every stage shares its worker code across query-row counts.

use super::*;
use crate::ShardView;
use crate::mid::MidOperationKind;

/// One contiguous workspace: optional field/component axes surround the
/// flattened query rows. The same declaration constructs mid tensors and checks
/// local call geometry; it does not prescribe their physical tile ownership.
struct RowWorkspace {
    precision: Precision,
    leading: Option<u32>,
    trailing: Option<u32>,
}

const SOFTMAX_WORKSPACES: [RowWorkspace; 3] = [
    RowWorkspace {
        precision: Precision::F32,
        leading: Some(2),
        trailing: None,
    }, // maximum, denominator
    RowWorkspace {
        precision: Precision::F32,
        leading: Some(2),
        trailing: Some(3),
    }, // max/sum, segment
    RowWorkspace {
        precision: Precision::F16,
        leading: None,
        trailing: Some(16),
    }, // masked FP8 tail
];

fn softmax_workspace_specs(precision: Precision, masked: bool) -> Option<&'static [RowWorkspace]> {
    let count = match precision {
        Precision::F16 => 2,
        Precision::F8F143 { .. } => 2 + usize::from(masked),
        _ => return None,
    };
    Some(&SOFTMAX_WORKSPACES[..count])
}

impl RowWorkspace {
    fn tensor(&self, probabilities: &crate::TensorType) -> Option<crate::TensorType> {
        let rank = probabilities.shape.0.len();
        if rank < 2 {
            return None;
        }
        let mut tensor = probabilities.clone();
        tensor.shape.0 = self
            .leading
            .into_iter()
            .chain(probabilities.shape.0[..rank - 1].iter().copied())
            .chain(self.trailing)
            .collect();
        tensor.format.precision = self.precision;
        tensor.format.layout.order = ElementOrder::RowMajor;
        tensor.format.layout.tiling = crate::tensor::project_tiling(probabilities, |axis| {
            (axis + 1 < rank).then_some(axis + usize::from(self.leading.is_some()))
        })?;
        Some(tensor)
    }

    fn accepts(&self, access: &KernelAccess, view: &ShardView, rows: u32) -> bool {
        if access.format.precision != self.precision
            || access.format.layout.order != ElementOrder::RowMajor
        {
            return false;
        }
        let mut dimensions = view.extents.as_slice();
        if let Some(width) = self.leading {
            let Some((axis, rest)) = dimensions.split_first() else {
                return false;
            };
            if axis.physical_end - axis.start != width {
                return false;
            }
            dimensions = rest;
        }
        if let Some(width) = self.trailing {
            let Some((axis, rest)) = dimensions.split_last() else {
                return false;
            };
            if axis.physical_end - axis.start != width {
                return false;
            }
            dimensions = rest;
        }
        element_count(dimensions) == Ok(rows)
    }
}

/// Persistent FP32 statistics and private FP32/F16 work storage. Probability
/// values contain no state bytes, and merge consumes only the statistics result.
pub(crate) fn softmax_workspaces(
    probabilities: &crate::TensorType,
    masked: bool,
) -> Option<Vec<crate::TensorType>> {
    softmax_workspace_specs(probabilities.format.precision, masked)?
        .iter()
        .map(|spec| spec.tensor(probabilities))
        .collect()
}

pub(super) fn call(run: &KernelRun) -> Result<KernelCall, KernelAbiError> {
    let output = run.requirements.outputs[0].format.precision;
    let (implementation, arguments) = match run.kernel {
        MidOperationKind::FlashAttention { .. } => {
            run.check_arity(3, 1)?;
            if output != Precision::F32
                || run
                    .requirements
                    .inputs
                    .iter()
                    .any(|input| input.format.precision != Precision::F16)
            {
                return Err(KernelAbiError::Unavailable(run.kernel.clone()));
            }
            (
                KernelImplementation::Attention(attention_shape(run)?),
                Vec::new(),
            )
        }
        MidOperationKind::AttentionSoftmax {
            head_dimension,
            key_columns,
            padded_key_columns,
        } => {
            let workspaces = softmax_workspace_specs(output, key_columns != padded_key_columns)
                .ok_or(KernelAbiError::RequirementMismatch)?;
            run.check_arity(1, 1 + workspaces.len())?;
            let rows = gemm_rows(run)?;
            if key_columns == 0
                || key_columns > padded_key_columns
                || run.requirements.inputs[0].format.precision != Precision::F16
                || run.requirements.inputs[0].format.layout.order
                    != ElementOrder::Amp(AmpOrder::Left)
                || run.requirements.outputs[0].format.layout.order
                    != ElementOrder::Amp(AmpOrder::Left)
                || matrix_extent(&run.outputs[0], false, true)? != padded_key_columns
                || input_matrix_extent(run, false, true)? != padded_key_columns
                || element_count(&run.inputs[0].extents[..run.inputs[0].extents.len() - 1])? != rows
                || workspaces
                    .iter()
                    .zip(run.requirements.outputs[1..].iter().zip(&run.outputs[1..]))
                    .any(|(workspace, (access, view))| !workspace.accepts(access, view, rows))
            {
                return Err(KernelAbiError::RequirementMismatch);
            }
            (
                KernelImplementation::Softmax(
                    head_dimension,
                    key_columns,
                    padded_key_columns,
                    output,
                ),
                vec![
                    rows,
                    key_columns,
                    u32::from(cost::f16_softmax_split_rows(
                        u64::from(rows),
                        u64::from(key_columns),
                        u64::from(padded_key_columns),
                    )),
                ],
            )
        }
        MidOperationKind::AttentionMerge {
            value_dimension,
            padded_value_dimension,
            initial,
            final_block,
        } => {
            if output != Precision::F32 && !(output == Precision::F16 && final_block) {
                return Err(KernelAbiError::Unavailable(run.kernel.clone()));
            }
            let previous = output == Precision::F16 && !initial;
            run.check_arity(if previous { 3 } else { 2 }, 1)?;
            let rows = gemm_rows(run)?;
            let accumulator_width = value_dimension
                .checked_add(2)
                .and_then(|width| width.div_ceil(16).checked_mul(16))
                .ok_or(KernelAbiError::ElementCountOverflow)?;
            if value_dimension == 0
                || value_dimension > padded_value_dimension
                || run.requirements.inputs[0].format.precision != Precision::F16
                || run.requirements.inputs[0].format.layout.order
                    != ElementOrder::Amp(AmpOrder::Left)
                || input_matrix_extent(run, false, true)? != padded_value_dimension
                || element_count(&run.inputs[0].extents[..run.inputs[0].extents.len() - 1])? != rows
                || run.requirements.outputs[0].format.layout.order != ElementOrder::RowMajor
                || matrix_extent(&run.outputs[0], false, true)?
                    != if output == Precision::F16 {
                        padded_value_dimension
                    } else {
                        accumulator_width
                    }
                || !SOFTMAX_WORKSPACES[0].accepts(&run.requirements.inputs[1], &run.inputs[1], rows)
                || (previous
                    && (run.requirements.inputs[2].format.precision != Precision::F32
                        || run.requirements.inputs[2].format.layout.order
                            != ElementOrder::RowMajor
                        || matrix_extent(&run.inputs[2], false, true)? != accumulator_width
                        || element_count(
                            &run.inputs[2].extents[..run.inputs[2].extents.len() - 1],
                        )? != rows))
            {
                return Err(KernelAbiError::RequirementMismatch);
            }
            // The initial FP16 stage has no previous accumulator. Supply its
            // unused ABI pointer slot here, without a fabricated mid operand.
            let arguments = (output == Precision::F16 && initial)
                .then_some(0)
                .into_iter()
                .chain([u32::from(initial), u32::from(final_block), rows])
                .collect();
            (
                KernelImplementation::Merge(value_dimension, padded_value_dimension, output),
                arguments,
            )
        }
        _ => return Err(KernelAbiError::RequirementMismatch),
    };
    Ok(KernelCall {
        implementation,
        arguments,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct AttentionKernelShape {
    pub(crate) matrices: u32,
    pub(crate) query_rows: u32,
    pub(crate) key_rows: u32,
    pub(crate) query_dimension: u32,
    pub(crate) value_dimension: u32,
    pub(crate) scale_bits: u32,
}

pub(crate) fn attention_shape(run: &KernelRun) -> Result<AttentionKernelShape, KernelAbiError> {
    let MidOperationKind::FlashAttention {
        options,
        accumulate,
    } = &run.kernel
    else {
        return Err(KernelAbiError::RequirementMismatch);
    };
    if options.causal || *accumulate != crate::AccumulationPrecision::F32 {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let [query, key, value] = run.inputs.as_slice() else {
        return Err(KernelAbiError::RequirementMismatch);
    };
    let extents = |view: &ShardView| {
        view.extents
            .iter()
            .map(|extent| extent.physical_end - extent.start)
            .collect::<Vec<_>>()
    };
    let query = extents(query);
    let key = extents(key);
    let value = extents(value);
    if query.len() < 2 || query.len() != key.len() || query.len() != value.len() {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let rank = query.len();
    if query[..rank - 2] != key[..rank - 2]
        || query[..rank - 2] != value[..rank - 2]
        || query[rank - 1] != key[rank - 1]
        || key[rank - 2] != value[rank - 2]
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let matrices = query[..rank - 2]
        .iter()
        .try_fold(1u32, |product, &extent| product.checked_mul(extent))
        .ok_or(KernelAbiError::ElementCountOverflow)?;
    let scale = options
        .scale
        .as_value()
        .unwrap_or_else(|| 1.0 / (query[rank - 1] as f32).sqrt());
    Ok(AttentionKernelShape {
        matrices,
        query_rows: query[rank - 2],
        key_rows: key[rank - 2],
        query_dimension: query[rank - 1],
        value_dimension: value[rank - 1],
        scale_bits: scale.to_bits(),
    })
}

impl KernelBuildPlan {
    pub(super) fn add_attention(&mut self, shape: AttentionKernelShape) {
        let suffix = format!(
            "m{}_q{}_k{}_d{}_v{}_{:08x}",
            shape.matrices,
            shape.query_rows,
            shape.key_rows,
            shape.query_dimension,
            shape.value_dimension,
            shape.scale_bits,
        );
        let call_symbol = format!("flash_attention_online_f16_{suffix}");
        let vertex = format!("FlashAttentionOnlineF16_{suffix}");
        let flags = vec![
            format!("-DATTENTION_MATRICES={}", shape.matrices),
            format!("-DATTENTION_QUERY_ROWS={}", shape.query_rows),
            format!("-DATTENTION_KEY_ROWS={}", shape.key_rows),
            format!("-DATTENTION_QUERY_DIMENSION={}", shape.query_dimension),
            format!("-DATTENTION_VALUE_DIMENSION={}", shape.value_dimension),
            format!("-DATTENTION_SCALE={}", f32::from_bits(shape.scale_bits)),
            format!("-DATTENTION_VERTEX_NAME={vertex}"),
        ];
        self.add_vertex(
            "flash_attention_online_f16.cpp",
            &call_symbol,
            &vertex,
            flags,
            &[3, 4, 5, 2],
        );
        self.symbols
            .insert(KernelImplementation::Attention(shape), call_symbol);
    }

    pub(super) fn add_attention_stages(
        &mut self,
        stages: BTreeSet<KernelImplementation>,
    ) -> Result<(), KernelAbiError> {
        let mut compiled = BTreeSet::new();
        for key in stages {
            let (name, symbol, source, flags) = match key {
                KernelImplementation::Softmax(head, keys, padded, precision) => {
                    let full = keys == padded;
                    let name = format!(
                        "attention_softmax_d{head}_p{padded}_{}",
                        if full { "full" } else { "tail" }
                    );
                    let symbol = match precision {
                        Precision::F16 => format!("{name}_f16"),
                        Precision::F8F143 { scale_exponent } => {
                            format!("{name}_f8_s{scale_exponent}")
                        }
                        _ => return Err(KernelAbiError::RequirementMismatch),
                    };
                    let name = symbol.replace('-', "m");
                    let symbol = name.clone();
                    let scale_bits = (1.0_f32 / (head as f32).sqrt()).to_bits();
                    let mut flags = vec![
                        format!("-DATTENTION_HEAD_DIMENSION={head}"),
                        format!("-DATTENTION_FULL_BLOCK={}", u8::from(full)),
                        format!("-DATTENTION_KEY_BLOCK_COLUMNS={padded}"),
                        format!("-DATTENTION_SCALE_BITS=0x{scale_bits:08x}"),
                        format!("-DATTENTION_SOFTMAX_SYMBOL={symbol}"),
                    ];
                    if let Precision::F8F143 { scale_exponent } = precision {
                        if !padded.is_multiple_of(32) {
                            return Err(KernelAbiError::RequirementMismatch);
                        }
                        flags.extend([
                            "-DATTENTION_OUTPUT_F8".into(),
                            format!("-DATTENTION_OUTPUT_SCALE={scale_exponent}"),
                        ]);
                    }
                    (name, symbol, "attention_softmax_f16.S", flags)
                }
                KernelImplementation::Merge(values, padded, output) => {
                    let suffix = match output {
                        Precision::F16 => "out16",
                        Precision::F32 => "out32",
                        _ => return Err(KernelAbiError::RequirementMismatch),
                    };
                    let name = format!("attention_merge_v{values}_p{padded}_{suffix}");
                    let symbol = format!("{name}_f16");
                    let flags = vec![
                        format!("-DATTENTION_VALUE_DIMENSION={values}"),
                        format!("-DATTENTION_PADDED_VALUE_DIMENSION={padded}"),
                        format!("-DATTENTION_MERGE_SYMBOL={symbol}"),
                        format!(
                            "-DATTENTION_MERGE_OUTPUT_F16={}",
                            u8::from(output == Precision::F16)
                        ),
                    ];
                    (name, symbol, "attention_merge_f16.S", flags)
                }
                _ => return Err(KernelAbiError::RequirementMismatch),
            };
            self.symbols.insert(key, symbol.clone());
            if compiled.insert(symbol.clone()) {
                self.compilations.push(KernelCompilation {
                    source,
                    name,
                    flags,
                });
            }
        }
        Ok(())
    }
}
