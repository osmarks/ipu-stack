//! Attention stage build recipes. Assembly workers take block sizes at runtime;
//! every stage shares its worker code across query-row counts.

use super::*;
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

    fn accepts(&self, geometry: TensorStorage<'_>, rows: u32) -> bool {
        if geometry.format.precision != self.precision
            || geometry.format.layout.order != ElementOrder::RowMajor
        {
            return false;
        }
        let mut dimensions = geometry.extents.iter().copied();
        for (width, leading) in [(self.leading, true), (self.trailing, false)] {
            if let Some(width) = width {
                let axis = if leading {
                    dimensions.next()
                } else {
                    dimensions.next_back()
                };
                if axis.is_none_or(|axis| axis.physical_end - axis.start != width) {
                    return false;
                }
            }
        }
        dimensions.try_fold(1u32, |count, axis| {
            count.checked_mul(axis.physical_end - axis.start)
        }) == Some(rows)
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

pub(super) fn call(
    kernel: &MidOperationKind,
    inputs: &[TensorStorage<'_>],
    outputs: &[TensorStorage<'_>],
    build: Option<&mut KernelObjects>,
) -> Result<KernelCall, KernelError> {
    let output = outputs
        .first()
        .ok_or(KernelError::RequirementMismatch)?
        .format
        .precision;
    let call = match *kernel {
        MidOperationKind::FlashAttention { .. } => {
            check_arity(inputs, outputs, 3, 1)?;
            if output != Precision::F32
                || inputs
                    .iter()
                    .any(|input| input.format.precision != Precision::F16)
            {
                return Err(KernelError::Unavailable(kernel.clone()));
            }
            flash(attention_shape(kernel, inputs)?, build)
        }
        MidOperationKind::AttentionSoftmax {
            head_dimension,
            key_columns,
            padded_key_columns,
        } => {
            let workspaces = softmax_workspace_specs(output, key_columns != padded_key_columns)
                .ok_or(KernelError::RequirementMismatch)?;
            check_arity(inputs, outputs, 1, 1 + workspaces.len())?;
            let rows = gemm_rows(outputs[0])?;
            if key_columns == 0
                || key_columns > padded_key_columns
                || (matches!(output, Precision::F8F143 { .. })
                    && !padded_key_columns.is_multiple_of(32))
                || inputs[0].format.precision != Precision::F16
                || inputs[0].format.layout.order != ElementOrder::Amp(AmpOrder::Left)
                || outputs[0].format.layout.order != ElementOrder::Amp(AmpOrder::Left)
                || outputs[0].matrix_extent(false, true)? != padded_key_columns
                || inputs[0].matrix_extent(false, true)? != padded_key_columns
                || u32::try_from(inputs[0].rows()).map_err(|_| KernelError::ElementCountOverflow)?
                    != rows
                || workspaces
                    .iter()
                    .zip(&outputs[1..])
                    .any(|(workspace, &geometry)| !workspace.accepts(geometry, rows))
            {
                return Err(KernelError::RequirementMismatch);
            }
            let full = key_columns == padded_key_columns;
            let head = head_dimension;
            let padded = padded_key_columns;
            let name = format!(
                "attention_softmax_d{head}_p{padded}_{}",
                if full { "full" } else { "tail" }
            );
            let symbol = match output {
                Precision::F16 => format!("{name}_f16"),
                Precision::F8F143 { scale_exponent } => {
                    format!("{name}_f8_s{scale_exponent}").replace('-', "m")
                }
                _ => return Err(KernelError::RequirementMismatch),
            };
            if let Some(build) = build {
                let scale_bits = (1.0_f32 / (head as f32).sqrt()).to_bits();
                let mut flags = vec![
                    format!("-DATTENTION_HEAD_DIMENSION={head}"),
                    format!("-DATTENTION_FULL_BLOCK={}", u8::from(full)),
                    format!("-DATTENTION_KEY_BLOCK_COLUMNS={padded}"),
                    format!("-DATTENTION_SCALE_BITS=0x{scale_bits:08x}"),
                    format!("-DATTENTION_SOFTMAX_SYMBOL={symbol}"),
                ];
                if let Precision::F8F143 { scale_exponent } = output {
                    flags.extend([
                        "-DATTENTION_OUTPUT_F8".into(),
                        format!("-DATTENTION_OUTPUT_SCALE={scale_exponent}"),
                    ]);
                }
                build.add_compilation(KernelCompilation {
                    source: "attention_softmax_f16.S",
                    name: symbol.clone(),
                    flags,
                });
            }
            let split = f16_softmax_split_rows(rows.into(), key_columns.into(), padded.into());
            KernelCall::new(
                symbol,
                vec![rows, key_columns, u32::from(split)],
                softmax_output_cycles(
                    rows.into(),
                    key_columns.into(),
                    padded.into(),
                    matches!(output, Precision::F8F143 { .. }),
                    split,
                ),
            )
        }
        MidOperationKind::AttentionMerge {
            value_dimension,
            padded_value_dimension,
            initial,
            final_block,
        } => {
            if output != Precision::F32 && !(output == Precision::F16 && final_block) {
                return Err(KernelError::Unavailable(kernel.clone()));
            }
            let previous = output == Precision::F16 && !initial;
            check_arity(inputs, outputs, if previous { 3 } else { 2 }, 1)?;
            let rows = gemm_rows(outputs[0])?;
            let accumulator_width = value_dimension
                .checked_add(2)
                .and_then(|width| width.div_ceil(16).checked_mul(16))
                .ok_or(KernelError::ElementCountOverflow)?;
            if value_dimension == 0
                || value_dimension > padded_value_dimension
                || inputs[0].format.precision != Precision::F16
                || inputs[0].format.layout.order != ElementOrder::Amp(AmpOrder::Left)
                || inputs[0].matrix_extent(false, true)? != padded_value_dimension
                || u32::try_from(inputs[0].rows()).map_err(|_| KernelError::ElementCountOverflow)?
                    != rows
                || outputs[0].format.layout.order != ElementOrder::RowMajor
                || outputs[0].matrix_extent(false, true)?
                    != if output == Precision::F16 {
                        padded_value_dimension
                    } else {
                        accumulator_width
                    }
                || !SOFTMAX_WORKSPACES[0].accepts(inputs[1], rows)
                || (previous
                    && (inputs[2].format.precision != Precision::F32
                        || inputs[2].format.layout.order != ElementOrder::RowMajor
                        || inputs[2].matrix_extent(false, true)? != accumulator_width
                        || u32::try_from(inputs[2].rows())
                            .map_err(|_| KernelError::ElementCountOverflow)?
                            != rows))
            {
                return Err(KernelError::RequirementMismatch);
            }
            // The initial FP16 stage has no previous accumulator. Supply its
            // unused ABI pointer slot here, without a fabricated mid operand.
            let arguments = (output == Precision::F16 && initial)
                .then_some(0)
                .into_iter()
                .chain([u32::from(initial), u32::from(final_block), rows])
                .collect();
            let values = value_dimension;
            let padded = padded_value_dimension;
            let suffix = if output == Precision::F16 {
                "out16"
            } else {
                "out32"
            };
            let name = format!("attention_merge_v{values}_p{padded}_{suffix}");
            let symbol = format!("{name}_f16");
            if let Some(build) = build {
                build.add_compilation(KernelCompilation {
                    source: "attention_merge_f16.S",
                    name,
                    flags: vec![
                        format!("-DATTENTION_VALUE_DIMENSION={values}"),
                        format!("-DATTENTION_PADDED_VALUE_DIMENSION={padded}"),
                        format!("-DATTENTION_MERGE_SYMBOL={symbol}"),
                        format!(
                            "-DATTENTION_MERGE_OUTPUT_F16={}",
                            u8::from(output == Precision::F16)
                        ),
                    ],
                });
            }
            KernelCall::new(
                symbol,
                arguments,
                f16_attention_merge_cycles(
                    rows.into(),
                    values.into(),
                    initial,
                    final_block,
                    output == Precision::F16,
                ),
            )
        }
        _ => return Err(KernelError::RequirementMismatch),
    };
    Ok(call)
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

fn attention_shape<'a>(
    kernel: &MidOperationKind,
    inputs: &[TensorStorage<'a>],
) -> Result<AttentionKernelShape, KernelError> {
    let MidOperationKind::FlashAttention {
        options,
        accumulate,
    } = kernel
    else {
        return Err(KernelError::RequirementMismatch);
    };
    if options.causal || *accumulate != AccumulationPrecision::F32 {
        return Err(KernelError::RequirementMismatch);
    }
    let (Some(query), Some(key), Some(value), None) = (
        inputs.first().copied(),
        inputs.get(1).copied(),
        inputs.get(2).copied(),
        inputs.get(3),
    ) else {
        return Err(KernelError::RequirementMismatch);
    };
    let rank = query.extents.len();
    if rank < 2
        || key.extents.len() != rank
        || value.extents.len() != rank
        || (0..rank - 2).any(|axis| {
            query.dimension(axis) != key.dimension(axis)
                || query.dimension(axis) != value.dimension(axis)
        })
        || query.dimension(rank - 1) != key.dimension(rank - 1)
        || key.dimension(rank - 2) != value.dimension(rank - 2)
    {
        return Err(KernelError::RequirementMismatch);
    }
    let matrices = query
        .widths()
        .take(rank - 2)
        .try_fold(1u32, |count, width| count.checked_mul(width))
        .ok_or(KernelError::ElementCountOverflow)?;
    let scale = options
        .scale
        .as_value()
        .unwrap_or_else(|| 1.0 / (query.dimension(rank - 1) as f32).sqrt());
    Ok(AttentionKernelShape {
        matrices,
        query_rows: query.dimension(rank - 2),
        key_rows: key.dimension(rank - 2),
        query_dimension: query.dimension(rank - 1),
        value_dimension: value.dimension(rank - 1),
        scale_bits: scale.to_bits(),
    })
}

fn flash(shape: AttentionKernelShape, build: Option<&mut KernelObjects>) -> KernelCall {
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
    if let Some(build) = build {
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
        build.add_vertex(
            "flash_attention_online_f16.cpp",
            &call_symbol,
            &vertex,
            flags,
            &[3, 4, 5, 2],
            "worker_call.S",
            Vec::new(),
        );
    }
    let cycles = u64::from(shape.matrices)
        .saturating_mul(u64::from(shape.query_rows))
        .saturating_mul(u64::from(shape.key_rows))
        .saturating_mul(u64::from(shape.query_dimension) + u64::from(shape.value_dimension))
        .saturating_mul(4)
        .div_ceil(6)
        .saturating_add(crate::estimate::IPU21_TARGET_COSTS.kernel_launch_cycles);
    KernelCall::new(call_symbol, Vec::new(), cycles)
}

/// Row-wise softmax: four-wide maxima and pipelined MIX/exp/store/sum take 41
/// issue groups per full 16-key panel. Masked pairs and zero padding use short
/// scalar loops; no tile program needs to be constructed to price them.
fn f16_softmax_whole_rows(rows: u64, keys: u64, padded_keys: u64) -> u64 {
    if rows == 0 {
        return 0;
    }
    let full_panels = keys / 16;
    let mut row = 28u64
        .saturating_add(2 * u64::from(full_panels != 0))
        .saturating_add(full_panels.saturating_mul(41));
    let launch = if keys == padded_keys {
        row = row.saturating_add(7);
        222u64
    } else {
        row = row.saturating_add(5 + 6 * u64::from(full_panels != 0));
        let pairs = (keys % 16) / 2;
        let zero_pairs = 8 - (keys % 16).div_ceil(2);
        let zero_panels = (padded_keys / 16).saturating_sub(full_panels.saturating_add(1));
        row = row
            .saturating_add(21)
            .saturating_add(if full_panels != 0 { 4 } else { 2 })
            .saturating_add(2 * u64::from(pairs != 0))
            .saturating_add(12 * pairs + 12 * (keys % 2))
            .saturating_add(u64::from(zero_pairs != 0) + 2 * zero_pairs)
            .saturating_add(u64::from(zero_panels != 0))
            .saturating_add(zero_panels.saturating_mul(10));
        234
    };
    launch.saturating_add(rows.div_ceil(6).saturating_mul(6).saturating_mul(row))
}

// Three local stages: partial maxima, exponentials/partial sums, final sums.
// A segment has ceil(padded_keys / 48) panels. 123 groups account for each
// segment's setup, row-state reductions, and address calculations; 408 cycles
// cover the launches and 21 groups per final worker wave reduce the sums.
fn f16_softmax_split_cycles(rows: u64, keys: u64, padded_keys: u64) -> u64 {
    if keys < 128 || rows == 0 {
        return u64::MAX;
    }
    let segment = padded_keys
        .div_ceil(48)
        .saturating_mul(41)
        .saturating_add(123);
    408u64
        .saturating_add(rows.div_ceil(2).saturating_mul(6).saturating_mul(segment))
        .saturating_add(rows.div_ceil(6).saturating_mul(126))
}

/// The ABI and the planner use the same choice; no tile program is built here.
pub(crate) fn f16_softmax_split_rows(rows: u64, keys: u64, padded_keys: u64) -> bool {
    f16_softmax_split_cycles(rows, keys, padded_keys)
        < f16_softmax_whole_rows(rows, keys, padded_keys)
}

/// The FP8 epilogue keeps the same row schedule, with two eight-value casts
/// per score panel, half-panel addressing and a bounded masked-tail drain.
fn softmax_output_cycles(rows: u64, keys: u64, padded_keys: u64, fp8: bool, split: bool) -> u64 {
    let base = if split {
        f16_softmax_split_cycles(rows, keys, padded_keys)
    } else {
        f16_softmax_whole_rows(rows, keys, padded_keys)
    };
    if !fp8 {
        return base;
    }
    base.saturating_add(
        rows.div_ceil(if split { 2 } else { 6 })
            .saturating_mul(6)
            .saturating_mul(
                padded_keys
                    .div_ceil(if split { 48 } else { 16 })
                    .saturating_mul(7)
                    .saturating_add(8),
            ),
    )
}

/// Merge preserves FP32 state. Pair loops issue four groups for initialization
/// and six for updates; final normalization is folded into the row coefficients.
pub(crate) fn f16_attention_merge_cycles(
    rows: u64,
    values: u64,
    initial: bool,
    final_block: bool,
    output_f16: bool,
) -> u64 {
    if rows == 0 {
        return 0;
    }
    let panels = values / 16;
    let row = (if initial { 24u64 } else { 31u64 })
        .saturating_add(if output_f16 { 3 } else { 0 })
        .saturating_add(u64::from(panels != 0))
        .saturating_add(panels.saturating_mul(3))
        .saturating_add(
            values
                .div_ceil(2)
                .saturating_mul((if initial { 4 } else { 6 }) + u64::from(output_f16)),
        )
        .saturating_add(if final_block {
            if initial { 3 } else { 5 }
        } else {
            0
        });
    222u64.saturating_add(rows.div_ceil(6).saturating_mul(6).saturating_mul(row))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_estimates_cover_measured_row_schedules() {
        // IPU21, explicit probability/statistics/scratch buffers in separate
        // elements, random inputs (typed-attention/softmax-*-random artifacts).
        // Bank conflicts and worker skew are outside this coarse issue model.
        for (rows, keys, padded, measured) in [
            (1, 64, 64, 1422u64),
            (7, 64, 64, 2640),
            (8, 64, 64, 3168),
            (1, 65, 80, 1770),
            (7, 65, 80, 3330),
            (8, 65, 80, 3774),
            (1, 729, 768, 5190),
            (7, 729, 768, 21456),
            (8, 729, 768, 22536),
        ] {
            let estimated = softmax_output_cycles(
                rows,
                keys,
                padded,
                false,
                f16_softmax_split_rows(rows, keys, padded),
            );
            assert!(
                estimated.abs_diff(measured) <= measured / 5,
                "rows={rows} keys={keys}: estimated={estimated} measured={measured}"
            );
        }
        assert_eq!(softmax_output_cycles(0, 64, 64, false, false), 0);
        assert_eq!(
            softmax_output_cycles(u64::MAX, u64::MAX, u64::MAX, false, false),
            u64::MAX
        );
        assert_eq!(
            f16_attention_merge_cycles(u64::MAX, u64::MAX, false, true, false),
            u64::MAX
        );
    }

    #[test]
    fn softmax_segmentation_prices_launches_and_worker_rounding() {
        assert!(f16_softmax_split_rows(1, 128, 128));
        assert!(!f16_softmax_split_rows(7, 128, 128));
        assert!(!f16_softmax_split_rows(8, 128, 128));
        assert!(!f16_softmax_split_rows(6, 729, 768));
        for rows in [7, 8] {
            assert!(f16_softmax_split_rows(rows, 729, 768));
        }
    }
}
