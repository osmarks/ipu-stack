//! Primitive prices over local tensor geometry, shared by compact mid costing
//! and final scheduled tile timelines. No tile IR is needed to evaluate them.

use super::*;
use crate::{AmpOrder, TileKernelSpec};

pub(crate) fn kernel_cycles(
    kernel: &TileKernelSpec,
    inputs: &[TensorType],
    output: &TensorType,
) -> u64 {
    let target = IPU21_TARGET_COSTS;
    let elements = output
        .shape
        .0
        .iter()
        .fold(1u64, |n, &width| n.saturating_mul(u64::from(width)));
    let rows = output.shape.0[..output.shape.0.len().saturating_sub(1)]
        .iter()
        .fold(1u64, |n, &width| n.saturating_mul(u64::from(width)));
    let work = match kernel {
        TileKernelSpec::Gemm {
            multiply,
            inner_block,
            output_columns,
            ..
        } => {
            let column_axis = output.shape.0.len().saturating_sub(
                if matches!(
                    output.format.layout.order,
                    ElementOrder::Amp(AmpOrder::TransposedOutput | AmpOrder::TransposedLeft)
                ) {
                    2
                } else {
                    1
                },
            );
            let rows = output
                .shape
                .0
                .iter()
                .enumerate()
                .filter(|(axis, _)| *axis != column_axis)
                .fold(1u64, |n, (_, &width)| n.saturating_mul(u64::from(width)));
            let columns = u64::from(*output_columns);
            let inner = u64::from(*inner_block);
            let interleaved = inputs.get(1).is_some_and(|input| {
                input.format.layout.memory_class == MemoryClass::Ipu21Interleaved
            });
            if *multiply == Precision::F16 && interleaved {
                return crate::kernel::cost::interleaved_f16_gemm_cycles(rows, inner, columns);
            }
            let (row_cycles, group_cycles) = match multiply {
                Precision::F16 => (rows, 1063),
                Precision::F32 => (rows.saturating_mul(4), 2126),
                Precision::F8F143 { .. } => {
                    return rows
                        .saturating_mul(columns)
                        .saturating_mul(inner)
                        .saturating_mul(2)
                        .div_ceil(256)
                        .saturating_add(target.kernel_launch_cycles);
                }
            };
            return 294u64.saturating_add(
                inner.div_ceil(16).saturating_mul(
                    columns
                        .saturating_mul(row_cycles)
                        .div_ceil(4)
                        .saturating_add(
                            columns
                                .div_ceil(16)
                                .saturating_mul(group_cycles)
                                .div_ceil(4),
                        ),
                ),
            );
        }
        TileKernelSpec::FillZero { bytes, .. } => u64::from(*bytes).div_ceil(48),
        TileKernelSpec::Gelu if output.format.precision == Precision::F16 => {
            return crate::kernel::cost::f16_gelu_cycles(elements);
        }
        TileKernelSpec::AttentionSoftmax {
            key_columns,
            padded_key_columns,
            ..
        } => {
            return crate::kernel::cost::f16_softmax_cycles(
                rows,
                u64::from(*key_columns),
                u64::from(*padded_key_columns),
            );
        }
        TileKernelSpec::Gelu => elements.saturating_mul(10),
        TileKernelSpec::Add => elements.div_ceil(16),
        TileKernelSpec::ReductionSum { partials } => {
            return crate::kernel::cost::f16_reduction_cycles(elements, u64::from(*partials));
        }
        TileKernelSpec::Cast { .. } => elements.div_ceil(8),
        TileKernelSpec::Rearrange { .. } => {
            return if output.format.layout.order == ElementOrder::RowMajor {
                elements
                    .saturating_mul(10)
                    .saturating_add(target.kernel_launch_cycles)
            } else {
                row_major_pack_cycles(output, elements)
            };
        }
        TileKernelSpec::AttentionMerge {
            value_dimension,
            initial,
            final_block,
            ..
        } => {
            return crate::kernel::cost::f16_attention_merge_cycles(
                rows,
                u64::from(*value_dimension),
                *initial,
                *final_block,
            );
        }
        TileKernelSpec::FlashAttention { .. } => {
            let [query, key, value] = inputs else {
                return u64::MAX;
            };
            let rank = query.shape.0.len();
            if rank < 2 || key.shape.0.len() != rank || value.shape.0.len() != rank {
                return u64::MAX;
            }
            return query.shape.0[..rank - 1]
                .iter()
                .fold(1u64, |n, &width| n.saturating_mul(u64::from(width)))
                .saturating_mul(u64::from(key.shape.0[rank - 2]))
                .saturating_mul(
                    u64::from(query.shape.0[rank - 1]) + u64::from(value.shape.0[rank - 1]),
                )
                .saturating_mul(4)
                .div_ceil(6)
                .saturating_add(target.kernel_launch_cycles);
        }
    };
    work.saturating_add(target.kernel_launch_cycles)
}
