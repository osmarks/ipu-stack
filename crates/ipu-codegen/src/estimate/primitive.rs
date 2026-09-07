//! Primitive prices over local tensor geometry, shared by compact mid costing
//! and final scheduled tile timelines. No tile IR is needed to evaluate them.

use super::*;
use crate::TileKernelSpec;

pub(crate) fn cast_cycles(from: Precision, to: Precision, elements: u64, panel_rows: u64) -> u64 {
    match (from, to) {
        (Precision::F16, Precision::F8F143 { .. }) => {
            if panel_rows != 0 {
                // Four vector conversions per row, in a 14-bundle repeat body.
                // Match whole-panel versus row sharing, including imbalance and
                // setup between panels, without expanding tile IR.
                let panels = elements.div_ceil(panel_rows.saturating_mul(32));
                if panel_rows <= 32 && panels >= 6 {
                    330 + panels.div_ceil(6).saturating_mul(120 + panel_rows * 84)
                } else {
                    330 + panels.saturating_mul(120 + panel_rows.div_ceil(6) * 84)
                }
            } else {
                330 + elements.div_ceil(48).saturating_mul(24)
            }
        }
        (Precision::F32, Precision::F16) => 330 + elements.div_ceil(12).saturating_mul(48),
        _ => {
            IPU21_TARGET_COSTS.kernel_launch_cycles
                + elements
                    .saturating_mul(from.bytes() + to.bytes())
                    .div_ceil(IPU21_TARGET_COSTS.local_copy_bytes_per_cycle)
        }
    }
}

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
                if output.format.layout.order.gemm_output_transposed() {
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
            // Native FP8 uses the same instruction sequence for 32 K
            // elements that the F16 kernel uses for 16.
            let inner =
                u64::from(*inner_block).div_ceil(if matches!(multiply, Precision::F8F143 { .. }) {
                    2
                } else {
                    1
                });
            let interleaved = inputs.get(1).is_some_and(|input| {
                input.format.layout.memory_class == MemoryClass::Ipu21Interleaved
            });
            if *multiply == Precision::F16
                && output.format.layout.order.gemm_output_group().is_some()
            {
                return crate::kernel::cost::f16_packed_gemm_cycles(
                    rows,
                    inner,
                    columns,
                    interleaved,
                );
            }
            if *multiply != Precision::F32 && interleaved {
                return crate::kernel::cost::interleaved_f16_gemm_cycles(rows, inner, columns);
            }
            let (row_cycles, group_cycles) = match multiply {
                Precision::F16 => (rows, 1063),
                Precision::F32 => (rows.saturating_mul(4), 2126),
                Precision::F8F143 { .. } => (rows, 1063),
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
        TileKernelSpec::Add => elements.saturating_mul(3),
        TileKernelSpec::LayerNorm => elements.saturating_mul(14),
        TileKernelSpec::ReductionSum { partials } => {
            return crate::kernel::cost::f16_reduction_cycles(elements, u64::from(*partials));
        }
        TileKernelSpec::Cast { from, to } => {
            let columns = u64::from(*output.shape.0.last().unwrap_or(&1));
            let panel_rows = output
                .format
                .layout
                .order
                .fp8_cast_panel_rows(elements / columns, columns);
            return cast_cycles(*from, *to, elements, panel_rows);
        }
        TileKernelSpec::Rearrange { from, .. } => {
            if from.order == ElementOrder::Amp(crate::AmpOrder::TransposedLeft)
                && output.format.precision == Precision::F16
                && output.format.layout.order == ElementOrder::RowMajor
                && let [outer @ .., rows, columns] = output.shape.0.as_slice()
            {
                return crate::kernel::cost::f16_transposed_unpack_cycles(
                    outer
                        .iter()
                        .fold(1u64, |count, &n| count.saturating_mul(u64::from(n))),
                    u64::from(*rows),
                    u64::from(*columns),
                );
            }
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
