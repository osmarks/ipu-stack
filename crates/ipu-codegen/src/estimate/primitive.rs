//! Primitive prices over local tensor geometry, shared by compact mid costing
//! and final scheduled tile timelines. No tile IR is needed to evaluate them.

use super::*;
use crate::{TensorFormat, TileKernelSpec};

pub(crate) fn cast_cycles(from: Precision, to: Precision, elements: u64, panel_rows: u64) -> u64 {
    match (from, to) {
        (Precision::F16, Precision::F8F143 { .. }) => {
            if panel_rows != 0 {
                // Price the 12-bundle fallback; the eight-bundle pipeline
                // additionally requires bank separation, unknown before placement.
                // Match whole-panel versus row sharing, including imbalance and
                // setup between panels, without expanding tile IR.
                let panels = elements.div_ceil(panel_rows.saturating_mul(32));
                if panel_rows <= 32 && panels >= 6 {
                    330 + panels.div_ceil(6).saturating_mul(120 + panel_rows * 72)
                } else {
                    330 + panels.saturating_mul(120 + panel_rows.div_ceil(6) * 72)
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

/// Borrow either planned dimensions or the physical extents of an emitted call.
#[derive(Clone, Copy)]
pub(crate) enum Geometry<'a> {
    Tensor(&'a TensorType),
    Storage(crate::storage::TensorStorage<'a>),
}
impl<'a> Geometry<'a> {
    pub(super) fn format(self) -> &'a TensorFormat {
        match self {
            Self::Tensor(t) => &t.format,
            Self::Storage(s) => s.format,
        }
    }
    pub(super) fn rank(self) -> usize {
        match self {
            Self::Tensor(t) => t.shape.0.len(),
            Self::Storage(s) => s.extents.len(),
        }
    }
    fn dimension(self, axis: usize) -> u32 {
        match self {
            Self::Tensor(t) => t.shape.0[axis],
            Self::Storage(s) => s.extents[axis].physical_end - s.extents[axis].start,
        }
    }
    pub(super) fn trailing_dimension(self, offset: usize) -> Option<u32> {
        self.rank()
            .checked_sub(offset + 1)
            .map(|axis| self.dimension(axis))
    }
    fn widths(self) -> impl DoubleEndedIterator<Item = u32> + ExactSizeIterator {
        (0..self.rank()).map(move |axis| self.dimension(axis))
    }
    fn elements(self) -> u64 {
        self.widths()
            .fold(1u64, |n, width| n.saturating_mul(u64::from(width)))
    }
}

pub(crate) fn kernel_cycles<'a>(
    kernel: &TileKernelSpec,
    inputs: impl Fn(usize) -> Option<Geometry<'a>>,
    output: Geometry<'a>,
) -> u64 {
    let target = IPU21_TARGET_COSTS;
    let elements = output.elements();
    let rows = output
        .widths()
        .take(output.rank().saturating_sub(1))
        .fold(1u64, |n, width| n.saturating_mul(u64::from(width)));
    if matches!(kernel, TileKernelSpec::Gelu | TileKernelSpec::LayerNorm)
        && matches!(output.format().precision, Precision::F8F143 { .. })
    {
        let Some(input) = inputs(0) else {
            return u64::MAX;
        };
        let width = u64::from(input.trailing_dimension(0).unwrap_or(0));
        return crate::kernel::cost::fp8_elementwise_cycles(
            matches!(kernel, TileKernelSpec::Gelu),
            input.elements().checked_div(width).unwrap_or(0),
            width,
            output.format().layout.order == ElementOrder::Amp(crate::AmpOrder::Left),
        );
    }
    let work = match kernel {
        TileKernelSpec::Gemm {
            multiply,
            inner_block,
            output_columns,
            ..
        } => {
            let column_axis = output.rank().saturating_sub(
                if output.format().layout.order.gemm_output_transposed() {
                    2
                } else {
                    1
                },
            );
            let rows = output
                .widths()
                .enumerate()
                .filter(|(axis, _)| *axis != column_axis)
                .fold(1u64, |n, (_, width)| n.saturating_mul(u64::from(width)));
            let columns = u64::from(*output_columns);
            // Native FP8 uses the same instruction sequence for 32 K
            // elements that the F16 kernel uses for 16.
            let inner =
                u64::from(*inner_block).div_ceil(if matches!(multiply, Precision::F8F143 { .. }) {
                    2
                } else {
                    1
                });
            let interleaved = inputs(1).is_some_and(|input| {
                input.format().layout.memory_class == MemoryClass::Ipu21Interleaved
            });
            if *multiply == Precision::F16
                && output.format().layout.order.gemm_output_group().is_some()
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
        TileKernelSpec::Gelu if output.format().precision == Precision::F16 => {
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
        TileKernelSpec::Add if output.format().precision == Precision::F16 => {
            return crate::kernel::cost::f16_add_cycles(
                elements,
                inputs(0).map_or(elements, Geometry::elements),
                inputs(1).map_or(elements, Geometry::elements),
            );
        }
        TileKernelSpec::Add => elements.saturating_mul(3),
        TileKernelSpec::BiasGelu => {
            return rows
                .saturating_mul(
                    crate::kernel::cost::f16_gelu_cycles(elements.checked_div(rows).unwrap_or(0))
                        .saturating_sub(330)
                        + 40,
                )
                .saturating_add(elements.saturating_mul(2))
                .saturating_add(330);
        }
        TileKernelSpec::AddLayerNorm | TileKernelSpec::LayerNorm => {
            return crate::kernel::cost::f16_layernorm_cycles(
                rows,
                u64::from(output.trailing_dimension(0).unwrap_or(0)),
                matches!(kernel, TileKernelSpec::AddLayerNorm),
            );
        }
        TileKernelSpec::LayerNormMoments | TileKernelSpec::AddLayerNormMoments => {
            let Some(input) = inputs(0) else {
                return u64::MAX;
            };
            let width = u64::from(input.trailing_dimension(0).unwrap_or(0));
            let rows = input.elements().checked_div(width).unwrap_or(0);
            let extra = if matches!(kernel, TileKernelSpec::AddLayerNormMoments) {
                rows * (108 + width.div_ceil(48) * 24)
            } else {
                0
            };
            return crate::kernel::cost::f16_layernorm_moments_cycles(rows, width) + extra;
        }
        TileKernelSpec::LayerNormApply { parts } => {
            return crate::kernel::cost::f16_layernorm_apply_cycles(
                rows,
                u64::from(output.trailing_dimension(0).unwrap_or(0)),
                *parts,
            );
        }
        TileKernelSpec::ReductionSum { partials } => {
            return crate::kernel::cost::f16_reduction_cycles(elements, u64::from(*partials));
        }
        TileKernelSpec::Cast { from, to } => {
            let columns = u64::from(output.trailing_dimension(0).unwrap_or(1));
            let panel_rows = output
                .format()
                .layout
                .order
                .fp8_cast_panel_rows(elements / columns, columns);
            let linear = output.format().layout.order == ElementOrder::Amp(crate::AmpOrder::Left)
                && panel_rows == 1
                && inputs(0).is_some_and(|input| input.elements() == elements);
            return cast_cycles(*from, *to, elements, if linear { 0 } else { panel_rows });
        }
        TileKernelSpec::Rearrange { from, .. } => {
            if from.order == ElementOrder::Amp(crate::AmpOrder::TransposedLeft)
                && output.format().precision == Precision::F16
                && output.format().layout.order == ElementOrder::RowMajor
                && let (Some(rows), Some(columns)) =
                    (output.trailing_dimension(1), output.trailing_dimension(0))
            {
                return crate::kernel::cost::f16_transposed_unpack_cycles(
                    output
                        .widths()
                        .take(output.rank() - 2)
                        .fold(1u64, |count, n| count.saturating_mul(u64::from(n))),
                    u64::from(rows),
                    u64::from(columns),
                );
            }
            return if output.format().layout.order == ElementOrder::RowMajor {
                elements
                    .saturating_mul(10)
                    .saturating_add(target.kernel_launch_cycles)
            } else {
                super::cycles::pack_geometry_cycles(output, elements)
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
                output.format().precision == Precision::F16,
            );
        }
        TileKernelSpec::FlashAttention { .. } => {
            let (Some(query), Some(key), Some(value), None) =
                (inputs(0), inputs(1), inputs(2), inputs(3))
            else {
                return u64::MAX;
            };
            let rank = query.rank();
            if rank < 2 || key.rank() != rank || value.rank() != rank {
                return u64::MAX;
            }
            return query
                .widths()
                .take(rank - 1)
                .fold(1u64, |n, width| n.saturating_mul(u64::from(width)))
                .saturating_mul(u64::from(key.dimension(rank - 2)))
                .saturating_mul(
                    u64::from(query.dimension(rank - 1)) + u64::from(value.dimension(rank - 1)),
                )
                .saturating_mul(4)
                .div_ceil(6)
                .saturating_add(target.kernel_launch_cycles);
        }
    };
    work.saturating_add(target.kernel_launch_cycles)
}
