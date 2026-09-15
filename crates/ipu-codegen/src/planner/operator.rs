//! Planner family choices, dispatch parameters and operand/result requirements.

use crate::graph::{AttentionOptions, GemmOptions};
use crate::kernel::AccumulationPrecision;
use crate::mid::MidOperationKind;
use crate::mid::ReductionStaging;
use crate::planner::catalogue::blocked_gemm_dispatch;
use crate::tensor::{
    AMP_COLUMN_MICRO, AMP_INNER_BLOCK, AMP_OUTPUT_COLUMN_BLOCK, AmpOrder, AxisFactorView,
    BlockMajorOrder, ElementOrder, GridOrder, Layout, Precision, TensorAxis, TensorFormat,
    TensorShape, TensorType,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperatorFamily {
    Gemm {
        options: GemmOptions,
        multiply: Precision,
        accumulate: AccumulationPrecision,
    },
    Gelu,
    LayerNorm,
    Add,
    View(AxisFactorView),
    Slice(crate::graph::AxisSlice),
    FlashAttention {
        options: AttentionOptions,
        accumulate: AccumulationPrecision,
    },
}

/// Physical matrix orientation used by a blocked GEMM implementation.
#[derive(
    Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
pub enum GemmOrientation {
    #[default]
    Normal,
    /// Compute `(rightᵀ × leftᵀ)ᵀ`. This preserves GEMM semantics while
    /// exchanging the physical row and output-column traversal dimensions.
    Swapped,
}

impl GemmOrientation {
    /// Semantic input indices consumed as physical left and right operands.
    pub(crate) fn operand_indices(self) -> (usize, usize) {
        match self {
            Self::Normal => (0, 1),
            Self::Swapped => (1, 0),
        }
    }

    /// Physical row and column axes in a tensor of at least rank two.
    pub(crate) fn matrix_axes(self, rank: usize) -> (usize, usize) {
        match self {
            Self::Normal => (rank - 2, rank - 1),
            Self::Swapped => (rank - 1, rank - 2),
        }
    }
}

/// Selected family algorithm and blocking. Family construction consumes this
/// choice and boundary tensor types to build executable mid operations; it is
/// never retained as an operation to be expanded by low.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OperatorDispatch {
    LayerNorm {
        parts: u16,
    },
    /// Run the family's local kernel independently on its selected shards.
    LocalKernel,
    BlockedGemm {
        inner_block: u32,
        output_column_block: u32,
        orientation: GemmOrientation,
        distribution: GemmDistribution,
    },
    Attention {
        fp8_scales: [Option<i8>; 2],
        query_key: Option<ProductGrid>,
        probability_value: Option<ProductGrid>,
        /// Full-key materialization or online softmax over successive blocks.
        materialized: bool,
        key_block_rows: u32,
        padded_query_dimension: u32,
        padded_value_dimension: u32,
    },
    /// Materialize a logical view with copies, exchanges, and any required
    /// storage-order transforms. The operator owns the coordinate mapping.
    View,
}

/// Which operand remains resident while a blocked whole-device GEMM is run.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GemmDistribution {
    #[default]
    OutputStationary,
    /// Distribute independent row, output-column, and K block ranges. Each
    /// row/K activation shard is replicated over the column groups, computes
    /// one local partial, and is reduced over K onto the output owner.
    ParallelReduction {
        row_partitions: u16,
        column_partitions: u16,
        inner_partitions: u16,
        /// Additional spatial partitions of each computed output block. Their
        /// product cannot exceed the K partition count; reduction roots are
        /// spread over former K-partition tiles rather than concentrated on
        /// one root per compute row/column block.
        result_row_partitions: u16,
        result_column_partitions: u16,
        reduction_staging: ReductionStaging,
    },
}

/// Per-batch product partition counts. Physical tile identities and local calls
/// remain the responsibility of low expansion.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProductGrid {
    pub rows: u16,
    pub columns: u16,
    pub inner: u16,
}

impl OperatorDispatch {
    /// Storage of one K group's partial result, before cross-group reduction.
    /// Estimation and lowering share the same grain and per-shard padding.
    pub(crate) fn gemm_partial_tensor(&self, output: &TensorType) -> TensorType {
        let Self::BlockedGemm {
            output_column_block,
            orientation,
            distribution:
                GemmDistribution::ParallelReduction {
                    row_partitions,
                    column_partitions,
                    ..
                },
            ..
        } = self
        else {
            return output.clone();
        };
        let make_layout = match (orientation, output.format.layout.order) {
            (GemmOrientation::Normal, ElementOrder::Amp(AmpOrder::Left)) => {
                Layout::amp_left_result_grid
            }
            (GemmOrientation::Swapped, ElementOrder::Amp(AmpOrder::TransposedLeft)) => {
                Layout::amp_transposed_left_result_grid
            }
            (GemmOrientation::Normal, _) => Layout::amp_output_grid,
            (GemmOrientation::Swapped, _) => Layout::amp_transposed_output_grid,
        };
        let mut partial = output.clone();
        partial.format.layout = make_layout(
            *output_column_block,
            row_partitions.saturating_mul(*column_partitions),
            *row_partitions,
            *column_partitions,
            GridOrder::ColumnsFast,
        );
        if output.format.layout.order.gemm_output_group().is_some() {
            partial.format.layout.order = output.format.layout.order;
        }
        let column_axis = orientation.matrix_axes(output.shape.0.len()).1;
        let source_axis = output
            .format
            .layout
            .tiling
            .axes
            .iter()
            .find(|axis| axis.axis.resolve(output.shape.0.len()) == Ok(column_axis));
        let partial_axis = partial
            .format
            .layout
            .tiling
            .axes
            .iter_mut()
            .find(|axis| axis.axis.resolve(output.shape.0.len()) == Ok(column_axis));
        if let Some((source, target)) = source_axis.zip(partial_axis) {
            target.block_size = source.block_size;
            target.padding_multiple = source.block_size;
            if source.partitions == *column_partitions {
                target.padding_multiple = source.padding_multiple;
                target.shard_padding_multiple = source.shard_padding_multiple;
            }
        }
        partial
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct OperandRequirement {
    pub format: TensorFormat,
    /// How a locally resident operand should be consumed when other tiles use
    /// an operator-local staging buffer for the same operand.
    pub local_staging: LocalOperandStaging,
    /// Whether a dispatch may populate and consume bounded operand slices
    /// instead of materializing the complete required format first.
    pub materialization: OperandMaterialization,
}

#[derive(
    Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
pub enum LocalOperandStaging {
    #[default]
    Direct,
    MatchRemote,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum OperandMaterialization {
    #[default]
    Complete,
    DispatchSlices,
}

impl OperandRequirement {
    pub fn new(format: TensorFormat) -> Self {
        Self {
            format,
            local_staging: LocalOperandStaging::Direct,
            materialization: OperandMaterialization::Complete,
        }
    }

    pub fn with_local_staging(mut self, staging: LocalOperandStaging) -> Self {
        self.local_staging = staging;
        self
    }

    pub fn with_materialization(mut self, materialization: OperandMaterialization) -> Self {
        self.materialization = materialization;
        self
    }
}

pub(super) fn default_dispatch(operator: OperatorFamily) -> OperatorDispatch {
    match operator {
        OperatorFamily::Gemm { .. } => blocked_gemm_dispatch(AMP_OUTPUT_COLUMN_BLOCK),
        OperatorFamily::View(_) | OperatorFamily::Slice(_) => OperatorDispatch::View,
        _ => OperatorDispatch::LocalKernel,
    }
}

impl OperatorFamily {
    pub(super) fn local_kernel(self) -> Option<MidOperationKind> {
        Some(match self {
            Self::LayerNorm => MidOperationKind::LayerNorm,
            Self::Gelu => MidOperationKind::Gelu,
            Self::Add => MidOperationKind::Add,
            Self::FlashAttention {
                options,
                accumulate,
            } => MidOperationKind::FlashAttention {
                options,
                accumulate,
            },
            Self::Gemm { .. } | Self::View(_) | Self::Slice(_) => return None,
        })
    }
}

/// Selected algorithm, input materialization, and result format. Local access
/// requirements belong to the kernel calls emitted from this plan.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct OperatorPlan {
    pub operator: OperatorFamily,
    pub dispatch: OperatorDispatch,
    pub inputs: Vec<OperandRequirement>,
    pub output: TensorFormat,
    /// None allocates a fresh result. Some requires one compatible, writable
    /// input from this preference list; an empty list makes the plan infeasible.
    pub reuse_inputs: Option<Vec<u16>>,
}

impl OperatorPlan {
    pub(super) fn can_reuse_input(
        &self,
        index: u16,
        inputs: &[TensorType],
        output: &TensorShape,
    ) -> bool {
        self.inputs
            .get(usize::from(index))
            .zip(inputs.get(usize::from(index)))
            .is_some_and(|(requirement, input)| {
                input.shape == *output && requirement.format == self.output
            })
    }

    /// Apply this plan's formats to the logical operand shapes.
    pub(super) fn tensor_types(
        &self,
        inputs: &[TensorType],
        output: &TensorShape,
    ) -> (Vec<TensorType>, TensorType) {
        let inputs = inputs
            .iter()
            .zip(&self.inputs)
            .map(|(input, requirement)| TensorType {
                shape: input.shape.clone(),
                format: requirement.format.clone(),
            })
            .collect();
        let output = TensorType {
            shape: output.clone(),
            format: self.output.clone(),
        };
        (inputs, output)
    }

    pub(super) fn supports(&self, inputs: &[TensorType], output: &TensorShape) -> bool {
        if self.inputs.len() != inputs.len()
            || self.reuse_inputs.as_ref().is_some_and(|indices| {
                !indices
                    .iter()
                    .any(|&index| self.can_reuse_input(index, inputs, output))
            })
        {
            return false;
        }
        let Ok(layouts) = self
            .inputs
            .iter()
            .zip(inputs)
            .map(|(requirement, input)| requirement.format.layout.resolve(&input.shape))
            .collect::<Result<Vec<_>, _>>()
        else {
            return false;
        };
        let Ok(output_layout) = self.output.layout.resolve(output) else {
            return false;
        };
        if !matches!(self.dispatch, OperatorDispatch::LocalKernel)
            && output_layout.has_empty_shards()
        {
            return false;
        }
        match (&self.operator, &self.dispatch) {
            (
                OperatorFamily::Gemm {
                    options, multiply, ..
                },
                OperatorDispatch::BlockedGemm {
                    inner_block,
                    output_column_block,
                    distribution,
                    orientation,
                },
            ) => {
                if inputs.len() != 2
                    || inputs.iter().any(|input| input.shape.0.len() < 2)
                    || output.0.len() < 2
                    || *inner_block == 0
                    || *output_column_block == 0
                    || options.transpose_left
                    || options.transpose_right
                {
                    return false;
                }
                let parallel = matches!(distribution, GemmDistribution::ParallelReduction { .. });
                if !parallel
                    && self.inputs[0].format.layout.tiling.tile_count
                        != self.output.layout.tiling.tile_count
                {
                    return false;
                }
                let (left, right) = orientation.operand_indices();
                let swapped = *orientation == GemmOrientation::Swapped;
                let left_order = if swapped {
                    AmpOrder::TransposedLeft
                } else {
                    AmpOrder::Left
                };
                let native_output = match (swapped, *multiply == Precision::F32) {
                    (false, false) => AmpOrder::Left,
                    (false, true) => AmpOrder::Output,
                    (true, false) => AmpOrder::TransposedLeft,
                    (true, true) => AmpOrder::TransposedOutput,
                };
                let right_matches = match self.inputs[right].format.layout.order {
                    ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. }) => !swapped,
                    ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix { .. }) => swapped,
                    _ => false,
                };
                if self.inputs[left].format.layout.order != ElementOrder::Amp(left_order)
                    || !right_matches
                    || !(self.output.layout.order == ElementOrder::Amp(native_output)
                        || (*multiply == Precision::F16
                            && self.output.layout.order.gemm_output_group().is_some()
                            && self.output.layout.order.gemm_output_transposed() == swapped))
                {
                    return false;
                }
                if let GemmDistribution::ParallelReduction {
                    row_partitions,
                    column_partitions,
                    inner_partitions,
                    result_row_partitions,
                    result_column_partitions,
                    ..
                } = distribution
                {
                    let rows = row_partitions.saturating_mul(*result_row_partitions);
                    let columns = column_partitions.saturating_mul(*result_column_partitions);
                    let row_axis = TensorAxis::FromEnd(if swapped { 1 } else { 2 });
                    let column_axis = TensorAxis::FromEnd(if swapped { 2 } else { 1 });
                    let parts = |axis| {
                        self.output
                            .layout
                            .tiling
                            .axes
                            .iter()
                            .find(|tiling| tiling.axis == axis)
                            .map(|tiling| tiling.partitions)
                    };
                    if *row_partitions == 0
                        || *column_partitions == 0
                        || *inner_partitions < 2
                        || *result_row_partitions == 0
                        || *result_column_partitions == 0
                        || result_row_partitions.saturating_mul(*result_column_partitions)
                            > *inner_partitions
                        || self.output.layout.tiling.tile_count != rows.saturating_mul(columns)
                        || parts(row_axis) != Some(rows)
                        || parts(column_axis) != Some(columns)
                    {
                        return false;
                    }
                }
                let left_layout = &layouts[left];
                let right_layout = &layouts[right];
                let inner_axis = orientation.matrix_axes(left_layout.padded_shape.0.len()).1;
                let right_column_axis =
                    orientation.matrix_axes(right_layout.padded_shape.0.len()).1;
                let output_column_axis = orientation.matrix_axes(output.0.len()).1;
                let (Some(output_axes), Some(right_axes)) =
                    (output_layout.axes(), right_layout.axes())
                else {
                    return false;
                };
                let output_columns = output_axes[output_column_axis].maximum_extent();
                let right_columns = if parallel {
                    right_axes[right_column_axis].maximum_extent()
                } else {
                    right_axes[right_column_axis].minimum_extent()
                };
                let grid = self
                    .inputs
                    .iter()
                    .any(|input| input.format.layout.tiling.replicas > 1)
                    || self.inputs[1]
                        .format
                        .layout
                        .tiling
                        .axes
                        .iter()
                        .any(|axis| axis.axis == TensorAxis::FromEnd(2) && axis.partitions > 1);
                let alignment = if parallel {
                    AMP_COLUMN_MICRO
                } else {
                    *output_column_block
                };
                !(grid && layouts.iter().any(|layout| layout.has_empty_shards()))
                    && left_layout.padded_shape.0[inner_axis].is_multiple_of(*inner_block)
                    && output_layout.padded_shape.0[output_column_axis].is_multiple_of(alignment)
                    && output_columns.is_multiple_of(alignment)
                    && (!parallel || output_columns <= *output_column_block)
                    && right_columns >= *output_column_block
            }
            (
                OperatorFamily::LayerNorm
                | OperatorFamily::Gelu
                | OperatorFamily::Add
                | OperatorFamily::FlashAttention { .. },
                OperatorDispatch::LocalKernel,
            )
            | (OperatorFamily::LayerNorm, OperatorDispatch::LayerNorm { .. }) => {
                self.inputs.iter().all(|input| {
                    input.format.layout.tiling.tile_count == self.output.layout.tiling.tile_count
                })
            }
            (
                OperatorFamily::FlashAttention {
                    options,
                    accumulate,
                },
                OperatorDispatch::Attention {
                    materialized,
                    key_block_rows,
                    padded_query_dimension,
                    padded_value_dimension,
                    ..
                },
            ) => {
                let [query, key, value] = self.inputs.as_slice() else {
                    return false;
                };
                !options.causal
                    && *accumulate == AccumulationPrecision::F32
                    && *key_block_rows != 0
                    && key_block_rows.is_multiple_of(AMP_INNER_BLOCK)
                    && (*materialized || *key_block_rows == AMP_INNER_BLOCK)
                    && *padded_query_dimension != 0
                    && *padded_value_dimension != 0
                    && query.format.layout.order == ElementOrder::Amp(AmpOrder::Left)
                    && key.format.layout.order == ElementOrder::Amp(AmpOrder::TransposedRight)
                    && matches!(
                        value.format.layout.order,
                        ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. })
                    )
                    && self.output.layout.order == ElementOrder::RowMajor
                    && query.format.layout.tiling.tile_count == self.output.layout.tiling.tile_count
                    && key.format.layout.tiling.tile_count == value.format.layout.tiling.tile_count
            }
            (operator, OperatorDispatch::View) => {
                let [input] = inputs else { return false };
                let shape = match operator {
                    OperatorFamily::View(view) => view.output_shape(&input.shape),
                    OperatorFamily::Slice(slice) => slice.output_shape(&input.shape),
                    _ => return false,
                };
                shape.as_ref() == Some(output)
                    && self.inputs[0].format.precision == self.output.precision
            }
            _ => false,
        }
    }
}
