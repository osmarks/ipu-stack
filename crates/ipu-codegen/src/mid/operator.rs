//! Executable operator and conversion contracts, plus their validation.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccumulationPrecision {
    F16,
    F32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MidOperator {
    Gemm {
        options: GemmOptions,
        multiply: Precision,
        accumulate: AccumulationPrecision,
    },
    Gelu,
    Add,
    View(AxisFactorView),
    FlashAttention {
        options: AttentionOptions,
        accumulate: AccumulationPrecision,
    },
}

/// A tile-local callable selected by a whole-device operator plan.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TileKernelSpec {
    /// Physical byte range initialized during tile expansion.
    FillZero {
        offset: u32,
        bytes: u32,
        /// Copy mappings cover every logical element; only padding needs initialization.
        padding_only: bool,
    },
    Gemm {
        multiply: Precision,
        accumulate: AccumulationPrecision,
        mode: GemmKernelMode,
        weights: GemmWeightLoad,
        inner_block: u32,
        output_columns: u32,
    },
    Gelu,
    ReductionSum {
        partials: u16,
    },
    Add,
    FlashAttention {
        options: AttentionOptions,
        accumulate: AccumulationPrecision,
    },
    AttentionSoftmax {
        head_dimension: u32,
        key_columns: u32,
        padded_key_columns: u32,
    },
    AttentionMerge {
        value_dimension: u32,
        padded_value_dimension: u32,
        key_block_columns: u32,
        initial: bool,
        final_block: bool,
    },
    Cast {
        from: Precision,
        to: Precision,
    },
    Rearrange {
        from: Layout,
        to: Layout,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GemmKernelMode {
    Initialize,
    Accumulate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GemmWeightLoad {
    Standard,
    Interleaved,
}

/// Physical matrix orientation used by a blocked GEMM implementation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

/// Shape-independent recipe which expands into ordered device-wide exchange
/// and tile-kernel phases after concrete shards are known.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OperatorDispatch {
    Pointwise {
        kernel: TileKernelSpec,
    },
    BlockedGemm {
        inner_block: u32,
        output_column_block: u32,
        orientation: GemmOrientation,
        distribution: GemmDistribution,
    },
    Attention {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeferredOutputPlan {
    pub source_input: usize,
    pub transform: AxisFactorView,
    /// Cost restored if no later consumer claims this offer.
    pub unfused_cycles: u64,
    /// Exchange portion of `unfused_cycles`, restored with the offer.
    pub unfused_exchange_cycles: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeferredInputPlan {
    pub producer: MidValueId,
    pub source: MidValueId,
    pub transform: AxisFactorView,
}

/// Which operand remains resident while a blocked whole-device GEMM is run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
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

/// Lifetime policy for partials reduced across a GEMM's K partitions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReductionStaging {
    /// Receive every remote partial into one packed buffer, then reduce once.
    #[default]
    Complete,
    /// Receive and accumulate one remote partial at a time. This minimizes
    /// temporary SRAM at the expense of additional exchange epochs and kernel
    /// launches.
    Streamed,
    /// Receive at most this many remote partials per exchange epoch.
    Batched(std::num::NonZeroU16),
}

impl ReductionStaging {
    pub(crate) fn remote_partials_per_stage(self, remote: u64) -> u64 {
        match self {
            Self::Complete => remote.max(1),
            Self::Streamed => 1,
            Self::Batched(limit) => u64::from(limit.get()).min(remote.max(1)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum EmptyOutputShardPolicy {
    Skip,
    Reject,
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

    pub(super) fn empty_output_shard_policy(&self) -> EmptyOutputShardPolicy {
        match self {
            Self::Pointwise { .. } => EmptyOutputShardPolicy::Skip,
            Self::View => EmptyOutputShardPolicy::Reject,
            Self::BlockedGemm { .. } | Self::Attention { .. } => EmptyOutputShardPolicy::Reject,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OperandRequirement {
    pub format: TensorFormat,
    pub alignment: u32,
    /// Bytes the kernel may access beyond the logical tensor payload.
    pub access_tail_bytes: u32,
    /// How a locally resident operand should be consumed when other tiles use
    /// an operator-local staging buffer for the same operand.
    pub local_staging: LocalOperandStaging,
    /// Whether a dispatch may populate and consume bounded operand slices
    /// instead of materializing the complete required format first.
    pub materialization: OperandMaterialization,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LocalOperandStaging {
    #[default]
    Direct,
    MatchRemote,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum OperandMaterialization {
    #[default]
    Complete,
    DispatchSlices,
}

impl OperandRequirement {
    pub fn new(format: TensorFormat, alignment: u32) -> Self {
        Self {
            format,
            alignment,
            access_tail_bytes: 0,
            local_staging: LocalOperandStaging::Direct,
            materialization: OperandMaterialization::Complete,
        }
    }

    pub fn with_access_tail(mut self, bytes: u32) -> Self {
        self.access_tail_bytes = bytes;
        self
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OutputAliasing {
    Fresh,
    MayAliasInputs(Vec<u16>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryOperand {
    Output,
    Input(u16),
}

pub(super) fn layout_has_empty_shards(layout: &Layout, shape: &TensorShape) -> bool {
    layout
        .resolve(shape)
        .map_or(true, |resolved| resolved.has_empty_shards())
}

pub(super) fn default_dispatch(operator: MidOperator) -> OperatorDispatch {
    match operator {
        MidOperator::Gemm { .. } => blocked_gemm_dispatch(AMP_OUTPUT_COLUMN_BLOCK),
        MidOperator::Gelu => OperatorDispatch::Pointwise {
            kernel: TileKernelSpec::Gelu,
        },
        MidOperator::Add => OperatorDispatch::Pointwise {
            kernel: TileKernelSpec::Add,
        },
        MidOperator::View(_) => OperatorDispatch::View,
        MidOperator::FlashAttention {
            options,
            accumulate,
        } => OperatorDispatch::Pointwise {
            kernel: TileKernelSpec::FlashAttention {
                options,
                accumulate,
            },
        },
    }
}

pub(super) fn alias_compatible(
    index: usize,
    requirements: &[OperandRequirement],
    inputs: &[TensorType],
    output_requirement: &OperandRequirement,
    output_shape: &TensorShape,
) -> bool {
    requirements
        .get(index)
        .zip(inputs.get(index))
        .is_some_and(|(requirement, input)| {
            input.shape == *output_shape && requirement.format == output_requirement.format
        })
}

pub(super) fn valid_requirement(requirement: &OperandRequirement, shape: &TensorShape) -> bool {
    requirement.alignment.is_power_of_two() && requirement.format.layout.resolve(shape).is_ok()
}

pub(super) fn valid_memory_operand(operand: MemoryOperand, input_count: usize) -> bool {
    match operand {
        MemoryOperand::Output => true,
        MemoryOperand::Input(index) => usize::from(index) < input_count,
    }
}

/// Operand storage and access constraints, shared by operator plans and
/// concrete kernel calls. A call binds formats to its actual operand buffers.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StorageRequirements {
    pub inputs: Vec<OperandRequirement>,
    pub output: OperandRequirement,
    pub output_aliasing: OutputAliasing,
    pub distinct_elements: Vec<Vec<MemoryOperand>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OperatorPlan {
    pub operator: MidOperator,
    pub dispatch: OperatorDispatch,
    pub requirements: StorageRequirements,
    /// A view transformation offered by this plan. It is materialized normally
    /// unless a later plan records a matching entry in `deferred_inputs`.
    pub deferred_output: Option<DeferredOutputPlan>,
}

/// Address-independent recipe for materializing a format conversion.
///
/// Layouts determine the logical shard regions and relative physical spans;
/// final tile identities and SRAM addresses remain a low-level concern.  The
/// same recipe is consumed by the cost model and by tile-program lowering so
/// planning cannot silently price a different conversion from the one emitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConversionStrategy {
    /// Run one tile-local kernel over corresponding resident shards.
    LocalKernel,
    /// Exchange logical intersections directly into the destination layout.
    DirectRetile,
    /// Exchange logical values into row-major staging, then transform locally
    /// into the destination element order.
    StageLogicalThenTransform,
}

pub fn layout_conversion_strategy(from: &Layout, to: &Layout) -> ConversionStrategy {
    if from.order == to.order {
        ConversionStrategy::DirectRetile
    } else {
        ConversionStrategy::StageLogicalThenTransform
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConversionPlan {
    pub input: OperandRequirement,
    pub output: OperandRequirement,
    pub strategy: ConversionStrategy,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq, Hash)]
pub enum OperatorPlanError {
    #[error("operator plan operand arity does not match its requirements")]
    OperandArity,
    #[error("operator plan dispatch does not match the selected operator")]
    DispatchMismatch,
    #[error("operator plan uses zero or incompatible block dimensions")]
    InvalidBlocking,
    #[error("operator plan requires corresponding activation and output tile groups")]
    IncompatibleTileGroups,
    #[error("operator dispatch does not support empty output shards")]
    EmptyOutputShard,
    #[error("blocked GEMM currently requires non-transposed AMP left/right/output formats")]
    UnsupportedGemmLayout,
}

impl OperatorPlan {
    /// Apply this plan's formats to the logical operand shapes.
    pub(super) fn tensor_types(
        &self,
        inputs: &[TensorType],
        output: &TensorShape,
    ) -> (Vec<TensorType>, TensorType) {
        let inputs = inputs
            .iter()
            .zip(&self.requirements.inputs)
            .map(|(input, requirement)| TensorType {
                shape: input.shape.clone(),
                format: requirement.format.clone(),
            })
            .collect();
        let output = TensorType {
            shape: output.clone(),
            format: self.requirements.output.format.clone(),
        };
        (inputs, output)
    }

    pub(super) fn supports(&self, inputs: &[TensorType], output: &TensorShape) -> bool {
        if self.requirements.inputs.len() != inputs.len()
            || !valid_requirement(&self.requirements.output, output)
            || !self
                .requirements
                .inputs
                .iter()
                .zip(inputs)
                .all(|(requirement, input)| valid_requirement(requirement, &input.shape))
        {
            return false;
        }
        let alias_valid = match &self.requirements.output_aliasing {
            OutputAliasing::Fresh => true,
            OutputAliasing::MayAliasInputs(indices) => {
                !indices.is_empty()
                    && indices.iter().any(|index| {
                        alias_compatible(
                            usize::from(*index),
                            &self.requirements.inputs,
                            inputs,
                            &self.requirements.output,
                            output,
                        )
                    })
            }
        };
        let requirements_valid = alias_valid
            && self.requirements.distinct_elements.iter().all(|operands| {
                operands.len() >= 2
                    && operands
                        .iter()
                        .all(|operand| valid_memory_operand(*operand, inputs.len()))
                    && operands
                        .iter()
                        .enumerate()
                        .all(|(index, operand)| !operands[..index].contains(operand))
            });
        if !requirements_valid {
            return false;
        }
        let (planned_inputs, planned_output) = self.tensor_types(inputs, output);
        self.validate(&planned_inputs, &planned_output).is_ok()
    }
    pub fn validate(
        &self,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> Result<(), OperatorPlanError> {
        if inputs.len() != self.requirements.inputs.len() {
            return Err(OperatorPlanError::OperandArity);
        }
        if self.dispatch.empty_output_shard_policy() == EmptyOutputShardPolicy::Reject
            && layout_has_empty_shards(&output.format.layout, &output.shape)
        {
            return Err(OperatorPlanError::EmptyOutputShard);
        }
        match (&self.operator, &self.dispatch) {
            (
                MidOperator::Gemm {
                    options, multiply, ..
                },
                OperatorDispatch::BlockedGemm {
                    inner_block,
                    output_column_block,
                    distribution,
                    orientation,
                },
            ) => {
                let [left, right] = inputs else {
                    return Err(OperatorPlanError::OperandArity);
                };
                if matches!(distribution, GemmDistribution::OutputStationary)
                    && left.format.layout.tiling.tile_count
                        != output.format.layout.tiling.tile_count
                {
                    return Err(OperatorPlanError::IncompatibleTileGroups);
                }
                let formats_match_orientation = match orientation {
                    GemmOrientation::Normal => {
                        matches!(left.format.layout.order, ElementOrder::Amp(AmpOrder::Left))
                            && matches!(
                                right.format.layout.order,
                                ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. })
                            )
                            && (output.format.layout.order
                                == ElementOrder::Amp(if *multiply == Precision::F16 {
                                    AmpOrder::Left
                                } else {
                                    AmpOrder::Output
                                })
                                || (*multiply == Precision::F16
                                    && output.format.layout.order.gemm_output_group().is_some()
                                    && !output.format.layout.order.gemm_output_transposed()))
                    }
                    GemmOrientation::Swapped => {
                        matches!(
                            left.format.layout.order,
                            ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix { .. })
                        ) && right.format.layout.order
                            == ElementOrder::Amp(AmpOrder::TransposedLeft)
                            && (output.format.layout.order
                                == ElementOrder::Amp(if *multiply == Precision::F16 {
                                    AmpOrder::TransposedLeft
                                } else {
                                    AmpOrder::TransposedOutput
                                })
                                || (*multiply == Precision::F16
                                    && output.format.layout.order.gemm_output_group().is_some()
                                    && output.format.layout.order.gemm_output_transposed()))
                    }
                };
                if options.transpose_left || options.transpose_right || !formats_match_orientation {
                    return Err(OperatorPlanError::UnsupportedGemmLayout);
                }
                if *inner_block == 0
                    || *output_column_block == 0
                    || left.shape.0.len() < 2
                    || output.shape.0.len() < 2
                {
                    return Err(OperatorPlanError::InvalidBlocking);
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
                    let result_rows = row_partitions.saturating_mul(*result_row_partitions);
                    let result_columns =
                        column_partitions.saturating_mul(*result_column_partitions);
                    let expected_tiles = result_rows.saturating_mul(result_columns);
                    let row_axis = match orientation {
                        GemmOrientation::Normal => TensorAxis::FromEnd(2),
                        GemmOrientation::Swapped => TensorAxis::FromEnd(1),
                    };
                    let column_axis = match orientation {
                        GemmOrientation::Normal => TensorAxis::FromEnd(1),
                        GemmOrientation::Swapped => TensorAxis::FromEnd(2),
                    };
                    let axis_partitions = |axis| {
                        output
                            .format
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
                        || output.format.layout.tiling.tile_count != expected_tiles
                        || axis_partitions(row_axis) != Some(result_rows)
                        || axis_partitions(column_axis) != Some(result_columns)
                    {
                        return Err(OperatorPlanError::InvalidBlocking);
                    }
                }
                let physical_left = match orientation {
                    GemmOrientation::Normal => left,
                    GemmOrientation::Swapped => right,
                };
                let left_layout = physical_left
                    .format
                    .layout
                    .resolve(&physical_left.shape)
                    .map_err(|_| OperatorPlanError::InvalidBlocking)?;
                let output_layout = output
                    .format
                    .layout
                    .resolve(&output.shape)
                    .map_err(|_| OperatorPlanError::InvalidBlocking)?;
                let output_column_axis = output_layout.padded_shape.0.len()
                    - match orientation {
                        GemmOrientation::Normal => 1,
                        GemmOrientation::Swapped => 2,
                    };
                let output_axis = &output_layout
                    .axes()
                    .ok_or(OperatorPlanError::InvalidBlocking)?[output_column_axis];
                let columns_per_output_shard = output_axis.maximum_extent();
                let physical_right = match orientation {
                    GemmOrientation::Normal => right,
                    GemmOrientation::Swapped => left,
                };
                let right_layout = physical_right
                    .format
                    .layout
                    .resolve(&physical_right.shape)
                    .map_err(|_| OperatorPlanError::InvalidBlocking)?;
                let right_column_axis = right_layout.padded_shape.0.len()
                    - match orientation {
                        GemmOrientation::Normal => 1,
                        GemmOrientation::Swapped => 2,
                    };
                let right_axis = &right_layout
                    .axes()
                    .ok_or(OperatorPlanError::InvalidBlocking)?[right_column_axis];
                let columns_per_right_shard =
                    if matches!(distribution, GemmDistribution::ParallelReduction { .. }) {
                        right_axis.maximum_extent()
                    } else {
                        right_axis.minimum_extent()
                    };
                let grid_plan = left.format.layout.tiling.replicas > 1
                    || right.format.layout.tiling.replicas > 1
                    || right
                        .format
                        .layout
                        .tiling
                        .axes
                        .iter()
                        .any(|axis| axis.axis == TensorAxis::FromEnd(2) && axis.partitions > 1);
                if grid_plan
                    && [&left_layout, &right_layout, &output_layout]
                        .into_iter()
                        .any(|layout| layout.has_empty_shards())
                {
                    return Err(OperatorPlanError::InvalidBlocking);
                }
                let physical_left_inner_axis = left_layout.padded_shape.0.len()
                    - match orientation {
                        GemmOrientation::Normal => 1,
                        GemmOrientation::Swapped => 2,
                    };
                let balanced_output_columns =
                    matches!(distribution, GemmDistribution::ParallelReduction { .. });
                let output_shard_alignment = if balanced_output_columns {
                    AMP_COLUMN_MICRO
                } else {
                    *output_column_block
                };
                if !left_layout.padded_shape.0[physical_left_inner_axis]
                    .is_multiple_of(*inner_block)
                    || !output_layout.padded_shape.0[output_column_axis]
                        .is_multiple_of(output_shard_alignment)
                    || !columns_per_output_shard.is_multiple_of(output_shard_alignment)
                    || (balanced_output_columns && columns_per_output_shard > *output_column_block)
                    || columns_per_right_shard < *output_column_block
                {
                    return Err(OperatorPlanError::InvalidBlocking);
                }
                Ok(())
            }
            (
                MidOperator::Gelu,
                OperatorDispatch::Pointwise {
                    kernel: TileKernelSpec::Gelu,
                    ..
                },
            )
            | (
                MidOperator::Add,
                OperatorDispatch::Pointwise {
                    kernel: TileKernelSpec::Add,
                    ..
                },
            ) => {
                let output_tiles = output.format.layout.tiling.tile_count;
                if inputs
                    .iter()
                    .any(|input| input.format.layout.tiling.tile_count != output_tiles)
                {
                    Err(OperatorPlanError::IncompatibleTileGroups)
                } else {
                    Ok(())
                }
            }
            (
                MidOperator::FlashAttention {
                    options,
                    accumulate,
                },
                OperatorDispatch::Attention {
                    materialized,
                    key_block_rows,
                    padded_query_dimension,
                    padded_value_dimension,
                },
            ) => {
                let [query, key, value] = inputs else {
                    return Err(OperatorPlanError::OperandArity);
                };
                if options.causal
                    || *accumulate != AccumulationPrecision::F32
                    || *key_block_rows == 0
                    || !key_block_rows.is_multiple_of(AMP_INNER_BLOCK)
                    || (!materialized && *key_block_rows != AMP_INNER_BLOCK)
                    || *padded_query_dimension == 0
                    || *padded_value_dimension == 0
                    || !matches!(query.format.layout.order, ElementOrder::Amp(AmpOrder::Left))
                    || !matches!(
                        key.format.layout.order,
                        ElementOrder::Amp(AmpOrder::TransposedRight)
                    )
                    || !matches!(
                        value.format.layout.order,
                        ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. })
                    )
                    || output.format.layout.order != ElementOrder::RowMajor
                    || query.format.layout.tiling.tile_count
                        != output.format.layout.tiling.tile_count
                    || key.format.layout.tiling.tile_count != value.format.layout.tiling.tile_count
                {
                    Err(OperatorPlanError::InvalidBlocking)
                } else {
                    Ok(())
                }
            }
            (
                MidOperator::FlashAttention {
                    options,
                    accumulate,
                },
                OperatorDispatch::Pointwise {
                    kernel:
                        TileKernelSpec::FlashAttention {
                            options: kernel_options,
                            accumulate: kernel_accumulate,
                        },
                    ..
                },
            ) if options == kernel_options && accumulate == kernel_accumulate => {
                let output_tiles = output.format.layout.tiling.tile_count;
                if inputs
                    .iter()
                    .any(|input| input.format.layout.tiling.tile_count != output_tiles)
                {
                    Err(OperatorPlanError::IncompatibleTileGroups)
                } else {
                    Ok(())
                }
            }
            (MidOperator::View(view), OperatorDispatch::View) => {
                let [input] = inputs else {
                    return Err(OperatorPlanError::OperandArity);
                };
                if view.output_shape(&input.shape).as_ref() != Some(&output.shape)
                    || input.format.precision != output.format.precision
                {
                    Err(OperatorPlanError::InvalidBlocking)
                } else {
                    Ok(())
                }
            }
            _ => Err(OperatorPlanError::DispatchMismatch),
        }
    }
}
