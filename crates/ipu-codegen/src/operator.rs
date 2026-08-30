//! Whole-device operator plans and tile-kernel specifications.

use crate::conversion::DeferredTransform;
use crate::graph::{AddOptions, AttentionOptions, GemmOptions, TensorShape};
use crate::ir::MidValueId;
use crate::layout::{
    AMP_COLUMN_MICRO, AMP_INNER_BLOCK, Layout, MemoryClass, NativeKernelOrder, StorageOrder,
    TensorAxis, TensorFormat, TensorType,
};

/// In-memory representation of one tensor element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Precision {
    /// F143 values scaled by a tensor-wide power of two.
    F8F143 {
        scale_exponent: i8,
    },
    F16,
    F32,
}

impl Precision {
    pub const fn bytes(self) -> u64 {
        match self {
            Self::F8F143 { .. } => 1,
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccumulationPrecision {
    F16,
    F32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MidOperator {
    Gemm {
        options: GemmOptions,
        multiply: Precision,
        accumulate: AccumulationPrecision,
    },
    Gelu,
    Add(AddOptions),
    FlashAttention {
        options: AttentionOptions,
        accumulate: AccumulationPrecision,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GemmKernelMode {
    Initialize,
    Accumulate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GemmWeightLoad {
    Standard,
    Interleaved,
}

/// Linearization of a GEMM's logical tile grid.
///
/// The order is part of the operand and output layouts because it determines
/// which tensor coordinates occupy adjacent logical (and therefore paired
/// physical) tiles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GridOrder {
    #[default]
    ColumnsFast,
    RowsFast,
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

/// Shape-independent recipe which expands into ordered device-wide exchange
/// and tile-kernel phases after concrete shards are known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OperatorDispatch {
    Pointwise(PointwiseInputMapping),
    BlockedGemm(BlockedGemmPlan),
    Attention(AttentionPlan),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AttentionPadding {
    pub query_dimension: u32,
    pub value_dimension: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AttentionBlocking {
    Flash {
        query_rows: u32,
        key_rows: u32,
    },
    Materialized {
        query_rows: u32,
        padded_key_rows: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttentionPlan {
    pub kernel: GemmKernelFamily,
    pub blocking: AttentionBlocking,
    pub padding: AttentionPadding,
}

impl AttentionPlan {
    pub fn gemm_blocks(&self) -> [GemmBlockShape; 2] {
        let key_columns = match self.blocking {
            AttentionBlocking::Flash { key_rows, .. } => key_rows,
            AttentionBlocking::Materialized {
                padded_key_rows, ..
            } => padded_key_rows,
        };
        [
            GemmBlockShape {
                inner: self.padding.query_dimension,
                output_columns: key_columns,
            },
            GemmBlockShape {
                inner: key_columns,
                output_columns: self.padding.value_dimension,
            },
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeferredInputPlan {
    pub producer: MidValueId,
    pub source: MidValueId,
    pub transform: DeferredTransform,
}

/// Which operand remains resident while a blocked whole-device GEMM is run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GemmDistribution {
    #[default]
    OutputStationary,
    /// Distribute independent row, output-column, and K block ranges. Each
    /// row/K activation shard is replicated over the column groups, computes
    /// one local partial, and is reduced over K onto the output owner.
    ParallelReduction(ParallelReductionPlan),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GemmGrid {
    pub rows: u16,
    pub columns: u16,
    pub inner: u16,
}

impl GemmGrid {
    pub const fn tile_count(self) -> u16 {
        self.rows
            .saturating_mul(self.columns)
            .saturating_mul(self.inner)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GemmResultGrid {
    pub rows: u16,
    pub columns: u16,
}

impl GemmResultGrid {
    pub const fn tile_count(self) -> u16 {
        self.rows.saturating_mul(self.columns)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParallelReductionPlan {
    pub compute: GemmGrid,
    pub staging: ReductionStaging,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GemmBlockShape {
    pub inner: u32,
    pub output_columns: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmKernelFamily {
    pub multiply: Precision,
    pub accumulate: AccumulationPrecision,
    pub weights: GemmWeightLoad,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmGeometry {
    pub block: GemmBlockShape,
    pub orientation: GemmOrientation,
    /// Spatial ownership of the final result. Parallel reductions may spread
    /// roots over former K-partition tiles.
    pub result: GemmResultGrid,
    pub order: GridOrder,
    pub distribution: GemmDistribution,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockedGemmPlan {
    pub kernel: GemmKernelFamily,
    pub geometry: GemmGeometry,
}

/// Exact blocked-GEMM geometry retained for planner diagnosis. Constraints
/// are keyed by the source graph operation and bypass beam pruning and
/// conservative whole-graph memory rejection. Concrete placement remains the
/// final authority on whether the resulting package fits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmPlanConstraint {
    pub source_operation: u32,
    pub geometry: GemmGeometry,
    pub weight_memory_class: MemoryClass,
    pub local_weight_staging: LocalOperandStaging,
}

/// Lifetime policy for partials reduced across a GEMM's K partitions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReductionStaging {
    /// Receive every remote partial into one packed buffer, then reduce once.
    #[default]
    Complete,
    /// Receive and accumulate one remote partial at a time. This minimizes
    /// temporary SRAM at the expense of additional exchange epochs and kernel
    /// launches.
    Streamed,
}

/// How a pointwise kernel's input shards are selected for each output shard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointwiseInputMapping {
    /// Each input view is selected by its logical overlap with the output and
    /// singleton dimensions may be broadcast.
    BroadcastToOutput,
    /// Each input must already have a shard resident on the output tile.
    TileLocal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmptyOutputShardPolicy {
    Skip,
    Reject,
}

impl OperatorDispatch {
    fn empty_output_shard_policy(&self) -> EmptyOutputShardPolicy {
        match self {
            Self::Pointwise(_) => EmptyOutputShardPolicy::Skip,
            Self::BlockedGemm(_) | Self::Attention(_) => EmptyOutputShardPolicy::Reject,
        }
    }
}

pub(crate) fn layout_has_empty_shards(layout: &Layout, shape: &TensorShape) -> bool {
    layout
        .resolve(shape)
        .map_or(true, |resolved| resolved.has_empty_shards())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperandRequirement {
    pub format: TensorFormat,
    pub allocation: AllocationRequirements,
    /// How a locally resident operand should be consumed when other tiles use
    /// an operator-local staging buffer for the same operand.
    pub local_staging: LocalOperandStaging,
    /// Whether a dispatch may populate and consume bounded operand slices
    /// instead of materializing the complete required format first.
    pub materialization: OperandMaterialization,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MemoryElementRequirement {
    #[default]
    Any,
    Distinct,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocationRequirements {
    pub alignment: u32,
    /// Bytes the kernel may access beyond the logical tensor payload.
    pub access_tail_bytes: u32,
    pub memory_element: MemoryElementRequirement,
}

impl AllocationRequirements {
    pub fn merge(&mut self, other: Self) {
        self.alignment = self.alignment.max(other.alignment);
        self.access_tail_bytes = self.access_tail_bytes.max(other.access_tail_bytes);
        if other.memory_element == MemoryElementRequirement::Distinct {
            self.memory_element = MemoryElementRequirement::Distinct;
        }
    }

    pub fn require_distinct_element(&mut self) {
        self.memory_element = MemoryElementRequirement::Distinct;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum LocalOperandStaging {
    #[default]
    Direct,
    MatchRemote,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OperandMaterialization {
    #[default]
    Complete,
    DispatchSlices,
}

impl OperandRequirement {
    pub fn new(format: TensorFormat, alignment: u32) -> Self {
        Self {
            format,
            allocation: AllocationRequirements {
                alignment,
                ..AllocationRequirements::default()
            },
            local_staging: LocalOperandStaging::Direct,
            materialization: OperandMaterialization::Complete,
        }
    }

    pub fn with_access_tail(mut self, bytes: u32) -> Self {
        self.allocation.access_tail_bytes = bytes;
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputAliasing {
    Fresh,
    MayAliasInputs(Vec<u16>),
    MustAliasInput(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryOperand {
    Output,
    Input(u16),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemorySpaceRequirements {
    /// Each group names operand ranges which must occupy distinct effective
    /// tile-memory elements.
    pub distinct_element_groups: Vec<Vec<MemoryOperand>>,
}

impl MemorySpaceRequirements {
    pub fn with_distinct_elements(
        mut self,
        operands: impl IntoIterator<Item = MemoryOperand>,
    ) -> Self {
        self.distinct_element_groups
            .push(operands.into_iter().collect());
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatorRequirements {
    pub inputs: Vec<OperandRequirement>,
    pub output: OperandRequirement,
    pub output_aliasing: OutputAliasing,
    pub memory_space: MemorySpaceRequirements,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatorPlan {
    pub operator: MidOperator,
    pub dispatch: OperatorDispatch,
    pub requirements: OperatorRequirements,
    /// Deferred producer results claimed by each input operand.
    pub deferred_inputs: Vec<Option<DeferredInputPlan>>,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
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

fn alias_compatible(
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

fn valid_requirement(requirement: &OperandRequirement, shape: &TensorShape) -> bool {
    requirement.allocation.alignment.is_power_of_two()
        && requirement.format.layout.resolve(shape).is_ok()
}

fn valid_memory_operand(operand: MemoryOperand, input_count: usize) -> bool {
    match operand {
        MemoryOperand::Output => true,
        MemoryOperand::Input(index) => usize::from(index) < input_count,
    }
}

impl OperatorPlan {
    pub(crate) fn candidate(
        operator: MidOperator,
        dispatch: OperatorDispatch,
        requirements: OperatorRequirements,
    ) -> Self {
        let input_count = requirements.inputs.len();
        Self {
            operator,
            dispatch,
            requirements,
            deferred_inputs: vec![None; input_count],
        }
    }

    pub(crate) fn supports(&self, inputs: &[TensorType], output: &TensorShape) -> bool {
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
            OutputAliasing::MustAliasInput(index) => alias_compatible(
                usize::from(*index),
                &self.requirements.inputs,
                inputs,
                &self.requirements.output,
                output,
            ),
        };
        if !alias_valid
            || !self
                .requirements
                .memory_space
                .distinct_element_groups
                .iter()
                .all(|operands| {
                    operands.len() >= 2
                        && operands
                            .iter()
                            .all(|operand| valid_memory_operand(*operand, inputs.len()))
                        && operands.iter().enumerate().all(|(index, operand)| {
                            !operands[..index].iter().any(|previous| previous == operand)
                        })
                })
        {
            return false;
        }
        let planned_inputs = inputs
            .iter()
            .zip(&self.requirements.inputs)
            .map(|(input, requirement)| TensorType {
                shape: input.shape.clone(),
                format: requirement.format.clone(),
            })
            .collect::<Vec<_>>();
        let planned_output = TensorType {
            shape: output.clone(),
            format: self.requirements.output.format.clone(),
        };
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
                OperatorDispatch::BlockedGemm(plan),
            ) => {
                let inner_block = &plan.geometry.block.inner;
                let output_column_block = &plan.geometry.block.output_columns;
                let distribution = &plan.geometry.distribution;
                let orientation = &plan.geometry.orientation;
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
                        matches!(
                            left.format.layout.order,
                            StorageOrder::Native(NativeKernelOrder::Left)
                        ) && matches!(
                            right.format.layout.order,
                            StorageOrder::Blocked(order) if order.is_matrix()
                        ) && output.format.layout.order
                            == StorageOrder::Native(if *multiply == Precision::F16 {
                                NativeKernelOrder::Left
                            } else {
                                NativeKernelOrder::Output
                            })
                    }
                    GemmOrientation::Swapped => {
                        matches!(
                            left.format.layout.order,
                            StorageOrder::Blocked(order) if order.is_transposed_matrix()
                        ) && right.format.layout.order
                            == StorageOrder::Native(NativeKernelOrder::TransposedLeft)
                            && output.format.layout.order
                                == StorageOrder::Native(if *multiply == Precision::F16 {
                                    NativeKernelOrder::TransposedLeft
                                } else {
                                    NativeKernelOrder::TransposedOutput
                                })
                    }
                };
                if options.transpose_left || options.transpose_right || !formats_match_orientation {
                    return Err(OperatorPlanError::UnsupportedGemmLayout);
                }
                let MidOperator::Gemm {
                    multiply,
                    accumulate,
                    ..
                } = &self.operator
                else {
                    return Err(OperatorPlanError::DispatchMismatch);
                };
                if plan.kernel.multiply != *multiply || plan.kernel.accumulate != *accumulate {
                    return Err(OperatorPlanError::DispatchMismatch);
                }
                if *inner_block == 0
                    || *output_column_block == 0
                    || left.shape.0.len() < 2
                    || output.shape.0.len() < 2
                {
                    return Err(OperatorPlanError::InvalidBlocking);
                }
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
                        .map_or(1, |tiling| tiling.partitions)
                };
                if output.format.layout.tiling.tile_count != plan.geometry.result.tile_count()
                    || axis_partitions(row_axis) != plan.geometry.result.rows
                    || axis_partitions(column_axis) != plan.geometry.result.columns
                {
                    return Err(OperatorPlanError::InvalidBlocking);
                }
                if let GemmDistribution::ParallelReduction(reduction) = distribution {
                    let result_rows = plan.geometry.result.rows;
                    let result_columns = plan.geometry.result.columns;
                    let result_row_partitions =
                        result_rows.checked_div(reduction.compute.rows).unwrap_or(0);
                    let result_column_partitions = result_columns
                        .checked_div(reduction.compute.columns)
                        .unwrap_or(0);
                    if reduction.compute.rows == 0
                        || reduction.compute.columns == 0
                        || reduction.compute.inner < 2
                        || result_rows == 0
                        || result_columns == 0
                        || !result_rows.is_multiple_of(reduction.compute.rows)
                        || !result_columns.is_multiple_of(reduction.compute.columns)
                        || result_row_partitions.saturating_mul(result_column_partitions)
                            > reduction.compute.inner
                        || axis_partitions(row_axis) != result_rows
                        || axis_partitions(column_axis) != result_columns
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
                let left_padded = left_layout.padded_shape();
                let output_padded = output_layout.padded_shape();
                let output_column_axis = output_padded.0.len()
                    - match orientation {
                        GemmOrientation::Normal => 1,
                        GemmOrientation::Swapped => 2,
                    };
                let columns_per_output_shard = output_layout
                    .maximum_axis_extent(output_column_axis)
                    .ok_or(OperatorPlanError::InvalidBlocking)?;
                let physical_right = match orientation {
                    GemmOrientation::Normal => right,
                    GemmOrientation::Swapped => left,
                };
                let right_layout = physical_right
                    .format
                    .layout
                    .resolve(&physical_right.shape)
                    .map_err(|_| OperatorPlanError::InvalidBlocking)?;
                let right_padded = right_layout.padded_shape();
                let right_column_axis = right_padded.0.len()
                    - match orientation {
                        GemmOrientation::Normal => 1,
                        GemmOrientation::Swapped => 2,
                    };
                let columns_per_right_shard = right_layout
                    .maximum_axis_extent(right_column_axis)
                    .ok_or(OperatorPlanError::InvalidBlocking)?;
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
                    && [left, right, output]
                        .into_iter()
                        .any(|tensor| !layout_shards_are_nonempty(tensor))
                {
                    return Err(OperatorPlanError::InvalidBlocking);
                }
                let physical_left_inner_axis = left_padded.0.len()
                    - match orientation {
                        GemmOrientation::Normal => 1,
                        GemmOrientation::Swapped => 2,
                    };
                let balanced_output_columns =
                    matches!(distribution, GemmDistribution::ParallelReduction(_));
                let output_shard_alignment = if balanced_output_columns {
                    AMP_COLUMN_MICRO
                } else {
                    *output_column_block
                };
                if !left_padded.0[physical_left_inner_axis].is_multiple_of(*inner_block)
                    || !output_padded.0[output_column_axis].is_multiple_of(output_shard_alignment)
                    || !columns_per_output_shard.is_multiple_of(output_shard_alignment)
                    || (balanced_output_columns && columns_per_output_shard > *output_column_block)
                    || columns_per_right_shard < *output_column_block
                {
                    return Err(OperatorPlanError::InvalidBlocking);
                }
                Ok(())
            }
            (MidOperator::Gelu | MidOperator::Add(_), OperatorDispatch::Pointwise(_)) => {
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
                OperatorDispatch::Attention(AttentionPlan {
                    kernel,
                    blocking:
                        AttentionBlocking::Flash {
                            query_rows,
                            key_rows,
                        },
                    padding,
                }),
            ) => {
                let [query, key, value] = inputs else {
                    return Err(OperatorPlanError::OperandArity);
                };
                if options.causal
                    || *accumulate != AccumulationPrecision::F32
                    || *query_rows == 0
                    || *key_rows != AMP_INNER_BLOCK
                    || padding.query_dimension == 0
                    || padding.value_dimension == 0
                    || !matches!(
                        query.format.layout.order,
                        StorageOrder::Native(NativeKernelOrder::Left)
                    )
                    || !matches!(
                        key.format.layout.order,
                        StorageOrder::Native(NativeKernelOrder::TransposedRight)
                    )
                    || !matches!(
                        value.format.layout.order,
                        StorageOrder::Blocked(order) if order.is_matrix()
                    )
                    || output.format.layout.order != StorageOrder::Linear
                    || query.format.layout.tiling.tile_count
                        != output.format.layout.tiling.tile_count
                    || key.format.layout.tiling.tile_count != value.format.layout.tiling.tile_count
                    || kernel.multiply != Precision::F16
                    || kernel.accumulate != *accumulate
                    || kernel.weights != GemmWeightLoad::Standard
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
                OperatorDispatch::Attention(AttentionPlan {
                    kernel,
                    blocking:
                        AttentionBlocking::Materialized {
                            query_rows,
                            padded_key_rows,
                        },
                    padding,
                }),
            ) => {
                let [query, key, value] = inputs else {
                    return Err(OperatorPlanError::OperandArity);
                };
                if options.causal
                    || *accumulate != AccumulationPrecision::F32
                    || *query_rows == 0
                    || *padded_key_rows == 0
                    || !padded_key_rows.is_multiple_of(AMP_INNER_BLOCK)
                    || padding.query_dimension == 0
                    || padding.value_dimension == 0
                    || !matches!(
                        query.format.layout.order,
                        StorageOrder::Native(NativeKernelOrder::Left)
                    )
                    || !matches!(
                        key.format.layout.order,
                        StorageOrder::Native(NativeKernelOrder::TransposedRight)
                    )
                    || !matches!(
                        value.format.layout.order,
                        StorageOrder::Blocked(order) if order.is_matrix()
                    )
                    || output.format.layout.order != StorageOrder::Linear
                    || query.format.layout.tiling.tile_count
                        != output.format.layout.tiling.tile_count
                    || key.format.layout.tiling.tile_count != value.format.layout.tiling.tile_count
                    || kernel.multiply != Precision::F16
                    || kernel.accumulate != *accumulate
                    || kernel.weights != GemmWeightLoad::Standard
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

fn layout_shards_are_nonempty(tensor: &TensorType) -> bool {
    tensor
        .format
        .layout
        .resolve(&tensor.shape)
        .is_ok_and(|resolved| !resolved.has_empty_shards())
}
