//! Whole-device operator plans and tile-kernel specifications.

use crate::graph::{AddOptions, AttentionOptions, GemmOptions, SplitHeadsOptions, TensorShape};
use crate::layout::{
    AMP_COLUMN_MICRO, AMP_INNER_BLOCK, AmpOrder, BlockMajorOrder, ElementOrder, Layout,
    MemoryClass, TensorAxis, TensorFormat, TensorType,
};
use crate::mid::MidValueId;

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
    SplitHeads(SplitHeadsOptions),
    FlashAttention {
        options: AttentionOptions,
        accumulate: AccumulationPrecision,
    },
}

/// A tile-local callable selected by a whole-device operator plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileKernelSpec {
    FillZero,
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
    Pointwise {
        kernel: TileKernelSpec,
        input_mapping: PointwiseInputMapping,
    },
    BlockedGemm(BlockedGemmPlan),
    Attention(AttentionPlan),
    /// Redistribute packed projection columns into independent attention
    /// streams. The byte mapping is performed directly by local copies and
    /// exchanges, without a tile kernel.
    SplitHeads,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttentionKernelFamily {
    pub query_key: TileKernelSpec,
    pub probability_value: TileKernelSpec,
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
    pub kernels: AttentionKernelFamily,
    pub blocking: AttentionBlocking,
    pub padding: AttentionPadding,
}

/// A logical value transformation whose physical materialization may be
/// deferred until a consumer requests bounded slices.  The transform is
/// independent of either producer or consumer operator kinds, so additional
/// view-like operators can participate without adding pairs of dispatch
/// special cases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeferredTransform {
    /// Split the final input axis into `parts` equal-width slices and fold the
    /// new part axis into the leading output axis.
    SplitLastAxisIntoLeading { parts: u32 },
}

impl DeferredTransform {
    /// Map one rectangular output slice back to a rectangular producer slice.
    /// Returning `None` means that the requested slice crosses a transform
    /// discontinuity and must be divided before dispatch.
    pub fn map_slice(
        self,
        source_shape: &TensorShape,
        output_shape: &TensorShape,
        output: &[(u32, u32)],
    ) -> Option<DeferredSliceMapping> {
        match self {
            Self::SplitLastAxisIntoLeading { parts } => {
                let [source_batch, source_rows, source_columns] = source_shape.0.as_slice() else {
                    return None;
                };
                let [output_streams, output_rows, output_columns] = output_shape.0.as_slice()
                else {
                    return None;
                };
                let [(stream_start, stream_end), rows, columns] = output else {
                    return None;
                };
                if parts == 0
                    || *stream_end != stream_start.checked_add(1)?
                    || *output_streams != source_batch.checked_mul(parts)?
                    || output_rows != source_rows
                    || source_columns != &output_columns.checked_mul(parts)?
                    || *stream_end > *output_streams
                    || rows.1 > *output_rows
                    || columns.1 > *output_columns
                {
                    return None;
                }
                let batch = stream_start / parts;
                let part = stream_start % parts;
                let column_base = part.checked_mul(*output_columns)?;
                Some(DeferredSliceMapping {
                    source_ranges: vec![
                        (batch, batch.checked_add(1)?),
                        *rows,
                        (
                            column_base.checked_add(columns.0)?,
                            column_base.checked_add(columns.1)?,
                        ),
                    ],
                    destination_source_axes: vec![1, 2],
                })
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeferredSliceMapping {
    pub source_ranges: Vec<(u32, u32)>,
    /// Source axes retained, in destination-axis order. Removed axes select a
    /// slice but do not occupy storage in the consumer's dispatch buffer.
    pub destination_source_axes: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DeferredOutputPlan {
    pub source_input: usize,
    pub transform: DeferredTransform,
    /// Cost restored if no later consumer claims this offer.
    pub unfused_cycles: u64,
    /// Exchange portion of `unfused_cycles`, restored with the offer.
    pub unfused_exchange_cycles: u64,
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
    /// Additional spatial partitions of each computed output block. Their
    /// product cannot exceed the K partition count; reduction roots are
    /// spread over former K-partition tiles rather than concentrated on one
    /// root per compute row/column block.
    pub result: GemmResultGrid,
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

impl GemmKernelFamily {
    pub fn kernel(self, mode: GemmKernelMode, block: GemmBlockShape) -> TileKernelSpec {
        TileKernelSpec::Gemm {
            multiply: self.multiply,
            accumulate: self.accumulate,
            mode,
            weights: self.weights,
            inner_block: block.inner,
            output_columns: block.output_columns,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmGeometry {
    pub block: GemmBlockShape,
    pub orientation: GemmOrientation,
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
            Self::Pointwise { .. } => EmptyOutputShardPolicy::Skip,
            Self::SplitHeads => EmptyOutputShardPolicy::Reject,
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
    /// A view transformation offered by this plan. It is materialized normally
    /// unless a later plan records a matching entry in `deferred_inputs`.
    pub deferred_output: Option<DeferredOutputPlan>,
    /// Deferred producer results claimed by each input operand.
    pub deferred_inputs: Vec<Option<DeferredInputPlan>>,
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

impl ConversionStrategy {
    pub const fn uses_intersections(self) -> bool {
        matches!(self, Self::DirectRetile | Self::StageLogicalThenTransform)
    }
}

pub fn layout_conversion_strategy(from: &Layout, to: &Layout) -> ConversionStrategy {
    if from.order == to.order {
        ConversionStrategy::DirectRetile
    } else {
        ConversionStrategy::StageLogicalThenTransform
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversionPlan {
    pub kernel: TileKernelSpec,
    pub input: OperandRequirement,
    pub output: OperandRequirement,
    pub strategy: ConversionStrategy,
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

pub(crate) fn alias_compatible(
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

pub(crate) fn valid_requirement(requirement: &OperandRequirement, shape: &TensorShape) -> bool {
    requirement.allocation.alignment.is_power_of_two()
        && requirement.format.layout.resolve(shape).is_ok()
}

pub(crate) fn valid_memory_operand(operand: MemoryOperand, input_count: usize) -> bool {
    match operand {
        MemoryOperand::Output => true,
        MemoryOperand::Input(index) => usize::from(index) < input_count,
    }
}

impl OperatorPlan {
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
                        matches!(left.format.layout.order, ElementOrder::Amp(AmpOrder::Left))
                            && matches!(
                                right.format.layout.order,
                                ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. })
                            )
                            && output.format.layout.order
                                == ElementOrder::Amp(if *multiply == Precision::F16 {
                                    AmpOrder::Left
                                } else {
                                    AmpOrder::Output
                                })
                    }
                    GemmOrientation::Swapped => {
                        matches!(
                            left.format.layout.order,
                            ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix { .. })
                        ) && right.format.layout.order
                            == ElementOrder::Amp(AmpOrder::TransposedLeft)
                            && output.format.layout.order
                                == ElementOrder::Amp(if *multiply == Precision::F16 {
                                    AmpOrder::TransposedLeft
                                } else {
                                    AmpOrder::TransposedOutput
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
                if let GemmDistribution::ParallelReduction(reduction) = distribution {
                    let result_rows = reduction.compute.rows.saturating_mul(reduction.result.rows);
                    let result_columns = reduction
                        .compute
                        .columns
                        .saturating_mul(reduction.result.columns);
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
                    if reduction.compute.rows == 0
                        || reduction.compute.columns == 0
                        || reduction.compute.inner < 2
                        || reduction.result.rows == 0
                        || reduction.result.columns == 0
                        || reduction.result.tile_count() > reduction.compute.inner
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
            (
                MidOperator::Gelu,
                OperatorDispatch::Pointwise {
                    kernel: TileKernelSpec::Gelu,
                    ..
                },
            )
            | (
                MidOperator::Add(_),
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
                OperatorDispatch::Attention(AttentionPlan {
                    kernels,
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
                    || !matches!(kernels.query_key, TileKernelSpec::Gemm { .. })
                    || !matches!(kernels.probability_value, TileKernelSpec::Gemm { .. })
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
                    kernels,
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
                    || !matches!(kernels.query_key, TileKernelSpec::Gemm { .. })
                    || !matches!(kernels.probability_value, TileKernelSpec::Gemm { .. })
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
            (MidOperator::SplitHeads(_), OperatorDispatch::SplitHeads) => {
                let [input] = inputs else {
                    return Err(OperatorPlanError::OperandArity);
                };
                if input.shape.0.len() != 3
                    || output.shape.0.len() != 3
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

fn layout_shards_are_nonempty(tensor: &TensorType) -> bool {
    tensor
        .format
        .layout
        .resolve(&tensor.shape)
        .is_ok_and(|resolved| !resolved.has_empty_shards())
}
