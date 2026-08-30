//! Whole-device operator plans and tile-kernel specifications.

use crate::graph::{AddOptions, AttentionOptions, GemmOptions, TensorShape};
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

impl GemmOrientation {
    pub const fn physical_left_input(self) -> usize {
        match self {
            Self::Normal => 0,
            Self::Swapped => 1,
        }
    }

    pub const fn physical_right_input(self) -> usize {
        1 - self.physical_left_input()
    }

    pub const fn row_axis(self) -> TensorAxis {
        match self {
            Self::Normal => TensorAxis::FromEnd(2),
            Self::Swapped => TensorAxis::FromEnd(1),
        }
    }

    pub const fn column_axis(self) -> TensorAxis {
        match self {
            Self::Normal => TensorAxis::FromEnd(1),
            Self::Swapped => TensorAxis::FromEnd(2),
        }
    }

    pub(crate) fn physical_order<T>(self, values: [T; 2]) -> [T; 2] {
        match self {
            Self::Normal => values,
            Self::Swapped => {
                let [left, right] = values;
                [right, left]
            }
        }
    }
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

impl AttentionBlocking {
    pub const fn query_rows(self) -> u32 {
        match self {
            Self::Flash { query_rows, .. } | Self::Materialized { query_rows, .. } => query_rows,
        }
    }

    pub const fn key_block_rows(self) -> u32 {
        match self {
            Self::Flash { key_rows, .. } => key_rows,
            Self::Materialized { .. } => AMP_INNER_BLOCK,
        }
    }
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
            AttentionBlocking::Materialized {
                padded_key_rows, ..
            } => padded_key_rows,
            blocking => blocking.key_block_rows(),
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
    /// Spatial row, output-column, and K partitions which invoke kernels.
    pub compute: GemmGrid,
    /// Spatial ownership of the final result. Parallel reductions may spread
    /// roots over former K-partition tiles.
    pub result: GemmResultGrid,
    pub order: GridOrder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockedGemmPlan {
    pub kernel: GemmKernelFamily,
    pub geometry: GemmGeometry,
}

impl BlockedGemmPlan {
    /// Tensor produced by the compute grid before a parallel K reduction.
    ///
    /// This is part of the selected whole-device plan: costing and low
    /// materialization must use the same partial ownership and padding.
    pub(crate) fn partial_tensor(&self, output: &TensorType) -> Option<TensorType> {
        if self.geometry.compute.inner < 2 {
            return Some(output.clone());
        }
        let output_rank = output.shape.0.len();
        let output_column_axis = self
            .geometry
            .orientation
            .column_axis()
            .resolve(output_rank)
            .ok()?;
        let column_tiling = *output
            .format
            .layout
            .tiling
            .axes
            .iter()
            .find(|axis| axis.axis.resolve(output_rank).ok() == Some(output_column_axis))?;
        let rows = self.geometry.compute.rows;
        let columns = self.geometry.compute.columns;
        let tiles = rows.checked_mul(columns)?;
        let left_order = match self.geometry.orientation {
            GemmOrientation::Normal => NativeKernelOrder::Left,
            GemmOrientation::Swapped => NativeKernelOrder::TransposedLeft,
        };
        let constructor = if output.format.layout.order == StorageOrder::Native(left_order) {
            Layout::amp_left_result_grid
        } else {
            Layout::amp_output_grid
        };
        let mut layout = constructor(
            self.geometry.orientation,
            self.geometry.block.output_columns,
            tiles,
            rows,
            columns,
            GridOrder::ColumnsFast,
        );
        let axis = layout
            .tiling
            .axes
            .iter_mut()
            .find(|axis| axis.axis.resolve(output_rank).ok() == Some(output_column_axis))?;
        axis.block_size = column_tiling.block_size;
        axis.padding_multiple = column_tiling.block_size;
        if column_tiling.partitions == columns {
            axis.block_size = column_tiling.block_size;
            axis.padding_multiple = column_tiling.padding_multiple;
            axis.shard_padding_multiple = column_tiling.shard_padding_multiple;
        }
        let partial = TensorType {
            shape: output.shape.clone(),
            format: TensorFormat {
                precision: output.format.precision,
                layout,
            },
        };
        partial.format.layout.resolve(&partial.shape).ok()?;
        Some(partial)
    }
}

/// Exact blocked-GEMM geometry retained for planner diagnosis. Constraints
/// are keyed by the source graph operation and bypass beam pruning and
/// conservative whole-graph memory rejection. Concrete placement remains the
/// final authority on whether the resulting package fits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmPlanConstraint {
    pub source_operation: u32,
    pub geometry: GemmGeometry,
    pub reduction_staging: Option<ReductionStaging>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmptyOutputShardPolicy {
    Skip,
    Reject,
}

impl crate::OperatorSchedule {
    fn empty_output_shard_policy(&self) -> EmptyOutputShardPolicy {
        match self.steps.first() {
            Some(crate::ScheduleStep::KernelMap(_)) => EmptyOutputShardPolicy::Skip,
            Some(crate::ScheduleStep::BlockedGemm(_) | crate::ScheduleStep::Attention(_)) => {
                EmptyOutputShardPolicy::Reject
            }
            Some(crate::ScheduleStep::Reduce { .. }) | None => EmptyOutputShardPolicy::Reject,
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
    /// Whether a schedule may populate and consume bounded operand slices
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LocalOperandStaging {
    Direct(MemoryClass),
    Staged(MemoryClass),
}

impl Default for LocalOperandStaging {
    fn default() -> Self {
        Self::Direct(MemoryClass::Standard)
    }
}

impl LocalOperandStaging {
    pub(crate) const fn memory_class(self) -> MemoryClass {
        match self {
            Self::Direct(memory_class) | Self::Staged(memory_class) => memory_class,
        }
    }

    pub(crate) const fn stages_local(self) -> bool {
        matches!(self, Self::Staged(_))
    }
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
            local_staging: LocalOperandStaging::default(),
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
    pub schedule: crate::OperatorSchedule,
    pub requirements: OperatorRequirements,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum OperatorPlanError {
    #[error("operator plan operand arity does not match its requirements")]
    OperandArity,
    #[error("operator plan schedule does not match the selected operator")]
    DispatchMismatch,
    #[error("operator plan uses zero or incompatible block dimensions")]
    InvalidBlocking,
    #[error("operator plan requires corresponding activation and output tile groups")]
    IncompatibleTileGroups,
    #[error("operator schedule does not support empty output shards")]
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
        schedule: crate::OperatorSchedule,
        requirements: OperatorRequirements,
    ) -> Self {
        Self {
            operator,
            schedule,
            requirements,
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
        if self.schedule.empty_output_shard_policy() == EmptyOutputShardPolicy::Reject
            && layout_has_empty_shards(&output.format.layout, &output.shape)
        {
            return Err(OperatorPlanError::EmptyOutputShard);
        }
        match (&self.operator, self.schedule.steps.as_slice()) {
            (
                MidOperator::Gemm {
                    options, multiply, ..
                },
                [crate::ScheduleStep::BlockedGemm(plan), ..],
            ) => {
                let inner_block = &plan.geometry.block.inner;
                let output_column_block = &plan.geometry.block.output_columns;
                let compute = plan.geometry.compute;
                let orientation = &plan.geometry.orientation;
                let [left, right] = inputs else {
                    return Err(OperatorPlanError::OperandArity);
                };
                if compute.inner == 1
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
                let row_axis = orientation.row_axis();
                let column_axis = orientation.column_axis();
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
                if compute.inner > 1 {
                    let result_rows = plan.geometry.result.rows;
                    let result_columns = plan.geometry.result.columns;
                    let result_row_partitions = result_rows.checked_div(compute.rows).unwrap_or(0);
                    let result_column_partitions =
                        result_columns.checked_div(compute.columns).unwrap_or(0);
                    if compute.rows == 0
                        || compute.columns == 0
                        || result_rows == 0
                        || result_columns == 0
                        || !result_rows.is_multiple_of(compute.rows)
                        || !result_columns.is_multiple_of(compute.columns)
                        || result_row_partitions.saturating_mul(result_column_partitions)
                            > compute.inner
                        || self.schedule.reduction_staging().is_none()
                        || axis_partitions(row_axis) != result_rows
                        || axis_partitions(column_axis) != result_columns
                    {
                        return Err(OperatorPlanError::InvalidBlocking);
                    }
                } else if compute.rows != plan.geometry.result.rows
                    || compute.columns != plan.geometry.result.columns
                    || self.schedule.reduction_staging().is_some()
                {
                    return Err(OperatorPlanError::InvalidBlocking);
                }
                let [physical_left, physical_right] = orientation.physical_order([left, right]);
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
                let output_column_axis = orientation
                    .column_axis()
                    .resolve(output_padded.0.len())
                    .map_err(|_| OperatorPlanError::InvalidBlocking)?;
                let columns_per_output_shard = output_layout
                    .maximum_axis_extent(output_column_axis)
                    .ok_or(OperatorPlanError::InvalidBlocking)?;
                let right_layout = physical_right
                    .format
                    .layout
                    .resolve(&physical_right.shape)
                    .map_err(|_| OperatorPlanError::InvalidBlocking)?;
                let right_padded = right_layout.padded_shape();
                let right_column_axis = orientation
                    .column_axis()
                    .resolve(right_padded.0.len())
                    .map_err(|_| OperatorPlanError::InvalidBlocking)?;
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
                let physical_left_inner_axis = orientation
                    .column_axis()
                    .resolve(left_padded.0.len())
                    .map_err(|_| OperatorPlanError::InvalidBlocking)?;
                let balanced_output_columns = compute.inner > 1;
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
            (MidOperator::Gelu | MidOperator::Add(_), [crate::ScheduleStep::KernelMap(map)]) => {
                if map.output != crate::ScheduleValue::Output
                    || map.inputs.len() != inputs.len()
                    || !map.inputs.iter().enumerate().all(|(index, (value, _))| {
                        *value == crate::ScheduleValue::Input(index as u16)
                    })
                {
                    return Err(OperatorPlanError::DispatchMismatch);
                }
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
                [
                    crate::ScheduleStep::Attention(AttentionPlan {
                        kernel,
                        blocking:
                            AttentionBlocking::Flash {
                                query_rows,
                                key_rows,
                            },
                        padding,
                    }),
                ],
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
                [
                    crate::ScheduleStep::Attention(AttentionPlan {
                        kernel,
                        blocking:
                            AttentionBlocking::Materialized {
                                query_rows,
                                padded_key_rows,
                            },
                        padding,
                    }),
                ],
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
