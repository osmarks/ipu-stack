//! Mid-level, layout-aware representation.
//!
//! This is the boundary between semantic graph operations and scheduling. It
//! records tensor shapes, storage precision, element order, axis tiling, and
//! memory-class requirements, but deliberately does not assign tile addresses
//! or emit exchange rows. [`lower`] tries a set of legal operator plans,
//! prices them with a [`CostModel`], and inserts explicit precision casts and
//! layout rearrangements at format boundaries.

use crate::graph::{
    AddOptions, AttentionOptions, ComputeGraph, GemmOptions, GraphInputKind, Operation,
    OperationId, OperationKind, Repeat, TensorShape, ValueId,
};
use std::collections::{BTreeMap, BTreeSet};

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

/// A tile-local callable selected by a whole-device operator plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileKernelSpec {
    Gemm {
        multiply: Precision,
        accumulate: AccumulationPrecision,
        mode: GemmKernelMode,
        weights: GemmWeightLoad,
    },
    Gelu,
    Add,
    FlashAttention {
        options: AttentionOptions,
        accumulate: AccumulationPrecision,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmKernelMode {
    Initialize,
    Accumulate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GemmWeightLoad {
    Standard,
    Interleaved,
}

/// Shape-independent recipe which expands into ordered device-wide exchange
/// and tile-kernel phases after concrete shards are known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OperatorDispatch {
    Pointwise {
        kernel: TileKernelSpec,
    },
    BlockedGemm {
        initialize: TileKernelSpec,
        accumulate: TileKernelSpec,
        inner_block: u32,
        output_column_block: u32,
    },
}

/// AMP packing role. Block dimensions are recorded by [`AxisTiling`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AmpOrder {
    Left,
    Right,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ElementOrder {
    RowMajor,
    Amp(AmpOrder),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryClass {
    Ipu21Standard,
    Ipu21Interleaved,
}

/// Maximum per-tile bytes attributed to each address/load class. The classes
/// share physical tile SRAM, so feasibility must check both the individual
/// interleaved-region limit and their combined size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryUsage {
    pub standard: u64,
    pub interleaved: u64,
}

impl MemoryUsage {
    pub const fn total(self) -> u64 {
        self.standard.saturating_add(self.interleaved)
    }

    fn add_class(&mut self, class: MemoryClass, bytes: u64) {
        let target = match class {
            MemoryClass::Ipu21Standard => &mut self.standard,
            MemoryClass::Ipu21Interleaved => &mut self.interleaved,
        };
        *target = target.saturating_add(bytes);
    }

    fn saturating_add(self, other: Self) -> Self {
        Self {
            standard: self.standard.saturating_add(other.standard),
            interleaved: self.interleaved.saturating_add(other.interleaved),
        }
    }

    fn component_max(self, other: Self) -> Self {
        Self {
            standard: self.standard.max(other.standard),
            interleaved: self.interleaved.max(other.interleaved),
        }
    }

    pub fn fits_ipu21(self) -> bool {
        self.interleaved <= u64::from(crate::memory::IPU21_INTERLEAVED_REGION_BYTES)
            && self.total() <= u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES)
    }
}

/// Storage visible at an operator boundary plus phase-local scratch. Peak is
/// the simultaneous requirement used for candidate feasibility.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryEstimate {
    pub live: MemoryUsage,
    pub temporary: MemoryUsage,
    pub peak: MemoryUsage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorAxis {
    FromStart(u16),
    FromEnd(u16),
}

impl TensorAxis {
    pub fn resolve(self, rank: usize) -> Result<usize, LayoutError> {
        match self {
            Self::FromStart(axis) if usize::from(axis) < rank => Ok(usize::from(axis)),
            Self::FromEnd(axis) if axis != 0 && usize::from(axis) <= rank => {
                Ok(rank - usize::from(axis))
            }
            _ => Err(LayoutError::AxisOutOfRange { axis: self, rank }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Padding {
    Reject,
    Zero,
}

/// Blocking and distribution of one logical tensor axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AxisTiling {
    pub axis: TensorAxis,
    /// Number of contiguous partitions distributed across the tile group.
    pub partitions: u16,
    /// Required physical block multiple. One imposes no blocking constraint.
    pub block_size: u32,
    pub padding: Padding,
    /// Optional physical-tile stride for this partition coordinate. When
    /// absent, axes are packed after the replica coordinate and preceding
    /// axes. Explicit strides allow operands of one operator to share a 2-D
    /// tile grid while replicating along different grid dimensions.
    pub tile_stride: Option<u16>,
}

impl AxisTiling {
    pub const fn new(axis: TensorAxis, partitions: u16, block_size: u32, padding: Padding) -> Self {
        Self {
            axis,
            partitions,
            block_size,
            padding,
            tile_stride: None,
        }
    }

    pub const fn with_tile_stride(mut self, tile_stride: u16) -> Self {
        self.tile_stride = Some(tile_stride);
        self
    }
}

/// Logical tile group and the tensor axes distributed or blocked within it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorTiling {
    pub tile_count: u16,
    pub replicas: u16,
    pub axes: Vec<AxisTiling>,
}

impl TensorTiling {
    pub fn replicated(tile_count: u16) -> Self {
        Self {
            tile_count,
            replicas: tile_count,
            axes: Vec::new(),
        }
    }

    pub fn sharded(axis: TensorAxis, tile_count: u16) -> Self {
        Self {
            tile_count,
            replicas: 1,
            axes: vec![AxisTiling::new(axis, tile_count, 1, Padding::Reject)],
        }
    }

    pub(crate) fn axis_strides(&self) -> Result<Vec<u32>, LayoutError> {
        let mut packed_stride = u32::from(self.replicas);
        self.axes
            .iter()
            .map(|axis| {
                let stride = axis.tile_stride.map_or(packed_stride, u32::from);
                packed_stride = packed_stride
                    .checked_mul(u32::from(axis.partitions))
                    .ok_or(LayoutError::TileCountOverflow)?;
                if stride == 0 {
                    return Err(LayoutError::EmptyAxisTiling);
                }
                Ok(stride)
            })
            .collect()
    }
}

/// Layout decisions which constrain operators and exchange generation without
/// assigning physical tile identities or SRAM addresses.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Layout {
    pub order: ElementOrder,
    pub tiling: TensorTiling,
    pub memory_class: MemoryClass,
}

impl Layout {
    pub fn row_major(tiling: TensorTiling) -> Self {
        Self {
            order: ElementOrder::RowMajor,
            tiling,
            memory_class: MemoryClass::Ipu21Standard,
        }
    }

    pub fn row_sharded(tile_count: u16) -> Self {
        Self::row_major(TensorTiling::sharded(TensorAxis::FromEnd(2), tile_count))
    }

    pub fn head_sharded(tile_count: u16) -> Self {
        Self::row_major(TensorTiling::sharded(TensorAxis::FromEnd(3), tile_count))
    }

    pub fn amp_left(inner: u16, tile_count: u16) -> Self {
        Self {
            order: ElementOrder::Amp(AmpOrder::Left),
            tiling: TensorTiling {
                tile_count,
                replicas: 1,
                axes: vec![
                    AxisTiling::new(TensorAxis::FromEnd(2), tile_count, 1, Padding::Reject),
                    AxisTiling::new(TensorAxis::FromEnd(1), 1, u32::from(inner), Padding::Zero),
                ],
            },
            memory_class: MemoryClass::Ipu21Standard,
        }
    }

    pub fn amp_right(inner: u16, tile_count: u16) -> Self {
        Self {
            order: ElementOrder::Amp(AmpOrder::Right),
            tiling: TensorTiling {
                tile_count,
                replicas: 1,
                axes: vec![
                    AxisTiling::new(TensorAxis::FromEnd(2), 1, u32::from(inner), Padding::Zero),
                    AxisTiling::new(TensorAxis::FromEnd(1), tile_count, 64, Padding::Zero),
                ],
            },
            memory_class: MemoryClass::Ipu21Standard,
        }
    }

    pub fn amp_output(tile_count: u16) -> Self {
        Self {
            order: ElementOrder::Amp(AmpOrder::Output),
            tiling: TensorTiling {
                tile_count,
                replicas: 1,
                axes: vec![
                    AxisTiling::new(TensorAxis::FromEnd(2), tile_count, 1, Padding::Reject),
                    AxisTiling::new(TensorAxis::FromEnd(1), 1, 64, Padding::Zero),
                ],
            },
            memory_class: MemoryClass::Ipu21Interleaved,
        }
    }

    /// AMP left operand on a row-by-column tile grid. The row shard is
    /// replicated across column groups so it is local to every output shard.
    pub fn amp_left_grid(
        inner: u16,
        tile_count: u16,
        row_partitions: u16,
        column_partitions: u16,
    ) -> Self {
        if column_partitions == 1 && row_partitions == tile_count {
            return Self::amp_left(inner, tile_count);
        }
        Self {
            order: ElementOrder::Amp(AmpOrder::Left),
            tiling: TensorTiling {
                tile_count,
                replicas: column_partitions,
                axes: vec![
                    AxisTiling::new(TensorAxis::FromEnd(2), row_partitions, 1, Padding::Reject),
                    AxisTiling::new(TensorAxis::FromEnd(1), 1, u32::from(inner), Padding::Zero),
                ],
            },
            memory_class: MemoryClass::Ipu21Standard,
        }
    }

    /// AMP right operand on a row-by-column tile grid. Each column shard is
    /// replicated across row groups so it is local to every output shard.
    pub fn amp_right_grid(
        inner: u16,
        tile_count: u16,
        row_partitions: u16,
        column_partitions: u16,
    ) -> Self {
        if row_partitions == 1 && column_partitions == tile_count {
            return Self::amp_right(inner, tile_count);
        }
        Self {
            order: ElementOrder::Amp(AmpOrder::Right),
            tiling: TensorTiling {
                tile_count,
                replicas: row_partitions,
                axes: vec![
                    AxisTiling::new(TensorAxis::FromEnd(2), 1, u32::from(inner), Padding::Zero),
                    AxisTiling::new(TensorAxis::FromEnd(1), column_partitions, 64, Padding::Zero)
                        .with_tile_stride(1),
                ],
            },
            memory_class: MemoryClass::Ipu21Standard,
        }
    }

    /// AMP right operand distributed across both K and output columns. A
    /// blocked GEMM streams each K panel to the other output-row groups that
    /// share its column coordinate instead of retaining a full replica.
    pub fn amp_right_streamed(
        inner: u16,
        tile_count: u16,
        row_partitions: u16,
        column_partitions: u16,
    ) -> Self {
        Self {
            order: ElementOrder::Amp(AmpOrder::Right),
            tiling: TensorTiling {
                tile_count,
                replicas: 1,
                axes: vec![
                    AxisTiling::new(TensorAxis::FromEnd(1), column_partitions, 64, Padding::Zero),
                    AxisTiling::new(
                        TensorAxis::FromEnd(2),
                        row_partitions,
                        u32::from(inner),
                        Padding::Zero,
                    ),
                ],
            },
            memory_class: MemoryClass::Ipu21Standard,
        }
    }

    /// AMP output distributed over both matrix axes on one tile grid.
    pub fn amp_output_grid(tile_count: u16, row_partitions: u16, column_partitions: u16) -> Self {
        if column_partitions == 1 && row_partitions == tile_count {
            return Self::amp_output(tile_count);
        }
        Self {
            order: ElementOrder::Amp(AmpOrder::Output),
            tiling: TensorTiling {
                tile_count,
                replicas: 1,
                axes: vec![
                    AxisTiling::new(TensorAxis::FromEnd(1), column_partitions, 64, Padding::Zero),
                    AxisTiling::new(TensorAxis::FromEnd(2), row_partitions, 1, Padding::Reject),
                ],
            },
            memory_class: MemoryClass::Ipu21Interleaved,
        }
    }

    /// Returns the physical extents after applying declared zero padding.
    pub fn padded_shape(&self, shape: &TensorShape) -> Result<TensorShape, LayoutError> {
        if self.tiling.tile_count == 0 || self.tiling.replicas == 0 {
            return Err(LayoutError::EmptyTileGroup);
        }
        let mut used_tiles = u32::from(self.tiling.replicas);
        let mut dimensions = shape.0.clone();
        let mut used_axes = Vec::with_capacity(self.tiling.axes.len());
        for tiling in &self.tiling.axes {
            if tiling.partitions == 0 || tiling.block_size == 0 {
                return Err(LayoutError::EmptyAxisTiling);
            }
            used_tiles = used_tiles
                .checked_mul(u32::from(tiling.partitions))
                .ok_or(LayoutError::TileCountOverflow)?;
            let axis = tiling.axis.resolve(dimensions.len())?;
            if used_axes.contains(&axis) {
                return Err(LayoutError::DuplicateAxis(axis));
            }
            used_axes.push(axis);
            let extent = dimensions[axis];
            let remainder = extent % tiling.block_size;
            if remainder != 0 {
                match tiling.padding {
                    Padding::Reject => {
                        return Err(LayoutError::IndivisibleAxis {
                            axis,
                            extent,
                            block_size: tiling.block_size,
                        });
                    }
                    Padding::Zero => {
                        dimensions[axis] = extent
                            .checked_add(tiling.block_size - remainder)
                            .ok_or(LayoutError::ExtentOverflow(axis))?;
                    }
                }
            }
        }
        if used_tiles != u32::from(self.tiling.tile_count) {
            return Err(LayoutError::TileCountMismatch {
                declared: self.tiling.tile_count,
                implied: used_tiles,
            });
        }
        let strides = self.tiling.axis_strides()?;
        let mut coordinates = BTreeMap::<Vec<u16>, u16>::new();
        for tile in 0..self.tiling.tile_count {
            let coordinate = self
                .tiling
                .axes
                .iter()
                .zip(&strides)
                .map(|(axis, stride)| {
                    ((u32::from(tile) / stride) % u32::from(axis.partitions)) as u16
                })
                .collect::<Vec<_>>();
            *coordinates.entry(coordinate).or_default() += 1;
        }
        if coordinates.len()
            != self
                .tiling
                .axes
                .iter()
                .map(|axis| usize::from(axis.partitions))
                .product::<usize>()
            || coordinates
                .values()
                .any(|copies| *copies != self.tiling.replicas)
        {
            return Err(LayoutError::InvalidTileMapping);
        }
        Ok(TensorShape(dimensions))
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum LayoutError {
    #[error("layout has an empty tile group")]
    EmptyTileGroup,
    #[error("axis tiling must have nonzero partitions and block size")]
    EmptyAxisTiling,
    #[error("tile count calculation overflowed")]
    TileCountOverflow,
    #[error("axis {axis:?} is outside rank {rank}")]
    AxisOutOfRange { axis: TensorAxis, rank: usize },
    #[error("axis {0} is tiled more than once")]
    DuplicateAxis(usize),
    #[error("axis {axis} extent {extent} is not divisible by block size {block_size}")]
    IndivisibleAxis {
        axis: usize,
        extent: u32,
        block_size: u32,
    },
    #[error("padded extent for axis {0} overflowed")]
    ExtentOverflow(usize),
    #[error("layout declares {declared} tiles but its tiling implies {implied}")]
    TileCountMismatch { declared: u16, implied: u32 },
    #[error("tile strides do not form the declared partition and replica mapping")]
    InvalidTileMapping,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorFormat {
    pub precision: Precision,
    pub layout: Layout,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorType {
    pub shape: TensorShape,
    pub format: TensorFormat,
}

impl TensorType {
    pub fn new(shape: impl IntoIterator<Item = u32>, precision: Precision, layout: Layout) -> Self {
        Self {
            shape: TensorShape::new(shape),
            format: TensorFormat { precision, layout },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperandRequirement {
    pub format: TensorFormat,
    pub alignment: u32,
    /// Bytes the kernel may access beyond the logical tensor payload.
    pub access_tail_bytes: u32,
}

impl OperandRequirement {
    pub fn new(format: TensorFormat, alignment: u32) -> Self {
        Self {
            format,
            alignment,
            access_tail_bytes: 0,
        }
    }

    pub fn with_access_tail(mut self, bytes: u32) -> Self {
        self.access_tail_bytes = bytes;
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryRelation {
    /// Operand ranges must not occupy the same effective tile-memory element.
    DistinctElements(Vec<MemoryOperand>),
}

/// Complete formats and placement requirements of one whole-device operator plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatorCandidate {
    pub operator: MidOperator,
    pub dispatch: OperatorDispatch,
    pub inputs: Vec<OperandRequirement>,
    pub output: OperandRequirement,
    pub output_aliasing: OutputAliasing,
    pub memory_relations: Vec<MemoryRelation>,
}

impl OperatorCandidate {
    pub fn new(
        operator: MidOperator,
        inputs: impl IntoIterator<Item = OperandRequirement>,
        output: OperandRequirement,
    ) -> Self {
        Self {
            operator,
            dispatch: default_dispatch(operator),
            inputs: inputs.into_iter().collect(),
            output,
            output_aliasing: OutputAliasing::Fresh,
            memory_relations: Vec::new(),
        }
    }

    pub fn with_dispatch(mut self, dispatch: OperatorDispatch) -> Self {
        self.dispatch = dispatch;
        self
    }

    pub fn with_output_aliasing(mut self, aliasing: OutputAliasing) -> Self {
        self.output_aliasing = aliasing;
        self
    }

    pub fn with_memory_relation(mut self, relation: MemoryRelation) -> Self {
        self.memory_relations.push(relation);
        self
    }

    fn supports(&self, inputs: &[TensorType], output: &TensorShape) -> bool {
        if self.inputs.len() != inputs.len()
            || !valid_requirement(&self.output, output)
            || !self
                .inputs
                .iter()
                .zip(inputs)
                .all(|(requirement, input)| valid_requirement(requirement, &input.shape))
        {
            return false;
        }
        let alias_valid = match &self.output_aliasing {
            OutputAliasing::Fresh => true,
            OutputAliasing::MayAliasInputs(indices) => {
                !indices.is_empty()
                    && indices.iter().any(|index| {
                        alias_compatible(
                            usize::from(*index),
                            &self.inputs,
                            inputs,
                            &self.output,
                            output,
                        )
                    })
            }
            OutputAliasing::MustAliasInput(index) => alias_compatible(
                usize::from(*index),
                &self.inputs,
                inputs,
                &self.output,
                output,
            ),
        };
        let requirements_valid = alias_valid
            && self.memory_relations.iter().all(|relation| match relation {
                MemoryRelation::DistinctElements(operands) => {
                    operands.len() >= 2
                        && operands
                            .iter()
                            .all(|operand| valid_memory_operand(*operand, inputs.len()))
                        && operands.iter().enumerate().all(|(index, operand)| {
                            !operands[..index].iter().any(|previous| previous == operand)
                        })
                }
            });
        if !requirements_valid {
            return false;
        }
        let planned_inputs = inputs
            .iter()
            .zip(&self.inputs)
            .map(|(input, requirement)| TensorType {
                shape: input.shape.clone(),
                format: requirement.format.clone(),
            })
            .collect::<Vec<_>>();
        let planned_output = TensorType {
            shape: output.clone(),
            format: self.output.format.clone(),
        };
        OperatorPlan {
            operator: self.operator,
            dispatch: self.dispatch.clone(),
            requirements: OperatorRequirements {
                inputs: self.inputs.clone(),
                output: self.output.clone(),
                output_aliasing: self.output_aliasing.clone(),
                memory_relations: self.memory_relations.clone(),
            },
        }
        .validate(&planned_inputs, &planned_output)
        .is_ok()
    }
}

fn default_dispatch(operator: MidOperator) -> OperatorDispatch {
    match operator {
        MidOperator::Gemm {
            options: _,
            multiply,
            accumulate,
        } => OperatorDispatch::BlockedGemm {
            initialize: TileKernelSpec::Gemm {
                multiply,
                accumulate,
                mode: GemmKernelMode::Initialize,
                weights: GemmWeightLoad::Standard,
            },
            accumulate: TileKernelSpec::Gemm {
                multiply,
                accumulate,
                mode: GemmKernelMode::Accumulate,
                weights: GemmWeightLoad::Standard,
            },
            inner_block: 64,
            output_column_block: 64,
        },
        MidOperator::Gelu => OperatorDispatch::Pointwise {
            kernel: TileKernelSpec::Gelu,
        },
        MidOperator::Add(_) => OperatorDispatch::Pointwise {
            kernel: TileKernelSpec::Add,
        },
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
    requirement.alignment.is_power_of_two() && requirement.format.layout.padded_shape(shape).is_ok()
}

fn valid_memory_operand(operand: MemoryOperand, input_count: usize) -> bool {
    match operand {
        MemoryOperand::Output => true,
        MemoryOperand::Input(index) => usize::from(index) < input_count,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineConfig {
    pub target: HardwareTarget,
    pub tile_count: u16,
    pub inputs: BTreeMap<ValueId, TensorFormat>,
    /// Graph-boundary tensors whose layout may be selected by their first
    /// consumer. Precision remains fixed, while packaging exposes the chosen
    /// physical layout directly through the host binding.
    pub automatic_inputs: BTreeMap<ValueId, Precision>,
    /// Signatures available independently to each operation. Earlier entries
    /// of the appropriate operation kind win when costs are equal.
    pub operator_candidates: Vec<OperatorCandidate>,
    pub scheduling: SchedulingPolicy,
    pub profiling: ProfilingConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardwareTarget {
    Ipu21,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulingPolicy {
    OperatorPlans,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProfilingConfig {
    pub enabled: bool,
}

impl PipelineConfig {
    pub fn new(tile_count: u16) -> Self {
        Self {
            target: HardwareTarget::Ipu21,
            tile_count,
            inputs: BTreeMap::new(),
            automatic_inputs: BTreeMap::new(),
            operator_candidates: default_operator_candidates(tile_count),
            scheduling: SchedulingPolicy::OperatorPlans,
            profiling: ProfilingConfig::default(),
        }
    }

    pub fn with_input(mut self, value: ValueId, format: TensorFormat) -> Self {
        self.inputs.insert(value, format);
        self.automatic_inputs.remove(&value);
        self
    }

    pub fn with_automatic_input(mut self, value: ValueId, precision: Precision) -> Self {
        self.inputs.remove(&value);
        self.automatic_inputs.insert(value, precision);
        self
    }
}

fn default_operator_candidates(tile_count: u16) -> Vec<OperatorCandidate> {
    let amp_output_f16 = TensorFormat {
        precision: Precision::F16,
        layout: Layout::amp_output(tile_count),
    };
    let amp_left_f16 = TensorFormat {
        precision: Precision::F16,
        layout: Layout::amp_left(64, tile_count),
    };
    let rows_f16 = TensorFormat {
        precision: Precision::F16,
        layout: Layout::row_sharded(tile_count),
    };
    let rows_f32 = TensorFormat {
        precision: Precision::F32,
        layout: Layout::row_sharded(tile_count),
    };
    let heads_f16 = TensorFormat {
        precision: Precision::F16,
        layout: Layout::head_sharded(tile_count),
    };
    let heads_f32 = TensorFormat {
        precision: Precision::F32,
        layout: Layout::head_sharded(tile_count),
    };
    let mut candidates = (1..=tile_count)
        .rev()
        .filter(|columns| tile_count.is_multiple_of(*columns))
        .flat_map(|columns| {
            let rows = tile_count / columns;
            [
                amp_grid_gemm_operator_candidate(Precision::F16, 64, 16, tile_count, rows, columns),
                amp_grid_gemm_operator_candidate(Precision::F32, 64, 32, tile_count, rows, columns),
                amp_streamed_gemm_operator_candidate(
                    Precision::F16,
                    64,
                    16,
                    tile_count,
                    rows,
                    columns,
                ),
                amp_streamed_gemm_operator_candidate(
                    Precision::F32,
                    64,
                    32,
                    tile_count,
                    rows,
                    columns,
                ),
            ]
        })
        .collect::<Vec<_>>();
    candidates.extend([
        amp_gemm_operator_candidate(Precision::F16, 64, 16, tile_count),
        amp_gemm_operator_candidate(Precision::F32, 64, 32, tile_count),
        OperatorCandidate::new(
            MidOperator::Gelu,
            [OperandRequirement::new(amp_output_f16, 8)],
            OperandRequirement::new(amp_left_f16, 8),
        ),
        pointwise_operator_candidate(MidOperator::Gelu, [rows_f16.clone()], rows_f16.clone()),
        pointwise_operator_candidate(MidOperator::Gelu, [rows_f32.clone()], rows_f32.clone()),
        pointwise_operator_candidate(
            MidOperator::Add(AddOptions::default()),
            [rows_f16.clone(), rows_f16.clone()],
            rows_f16,
        ),
        pointwise_operator_candidate(
            MidOperator::Add(AddOptions::default()),
            [rows_f32.clone(), rows_f32.clone()],
            rows_f32,
        ),
        pointwise_operator_candidate(
            MidOperator::FlashAttention {
                options: AttentionOptions::default(),
                accumulate: AccumulationPrecision::F32,
            },
            [heads_f16.clone(), heads_f16.clone(), heads_f16.clone()],
            heads_f16,
        ),
        pointwise_operator_candidate(
            MidOperator::FlashAttention {
                options: AttentionOptions::default(),
                accumulate: AccumulationPrecision::F32,
            },
            [heads_f32.clone(), heads_f32.clone(), heads_f32.clone()],
            heads_f32,
        ),
    ]);
    candidates
}

fn pointwise_operator_candidate(
    operator: MidOperator,
    inputs: impl IntoIterator<Item = TensorFormat>,
    output: TensorFormat,
) -> OperatorCandidate {
    let candidate = OperatorCandidate::new(
        operator,
        inputs
            .into_iter()
            .map(|format| OperandRequirement::new(format, 8)),
        OperandRequirement::new(output, 8),
    );
    match operator {
        MidOperator::Gelu => {
            candidate.with_output_aliasing(OutputAliasing::MayAliasInputs(vec![0]))
        }
        MidOperator::Add(_) => {
            candidate.with_output_aliasing(OutputAliasing::MayAliasInputs(vec![0, 1]))
        }
        _ => candidate,
    }
}

fn amp_gemm_operator_candidate(
    precision: Precision,
    inner: u16,
    left_tail: u32,
    tile_count: u16,
) -> OperatorCandidate {
    OperatorCandidate::new(
        MidOperator::Gemm {
            options: GemmOptions::default(),
            multiply: precision,
            accumulate: AccumulationPrecision::F32,
        },
        [
            OperandRequirement::new(
                TensorFormat {
                    precision,
                    layout: Layout::amp_left(inner, tile_count),
                },
                32,
            )
            .with_access_tail(left_tail),
            OperandRequirement::new(
                TensorFormat {
                    precision,
                    layout: Layout::amp_right(inner, tile_count),
                },
                32,
            ),
        ],
        OperandRequirement::new(
            TensorFormat {
                precision,
                layout: Layout::amp_output(tile_count),
            },
            32,
        ),
    )
    .with_memory_relation(MemoryRelation::DistinctElements(vec![
        MemoryOperand::Output,
        MemoryOperand::Input(0),
    ]))
}

fn amp_grid_gemm_operator_candidate(
    precision: Precision,
    inner: u16,
    left_tail: u32,
    tile_count: u16,
    row_partitions: u16,
    column_partitions: u16,
) -> OperatorCandidate {
    OperatorCandidate::new(
        MidOperator::Gemm {
            options: GemmOptions::default(),
            multiply: precision,
            accumulate: AccumulationPrecision::F32,
        },
        [
            OperandRequirement::new(
                TensorFormat {
                    precision,
                    layout: Layout::amp_left_grid(
                        inner,
                        tile_count,
                        row_partitions,
                        column_partitions,
                    ),
                },
                32,
            )
            .with_access_tail(left_tail),
            OperandRequirement::new(
                TensorFormat {
                    precision,
                    layout: Layout::amp_right_grid(
                        inner,
                        tile_count,
                        row_partitions,
                        column_partitions,
                    ),
                },
                32,
            ),
        ],
        OperandRequirement::new(
            TensorFormat {
                precision,
                layout: Layout::amp_output_grid(tile_count, row_partitions, column_partitions),
            },
            32,
        ),
    )
    .with_memory_relation(MemoryRelation::DistinctElements(vec![
        MemoryOperand::Output,
        MemoryOperand::Input(0),
    ]))
}

fn amp_streamed_gemm_operator_candidate(
    precision: Precision,
    inner: u16,
    left_tail: u32,
    tile_count: u16,
    row_partitions: u16,
    column_partitions: u16,
) -> OperatorCandidate {
    OperatorCandidate::new(
        MidOperator::Gemm {
            options: GemmOptions::default(),
            multiply: precision,
            accumulate: AccumulationPrecision::F32,
        },
        [
            OperandRequirement::new(
                TensorFormat {
                    precision,
                    layout: Layout::amp_left_grid(
                        inner,
                        tile_count,
                        row_partitions,
                        column_partitions,
                    ),
                },
                32,
            )
            .with_access_tail(left_tail),
            OperandRequirement::new(
                TensorFormat {
                    precision,
                    layout: Layout::amp_right_streamed(
                        inner,
                        tile_count,
                        row_partitions,
                        column_partitions,
                    ),
                },
                32,
            ),
        ],
        OperandRequirement::new(
            TensorFormat {
                precision,
                layout: Layout::amp_output_grid(tile_count, row_partitions, column_partitions),
            },
            32,
        ),
    )
    .with_memory_relation(MemoryRelation::DistinctElements(vec![
        MemoryOperand::Output,
        MemoryOperand::Input(0),
    ]))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MidValueId(u32);

impl MidValueId {
    pub const fn index(self) -> u32 {
        self.0
    }

    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidValue {
    pub id: MidValueId,
    pub tensor_type: TensorType,
    /// Semantic value represented by this value; conversions retain the same
    /// origin. Region arguments also refer to their high-level argument ID.
    pub origin: ValueId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MidOperationKind {
    Operator(MidOperator),
    CastPrecision { from: Precision, to: Precision },
    Rearrange { from: Layout, to: Layout },
    Repeat(MidRepeat),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidOperation {
    pub source: Option<OperationId>,
    pub inputs: Vec<MidValueId>,
    pub results: Vec<MidValueId>,
    pub kind: MidOperationKind,
    pub operator_plan: Option<OperatorPlan>,
    pub conversion_plan: Option<ConversionPlan>,
    pub estimated_cost: u64,
    pub memory: MemoryEstimate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatorRequirements {
    pub inputs: Vec<OperandRequirement>,
    pub output: OperandRequirement,
    pub output_aliasing: OutputAliasing,
    pub memory_relations: Vec<MemoryRelation>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatorPlan {
    pub operator: MidOperator,
    pub dispatch: OperatorDispatch,
    pub requirements: OperatorRequirements,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConversionDispatch {
    Local,
    Intersections,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversionPlan {
    pub kernel: TileKernelSpec,
    pub input: OperandRequirement,
    pub output: OperandRequirement,
    pub dispatch: ConversionDispatch,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum OperatorPlanError {
    #[error("operator plan operand arity does not match its requirements")]
    OperandArity,
    #[error("operator plan dispatch does not match the selected operator")]
    DispatchMismatch,
    #[error("operator plan uses zero or incompatible block dimensions")]
    InvalidBlocking,
    #[error("operator plan requires corresponding input and output tile groups")]
    IncompatibleTileGroups,
    #[error("blocked GEMM currently requires non-transposed AMP left/right/output formats")]
    UnsupportedGemmLayout,
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
        let output_tiles = output.format.layout.tiling.tile_count;
        if inputs
            .iter()
            .any(|input| input.format.layout.tiling.tile_count != output_tiles)
        {
            return Err(OperatorPlanError::IncompatibleTileGroups);
        }
        match (&self.operator, &self.dispatch) {
            (
                MidOperator::Gemm { options, .. },
                OperatorDispatch::BlockedGemm {
                    initialize,
                    accumulate,
                    inner_block,
                    output_column_block,
                },
            ) => {
                let [left, right] = inputs else {
                    return Err(OperatorPlanError::OperandArity);
                };
                if options.transpose_left
                    || options.transpose_right
                    || !matches!(left.format.layout.order, ElementOrder::Amp(AmpOrder::Left))
                    || !matches!(
                        right.format.layout.order,
                        ElementOrder::Amp(AmpOrder::Right)
                    )
                    || !matches!(
                        output.format.layout.order,
                        ElementOrder::Amp(AmpOrder::Output)
                    )
                {
                    return Err(OperatorPlanError::UnsupportedGemmLayout);
                }
                let (
                    TileKernelSpec::Gemm {
                        multiply: init_multiply,
                        accumulate: init_accumulate,
                        mode: GemmKernelMode::Initialize,
                        ..
                    },
                    TileKernelSpec::Gemm {
                        multiply: next_multiply,
                        accumulate: next_accumulate,
                        mode: GemmKernelMode::Accumulate,
                        ..
                    },
                    MidOperator::Gemm {
                        multiply,
                        accumulate,
                        ..
                    },
                ) = (initialize, accumulate, &self.operator)
                else {
                    return Err(OperatorPlanError::DispatchMismatch);
                };
                if init_multiply != multiply
                    || next_multiply != multiply
                    || init_accumulate != accumulate
                    || next_accumulate != accumulate
                {
                    return Err(OperatorPlanError::DispatchMismatch);
                }
                if *inner_block == 0
                    || *output_column_block == 0
                    || left.shape.0.len() < 2
                    || output.shape.0.len() < 2
                {
                    return Err(OperatorPlanError::InvalidBlocking);
                }
                let left_padded = left
                    .format
                    .layout
                    .padded_shape(&left.shape)
                    .map_err(|_| OperatorPlanError::InvalidBlocking)?;
                let output_padded = output
                    .format
                    .layout
                    .padded_shape(&output.shape)
                    .map_err(|_| OperatorPlanError::InvalidBlocking)?;
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
                    && (!layout_shards_are_even(left)
                        || !layout_shards_are_even(right)
                        || !layout_shards_are_even(output))
                {
                    return Err(OperatorPlanError::InvalidBlocking);
                }
                if !left_padded.0.last().unwrap().is_multiple_of(*inner_block)
                    || !output_padded
                        .0
                        .last()
                        .unwrap()
                        .is_multiple_of(*output_column_block)
                {
                    return Err(OperatorPlanError::InvalidBlocking);
                }
                Ok(())
            }
            (
                MidOperator::Gelu,
                OperatorDispatch::Pointwise {
                    kernel: TileKernelSpec::Gelu,
                },
            )
            | (
                MidOperator::Add(_),
                OperatorDispatch::Pointwise {
                    kernel: TileKernelSpec::Add,
                },
            ) => Ok(()),
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
                },
            ) if options == kernel_options && accumulate == kernel_accumulate => Ok(()),
            _ => Err(OperatorPlanError::DispatchMismatch),
        }
    }
}

fn layout_shards_are_even(tensor: &TensorType) -> bool {
    let Ok(padded) = tensor.format.layout.padded_shape(&tensor.shape) else {
        return false;
    };
    tensor.format.layout.tiling.axes.iter().all(|axis| {
        axis.axis.resolve(padded.0.len()).is_ok_and(|index| {
            let blocks = padded.0[index] / axis.block_size;
            blocks >= u32::from(axis.partitions)
                && blocks.is_multiple_of(u32::from(axis.partitions))
        })
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidRegion {
    pub arguments: Vec<MidValueId>,
    pub operations: Vec<MidOperation>,
    pub yields: Vec<MidValueId>,
    pub estimated_cost: u64,
    pub peak_memory: MemoryUsage,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidRepeat {
    pub count: u32,
    pub carried_inputs: usize,
    pub invariant_inputs: usize,
    /// One normalized value list for each iterated body argument. Keeping the
    /// lists on the structured operation avoids unrolling layer parameters.
    pub iterated_inputs: Vec<Vec<MidValueId>>,
    pub body: MidRegion,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidInput {
    pub name: String,
    pub kind: GraphInputKind,
    pub value: MidValueId,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MidGraph {
    pub inputs: Vec<MidInput>,
    pub values: Vec<MidValue>,
    pub operations: Vec<MidOperation>,
    pub outputs: Vec<MidValueId>,
    pub estimated_cost: u64,
    pub peak_memory: MemoryUsage,
}

/// Deliberately small and replaceable cost interface. Units are arbitrary but
/// must be comparable within one lowering run.
pub trait CostModel {
    fn operator_cost(
        &self,
        operator: MidOperator,
        dispatch: &OperatorDispatch,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> u64;
    fn cast_cost(&self, input: &TensorType, to: Precision) -> u64;
    fn rearrange_cost(
        &self,
        shape: &TensorShape,
        precision: Precision,
        from: &Layout,
        to: &Layout,
    ) -> u64;
}

/// IPU21 analytical cycle model. Its structure follows the SDK planners:
/// serial compute, exchange, and rearrangement phases are summed, while
/// competing arithmetic and operand-feed limits within a kernel are maxed.
/// Constants remain deliberately explicit so hardware measurements can refine
/// them without changing the planning interface.
#[derive(Clone, Copy, Debug, Default)]
pub struct ToyCostModel;

impl CostModel for ToyCostModel {
    fn operator_cost(
        &self,
        operator: MidOperator,
        dispatch: &OperatorDispatch,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> u64 {
        let elements = physical_elements(&output.shape, &output.format.layout);
        match operator {
            MidOperator::Gemm { multiply, .. } => {
                let left_shape = inputs[0]
                    .format
                    .layout
                    .padded_shape(&inputs[0].shape)
                    .unwrap_or_else(|_| inputs[0].shape.clone());
                let k = left_shape.0.last().copied().unwrap_or(1) as u64;
                let flops_per_cycle: u64 = match multiply {
                    Precision::F8F143 { .. } => 256,
                    Precision::F16 => 128,
                    Precision::F32 => 32,
                };
                let output_elements_per_tile =
                    maximum_shard_bytes(output).div_ceil(output.format.precision.bytes());
                let output_columns_per_tile =
                    maximum_axis_shard_extent(output, output.shape.0.len().saturating_sub(1));
                let arithmetic = output_elements_per_tile
                    .saturating_mul(2)
                    .saturating_mul(k)
                    .div_ceil(flops_per_cycle);
                let right = inputs.get(1);
                let right_bytes_consumed = right.map_or(u64::MAX, |right| {
                    output_columns_per_tile
                        .saturating_mul(k)
                        .saturating_mul(right.format.precision.bytes())
                });
                let resident_interleaved_weights = right.is_some_and(|right| {
                    right.format.layout.memory_class == MemoryClass::Ipu21Interleaved
                });
                let staged_weights = right.is_some_and(|right| {
                    gemm_requires_staging(dispatch, right, output)
                        && right.format.precision == Precision::F16
                });
                let weight_feed = right_bytes_consumed.div_ceil(
                    if resident_interleaved_weights || staged_weights {
                        IPU21_INTERLEAVED_LOAD_BYTES_PER_CYCLE
                    } else {
                        IPU21_STANDARD_LOAD_BYTES_PER_CYCLE
                    },
                );
                let packing = if staged_weights {
                    right_bytes_consumed
                        .saturating_mul(2)
                        .div_ceil(IPU21_LOCAL_COPY_BYTES_PER_CYCLE)
                } else {
                    0
                };
                let remote_bytes = gemm_remote_bytes_per_tile(inputs, output);
                let exchange = remote_bytes.div_ceil(gemm_exchange_bytes_per_cycle(inputs));
                let calls = match dispatch {
                    OperatorDispatch::BlockedGemm {
                        inner_block,
                        output_column_block,
                        ..
                    } => k
                        .div_ceil(u64::from(*inner_block))
                        .saturating_mul(output_columns_per_tile)
                        .div_ceil(u64::from(*output_column_block))
                        .saturating_mul(IPU21_GEMM_CALL_OVERHEAD_CYCLES),
                    OperatorDispatch::Pointwise { .. } => 0,
                };
                let memory = operator_memory_estimate(dispatch, inputs, output);
                let capacity_penalty = if !memory.peak.fits_ipu21() {
                    u64::MAX / 8
                } else {
                    0
                };
                arithmetic
                    .max(weight_feed)
                    .saturating_add(packing)
                    .saturating_add(exchange)
                    .saturating_add(calls)
                    .saturating_add(capacity_penalty)
            }
            MidOperator::FlashAttention { .. } => elements.saturating_mul(8).div_ceil(32),
            MidOperator::Gelu => elements.saturating_mul(6).div_ceil(16),
            MidOperator::Add(_) => elements.div_ceil(16),
        }
    }

    fn cast_cost(&self, input: &TensorType, to: Precision) -> u64 {
        let input_bytes = maximum_shard_bytes(input);
        let output_bytes = input_bytes
            .div_ceil(input.format.precision.bytes())
            .saturating_mul(to.bytes());
        input_bytes
            .saturating_add(output_bytes)
            .div_ceil(IPU21_LOCAL_COPY_BYTES_PER_CYCLE)
    }

    fn rearrange_cost(
        &self,
        shape: &TensorShape,
        precision: Precision,
        from: &Layout,
        to: &Layout,
    ) -> u64 {
        let input = TensorType::new(shape.0.iter().copied(), precision, from.clone());
        let output = TensorType::new(shape.0.iter().copied(), precision, to.clone());
        let local_bytes = maximum_shard_bytes(&input)
            .max(maximum_shard_bytes(&output))
            .saturating_mul(2);
        let local_cycles = local_bytes.div_ceil(IPU21_LOCAL_COPY_BYTES_PER_CYCLE);
        let exchange_cycles = if from.tiling == to.tiling {
            0
        } else {
            maximum_shard_bytes(&output).div_ceil(IPU21_EXCHANGE_BYTES_PER_CYCLE)
        };
        local_cycles.saturating_add(exchange_cycles)
    }
}

const IPU21_EXCHANGE_BYTES_PER_CYCLE: u64 = 8;
const IPU21_STANDARD_LOAD_BYTES_PER_CYCLE: u64 = 8;
const IPU21_INTERLEAVED_LOAD_BYTES_PER_CYCLE: u64 = 16;
const IPU21_LOCAL_COPY_BYTES_PER_CYCLE: u64 = 8;
const IPU21_GEMM_CALL_OVERHEAD_CYCLES: u64 = 48;

fn physical_elements(shape: &TensorShape, layout: &Layout) -> u64 {
    layout
        .padded_shape(shape)
        .map_or_else(|_| shape.elements(), |shape| shape.elements())
        .saturating_mul(u64::from(layout.tiling.replicas))
}

fn maximum_shard_bytes(tensor: &TensorType) -> u64 {
    let Ok(padded) = tensor.format.layout.padded_shape(&tensor.shape) else {
        return u64::MAX;
    };
    padded
        .0
        .iter()
        .enumerate()
        .map(|(index, &extent)| {
            tensor
                .format
                .layout
                .tiling
                .axes
                .iter()
                .find(|axis| axis.axis.resolve(padded.0.len()) == Ok(index))
                .map_or(u64::from(extent), |axis| {
                    let blocks = extent / axis.block_size;
                    u64::from(blocks.div_ceil(u32::from(axis.partitions)) * axis.block_size)
                })
        })
        .product::<u64>()
        .saturating_mul(tensor.format.precision.bytes())
}

fn maximum_axis_shard_extent(tensor: &TensorType, axis: usize) -> u64 {
    (0..tensor.format.layout.tiling.tile_count)
        .filter_map(|tile| tile_axis_range(tensor, axis, tile))
        .map(|range| u64::from(range.end - range.start))
        .max()
        .unwrap_or(u64::MAX)
}

fn tensor_memory(tensor: &TensorType) -> MemoryUsage {
    let mut usage = MemoryUsage::default();
    usage.add_class(
        tensor.format.layout.memory_class,
        maximum_shard_bytes(tensor),
    );
    usage
}

fn operator_memory_estimate(
    dispatch: &OperatorDispatch,
    inputs: &[TensorType],
    output: &TensorType,
) -> MemoryEstimate {
    let live = inputs.iter().fold(tensor_memory(output), |usage, input| {
        usage.saturating_add(tensor_memory(input))
    });
    let mut temporary = MemoryUsage::default();
    if let (
        OperatorDispatch::BlockedGemm {
            inner_block,
            output_column_block,
            ..
        },
        Some(right),
    ) = (dispatch, inputs.get(1))
        && right.format.precision == Precision::F16
        && gemm_requires_staging(dispatch, right, output)
    {
        // One packed panel is reused across K phases. Exchange receive storage
        // is a separately reserved architectural window and is not counted as
        // planned data SRAM.
        temporary.interleaved = u64::from(*inner_block)
            .saturating_mul(u64::from(*output_column_block))
            .saturating_mul(right.format.precision.bytes());
    }
    MemoryEstimate {
        live,
        temporary,
        peak: live.saturating_add(temporary),
    }
}

fn gemm_requires_staging(
    dispatch: &OperatorDispatch,
    right: &TensorType,
    output: &TensorType,
) -> bool {
    if right.format.layout.memory_class == MemoryClass::Ipu21Interleaved {
        return false;
    }
    let OperatorDispatch::BlockedGemm {
        inner_block,
        output_column_block: _,
        ..
    } = dispatch
    else {
        return false;
    };
    let rank = right.shape.0.len();
    let output_rank = output.shape.0.len();
    if rank < 2 || output_rank < 2 {
        return true;
    }
    let streamed = right
        .format
        .layout
        .tiling
        .axes
        .iter()
        .any(|axis| axis.axis == TensorAxis::FromEnd(2) && axis.partitions > 1);
    let k = right.shape.0[rank - 2];
    let columns = maximum_axis_shard_extent(output, output_rank - 1);
    streamed || (k > *inner_block && columns > 16)
}

fn gemm_exchange_bytes_per_cycle(inputs: &[TensorType]) -> u64 {
    let Some(right) = inputs.get(1) else {
        return IPU21_EXCHANGE_BYTES_PER_CYCLE;
    };
    let inner_sharded = right
        .format
        .layout
        .tiling
        .axes
        .iter()
        .any(|axis| axis.axis == TensorAxis::FromEnd(2) && axis.partitions > 1);
    let column_partitions = right
        .format
        .layout
        .tiling
        .axes
        .iter()
        .find(|axis| axis.axis == TensorAxis::FromEnd(1))
        .map_or(1, |axis| axis.partitions);
    // IPU21 pairs adjacent tiles on one shared exchange bus. With one output
    // column partition, a streamed K panel is a full broadcast to consecutive
    // tiles and both receivers can consume it in the same cycle.
    if inner_sharded && column_partitions == 1 {
        IPU21_EXCHANGE_BYTES_PER_CYCLE * 2
    } else {
        IPU21_EXCHANGE_BYTES_PER_CYCLE
    }
}

fn conversion_memory_estimate(input: &TensorType, output: &TensorType) -> MemoryEstimate {
    let live = tensor_memory(input).saturating_add(tensor_memory(output));
    MemoryEstimate {
        live,
        temporary: MemoryUsage::default(),
        peak: live,
    }
}

fn region_peak_memory(
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
) -> MemoryUsage {
    let mut uses = BTreeMap::<MidValueId, u32>::new();
    for input in operations.iter().flat_map(|operation| &operation.inputs) {
        *uses.entry(*input).or_default() += 1;
    }
    for output in outputs {
        *uses.entry(*output).or_default() += 1;
    }
    let mut live_values = BTreeSet::new();
    let mut live = MemoryUsage::default();
    for id in initial {
        if live_values.insert(*id) {
            live = live.saturating_add(tensor_memory(&values[id.index() as usize].tensor_type));
        }
    }
    let mut peak = live;
    for operation in operations {
        let mut during = live.saturating_add(operation.memory.temporary);
        for result in &operation.results {
            if !live_values.contains(result) {
                during = during
                    .saturating_add(tensor_memory(&values[result.index() as usize].tensor_type));
            }
        }
        peak = peak.component_max(during);
        for input in &operation.inputs {
            if let Some(remaining) = uses.get_mut(input) {
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 && live_values.remove(input) {
                    let bytes = tensor_memory(&values[input.index() as usize].tensor_type);
                    live.standard = live.standard.saturating_sub(bytes.standard);
                    live.interleaved = live.interleaved.saturating_sub(bytes.interleaved);
                }
            }
        }
        for result in &operation.results {
            if uses.get(result).copied().unwrap_or(0) != 0 && live_values.insert(*result) {
                live = live
                    .saturating_add(tensor_memory(&values[result.index() as usize].tensor_type));
            }
        }
    }
    peak.component_max(live)
}

fn gemm_remote_bytes_per_tile(inputs: &[TensorType], output: &TensorType) -> u64 {
    let [left, right] = inputs else {
        return u64::MAX;
    };
    if left.shape.0.len() < 2 || right.shape.0.len() < 2 || output.shape.0.len() < 2 {
        return u64::MAX;
    }
    let tiles = output.format.layout.tiling.tile_count;
    let output_row_axis = output.shape.0.len() - 2;
    let output_column_axis = output.shape.0.len() - 1;
    let left_row_axis = left.shape.0.len() - 2;
    let right_column_axis = right.shape.0.len() - 1;
    let right_inner_axis = right.shape.0.len() - 2;
    let k = left.shape.0[left.shape.0.len() - 1];
    (0..tiles).fold(0u64, |maximum, tile| {
        let Some(output_rows) = tile_axis_range(output, output_row_axis, tile) else {
            return u64::MAX;
        };
        let Some(output_columns) = tile_axis_range(output, output_column_axis, tile) else {
            return u64::MAX;
        };
        let Some(left_rows) = tile_axis_range(left, left_row_axis, tile) else {
            return u64::MAX;
        };
        let Some(right_columns) = tile_axis_range(right, right_column_axis, tile) else {
            return u64::MAX;
        };
        let Some(right_inner) = tile_axis_range(right, right_inner_axis, tile) else {
            return u64::MAX;
        };
        let left_remote = if left_rows.start > output_rows.start || left_rows.end < output_rows.end
        {
            u64::from(output_rows.end - output_rows.start)
                .saturating_mul(u64::from(k))
                .saturating_mul(left.format.precision.bytes())
        } else {
            0
        };
        let right_remote = if right_columns.start > output_columns.start
            || right_columns.end < output_columns.end
            || right_inner.start != 0
            || right_inner.end < k
        {
            u64::from(output_columns.end - output_columns.start)
                .saturating_mul(u64::from(k))
                .saturating_mul(right.format.precision.bytes())
        } else {
            0
        };
        maximum.max(left_remote.saturating_add(right_remote))
    })
}

fn tile_axis_range(tensor: &TensorType, axis: usize, tile: u16) -> Option<std::ops::Range<u32>> {
    let layout = &tensor.format.layout;
    let padded = layout.padded_shape(&tensor.shape).ok()?;
    let Some((tiling, stride)) = layout
        .tiling
        .axes
        .iter()
        .zip(layout.tiling.axis_strides().ok()?)
        .find(|(tiling, _)| tiling.axis.resolve(padded.0.len()) == Ok(axis))
    else {
        return Some(0..padded.0[axis]);
    };
    let blocks = padded.0[axis] / tiling.block_size;
    let partitions = u32::from(tiling.partitions);
    let coordinate = (u32::from(tile) / stride) % partitions;
    let short = blocks / partitions;
    let long = blocks % partitions;
    let start_blocks = coordinate * short + coordinate.min(long);
    let shard_blocks = short + u32::from(coordinate < long);
    Some(start_blocks * tiling.block_size..(start_blocks + shard_blocks) * tiling.block_size)
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LoweringError {
    #[error("mid-level lowering requires a nonzero tile count")]
    EmptyTileGroup,
    #[error("no tensor type was supplied for graph input {0:?}")]
    MissingInputType(ValueId),
    #[error("graph has no stored shape for value {0:?}")]
    MissingShape(ValueId),
    #[error("operation {0:?} has no legal format candidate")]
    NoCandidate(OperationId),
    #[error("internal lowering error: value {0:?} is unavailable")]
    UnknownValue(ValueId),
}

pub type LoweringResult<T> = std::result::Result<T, LoweringError>;

#[tracing::instrument(
    name = "ipu_codegen.mid.lower",
    skip(graph, config, costs),
    fields(tile_count = config.tile_count, operations = graph.operations().len())
)]
pub fn lower(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
) -> LoweringResult<MidGraph> {
    if config.tile_count == 0 {
        return Err(LoweringError::EmptyTileGroup);
    }
    let mut state = LoweringState::default();
    let mut values = BTreeMap::new();
    let mut inputs = Vec::with_capacity(graph.inputs().len());
    for input in graph.inputs() {
        let (format, automatic) = if let Some(format) = config.inputs.get(&input.value) {
            (format.clone(), false)
        } else if let Some(&precision) = config.automatic_inputs.get(&input.value) {
            (
                TensorFormat {
                    precision,
                    layout: Layout::row_sharded(config.tile_count),
                },
                true,
            )
        } else {
            return Err(LoweringError::MissingInputType(input.value));
        };
        let tensor_type = TensorType {
            shape: input.shape.clone(),
            format,
        };
        let value = state.value(input.value, tensor_type);
        if automatic {
            state.automatic_inputs.insert(value);
        }
        values.insert(input.value, value);
        inputs.push(MidInput {
            name: input.name.clone(),
            kind: input.kind,
            value,
        });
    }
    let operations = lower_operations(
        graph.operations(),
        &mut values,
        graph.value_shapes(),
        graph,
        config,
        costs,
        &mut state,
    )?;
    let outputs = graph
        .outputs()
        .iter()
        .map(|value| lookup(&values, *value))
        .collect::<LoweringResult<Vec<_>>>()?;
    let estimated_cost = operations
        .iter()
        .map(|operation| operation.estimated_cost)
        .sum();
    let initial = inputs.iter().map(|input| input.value).collect::<Vec<_>>();
    let peak_memory = region_peak_memory(&initial, &operations, &outputs, &state.values);
    tracing::info!(
        values = state.values.len(),
        operations = operations.len(),
        estimated_cost,
        "selected operator plans"
    );
    Ok(MidGraph {
        inputs,
        values: state.values,
        operations,
        outputs,
        estimated_cost,
        peak_memory,
    })
}

#[derive(Default)]
struct LoweringState {
    values: Vec<MidValue>,
    automatic_inputs: BTreeSet<MidValueId>,
}

impl LoweringState {
    fn value(&mut self, origin: ValueId, tensor_type: TensorType) -> MidValueId {
        let id = MidValueId(self.values.len() as u32);
        self.values.push(MidValue {
            id,
            tensor_type,
            origin,
        });
        id
    }

    fn get(&self, id: MidValueId) -> &MidValue {
        &self.values[id.0 as usize]
    }

    fn retarget_automatic_input(&mut self, id: MidValueId, layout: Layout) -> bool {
        if !self.automatic_inputs.remove(&id) {
            return false;
        }
        self.values[id.0 as usize].tensor_type.format.layout = layout;
        true
    }
}

fn lower_operations(
    source: &[Operation],
    values: &mut BTreeMap<ValueId, MidValueId>,
    shapes: &BTreeMap<ValueId, TensorShape>,
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
    state: &mut LoweringState,
) -> LoweringResult<Vec<MidOperation>> {
    let mut operations = Vec::new();
    for operation in source {
        if let OperationKind::Repeat(repeat) = &operation.kind {
            lower_repeat(
                operation,
                repeat,
                values,
                graph,
                config,
                costs,
                state,
                &mut operations,
            )?;
            continue;
        }
        let input_ids = operation
            .inputs
            .iter()
            .map(|value| lookup(values, *value))
            .collect::<LoweringResult<Vec<_>>>()?;
        let input_types = input_ids
            .iter()
            .map(|value| state.get(*value).tensor_type.clone())
            .collect::<Vec<_>>();
        let output_shape = shapes
            .get(&operation.results[0])
            .cloned()
            .ok_or(LoweringError::MissingShape(operation.results[0]))?;
        let plans = plans(operation, &input_types, &output_shape, config);
        let plan = plans
            .into_iter()
            .filter(|plan| {
                let grid_plan = matches!(plan.operator, MidOperator::Gemm { .. })
                    && (plan
                        .requirements
                        .inputs
                        .iter()
                        .any(|input| input.format.layout.tiling.replicas > 1)
                        || plan.requirements.inputs.get(1).is_some_and(|right| {
                            right.format.layout.tiling.axes.iter().any(|axis| {
                                axis.axis == TensorAxis::FromEnd(2) && axis.partitions > 1
                            })
                        }));
                !grid_plan
                    || input_ids
                        .iter()
                        .zip(&plan.requirements.inputs)
                        .all(|(id, requirement)| {
                            state.automatic_inputs.contains(id)
                                || state.get(*id).tensor_type.format.layout
                                    == requirement.format.layout
                        })
            })
            .enumerate()
            .map(|(order, plan)| {
                let conversion = input_types
                    .iter()
                    .zip(&input_ids)
                    .zip(&plan.requirements.inputs)
                    .map(|((from, id), to)| {
                        if state.automatic_inputs.contains(id) {
                            if from.format.precision != to.format.precision {
                                costs.cast_cost(from, to.format.precision)
                            } else {
                                0
                            }
                        } else {
                            conversion_cost(from, &to.format, costs)
                        }
                    })
                    .sum::<u64>();
                let output = TensorType {
                    shape: output_shape.clone(),
                    format: plan.requirements.output.format.clone(),
                };
                let planned_inputs = input_types
                    .iter()
                    .zip(&plan.requirements.inputs)
                    .map(|(input, requirement)| TensorType {
                        shape: input.shape.clone(),
                        format: requirement.format.clone(),
                    })
                    .collect::<Vec<_>>();
                let cost = conversion
                    + costs.operator_cost(plan.operator, &plan.dispatch, &planned_inputs, &output);
                ((cost, order), plan)
            })
            .min_by_key(|(cost, _)| *cost)
            .ok_or(LoweringError::NoCandidate(operation.id))?
            .1;
        tracing::debug!(
            operation = operation.id.index(),
            operator = ?plan.operator,
            "selected operator candidate"
        );
        let converted = input_ids
            .into_iter()
            .zip(&plan.requirements.inputs)
            .map(|(value, requirement)| {
                ensure_format(
                    value,
                    requirement.format.clone(),
                    operation.id,
                    costs,
                    state,
                    &mut operations,
                )
            })
            .collect::<Vec<_>>();
        let result = state.value(
            operation.results[0],
            TensorType {
                shape: output_shape,
                format: plan.requirements.output.format.clone(),
            },
        );
        let converted_types = converted
            .iter()
            .map(|value| state.get(*value).tensor_type.clone())
            .collect::<Vec<_>>();
        let operator_cost = costs.operator_cost(
            plan.operator,
            &plan.dispatch,
            &converted_types,
            &state.get(result).tensor_type,
        );
        let memory = operator_memory_estimate(
            &plan.dispatch,
            &converted_types,
            &state.get(result).tensor_type,
        );
        operations.push(MidOperation {
            source: Some(operation.id),
            inputs: converted,
            results: vec![result],
            kind: MidOperationKind::Operator(plan.operator),
            operator_plan: Some(OperatorPlan {
                operator: plan.operator,
                dispatch: plan.dispatch,
                requirements: plan.requirements,
            }),
            conversion_plan: None,
            estimated_cost: operator_cost,
            memory,
        });
        values.insert(operation.results[0], result);
    }
    Ok(operations)
}

#[derive(Clone)]
struct Plan {
    operator: MidOperator,
    dispatch: OperatorDispatch,
    requirements: OperatorRequirements,
}

fn plans(
    operation: &Operation,
    inputs: &[TensorType],
    output: &TensorShape,
    config: &PipelineConfig,
) -> Vec<Plan> {
    config
        .operator_candidates
        .iter()
        .filter(|candidate| {
            operator_matches(&operation.kind, candidate.operator)
                && candidate.supports(inputs, output)
        })
        .map(|candidate| Plan {
            operator: candidate.operator,
            dispatch: candidate.dispatch.clone(),
            requirements: OperatorRequirements {
                inputs: candidate.inputs.clone(),
                output: candidate.output.clone(),
                output_aliasing: resolved_output_aliasing(candidate, inputs, output),
                memory_relations: candidate.memory_relations.clone(),
            },
        })
        .collect()
}

fn resolved_output_aliasing(
    candidate: &OperatorCandidate,
    inputs: &[TensorType],
    output: &TensorShape,
) -> OutputAliasing {
    match &candidate.output_aliasing {
        OutputAliasing::MayAliasInputs(indices) => OutputAliasing::MayAliasInputs(
            indices
                .iter()
                .copied()
                .filter(|index| {
                    alias_compatible(
                        usize::from(*index),
                        &candidate.inputs,
                        inputs,
                        &candidate.output,
                        output,
                    )
                })
                .collect(),
        ),
        aliasing => aliasing.clone(),
    }
}

fn operator_matches(operation: &OperationKind, operator: MidOperator) -> bool {
    match (operation, operator) {
        (OperationKind::Gemm(expected), MidOperator::Gemm { options, .. }) => *expected == options,
        (OperationKind::Gelu, MidOperator::Gelu) => true,
        (OperationKind::Add(expected), MidOperator::Add(options)) => *expected == options,
        (OperationKind::FlashAttention(expected), MidOperator::FlashAttention { options, .. }) => {
            *expected == options
        }
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_repeat(
    operation: &Operation,
    repeat: &Repeat,
    values: &mut BTreeMap<ValueId, MidValueId>,
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
    state: &mut LoweringState,
    operations: &mut Vec<MidOperation>,
) -> LoweringResult<()> {
    let inputs = operation
        .inputs
        .iter()
        .map(|value| lookup(values, *value))
        .collect::<LoweringResult<Vec<_>>>()?;
    let mut argument_types = inputs
        .iter()
        .map(|value| state.get(*value).tensor_type.clone())
        .collect::<Vec<_>>();
    let mut iterated_inputs = Vec::with_capacity(repeat.iterated_inputs.len());
    for sequence_id in &repeat.iterated_inputs {
        let sequence = &graph.sequences()[sequence_id.index() as usize];
        let first = lookup(values, sequence.values[0])?;
        let first_type = state.get(first).tensor_type.clone();
        let normalized = sequence
            .values
            .iter()
            .map(|value| lookup(values, *value))
            .collect::<LoweringResult<Vec<_>>>()?
            .into_iter()
            .map(|value| {
                ensure_format(
                    value,
                    first_type.format.clone(),
                    operation.id,
                    costs,
                    state,
                    operations,
                )
            })
            .collect();
        iterated_inputs.push(normalized);
        argument_types.push(first_type);
    }
    let mut body_values = BTreeMap::new();
    let mut arguments = Vec::new();
    for (&origin, tensor_type) in repeat.body.arguments.iter().zip(argument_types) {
        let value = state.value(origin, tensor_type);
        body_values.insert(origin, value);
        arguments.push(value);
    }
    let mut body_operations = lower_operations(
        &repeat.body.operations,
        &mut body_values,
        &repeat.body.value_shapes,
        graph,
        config,
        costs,
        state,
    )?;
    let mut yields = Vec::new();
    for (index, high_yield) in repeat.body.yields.iter().enumerate() {
        let value = lookup(&body_values, *high_yield)?;
        let target = state.get(inputs[index]).tensor_type.format.clone();
        yields.push(ensure_format(
            value,
            target,
            operation.id,
            costs,
            state,
            &mut body_operations,
        ));
    }
    let body_cost = body_operations
        .iter()
        .map(|operation| operation.estimated_cost)
        .sum();
    let body_peak = region_peak_memory(&arguments, &body_operations, &yields, &state.values);
    let mut results = Vec::new();
    for (origin, input) in operation.results.iter().zip(&inputs) {
        let result = state.value(*origin, state.get(*input).tensor_type.clone());
        values.insert(*origin, result);
        results.push(result);
    }
    operations.push(MidOperation {
        source: Some(operation.id),
        inputs,
        results,
        kind: MidOperationKind::Repeat(MidRepeat {
            count: repeat.count,
            carried_inputs: repeat.carried_inputs,
            invariant_inputs: repeat.invariant_inputs,
            iterated_inputs,
            body: MidRegion {
                arguments,
                operations: body_operations,
                yields,
                estimated_cost: body_cost,
                peak_memory: body_peak,
            },
        }),
        operator_plan: None,
        conversion_plan: None,
        estimated_cost: body_cost.saturating_mul(u64::from(repeat.count)),
        memory: MemoryEstimate {
            live: body_peak,
            temporary: MemoryUsage::default(),
            peak: body_peak,
        },
    });
    Ok(())
}

fn ensure_format(
    mut value: MidValueId,
    target: TensorFormat,
    source: OperationId,
    costs: &impl CostModel,
    state: &mut LoweringState,
    operations: &mut Vec<MidOperation>,
) -> MidValueId {
    if state.retarget_automatic_input(value, target.layout.clone())
        && state.get(value).tensor_type.format.precision == target.precision
    {
        return value;
    }
    let original = state.get(value).clone();
    if original.tensor_type.format.precision != target.precision {
        let mut tensor_type = original.tensor_type.clone();
        let from = tensor_type.format.precision;
        tensor_type.format.precision = target.precision;
        let result = state.value(original.origin, tensor_type.clone());
        let memory = conversion_memory_estimate(&original.tensor_type, &tensor_type);
        operations.push(MidOperation {
            source: Some(source),
            inputs: vec![value],
            results: vec![result],
            kind: MidOperationKind::CastPrecision {
                from,
                to: target.precision,
            },
            operator_plan: None,
            conversion_plan: Some(ConversionPlan {
                kernel: TileKernelSpec::Cast {
                    from,
                    to: target.precision,
                },
                input: OperandRequirement::new(original.tensor_type.format.clone(), 8),
                output: OperandRequirement::new(tensor_type.format.clone(), 8),
                dispatch: ConversionDispatch::Local,
            }),
            estimated_cost: costs.cast_cost(&original.tensor_type, target.precision),
            memory,
        });
        value = result;
    }
    let current = state.get(value).clone();
    if current.tensor_type.format.layout != target.layout {
        let mut tensor_type = current.tensor_type.clone();
        let from = tensor_type.format.layout.clone();
        tensor_type.format.layout = target.layout.clone();
        let result = state.value(current.origin, tensor_type.clone());
        let memory = conversion_memory_estimate(&current.tensor_type, &tensor_type);
        operations.push(MidOperation {
            source: Some(source),
            inputs: vec![value],
            results: vec![result],
            kind: MidOperationKind::Rearrange {
                from: from.clone(),
                to: target.layout.clone(),
            },
            operator_plan: None,
            conversion_plan: Some(ConversionPlan {
                kernel: TileKernelSpec::Rearrange {
                    from: from.clone(),
                    to: target.layout.clone(),
                },
                input: OperandRequirement::new(current.tensor_type.format.clone(), 8),
                output: OperandRequirement::new(tensor_type.format.clone(), 8),
                dispatch: ConversionDispatch::Intersections,
            }),
            estimated_cost: costs.rearrange_cost(
                &tensor_type.shape,
                tensor_type.format.precision,
                &from,
                &target.layout,
            ),
            memory,
        });
        value = result;
    }
    value
}

fn conversion_cost(from: &TensorType, to: &TensorFormat, costs: &impl CostModel) -> u64 {
    let cast = if from.format.precision != to.precision {
        costs.cast_cost(from, to.precision)
    } else {
        0
    };
    let rearrange = if from.format.layout != to.layout {
        costs.rearrange_cost(&from.shape, to.precision, &from.format.layout, &to.layout)
    } else {
        0
    };
    cast.saturating_add(rearrange)
}

fn lookup(values: &BTreeMap<ValueId, MidValueId>, value: ValueId) -> LoweringResult<MidValueId> {
    values
        .get(&value)
        .copied()
        .ok_or(LoweringError::UnknownValue(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RANDOM_CASES: usize = 128;

    fn dimension(random: &mut fastrand::Rng) -> u32 {
        random.u32(1..=128)
    }

    fn precision(random: &mut fastrand::Rng) -> Precision {
        if random.bool() {
            Precision::F16
        } else {
            Precision::F32
        }
    }

    fn format(precision: Precision, layout: Layout) -> TensorFormat {
        TensorFormat { precision, layout }
    }

    fn random_format(random: &mut fastrand::Rng, tiles: u16) -> TensorFormat {
        let tiling = if random.bool() {
            TensorTiling::replicated(tiles)
        } else {
            TensorTiling::sharded(TensorAxis::FromEnd(2), tiles)
        };
        let mut layout = Layout::row_major(tiling);
        if random.bool() {
            layout.memory_class = MemoryClass::Ipu21Interleaved;
        }
        format(precision(random), layout)
    }

    fn value(lowered: &MidGraph, id: MidValueId) -> &MidValue {
        &lowered.values[id.index() as usize]
    }

    #[test]
    fn randomized_memory_usage_checks_coupled_ipu21_capacity() {
        let mut random = fastrand::Rng::with_seed(0x6d65_6d32);
        for _ in 0..RANDOM_CASES {
            let usage = MemoryUsage {
                standard: random.u64(0..=u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES) * 2),
                interleaved: random
                    .u64(0..=u64::from(crate::memory::IPU21_INTERLEAVED_REGION_BYTES) * 2),
            };
            assert_eq!(
                usage.fits_ipu21(),
                usage.interleaved <= u64::from(crate::memory::IPU21_INTERLEAVED_REGION_BYTES)
                    && usage.total() <= u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES)
            );
        }
    }

    #[test]
    fn randomized_cycle_model_rewards_direct_interleaved_weight_loads() {
        let mut random = fastrand::Rng::with_seed(0x6379_636c);
        for _ in 0..RANDOM_CASES {
            let rows = 1_u16 << random.u32(0..=2);
            let columns = 1_u16 << random.u32(0..=2);
            let tiles = rows * columns;
            let m = u32::from(rows) * random.u32(1..=4);
            let k = 64 * random.u32(2..=4);
            let n = u32::from(columns) * 64;
            let left = TensorType::new(
                [m, k],
                Precision::F16,
                Layout::amp_left_grid(64, tiles, rows, columns),
            );
            let mut standard_layout = Layout::amp_right_grid(64, tiles, rows, columns);
            let mut direct_layout = standard_layout.clone();
            direct_layout.memory_class = MemoryClass::Ipu21Interleaved;
            standard_layout.memory_class = MemoryClass::Ipu21Standard;
            let standard = TensorType::new([k, n], Precision::F16, standard_layout);
            let direct = TensorType::new([k, n], Precision::F16, direct_layout);
            let output = TensorType::new(
                [m, n],
                Precision::F16,
                Layout::amp_output_grid(tiles, rows, columns),
            );
            let operator = MidOperator::Gemm {
                options: GemmOptions::default(),
                multiply: Precision::F16,
                accumulate: AccumulationPrecision::F32,
            };
            let dispatch = default_dispatch(operator);
            let standard_cost =
                ToyCostModel.operator_cost(operator, &dispatch, &[left.clone(), standard], &output);
            let direct_cost =
                ToyCostModel.operator_cost(operator, &dispatch, &[left, direct], &output);
            assert!(direct_cost < standard_cost);
        }
    }

    fn assert_conversions_are_explicit(lowered: &MidGraph, operations: &[MidOperation]) {
        for operation in operations {
            let [input] = operation.inputs.as_slice() else {
                continue;
            };
            let [result] = operation.results.as_slice() else {
                continue;
            };
            let before = &value(lowered, *input).tensor_type;
            let after = &value(lowered, *result).tensor_type;
            match &operation.kind {
                MidOperationKind::CastPrecision { from, to } => {
                    assert_eq!(*from, before.format.precision);
                    assert_eq!(*to, after.format.precision);
                    assert_eq!(before.shape, after.shape);
                    assert_eq!(before.format.layout, after.format.layout);
                }
                MidOperationKind::Rearrange { from, to } => {
                    assert_eq!(from, &before.format.layout);
                    assert_eq!(to, &after.format.layout);
                    assert_eq!(before.shape, after.shape);
                    assert_eq!(before.format.precision, after.format.precision);
                }
                MidOperationKind::Operator(_) | MidOperationKind::Repeat(_) => {}
            }
        }
    }

    fn assert_operator_signature(
        lowered: &MidGraph,
        operation: &MidOperation,
        inputs: &[TensorFormat],
        output: TensorFormat,
    ) {
        assert_eq!(operation.inputs.len(), inputs.len());
        for (&value_id, expected) in operation.inputs.iter().zip(inputs) {
            assert_eq!(&value(lowered, value_id).tensor_type.format, expected);
        }
        assert_eq!(
            value(lowered, operation.results[0]).tensor_type.format,
            output
        );
    }

    struct ColumnParityCost;

    impl CostModel for ColumnParityCost {
        fn operator_cost(
            &self,
            operator: MidOperator,
            _dispatch: &OperatorDispatch,
            _inputs: &[TensorType],
            output: &TensorType,
        ) -> u64 {
            let preferred = if output.shape.0.last().unwrap().is_multiple_of(2) {
                Precision::F16
            } else {
                Precision::F32
            };
            match operator {
                MidOperator::Gemm { multiply, .. } if multiply == preferred => 0,
                MidOperator::Gemm { .. } => 1,
                _ => 0,
            }
        }

        fn cast_cost(&self, _input: &TensorType, _to: Precision) -> u64 {
            0
        }

        fn rearrange_cost(
            &self,
            _shape: &TensorShape,
            _precision: Precision,
            _from: &Layout,
            _to: &Layout,
        ) -> u64 {
            0
        }
    }

    #[test]
    fn randomized_axis_tiling_applies_or_rejects_padding() {
        let mut random = fastrand::Rng::with_seed(0x7469_6c65);
        for case in 0..RANDOM_CASES {
            let rank = random.usize(1..=6);
            let axis = random.usize(0..rank);
            let extent = dimension(&mut random);
            let block_size = random.u32(1..=32);
            let partitions = random.u16(1..=16);
            let replicas = random.u16(1..=4);
            let padding = if random.bool() {
                Padding::Reject
            } else {
                Padding::Zero
            };
            let mut shape = (0..rank)
                .map(|_| dimension(&mut random))
                .collect::<Vec<_>>();
            shape[axis] = extent;
            let layout = Layout::row_major(TensorTiling {
                tile_count: partitions * replicas,
                replicas,
                axes: vec![AxisTiling::new(
                    TensorAxis::FromStart(axis as u16),
                    partitions,
                    block_size,
                    padding,
                )],
            });

            let result = layout.padded_shape(&TensorShape(shape.clone()));
            if padding == Padding::Reject && !extent.is_multiple_of(block_size) {
                assert!(
                    matches!(result, Err(LayoutError::IndivisibleAxis { .. })),
                    "random case {case}"
                );
            } else {
                let padded = result.unwrap();
                let expected = extent.div_ceil(block_size) * block_size;
                assert_eq!(padded.0[axis], expected, "random case {case}");
                for (other, original) in shape.iter().enumerate() {
                    if other != axis {
                        assert_eq!(padded.0[other], *original, "random case {case}");
                    }
                }
            }
        }
    }

    #[test]
    fn randomized_gemm_lowering_makes_every_format_boundary_explicit() {
        let mut random = fastrand::Rng::with_seed(0x6d69_6467);
        for case in 0..RANDOM_CASES {
            let (rows, inner, columns) = (
                dimension(&mut random),
                dimension(&mut random) * 64,
                dimension(&mut random) * 64,
            );
            let batches = (0..random.usize(0..=3))
                .map(|_| dimension(&mut random))
                .collect::<Vec<_>>();
            let tiles = random.u16(1..=64);
            let multiply = precision(&mut random);
            let left_format = format(
                precision(&mut random),
                Layout::amp_left([8, 16, 32][random.usize(0..3)], tiles),
            );
            let right_format = format(
                if random.bool() {
                    precision(&mut random)
                } else {
                    Precision::F8F143 {
                        scale_exponent: random.i8(-16..=16),
                    }
                },
                Layout::amp_right([8, 16, 32][random.usize(0..3)], tiles),
            );
            let output_format = format(precision(&mut random), Layout::amp_output(tiles));
            let accumulate = if random.bool() {
                AccumulationPrecision::F16
            } else {
                AccumulationPrecision::F32
            };
            let candidate = OperatorCandidate::new(
                MidOperator::Gemm {
                    options: GemmOptions::default(),
                    multiply,
                    accumulate,
                },
                [
                    OperandRequirement::new(left_format.clone(), 32),
                    OperandRequirement::new(right_format.clone(), 32),
                ],
                OperandRequirement::new(output_format.clone(), 32),
            );
            let mut left_shape = batches.clone();
            left_shape.extend([rows, inner]);
            let mut right_shape = batches;
            right_shape.extend([inner, columns]);

            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", left_shape).unwrap();
            let right = graph.parameter("right", right_shape).unwrap();
            let product = graph.gemm(left, right).unwrap();
            graph.set_outputs([product]).unwrap();
            let linear = Layout::row_sharded(tiles);
            let mut config = PipelineConfig::new(tiles)
                .with_input(left, format(precision(&mut random), linear.clone()))
                .with_input(right, format(precision(&mut random), linear));
            config.operator_candidates = vec![candidate.clone()];

            let lowered = lower(&graph, &config, &ToyCostModel).unwrap();
            let operator = lowered
                .operations
                .iter()
                .find(|operation| matches!(operation.kind, MidOperationKind::Operator(_)))
                .unwrap();
            let MidOperationKind::Operator(MidOperator::Gemm {
                multiply: selected_multiply,
                accumulate: selected_accumulate,
                ..
            }) = operator.kind
            else {
                panic!("random case {case}: expected GEMM");
            };
            assert_eq!(selected_multiply, multiply, "random case {case}");
            assert_eq!(selected_accumulate, accumulate, "random case {case}");
            assert_eq!(
                &value(&lowered, operator.inputs[0]).tensor_type.format,
                &candidate.inputs[0].format,
                "random case {case}"
            );
            assert_eq!(
                &value(&lowered, operator.inputs[1]).tensor_type.format,
                &candidate.inputs[1].format,
                "random case {case}"
            );
            let output = value(&lowered, lowered.outputs[0]);
            let expected_shape = graph.value_shape(product).unwrap().clone();
            assert_eq!(
                output.tensor_type.shape, expected_shape,
                "random case {case}"
            );
            assert_eq!(
                &output.tensor_type.format, &candidate.output.format,
                "random case {case}"
            );
            assert_conversions_are_explicit(&lowered, &lowered.operations);
        }
    }

    #[test]
    fn randomized_gemms_choose_precision_independently_within_one_graph() {
        let mut random = fastrand::Rng::with_seed(0x6d75_6c74);
        for case in 0..RANDOM_CASES {
            let rows = dimension(&mut random);
            let inner = dimension(&mut random);
            let even_columns = random.u32(1..=64) * 2;
            let odd_columns = random.u32(1..=64) * 2 - 1;
            let tiles = random.u16(1..=64);
            let layout = Layout::row_sharded(tiles);
            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", [rows, inner]).unwrap();
            let even_right = graph.parameter("even", [inner, even_columns]).unwrap();
            let odd_right = graph.parameter("odd", [inner, odd_columns]).unwrap();
            let even = graph.gemm(left, even_right).unwrap();
            let odd = graph.gemm(left, odd_right).unwrap();
            graph.set_outputs([even, odd]).unwrap();
            let input_format = format(precision(&mut random), layout);
            let config = PipelineConfig::new(tiles)
                .with_input(left, input_format.clone())
                .with_input(even_right, input_format.clone())
                .with_input(odd_right, input_format);

            let lowered = lower(&graph, &config, &ColumnParityCost).unwrap();
            let chosen = lowered
                .operations
                .iter()
                .filter_map(|operation| match operation.kind {
                    MidOperationKind::Operator(MidOperator::Gemm { multiply, .. }) => {
                        Some(multiply)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                chosen,
                vec![Precision::F16, Precision::F32],
                "random case {case}"
            );
            for operation in lowered.operations.iter().filter(|operation| {
                matches!(
                    operation.kind,
                    MidOperationKind::Operator(MidOperator::Gemm { .. })
                )
            }) {
                let requirements = &operation.operator_plan.as_ref().unwrap().requirements;
                assert!(
                    requirements
                        .inputs
                        .iter()
                        .chain([&requirements.output])
                        .all(|requirement| requirement.alignment == 32)
                );
                assert_eq!(
                    requirements.output.format.layout.memory_class,
                    MemoryClass::Ipu21Interleaved
                );
                assert_eq!(
                    requirements.memory_relations,
                    [MemoryRelation::DistinctElements(vec![
                        MemoryOperand::Output,
                        MemoryOperand::Input(0),
                    ])]
                );
                let expected_tail = match operation.kind {
                    MidOperationKind::Operator(MidOperator::Gemm {
                        multiply: Precision::F16,
                        ..
                    }) => 16,
                    MidOperationKind::Operator(MidOperator::Gemm {
                        multiply: Precision::F32,
                        ..
                    }) => 32,
                    _ => unreachable!(),
                };
                assert_eq!(requirements.inputs[0].access_tail_bytes, expected_tail);
            }
        }
    }

    #[test]
    fn randomized_non_gemm_lowering_honors_operator_plans() {
        let mut random = fastrand::Rng::with_seed(0x6164_642b);
        for case in 0..RANDOM_CASES {
            let tiles = random.u16(1..=64);
            let batch = dimension(&mut random);
            let query_rows = dimension(&mut random);
            let key_rows = dimension(&mut random);
            let channels = dimension(&mut random);
            let value_channels = dimension(&mut random);
            let mut graph = ComputeGraph::new();
            let activation = graph
                .host_input("activation", [batch, query_rows, channels])
                .unwrap();
            let residual = graph
                .host_input("residual", [batch, query_rows, channels])
                .unwrap();
            let query = graph
                .host_input("query", [batch, query_rows, channels])
                .unwrap();
            let key = graph
                .host_input("key", [batch, key_rows, channels])
                .unwrap();
            let attention_value = graph
                .host_input("value", [batch, key_rows, value_channels])
                .unwrap();
            let activated = graph.gelu(activation).unwrap();
            let sum = graph.add(activated, residual).unwrap();
            let attended = graph.flash_attention(query, key, attention_value).unwrap();
            graph.set_outputs([sum, attended]).unwrap();

            let gelu_input = random_format(&mut random, tiles);
            let gelu_output = gelu_input.clone();
            let add_left = random_format(&mut random, tiles);
            let add_right = random_format(&mut random, tiles);
            let add_output = add_left.clone();
            let attention_query = random_format(&mut random, tiles);
            let attention_key = random_format(&mut random, tiles);
            let attention_value_format = random_format(&mut random, tiles);
            let attention_output = random_format(&mut random, tiles);
            let attention_accumulate = if random.bool() {
                AccumulationPrecision::F16
            } else {
                AccumulationPrecision::F32
            };
            let mut config = PipelineConfig::new(tiles)
                .with_input(activation, random_format(&mut random, tiles))
                .with_input(residual, random_format(&mut random, tiles))
                .with_input(query, random_format(&mut random, tiles))
                .with_input(key, random_format(&mut random, tiles))
                .with_input(attention_value, random_format(&mut random, tiles));
            config.operator_candidates = vec![
                OperatorCandidate::new(
                    MidOperator::Gelu,
                    [OperandRequirement::new(gelu_input.clone(), 8)],
                    OperandRequirement::new(gelu_output.clone(), 8),
                )
                .with_output_aliasing(OutputAliasing::MayAliasInputs(vec![0])),
                OperatorCandidate::new(
                    MidOperator::Add(AddOptions::default()),
                    [
                        OperandRequirement::new(add_left.clone(), 8),
                        OperandRequirement::new(add_right.clone(), 8),
                    ],
                    OperandRequirement::new(add_output.clone(), 8),
                )
                .with_output_aliasing(OutputAliasing::MayAliasInputs(vec![0])),
                OperatorCandidate::new(
                    MidOperator::FlashAttention {
                        options: AttentionOptions::default(),
                        accumulate: attention_accumulate,
                    },
                    [
                        OperandRequirement::new(attention_query.clone(), 8),
                        OperandRequirement::new(attention_key.clone(), 8),
                        OperandRequirement::new(attention_value_format.clone(), 8),
                    ],
                    OperandRequirement::new(attention_output.clone(), 8),
                ),
            ];

            let lowered = lower(&graph, &config, &ToyCostModel).unwrap();
            let operators = lowered
                .operations
                .iter()
                .filter(|operation| matches!(operation.kind, MidOperationKind::Operator(_)))
                .collect::<Vec<_>>();
            assert_eq!(operators.len(), 3, "random case {case}");
            assert!(matches!(
                operators[0].kind,
                MidOperationKind::Operator(MidOperator::Gelu)
            ));
            assert_operator_signature(&lowered, operators[0], &[gelu_input], gelu_output.clone());
            assert_eq!(
                operators[0]
                    .operator_plan
                    .as_ref()
                    .unwrap()
                    .requirements
                    .output_aliasing,
                OutputAliasing::MayAliasInputs(vec![0])
            );
            assert!(matches!(
                operators[1].kind,
                MidOperationKind::Operator(MidOperator::Add(_))
            ));
            assert_operator_signature(&lowered, operators[1], &[add_left, add_right], add_output);
            assert_eq!(
                operators[1]
                    .operator_plan
                    .as_ref()
                    .unwrap()
                    .requirements
                    .output_aliasing,
                OutputAliasing::MayAliasInputs(vec![0])
            );
            assert!(matches!(
                operators[2].kind,
                MidOperationKind::Operator(MidOperator::FlashAttention { accumulate, .. })
                    if accumulate == attention_accumulate
            ));
            assert_operator_signature(
                &lowered,
                operators[2],
                &[attention_query, attention_key, attention_value_format],
                attention_output,
            );
            assert_conversions_are_explicit(&lowered, &lowered.operations);
        }
    }

    #[test]
    fn randomized_repeat_lowering_retains_sequences_without_unrolling() {
        let mut random = fastrand::Rng::with_seed(0x7265_7065);
        for case in 0..RANDOM_CASES {
            let size = dimension(&mut random);
            let count = random.u32(1..=12);
            let tiles = random.u16(1..=64);
            let layout = Layout::row_sharded(tiles);
            let carried_format = format(precision(&mut random), layout.clone());
            let mut graph = ComputeGraph::new();
            let carried = graph.host_input("state", [size, size]).unwrap();
            let weights = (0..count)
                .map(|index| graph.parameter(format!("weight.{index}"), [size, size]))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let sequence = graph.value_sequence("weights", weights.clone()).unwrap();
            let output = graph
                .repeat(count, [carried], [], [sequence], |body, arguments| {
                    Ok(vec![
                        body.gemm(arguments.carried[0], arguments.iterated[0])?,
                    ])
                })
                .unwrap()[0];
            graph.set_outputs([output]).unwrap();
            let mut config = PipelineConfig::new(tiles).with_input(carried, carried_format.clone());
            for weight in weights {
                config
                    .inputs
                    .insert(weight, format(precision(&mut random), layout.clone()));
            }

            let lowered = lower(&graph, &config, &ToyCostModel).unwrap();
            let repeat = lowered
                .operations
                .iter()
                .find_map(|operation| match &operation.kind {
                    MidOperationKind::Repeat(repeat) => Some(repeat),
                    _ => None,
                })
                .unwrap();
            assert_eq!(repeat.count, count, "random case {case}");
            assert_eq!(repeat.iterated_inputs.len(), 1, "random case {case}");
            assert_eq!(
                repeat.iterated_inputs[0].len(),
                count as usize,
                "random case {case}"
            );
            let sequence_format = &value(&lowered, repeat.iterated_inputs[0][0])
                .tensor_type
                .format;
            assert!(repeat.iterated_inputs[0].iter().all(|value_id| {
                &value(&lowered, *value_id).tensor_type.format == sequence_format
            }));
            assert_eq!(
                &value(&lowered, repeat.body.yields[0]).tensor_type.format,
                &carried_format,
                "random case {case}"
            );
            assert_eq!(
                &value(&lowered, lowered.outputs[0]).tensor_type.format,
                &carried_format,
                "random case {case}"
            );
            assert_conversions_are_explicit(&lowered, &lowered.operations);
            assert_conversions_are_explicit(&lowered, &repeat.body.operations);
        }
    }
}
