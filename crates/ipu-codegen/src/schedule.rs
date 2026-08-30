//! Parametric whole-device work selected by mid-level planning.

use crate::{
    GemmGeometry, GemmKernelFamily, GemmOrientation, GridOrder, Layout, MidOperator,
    NativeKernelOrder, OperatorRequirements, ReductionStaging, StorageOrder, TensorFormat,
    TensorType, TileKernelSpec,
};

/// A value consumed or produced by an operator schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScheduleValue {
    Input(u16),
    Temporary(u16),
    Output,
}

/// How a mapped kernel obtains one input for each output shard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleAccess {
    /// Select every logical intersection, including singleton broadcasting.
    LogicalOverlap,
    /// Consume the corresponding shard already resident on the kernel tile.
    TileLocal,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelMap {
    pub kernel: TileKernelSpec,
    pub inputs: Vec<(ScheduleValue, ScheduleAccess)>,
    pub output: ScheduleValue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmMap {
    pub inputs: [ScheduleValue; 2],
    pub output: ScheduleValue,
    pub kernel: GemmKernelFamily,
    pub geometry: GemmGeometry,
}

impl GemmMap {
    /// Tensor produced by the compute grid before a parallel K reduction.
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
            Self::Materialized { .. } => crate::AMP_INNER_BLOCK,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionMap {
    pub inputs: [ScheduleValue; 3],
    pub output: ScheduleValue,
    pub kernel: GemmKernelFamily,
    pub blocking: AttentionBlocking,
    pub query_dimension: u32,
    pub value_dimension: u32,
}

impl AttentionMap {
    pub fn gemm_blocks(&self) -> [crate::GemmBlockShape; 2] {
        let key_columns = match self.blocking {
            AttentionBlocking::Materialized {
                padded_key_rows, ..
            } => padded_key_rows,
            blocking => blocking.key_block_rows(),
        };
        [
            crate::GemmBlockShape {
                inner: self.query_dimension,
                output_columns: key_columns,
            },
            crate::GemmBlockShape {
                inner: key_columns,
                output_columns: self.value_dimension,
            },
        ]
    }
}

/// Ordered, symbolic whole-device work. Placement binds values to concrete
/// shards and tiles; low lowering only materializes the selected steps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScheduleStep {
    KernelMap(KernelMap),
    Gemm(GemmMap),
    Reduce {
        input: ScheduleValue,
        output: ScheduleValue,
        staging: ReductionStaging,
    },
    Attention(AttentionMap),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatorSchedule {
    pub operator: MidOperator,
    pub steps: Vec<ScheduleStep>,
    pub requirements: OperatorRequirements,
}
