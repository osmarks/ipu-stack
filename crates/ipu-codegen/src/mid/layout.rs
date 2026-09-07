//! Tensor formats, ownership, padding, and logical shard geometry.

use super::*;

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

/// AMP packing role. Block dimensions are recorded by [`AxisTiling`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AmpOrder {
    Left,
    /// A semantic `[K, N]` matrix packed as the left operand `[N, K]`.
    TransposedLeft,
    /// Semantic `[key, channel]` storage packed as the right operand of
    /// `query * key.transpose()`.
    TransposedRight,
    Output,
    /// A semantic `[M, N]` output packed as the physical output `[N, M]`.
    TransposedOutput,
}

/// Ordinary matrix elements grouped into contiguous rectangular blocks.
///
/// Unlike [`AmpOrder`], this is an SRAM storage layout rather than an AMP
/// operand micro-layout. Kernels route each naturally ordered group into the
/// required AMP register slots with `ld*putcs` destination permutations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BlockMajorOrder {
    /// The final two semantic axes are `[rows, columns]`.
    Matrix { row_block: u16, column_block: u16 },
    /// The final two semantic axes are stored as `[columns, rows]`.
    TransposedMatrix { row_block: u16, column_block: u16 },
}

pub const AMP_INNER_BLOCK: u32 = 64;
pub(super) const AMP_NARROW_OUTPUT_COLUMN_BLOCK: u32 = 32;
pub const AMP_OUTPUT_COLUMN_BLOCK: u32 = 64;
pub(super) const AMP_WIDE_OUTPUT_COLUMN_BLOCK: u32 = 128;
pub const AMP_COLUMN_MICRO: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ElementOrder {
    RowMajor,
    BlockMajor(BlockMajorOrder),
    Amp(AmpOrder),
}

/// Physical traversal within one 16-by-16 matrix fragment. Layouts
/// with the same order can exchange whole panels while changing their outer
/// ownership and panel sequence, without an intermediate rearrangement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MicroPanelOrder {
    RowsThenColumns,
    ColumnsThenRows,
}

impl ElementOrder {
    /// Number of rows in each pair of F16 panels regrouped by an FP8 cast.
    /// A zero result denotes an order whose linearization is precision-independent.
    pub(crate) fn fp8_cast_panel_rows(self, rows: u64, columns: u64) -> u64 {
        match self {
            Self::Amp(AmpOrder::Left) => rows,
            Self::Amp(AmpOrder::TransposedLeft) => columns,
            Self::Amp(AmpOrder::TransposedRight) => 16,
            Self::BlockMajor(
                BlockMajorOrder::Matrix { column_block, .. }
                | BlockMajorOrder::TransposedMatrix { column_block, .. },
            ) => u64::from(column_block),
            _ => 0,
        }
    }

    /// GEMM stores complete 16-row micro-panels in this column-group order.
    pub(crate) fn gemm_output_group(self) -> Option<u32> {
        match self {
            Self::BlockMajor(
                BlockMajorOrder::Matrix {
                    row_block,
                    column_block: 16,
                }
                | BlockMajorOrder::TransposedMatrix {
                    row_block,
                    column_block: 16,
                },
            ) if row_block >= 16 && row_block.is_power_of_two() => Some(u32::from(row_block)),
            _ => None,
        }
    }

    pub(crate) fn gemm_output_transposed(self) -> bool {
        matches!(
            self,
            Self::Amp(AmpOrder::TransposedLeft | AmpOrder::TransposedOutput)
                | Self::BlockMajor(BlockMajorOrder::Matrix { .. })
        )
    }

    /// This packing is consumed as contiguous K-major panels, while a generic
    /// intersection rearrangement produces rectangular tensor-coordinate
    /// views. It must therefore be selected for an automatic input or produced
    /// by a specialized operator/local staging path.
    pub(super) fn requires_direct_population(&self) -> bool {
        matches!(
            self,
            Self::BlockMajor(BlockMajorOrder::TransposedMatrix { .. })
                | Self::Amp(AmpOrder::TransposedRight)
        )
    }

    /// Whether a row-major logical staging shard can be transformed locally
    /// into this order by the generated conversion kernels.
    pub(super) fn supports_row_major_population(self) -> bool {
        matches!(
            self,
            Self::RowMajor
                | Self::BlockMajor(BlockMajorOrder::Matrix { .. })
                | Self::Amp(AmpOrder::Left | AmpOrder::TransposedRight)
        )
    }

    pub(crate) const fn micro_panel_order(self) -> Option<MicroPanelOrder> {
        match self {
            Self::Amp(AmpOrder::Left | AmpOrder::TransposedRight)
            | Self::BlockMajor(BlockMajorOrder::TransposedMatrix { .. }) => {
                Some(MicroPanelOrder::RowsThenColumns)
            }
            Self::Amp(AmpOrder::TransposedLeft)
            | Self::BlockMajor(BlockMajorOrder::Matrix { .. }) => {
                Some(MicroPanelOrder::ColumnsThenRows)
            }
            Self::RowMajor | Self::Amp(AmpOrder::Output | AmpOrder::TransposedOutput) => None,
        }
    }

    /// Smallest column span which remains a self-contained physical fragment
    /// when canonical linear ownership divides a matrix into row segments.
    pub(crate) fn retained_linear_column_grain(self, precision: Precision) -> Option<u32> {
        match self {
            Self::RowMajor => Some(1),
            Self::Amp(AmpOrder::Left) => Some(match precision {
                Precision::F8F143 { .. } => 32,
                Precision::F16 => 16,
                Precision::F32 => 8,
            }),
            Self::Amp(AmpOrder::Output) => Some(AMP_COLUMN_MICRO),
            Self::BlockMajor(_) | Self::Amp(_) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryClass {
    Ipu21Standard,
    Ipu21Interleaved,
}

/// Maximum per-tile bytes attributed to each address/load class. The classes
/// share physical tile SRAM, so feasibility must check both the individual
/// interleaved-region limit and their combined size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorAxis {
    FromStart(u16),
    FromEnd(u16),
    /// Grain-aligned ownership intervals in canonical logical element order.
    Linear,
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
    /// Number of equal semantic groups which are padded independently before
    /// partitioning. Partitions must subdivide groups evenly. This keeps, for
    /// example, attention-head boundaries intact without coupling the number
    /// of column shards to the number of heads.
    pub padding_groups: u16,
    /// Required physical block multiple. One imposes no blocking constraint.
    pub block_size: u32,
    /// Physical extent multiple, independently of the grain distributed
    /// between partitions. This permits fine-grained ownership of an axis
    /// whose producer operates on wider padded blocks.
    pub padding_multiple: u32,
    /// Physical extent multiple applied independently to every partition.
    /// Unlike `padding_multiple`, this does not change semantic partition
    /// boundaries: padding belongs to the allocation owned by that partition.
    pub shard_padding_multiple: u32,
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
            padding_groups: 1,
            block_size,
            padding_multiple: block_size,
            shard_padding_multiple: 1,
            padding,
            tile_stride: None,
        }
    }

    pub const fn with_tile_stride(mut self, tile_stride: u16) -> Self {
        self.tile_stride = Some(tile_stride);
        self
    }

    pub const fn with_padding_multiple(mut self, padding_multiple: u32) -> Self {
        self.padding_multiple = padding_multiple;
        self
    }

    pub const fn with_shard_padding_multiple(mut self, shard_padding_multiple: u32) -> Self {
        self.shard_padding_multiple = shard_padding_multiple;
        self
    }

    pub const fn with_padding_groups(mut self, padding_groups: u16) -> Self {
        self.padding_groups = padding_groups;
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
    pub fn linear(tile_count: u16, grain: u32) -> Self {
        Self {
            tile_count,
            replicas: 1,
            axes: vec![AxisTiling::new(
                TensorAxis::Linear,
                tile_count,
                grain,
                Padding::Reject,
            )],
        }
    }

    pub fn linear_grain(&self) -> Option<u32> {
        match self.axes.as_slice() {
            [axis] if axis.axis == TensorAxis::Linear => Some(axis.block_size),
            _ => None,
        }
    }

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
}

/// Layout decisions which constrain operators and exchange generation without
/// assigning physical tile identities or SRAM addresses.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Layout {
    pub order: ElementOrder,
    pub tiling: TensorTiling,
    pub memory_class: MemoryClass,
}

/// Half-open bounds along one tensor axis. `physical_end` includes zero padding;
/// `logical_end` excludes it. An empty shard uses `logical_end == start`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShardExtent {
    pub axis: u16,
    pub start: u32,
    pub logical_end: u32,
    pub physical_end: u32,
}

impl Layout {
    pub(crate) fn validate_tile_count(&self, available: u16) -> Result<(), LayoutError> {
        let declared = self.tiling.tile_count;
        if declared == 0 {
            return Err(LayoutError::EmptyTileGroup);
        }
        if declared > available {
            return Err(LayoutError::TileCapacity {
                declared,
                available,
            });
        }
        Ok(())
    }

    /// Logical ownership and padded bounds, independent of physical placement.
    pub fn shard_extents(
        &self,
        shape: &TensorShape,
    ) -> Result<Vec<(u16, Vec<ShardExtent>)>, LayoutError> {
        self.resolve(shape)?.shard_extents()
    }

    pub fn row_major(tiling: TensorTiling) -> Self {
        Self {
            order: ElementOrder::RowMajor,
            tiling,
            memory_class: MemoryClass::Ipu21Standard,
        }
    }

    pub fn logical_linear(tile_count: u16, grain: u32) -> Self {
        Self::row_major(TensorTiling::linear(tile_count, grain))
    }

    pub fn with_retained_order_linear_ownership(&self, tile_count: u16, grain: u32) -> Self {
        Self {
            order: self.order,
            tiling: TensorTiling::linear(tile_count, grain),
            memory_class: self.memory_class,
        }
    }

    pub fn row_sharded(tile_count: u16) -> Self {
        Self::row_major(TensorTiling::sharded(TensorAxis::FromEnd(2), tile_count))
    }

    pub fn head_sharded(tile_count: u16) -> Self {
        Self::row_major(TensorTiling::sharded(TensorAxis::FromEnd(3), tile_count))
    }

    pub(super) fn attention_tiling(heads: u16, query_partitions: u16) -> TensorTiling {
        TensorTiling {
            tile_count: heads.saturating_mul(query_partitions),
            replicas: 1,
            axes: vec![
                AxisTiling::new(TensorAxis::FromEnd(2), query_partitions, 1, Padding::Reject)
                    .with_tile_stride(heads),
                AxisTiling::new(TensorAxis::FromEnd(3), heads, 1, Padding::Reject)
                    .with_tile_stride(1),
            ],
        }
    }

    pub fn attention_query(heads: u16, query_partitions: u16) -> Self {
        let mut layout = Self::attention_output(heads, query_partitions);
        layout.order = ElementOrder::Amp(AmpOrder::Left);
        layout
    }

    pub fn attention_key(heads: u16, key_partitions: u16) -> Self {
        let axes = vec![
            AxisTiling::new(TensorAxis::FromEnd(3), heads, 1, Padding::Reject).with_tile_stride(1),
            AxisTiling::new(
                TensorAxis::FromEnd(2),
                key_partitions,
                AMP_INNER_BLOCK,
                Padding::Zero,
            )
            .with_tile_stride(heads),
            AxisTiling::new(TensorAxis::FromEnd(1), 1, AMP_COLUMN_MICRO, Padding::Zero),
        ];
        Self {
            order: ElementOrder::Amp(AmpOrder::TransposedRight),
            tiling: TensorTiling {
                tile_count: heads.saturating_mul(key_partitions),
                replicas: 1,
                axes,
            },
            memory_class: MemoryClass::Ipu21Standard,
        }
    }

    pub fn attention_block_major_key_value(heads: u16, key_partitions: u16) -> Self {
        let mut layout = Self::attention_key(heads, key_partitions);
        layout.order = ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
            row_block: AMP_INNER_BLOCK as u16,
            column_block: AMP_COLUMN_MICRO as u16,
        });
        layout
    }

    pub fn attention_output(heads: u16, query_partitions: u16) -> Self {
        let mut tiling = Self::attention_tiling(heads, query_partitions);
        tiling.axes.push(AxisTiling::new(
            TensorAxis::FromEnd(1),
            1,
            AMP_COLUMN_MICRO,
            Padding::Zero,
        ));
        Self::row_major(tiling)
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

    pub fn block_major_matrix(row_block: u16, tile_count: u16) -> Self {
        Self::block_major_matrix_storage(
            row_block,
            AMP_OUTPUT_COLUMN_BLOCK,
            tile_count,
            1,
            1,
            MemoryClass::Ipu21Standard,
        )
    }

    pub fn amp_output(tile_count: u16) -> Self {
        let mut layout = Self::amp_left(AMP_OUTPUT_COLUMN_BLOCK as u16, tile_count);
        layout.order = ElementOrder::Amp(AmpOrder::Output);
        layout.memory_class = MemoryClass::Ipu21Interleaved;
        layout
    }

    /// F16 AMP result stored in the same within-panel order as a following
    /// left operand. The GEMM coefficient routing makes the native accumulator
    /// drain land in this order without a post-compute permutation.
    pub fn amp_left_result(tile_count: u16) -> Self {
        let mut layout = Self::amp_output(tile_count);
        layout.order = ElementOrder::Amp(AmpOrder::Left);
        layout
    }

    /// AMP left operand on a row-by-column tile grid. The row shard is
    /// replicated across column groups so it is local to every output shard.
    pub fn amp_left_grid(
        inner: u16,
        tile_count: u16,
        row_partitions: u16,
        column_partitions: u16,
        grid_order: GridOrder,
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
                    AxisTiling::new(TensorAxis::FromEnd(2), row_partitions, 1, Padding::Reject)
                        .with_tile_stride(match grid_order {
                            GridOrder::ColumnsFast => column_partitions,
                            GridOrder::RowsFast => 1,
                        }),
                    AxisTiling::new(TensorAxis::FromEnd(1), 1, u32::from(inner), Padding::Zero),
                ],
            },
            memory_class: MemoryClass::Ipu21Standard,
        }
    }

    /// AMP left operand for a row-by-column-by-K dispatch grid. K and rows
    /// are true shards; the column coordinate is a replica because the same
    /// activation range is consumed by each output-column group.
    pub fn amp_left_parallel_grid(
        inner: u16,
        tile_count: u16,
        row_partitions: u16,
        column_partitions: u16,
        inner_partitions: u16,
    ) -> Self {
        Self {
            order: ElementOrder::Amp(AmpOrder::Left),
            tiling: TensorTiling {
                tile_count,
                replicas: column_partitions,
                axes: vec![
                    AxisTiling::new(
                        TensorAxis::FromEnd(1),
                        inner_partitions,
                        u32::from(inner),
                        Padding::Zero,
                    ),
                    AxisTiling::new(TensorAxis::FromEnd(2), row_partitions, 1, Padding::Reject),
                ],
            },
            memory_class: MemoryClass::Ipu21Standard,
        }
    }

    /// A semantic right-hand matrix `[K, N]` packed as the physical left
    /// operand `[N, K]` on a row-by-column-by-K dispatch grid.
    pub fn amp_transposed_left_parallel_grid(
        inner: u16,
        tile_count: u16,
        row_partitions: u16,
        column_partitions: u16,
        inner_partitions: u16,
    ) -> Self {
        Self::amp_left_parallel_grid(
            inner,
            tile_count,
            row_partitions,
            column_partitions,
            inner_partitions,
        )
        .transpose_matrix_axes(ElementOrder::Amp(AmpOrder::TransposedLeft))
    }

    /// Block-major matrix storage on a row-by-column tile grid. Each column
    /// shard is replicated across row groups so it is local to every consumer.
    pub fn block_major_matrix_grid(
        inner: u16,
        output_column_block: u32,
        tile_count: u16,
        row_partitions: u16,
        column_partitions: u16,
        grid_order: GridOrder,
    ) -> Self {
        Self {
            order: ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                row_block: inner,
                column_block: AMP_COLUMN_MICRO as u16,
            }),
            tiling: TensorTiling {
                tile_count,
                replicas: row_partitions,
                axes: vec![
                    AxisTiling::new(TensorAxis::FromEnd(2), 1, u32::from(inner), Padding::Zero),
                    AxisTiling::new(
                        TensorAxis::FromEnd(1),
                        column_partitions,
                        output_column_block,
                        Padding::Zero,
                    )
                    .with_tile_stride(match grid_order {
                        GridOrder::ColumnsFast => 1,
                        GridOrder::RowsFast => row_partitions,
                    }),
                ],
            },
            memory_class: MemoryClass::Ipu21Standard,
        }
    }

    /// Matrix storage with complete row-by-column blocks contiguous in the
    /// selected memory class. Column and row sharding select the owner set;
    /// `copies` controls persistent replication independently of consumers.
    pub fn block_major_matrix_storage(
        inner_block: u16,
        output_column_block: u32,
        column_partitions: u16,
        inner_partitions: u16,
        copies: u16,
        memory_class: MemoryClass,
    ) -> Self {
        let tile_count = column_partitions
            .checked_mul(inner_partitions)
            .and_then(|tiles| tiles.checked_mul(copies))
            .unwrap_or(0);
        Self {
            order: ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                row_block: inner_block,
                column_block: AMP_COLUMN_MICRO as u16,
            }),
            tiling: TensorTiling {
                tile_count,
                replicas: copies,
                axes: vec![
                    AxisTiling::new(
                        TensorAxis::FromEnd(1),
                        column_partitions,
                        output_column_block,
                        Padding::Zero,
                    )
                    .with_tile_stride(1),
                    AxisTiling::new(
                        TensorAxis::FromEnd(2),
                        inner_partitions,
                        u32::from(inner_block),
                        Padding::Zero,
                    ),
                ],
            },
            memory_class,
        }
    }

    /// A semantic matrix `[M, K]` stored in transposed contiguous blocks.
    pub fn transposed_block_major_matrix_storage(
        inner_block: u16,
        output_column_block: u32,
        column_partitions: u16,
        inner_partitions: u16,
        copies: u16,
        memory_class: MemoryClass,
    ) -> Self {
        Self::block_major_matrix_storage(
            inner_block,
            output_column_block,
            column_partitions,
            inner_partitions,
            copies,
            memory_class,
        )
        .transpose_matrix_axes(ElementOrder::BlockMajor(
            BlockMajorOrder::TransposedMatrix {
                row_block: inner_block,
                column_block: AMP_COLUMN_MICRO as u16,
            },
        ))
    }

    /// AMP output distributed over both matrix axes on one tile grid.
    pub fn amp_output_grid(
        output_column_block: u32,
        tile_count: u16,
        row_partitions: u16,
        column_partitions: u16,
        grid_order: GridOrder,
    ) -> Self {
        if output_column_block == AMP_OUTPUT_COLUMN_BLOCK
            && column_partitions == 1
            && row_partitions == tile_count
        {
            return Self::amp_output(tile_count);
        }
        Self::amp_output_grid_storage(
            output_column_block,
            tile_count,
            row_partitions,
            column_partitions,
            grid_order,
        )
    }

    fn amp_output_grid_storage(
        output_column_block: u32,
        tile_count: u16,
        row_partitions: u16,
        column_partitions: u16,
        grid_order: GridOrder,
    ) -> Self {
        Self {
            order: ElementOrder::Amp(AmpOrder::Output),
            tiling: TensorTiling {
                tile_count,
                replicas: 1,
                axes: vec![
                    AxisTiling::new(
                        TensorAxis::FromEnd(1),
                        column_partitions,
                        output_column_block,
                        Padding::Zero,
                    )
                    .with_tile_stride(match grid_order {
                        GridOrder::ColumnsFast => 1,
                        GridOrder::RowsFast => row_partitions,
                    }),
                    AxisTiling::new(TensorAxis::FromEnd(2), row_partitions, 1, Padding::Reject)
                        .with_tile_stride(match grid_order {
                            GridOrder::ColumnsFast => column_partitions,
                            GridOrder::RowsFast => 1,
                        }),
                ],
            },
            memory_class: MemoryClass::Ipu21Interleaved,
        }
    }

    pub fn amp_left_result_grid(
        output_column_block: u32,
        tile_count: u16,
        row_partitions: u16,
        column_partitions: u16,
        grid_order: GridOrder,
    ) -> Self {
        let mut layout = Self::amp_output_grid(
            output_column_block,
            tile_count,
            row_partitions,
            column_partitions,
            grid_order,
        );
        layout.order = ElementOrder::Amp(AmpOrder::Left);
        layout
    }

    /// A semantic output `[M, N]` packed as the physical AMP output `[N, M]`.
    pub fn amp_transposed_output_grid(
        output_column_block: u32,
        tile_count: u16,
        row_partitions: u16,
        column_partitions: u16,
        grid_order: GridOrder,
    ) -> Self {
        Self::amp_output_grid_storage(
            output_column_block,
            tile_count,
            row_partitions,
            column_partitions,
            grid_order,
        )
        .transpose_matrix_axes(ElementOrder::Amp(AmpOrder::TransposedOutput))
    }

    /// Reinterpret a physical matrix layout with its two logical matrix axes swapped.
    /// The caller supplies the corresponding transposed physical element order.
    fn transpose_matrix_axes(mut self, order: ElementOrder) -> Self {
        for axis in &mut self.tiling.axes {
            axis.axis = match axis.axis {
                TensorAxis::FromEnd(1) => TensorAxis::FromEnd(2),
                TensorAxis::FromEnd(2) => TensorAxis::FromEnd(1),
                other => other,
            };
        }
        self.order = order;
        self
    }

    pub fn amp_transposed_left_result_grid(
        output_column_block: u32,
        tile_count: u16,
        row_partitions: u16,
        column_partitions: u16,
        grid_order: GridOrder,
    ) -> Self {
        let mut layout = Self::amp_transposed_output_grid(
            output_column_block,
            tile_count,
            row_partitions,
            column_partitions,
            grid_order,
        );
        layout.order = ElementOrder::Amp(AmpOrder::TransposedLeft);
        layout
    }

    /// AMP output storage sharded by rows and replicated across column groups.
    pub fn amp_output_replicated_grid(
        tile_count: u16,
        row_partitions: u16,
        column_replicas: u16,
    ) -> Self {
        if column_replicas == 1 && row_partitions == tile_count {
            return Self::amp_output(tile_count);
        }
        Self {
            order: ElementOrder::Amp(AmpOrder::Output),
            tiling: TensorTiling {
                tile_count,
                replicas: column_replicas,
                axes: vec![
                    AxisTiling::new(TensorAxis::FromEnd(2), row_partitions, 1, Padding::Reject),
                    AxisTiling::new(TensorAxis::FromEnd(1), 1, 64, Padding::Zero),
                ],
            },
            memory_class: MemoryClass::Ipu21Interleaved,
        }
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum LayoutError {
    #[error("layout requires {declared} tiles but the graph has {available}")]
    TileCapacity { declared: u16, available: u16 },
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
    #[error("shard extent {extent} is not divisible by block size {block_size}")]
    IndivisibleShard { extent: u32, block_size: u32 },
    #[error(
        "axis extent {extent} cannot be divided into {groups} padding groups and {partitions} partitions"
    )]
    InvalidPaddingGroups {
        groups: u16,
        partitions: u16,
        extent: u32,
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

impl TensorFormat {
    pub(crate) fn supports_micro_panel_exchange(&self, destination: &Self) -> bool {
        matches!(self.precision, Precision::F16 | Precision::F8F143 { .. })
            && destination.precision == self.precision
            && self.layout.order.micro_panel_order().is_some()
            && self.layout.order.micro_panel_order() == destination.layout.order.micro_panel_order()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorType {
    pub shape: TensorShape,
    pub format: TensorFormat,
}

impl TensorType {
    /// Keep producer ownership while completing FP8's 32-element panels.
    pub(crate) fn fp8_cast_layout(&self) -> Option<Layout> {
        let mut layout = self.format.layout.clone();
        let axis_from_end = match layout.order {
            ElementOrder::Amp(AmpOrder::Left) => {
                if layout.resolve(&self.shape).ok().is_some_and(|resolved| {
                    resolved
                        .axes()
                        .and_then(|axes| axes.last())
                        .is_some_and(|axis| axis.extents_are_multiple_of(32))
                }) {
                    return Some(layout);
                }
                // A narrow final tail is harmless, but padding every producer
                // panel would turn a bulk exchange into strided short packets.
                if !layout.resolve(&self.shape).ok().is_some_and(|resolved| {
                    resolved
                        .axes()
                        .and_then(|axes| axes.last())
                        .is_some_and(|axis| axis.complete_panels_except_tail(32))
                }) {
                    return None;
                }
                let axis = layout.tiling.axes.iter_mut().find(|axis| {
                    matches!(axis.axis, TensorAxis::FromEnd(1))
                        || axis.axis == TensorAxis::FromStart((self.shape.0.len() - 1) as u16)
                })?;
                axis.shard_padding_multiple = axis.shard_padding_multiple.max(32);
                return Some(layout);
            }
            ElementOrder::RowMajor
            | ElementOrder::Amp(AmpOrder::Output | AmpOrder::TransposedOutput) => {
                return Some(layout);
            }
            ElementOrder::Amp(AmpOrder::TransposedRight) => 1,
            ElementOrder::Amp(AmpOrder::TransposedLeft) => 2,
            ElementOrder::BlockMajor(
                BlockMajorOrder::Matrix { row_block, .. }
                | BlockMajorOrder::TransposedMatrix { row_block, .. },
            ) => return row_block.is_multiple_of(32).then_some(layout),
        };
        let valid = layout.resolve(&self.shape).ok().is_some_and(|resolved| {
            resolved.axes().is_some_and(|axes| {
                axes.len()
                    .checked_sub(axis_from_end)
                    .is_some_and(|axis| axes[axis].extents_are_multiple_of(32))
            })
        });
        valid.then_some(layout)
    }

    pub fn new(shape: impl IntoIterator<Item = u32>, precision: Precision, layout: Layout) -> Self {
        Self {
            shape: TensorShape::new(shape),
            format: TensorFormat { precision, layout },
        }
    }
}
