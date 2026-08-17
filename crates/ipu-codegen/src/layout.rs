use crate::graph::TensorShape;
use crate::operator::{GridOrder, Precision};
use std::ops::Range;

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
pub(crate) const AMP_NARROW_OUTPUT_COLUMN_BLOCK: u32 = 32;
pub const AMP_OUTPUT_COLUMN_BLOCK: u32 = 64;
pub(crate) const AMP_WIDE_OUTPUT_COLUMN_BLOCK: u32 = 128;
pub const AMP_COLUMN_MICRO: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ElementOrder {
    RowMajor,
    BlockMajor(BlockMajorOrder),
    Amp(AmpOrder),
}

/// Physical traversal within one 16-by-16 F16 matrix micro-panel. Layouts
/// with the same order can exchange whole panels while changing their outer
/// ownership and panel sequence, without an intermediate rearrangement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum F16MicroPanelOrder {
    RowsThenColumns,
    ColumnsThenRows,
}

impl ElementOrder {
    /// This packing is consumed as contiguous K-major panels, while a generic
    /// intersection rearrangement produces rectangular tensor-coordinate
    /// views. It must therefore be selected for an automatic input or produced
    /// by a specialized operator/local staging path.
    pub(crate) fn requires_direct_population(&self) -> bool {
        matches!(
            self,
            Self::BlockMajor(BlockMajorOrder::TransposedMatrix { .. })
                | Self::Amp(AmpOrder::TransposedRight)
        )
    }

    /// Whether a row-major logical staging shard can be transformed locally
    /// into this order by the generated conversion kernels.
    pub(crate) fn supports_row_major_population(self) -> bool {
        matches!(
            self,
            Self::RowMajor
                | Self::BlockMajor(BlockMajorOrder::Matrix { .. })
                | Self::Amp(AmpOrder::Left | AmpOrder::TransposedRight)
        )
    }

    pub(crate) const fn f16_micro_panel_order(self) -> Option<F16MicroPanelOrder> {
        match self {
            Self::Amp(AmpOrder::Left | AmpOrder::TransposedRight)
            | Self::BlockMajor(BlockMajorOrder::TransposedMatrix { .. }) => {
                Some(F16MicroPanelOrder::RowsThenColumns)
            }
            Self::Amp(AmpOrder::TransposedLeft)
            | Self::BlockMajor(BlockMajorOrder::Matrix { .. }) => {
                Some(F16MicroPanelOrder::ColumnsThenRows)
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
    Standard,
    Interleaved,
}

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
            memory_class: MemoryClass::Standard,
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

    fn attention_tiling(heads: u16, query_partitions: u16) -> TensorTiling {
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
        let mut tiling = Self::attention_tiling(heads, query_partitions);
        tiling.axes.push(AxisTiling::new(
            TensorAxis::FromEnd(1),
            1,
            AMP_COLUMN_MICRO,
            Padding::Zero,
        ));
        Self {
            order: ElementOrder::Amp(AmpOrder::Left),
            tiling,
            memory_class: MemoryClass::Standard,
        }
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
            memory_class: MemoryClass::Standard,
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
            memory_class: MemoryClass::Standard,
        }
    }

    pub fn block_major_matrix(row_block: u16, tile_count: u16) -> Self {
        Self::block_major_matrix_storage(
            row_block,
            AMP_OUTPUT_COLUMN_BLOCK,
            tile_count,
            1,
            1,
            MemoryClass::Standard,
        )
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
            memory_class: MemoryClass::Interleaved,
        }
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
            memory_class: MemoryClass::Standard,
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
            memory_class: MemoryClass::Standard,
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
        Self {
            order: ElementOrder::Amp(AmpOrder::TransposedLeft),
            tiling: TensorTiling {
                tile_count,
                replicas: column_partitions,
                axes: vec![
                    AxisTiling::new(
                        TensorAxis::FromEnd(2),
                        inner_partitions,
                        u32::from(inner),
                        Padding::Zero,
                    ),
                    AxisTiling::new(TensorAxis::FromEnd(1), row_partitions, 1, Padding::Reject),
                ],
            },
            memory_class: MemoryClass::Standard,
        }
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
            memory_class: MemoryClass::Standard,
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
        let tile_count = column_partitions
            .checked_mul(inner_partitions)
            .and_then(|tiles| tiles.checked_mul(copies))
            .unwrap_or(0);
        Self {
            order: ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix {
                row_block: inner_block,
                column_block: AMP_COLUMN_MICRO as u16,
            }),
            tiling: TensorTiling {
                tile_count,
                replicas: copies,
                axes: vec![
                    AxisTiling::new(
                        TensorAxis::FromEnd(2),
                        column_partitions,
                        output_column_block,
                        Padding::Zero,
                    )
                    .with_tile_stride(1),
                    AxisTiling::new(
                        TensorAxis::FromEnd(1),
                        inner_partitions,
                        u32::from(inner_block),
                        Padding::Zero,
                    ),
                ],
            },
            memory_class,
        }
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
            memory_class: MemoryClass::Interleaved,
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
        Self {
            order: ElementOrder::Amp(AmpOrder::TransposedOutput),
            tiling: TensorTiling {
                tile_count,
                replicas: 1,
                axes: vec![
                    AxisTiling::new(
                        TensorAxis::FromEnd(2),
                        column_partitions,
                        output_column_block,
                        Padding::Zero,
                    )
                    .with_tile_stride(match grid_order {
                        GridOrder::ColumnsFast => 1,
                        GridOrder::RowsFast => row_partitions,
                    }),
                    AxisTiling::new(TensorAxis::FromEnd(1), row_partitions, 1, Padding::Reject)
                        .with_tile_stride(match grid_order {
                            GridOrder::ColumnsFast => column_partitions,
                            GridOrder::RowsFast => 1,
                        }),
                ],
            },
            memory_class: MemoryClass::Interleaved,
        }
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
            memory_class: MemoryClass::Interleaved,
        }
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
    #[error("tensor rank {0} cannot be represented by shard axis identifiers")]
    RankTooLarge(usize),
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
    pub(crate) fn supports_f16_micro_panel_exchange(&self, destination: &Self) -> bool {
        self.precision == Precision::F16
            && destination.precision == Precision::F16
            && self.layout.order.f16_micro_panel_order().is_some()
            && self.layout.order.f16_micro_panel_order()
                == destination.layout.order.f16_micro_panel_order()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

/// Half-open bounds along one tensor axis. `physical_end` includes any zero
/// padding while `logical_end` never exceeds the semantic tensor shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShardExtent {
    pub axis: u16,
    pub start: u32,
    pub logical_end: u32,
    pub physical_end: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorRegion {
    pub extents: Vec<ShardExtent>,
}

impl TensorRegion {
    pub fn new(extents: impl IntoIterator<Item = ShardExtent>) -> Self {
        Self {
            extents: extents.into_iter().collect(),
        }
    }

    pub fn logical_bounds(bounds: impl IntoIterator<Item = (u32, u32)>) -> Option<Self> {
        bounds
            .into_iter()
            .enumerate()
            .map(|(axis, (start, end))| {
                Some(ShardExtent {
                    axis: u16::try_from(axis).ok()?,
                    start,
                    logical_end: end,
                    physical_end: end,
                })
            })
            .collect::<Option<Vec<_>>>()
            .map(Self::new)
    }

    pub fn logical(&self) -> Self {
        Self::new(self.extents.iter().map(|extent| ShardExtent {
            physical_end: extent.logical_end,
            ..*extent
        }))
    }

    pub fn physical(&self) -> Self {
        Self::new(self.extents.iter().map(|extent| ShardExtent {
            logical_end: extent.physical_end,
            ..*extent
        }))
    }

    pub fn intersection(&self, other: &Self) -> Option<Self> {
        if self.extents.len() != other.extents.len() {
            return None;
        }
        self.extents
            .iter()
            .zip(&other.extents)
            .map(|(left, right)| {
                if left.axis != right.axis {
                    return None;
                }
                let start = left.start.max(right.start);
                let end = left.logical_end.min(right.logical_end);
                (start < end).then_some(ShardExtent {
                    axis: left.axis,
                    start,
                    logical_end: end,
                    physical_end: end,
                })
            })
            .collect::<Option<Vec<_>>>()
            .map(Self::new)
    }

    pub fn logical_elements(&self) -> u64 {
        self.extents.iter().fold(1, |elements, extent| {
            elements.saturating_mul(u64::from(extent.logical_end.saturating_sub(extent.start)))
        })
    }
}

impl std::ops::Deref for TensorRegion {
    type Target = [ShardExtent];

    fn deref(&self) -> &Self::Target {
        &self.extents
    }
}

impl std::ops::DerefMut for TensorRegion {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.extents
    }
}

impl<'a> IntoIterator for &'a TensorRegion {
    type Item = &'a ShardExtent;
    type IntoIter = std::slice::Iter<'a, ShardExtent>;

    fn into_iter(self) -> Self::IntoIter {
        self.extents.iter()
    }
}

impl<'a> IntoIterator for &'a mut TensorRegion {
    type Item = &'a mut ShardExtent;
    type IntoIter = std::slice::IterMut<'a, ShardExtent>;

    fn into_iter(self) -> Self::IntoIter {
        self.extents.iter_mut()
    }
}

impl From<Vec<ShardExtent>> for TensorRegion {
    fn from(extents: Vec<ShardExtent>) -> Self {
        Self { extents }
    }
}

impl FromIterator<ShardExtent> for TensorRegion {
    fn from_iter<T: IntoIterator<Item = ShardExtent>>(iter: T) -> Self {
        Self::new(iter)
    }
}

/// One rectangular region owned by a tile. Linear ownership can give a tile
/// several regions when its interval crosses row boundaries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedShard {
    pub tile: u16,
    pub extents: TensorRegion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolvedAxis {
    index: usize,
    tiling: AxisTiling,
    tile_stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResolvedOwnership {
    Linear { grain: u32 },
    Axes,
}

/// Shape-dependent ownership implied by a [`Layout`].
///
/// This is the canonical source for padded dimensions, shard ranges, and
/// physical allocation sizes. Element order is intentionally left unresolved:
/// storage code maps these tensor regions into row-major, block-major, or AMP
/// byte spans.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedLayout {
    shape: TensorShape,
    padded_shape: TensorShape,
    tile_count: u16,
    replicas: u16,
    axes: Vec<ResolvedAxis>,
    ownership: ResolvedOwnership,
}

impl Layout {
    pub(crate) fn resolve(&self, shape: &TensorShape) -> Result<ResolvedLayout, LayoutError> {
        if self.tiling.tile_count == 0 || self.tiling.replicas == 0 {
            return Err(LayoutError::EmptyTileGroup);
        }
        if shape.0.len() > usize::from(u16::MAX) {
            return Err(LayoutError::RankTooLarge(shape.0.len()));
        }
        if let Some(grain) = self.tiling.linear_grain() {
            let elements = shape.elements();
            if shape.0.is_empty()
                || grain == 0
                || elements / u64::from(grain) < u64::from(self.tiling.tile_count)
                || !elements.is_multiple_of(u64::from(grain))
            {
                return Err(LayoutError::EmptyAxisTiling);
            }
            return Ok(ResolvedLayout {
                shape: shape.clone(),
                padded_shape: shape.clone(),
                tile_count: self.tiling.tile_count,
                replicas: self.tiling.replicas,
                axes: Vec::new(),
                ownership: ResolvedOwnership::Linear { grain },
            });
        }

        let mut used_tiles = u32::from(self.tiling.replicas);
        let mut dimensions = shape.0.clone();
        let mut used_axes = Vec::with_capacity(self.tiling.axes.len());
        for tiling in &self.tiling.axes {
            if tiling.partitions == 0
                || tiling.padding_groups == 0
                || tiling.block_size == 0
                || tiling.padding_multiple == 0
                || tiling.shard_padding_multiple == 0
            {
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
            if !u32::from(tiling.partitions).is_multiple_of(u32::from(tiling.padding_groups))
                || !extent.is_multiple_of(u32::from(tiling.padding_groups))
            {
                return Err(LayoutError::InvalidPaddingGroups {
                    groups: tiling.padding_groups,
                    partitions: tiling.partitions,
                    extent,
                });
            }
            let group_extent = extent / u32::from(tiling.padding_groups);
            let remainder = group_extent % tiling.padding_multiple;
            if remainder != 0 && tiling.padding == Padding::Reject {
                return Err(LayoutError::IndivisibleAxis {
                    axis,
                    extent: group_extent,
                    block_size: tiling.padding_multiple,
                });
            }
            if remainder != 0 {
                let padded_group_extent = group_extent
                    .checked_add(tiling.padding_multiple - remainder)
                    .ok_or(LayoutError::ExtentOverflow(axis))?;
                dimensions[axis] = padded_group_extent
                    .checked_mul(u32::from(tiling.padding_groups))
                    .ok_or(LayoutError::ExtentOverflow(axis))?;
            }
            let padded_group_extent = dimensions[axis] / u32::from(tiling.padding_groups);
            if !padded_group_extent.is_multiple_of(tiling.block_size) {
                return Err(LayoutError::IndivisibleAxis {
                    axis,
                    extent: padded_group_extent,
                    block_size: tiling.block_size,
                });
            }
        }
        if used_tiles != u32::from(self.tiling.tile_count) {
            return Err(LayoutError::TileCountMismatch {
                declared: self.tiling.tile_count,
                implied: used_tiles,
            });
        }

        let strides = self.tiling.axis_strides()?;
        validate_tile_mapping(
            &self.tiling.axes,
            self.tiling.replicas,
            &strides,
            self.tiling.tile_count,
        )?;
        let axes = self
            .tiling
            .axes
            .iter()
            .copied()
            .zip(strides)
            .map(|(tiling, tile_stride)| {
                Ok(ResolvedAxis {
                    index: tiling.axis.resolve(dimensions.len())?,
                    tiling,
                    tile_stride,
                })
            })
            .collect::<Result<Vec<_>, LayoutError>>()?;
        let resolved = ResolvedLayout {
            shape: shape.clone(),
            padded_shape: TensorShape(dimensions),
            tile_count: self.tiling.tile_count,
            replicas: self.tiling.replicas,
            axes,
            ownership: ResolvedOwnership::Axes,
        };
        // Every shard has one of two adjacent block counts. Validate one long,
        // one short, and the final coordinate rather than scanning all tiles.
        for axis in &resolved.axes {
            let (partitions_per_group, _, _, long_shards) =
                axis.partition_geometry(&resolved.padded_shape);
            axis.bounds(&resolved.shape, &resolved.padded_shape, 0)?;
            if long_shards < partitions_per_group {
                axis.bounds(&resolved.shape, &resolved.padded_shape, long_shards)?;
            }
            axis.bounds(
                &resolved.shape,
                &resolved.padded_shape,
                u32::from(axis.tiling.partitions) - 1,
            )?;
        }
        Ok(resolved)
    }
}

impl ResolvedLayout {
    pub(crate) fn padded_shape(&self) -> &TensorShape {
        &self.padded_shape
    }

    pub(crate) const fn tile_count(&self) -> u16 {
        self.tile_count
    }

    pub(crate) fn total_elements(&self) -> u64 {
        match self.ownership {
            ResolvedOwnership::Linear { .. } => self.shape.elements(),
            ResolvedOwnership::Axes => self
                .padded_shape
                .0
                .iter()
                .enumerate()
                .map(|(index, &extent)| {
                    self.axis(index).map_or(u64::from(extent), |axis| {
                        axis.total_physical_extent(&self.padded_shape)
                    })
                })
                .product::<u64>()
                .saturating_mul(u64::from(self.replicas)),
        }
    }

    pub(crate) fn maximum_tile_elements(&self) -> u64 {
        match self.ownership {
            ResolvedOwnership::Linear { grain } => {
                let grains = self.shape.elements() / u64::from(grain);
                grains
                    .div_ceil(u64::from(self.tile_count))
                    .saturating_mul(u64::from(grain))
            }
            ResolvedOwnership::Axes => self
                .padded_shape
                .0
                .iter()
                .enumerate()
                .map(|(index, &extent)| {
                    self.axis(index).map_or(u64::from(extent), |axis| {
                        u64::from(axis.maximum_physical_extent(&self.padded_shape))
                    })
                })
                .product(),
        }
    }

    pub(crate) fn tile_elements(&self, tile: u16) -> Option<u64> {
        if tile >= self.tile_count {
            return None;
        }
        match self.ownership {
            ResolvedOwnership::Linear { grain } => {
                let grains = self.shape.elements() / u64::from(grain);
                let short = grains / u64::from(self.tile_count);
                let long = grains % u64::from(self.tile_count);
                Some((short + u64::from(u64::from(tile) < long)).saturating_mul(u64::from(grain)))
            }
            ResolvedOwnership::Axes => self.padded_shape.0.iter().enumerate().try_fold(
                1_u64,
                |elements, (index, &extent)| {
                    let width = self.axis(index).map_or(u64::from(extent), |axis| {
                        let coordinate = (u32::from(tile) / axis.tile_stride)
                            % u32::from(axis.tiling.partitions);
                        let (start, _, end) = axis
                            .bounds(&self.shape, &self.padded_shape, coordinate)
                            .expect("resolved axis bounds remain valid");
                        u64::from(end - start)
                    });
                    elements.checked_mul(width)
                },
            ),
        }
    }

    pub(crate) fn has_empty_shards(&self) -> bool {
        match self.ownership {
            ResolvedOwnership::Linear { .. } => false,
            ResolvedOwnership::Axes => self.axes.iter().any(|axis| {
                let (partitions_per_group, _, _, _) = axis.partition_geometry(&self.padded_shape);
                let (start, logical_end, _) = axis
                    .bounds(&self.shape, &self.padded_shape, partitions_per_group - 1)
                    .expect("resolved axis bounds remain valid");
                start == logical_end
            }),
        }
    }

    pub(crate) fn maximum_axis_extent(&self, axis: usize) -> Option<u32> {
        let &padded_extent = self.padded_shape.0.get(axis)?;
        if matches!(self.ownership, ResolvedOwnership::Linear { .. }) {
            return Some(padded_extent);
        }
        Some(self.axis(axis).map_or(padded_extent, |resolved| {
            resolved.maximum_physical_extent(&self.padded_shape)
        }))
    }

    /// Physical range of one axis assigned to `tile`. Linear ownership may
    /// cross higher-dimensional rows, so it conservatively returns the whole
    /// axis just as the previous traffic estimator did.
    pub(crate) fn tile_axis_range(&self, tile: u16, axis: usize) -> Option<Range<u32>> {
        if tile >= self.tile_count {
            return None;
        }
        let &padded_extent = self.padded_shape.0.get(axis)?;
        if matches!(self.ownership, ResolvedOwnership::Linear { .. }) {
            return Some(0..padded_extent);
        }
        Some(self.axis(axis).map_or(0..padded_extent, |resolved| {
            let coordinate =
                (u32::from(tile) / resolved.tile_stride) % u32::from(resolved.tiling.partitions);
            let (start, _, physical_end) = resolved
                .bounds(&self.shape, &self.padded_shape, coordinate)
                .expect("resolved axis bounds remain valid");
            start..physical_end
        }))
    }

    pub(crate) fn axis_bounds(&self, axis: usize, coordinate: u32) -> Option<(u32, u32, u32)> {
        self.axis(axis)?
            .bounds(&self.shape, &self.padded_shape, coordinate)
            .ok()
    }

    pub(crate) fn shard_extents(&self) -> Vec<ResolvedShard> {
        let shards = match self.ownership {
            ResolvedOwnership::Linear { grain } => self.linear_shard_extents(grain),
            ResolvedOwnership::Axes => (0..self.tile_count)
                .map(|tile| {
                    let extents = self
                        .padded_shape
                        .0
                        .iter()
                        .enumerate()
                        .map(|(index, &padded_extent)| {
                            let (start, logical_end, physical_end) = self.axis(index).map_or(
                                (0, self.shape.0[index], padded_extent),
                                |axis| {
                                    let coordinate = (u32::from(tile) / axis.tile_stride)
                                        % u32::from(axis.tiling.partitions);
                                    axis.bounds(&self.shape, &self.padded_shape, coordinate)
                                        .expect("resolved axis bounds remain valid")
                                },
                            );
                            ShardExtent {
                                axis: index as u16,
                                start,
                                logical_end,
                                physical_end,
                            }
                        })
                        .collect();
                    ResolvedShard { tile, extents }
                })
                .collect(),
        };
        debug_assert!({
            let mut materialized = vec![0_u64; usize::from(self.tile_count)];
            for shard in &shards {
                let elements = shard.extents.iter().fold(1_u64, |elements, extent| {
                    elements.saturating_mul(u64::from(extent.physical_end - extent.start))
                });
                materialized[usize::from(shard.tile)] =
                    materialized[usize::from(shard.tile)].saturating_add(elements);
            }
            materialized
                .into_iter()
                .enumerate()
                .all(|(tile, elements)| self.tile_elements(tile as u16) == Some(elements))
        });
        shards
    }

    fn axis(&self, index: usize) -> Option<&ResolvedAxis> {
        self.axes.iter().find(|axis| axis.index == index)
    }

    fn linear_shard_extents(&self, grain: u32) -> Vec<ResolvedShard> {
        let elements = self.shape.elements();
        let grains = elements / u64::from(grain);
        let tiles = u64::from(self.tile_count);
        let rank = self.shape.0.len();
        let width = u64::from(self.shape.0[rank - 1]);
        let mut shards = Vec::new();
        for tile in 0..self.tile_count {
            let start_grain =
                u64::from(tile) * (grains / tiles) + u64::from(tile).min(grains % tiles);
            let tile_grains = grains / tiles + u64::from(u64::from(tile) < grains % tiles);
            let start = start_grain * u64::from(grain);
            let end = start + tile_grains * u64::from(grain);
            let first_row = start / width;
            let last_row = end.div_ceil(width);
            for row in first_row..last_row {
                let mut coordinates = vec![0u32; rank.saturating_sub(1)];
                let mut linear_row = row;
                for axis in (0..rank.saturating_sub(1)).rev() {
                    let extent = u64::from(self.shape.0[axis]);
                    coordinates[axis] = (linear_row % extent) as u32;
                    linear_row /= extent;
                }
                let mut extents = coordinates
                    .into_iter()
                    .enumerate()
                    .map(|(axis, coordinate)| ShardExtent {
                        axis: axis as u16,
                        start: coordinate,
                        logical_end: coordinate + 1,
                        physical_end: coordinate + 1,
                    })
                    .collect::<Vec<_>>();
                let column_start = if row == first_row { start % width } else { 0 };
                let column_end = if row + 1 == last_row && !end.is_multiple_of(width) {
                    end % width
                } else {
                    width
                };
                extents.push(ShardExtent {
                    axis: (rank - 1) as u16,
                    start: column_start as u32,
                    logical_end: column_end as u32,
                    physical_end: column_end as u32,
                });
                shards.push(ResolvedShard {
                    tile,
                    extents: extents.into(),
                });
            }
        }
        shards
    }
}

impl ResolvedAxis {
    fn partition_geometry(self, padded_shape: &TensorShape) -> (u32, u32, u32, u32) {
        let groups = u32::from(self.tiling.padding_groups);
        let partitions_per_group = u32::from(self.tiling.partitions) / groups;
        let padded_group_extent = padded_shape.0[self.index] / groups;
        let blocks = padded_group_extent / self.tiling.block_size;
        let short_blocks = blocks / partitions_per_group;
        let long_shards = blocks % partitions_per_group;
        (partitions_per_group, blocks, short_blocks, long_shards)
    }

    fn padded_block_width(self, blocks: u32) -> u32 {
        let allocated = blocks * self.tiling.block_size;
        allocated
            .div_ceil(self.tiling.shard_padding_multiple)
            .saturating_mul(self.tiling.shard_padding_multiple)
    }

    fn maximum_physical_extent(self, padded_shape: &TensorShape) -> u32 {
        let (_, _, short_blocks, long_shards) = self.partition_geometry(padded_shape);
        self.padded_block_width(short_blocks + u32::from(long_shards != 0))
    }

    fn total_physical_extent(self, padded_shape: &TensorShape) -> u64 {
        let (partitions_per_group, _, short_blocks, long_shards) =
            self.partition_geometry(padded_shape);
        let short_shards = partitions_per_group - long_shards;
        let group_width = u64::from(self.padded_block_width(short_blocks))
            .saturating_mul(u64::from(short_shards))
            .saturating_add(
                u64::from(self.padded_block_width(short_blocks + 1))
                    .saturating_mul(u64::from(long_shards)),
            );
        group_width.saturating_mul(u64::from(self.tiling.padding_groups))
    }

    fn bounds(
        self,
        shape: &TensorShape,
        padded_shape: &TensorShape,
        coordinate: u32,
    ) -> Result<(u32, u32, u32), LayoutError> {
        resolve_axis_bounds(
            self.tiling,
            padded_shape.0[self.index],
            shape.0[self.index],
            coordinate,
            self.index,
        )
    }
}

fn resolve_axis_bounds(
    tiling: AxisTiling,
    padded_extent: u32,
    logical_extent: u32,
    coordinate: u32,
    axis: usize,
) -> Result<(u32, u32, u32), LayoutError> {
    let partitions = u32::from(tiling.partitions);
    let groups = u32::from(tiling.padding_groups);
    if groups == 0
        || coordinate >= partitions
        || !partitions.is_multiple_of(groups)
        || !padded_extent.is_multiple_of(groups)
        || !logical_extent.is_multiple_of(groups)
    {
        return Err(LayoutError::InvalidPaddingGroups {
            groups: tiling.padding_groups,
            partitions: tiling.partitions,
            extent: logical_extent,
        });
    }
    let partitions_per_group = partitions / groups;
    let group = coordinate / partitions_per_group;
    let coordinate_in_group = coordinate % partitions_per_group;
    let padded_group_extent = padded_extent / groups;
    let logical_group_extent = logical_extent / groups;
    if !padded_group_extent.is_multiple_of(tiling.block_size) {
        return Err(LayoutError::IndivisibleAxis {
            axis,
            extent: padded_group_extent,
            block_size: tiling.block_size,
        });
    }
    let blocks = padded_group_extent / tiling.block_size;
    let short_size = blocks / partitions_per_group;
    let long_shards = blocks % partitions_per_group;
    let start_blocks = coordinate_in_group * short_size + coordinate_in_group.min(long_shards);
    let shard_blocks = short_size + u32::from(coordinate_in_group < long_shards);
    let start_in_group = start_blocks
        .checked_mul(tiling.block_size)
        .ok_or(LayoutError::ExtentOverflow(axis))?;
    let allocated = shard_blocks
        .checked_mul(tiling.block_size)
        .ok_or(LayoutError::ExtentOverflow(axis))?;
    let remainder = allocated % tiling.shard_padding_multiple;
    if remainder != 0 && tiling.padding == Padding::Reject {
        return Err(LayoutError::IndivisibleShard {
            extent: allocated,
            block_size: tiling.shard_padding_multiple,
        });
    }
    let physical_width = if remainder == 0 {
        allocated
    } else {
        allocated
            .checked_add(tiling.shard_padding_multiple - remainder)
            .ok_or(LayoutError::ExtentOverflow(axis))?
    };
    let group_logical_base = group
        .checked_mul(logical_group_extent)
        .ok_or(LayoutError::ExtentOverflow(axis))?;
    let start = group_logical_base
        .checked_add(start_in_group)
        .ok_or(LayoutError::ExtentOverflow(axis))?;
    let logical_end = group_logical_base
        .checked_add(
            start_in_group
                .checked_add(allocated)
                .ok_or(LayoutError::ExtentOverflow(axis))?
                .min(logical_group_extent)
                .max(start_in_group),
        )
        .ok_or(LayoutError::ExtentOverflow(axis))?;
    let physical_end = start
        .checked_add(physical_width)
        .ok_or(LayoutError::ExtentOverflow(axis))?;
    Ok((start, logical_end, physical_end))
}

fn validate_tile_mapping(
    axes: &[AxisTiling],
    replicas: u16,
    strides: &[u32],
    tile_count: u16,
) -> Result<(), LayoutError> {
    if has_regular_tile_mapping(axes, replicas, strides) {
        return Ok(());
    }
    let coordinate_count = axes
        .iter()
        .try_fold(1usize, |count, axis| {
            count.checked_mul(usize::from(axis.partitions))
        })
        .ok_or(LayoutError::TileCountOverflow)?;
    let mut coordinate_copies = vec![0u16; coordinate_count];
    for tile in 0..tile_count {
        let coordinate = axes
            .iter()
            .zip(strides)
            .try_fold(0usize, |coordinate, (axis, stride)| {
                coordinate
                    .checked_mul(usize::from(axis.partitions))
                    .and_then(|coordinate| {
                        coordinate.checked_add(
                            ((u32::from(tile) / stride) % u32::from(axis.partitions)) as usize,
                        )
                    })
            })
            .ok_or(LayoutError::TileCountOverflow)?;
        coordinate_copies[coordinate] = coordinate_copies[coordinate]
            .checked_add(1)
            .ok_or(LayoutError::TileCountOverflow)?;
    }
    if coordinate_copies.iter().any(|copies| *copies != replicas) {
        return Err(LayoutError::InvalidTileMapping);
    }
    Ok(())
}

fn has_regular_tile_mapping(axes: &[AxisTiling], replicas: u16, strides: &[u32]) -> bool {
    let mut digits = axes
        .iter()
        .zip(strides)
        .filter(|(axis, _)| axis.partitions > 1)
        .map(|(axis, &stride)| (stride, u32::from(axis.partitions)))
        .collect::<Vec<_>>();
    digits.sort_unstable();
    let Some(&(base, _)) = digits.first() else {
        return true;
    };
    if base == 0 || !u32::from(replicas).is_multiple_of(base) {
        return false;
    }
    let mut expected_stride = base;
    for (stride, partitions) in digits {
        if stride != expected_stride {
            return false;
        }
        let Some(next) = expected_stride.checked_mul(partitions) else {
            return false;
        };
        expected_stride = next;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    const RANDOM_CASES: usize = 128;

    #[test]
    fn randomized_axis_layout_resolution_matches_materialized_ownership() {
        let mut random = fastrand::Rng::with_seed(0x7265_736f_6c76_6564);
        for case in 0..RANDOM_CASES {
            let rank = random.usize(2..=5);
            let first_axis = random.usize(0..rank);
            let second_axis = (first_axis + random.usize(1..rank)) % rank;
            let first_partitions = random.u16(1..=8);
            let second_partitions = random.u16(1..=8);
            let replicas = random.u16(1..=3);
            let first_block = random.u32(1..=16);
            let second_block = random.u32(1..=16);
            let mut shape = (0..rank).map(|_| random.u32(1..=97)).collect::<Vec<_>>();
            shape[first_axis] = random.u32(1..=97);
            shape[second_axis] = random.u32(1..=97);
            let layout = Layout {
                order: crate::layout::ElementOrder::RowMajor,
                tiling: TensorTiling {
                    tile_count: first_partitions * second_partitions * replicas,
                    replicas,
                    axes: vec![
                        AxisTiling::new(
                            TensorAxis::FromStart(first_axis as u16),
                            first_partitions,
                            first_block,
                            Padding::Zero,
                        )
                        .with_shard_padding_multiple(random.u32(1..=16)),
                        AxisTiling::new(
                            TensorAxis::FromStart(second_axis as u16),
                            second_partitions,
                            second_block,
                            Padding::Zero,
                        )
                        .with_shard_padding_multiple(random.u32(1..=16)),
                    ],
                },
                memory_class: MemoryClass::Standard,
            };
            let resolved = layout.resolve(&TensorShape(shape.clone())).unwrap();
            let shards = resolved.shard_extents();
            assert_eq!(
                shards.len(),
                usize::from(resolved.tile_count()),
                "case {case}"
            );

            let tile_elements = shards
                .iter()
                .map(|shard| {
                    shard.extents.iter().fold(1_u64, |elements, extent| {
                        elements * u64::from(extent.physical_end - extent.start)
                    })
                })
                .collect::<Vec<_>>();
            assert_eq!(
                tile_elements.iter().sum::<u64>(),
                resolved.total_elements(),
                "case {case}"
            );
            assert_eq!(
                tile_elements.iter().copied().max().unwrap(),
                resolved.maximum_tile_elements(),
                "case {case}"
            );
            for (tile, &elements) in tile_elements.iter().enumerate() {
                assert_eq!(
                    resolved.tile_elements(tile as u16),
                    Some(elements),
                    "case {case}"
                );
            }

            for axis in [first_axis, second_axis] {
                let ranges = shards
                    .iter()
                    .map(|shard| {
                        let extent = shard.extents[axis];
                        (extent.start, extent.logical_end)
                    })
                    .filter(|(start, end)| start < end)
                    .collect::<BTreeSet<_>>();
                let mut cursor = 0;
                for (start, end) in ranges {
                    assert_eq!(start, cursor, "case {case}, axis {axis}");
                    cursor = end;
                }
                assert_eq!(cursor, shape[axis], "case {case}, axis {axis}");
            }
        }
    }

    #[test]
    fn randomized_linear_resolution_covers_each_logical_element_once() {
        let mut random = fastrand::Rng::with_seed(0x6c69_6e65_6172_697a);
        for case in 0..RANDOM_CASES {
            let rank = random.usize(1..=5);
            let shape = TensorShape((0..rank).map(|_| random.u32(1..=8)).collect::<Vec<_>>());
            let elements = shape.elements();
            let divisors = (1..=elements.min(32))
                .filter(|grain| elements.is_multiple_of(*grain))
                .collect::<Vec<_>>();
            let grain = divisors[random.usize(..divisors.len())] as u32;
            let grains = elements / u64::from(grain);
            let tile_count = random.u16(1..=u16::try_from(grains.min(32)).unwrap());
            let layout = Layout::row_major(TensorTiling::linear(tile_count, grain));
            let resolved = layout.resolve(&shape).unwrap();
            let shards = resolved.shard_extents();
            let materialized = shards.iter().flat_map(|shard| &shard.extents).count();
            assert!(materialized >= usize::from(tile_count), "case {case}");
            assert_eq!(
                shards
                    .iter()
                    .map(|shard| shard.extents.iter().fold(1_u64, |count, extent| {
                        count * u64::from(extent.logical_end - extent.start)
                    }))
                    .sum::<u64>(),
                elements,
                "case {case}"
            );
            assert_eq!(resolved.total_elements(), elements, "case {case}");
        }
    }
}
