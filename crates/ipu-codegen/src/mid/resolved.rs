//! Resolved ownership geometry shared by validation, estimates, and lowering.
//!
//! Axis partitions are resolved once, independently of element encoding and
//! physical addresses. Replicas select the same bounds without duplicating them.

use super::{AxisTiling, Layout, LayoutError, Padding, ShardExtent, TensorTiling};
use crate::graph::TensorShape;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedAxis {
    partitions: Vec<ShardExtent>,
    stride: u32,
}

impl ResolvedAxis {
    pub(crate) fn extent_sizes(&self) -> impl Iterator<Item = u32> + '_ {
        self.partitions
            .iter()
            .map(|part| part.physical_end - part.start)
    }

    pub(crate) fn complete_panels_except_tail(&self, grain: u32) -> bool {
        self.partitions
            .iter()
            .take(self.partitions.len().saturating_sub(1))
            .all(|part| (part.physical_end - part.start).is_multiple_of(grain))
    }

    pub(crate) fn extents_are_multiple_of(&self, grain: u32) -> bool {
        self.partitions
            .iter()
            .all(|part| (part.physical_end - part.start).is_multiple_of(grain))
    }

    pub(crate) fn same_partitioning(&self, other: &Self) -> bool {
        self.partitions == other.partitions
            && (self.partitions.len() <= 1 || self.stride == other.stride)
    }

    pub(crate) fn extent(&self, tile: u16) -> ShardExtent {
        self.partitions[(u32::from(tile) / self.stride) as usize % self.partitions.len()]
    }

    pub(crate) fn maximum_extent(&self) -> u32 {
        self.partitions
            .iter()
            .map(|part| part.physical_end - part.start)
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn minimum_extent(&self) -> u32 {
        self.partitions
            .iter()
            .map(|part| part.physical_end - part.start)
            .min()
            .unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ResolvedLayout {
    pub(crate) padded_shape: TensorShape,
    shape: TensorShape,
    tile_count: u16,
    replicas: u16,
    // Linear ownership can span several row fragments per tile, so it has no
    // independent axis partitions. Expand it only when concrete shards are needed.
    linear_grain: Option<u32>,
    axes: Vec<ResolvedAxis>,
}

impl Layout {
    pub(crate) fn resolve(&self, shape: &TensorShape) -> Result<ResolvedLayout, LayoutError> {
        let padded_shape = self.padded_shape(shape)?;
        let linear_grain = self.tiling.linear_grain();
        let mut axes = Vec::new();
        if linear_grain.is_none() {
            let strides = self.tiling.axis_strides()?;
            for (axis, &physical_end) in padded_shape.0.iter().enumerate() {
                let axis_id = u16::try_from(axis).map_err(|_| LayoutError::ExtentOverflow(axis))?;
                let (partitions, stride) = match self
                    .tiling
                    .axes
                    .iter()
                    .zip(&strides)
                    .find(|(tiling, _)| tiling.axis.resolve(shape.0.len()) == Ok(axis))
                {
                    Some((tiling, &stride)) => {
                        let partitions = (0..u32::from(tiling.partitions))
                            .map(|coordinate| {
                                let (start, logical_end, physical_end) =
                                    tiling.shard_bounds(physical_end, shape.0[axis], coordinate)?;
                                Ok(ShardExtent {
                                    axis: axis_id,
                                    start,
                                    logical_end,
                                    physical_end,
                                })
                            })
                            .collect::<Result<Vec<_>, LayoutError>>()?;
                        (partitions, stride)
                    }
                    None => (
                        vec![ShardExtent {
                            axis: axis_id,
                            start: 0,
                            logical_end: shape.0[axis],
                            physical_end,
                        }],
                        1,
                    ),
                };
                axes.push(ResolvedAxis { partitions, stride });
            }
        }
        Ok(ResolvedLayout {
            padded_shape,
            shape: shape.clone(),
            tile_count: self.tiling.tile_count,
            replicas: self.tiling.replicas,
            linear_grain,
            axes,
        })
    }
}

impl ResolvedLayout {
    pub(crate) fn axes(&self) -> Option<&[ResolvedAxis]> {
        self.linear_grain.is_none().then_some(&self.axes)
    }

    pub(crate) fn has_empty_shards(&self) -> bool {
        self.axes.iter().any(|axis| {
            axis.partitions
                .iter()
                .any(|part| part.start == part.logical_end)
        })
    }

    pub(crate) fn physical_elements(&self) -> u64 {
        if self.linear_grain.is_some() {
            return self.shape.elements();
        }
        self.axes
            .iter()
            .map(|axis| {
                axis.partitions
                    .iter()
                    .map(|part| u64::from(part.physical_end - part.start))
                    .sum::<u64>()
            })
            .fold(u64::from(self.replicas), u64::saturating_mul)
    }

    pub(crate) fn maximum_tile_elements(&self) -> u64 {
        if let Some(grain) = self.linear_grain {
            return (self.shape.elements() / u64::from(grain))
                .div_ceil(u64::from(self.tile_count))
                .saturating_mul(u64::from(grain));
        }
        self.axes
            .iter()
            .map(|axis| u64::from(axis.maximum_extent()))
            .fold(1, u64::saturating_mul)
    }

    pub(crate) fn shard_extents(&self) -> Result<Vec<(u16, Vec<ShardExtent>)>, LayoutError> {
        if let Some(grain) = self.linear_grain {
            let shape = &self.shape;
            let rank = shape.0.len();
            let elements = shape.elements();
            let grains = elements / u64::from(grain);
            let tiles = u64::from(self.tile_count);
            let width = u64::from(*shape.0.last().ok_or(LayoutError::EmptyAxisTiling)?);
            let mut all = Vec::new();
            for tile in 0..self.tile_count {
                let start_grain =
                    u64::from(tile) * (grains / tiles) + u64::from(tile).min(grains % tiles);
                let tile_grains = grains / tiles + u64::from(u64::from(tile) < grains % tiles);
                let start = start_grain * u64::from(grain);
                let end = start + tile_grains * u64::from(grain);
                let first_row = start / width;
                let last_row = end.div_ceil(width);
                for row in first_row..last_row {
                    let column_start = if row == first_row { start % width } else { 0 };
                    let column_end = if row + 1 == last_row && !end.is_multiple_of(width) {
                        end % width
                    } else {
                        width
                    };
                    let mut coordinates = vec![0u32; rank.saturating_sub(1)];
                    let mut linear_row = row;
                    for axis in (0..rank.saturating_sub(1)).rev() {
                        let extent = u64::from(shape.0[axis]);
                        coordinates[axis] = u32::try_from(linear_row % extent)
                            .map_err(|_| LayoutError::ExtentOverflow(rank))?;
                        linear_row /= extent;
                    }
                    let mut region = coordinates
                        .into_iter()
                        .enumerate()
                        .map(|(axis, coordinate)| {
                            Ok(ShardExtent {
                                axis: u16::try_from(axis)
                                    .map_err(|_| LayoutError::ExtentOverflow(rank))?,
                                start: coordinate,
                                logical_end: coordinate + 1,
                                physical_end: coordinate + 1,
                            })
                        })
                        .collect::<Result<Vec<_>, LayoutError>>()?;
                    region.push(ShardExtent {
                        axis: u16::try_from(rank - 1)
                            .map_err(|_| LayoutError::ExtentOverflow(rank))?,
                        start: u32::try_from(column_start)
                            .map_err(|_| LayoutError::ExtentOverflow(rank))?,
                        logical_end: u32::try_from(column_end)
                            .map_err(|_| LayoutError::ExtentOverflow(rank))?,
                        physical_end: u32::try_from(column_end)
                            .map_err(|_| LayoutError::ExtentOverflow(rank))?,
                    });
                    all.push((tile, region));
                }
            }
            return Ok(all);
        }
        Ok((0..self.tile_count)
            .map(|tile| {
                (
                    tile,
                    self.axes.iter().map(|axis| axis.extent(tile)).collect(),
                )
            })
            .collect())
    }
}

impl AxisTiling {
    pub(crate) fn shard_bounds(
        self,
        padded_extent: u32,
        logical_extent: u32,
        coordinate: u32,
    ) -> Result<(u32, u32, u32), LayoutError> {
        let partitions = u32::from(self.partitions);
        let groups = u32::from(self.padding_groups);
        if groups == 0
            || coordinate >= partitions
            || !partitions.is_multiple_of(groups)
            || !padded_extent.is_multiple_of(groups)
            || !logical_extent.is_multiple_of(groups)
        {
            return Err(LayoutError::InvalidPaddingGroups {
                groups: self.padding_groups,
                partitions: self.partitions,
                extent: logical_extent,
            });
        }
        let partitions_per_group = partitions / groups;
        let group = coordinate / partitions_per_group;
        let coordinate_in_group = coordinate % partitions_per_group;
        let padded_group_extent = padded_extent / groups;
        let logical_group_extent = logical_extent / groups;
        if !padded_group_extent.is_multiple_of(self.block_size) {
            return Err(LayoutError::IndivisibleAxis {
                axis: 0,
                extent: padded_group_extent,
                block_size: self.block_size,
            });
        }
        let blocks = padded_group_extent / self.block_size;
        let short_size = blocks / partitions_per_group;
        let long_shards = blocks % partitions_per_group;
        let start_blocks = coordinate_in_group * short_size + coordinate_in_group.min(long_shards);
        let shard_blocks = short_size + u32::from(coordinate_in_group < long_shards);
        let start_in_group = start_blocks
            .checked_mul(self.block_size)
            .ok_or(LayoutError::ExtentOverflow(0))?;
        let allocated = shard_blocks
            .checked_mul(self.block_size)
            .ok_or(LayoutError::ExtentOverflow(0))?;
        let remainder = allocated % self.shard_padding_multiple;
        if remainder != 0 && self.padding == Padding::Reject {
            return Err(LayoutError::IndivisibleShard {
                extent: allocated,
                block_size: self.shard_padding_multiple,
            });
        }
        let physical_width = if remainder == 0 {
            allocated
        } else {
            allocated
                .checked_add(self.shard_padding_multiple - remainder)
                .ok_or(LayoutError::ExtentOverflow(0))?
        };
        let group_logical_base = group
            .checked_mul(logical_group_extent)
            .ok_or(LayoutError::ExtentOverflow(0))?;
        let start = group_logical_base
            .checked_add(start_in_group)
            .ok_or(LayoutError::ExtentOverflow(0))?;
        let logical_end = group_logical_base
            .checked_add(
                start_in_group
                    .checked_add(allocated)
                    .ok_or(LayoutError::ExtentOverflow(0))?
                    .min(logical_group_extent)
                    .max(start_in_group),
            )
            .ok_or(LayoutError::ExtentOverflow(0))?;
        let physical_end = start
            .checked_add(physical_width)
            .ok_or(LayoutError::ExtentOverflow(0))?;
        Ok((start, logical_end, physical_end))
    }
}

impl TensorTiling {
    pub(super) fn axis_strides(&self) -> Result<Vec<u32>, LayoutError> {
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

impl Layout {
    /// Returns the physical extents after applying declared zero padding.
    pub fn padded_shape(&self, shape: &TensorShape) -> Result<TensorShape, LayoutError> {
        if self.tiling.tile_count == 0 || self.tiling.replicas == 0 {
            return Err(LayoutError::EmptyTileGroup);
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
            return Ok(shape.clone());
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
            if remainder != 0 {
                match tiling.padding {
                    Padding::Reject => {
                        return Err(LayoutError::IndivisibleAxis {
                            axis,
                            extent: group_extent,
                            block_size: tiling.padding_multiple,
                        });
                    }
                    Padding::Zero => {}
                }
            }
            if remainder != 0 && tiling.padding == Padding::Zero {
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
        if has_regular_tile_mapping(&self.tiling, &strides) {
            return Ok(TensorShape(dimensions));
        }
        let coordinate_count = self
            .tiling
            .axes
            .iter()
            .try_fold(1usize, |count, axis| {
                count.checked_mul(usize::from(axis.partitions))
            })
            .ok_or(LayoutError::TileCountOverflow)?;
        let mut coordinate_copies = vec![0u16; coordinate_count];
        for tile in 0..self.tiling.tile_count {
            let coordinate = self
                .tiling
                .axes
                .iter()
                .zip(&strides)
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
        if coordinate_copies
            .iter()
            .any(|copies| *copies != self.tiling.replicas)
        {
            return Err(LayoutError::InvalidTileMapping);
        }
        Ok(TensorShape(dimensions))
    }
}

fn has_regular_tile_mapping(tiling: &TensorTiling, strides: &[u32]) -> bool {
    let mut digits = tiling
        .axes
        .iter()
        .zip(strides)
        .filter(|(axis, _)| axis.partitions > 1)
        .map(|(axis, &stride)| (stride, u32::from(axis.partitions)))
        .collect::<Vec<_>>();
    digits.sort_unstable();
    let Some(&(base, _)) = digits.first() else {
        return true;
    };
    if base == 0 || !u32::from(tiling.replicas).is_multiple_of(base) {
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
