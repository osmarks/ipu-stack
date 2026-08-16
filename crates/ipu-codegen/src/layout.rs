use crate::graph::TensorShape;
use crate::mid::{AxisTiling, Layout, LayoutError, Padding};
use std::ops::Range;

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
    use crate::mid::{MemoryClass, TensorAxis, TensorTiling};
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
                order: crate::mid::ElementOrder::RowMajor,
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
