//! Resolved ownership geometry shared by validation, estimates, and lowering.
//!
//! Axis partitions are resolved once, independently of element encoding and
//! physical addresses. Replicas select the same bounds without duplicating them.

use super::{AxisTiling, Layout, LayoutError, Padding, ShardExtent, TensorTiling};
use super::{TensorAxis, TensorShape, TensorType};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedAxis {
    partitions: Vec<ShardExtent>,
    stride: u32,
}

impl ResolvedAxis {
    pub(crate) fn partitions(&self) -> &[ShardExtent] {
        &self.partitions
    }

    pub(crate) fn extent_sizes(&self) -> impl Iterator<Item = u32> + '_ {
        self.partitions
            .iter()
            .map(|part| part.physical_end - part.start)
    }

    pub(crate) fn complete_panels_except_tail(&self, grain: u32) -> bool {
        self.extent_sizes()
            .take(self.partitions.len().saturating_sub(1))
            .all(|extent| extent.is_multiple_of(grain))
    }

    pub(crate) fn extents_are_multiple_of(&self, grain: u32) -> bool {
        self.extent_sizes()
            .all(|extent| extent.is_multiple_of(grain))
    }

    pub(crate) fn same_partitioning(&self, other: &Self) -> bool {
        self.partitions == other.partitions
            && (self.partitions.len() <= 1 || self.stride == other.stride)
    }

    pub(crate) fn extent(&self, tile: u16) -> ShardExtent {
        self.partitions[(u32::from(tile) / self.stride) as usize % self.partitions.len()]
    }

    pub(crate) fn maximum_extent(&self) -> u32 {
        self.extent_sizes().max().unwrap_or(0)
    }

    pub(crate) fn minimum_extent(&self) -> u32 {
        self.extent_sizes().min().unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ResolvedLayout {
    pub(crate) padded_shape: TensorShape,
    shape: TensorShape,
    tile_count: u16,
    #[cfg(test)]
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
            #[cfg(test)]
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

    #[cfg(test)]
    pub(crate) fn physical_elements(&self) -> u64 {
        if let Some(grain) = self.linear_grain {
            return self.shape.elements().div_ceil(u64::from(grain)) * u64::from(grain);
        }
        self.axes
            .iter()
            .map(|axis| axis.extent_sizes().map(u64::from).sum::<u64>())
            .fold(u64::from(self.replicas), u64::saturating_mul)
    }

    pub(crate) fn maximum_tile_elements(&self) -> u64 {
        if let Some(grain) = self.linear_grain {
            return self
                .shape
                .elements()
                .div_ceil(u64::from(grain))
                .div_ceil(u64::from(self.tile_count))
                .saturating_mul(u64::from(grain));
        }
        self.axes
            .iter()
            .map(|axis| u64::from(axis.maximum_extent()))
            .fold(1, u64::saturating_mul)
    }

    /// Physical payload on one logical owner, without expanding row fragments.
    pub(crate) fn tile_elements(&self, tile: u16) -> u64 {
        if tile >= self.tile_count {
            return 0;
        }
        if let Some(grain) = self.linear_grain {
            let grains = self.shape.elements().div_ceil(u64::from(grain));
            let tiles = u64::from(self.tile_count);
            return (grains / tiles + u64::from(u64::from(tile) < grains % tiles))
                * u64::from(grain);
        }
        self.axes
            .iter()
            .map(|axis| {
                let extent = axis.extent(tile);
                u64::from(extent.physical_end - extent.start)
            })
            .product()
    }

    pub(crate) fn shard_extents(&self) -> Result<Vec<(u16, Vec<ShardExtent>)>, LayoutError> {
        if self.linear_grain.is_some() {
            let shape = &self.shape;
            let rank = shape.0.len();
            let width = u64::from(*shape.0.last().ok_or(LayoutError::EmptyAxisTiling)?);
            let mut all = Vec::new();
            let mut cursor = 0;
            for tile in 0..self.tile_count {
                let start = cursor;
                let physical_end = start + self.tile_elements(tile);
                let end = physical_end.min(shape.elements());
                cursor = physical_end;
                let first_row = start / width;
                let last_row = end.div_ceil(width);
                for row in first_row..last_row {
                    let column_start = if row == first_row { start % width } else { 0 };
                    let column_end = if row + 1 == last_row && !end.is_multiple_of(width) {
                        end % width
                    } else {
                        width
                    };
                    let mut region = Vec::with_capacity(rank);
                    let mut linear_row = row;
                    for axis in (0..rank.saturating_sub(1)).rev() {
                        let extent = u64::from(shape.0[axis]);
                        let coordinate = u32::try_from(linear_row % extent)
                            .map_err(|_| LayoutError::ExtentOverflow(rank))?;
                        linear_row /= extent;
                        region.push(ShardExtent {
                            axis: u16::try_from(axis)
                                .map_err(|_| LayoutError::ExtentOverflow(rank))?,
                            start: coordinate,
                            logical_end: coordinate + 1,
                            physical_end: coordinate + 1,
                        });
                    }
                    region.reverse();
                    region.push(ShardExtent {
                        axis: u16::try_from(rank - 1)
                            .map_err(|_| LayoutError::ExtentOverflow(rank))?,
                        start: u32::try_from(column_start)
                            .map_err(|_| LayoutError::ExtentOverflow(rank))?,
                        logical_end: u32::try_from(column_end)
                            .map_err(|_| LayoutError::ExtentOverflow(rank))?,
                        // Flat padding belongs only to the final local row;
                        // it must not become another logical row or wrap axes.
                        physical_end: u32::try_from(
                            column_end
                                + if row + 1 == last_row {
                                    physical_end - end
                                } else {
                                    0
                                },
                        )
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
        let physical_width = allocated
            .checked_next_multiple_of(self.shard_padding_multiple)
            .ok_or(LayoutError::ExtentOverflow(0))?;
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

impl Layout {
    /// Returns axis-wise padded extents. Flat ownership padding is allocation
    /// tail space, not a rectangular extension of the logical tensor; its size
    /// is described by resolved tile sizes and shard physical bounds instead.
    pub fn padded_shape(&self, shape: &TensorShape) -> Result<TensorShape, LayoutError> {
        if self.tiling.tile_count == 0 || self.tiling.replicas == 0 {
            return Err(LayoutError::EmptyTileGroup);
        }
        if let Some(grain) = self.tiling.linear_grain() {
            let elements = shape.elements();
            if shape.0.is_empty()
                || grain == 0
                || elements.div_ceil(u64::from(grain)) < u64::from(self.tiling.tile_count)
                || (self.tiling.axes[0].padding == Padding::Reject
                    && !elements.is_multiple_of(u64::from(grain)))
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
                let padded_group_extent = group_extent
                    .checked_next_multiple_of(tiling.padding_multiple)
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
        // The validated tile count is replicas times the axis-partition product.
        let coordinate_count = usize::from(self.tiling.tile_count / self.tiling.replicas);
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

/// Project the output's ownership onto non-broadcast operand dimensions.
/// Replicating a whole multi-row parameter and then selecting its columns
/// leaves strided views; partition it before dispatch instead.
pub(crate) fn broadcast_operand_tiling(
    input: &TensorType,
    output: &TensorType,
) -> Option<TensorTiling> {
    let indexing = crate::tensor::Broadcast::new(&input.shape.0, &output.shape.0)?;
    if input.shape == output.shape {
        return Some(output.format.layout.tiling.clone());
    }
    project_tiling(output, |axis| indexing.input_axis(axis))
}

pub(crate) fn axis_tiling(tensor: &TensorType, axis: usize) -> Option<&AxisTiling> {
    tensor
        .format
        .layout
        .tiling
        .axes
        .iter()
        .find(|dim| dim.axis.resolve(tensor.shape.0.len()) == Ok(axis))
}

/// Keep physical tile strides while omitted distributed axes become replicas.
pub(crate) fn project_tiling(
    output: &TensorType,
    map_axis: impl Fn(usize) -> Option<usize>,
) -> Option<TensorTiling> {
    let tiling = &output.format.layout.tiling;
    let mut replicas = tiling.replicas;
    let mut axes = Vec::new();
    for (dim, stride) in tiling.axes.iter().zip(tiling.axis_strides().ok()?) {
        if let Some(mapped) = map_axis(dim.axis.resolve(output.shape.0.len()).ok()?) {
            let mut dim = *dim;
            dim.axis = TensorAxis::FromStart(u16::try_from(mapped).ok()?);
            dim.tile_stride = Some(u16::try_from(stride).ok()?);
            axes.push(dim);
        } else {
            replicas = replicas.checked_mul(dim.partitions)?;
        }
    }
    Some(TensorTiling {
        tile_count: tiling.tile_count,
        replicas,
        axes,
    })
}

pub(crate) fn same_distribution(a: &TensorType, b: &TensorType) -> bool {
    a.shape == b.shape
        && a.format.layout.tiling.tile_count == b.format.layout.tiling.tile_count
        && a.format
            .layout
            .resolve(&a.shape)
            .ok()
            .zip(b.format.layout.resolve(&b.shape).ok())
            .is_some_and(|(left, right)| {
                left.padded_shape == right.padded_shape
                    && (a.format.layout.tiling == b.format.layout.tiling
                        || left.axes().zip(right.axes()).is_some_and(|(a, b)| {
                            a.len() == b.len()
                                && a.iter().zip(b).all(|(a, b)| a.same_partitioning(b))
                        }))
            })
}

#[cfg(test)]
mod linear_padding_tests {
    use super::*;

    #[test]
    fn padded_linear_shards_cover_the_tensor_once_and_account_for_only_one_tail() {
        let mut rng = fastrand::Rng::with_seed(0xf1a7_7a11);
        for _ in 0..256 {
            let shape = TensorShape((0..rng.usize(1..=4)).map(|_| rng.u32(1..=13)).collect());
            let grain = rng.u32(1..=32);
            let elements = shape.elements();
            let tiles = rng.u16(1..=elements.div_ceil(u64::from(grain)).min(32) as u16);
            let mut layout = Layout::logical_linear(tiles, grain);
            assert_eq!(
                layout.resolve(&shape).is_ok(),
                elements.is_multiple_of(u64::from(grain))
            );
            layout.tiling.axes[0].padding = Padding::Zero;
            let resolved = layout.resolve(&shape).unwrap();
            let mut cursor = 0;
            let mut allocated = vec![0; usize::from(tiles)];
            let format = crate::TensorFormat {
                precision: crate::Precision::F16,
                layout: layout.clone(),
            };
            for (tile, extents) in resolved.shard_extents().unwrap() {
                let start = extents
                    .iter()
                    .zip(&shape.0)
                    .fold(0u64, |offset, (extent, &width)| {
                        offset * u64::from(width) + u64::from(extent.start)
                    });
                assert_eq!(start, cursor);
                let last = extents.last().unwrap();
                cursor += u64::from(last.logical_end - last.start);
                let bytes = crate::storage::storage_bytes(crate::storage::TensorStorage {
                    format: &format,
                    extents: &extents,
                })
                .unwrap();
                allocated[usize::from(tile)] += u64::from(bytes) / 2;
                if cursor != elements {
                    assert_eq!(last.logical_end, last.physical_end);
                }
            }
            assert_eq!(cursor, elements);
            assert_eq!(
                allocated.iter().sum::<u64>(),
                elements.div_ceil(u64::from(grain)) * u64::from(grain)
            );
            assert_eq!(
                allocated.iter().copied().max().unwrap(),
                resolved.maximum_tile_elements()
            );
            assert_eq!(allocated.iter().sum::<u64>(), resolved.physical_elements());
            for (tile, &size) in allocated.iter().enumerate() {
                assert_eq!(size, resolved.tile_elements(tile as u16));
            }
            if shape.0.len() == 1 {
                let mut axis = layout.clone();
                axis.tiling.axes[0].axis = TensorAxis::FromEnd(1);
                assert_eq!(
                    resolved.shard_extents().unwrap(),
                    axis.resolve(&shape).unwrap().shard_extents().unwrap()
                );
            }
        }
    }
}
