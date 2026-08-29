//! Address-independent layout-conversion routes and copy geometry.

use crate::graph::TensorShape;
use crate::layout::{
    AMP_COLUMN_MICRO, Layout, LayoutError, NativeKernelOrder, StorageOrder, TensorRegion,
};
use crate::operator::{OperandRequirement, Precision, TileKernelSpec};
use crate::storage::{StorageError, amp_micro_dimension, physical_byte_offset};
use std::collections::BTreeMap;

/// Address-independent strategy for materializing a format conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConversionStrategy {
    LocalKernel,
    DirectRetile,
    DirectLogical,
    StageLogicalThenTransform,
}

impl ConversionStrategy {
    pub const fn uses_intersections(self) -> bool {
        !matches!(self, Self::LocalKernel)
    }
}

pub fn layout_conversion_strategy(
    precision: Precision,
    from: &Layout,
    to: &Layout,
) -> ConversionStrategy {
    if from.order == to.order {
        ConversionStrategy::DirectRetile
    } else if precision == Precision::F32
        || matches!(from.order, StorageOrder::Linear)
            && matches!(
                to.order,
                StorageOrder::Native(NativeKernelOrder::Left | NativeKernelOrder::Output)
            )
        || matches!(to.order, StorageOrder::Linear)
            && matches!(
                from.order,
                StorageOrder::Native(NativeKernelOrder::Left | NativeKernelOrder::Output)
            )
        || matches!(
            (from.order, to.order),
            (StorageOrder::Blocked(source), StorageOrder::Blocked(destination))
                if source.axes == destination.axes
                    && source.permutation == destination.permutation
        )
    {
        ConversionStrategy::DirectLogical
    } else {
        ConversionStrategy::StageLogicalThenTransform
    }
}

/// Resolved copy recipe consumed by costing and low-level lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversionPlan {
    pub kernel: TileKernelSpec,
    pub input: OperandRequirement,
    pub output: OperandRequirement,
    pub strategy: ConversionStrategy,
    pub mappings: Vec<ConversionMapping>,
}

/// One arbitrary-rank affine copy nest within a conversion route.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CopyGeometry {
    pub source_offset: u32,
    pub destination_offset: u32,
    pub contiguous_bytes: u32,
    /// Inner to outer affine dimensions.
    pub dimensions: Vec<CopyDimension>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CopyDimension {
    pub count: u32,
    pub source_stride: u32,
    pub destination_stride: u32,
}

impl CopyGeometry {
    pub fn copy_count(&self) -> u64 {
        self.dimensions.iter().fold(1, |copies, dimension| {
            copies.saturating_mul(u64::from(dimension.count))
        })
    }

    pub fn bytes(&self) -> u64 {
        u64::from(self.contiguous_bytes).saturating_mul(self.copy_count())
    }

    pub(crate) fn offsets(&self) -> Option<Vec<(u32, u32)>> {
        let mut offsets = vec![(self.source_offset, self.destination_offset)];
        for dimension in &self.dimensions {
            let inner = offsets.clone();
            offsets.clear();
            offsets.reserve(inner.len().checked_mul(dimension.count as usize)?);
            for index in 0..dimension.count {
                let source_delta = index.checked_mul(dimension.source_stride)?;
                let destination_delta = index.checked_mul(dimension.destination_stride)?;
                for &(source, destination) in &inner {
                    offsets.push((
                        source.checked_add(source_delta)?,
                        destination.checked_add(destination_delta)?,
                    ));
                }
            }
        }
        Some(offsets)
    }
}

/// A semantic intersection between one resolved source shard and one
/// resolved destination shard. Regions remain address-independent; low
/// lowering binds them to the corresponding shard IDs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversionMapping {
    pub source_shard: u32,
    pub source_tile: u16,
    pub source_storage: TensorRegion,
    pub destination_shard: u32,
    pub destination_tile: u16,
    pub destination_storage: TensorRegion,
    pub region: TensorRegion,
    /// Local copies from the source layout into word-aligned transfer staging.
    pub source_copies: Vec<CopyGeometry>,
    pub copies: Vec<CopyGeometry>,
    /// Local copies from word-aligned transfer staging into the planned
    /// destination storage.
    pub destination_copies: Vec<CopyGeometry>,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ConversionGeometryError {
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error("layout conversion has no regular block geometry")]
    Unsupported,
    #[error("layout conversion geometry overflowed")]
    Overflow,
}

/// Resolves ownership intersections and their regular local/remote copy
/// nests from tensor shape and layouts. No SRAM addresses or low-level shard
/// identities participate in this decision.
pub(crate) fn plan_conversion(
    shape: &TensorShape,
    precision: Precision,
    from: &Layout,
    to: &Layout,
    strategy: ConversionStrategy,
) -> Result<Vec<ConversionMapping>, ConversionGeometryError> {
    if strategy == ConversionStrategy::LocalKernel {
        return Ok(Vec::new());
    }
    let sources = from.resolve(shape)?.shard_extents();
    let destinations = to.resolve(shape)?.shard_extents();
    let mut mappings = Vec::new();
    for (destination_index, destination) in destinations.into_iter().enumerate() {
        let mut intersections = BTreeMap::<TensorRegion, _>::new();
        for (source_index, source) in sources.iter().enumerate() {
            let physical_retile =
                from.order == to.order && strategy == ConversionStrategy::DirectRetile;
            let source_region = if physical_retile {
                source.extents.physical()
            } else {
                source.extents.logical()
            };
            let destination_region = if physical_retile {
                destination.extents.physical()
            } else {
                destination.extents.logical()
            };
            let Some(region) = source_region.intersection(&destination_region) else {
                continue;
            };
            let selected = intersections
                .entry(region)
                .or_insert((source_index, source));
            if source.tile == destination.tile {
                *selected = (source_index, source);
            }
        }
        for (region, (source_index, source)) in intersections {
            let destination_storage = if strategy == ConversionStrategy::StageLogicalThenTransform {
                destination.extents.logical()
            } else {
                destination.extents.clone()
            };
            let (source_order, destination_order, logical_order) = match strategy {
                ConversionStrategy::DirectRetile if from.order == to.order => {
                    (from.order, to.order, false)
                }
                ConversionStrategy::DirectLogical => (from.order, to.order, true),
                ConversionStrategy::StageLogicalThenTransform => (
                    from.order,
                    StorageOrder::Linear,
                    from.order != StorageOrder::Linear,
                ),
                ConversionStrategy::DirectRetile | ConversionStrategy::LocalKernel => {
                    return Err(ConversionGeometryError::Unsupported);
                }
            };
            let direct = copy_geometries(
                precision,
                source_order,
                destination_order,
                &source.extents,
                &destination_storage,
                &region,
                source.tile == destination.tile,
                logical_order,
            );
            let (source_copies, copies, destination_copies) = match direct {
                Ok(copies) => (Vec::new(), copies, Vec::new()),
                Err(ConversionGeometryError::Unsupported) => {
                    let logical_storage = region.logical();
                    let source_copies = copy_geometries(
                        precision,
                        source_order,
                        StorageOrder::Linear,
                        &source.extents,
                        &logical_storage,
                        &region,
                        true,
                        source_order != StorageOrder::Linear,
                    )?;
                    let destination_copies = copy_geometries(
                        precision,
                        StorageOrder::Linear,
                        destination_order,
                        &logical_storage,
                        &destination_storage,
                        &region,
                        true,
                        destination_order != StorageOrder::Linear,
                    )?;
                    let bytes = region
                        .iter()
                        .try_fold(precision.bytes() as u32, |bytes, extent| {
                            bytes.checked_mul(extent.logical_end - extent.start)
                        })
                        .ok_or(ConversionGeometryError::Overflow)?;
                    let padded_bytes = bytes.div_ceil(4) * 4;
                    (
                        source_copies,
                        vec![CopyGeometry {
                            source_offset: 0,
                            destination_offset: 0,
                            contiguous_bytes: padded_bytes,
                            dimensions: Vec::new(),
                        }],
                        destination_copies,
                    )
                }
                Err(error) => return Err(error),
            };
            mappings.push(ConversionMapping {
                source_shard: u32::try_from(source_index)
                    .map_err(|_| ConversionGeometryError::Overflow)?,
                source_tile: source.tile,
                source_storage: source.extents.clone(),
                destination_shard: u32::try_from(destination_index)
                    .map_err(|_| ConversionGeometryError::Overflow)?,
                destination_tile: destination.tile,
                destination_storage,
                region,
                source_copies,
                copies,
                destination_copies,
            });
        }
    }
    if mappings.is_empty() {
        return Err(ConversionGeometryError::Unsupported);
    }
    Ok(mappings)
}

fn copy_geometries(
    precision: Precision,
    source_order: StorageOrder,
    destination_order: StorageOrder,
    source_storage: &TensorRegion,
    destination_storage: &TensorRegion,
    region: &TensorRegion,
    local: bool,
    logical_order: bool,
) -> Result<Vec<CopyGeometry>, ConversionGeometryError> {
    if source_storage.len() != destination_storage.len() || source_storage.len() != region.len() {
        return Err(ConversionGeometryError::Unsupported);
    }
    let dimensions = if logical_order {
        vec![1; region.len()]
    } else {
        if source_order != destination_order {
            return Err(ConversionGeometryError::Unsupported);
        }
        physical_block_dimensions(source_order, precision, region)?
    };
    let geometries = match affine_geometries(
        source_order,
        destination_order,
        precision,
        source_storage,
        destination_storage,
        region,
        &dimensions,
        logical_order,
    ) {
        Ok(geometries) => geometries,
        Err(ConversionGeometryError::Unsupported) if local && logical_order => {
            let scalar = vec![1; region.len()];
            affine_geometries(
                source_order,
                destination_order,
                precision,
                source_storage,
                destination_storage,
                region,
                &scalar,
                true,
            )?
        }
        Err(error) => return Err(error),
    };
    let alignment = if local { precision.bytes() as u32 } else { 4 };
    for geometry in &geometries {
        if !geometry.source_offset.is_multiple_of(alignment)
            || !geometry.destination_offset.is_multiple_of(alignment)
            || geometry.contiguous_bytes == 0
            || !geometry.contiguous_bytes.is_multiple_of(alignment)
        {
            return Err(ConversionGeometryError::Unsupported);
        }
    }
    Ok(geometries)
}

fn affine_geometries(
    source_order: StorageOrder,
    destination_order: StorageOrder,
    precision: Precision,
    source_storage: &TensorRegion,
    destination_storage: &TensorRegion,
    region: &TensorRegion,
    dimensions: &[u32],
    logical_order: bool,
) -> Result<Vec<CopyGeometry>, ConversionGeometryError> {
    let mut pending = vec![region.clone()];
    let mut geometries = Vec::new();
    while let Some(region) = pending.pop() {
        if logical_order {
            let cells = [
                (
                    affine_cell_dimensions(source_order, precision, region.len())?,
                    source_storage,
                ),
                (
                    affine_cell_dimensions(destination_order, precision, region.len())?,
                    destination_storage,
                ),
            ];
            let mut split = None;
            'orders: for (dimensions, storage) in cells {
                for (axis, (&dimension, extent)) in
                    dimensions.iter().zip(storage.iter()).enumerate()
                {
                    if dimension == 0 {
                        continue;
                    }
                    let local = region[axis].start - extent.start;
                    let boundary = extent
                        .start
                        .checked_add(
                            local
                                .checked_div(dimension)
                                .and_then(|cell| cell.checked_add(1))
                                .and_then(|cell| cell.checked_mul(dimension))
                                .ok_or(ConversionGeometryError::Overflow)?,
                        )
                        .ok_or(ConversionGeometryError::Overflow)?;
                    if boundary < region[axis].logical_end {
                        split = Some((axis, boundary));
                        break 'orders;
                    }
                }
            }
            if let Some((axis, boundary)) = split {
                let mut first = region.clone();
                let mut second = region;
                first[axis].logical_end = boundary;
                first[axis].physical_end = boundary;
                second[axis].start = boundary;
                pending.push(second);
                pending.push(first);
                continue;
            }
        }
        match affine_geometry(
            source_order,
            destination_order,
            precision,
            source_storage,
            destination_storage,
            &region,
            dimensions,
        ) {
            Ok(geometry) => geometries.push(geometry),
            Err(ConversionGeometryError::Unsupported) => {
                let Some((axis, count)) = region
                    .iter()
                    .enumerate()
                    .filter_map(|(axis, extent)| {
                        let count = (extent.logical_end - extent.start) / dimensions[axis];
                        (count > 1).then_some((axis, count))
                    })
                    .max_by_key(|&(_, count)| count)
                else {
                    return Err(ConversionGeometryError::Unsupported);
                };
                let split = region[axis].start + count.div_ceil(2) * dimensions[axis];
                let mut first = region.clone();
                let mut second = region;
                first[axis].logical_end = split;
                first[axis].physical_end = split;
                second[axis].start = split;
                pending.push(second);
                pending.push(first);
            }
            Err(error) => return Err(error),
        }
    }
    compact_geometries(geometries)
}

fn compact_geometries(
    mut geometries: Vec<CopyGeometry>,
) -> Result<Vec<CopyGeometry>, ConversionGeometryError> {
    geometries.sort_by_key(|geometry| geometry.source_offset);
    loop {
        let mut compacted = Vec::new();
        let mut merged = false;
        let mut index = 0;
        while index < geometries.len() {
            let mut geometry = geometries[index].clone();
            let mut end = index + 1;
            let Some(second) = geometries.get(end).filter(|second| {
                second.contiguous_bytes == geometry.contiguous_bytes
                    && second.dimensions == geometry.dimensions
            }) else {
                compacted.push(geometry);
                index = end;
                continue;
            };
            let Some(source_stride) = second.source_offset.checked_sub(geometry.source_offset)
            else {
                compacted.push(geometry);
                index = end;
                continue;
            };
            let Some(destination_stride) = second
                .destination_offset
                .checked_sub(geometry.destination_offset)
            else {
                compacted.push(geometry);
                index = end;
                continue;
            };
            if source_stride == 0 || destination_stride == 0 {
                compacted.push(geometry);
                index = end;
                continue;
            }
            end += 1;
            while let Some(next) = geometries.get(end) {
                let previous = &geometries[end - 1];
                if next.contiguous_bytes != geometry.contiguous_bytes
                    || next.dimensions != geometry.dimensions
                    || next.source_offset.checked_sub(previous.source_offset) != Some(source_stride)
                    || next
                        .destination_offset
                        .checked_sub(previous.destination_offset)
                        != Some(destination_stride)
                {
                    break;
                }
                end += 1;
            }
            geometry.dimensions.push(CopyDimension {
                count: u32::try_from(end - index).map_err(|_| ConversionGeometryError::Overflow)?,
                source_stride,
                destination_stride,
            });
            compacted.push(geometry);
            merged = true;
            index = end;
        }
        geometries = compacted;
        if !merged {
            return Ok(geometries);
        }
    }
}

fn affine_cell_dimensions(
    order: StorageOrder,
    precision: Precision,
    rank: usize,
) -> Result<Vec<u32>, ConversionGeometryError> {
    let mut dimensions = vec![0; rank];
    if rank < 2 || order == StorageOrder::Linear {
        return Ok(dimensions);
    }
    let row = rank - 2;
    let column = rank - 1;
    match order {
        StorageOrder::Linear => {}
        StorageOrder::Native(NativeKernelOrder::Left) => {
            dimensions[column] = amp_micro_dimension(precision);
        }
        StorageOrder::Native(NativeKernelOrder::TransposedLeft) => {
            dimensions[row] = amp_micro_dimension(precision);
        }
        StorageOrder::Native(NativeKernelOrder::TransposedRight) => {
            dimensions[row] = amp_micro_dimension(precision);
            dimensions[column] = AMP_COLUMN_MICRO;
        }
        StorageOrder::Native(NativeKernelOrder::Output) => dimensions[column] = 2,
        StorageOrder::Native(NativeKernelOrder::TransposedOutput) => dimensions[row] = 2,
        StorageOrder::Blocked(order) => {
            let [row, column] = order.physical_axes(rank)?;
            dimensions[row] = amp_micro_dimension(precision);
            dimensions[column] = u32::from(order.block_shape[1]);
        }
    }
    Ok(dimensions)
}

fn affine_geometry(
    source_order: StorageOrder,
    destination_order: StorageOrder,
    precision: Precision,
    source_storage: &TensorRegion,
    destination_storage: &TensorRegion,
    region: &TensorRegion,
    dimensions: &[u32],
) -> Result<CopyGeometry, ConversionGeometryError> {
    let rank = region.len();
    if rank == 0
        || source_storage.len() != rank
        || destination_storage.len() != rank
        || dimensions.len() != rank
    {
        return Err(ConversionGeometryError::Unsupported);
    }
    for (axis, (((source, destination), region), &block)) in source_storage
        .iter()
        .zip(destination_storage.iter())
        .zip(region.iter())
        .zip(dimensions)
        .enumerate()
    {
        let source_must_align = !(source_order == StorageOrder::Linear && axis + 1 == rank);
        let destination_must_align =
            !(destination_order == StorageOrder::Linear && axis + 1 == rank);
        if source.axis != region.axis
            || destination.axis != region.axis
            || region.start < source.start
            || region.start < destination.start
            || region.logical_end > source.physical_end
            || region.logical_end > destination.physical_end
            || block == 0
            || source_must_align && !(region.start - source.start).is_multiple_of(block)
            || destination_must_align && !(region.start - destination.start).is_multiple_of(block)
            || !(region.logical_end - region.start).is_multiple_of(block)
        {
            return Err(ConversionGeometryError::Unsupported);
        }
    }
    let block_bytes = dimensions
        .iter()
        .try_fold(1u32, |elements, dimension| elements.checked_mul(*dimension))
        .and_then(|elements| elements.checked_mul(precision.bytes() as u32))
        .ok_or(ConversionGeometryError::Overflow)?;
    let starts = region.iter().map(|extent| extent.start).collect::<Vec<_>>();
    let source_offset = physical_byte_offset(source_order, precision, source_storage, &starts)?;
    let destination_offset =
        physical_byte_offset(destination_order, precision, destination_storage, &starts)?;
    let mut copy_dimensions = Vec::new();
    let mut axis_strides = vec![None; rank];
    for axis in 0..rank {
        let count = (region[axis].logical_end - region[axis].start) / dimensions[axis];
        if count <= 1 {
            continue;
        }
        let mut next = starts.clone();
        next[axis] += dimensions[axis];
        let source_stride = physical_byte_offset(source_order, precision, source_storage, &next)?
            .checked_sub(source_offset)
            .ok_or(ConversionGeometryError::Unsupported)?;
        let destination_stride =
            physical_byte_offset(destination_order, precision, destination_storage, &next)?
                .checked_sub(destination_offset)
                .ok_or(ConversionGeometryError::Unsupported)?;
        let mut last = starts.clone();
        last[axis] += (count - 1) * dimensions[axis];
        let source_end = source_offset
            .checked_add((count - 1) * source_stride)
            .ok_or(ConversionGeometryError::Overflow)?;
        let destination_end = destination_offset
            .checked_add((count - 1) * destination_stride)
            .ok_or(ConversionGeometryError::Overflow)?;
        if physical_byte_offset(source_order, precision, source_storage, &last)? != source_end
            || physical_byte_offset(destination_order, precision, destination_storage, &last)?
                != destination_end
        {
            return Err(ConversionGeometryError::Unsupported);
        }
        axis_strides[axis] = Some((source_stride, destination_stride));
        copy_dimensions.push(CopyDimension {
            count,
            source_stride,
            destination_stride,
        });
    }
    let mut last = starts;
    let mut expected_source = source_offset;
    let mut expected_destination = destination_offset;
    for (axis, &dimension) in dimensions.iter().enumerate() {
        let count = (region[axis].logical_end - region[axis].start) / dimension;
        last[axis] += (count - 1) * dimension;
        if let Some((source_stride, destination_stride)) = axis_strides[axis] {
            expected_source = expected_source
                .checked_add((count - 1) * source_stride)
                .ok_or(ConversionGeometryError::Overflow)?;
            expected_destination = expected_destination
                .checked_add((count - 1) * destination_stride)
                .ok_or(ConversionGeometryError::Overflow)?;
        }
    }
    if physical_byte_offset(source_order, precision, source_storage, &last)? != expected_source
        || physical_byte_offset(destination_order, precision, destination_storage, &last)?
            != expected_destination
    {
        return Err(ConversionGeometryError::Unsupported);
    }
    copy_dimensions
        .sort_by_key(|dimension| dimension.source_stride.max(dimension.destination_stride));
    Ok(CopyGeometry {
        source_offset,
        destination_offset,
        contiguous_bytes: block_bytes,
        dimensions: copy_dimensions,
    })
}

fn physical_block_dimensions(
    order: StorageOrder,
    precision: Precision,
    region: &TensorRegion,
) -> Result<Vec<u32>, ConversionGeometryError> {
    let rank = region.len();
    let mut dimensions = vec![1u32; rank];
    if rank == 1 {
        dimensions[0] = region[0].logical_end - region[0].start;
    } else {
        let row = rank - 2;
        let column = rank - 1;
        match order {
            StorageOrder::Linear => {
                dimensions[column] = region[column].logical_end - region[column].start;
            }
            StorageOrder::Native(NativeKernelOrder::Left) => {
                dimensions[column] = amp_micro_dimension(precision);
            }
            StorageOrder::Native(NativeKernelOrder::Output) => {
                dimensions[column] = AMP_COLUMN_MICRO;
            }
            StorageOrder::Native(NativeKernelOrder::TransposedLeft) => {
                dimensions[row] = amp_micro_dimension(precision);
            }
            StorageOrder::Native(NativeKernelOrder::TransposedOutput) => {
                dimensions[row] = AMP_COLUMN_MICRO;
            }
            StorageOrder::Native(NativeKernelOrder::TransposedRight) => {
                dimensions[row] = amp_micro_dimension(precision);
                dimensions[column] = AMP_COLUMN_MICRO;
            }
            StorageOrder::Blocked(order) => {
                let [row, column] = order.physical_axes(rank)?;
                dimensions[row] = amp_micro_dimension(precision);
                dimensions[column] = u32::from(order.block_shape[1]);
            }
        }
    }
    Ok(dimensions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{BlockedOrder, ShardExtent, TensorAxis};
    use std::collections::BTreeMap;

    #[test]
    fn randomized_affine_geometries_preserve_logical_elements() {
        let mut random = fastrand::Rng::with_seed(0x636f_7079_6765_6f6d);
        for case in 0..128 {
            let rank = random.usize(2..=4);
            let mut shape = (0..rank).map(|_| random.u32(1..=3)).collect::<Vec<_>>();
            shape[rank - 2] = 16 * random.u32(1..=3);
            shape[rank - 1] = 16 * random.u32(1..=3);
            let first_axis = random.usize(..rank);
            let second_axis = (first_axis + random.usize(1..rank)) % rank;
            shape[first_axis] = 16 * random.u32(1..=3);
            shape[second_axis] = 16 * random.u32(1..=3);
            let arbitrary_blocked = StorageOrder::Blocked(BlockedOrder {
                axes: [
                    TensorAxis::FromStart(first_axis as u16),
                    TensorAxis::FromStart(second_axis as u16),
                ],
                block_shape: [16, 16],
                permutation: if random.bool() { [0, 1] } else { [1, 0] },
            });
            let orders = [
                StorageOrder::Linear,
                StorageOrder::Blocked(BlockedOrder::matrix(16, 16)),
                StorageOrder::Blocked(BlockedOrder::transposed_matrix(16, 16)),
                arbitrary_blocked,
                StorageOrder::Native(NativeKernelOrder::Left),
                StorageOrder::Native(NativeKernelOrder::TransposedLeft),
                StorageOrder::Native(NativeKernelOrder::TransposedRight),
                StorageOrder::Native(NativeKernelOrder::Output),
                StorageOrder::Native(NativeKernelOrder::TransposedOutput),
            ];
            let storage = shape
                .iter()
                .enumerate()
                .map(|(axis, &end)| ShardExtent {
                    axis: axis as u16,
                    start: 0,
                    logical_end: end,
                    physical_end: end,
                })
                .collect::<Vec<_>>()
                .into();
            let precision = if random.bool() {
                Precision::F16
            } else {
                Precision::F32
            };
            let source_order = orders[random.usize(..orders.len())];
            let destination_order = orders[random.usize(..orders.len())];
            let geometries = copy_geometries(
                precision,
                source_order,
                destination_order,
                &storage,
                &storage,
                &storage,
                true,
                true,
            )
            .unwrap_or_else(|error| {
                panic!("case {case}: {source_order:?} -> {destination_order:?}: {error}")
            });
            let element_bytes = precision.bytes() as u32;
            let mut copied = BTreeMap::new();
            for geometry in geometries {
                for (source, destination) in geometry.offsets().unwrap() {
                    for byte in (0..geometry.contiguous_bytes).step_by(element_bytes as usize) {
                        assert_eq!(
                            copied.insert(source + byte, destination + byte),
                            None,
                            "case {case}: duplicate source element"
                        );
                    }
                }
            }
            let mut coordinates = vec![0; rank];
            loop {
                let source =
                    physical_byte_offset(source_order, precision, &storage, &coordinates).unwrap();
                let destination =
                    physical_byte_offset(destination_order, precision, &storage, &coordinates)
                        .unwrap();
                assert_eq!(
                    copied.remove(&source),
                    Some(destination),
                    "case {case}: {shape:?} {precision:?} {source_order:?} -> {destination_order:?} at {coordinates:?}"
                );
                let mut axis = rank;
                loop {
                    if axis == 0 {
                        assert!(copied.is_empty(), "case {case}: extra copied elements");
                        break;
                    }
                    axis -= 1;
                    coordinates[axis] += 1;
                    if coordinates[axis] < shape[axis] {
                        break;
                    }
                    coordinates[axis] = 0;
                }
                if axis == 0 && coordinates.iter().all(|&coordinate| coordinate == 0) {
                    break;
                }
            }
        }
    }
}
