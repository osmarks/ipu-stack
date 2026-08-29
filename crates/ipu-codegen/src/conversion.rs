//! Address-independent layout-conversion routes and copy geometry.

use crate::graph::TensorShape;
use crate::layout::{
    AMP_COLUMN_MICRO, AmpOrder, BlockMajorOrder, ElementOrder, Layout, LayoutError, TensorRegion,
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
        || matches!(from.order, ElementOrder::RowMajor)
            && matches!(
                to.order,
                ElementOrder::Amp(AmpOrder::Left | AmpOrder::Output)
            )
        || matches!(to.order, ElementOrder::RowMajor)
            && matches!(
                from.order,
                ElementOrder::Amp(AmpOrder::Left | AmpOrder::Output)
            )
        || matches!(
            (from.order, to.order),
            (
                ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. }),
                ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. })
            ) | (
                ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix { .. }),
                ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix { .. })
            )
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
    pub copies: Vec<CopyGeometry>,
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

#[derive(Clone)]
struct StorageBlock {
    origin: Vec<u32>,
    offset: u32,
    bytes: u32,
}

#[derive(Clone)]
struct CopyRun {
    source_offset: u32,
    destination_offset: u32,
    bytes: u32,
}

/// Resolves ownership intersections and their regular local/remote copy
/// nests from tensor shape and layouts. No SRAM addresses or low-level shard
/// identities participate in this decision.
pub(crate) fn plan_conversion_mappings(
    shape: &TensorShape,
    precision: Precision,
    from: &Layout,
    to: &Layout,
    strategy: ConversionStrategy,
) -> Result<Vec<ConversionMapping>, ConversionGeometryError> {
    plan_mappings(shape, precision, from, to, strategy, true)
}

pub(crate) fn plan_semantic_mappings(
    shape: &TensorShape,
    precision: Precision,
    from: &Layout,
    to: &Layout,
    strategy: ConversionStrategy,
) -> Result<Vec<ConversionMapping>, ConversionGeometryError> {
    plan_mappings(shape, precision, from, to, strategy, false)
}

fn plan_mappings(
    shape: &TensorShape,
    precision: Precision,
    from: &Layout,
    to: &Layout,
    strategy: ConversionStrategy,
    regular: bool,
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
            let Some(region) = source
                .extents
                .logical()
                .intersection(&destination.extents.logical())
            else {
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
            let copies = if regular {
                let (source_order, destination_order, logical_order) = match strategy {
                    ConversionStrategy::DirectRetile if from.order == to.order => {
                        (from.order, to.order, false)
                    }
                    ConversionStrategy::DirectLogical => (from.order, to.order, true),
                    ConversionStrategy::StageLogicalThenTransform
                        if from.order == ElementOrder::RowMajor =>
                    {
                        (ElementOrder::RowMajor, ElementOrder::RowMajor, false)
                    }
                    ConversionStrategy::DirectRetile
                    | ConversionStrategy::StageLogicalThenTransform
                    | ConversionStrategy::LocalKernel => {
                        return Err(ConversionGeometryError::Unsupported);
                    }
                };
                copy_geometries(
                    precision,
                    source_order,
                    destination_order,
                    &source.extents,
                    &destination_storage,
                    &region,
                    source.tile == destination.tile,
                    logical_order,
                )?
            } else {
                Vec::new()
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
                copies,
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
    source_order: ElementOrder,
    destination_order: ElementOrder,
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
        logical_block_dimensions(source_order, destination_order, precision, region)?
    } else {
        if source_order != destination_order {
            return Err(ConversionGeometryError::Unsupported);
        }
        physical_block_dimensions(source_order, precision, region)?
    };
    let source = storage_blocks(source_order, precision, source_storage, region, &dimensions)?;
    let destination = storage_blocks(
        destination_order,
        precision,
        destination_storage,
        region,
        &dimensions,
    )?;
    let mut destinations = destination
        .into_iter()
        .map(|block| (block.origin.clone(), block))
        .collect::<BTreeMap<_, _>>();
    let mut source = source;
    source.sort_by_key(|block| block.offset);
    let mut runs = Vec::<CopyRun>::with_capacity(source.len());
    for block in source {
        let destination = destinations
            .remove(&block.origin)
            .ok_or(ConversionGeometryError::Unsupported)?;
        if destination.bytes != block.bytes {
            return Err(ConversionGeometryError::Unsupported);
        }
        if let Some(previous) = runs.last_mut()
            && previous.source_offset.checked_add(previous.bytes) == Some(block.offset)
            && previous.destination_offset.checked_add(previous.bytes) == Some(destination.offset)
        {
            previous.bytes = previous
                .bytes
                .checked_add(block.bytes)
                .ok_or(ConversionGeometryError::Overflow)?;
            continue;
        }
        runs.push(CopyRun {
            source_offset: block.offset,
            destination_offset: destination.offset,
            bytes: block.bytes,
        });
    }
    if !destinations.is_empty() {
        return Err(ConversionGeometryError::Unsupported);
    }
    let mut geometries = runs
        .into_iter()
        .map(|run| CopyGeometry {
            source_offset: run.source_offset,
            destination_offset: run.destination_offset,
            contiguous_bytes: run.bytes,
            dimensions: Vec::new(),
        })
        .collect::<Vec<_>>();
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
                    || next.source_offset.checked_sub(previous.source_offset)
                        != Some(source_stride)
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
                count: u32::try_from(end - index)
                    .map_err(|_| ConversionGeometryError::Overflow)?,
                source_stride,
                destination_stride,
            });
            compacted.push(geometry);
            merged = true;
            index = end;
        }
        geometries = compacted;
        if !merged {
            break;
        }
    }
    let alignment = if local {
        precision.bytes() as u32
    } else {
        4
    };
    for geometry in &geometries {
        if !geometry.source_offset.is_multiple_of(alignment)
            || !geometry.destination_offset.is_multiple_of(alignment)
            || geometry.contiguous_bytes == 0
            || !geometry.contiguous_bytes.is_multiple_of(alignment)
        {
            return Err(ConversionGeometryError::Unsupported);
        }
    }
    if local {
        geometries = split_unsupported_local_nests(geometries)?;
    }
    Ok(geometries)
}

fn split_unsupported_local_nests(
    geometries: Vec<CopyGeometry>,
) -> Result<Vec<CopyGeometry>, ConversionGeometryError> {
    let mut supported = Vec::new();
    for mut geometry in geometries {
        let retain_inner = geometry.dimensions.first().is_some_and(|dimension| {
            geometry.contiguous_bytes.is_multiple_of(8)
                && u64::from(geometry.contiguous_bytes) * u64::from(dimension.count) <= 512
        });
        let outer = geometry.dimensions.split_off(usize::from(retain_inner));
        let offsets = CopyGeometry {
            dimensions: outer,
            ..geometry.clone()
        }
        .offsets()
        .ok_or(ConversionGeometryError::Overflow)?;
        for (source_offset, destination_offset) in offsets {
            supported.push(CopyGeometry {
                source_offset,
                destination_offset,
                ..geometry.clone()
            });
        }
    }
    Ok(supported)
}

fn storage_blocks(
    order: ElementOrder,
    precision: Precision,
    storage: &TensorRegion,
    region: &TensorRegion,
    dimensions: &[u32],
) -> Result<Vec<StorageBlock>, ConversionGeometryError> {
    let rank = storage.len();
    if rank == 0 || rank != region.len() || rank != dimensions.len() {
        return Err(ConversionGeometryError::Unsupported);
    }
    let dimensions = dimensions.to_vec();
    for (axis, ((storage, region), &block)) in storage
        .iter()
        .zip(region.iter())
        .zip(&dimensions)
        .enumerate()
    {
        let start_must_align = !(order == ElementOrder::RowMajor && axis + 1 == rank);
        if storage.axis != region.axis
            || region.start < storage.start
            || region.logical_end > storage.logical_end
            || block == 0
            || start_must_align && !(region.start - storage.start).is_multiple_of(block)
            || !(region.logical_end - region.start).is_multiple_of(block)
        {
            return Err(ConversionGeometryError::Unsupported);
        }
    }
    let block_elements = dimensions
        .iter()
        .try_fold(1u32, |elements, dimension| elements.checked_mul(*dimension));
    let block_bytes = block_elements
        .and_then(|elements| elements.checked_mul(precision.bytes() as u32))
        .ok_or(ConversionGeometryError::Overflow)?;
    let starts = region.iter().map(|extent| extent.start).collect::<Vec<_>>();
    let ends = region
        .iter()
        .map(|extent| extent.logical_end)
        .collect::<Vec<_>>();
    let mut coordinates = starts.clone();
    let mut blocks = Vec::new();
    loop {
        blocks.push(StorageBlock {
            origin: coordinates.clone(),
            offset: physical_byte_offset(order, precision, storage, &coordinates)?,
            bytes: block_bytes,
        });
        let mut axis = rank;
        loop {
            if axis == 0 {
                return Ok(blocks);
            }
            axis -= 1;
            coordinates[axis] = coordinates[axis]
                .checked_add(dimensions[axis])
                .ok_or(ConversionGeometryError::Overflow)?;
            if coordinates[axis] < ends[axis] {
                break;
            }
            coordinates[axis] = starts[axis];
        }
    }
}

fn physical_block_dimensions(
    order: ElementOrder,
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
            ElementOrder::RowMajor => {
                dimensions[column] = region[column].logical_end - region[column].start;
            }
            ElementOrder::Amp(AmpOrder::Left) => {
                dimensions[column] = amp_micro_dimension(precision);
            }
            ElementOrder::Amp(AmpOrder::Output) => {
                dimensions[column] = AMP_COLUMN_MICRO;
            }
            ElementOrder::Amp(AmpOrder::TransposedLeft) => {
                dimensions[row] = amp_micro_dimension(precision);
            }
            ElementOrder::Amp(AmpOrder::TransposedOutput) => {
                dimensions[row] = AMP_COLUMN_MICRO;
            }
            ElementOrder::Amp(AmpOrder::TransposedRight) => {
                dimensions[row] = amp_micro_dimension(precision);
                dimensions[column] = AMP_COLUMN_MICRO;
            }
            ElementOrder::BlockMajor(BlockMajorOrder::Matrix { column_block, .. }) => {
                dimensions[row] = amp_micro_dimension(precision);
                dimensions[column] = u32::from(column_block);
            }
            ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix {
                column_block, ..
            }) => {
                dimensions[row] = u32::from(column_block);
                dimensions[column] = amp_micro_dimension(precision);
            }
        }
    }
    Ok(dimensions)
}

fn logical_block_dimensions(
    source: ElementOrder,
    destination: ElementOrder,
    precision: Precision,
    region: &TensorRegion,
) -> Result<Vec<u32>, ConversionGeometryError> {
    let rank = region.len();
    if rank == 0 {
        return Err(ConversionGeometryError::Unsupported);
    }
    let mut dimensions = vec![1; rank];
    match (source, destination) {
        (
            ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                column_block: source_columns,
                ..
            }),
            ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                column_block: destination_columns,
                ..
            }),
        ) if rank >= 2 => {
            dimensions[rank - 2] = amp_micro_dimension(precision);
            dimensions[rank - 1] = gcd(u32::from(source_columns), u32::from(destination_columns));
            return Ok(dimensions);
        }
        (
            ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix {
                column_block: source_rows,
                ..
            }),
            ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix {
                column_block: destination_rows,
                ..
            }),
        ) if rank >= 2 => {
            dimensions[rank - 2] = gcd(u32::from(source_rows), u32::from(destination_rows));
            dimensions[rank - 1] = amp_micro_dimension(precision);
            return Ok(dimensions);
        }
        _ => {}
    }
    let contiguous_columns = |order| match order {
        ElementOrder::RowMajor => Some(region[rank - 1].logical_end - region[rank - 1].start),
        ElementOrder::Amp(AmpOrder::Left) => Some(amp_micro_dimension(precision)),
        ElementOrder::Amp(AmpOrder::Output) => Some(2),
        ElementOrder::BlockMajor(_)
        | ElementOrder::Amp(
            AmpOrder::TransposedLeft | AmpOrder::TransposedRight | AmpOrder::TransposedOutput,
        ) if precision == Precision::F32 => Some(1),
        ElementOrder::BlockMajor(_) | ElementOrder::Amp(_) => None,
    };
    let source_columns = contiguous_columns(source).ok_or(ConversionGeometryError::Unsupported)?;
    let destination_columns =
        contiguous_columns(destination).ok_or(ConversionGeometryError::Unsupported)?;
    let columns = gcd(source_columns, destination_columns);
    if columns
        .checked_mul(precision.bytes() as u32)
        .is_none_or(|bytes| bytes < 4)
    {
        return Err(ConversionGeometryError::Unsupported);
    }
    dimensions[rank - 1] = columns;
    Ok(dimensions)
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}
