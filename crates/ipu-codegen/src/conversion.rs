//! Address-independent layout-conversion routes and copy geometry.

use crate::graph::TensorShape;
use crate::layout::{
    AMP_COLUMN_MICRO, AmpOrder, BlockMajorOrder, ElementOrder, Layout, LayoutError, TensorRegion,
};
use crate::operator::{ConversionStrategy, Precision};
use crate::storage::{StorageError, amp_micro_dimension, physical_byte_offset};
use std::collections::BTreeMap;

/// One regular copy nest within a conversion route.
///
/// `contiguous_axes` identifies the semantic axes folded into each row.
/// `repeated_axis` identifies the next layout axis traversed by the two byte
/// strides. A single-row geometry is an ordinary contiguous copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CopyGeometry {
    pub source_offset: u32,
    pub destination_offset: u32,
    /// Half-open semantic axis range folded into one contiguous row.
    pub contiguous_axes: (u16, u16),
    pub repeated_axis: Option<u16>,
    pub rows: u32,
    pub row_bytes: u32,
    pub source_stride: u32,
    pub destination_stride: u32,
}

impl CopyGeometry {
    pub fn bytes(&self) -> u64 {
        u64::from(self.rows) * u64::from(self.row_bytes)
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
    contiguous_start: u16,
}

#[derive(Clone)]
struct CopyRun {
    origin: Vec<u32>,
    source_offset: u32,
    destination_offset: u32,
    bytes: u32,
    contiguous_start: u16,
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
            let local = source.tile == destination.tile;
            let destination_storage = if strategy == ConversionStrategy::StageLogicalThenTransform {
                destination.extents.logical()
            } else {
                destination.extents.clone()
            };
            let copies = copy_geometries(
                precision,
                source_order,
                destination_order,
                &source.extents,
                &destination_storage,
                &region,
                local,
                logical_order,
            )?;
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
            previous.contiguous_start = previous
                .contiguous_start
                .min(block.contiguous_start)
                .min(destination.contiguous_start);
            continue;
        }
        runs.push(CopyRun {
            origin: block.origin,
            source_offset: block.offset,
            destination_offset: destination.offset,
            bytes: block.bytes,
            contiguous_start: block.contiguous_start.min(destination.contiguous_start),
        });
    }
    if !destinations.is_empty() {
        return Err(ConversionGeometryError::Unsupported);
    }
    let rank = u16::try_from(region.len()).map_err(|_| ConversionGeometryError::Overflow)?;
    let mut geometries = Vec::new();
    let mut index = 0;
    while index < runs.len() {
        let first = &runs[index];
        let mut end = index + 1;
        let mut repeated_axis = None;
        let mut source_stride = first.bytes;
        let mut destination_stride = first.bytes;
        if let Some(second) = runs.get(end)
            && second.bytes == first.bytes
        {
            repeated_axis = changed_axis(&first.origin, &second.origin);
            source_stride = second
                .source_offset
                .checked_sub(first.source_offset)
                .unwrap_or(0);
            destination_stride = second
                .destination_offset
                .checked_sub(first.destination_offset)
                .unwrap_or(0);
            if repeated_axis.is_some() && source_stride != 0 && destination_stride != 0 {
                end += 1;
                while let Some(run) = runs.get(end) {
                    let previous = &runs[end - 1];
                    if run.bytes != first.bytes
                        || changed_axis(&previous.origin, &run.origin) != repeated_axis
                        || run.source_offset.checked_sub(previous.source_offset)
                            != Some(source_stride)
                        || run
                            .destination_offset
                            .checked_sub(previous.destination_offset)
                            != Some(destination_stride)
                    {
                        break;
                    }
                    end += 1;
                }
            } else {
                repeated_axis = None;
                source_stride = first.bytes;
                destination_stride = first.bytes;
                end = index + 1;
            }
        }
        let rows = u32::try_from(end - index).map_err(|_| ConversionGeometryError::Overflow)?;
        let geometry = CopyGeometry {
            source_offset: first.source_offset,
            destination_offset: first.destination_offset,
            contiguous_axes: (first.contiguous_start, rank),
            repeated_axis,
            rows,
            row_bytes: first.bytes,
            source_stride,
            destination_stride,
        };
        if geometry.source_offset & 0b11 != 0
            || geometry.destination_offset & 0b11 != 0
            || geometry.row_bytes == 0
            || geometry.row_bytes & 0b11 != 0
        {
            return Err(ConversionGeometryError::Unsupported);
        }
        if local && geometry.rows > 1 && (geometry.row_bytes & 0b111 != 0 || geometry.bytes() > 512)
        {
            for row in 0..geometry.rows {
                geometries.push(CopyGeometry {
                    source_offset: geometry
                        .source_offset
                        .checked_add(
                            row.checked_mul(geometry.source_stride)
                                .ok_or(ConversionGeometryError::Overflow)?,
                        )
                        .ok_or(ConversionGeometryError::Overflow)?,
                    destination_offset: geometry
                        .destination_offset
                        .checked_add(
                            row.checked_mul(geometry.destination_stride)
                                .ok_or(ConversionGeometryError::Overflow)?,
                        )
                        .ok_or(ConversionGeometryError::Overflow)?,
                    contiguous_axes: geometry.contiguous_axes,
                    repeated_axis: None,
                    rows: 1,
                    row_bytes: geometry.row_bytes,
                    source_stride: geometry.row_bytes,
                    destination_stride: geometry.row_bytes,
                });
            }
        } else {
            geometries.push(geometry);
        }
        index = end;
    }
    Ok(geometries)
}

fn changed_axis(left: &[u32], right: &[u32]) -> Option<u16> {
    let mut changed = None;
    for (axis, (&left, &right)) in left.iter().zip(right).enumerate() {
        if left == right {
            continue;
        }
        if right < left || changed.is_some() {
            return None;
        }
        changed = u16::try_from(axis).ok();
    }
    changed
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
    let contiguous_start = dimensions
        .iter()
        .position(|dimension| *dimension > 1)
        .unwrap_or(rank - 1);
    let contiguous_start =
        u16::try_from(contiguous_start).map_err(|_| ConversionGeometryError::Overflow)?;
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
            contiguous_start,
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
            dimensions[rank - 1] = gcd(
                u32::from(source_columns),
                u32::from(destination_columns),
            );
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
