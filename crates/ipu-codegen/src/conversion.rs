//! Address-independent layout-conversion routes and copy geometry.

use crate::graph::TensorShape;
use crate::layout::{
    AMP_COLUMN_MICRO, Layout, LayoutError, NativeKernelOrder, StorageOrder, TensorRegion,
    TensorType,
};
use crate::mid::{MidGraph, MidOperation, MidOperationKind, MidValue};
use crate::operator::Precision;
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
    pub source_region: TensorRegion,
    pub destination_region: TensorRegion,
    /// Local copies from the source layout into word-aligned transfer staging.
    pub source_copies: Vec<CopyGeometry>,
    pub copies: Vec<CopyGeometry>,
    /// Local copies from word-aligned transfer staging into the planned
    /// destination storage.
    pub destination_copies: Vec<CopyGeometry>,
}

/// A logical view whose physical materialization may be deferred until a
/// consumer requests bounded slices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeferredTransform {
    /// Split the final input axis into `parts` equal-width slices and fold the
    /// new part axis into the leading output axis.
    SplitLastAxisIntoLeading { parts: u32 },
}

impl DeferredTransform {
    pub fn map_slices(
        self,
        source_shape: &TensorShape,
        output_shape: &TensorShape,
        output: &TensorRegion,
    ) -> Option<Vec<DeferredSliceMapping>> {
        let streams = output.extents.first()?;
        (streams.start..streams.logical_end)
            .map(|stream| {
                let mut slice = output.clone();
                slice.extents[0].start = stream;
                slice.extents[0].logical_end = stream.checked_add(1)?;
                slice.extents[0].physical_end = stream.checked_add(1)?;
                self.map_slice(source_shape, output_shape, &slice)
            })
            .collect()
    }

    pub(crate) fn map_slice(
        self,
        source_shape: &TensorShape,
        output_shape: &TensorShape,
        output: &TensorRegion,
    ) -> Option<DeferredSliceMapping> {
        let Self::SplitLastAxisIntoLeading { parts } = self;
        let [source_batch, source_rows, source_columns] = source_shape.0.as_slice() else {
            return None;
        };
        let [output_streams, output_rows, output_columns] = output_shape.0.as_slice() else {
            return None;
        };
        let [stream, rows, columns] = output.extents.as_slice() else {
            return None;
        };
        if parts == 0
            || stream.axis != 0
            || rows.axis != 1
            || columns.axis != 2
            || stream.logical_end != stream.start.checked_add(1)?
            || *output_streams != source_batch.checked_mul(parts)?
            || output_rows != source_rows
            || source_columns != &output_columns.checked_mul(parts)?
            || stream.logical_end > *output_streams
            || rows.logical_end > *output_rows
            || columns.logical_end > *output_columns
        {
            return None;
        }
        let batch = stream.start / parts;
        let column_base = (stream.start % parts).checked_mul(*output_columns)?;
        Some(DeferredSliceMapping {
            source: TensorRegion::new([
                crate::ShardExtent {
                    axis: 0,
                    start: batch,
                    logical_end: batch.checked_add(1)?,
                    physical_end: batch.checked_add(1)?,
                },
                *rows,
                crate::ShardExtent {
                    axis: 2,
                    start: column_base.checked_add(columns.start)?,
                    logical_end: column_base.checked_add(columns.logical_end)?,
                    physical_end: column_base.checked_add(columns.logical_end)?,
                },
            ]),
            destination: output.clone(),
            source_axes: vec![None, Some(1), Some(2)],
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeferredSliceMapping {
    pub source: TensorRegion,
    pub destination: TensorRegion,
    /// Source axis corresponding to each destination axis. `None` selects a
    /// fixed source coordinate.
    pub source_axes: Vec<Option<usize>>,
}

impl DeferredSliceMapping {
    pub fn project_source(&self, source: &TensorRegion) -> Option<TensorRegion> {
        if source.len() != self.source.len() || self.source_axes.len() != self.destination.len() {
            return None;
        }
        self.source_axes
            .iter()
            .zip(self.destination.iter())
            .enumerate()
            .map(|(axis, (source_axis, destination))| {
                let Some(source_axis) = source_axis else {
                    return Some(*destination);
                };
                let selected = source.get(*source_axis)?;
                let base = self.source.get(*source_axis)?.start;
                Some(crate::ShardExtent {
                    axis: u16::try_from(axis).ok()?,
                    start: destination
                        .start
                        .checked_add(selected.start.checked_sub(base)?)?,
                    logical_end: destination
                        .start
                        .checked_add(selected.logical_end.checked_sub(base)?)?,
                    physical_end: destination
                        .start
                        .checked_add(selected.logical_end.checked_sub(base)?)?,
                })
            })
            .collect()
    }
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
            mappings.push(plan_mapping(
                precision,
                from.order,
                to.order,
                strategy,
                ConversionMapping {
                    source_shard: u32::try_from(source_index)
                        .map_err(|_| ConversionGeometryError::Overflow)?,
                    source_tile: source.tile,
                    source_storage: source.extents.clone(),
                    destination_shard: u32::try_from(destination_index)
                        .map_err(|_| ConversionGeometryError::Overflow)?,
                    destination_tile: destination.tile,
                    destination_storage,
                    source_region: region.clone(),
                    destination_region: region,
                    source_copies: Vec::new(),
                    copies: Vec::new(),
                    destination_copies: Vec::new(),
                },
            )?);
        }
    }
    if mappings.is_empty() {
        return Err(ConversionGeometryError::Unsupported);
    }
    Ok(mappings)
}

pub(crate) fn plan_view_conversion(
    source: &TensorType,
    destination: &TensorType,
    transform: DeferredTransform,
) -> Result<(ConversionStrategy, Vec<ConversionMapping>), ConversionGeometryError> {
    let strategy = layout_conversion_strategy(
        source.format.precision,
        &source.format.layout,
        &destination.format.layout,
    );
    let sources = source.format.layout.resolve(&source.shape)?.shard_extents();
    let destinations = destination
        .format
        .layout
        .resolve(&destination.shape)?
        .shard_extents();
    let mut mappings = Vec::new();
    for (destination_index, output) in destinations.into_iter().enumerate() {
        let destination_storage = if strategy == ConversionStrategy::StageLogicalThenTransform {
            output.extents.logical()
        } else {
            output.extents.clone()
        };
        for view in transform
            .map_slices(&source.shape, &destination.shape, &output.extents.logical())
            .ok_or(ConversionGeometryError::Unsupported)?
        {
            let mut intersections = BTreeMap::<TensorRegion, _>::new();
            for (source_index, input) in sources.iter().enumerate() {
                let Some(region) = input.extents.logical().intersection(&view.source) else {
                    continue;
                };
                let selected = intersections.entry(region).or_insert((source_index, input));
                if input.tile == output.tile {
                    *selected = (source_index, input);
                }
            }
            for (source_region, (source_index, input)) in intersections {
                let destination_region = view
                    .project_source(&source_region)
                    .ok_or(ConversionGeometryError::Unsupported)?;
                mappings.push(plan_mapping(
                    source.format.precision,
                    source.format.layout.order,
                    destination.format.layout.order,
                    strategy,
                    ConversionMapping {
                        source_shard: u32::try_from(source_index)
                            .map_err(|_| ConversionGeometryError::Overflow)?,
                        source_tile: input.tile,
                        source_storage: input.extents.clone(),
                        destination_shard: u32::try_from(destination_index)
                            .map_err(|_| ConversionGeometryError::Overflow)?,
                        destination_tile: output.tile,
                        destination_storage: destination_storage.clone(),
                        source_region,
                        destination_region,
                        source_copies: Vec::new(),
                        copies: Vec::new(),
                        destination_copies: Vec::new(),
                    },
                )?);
            }
        }
    }
    if mappings.is_empty() {
        return Err(ConversionGeometryError::Unsupported);
    }
    Ok((strategy, mappings))
}

pub(crate) fn finalize_conversion_plans(
    graph: &mut MidGraph,
) -> Result<(), ConversionGeometryError> {
    finalize_operations(&mut graph.operations, &graph.values)
}

fn finalize_operations(
    operations: &mut [MidOperation],
    values: &[MidValue],
) -> Result<(), ConversionGeometryError> {
    for operation in operations {
        match &mut operation.kind {
            MidOperationKind::View(transform, strategy, mappings) => {
                let source = &values
                    .get(operation.inputs[0].index() as usize)
                    .ok_or(ConversionGeometryError::Unsupported)?
                    .tensor_type;
                let destination = &values
                    .get(operation.results[0].index() as usize)
                    .ok_or(ConversionGeometryError::Unsupported)?
                    .tensor_type;
                let (resolved, planned) = plan_view_conversion(source, destination, *transform)?;
                if resolved != *strategy {
                    return Err(ConversionGeometryError::Unsupported);
                }
                *mappings = planned;
            }
            MidOperationKind::Rearrange(strategy, _, mappings) => {
                let source = &values
                    .get(operation.inputs[0].index() as usize)
                    .ok_or(ConversionGeometryError::Unsupported)?
                    .tensor_type;
                let destination = &values
                    .get(operation.results[0].index() as usize)
                    .ok_or(ConversionGeometryError::Unsupported)?
                    .tensor_type;
                *mappings = plan_conversion(
                    &destination.shape,
                    destination.format.precision,
                    &source.format.layout,
                    &destination.format.layout,
                    *strategy,
                )?;
            }
            MidOperationKind::Repeat(repeat) => {
                finalize_operations(&mut repeat.body.operations, values)?;
            }
            MidOperationKind::Operator(_) | MidOperationKind::CastPrecision => {}
        }
    }
    Ok(())
}

fn plan_mapping(
    precision: Precision,
    from: StorageOrder,
    to: StorageOrder,
    strategy: ConversionStrategy,
    mut mapping: ConversionMapping,
) -> Result<ConversionMapping, ConversionGeometryError> {
    let (source_order, destination_order, logical_order) = match strategy {
        ConversionStrategy::DirectRetile if from == to => (from, to, false),
        ConversionStrategy::DirectLogical => (from, to, true),
        ConversionStrategy::StageLogicalThenTransform => {
            (from, StorageOrder::Linear, from != StorageOrder::Linear)
        }
        ConversionStrategy::DirectRetile | ConversionStrategy::LocalKernel => {
            return Err(ConversionGeometryError::Unsupported);
        }
    };
    let direct = copy_geometries_between(
        precision,
        (
            source_order,
            &mapping.source_storage,
            &mapping.source_region,
        ),
        (
            destination_order,
            &mapping.destination_storage,
            &mapping.destination_region,
        ),
        mapping.source_tile == mapping.destination_tile,
        logical_order,
    );
    let (source_copies, copies, destination_copies) = match direct {
        Ok(copies) => (Vec::new(), copies, Vec::new()),
        Err(ConversionGeometryError::Unsupported) => {
            let logical_storage = mapping.destination_region.logical();
            let source_copies = copy_geometries_between(
                precision,
                (
                    source_order,
                    &mapping.source_storage,
                    &mapping.source_region,
                ),
                (
                    StorageOrder::Linear,
                    &logical_storage,
                    &mapping.destination_region,
                ),
                true,
                source_order != StorageOrder::Linear,
            )?;
            let destination_copies = copy_geometries(
                precision,
                StorageOrder::Linear,
                destination_order,
                &logical_storage,
                &mapping.destination_storage,
                &mapping.destination_region,
                true,
                destination_order != StorageOrder::Linear,
            )?;
            let bytes = mapping
                .source_region
                .logical_elements()
                .checked_mul(precision.bytes())
                .and_then(|bytes| u32::try_from(bytes).ok())
                .ok_or(ConversionGeometryError::Overflow)?;
            (
                source_copies,
                vec![CopyGeometry {
                    source_offset: 0,
                    destination_offset: 0,
                    contiguous_bytes: bytes.div_ceil(4) * 4,
                    dimensions: Vec::new(),
                }],
                destination_copies,
            )
        }
        Err(error) => return Err(error),
    };
    mapping.source_copies = source_copies;
    mapping.copies = copies;
    mapping.destination_copies = destination_copies;
    Ok(mapping)
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

fn copy_geometries_between(
    precision: Precision,
    source: (StorageOrder, &TensorRegion, &TensorRegion),
    destination: (StorageOrder, &TensorRegion, &TensorRegion),
    local: bool,
    logical_order: bool,
) -> Result<Vec<CopyGeometry>, ConversionGeometryError> {
    let (source_order, source_storage, source_region) = source;
    let (destination_order, destination_storage, destination_region) = destination;
    if source_storage.len() != destination_storage.len()
        || source_storage.len() != source_region.len()
        || source_region.len() != destination_region.len()
    {
        return Err(ConversionGeometryError::Unsupported);
    }
    let mut normalized_source = Vec::with_capacity(source_region.len());
    let mut normalized_destination = Vec::with_capacity(source_region.len());
    let mut normalized_region = Vec::with_capacity(source_region.len());
    for (axis, (((source_storage, destination_storage), source), destination)) in source_storage
        .iter()
        .zip(destination_storage.iter())
        .zip(source_region.iter())
        .zip(destination_region.iter())
        .enumerate()
    {
        let source_width = source.logical_end - source.start;
        if source_width != destination.logical_end - destination.start
            || source.start < source_storage.start
            || destination.start < destination_storage.start
        {
            return Err(ConversionGeometryError::Unsupported);
        }
        let source_delta = source.start - source_storage.start;
        let destination_delta = destination.start - destination_storage.start;
        let common_start = source_delta.max(destination_delta);
        let source_start = common_start - source_delta;
        let destination_start = common_start - destination_delta;
        let axis = u16::try_from(axis).map_err(|_| ConversionGeometryError::Overflow)?;
        let extent = |start: u32,
                      storage: &crate::ShardExtent|
         -> Result<crate::ShardExtent, ConversionGeometryError> {
            Ok(crate::ShardExtent {
                axis,
                start,
                logical_end: start
                    .checked_add(storage.logical_end - storage.start)
                    .ok_or(ConversionGeometryError::Overflow)?,
                physical_end: start
                    .checked_add(storage.physical_end - storage.start)
                    .ok_or(ConversionGeometryError::Overflow)?,
            })
        };
        normalized_source.push(extent(source_start, source_storage)?);
        normalized_destination.push(extent(destination_start, destination_storage)?);
        normalized_region.push(crate::ShardExtent {
            axis,
            start: common_start,
            logical_end: common_start
                .checked_add(source_width)
                .ok_or(ConversionGeometryError::Overflow)?,
            physical_end: common_start
                .checked_add(source_width)
                .ok_or(ConversionGeometryError::Overflow)?,
        });
    }
    copy_geometries(
        precision,
        source_order,
        destination_order,
        &normalized_source.into(),
        &normalized_destination.into(),
        &normalized_region.into(),
        local,
        logical_order,
    )
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
    use crate::layout::{
        AxisTiling, BlockedOrder, MemoryClass, Padding, ShardExtent, TensorAxis, TensorTiling,
    };
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
            let source_starts = shape
                .iter()
                .map(|_| 16 * random.u32(0..=1))
                .collect::<Vec<_>>();
            let destination_starts = shape
                .iter()
                .map(|_| 16 * random.u32(0..=1))
                .collect::<Vec<_>>();
            let mut storage = |starts: &[u32]| {
                shape
                    .iter()
                    .zip(starts)
                    .enumerate()
                    .map(|(axis, (&width, &start))| ShardExtent {
                        axis: axis as u16,
                        start: 0,
                        logical_end: start + width + 16 * random.u32(0..=1),
                        physical_end: start + width + 16,
                    })
                    .collect::<Vec<_>>()
                    .into()
            };
            let region = |starts: &[u32]| {
                shape
                    .iter()
                    .zip(starts)
                    .enumerate()
                    .map(|(axis, (&width, &start))| ShardExtent {
                        axis: axis as u16,
                        start,
                        logical_end: start + width,
                        physical_end: start + width,
                    })
                    .collect::<Vec<_>>()
                    .into()
            };
            let source_storage = storage(&source_starts);
            let destination_storage = storage(&destination_starts);
            let source_region = region(&source_starts);
            let destination_region = region(&destination_starts);
            let precision = if random.bool() {
                Precision::F16
            } else {
                Precision::F32
            };
            let source_order = orders[random.usize(..orders.len())];
            let destination_order = orders[random.usize(..orders.len())];
            let geometries = copy_geometries_between(
                precision,
                (source_order, &source_storage, &source_region),
                (destination_order, &destination_storage, &destination_region),
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
                let source_coordinates = coordinates
                    .iter()
                    .zip(&source_starts)
                    .map(|(coordinate, start)| coordinate + start)
                    .collect::<Vec<_>>();
                let destination_coordinates = coordinates
                    .iter()
                    .zip(&destination_starts)
                    .map(|(coordinate, start)| coordinate + start)
                    .collect::<Vec<_>>();
                let source = physical_byte_offset(
                    source_order,
                    precision,
                    &source_storage,
                    &source_coordinates,
                )
                .unwrap();
                let destination = physical_byte_offset(
                    destination_order,
                    precision,
                    &destination_storage,
                    &destination_coordinates,
                )
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

    #[test]
    fn randomized_native_results_can_be_linearized_directly() {
        let mut random = fastrand::Rng::with_seed(0x7374_6167_6564_6c69);
        for case in 0..64 {
            let row_partitions = random.u16(1..=8);
            let column_partitions = random.u16(1..=8);
            let rows = 16 * u32::from(row_partitions) * random.u32(1..=4) - random.u32(0..16);
            let columns = 2 * u32::from(column_partitions) * random.u32(1..=8) - random.u32(0..2);
            let tile_count = row_partitions * column_partitions;
            let source = Layout {
                order: StorageOrder::Native(NativeKernelOrder::TransposedLeft),
                tiling: TensorTiling {
                    tile_count,
                    replicas: 1,
                    axes: vec![
                        AxisTiling::new(TensorAxis::FromEnd(2), row_partitions, 16, Padding::Zero)
                            .with_tile_stride(1),
                        AxisTiling::new(
                            TensorAxis::FromEnd(1),
                            column_partitions,
                            2,
                            Padding::Zero,
                        )
                        .with_tile_stride(row_partitions),
                    ],
                },
                memory_class: MemoryClass::Interleaved,
            };
            let destination = Layout::logical_linear(tile_count, 4);
            let shape = if random.bool() {
                TensorShape::new([rows, columns])
            } else {
                TensorShape::new([1, rows, columns])
            };
            if source.resolve(&shape).is_err() || destination.resolve(&shape).is_err() {
                continue;
            }
            let strategy = layout_conversion_strategy(Precision::F16, &source, &destination);
            assert_eq!(
                strategy,
                ConversionStrategy::DirectLogical,
                "random case {case}"
            );
            plan_conversion(&shape, Precision::F16, &source, &destination, strategy)
                .unwrap_or_else(|error| panic!("random case {case}: {error}"));
        }
    }
}
