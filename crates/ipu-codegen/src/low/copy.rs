//! Relative copy operations and materialization policy, before placement.

use crate::storage::{ByteSpan, StorageError, StorageResult};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum CopyOrder {
    /// Preserve tensor coordinates, converting between physical layouts.
    #[default]
    Semantic,
    /// Preserve allocation order, treating both views as packed byte spans.
    Physical,
    /// Row-major grid of 16-by-16 panels, physical order within each panel.
    Panels,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyOperation<Buffer> {
    pub source: Buffer,
    pub source_offset: u32,
    pub destination: Buffer,
    pub destination_offset: u32,
    pub bytes: u32,
    pub pattern: CopyPattern,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CopyPattern {
    #[default]
    Contiguous,
    Strided {
        rows: u32,
        row_bytes: u32,
        source_stride: u32,
        destination_stride: u32,
    },
}

impl<Buffer: Clone + PartialEq> CopyOperation<Buffer> {
    pub(crate) fn from_traversals(
        source: Buffer,
        destination: Buffer,
        source_traversal: &ByteTraversal,
        destination_traversal: &ByteTraversal,
    ) -> StorageResult<Vec<Self>> {
        if source_traversal.byte_len() != destination_traversal.byte_len() {
            return Err(StorageError::InvalidView);
        }
        if source_traversal.byte_len() == 0 {
            return Ok(Vec::new());
        }
        if let (Some(mut left), Some(mut right)) = (
            source_traversal.regular_span(),
            destination_traversal.regular_span(),
        ) {
            // Split a contiguous endpoint symbolically to match the other side.
            if left.rows == 1 && right.rows > 1 {
                left.bytes = right.bytes;
                left.stride = right.bytes;
                left.rows = right.rows;
            } else if right.rows == 1 && left.rows > 1 {
                right.bytes = left.bytes;
                right.stride = left.bytes;
                right.rows = left.rows;
            }
            if left.bytes == right.bytes
                && left.rows == right.rows
                && let Some(pattern) = row_copy_pattern(
                    left.rows,
                    left.bytes,
                    left.offset,
                    right.offset,
                    left.stride,
                    right.stride,
                )
            {
                return Ok(vec![Self {
                    source,
                    destination,
                    source_offset: left.offset,
                    destination_offset: right.offset,
                    bytes: left.bytes * left.rows,
                    pattern,
                }]);
            }
        }
        Self::from_spans(
            source,
            destination,
            source_traversal.spans(),
            destination_traversal.spans(),
        )
    }

    pub(crate) fn from_spans(
        source: Buffer,
        destination: Buffer,
        source_spans: impl IntoIterator<Item = ByteSpan>,
        destination_spans: impl IntoIterator<Item = ByteSpan>,
    ) -> StorageResult<Vec<Self>> {
        let can_reorder = source != destination;
        let mut copies = Vec::new();
        for_each_copy_span(
            source_spans,
            destination_spans,
            |source_offset, destination_offset, bytes| {
                copies.push(Self {
                    source: source.clone(),
                    destination: destination.clone(),
                    source_offset,
                    destination_offset,
                    bytes,
                    pattern: CopyPattern::Contiguous,
                });
                Ok(())
            },
        )?;
        let original = coalesce_copies(copies.clone());
        if !can_reorder || original.len() < 2 {
            return Ok(original);
        }
        // Local copies read one live allocation and write another. Traversing
        // disjoint destination spans in physical order can turn a blocked
        // transpose's many short launches into a few long strided copies.
        copies.sort_by_key(|copy| (copy.destination_offset, copy.source_offset));
        if copies.windows(2).any(|pair| {
            pair[0]
                .destination_offset
                .checked_add(pair[0].bytes)
                .is_none_or(|end| end > pair[1].destination_offset)
        }) {
            return Ok(original);
        }
        let reordered = coalesce_copies(copies);
        Ok(if reordered.len() < original.len() {
            reordered
        } else {
            original
        })
    }
}

/// Zip two span streams by byte position, retaining each stream's boundaries.
/// This is shared by local copy generation and direct-exchange costing.
pub(crate) fn for_each_copy_span(
    source: impl IntoIterator<Item = ByteSpan>,
    destination: impl IntoIterator<Item = ByteSpan>,
    mut visit: impl FnMut(u32, u32, u32) -> StorageResult<()>,
) -> StorageResult<()> {
    let mut sources = source.into_iter().filter(|span| span.bytes != 0);
    let mut destinations = destination.into_iter().filter(|span| span.bytes != 0);
    let mut source = sources.next();
    let mut destination = destinations.next();
    while let (Some(left), Some(right)) = (&mut source, &mut destination) {
        let bytes = left.bytes.min(right.bytes);
        visit(left.offset, right.offset, bytes)?;
        left.offset = left
            .offset
            .checked_add(bytes)
            .ok_or(StorageError::Overflow)?;
        right.offset = right
            .offset
            .checked_add(bytes)
            .ok_or(StorageError::Overflow)?;
        left.bytes -= bytes;
        right.bytes -= bytes;
        if left.bytes == 0 {
            source = sources.next();
        }
        if right.bytes == 0 {
            destination = destinations.next();
        }
    }
    if source.is_some() || destination.is_some() {
        return Err(StorageError::InvalidView);
    }
    Ok(())
}

const PARALLEL_STRIDED_COPY_MAX_BYTES: u32 = 512;

/// Shared launch policy for symbolic rows and the irregular-span fallback.
fn row_copy_pattern(
    rows: u32,
    row_bytes: u32,
    source: u32,
    destination: u32,
    source_stride: u32,
    destination_stride: u32,
) -> Option<CopyPattern> {
    if rows == 1 {
        return Some(CopyPattern::Contiguous);
    }
    let geometry = [
        row_bytes,
        source,
        destination,
        source_stride,
        destination_stride,
    ];
    if source_stride == 0
        || destination_stride == 0
        || !geometry.iter().all(|n| n.is_multiple_of(4))
    {
        return None;
    }
    // Wide copies with too few rows underuse the workers in the strided kernel.
    if rows < 6
        && row_bytes.saturating_mul(rows) > PARALLEL_STRIDED_COPY_MAX_BYTES
        && geometry.iter().all(|n| n.is_multiple_of(8))
    {
        return None;
    }
    Some(
        if source_stride == row_bytes && destination_stride == row_bytes {
            CopyPattern::Contiguous
        } else {
            CopyPattern::Strided {
                rows,
                row_bytes,
                source_stride,
                destination_stride,
            }
        },
    )
}

fn coalesce_copies<Buffer: Clone>(
    copies: Vec<CopyOperation<Buffer>>,
) -> Vec<CopyOperation<Buffer>> {
    let mut coalesced = Vec::new();
    let mut index = 0;
    while index < copies.len() {
        let first = &copies[index];
        let Some(second) = copies.get(index + 1) else {
            coalesced.push(first.clone());
            break;
        };
        if first.bytes != second.bytes || first.bytes == 0 || !first.bytes.is_multiple_of(4) {
            coalesced.push(first.clone());
            index += 1;
            continue;
        }
        let source_stride = second.source_offset.saturating_sub(first.source_offset);
        let destination_stride = second
            .destination_offset
            .saturating_sub(first.destination_offset);
        if source_stride == 0
            || destination_stride == 0
            || !first.source_offset.is_multiple_of(4)
            || !first.destination_offset.is_multiple_of(4)
            || !source_stride.is_multiple_of(4)
            || !destination_stride.is_multiple_of(4)
        {
            coalesced.push(first.clone());
            index += 1;
            continue;
        }
        let mut end = index + 2;
        while let Some(copy) = copies.get(end) {
            let previous = &copies[end - 1];
            if copy.bytes != first.bytes
                || copy.source_offset.checked_sub(previous.source_offset) != Some(source_stride)
                || copy
                    .destination_offset
                    .checked_sub(previous.destination_offset)
                    != Some(destination_stride)
            {
                break;
            }
            end += 1;
        }
        let rows = u32::try_from(end - index).unwrap_or(u32::MAX);
        if let Some(pattern) = row_copy_pattern(
            rows,
            first.bytes,
            first.source_offset,
            first.destination_offset,
            source_stride,
            destination_stride,
        ) {
            let mut copy = first.clone();
            copy.bytes = copy.bytes.saturating_mul(rows);
            copy.pattern = pattern;
            coalesced.push(copy);
        } else {
            coalesced.extend(copies[index..end].iter().cloned());
        }
        index = end;
    }
    coalesced
}

use crate::storage::{ByteTraversal, TensorStorage, byte_traversal, storage_bytes};
use crate::{
    AmpOrder, BlockMajorOrder, ElementOrder, Layout, MemoryClass, ShardExtent, TensorFormat,
    TensorTiling, TensorType, TileKernelSpec,
};

pub(crate) struct CopyMapping<'a> {
    pub source: TensorStorage<'a>,
    pub source_extents: &'a [ShardExtent],
    pub destination_extents: &'a [ShardExtent],
}

pub(crate) struct CopyStaging {
    pub tensor_type: TensorType,
    pub extents: Vec<ShardExtent>,
    pub kernel: Option<TileKernelSpec>,
}

/// Physical realization of a selected mid copy: direct movement or destination
/// packing, plus initialization of storage the source does not populate.
pub(crate) struct CopyPlan {
    pub clear_ranges: Vec<ByteSpan>,
    pub staging: Option<CopyStaging>,
}

impl CopyPlan {
    pub(crate) fn for_destination(
        destination: &TensorType,
        extents: &[ShardExtent],
        mappings: &[CopyMapping<'_>],
        order: CopyOrder,
    ) -> StorageResult<Self> {
        if order != CopyOrder::Semantic {
            return Ok(Self {
                clear_ranges: uncovered_copy_bytes(
                    TensorStorage {
                        format: &destination.format,
                        extents,
                    },
                    mappings,
                    order,
                )?,
                staging: None,
            });
        }
        let storage = TensorStorage {
            format: &destination.format,
            extents,
        };
        let mut fragments = 0u64;
        let mut word_aligned = true;
        let mut destination_unaligned = false;
        for mapping in mappings {
            let source = byte_traversal(mapping.source, mapping.source_extents, false)?;
            let target = byte_traversal(storage, mapping.destination_extents, false)?;
            destination_unaligned |= !target.word_aligned();
            word_aligned &= source.word_aligned() && !destination_unaligned;
            if word_aligned {
                match source.copy_fragments(&target, ipu_exchange::MAX_TRANSFER_WORDS * 4) {
                    Ok(count) => fragments = fragments.saturating_add(count),
                    Err(_) => word_aligned = false,
                }
            }
        }

        let bytes = u64::from(storage_bytes(storage)?);
        let padding = extents
            .iter()
            .any(|extent| extent.physical_end > extent.logical_end);
        let clear_cycles = if padding {
            crate::estimate::IPU21_TARGET_COSTS
                .kernel_launch_cycles
                .saturating_add(bytes.div_ceil(8 * 6))
        } else {
            0
        };
        let fragment_cycles = fragments
            .saturating_mul(crate::estimate::IPU21_LOGICAL_FRAGMENT_CYCLES)
            .saturating_add(clear_cycles);
        let pack_cycles = crate::estimate::row_major_pack_cycles(
            destination,
            bytes.div_ceil(destination.format.precision.bytes().max(1)),
        );
        let direct_word_exchange = word_aligned && fragment_cycles < pack_cycles;
        let transform = destination_unaligned
            || mappings.iter().any(|mapping| {
                mapping.source.format.layout.order != destination.format.layout.order
            });
        let staging = (transform && !direct_word_exchange).then(|| {
            let mut extents = extents.to_vec();
            for extent in &mut extents {
                extent.physical_end = extent.logical_end;
            }
            let tensor_type = TensorType {
                shape: destination.shape.clone(),
                format: TensorFormat {
                    precision: destination.format.precision,
                    layout: Layout {
                        order: ElementOrder::RowMajor,
                        tiling: TensorTiling::replicated(1),
                        memory_class: MemoryClass::Ipu21Standard,
                    },
                },
            };
            let kernel = (destination.format.precision == super::Precision::F16
                && matches!(
                    destination.format.layout.order,
                    ElementOrder::Amp(AmpOrder::Left | AmpOrder::TransposedRight)
                        | ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. })
                ))
            .then(|| TileKernelSpec::Rearrange {
                from: tensor_type.format.layout.clone(),
                to: destination.format.layout.clone(),
            });
            CopyStaging {
                tensor_type,
                extents,
                kernel,
            }
        });
        Ok(Self {
            // Packing writes every physical output element, including zero
            // padding. A semantic copy without a packer writes logical bytes.
            clear_ranges: if staging
                .as_ref()
                .is_some_and(|staging| staging.kernel.is_some())
            {
                Vec::new()
            } else {
                uncovered_copy_bytes(storage, mappings, CopyOrder::Semantic)?
            },
            staging,
        })
    }
}

/// Byte coverage, not summed volume: overlapping mappings cannot hide holes.
/// Clears run before copies, so round holes outward to the fill kernel's eight
/// byte granularity; neighboring logical bytes are subsequently overwritten.
pub(crate) fn uncovered_copy_bytes(
    storage: TensorStorage<'_>,
    mappings: &[CopyMapping<'_>],
    order: CopyOrder,
) -> StorageResult<Vec<ByteSpan>> {
    let bytes = storage_bytes(storage)?;
    let covered = mappings
        .iter()
        .map(|mapping| {
            let mut extents = mapping.destination_extents.to_vec();
            if order == CopyOrder::Semantic {
                for extent in &mut extents {
                    extent.physical_end = extent.logical_end;
                }
            }
            byte_traversal(storage, &extents, true)
        })
        .collect::<StorageResult<Vec<_>>>()?;
    let covered = ByteTraversal::physical_union(covered);
    let mut cursor = 0u32;
    let mut holes: Vec<ByteSpan> = Vec::new();
    for span in covered.spans().chain(std::iter::once(ByteSpan {
        offset: bytes,
        bytes: 0,
    })) {
        let end = span
            .offset
            .checked_add(span.bytes)
            .ok_or(StorageError::Overflow)?;
        if end > bytes {
            return Err(StorageError::InvalidView);
        }
        if span.offset > cursor {
            let start = cursor / 8 * 8;
            let end = span
                .offset
                .div_ceil(8)
                .checked_mul(8)
                .ok_or(StorageError::Overflow)?
                .min(bytes);
            if let Some(previous) = holes.last_mut()
                && previous.offset + previous.bytes >= start
            {
                previous.bytes = end - previous.offset;
            } else {
                holes.push(ByteSpan {
                    offset: start,
                    bytes: end - start,
                });
            }
        }
        cursor = cursor.max(end);
    }
    Ok(holes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_initialize_uncovered_padding_and_view_tails() {
        let source = TensorType::new(
            [2, 50],
            super::super::Precision::F16,
            Layout::row_major(TensorTiling::replicated(1)),
        );
        let copied = [
            ShardExtent {
                axis: 0,
                start: 0,
                logical_end: 2,
                physical_end: 2,
            },
            ShardExtent {
                axis: 1,
                start: 0,
                logical_end: 50,
                physical_end: 50,
            },
        ];
        for logical_end in [50, 64] {
            let mut extents = copied;
            extents[1].logical_end = logical_end;
            extents[1].physical_end = 64;
            let mapping = CopyMapping {
                source: TensorStorage {
                    format: &source.format,
                    extents: &copied,
                },
                source_extents: &copied,
                destination_extents: &copied,
            };
            for order in [CopyOrder::Physical, CopyOrder::Semantic] {
                let plan = CopyPlan::for_destination(
                    &source,
                    &extents,
                    std::slice::from_ref(&mapping),
                    order,
                )
                .unwrap();
                let expected = vec![
                    ByteSpan {
                        offset: 96,
                        bytes: 32,
                    },
                    ByteSpan {
                        offset: 224,
                        bytes: 32,
                    },
                ];
                assert_eq!(plan.clear_ranges, expected);
                // Counting copied elements would incorrectly classify these
                // overlapping writes as covering the whole allocation.
                assert_eq!(
                    uncovered_copy_bytes(
                        TensorStorage {
                            format: &source.format,
                            extents: &extents
                        },
                        &[mapping_ref(&mapping), mapping_ref(&mapping)],
                        order,
                    )
                    .unwrap(),
                    expected,
                );
            }
        }
    }

    fn mapping_ref<'a>(mapping: &CopyMapping<'a>) -> CopyMapping<'a> {
        CopyMapping {
            source: mapping.source,
            source_extents: mapping.source_extents,
            destination_extents: mapping.destination_extents,
        }
    }

    #[test]
    fn semantic_staging_coverage_ignores_original_physical_padding() {
        let tensor = TensorType::new(
            [2, 50],
            super::super::Precision::F16,
            Layout::row_major(TensorTiling::replicated(1)),
        );
        let extents = [
            ShardExtent {
                axis: 0,
                start: 0,
                logical_end: 2,
                physical_end: 2,
            },
            ShardExtent {
                axis: 1,
                start: 0,
                logical_end: 50,
                physical_end: 50,
            },
        ];
        let mut padded = extents;
        padded[1].physical_end = 64;
        let storage = TensorStorage {
            format: &tensor.format,
            extents: &extents,
        };
        let mapping = CopyMapping {
            source: storage,
            source_extents: &extents,
            destination_extents: &padded,
        };
        assert!(
            uncovered_copy_bytes(storage, &[mapping], CopyOrder::Semantic)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn randomized_copy_holes_match_byte_coverage() {
        let mut random = fastrand::Rng::with_seed(0x686f_6c65);
        for _ in 0..256 {
            let rows = random.u32(1..=8);
            let columns = random.u32(1..=8) * 4;
            let tensor = TensorType::new(
                [rows, columns],
                super::super::Precision::F16,
                Layout::row_major(TensorTiling::replicated(1)),
            );
            let extents = [
                ShardExtent {
                    axis: 0,
                    start: 0,
                    logical_end: rows,
                    physical_end: rows,
                },
                ShardExtent {
                    axis: 1,
                    start: 0,
                    logical_end: columns,
                    physical_end: columns,
                },
            ];
            let storage = TensorStorage {
                format: &tensor.format,
                extents: &extents,
            };
            let mut covered = vec![false; (rows * columns * 2) as usize];
            let rectangles = (0..random.usize(0..=12))
                .map(|_| {
                    let row = random.u32(0..rows);
                    let column = random.u32(0..columns);
                    let end_row = random.u32(row + 1..=rows);
                    let end_column = random.u32(column + 1..=columns);
                    for r in row..end_row {
                        for c in column..end_column {
                            let byte = ((r * columns + c) * 2) as usize;
                            covered[byte..byte + 2].fill(true);
                        }
                    }
                    [
                        ShardExtent {
                            axis: 0,
                            start: row,
                            logical_end: end_row,
                            physical_end: end_row,
                        },
                        ShardExtent {
                            axis: 1,
                            start: column,
                            logical_end: end_column,
                            physical_end: end_column,
                        },
                    ]
                })
                .collect::<Vec<_>>();
            let mappings = rectangles
                .iter()
                .map(|rectangle| CopyMapping {
                    source: storage,
                    source_extents: rectangle,
                    destination_extents: rectangle,
                })
                .collect::<Vec<_>>();
            for order in [CopyOrder::Physical, CopyOrder::Semantic] {
                let mut cleared = vec![false; covered.len()];
                for range in uncovered_copy_bytes(storage, &mappings, order).unwrap() {
                    assert_eq!(range.offset % 8, 0);
                    assert_eq!(range.bytes % 8, 0);
                    cleared[range.offset as usize..(range.offset + range.bytes) as usize]
                        .fill(true);
                }
                for (written, initialized) in covered.chunks_exact(8).zip(cleared.chunks_exact(8)) {
                    assert!(
                        initialized
                            .iter()
                            .all(|&clear| clear == written.contains(&false))
                    );
                }
            }
        }
    }

    #[test]
    fn packed_block_transpose_uses_long_rows_instead_of_tiny_launches() {
        let source = [ByteSpan {
            offset: 0,
            bytes: 164 * 6 * 32,
        }];
        let destination = (0..164)
            .flat_map(|column| {
                (0..6).map(move |row| ByteSpan {
                    offset: (row * 164 + column) * 32,
                    bytes: 32,
                })
            })
            .collect::<Vec<_>>();
        let copies =
            CopyOperation::from_spans(0, 1, source.iter().copied(), destination.iter().copied())
                .unwrap();
        assert_eq!(copies.len(), 6);
        for (row, copy) in copies.iter().enumerate() {
            assert_eq!(copy.source_offset, row as u32 * 32);
            assert_eq!(copy.destination_offset, row as u32 * 164 * 32);
            assert_eq!(
                copy.pattern,
                CopyPattern::Strided {
                    rows: 164,
                    row_bytes: 32,
                    source_stride: 192,
                    destination_stride: 32,
                }
            );
        }
        // Do not reorder a request that explicitly aliases its source.
        assert_eq!(
            CopyOperation::from_spans(0, 0, source.iter().copied(), destination.iter().copied())
                .unwrap()
                .len(),
            164
        );
    }

    #[test]
    fn copy_runs_preserve_byte_mapping_across_span_boundaries() {
        let mut random = fastrand::Rng::with_seed(0x636f_7079);
        for _ in 0..1000 {
            let bytes = random.u32(1..=1024);
            let mut spans = || {
                let mut remaining = bytes;
                let mut offset = random.u32(0..32);
                let mut result = Vec::new();
                while remaining > 0 {
                    let count = random.u32(1..=remaining.min(64));
                    result.push(ByteSpan {
                        offset,
                        bytes: count,
                    });
                    offset += count + random.u32(0..16);
                    remaining -= count;
                }
                result
            };
            let source = spans();
            let destination = spans();
            let flatten = |spans: &[ByteSpan]| {
                spans
                    .iter()
                    .flat_map(|span| span.offset..span.offset + span.bytes)
                    .collect::<Vec<_>>()
            };
            let mut expected = flatten(&source)
                .into_iter()
                .zip(flatten(&destination))
                .collect::<Vec<_>>();
            let copies = CopyOperation::from_spans(
                0,
                1,
                source.iter().copied(),
                destination.iter().copied(),
            )
            .unwrap();
            let mut actual = copies
                .into_iter()
                .flat_map(|copy| {
                    let (rows, width, source_stride, destination_stride) = match copy.pattern {
                        CopyPattern::Contiguous => (1, copy.bytes, 0, 0),
                        CopyPattern::Strided {
                            rows,
                            row_bytes,
                            source_stride,
                            destination_stride,
                        } => (rows, row_bytes, source_stride, destination_stride),
                    };
                    (0..rows).flat_map(move |row| {
                        (0..width).map(move |byte| {
                            (
                                copy.source_offset + row * source_stride + byte,
                                copy.destination_offset + row * destination_stride + byte,
                            )
                        })
                    })
                })
                .collect::<Vec<_>>();
            actual.sort_unstable();
            expected.sort_unstable();
            assert_eq!(actual, expected);
        }
        assert!(
            CopyOperation::from_spans(
                0,
                1,
                [ByteSpan {
                    offset: 0,
                    bytes: 4
                }],
                []
            )
            .is_err()
        );
    }

    #[test]
    fn four_byte_strided_copies_batch_rows_and_preserve_gaps() {
        // Include the attention query's 960-byte local intersection, fewer
        // rows than workers, and several rounds of the six-worker row loop.
        for rows in [2, 5, 6, 8, 19] {
            for (offset, width, source_stride, destination_stride) in
                [(108, 120, 228, 160), (4, 12, 20, 28), (0, 4, 12, 8)]
            {
                let spans = |start, stride| {
                    (0..rows)
                        .map(|row| ByteSpan {
                            offset: start + row * stride,
                            bytes: width,
                        })
                        .collect::<Vec<_>>()
                };
                let copies = CopyOperation::from_spans(
                    crate::BlockValueId::from_index(0),
                    crate::BlockValueId::from_index(1),
                    spans(offset, source_stride),
                    spans(0, destination_stride),
                )
                .unwrap();
                assert_eq!(copies.len(), 1);
                let (symbol, args) = crate::tile::local_copy_call(&copies[0]).unwrap();
                assert_eq!(symbol, crate::COPY_STRIDED_U32_SYMBOL);
                assert_eq!(args, [width / 4, rows, source_stride, destination_stride]);
                let source = (0..offset + rows * source_stride)
                    .map(|i| (i % 251) as u8)
                    .collect::<Vec<_>>();
                let mut actual = vec![255; (rows * destination_stride) as usize];
                let mut expected = actual.clone();
                for row in 0..rows {
                    let src = (offset + row * source_stride) as usize;
                    let dst = (row * destination_stride) as usize;
                    expected[dst..dst + width as usize]
                        .copy_from_slice(&source[src..src + width as usize]);
                }
                for worker in 0..6 {
                    for row in (worker..args[1]).step_by(6) {
                        for word in 0..args[0] {
                            let src = (copies[0].source_offset + row * args[2] + word * 4) as usize;
                            let dst =
                                (copies[0].destination_offset + row * args[3] + word * 4) as usize;
                            actual[dst..dst + 4].copy_from_slice(&source[src..src + 4]);
                        }
                    }
                }
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn long_strided_copies_keep_all_rows_in_one_launch() {
        for rows in [2, 64, 65, 164] {
            let source = (0..rows)
                .map(|row| ByteSpan {
                    offset: row * 16,
                    bytes: 8,
                })
                .collect::<Vec<_>>();
            let destination = (0..rows)
                .map(|row| ByteSpan {
                    offset: row * 24,
                    bytes: 8,
                })
                .collect::<Vec<_>>();
            let copies = CopyOperation::from_spans(
                0,
                1,
                source.iter().copied(),
                destination.iter().copied(),
            )
            .unwrap();
            assert_eq!(copies.len(), 1);
            assert_eq!(
                copies[0].pattern,
                CopyPattern::Strided {
                    rows,
                    row_bytes: 8,
                    source_stride: 16,
                    destination_stride: 24,
                }
            );
        }
        let source = [
            ByteSpan {
                offset: 0,
                bytes: 1024,
            },
            ByteSpan {
                offset: 2048,
                bytes: 1024,
            },
        ];
        let destination = [
            ByteSpan {
                offset: 0,
                bytes: 1024,
            },
            ByteSpan {
                offset: 4096,
                bytes: 1024,
            },
        ];
        let copies =
            CopyOperation::from_spans(0, 1, source.iter().copied(), destination.iter().copied())
                .unwrap();
        assert_eq!(copies.len(), 2);
        assert!(
            copies
                .iter()
                .all(|copy| copy.pattern == CopyPattern::Contiguous)
        );
    }
}
