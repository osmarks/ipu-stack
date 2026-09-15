//! Allocation-relative movement geometry. This module does not select kernels,
//! price transfers, allocate scratch, or round coverage to an implementation's stores.

use super::{
    ByteSpan, ByteTraversal, StorageError, StorageResult, TensorStorage, byte_traversal,
    storage_bytes,
};
use crate::tensor::ShardExtent;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CopyOrder {
    /// Preserve tensor coordinates, converting between physical layouts.
    #[default]
    Semantic,
    /// Preserve allocation order, treating both views as packed byte spans.
    Physical,
    /// Row-major grid of 16-by-16 panels, physical order within each panel.
    Panels,
}

/// Exact byte coverage, not summed volume: overlapping mappings cannot hide holes.
/// Any alignment required by a clearing implementation belongs to its caller.
pub(crate) fn uncovered_bytes<'a>(
    storage: TensorStorage<'_>,
    views: impl IntoIterator<Item = &'a [ShardExtent]>,
    order: CopyOrder,
) -> StorageResult<Vec<ByteSpan>> {
    uncovered_ranges(
        storage_bytes(storage)?,
        &copy_coverage(storage, views, order)?,
    )
}

fn copy_coverage<'a>(
    storage: TensorStorage<'_>,
    views: impl IntoIterator<Item = &'a [ShardExtent]>,
    order: CopyOrder,
) -> StorageResult<ByteTraversal> {
    let covered = views
        .into_iter()
        .map(|view| {
            let mut extents = view.to_vec();
            if order == CopyOrder::Semantic {
                for extent in &mut extents {
                    extent.physical_end = extent.logical_end;
                }
            }
            byte_traversal(storage, &extents, true)
        })
        .collect::<StorageResult<Vec<_>>>()?;
    Ok(ByteTraversal::physical_union(covered))
}

pub(super) fn uncovered_ranges(
    bytes: u32,
    covered: &ByteTraversal,
) -> StorageResult<Vec<ByteSpan>> {
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
            holes.push(ByteSpan {
                offset: cursor,
                bytes: span.offset - cursor,
            });
        }
        cursor = cursor.max(end);
    }
    Ok(holes)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{CopyMapping, GeometryCache};
    use crate::tensor::{Layout, Precision, TensorTiling, TensorType};

    #[test]
    fn copies_initialize_uncovered_padding_and_view_tails() {
        let source = TensorType::new(
            [2, 50],
            Precision::F16,
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
                let geometry = GeometryCache::default()
                    .destination(
                        TensorStorage {
                            format: &source.format,
                            extents: &extents,
                        },
                        std::slice::from_ref(&mapping),
                        order,
                        256,
                    )
                    .unwrap();
                let expected = vec![
                    ByteSpan {
                        offset: 100,
                        bytes: 28,
                    },
                    ByteSpan {
                        offset: 228,
                        bytes: 28,
                    },
                ];
                assert_eq!(geometry.uncovered().unwrap(), expected);
                // Counting copied elements would incorrectly classify these
                // overlapping writes as covering the whole allocation.
                assert_eq!(
                    uncovered_bytes(
                        TensorStorage {
                            format: &source.format,
                            extents: &extents
                        },
                        [mapping.destination_extents, mapping.destination_extents],
                        order,
                    )
                    .unwrap(),
                    expected,
                );
            }
        }
    }

    #[test]
    fn semantic_staging_coverage_ignores_original_physical_padding() {
        let tensor = TensorType::new(
            [2, 50],
            Precision::F16,
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
            uncovered_bytes(storage, [mapping.destination_extents], CopyOrder::Semantic)
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
                Precision::F16,
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
                for range in uncovered_bytes(
                    storage,
                    mappings.iter().map(|m| m.destination_extents),
                    order,
                )
                .unwrap()
                {
                    cleared[range.offset as usize..(range.offset + range.bytes) as usize]
                        .fill(true);
                }
                assert!(
                    covered
                        .iter()
                        .zip(cleared)
                        .all(|(&written, clear)| written != clear)
                );
            }
        }
    }
}
