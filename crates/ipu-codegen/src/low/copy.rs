//! Relative copy operations and materialization policy, before placement.

use crate::storage::{ByteSpan, StorageError, StorageResult};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CopyOrder {
    /// Preserve tensor coordinates, converting between physical layouts.
    #[default]
    Semantic,
    /// Preserve allocation order, treating both views as packed byte spans.
    Physical,
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

impl<Buffer: Clone> CopyOperation<Buffer> {
    pub(crate) fn from_spans(
        source: Buffer,
        destination: Buffer,
        source_spans: &[ByteSpan],
        destination_spans: &[ByteSpan],
    ) -> StorageResult<Vec<Self>> {
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
        Ok(coalesce_copies(copies))
    }
}

/// Zip two span streams by byte position, retaining each stream's boundaries.
/// This is shared by local copy generation and direct-exchange costing.
pub(crate) fn for_each_copy_span(
    source: &[ByteSpan],
    destination: &[ByteSpan],
    mut visit: impl FnMut(u32, u32, u32) -> StorageResult<()>,
) -> StorageResult<()> {
    let mut sources = source.iter().copied().filter(|span| span.bytes != 0);
    let mut destinations = destination.iter().copied().filter(|span| span.bytes != 0);
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
        if first.bytes != second.bytes || first.bytes == 0 || !first.bytes.is_multiple_of(8) {
            coalesced.push(first.clone());
            index += 1;
            continue;
        }
        let source_stride = second.source_offset.saturating_sub(first.source_offset);
        let destination_stride = second
            .destination_offset
            .saturating_sub(first.destination_offset);
        if source_stride == 0 || destination_stride == 0 {
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
        // Larger strided regions are deliberately left as contiguous rows:
        // spreading them over workers loses more to bank contention than it
        // saves in call overhead on IPU21.
        if first.bytes.saturating_mul(rows) > PARALLEL_STRIDED_COPY_MAX_BYTES {
            coalesced.extend(copies[index..end].iter().cloned());
            index = end;
            continue;
        }
        if source_stride == first.bytes && destination_stride == first.bytes {
            let mut copy = first.clone();
            copy.bytes = copy.bytes.saturating_mul(rows);
            coalesced.push(copy);
        } else {
            let mut copy = first.clone();
            copy.bytes = copy.bytes.saturating_mul(rows);
            copy.pattern = CopyPattern::Strided {
                rows,
                row_bytes: first.bytes,
                source_stride,
                destination_stride,
            };
            coalesced.push(copy);
        }
        index = end;
    }
    coalesced
}

use crate::storage::{TensorStorage, logical_byte_spans, physical_byte_spans, storage_bytes};
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
        if order == CopyOrder::Physical {
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
            let source = logical_byte_spans(mapping.source, mapping.source_extents)?;
            let target = logical_byte_spans(storage, mapping.destination_extents)?;
            let aligned =
                |span: &ByteSpan| span.offset.is_multiple_of(4) && span.bytes.is_multiple_of(4);
            destination_unaligned |= !target.iter().all(aligned);
            word_aligned &= source.iter().all(aligned) && !destination_unaligned;
            if word_aligned {
                word_aligned = for_each_copy_span(&source, &target, |_, _, bytes| {
                    fragments = fragments.saturating_add(
                        u64::from(bytes).div_ceil(u64::from(ipu_exchange::MAX_TRANSFER_WORDS) * 4),
                    );
                    Ok(())
                })
                .is_ok();
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
    let mut covered = Vec::new();
    for mapping in mappings {
        covered.extend(match order {
            CopyOrder::Physical => physical_byte_spans(storage, mapping.destination_extents)?,
            CopyOrder::Semantic => {
                let mut logical = mapping.destination_extents.to_vec();
                for extent in &mut logical {
                    extent.physical_end = extent.logical_end;
                }
                logical_byte_spans(storage, &logical)?
            }
        });
    }
    covered.sort_unstable_by_key(|span| span.offset);
    let mut cursor = 0u32;
    let mut holes: Vec<ByteSpan> = Vec::new();
    for span in covered.into_iter().chain(std::iter::once(ByteSpan {
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
            let expected = flatten(&source)
                .into_iter()
                .zip(flatten(&destination))
                .collect::<Vec<_>>();
            let copies = CopyOperation::from_spans(0, 1, &source, &destination).unwrap();
            let actual = copies
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
            assert_eq!(actual, expected);
        }
        assert!(
            CopyOperation::from_spans(
                0,
                1,
                &[ByteSpan {
                    offset: 0,
                    bytes: 4
                }],
                &[]
            )
            .is_err()
        );
    }

    #[test]
    fn copy_runs_respect_worker_striding_limit() {
        for rows in [2, 64, 65] {
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
            let copies = CopyOperation::from_spans(0, 1, &source, &destination).unwrap();
            if rows <= 64 {
                assert_eq!(copies.len(), 1);
                assert_eq!(
                    copies[0].pattern,
                    CopyPattern::Strided {
                        rows,
                        row_bytes: 8,
                        source_stride: 16,
                        destination_stride: 24
                    }
                );
            } else {
                assert_eq!(copies.len(), rows as usize);
                assert!(
                    copies
                        .iter()
                        .all(|copy| copy.pattern == CopyPattern::Contiguous)
                );
            }
        }
    }
}
