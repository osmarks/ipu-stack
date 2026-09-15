//! Form contiguous or strided launches from matched byte geometry.
//! Reorder only independent buffers with disjoint destination spans.
//! Worker utilization and instruction alignment determine useful row grouping.

use super::*;
#[cfg(test)]
use crate::storage::ByteSpan;
use crate::storage::{CopyPair, StorageResult};

impl<Buffer: Clone + PartialEq> CopyOperation<Buffer> {
    pub(crate) fn from_pair(
        source: Buffer,
        destination: Buffer,
        same_buffer: bool,
        pair: &CopyPair,
    ) -> StorageResult<Vec<Self>> {
        if pair.bytes == 0 {
            return Ok(Vec::new());
        }
        if let [[left, right]] = pair.rows.as_slice()
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
                bytes: left
                    .bytes
                    .checked_mul(left.rows)
                    .ok_or(StorageError::Overflow)?,
                pattern,
            }]);
        }
        let mut copies = Vec::new();
        for [left, right] in &pair.rows {
            for row in 0..left.rows {
                copies.push(Self {
                    source: source.clone(),
                    destination: destination.clone(),
                    source_offset: left.offset + row * left.stride,
                    destination_offset: right.offset + row * right.stride,
                    bytes: left.bytes,
                    pattern: CopyPattern::Contiguous,
                });
            }
        }
        let original = coalesce_copies(&copies);
        if same_buffer || original.len() < 2 {
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
        let reordered = coalesce_copies(&copies);
        Ok(if reordered.len() < original.len() {
            reordered
        } else {
            original
        })
    }

    #[cfg(test)]
    pub(crate) fn from_traversals(
        source: Buffer,
        destination: Buffer,
        a: &crate::storage::ByteTraversal,
        b: &crate::storage::ByteTraversal,
    ) -> StorageResult<Vec<Self>> {
        let same_buffer = source == destination;
        Self::from_pair(source, destination, same_buffer, &CopyPair::new(a, b)?)
    }

    #[cfg(test)]
    pub(crate) fn from_spans(
        source: Buffer,
        destination: Buffer,
        source_spans: impl IntoIterator<Item = ByteSpan>,
        destination_spans: impl IntoIterator<Item = ByteSpan>,
    ) -> StorageResult<Vec<Self>> {
        let same_buffer = source == destination;
        Self::from_pair(
            source,
            destination,
            same_buffer,
            &CopyPair::from_spans(source_spans, destination_spans)?,
        )
    }
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
    if rows < WORKER_CONTEXTS
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

fn coalesce_copies<Buffer: Clone>(copies: &[CopyOperation<Buffer>]) -> Vec<CopyOperation<Buffer>> {
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

#[cfg(test)]
mod planning_tests {
    use super::*;

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
