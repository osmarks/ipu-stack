//! Relative copy operations and materialization policy, before placement.

use crate::storage::{ByteSpan, ByteTraversal, StorageError, StorageResult, for_each_copy_span};

/// Requested realization of a whole-device coordinate copy. Explicit requests
/// are checked by movement lowering; Automatic selects from the actual geometry.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum CopyPolicy {
    #[default]
    Automatic,
    /// Run a rearrangement kernel on corresponding resident shards.
    LocalKernel,
    /// Transfer compatible physical spans directly to their destination.
    DirectRetile,
    /// Transfer logical values through row-major staging, then pack locally.
    StageLogicalThenTransform,
}

pub fn default_copy_policy(from: &crate::Layout, to: &crate::Layout) -> CopyPolicy {
    if from.order == to.order {
        CopyPolicy::DirectRetile
    } else {
        CopyPolicy::StageLogicalThenTransform
    }
}

/// Destination preparation for a selected copy. This is separate from its
/// logical/physical traversal policy; changing it does not change tensor values.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum PackingPolicy {
    #[default]
    Automatic,
    /// Use direct word movement without destination packing scratch.
    Direct,
    /// Populate row-major scratch, then pack into the destination.
    Staged,
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
        if let Some((left, right)) = source_traversal.regular_copy(destination_traversal)
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
        let original = coalesce_copies(&copies);
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
        let reordered = coalesce_copies(&copies);
        Ok(if reordered.len() < original.len() {
            reordered
        } else {
            original
        })
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
mod tests {
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
