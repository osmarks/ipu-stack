//! Local byte-copy family: checked helper choice, access footprint, ABI and cost.
//! Movement construction supplies byte geometry; no numerical tensor type is
//! invented for it. Graph rewrites rebind a changed descriptor explicitly.

use super::{KernelCall, StorageError};
use crate::low::storage::{StorageAccess, bind_storage};
#[cfg(test)]
use crate::storage::ByteSpan;
use crate::storage::{CopyPair, StorageResult};
use crate::{BlockValue, BlockValueId, CopyOperation, CopyPattern, LocalCopy};

use ipu_target::ipu21::WORKER_CONTEXTS;

/// Coarse launch allowance when mid has not selected a local helper yet.
pub(crate) const WORKER_CALL_CYCLES: u64 = 288;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CopyKernel {
    U16,
    U32,
    U64,
    StridedU32,
    StridedU64,
}
impl CopyKernel {
    fn width(self) -> u32 {
        match self {
            Self::U16 => 2,
            Self::U32 | Self::StridedU32 => 4,
            Self::U64 | Self::StridedU64 => 8,
        }
    }
    fn symbol(self) -> &'static str {
        match self {
            Self::U16 => crate::kernel::abi::COPY_U16_SYMBOL,
            Self::U32 => crate::kernel::abi::COPY_U32_SYMBOL,
            Self::U64 => crate::kernel::abi::COPY_U64_SYMBOL,
            Self::StridedU32 => crate::kernel::abi::COPY_STRIDED_U32_SYMBOL,
            Self::StridedU64 => crate::kernel::abi::COPY_STRIDED_U64_SYMBOL,
        }
    }
}

/// A local copy in the executable low graph. The descriptor and helper cannot
/// be mutated independently: coalescing constructs another checked binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyRun {
    movement: LocalCopy,
    kernel: CopyKernel,
    access: [StorageAccess; 2],
}

impl CopyRun {
    pub(crate) fn bind(movement: LocalCopy, shards: &[BlockValue]) -> Result<Self, StorageError> {
        let aligned = |width| {
            movement.source_offset.is_multiple_of(width)
                && movement.destination_offset.is_multiple_of(width)
        };
        let (rows, row_bytes, source_stride, destination_stride) = match movement.pattern {
            CopyPattern::Contiguous => (1, movement.bytes, 0, 0),
            CopyPattern::Strided {
                rows,
                row_bytes,
                source_stride,
                destination_stride,
            } if rows >= 2 && row_bytes.checked_mul(rows) == Some(movement.bytes) => {
                (rows, row_bytes, source_stride, destination_stride)
            }
            _ => return Err(StorageError::InvalidView),
        };
        if row_bytes == 0 {
            return Err(StorageError::InvalidView);
        }
        let kernel = if rows != 1 {
            [CopyKernel::StridedU64, CopyKernel::StridedU32]
                .into_iter()
                .find(|kernel| {
                    let width = kernel.width();
                    aligned(width)
                        && row_bytes.is_multiple_of(width)
                        && source_stride.is_multiple_of(width)
                        && destination_stride.is_multiple_of(width)
                })
        } else if aligned(8)
            && movement.bytes >= WORKER_CONTEXTS * 8
            && movement.bytes.is_multiple_of(8)
        {
            Some(CopyKernel::U64)
        } else if aligned(4) && movement.bytes.is_multiple_of(4) {
            Some(CopyKernel::U32)
        } else if aligned(2) && movement.bytes.is_multiple_of(2) {
            Some(CopyKernel::U16)
        } else {
            None
        }
        .ok_or(StorageError::InvalidView)?;

        let mut access = [StorageAccess::default(); 2];
        let mut tile = None;
        for (index, (id, offset, stride)) in [
            (movement.source, movement.source_offset, source_stride),
            (
                movement.destination,
                movement.destination_offset,
                destination_stride,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let (shard, _) = bind_storage(shards, id)?;
            if tile
                .replace(shard.tile)
                .is_some_and(|previous| previous != shard.tile)
            {
                return Err(StorageError::InvalidView);
            }
            let bytes = crate::shard_storage_bytes(shard)?;
            let end = (rows - 1)
                .checked_mul(stride)
                .and_then(|step| offset.checked_add(step))
                .and_then(|start| start.checked_add(row_bytes))
                .ok_or(StorageError::Overflow)?;
            if end > bytes {
                return Err(StorageError::InvalidView);
            }
            // The halfword helper reads/modifies an aligned 32-bit word. Align
            // the base so it needs no prefix; retain any access past a short tail.
            let alignment = kernel.width().max(4);
            let physical_end = end
                .checked_next_multiple_of(alignment)
                .ok_or(StorageError::Overflow)?;
            access[index] = StorageAccess {
                alignment,
                access_tail_bytes: physical_end.saturating_sub(bytes),
            };
        }
        Ok(Self {
            movement,
            kernel,
            access,
        })
    }

    pub(crate) fn movement(&self) -> &LocalCopy {
        &self.movement
    }

    pub(crate) fn accesses(&self) -> impl Iterator<Item = (BlockValueId, StorageAccess)> {
        [self.movement.source, self.movement.destination]
            .into_iter()
            .zip(self.access)
    }

    pub(crate) fn symbol(&self) -> &'static str {
        self.kernel.symbol()
    }

    pub(crate) fn call(&self) -> KernelCall {
        let words = self.movement.bytes / self.kernel.width();
        let arguments = match self.movement.pattern {
            CopyPattern::Strided {
                rows,
                row_bytes,
                source_stride,
                destination_stride,
            } => vec![
                row_bytes / self.kernel.width(),
                rows,
                source_stride,
                destination_stride,
            ],
            CopyPattern::Contiguous if self.kernel == CopyKernel::U64 => {
                vec![words / WORKER_CONTEXTS, words % WORKER_CONTEXTS]
            }
            CopyPattern::Contiguous => vec![words],
        };
        KernelCall::exact(self.symbol(), arguments)
    }

    pub(crate) fn cycles(&self) -> u64 {
        let words = u64::from(self.movement.bytes / self.kernel.width());
        let workers = u64::from(WORKER_CONTEXTS);
        match self.kernel {
            // Supervisor loops: measured setup and steady-state issue cost.
            CopyKernel::U16 => 112 + words * 129,
            CopyKernel::U32 => 104 + words * 23,
            CopyKernel::U64 => 246 + words.div_ceil(workers) * workers * 2,
            CopyKernel::StridedU32 | CopyKernel::StridedU64 => {
                let CopyPattern::Strided {
                    rows, row_bytes, ..
                } = self.movement.pattern
                else {
                    unreachable!()
                };
                // Current loops issue one load and one store per word, plus
                // RPT and five row-update instructions. Each worker owns rows
                // context, context + WORKER_CONTEXTS, ... .
                294 + u64::from(rows).div_ceil(workers)
                    * workers
                    * (2 * u64::from(row_bytes / self.kernel.width()) + 6)
            }
        }
    }
}

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
        Self::group(copies, !same_buffer)
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

    fn group(mut copies: Vec<Self>, can_reorder: bool) -> StorageResult<Vec<Self>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::ByteSpan;
    fn buffers(bytes: &[u32]) -> Vec<BlockValue> {
        bytes
            .iter()
            .enumerate()
            .map(|(index, &bytes)| BlockValue {
                id: BlockValueId::from_index(index as u32),
                tile: 0,
                tensor_type: crate::TensorType::new(
                    [bytes / 2],
                    crate::Precision::F16,
                    crate::Layout::row_sharded(1),
                ),
                extents: vec![crate::ShardExtent {
                    axis: 0,
                    start: 0,
                    logical_end: bytes / 2,
                    physical_end: bytes / 2,
                }],
                definition: crate::ShardDefinition::Staging,
            })
            .collect()
    }

    #[test]
    fn binding_checks_byte_bounds_and_reserves_halfword_accesses() {
        let mut shards = buffers(&[10, 18]);
        let copy = LocalCopy {
            source: BlockValueId(0),
            source_offset: 8,
            destination: BlockValueId(1),
            destination_offset: 16,
            bytes: 2,
            pattern: CopyPattern::Contiguous,
        };
        let bound = CopyRun::bind(copy.clone(), &shards).unwrap();
        for (shard, access) in bound.accesses() {
            let bytes = crate::shard_storage_bytes(&shards[shard.index() as usize]).unwrap();
            assert_eq!((bytes + access.access_tail_bytes) % 4, 0);
            assert_eq!(access.access_tail_bytes, 2);
        }
        let mut invalid = copy.clone();
        invalid.bytes = 4;
        assert!(CopyRun::bind(invalid, &shards).is_err());
        for offset in [1, 9, 10, u32::MAX] {
            let mut invalid = copy.clone();
            invalid.source_offset = offset;
            assert!(CopyRun::bind(invalid, &shards).is_err());
        }
        shards[1].tile = 1;
        assert!(CopyRun::bind(copy.clone(), &shards).is_err());
        shards[1].tile = 0;
        shards[0].definition = crate::ShardDefinition::Unmaterialized;
        assert!(CopyRun::bind(copy, &shards).is_err());

        let shards = buffers(&[64, 64]);
        let mut copy = LocalCopy {
            source: BlockValueId(0),
            source_offset: 0,
            destination: BlockValueId(1),
            destination_offset: 0,
            bytes: 16,
            pattern: CopyPattern::Strided {
                rows: 2,
                row_bytes: 8,
                source_stride: 56,
                destination_stride: 56,
            },
        };
        CopyRun::bind(copy.clone(), &shards).unwrap();
        for stride in [60, 64, u32::MAX - 3] {
            copy.pattern = CopyPattern::Strided {
                rows: 2,
                row_bytes: 8,
                source_stride: stride,
                destination_stride: 56,
            };
            assert!(CopyRun::bind(copy.clone(), &shards).is_err());
        }
    }
    #[test]
    fn randomized_local_copy_calls_respect_alignment_and_worker_counts() {
        let mut random = fastrand::Rng::with_seed(0x636f_7079);
        for _ in 0..1_000 {
            let words = random.u32(1..=4_096);
            let bytes = words * 4;
            let copy = crate::LocalCopy {
                source: crate::BlockValueId::from_index(0),
                source_offset: 2 * random.u32(0..4),
                destination: crate::BlockValueId::from_index(1),
                destination_offset: 2 * random.u32(0..4),
                bytes,
                pattern: crate::CopyPattern::Contiguous,
            };
            let shards = buffers(&[bytes + 8, bytes + 8]);
            let run = CopyRun::bind(copy.clone(), &shards).unwrap();
            let symbol = run.symbol();
            let arguments = run.call().arguments;
            if symbol == crate::kernel::abi::COPY_U64_SYMBOL {
                assert!(copy.source_offset.is_multiple_of(8));
                assert!(copy.destination_offset.is_multiple_of(8));
                assert!(arguments[0] != 0);
                assert_eq!((arguments[0] * 6 + arguments[1]) * 8, bytes);
                assert!(arguments[1] < 6);
            } else if symbol == crate::kernel::abi::COPY_U32_SYMBOL {
                assert!(copy.source_offset.is_multiple_of(4));
                assert!(copy.destination_offset.is_multiple_of(4));
                assert_eq!(arguments, [words]);
            } else {
                assert_eq!(symbol, crate::kernel::abi::COPY_U16_SYMBOL);
                assert_eq!(arguments, [bytes / 2]);
            }
        }
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
                let copies = LocalCopy::from_spans(
                    crate::BlockValueId::from_index(0),
                    crate::BlockValueId::from_index(1),
                    spans(offset, source_stride),
                    spans(0, destination_stride),
                )
                .unwrap();
                assert_eq!(copies.len(), 1);
                let shards = buffers(&[offset + rows * source_stride, rows * destination_stride]);
                let run = CopyRun::bind(copies[0].clone(), &shards).unwrap();
                let symbol = run.symbol();
                let args = run.call().arguments;
                assert_eq!(symbol, crate::kernel::abi::COPY_STRIDED_U32_SYMBOL);
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
}
