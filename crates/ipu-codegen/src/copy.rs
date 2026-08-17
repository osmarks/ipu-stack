//! Address-independent plans for tile-local memory copies.

use crate::storage::ByteSpan;

/// Target policy used while turning storage spans into tile-kernel calls.
///
/// The policy is selected before low-level lowering.  Lowering binds the
/// resulting relative offsets to shards, but does not choose between copy
/// implementations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CopyPlan {
    pub parallel_strided_max_bytes: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

/// One relative copy invocation selected from two storage-span sequences.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CopyRun {
    pub source_offset: u32,
    pub destination_offset: u32,
    /// Total bytes copied by this invocation, including all strided rows.
    pub bytes: u32,
    pub pattern: CopyPattern,
}

pub(crate) fn plan_transfer_runs(
    source_spans: &[ByteSpan],
    destination_spans: &[ByteSpan],
    maximum_bytes: u32,
    alignment: u32,
) -> Option<Vec<CopyRun>> {
    if maximum_bytes == 0
        || alignment == 0
        || source_spans.iter().chain(destination_spans).any(|span| {
            !span.offset.is_multiple_of(alignment) || !span.bytes.is_multiple_of(alignment)
        })
    {
        return None;
    }
    let mut runs = Vec::new();
    let mut source_index = 0usize;
    let mut destination_index = 0usize;
    let mut source_offset = 0u32;
    let mut destination_offset = 0u32;
    while source_index < source_spans.len() && destination_index < destination_spans.len() {
        let source = source_spans[source_index];
        let destination = destination_spans[destination_index];
        let bytes = (source.bytes - source_offset)
            .min(destination.bytes - destination_offset)
            .min(maximum_bytes);
        if bytes == 0 || !bytes.is_multiple_of(alignment) {
            return None;
        }
        runs.push(CopyRun {
            source_offset: source.offset + source_offset,
            destination_offset: destination.offset + destination_offset,
            bytes,
            pattern: CopyPattern::Contiguous,
        });
        source_offset += bytes;
        destination_offset += bytes;
        if source_offset == source.bytes {
            source_index += 1;
            source_offset = 0;
        }
        if destination_offset == destination.bytes {
            destination_index += 1;
            destination_offset = 0;
        }
    }
    if source_index != source_spans.len() || destination_index != destination_spans.len() {
        return None;
    }
    Some(runs)
}

/// Plans calls over corresponding physical spans.
///
/// Contiguous adjacent rows become one call.  Short regular non-contiguous
/// runs use the parallel strided kernel selected by the target policy.  The
/// result contains only relative offsets, so both planning estimates and
/// shard-bound low lowering can consume it.
pub(crate) fn plan_copy_runs(
    source_spans: &[ByteSpan],
    destination_spans: &[ByteSpan],
    plan: CopyPlan,
) -> Option<Vec<CopyRun>> {
    let rows = plan_transfer_runs(source_spans, destination_spans, u32::MAX, 1)?;

    let mut runs = Vec::new();
    let mut index = 0;
    while index < rows.len() {
        let first = rows[index];
        let Some(second) = rows.get(index + 1).copied() else {
            runs.push(first);
            break;
        };
        if first.bytes != second.bytes || first.bytes == 0 || !first.bytes.is_multiple_of(8) {
            runs.push(first);
            index += 1;
            continue;
        }
        let Some(source_stride) = second.source_offset.checked_sub(first.source_offset) else {
            runs.push(first);
            index += 1;
            continue;
        };
        let Some(destination_stride) = second
            .destination_offset
            .checked_sub(first.destination_offset)
        else {
            runs.push(first);
            index += 1;
            continue;
        };
        if source_stride == 0 || destination_stride == 0 {
            runs.push(first);
            index += 1;
            continue;
        }
        let mut end = index + 2;
        while let Some(copy) = rows.get(end) {
            let previous = rows[end - 1];
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
        let row_count = u32::try_from(end - index).unwrap_or(u32::MAX);
        let total_bytes = first.bytes.saturating_mul(row_count);
        if source_stride == first.bytes && destination_stride == first.bytes {
            runs.push(CopyRun {
                bytes: total_bytes,
                ..first
            });
        } else if total_bytes <= plan.parallel_strided_max_bytes {
            runs.push(CopyRun {
                bytes: total_bytes,
                pattern: CopyPattern::Strided {
                    rows: row_count,
                    row_bytes: first.bytes,
                    source_stride,
                    destination_stride,
                },
                ..first
            });
        } else {
            runs.extend_from_slice(&rows[index..end]);
        }
        index = end;
    }
    Some(runs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn randomized_copy_plans_preserve_span_streams() {
        let mut random = fastrand::Rng::with_seed(0x636f_7079_7275_6e73);
        for case in 0..256 {
            let row_bytes = random.u32(1..=16) * 8;
            let rows = random.u32(1..=32);
            let source_stride = row_bytes + random.u32(0..=8) * 8;
            let destination_stride = row_bytes + random.u32(0..=8) * 8;
            let source = (0..rows)
                .map(|row| ByteSpan {
                    offset: 16 + row * source_stride,
                    bytes: row_bytes,
                })
                .collect::<Vec<_>>();
            let destination = (0..rows)
                .map(|row| ByteSpan {
                    offset: 32 + row * destination_stride,
                    bytes: row_bytes,
                })
                .collect::<Vec<_>>();
            let maximum = random.u32(1..=32) * row_bytes;
            let runs = plan_copy_runs(
                &source,
                &destination,
                CopyPlan {
                    parallel_strided_max_bytes: maximum,
                },
            )
            .unwrap();
            assert_eq!(
                runs.iter().map(|run| u64::from(run.bytes)).sum::<u64>(),
                u64::from(row_bytes) * u64::from(rows),
                "case {case}"
            );
            for run in runs {
                match run.pattern {
                    CopyPattern::Contiguous => {}
                    CopyPattern::Strided {
                        rows,
                        row_bytes,
                        source_stride,
                        destination_stride,
                    } => {
                        assert!(rows >= 2, "case {case}");
                        assert_eq!(run.bytes, rows * row_bytes, "case {case}");
                        assert!(source_stride >= row_bytes, "case {case}");
                        assert!(destination_stride >= row_bytes, "case {case}");
                        assert!(run.bytes <= maximum, "case {case}");
                    }
                }
            }
        }
    }
}
