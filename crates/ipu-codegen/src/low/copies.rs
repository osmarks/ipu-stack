//! Local byte-span emission and physical copy coalescing.

use super::*;

pub(super) fn append_span_copies(
    shards: &[LowShard],
    source: &ShardView,
    destination: &ShardView,
    tile: u16,
    copies: &mut Vec<(u16, LocalCopy)>,
) -> LowLoweringResult<()> {
    let source_spans = view_byte_spans(&shards[source.shard.index() as usize], source)?;
    let destination_spans =
        view_byte_spans(&shards[destination.shard.index() as usize], destination)?;
    append_byte_span_copies(
        source,
        destination,
        tile,
        &source_spans,
        &destination_spans,
        copies,
    )
}

pub(super) fn append_byte_span_copies(
    source: &ShardView,
    destination: &ShardView,
    tile: u16,
    source_spans: &[ByteSpan],
    destination_spans: &[ByteSpan],
    copies: &mut Vec<(u16, LocalCopy)>,
) -> LowLoweringResult<()> {
    let mut pending = Vec::new();
    let mut source_index = 0usize;
    let mut destination_index = 0usize;
    let mut source_offset = 0u32;
    let mut destination_offset = 0u32;
    while source_index < source_spans.len() && destination_index < destination_spans.len() {
        let source_span = source_spans[source_index];
        let destination_span = destination_spans[destination_index];
        let bytes =
            (source_span.bytes - source_offset).min(destination_span.bytes - destination_offset);
        pending.push(LocalCopy {
            source: source.shard,
            source_offset: source_span.offset + source_offset,
            destination: destination.shard,
            destination_offset: destination_span.offset + destination_offset,
            bytes,
            pattern: LocalCopyPattern::Contiguous,
        });
        source_offset += bytes;
        destination_offset += bytes;
        if source_offset == source_span.bytes {
            source_index += 1;
            source_offset = 0;
        }
        if destination_offset == destination_span.bytes {
            destination_index += 1;
            destination_offset = 0;
        }
    }
    if source_index != source_spans.len() || destination_index != destination_spans.len() {
        return Err(LowLoweringError::InvalidConversionPlan);
    }
    copies.extend(
        coalesce_local_copies(pending)
            .into_iter()
            .map(|copy| (tile, copy)),
    );
    Ok(())
}

pub(super) const PARALLEL_STRIDED_COPY_MAX_BYTES: u32 = 512;

pub(super) fn coalesce_local_copies(copies: Vec<LocalCopy>) -> Vec<LocalCopy> {
    let mut coalesced = Vec::new();
    let mut index = 0;
    while index < copies.len() {
        let first = &copies[index];
        let Some(second) = copies.get(index + 1) else {
            coalesced.push(first.clone());
            break;
        };
        if first.source != second.source
            || first.destination != second.destination
            || first.bytes != second.bytes
            || first.bytes == 0
            || !first.bytes.is_multiple_of(8)
        {
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
            if copy.source != first.source
                || copy.destination != first.destination
                || copy.bytes != first.bytes
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
            copy.pattern = LocalCopyPattern::Strided {
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

pub(super) fn append_logical_span_copies(
    shards: &[LowShard],
    source: &ShardView,
    destination: &ShardView,
    tile: u16,
    copies: &mut Vec<(u16, LocalCopy)>,
) -> LowLoweringResult<()> {
    let source_spans = logical_view_byte_spans(&shards[source.shard.index() as usize], source)?;
    let destination_spans =
        logical_view_byte_spans(&shards[destination.shard.index() as usize], destination)?;
    append_byte_span_copies(
        source,
        destination,
        tile,
        &source_spans,
        &destination_spans,
        copies,
    )
}
