//! Move local copies out of exchange boundaries; transfer ordering belongs to
//! the physical exchange scheduler, which handles receive-then-forward chains.
use super::*;

impl TileGraphBuilder {
    pub(super) fn group_exchange_copies(
        &self,
        previous: ExchangePhaseId,
        next: &[LogicalExchange],
        region: &mut BlockRegion,
    ) -> ExpansionResult<bool> {
        let Some(boundary) = region
            .operations
            .iter()
            .rposition(|op| *op == BlockOperation::Exchange(previous))
        else {
            return Ok(false);
        };
        let between = &region.operations[boundary + 1..];
        // Preserve compute, checkpoints and Repeat boundaries. Moving the whole
        // copy group preserves dependencies between copies as well.
        if !between
            .iter()
            .all(|op| matches!(op, BlockOperation::Copy { .. }))
        {
            return Ok(false);
        }
        if self.copies_commute(between, next)? {
            return Ok(true);
        }
        if self.copies_commute(between, &self.phases[previous.index() as usize].transfers)? {
            region.operations[boundary..].rotate_left(1);
            return Ok(true);
        }
        Ok(false)
    }

    fn copies_commute(
        &self,
        operations: &[BlockOperation],
        transfers: &[LogicalExchange],
    ) -> ExpansionResult<bool> {
        // Index only allocations touched by the copies. Large materializations
        // must not compare every copy with every transfer in the phase.
        let copies = operations
            .iter()
            .map(|operation| {
                let BlockOperation::Copy { copy, .. } = operation else {
                    unreachable!()
                };
                &self.local_copies[copy.0 as usize]
            })
            .collect::<Vec<_>>();
        let mut accesses = BTreeMap::<BlockValueId, Vec<(&ShardView, CopyOrder, bool)>>::new();
        for copy in &copies {
            accesses.entry(self.storage_root(copy.source)).or_default();
            accesses
                .entry(self.storage_root(copy.destination))
                .or_default();
        }
        for transfer in transfers {
            let order = transfer.span_order(&self.shards);
            if let Some(views) = accesses.get_mut(&self.storage_root(transfer.source.shard)) {
                views.push((&transfer.source, order, false));
            }
            for view in &transfer.destinations {
                if let Some(views) = accesses.get_mut(&self.storage_root(view.shard)) {
                    views.push((view, order, true));
                }
            }
        }
        for copy in copies {
            // Moving in either direction must preserve RAW, WAR and WAW;
            // read/read overlap is harmless. Stream geometry only for aliases.
            for source in [true, false] {
                let root = self.storage_root(if source {
                    copy.source
                } else {
                    copy.destination
                });
                for &(view, order, write) in &accesses[&root] {
                    if (!source || write) && self.copy_overlaps_view(copy, source, view, order)? {
                        return Ok(false);
                    }
                }
            }
        }
        Ok(true)
    }

    fn copy_overlaps_view(
        &self,
        copy: &LocalCopy,
        source: bool,
        view: &ShardView,
        order: CopyOrder,
    ) -> ExpansionResult<bool> {
        let (shard, offset) = if source {
            (copy.source, copy.source_offset)
        } else {
            (copy.destination, copy.destination_offset)
        };
        if self.storage_root(shard) != self.storage_root(view.shard) {
            return Ok(false);
        }
        let (rows, bytes, stride) = match copy.pattern {
            CopyPattern::Contiguous => (1, copy.bytes, 0),
            CopyPattern::Strided {
                rows,
                row_bytes,
                source_stride,
                destination_stride,
            } => (
                rows,
                row_bytes,
                if source {
                    source_stride
                } else {
                    destination_stride
                },
            ),
        };
        let traversal =
            view_byte_traversal(&self.shards[view.shard.index() as usize], view, order)?;
        Ok(traversal
            .spans()
            .any(|span| strided_overlap(offset, rows, bytes, stride, span)))
    }
}

/// Compare a span with an affine row sequence without enumerating its rows.
fn strided_overlap(offset: u32, rows: u32, bytes: u32, stride: u32, span: ByteSpan) -> bool {
    if rows == 0 || bytes == 0 || span.bytes == 0 {
        return false;
    }
    let (offset, bytes, stride) = (u64::from(offset), u64::from(bytes), u64::from(stride));
    let start = u64::from(span.offset);
    let end = start + u64::from(span.bytes);
    if stride == 0 {
        return offset < end && start < offset + bytes;
    }
    let first = if start < offset + bytes {
        0
    } else {
        (start - offset - bytes) / stride + 1
    };
    first < u64::from(rows) && offset + first * stride < end
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn affine_overlap_matches_explicit_rows() {
        let mut random = fastrand::Rng::with_seed(1729);
        for _ in 0..10_000 {
            let offset = random.u32(0..64);
            let rows = random.u32(0..16);
            let bytes = random.u32(0..32);
            let stride = random.u32(0..64);
            let span = ByteSpan {
                offset: random.u32(0..1024),
                bytes: random.u32(0..128),
            };
            let expected = bytes != 0
                && span.bytes != 0
                && (0..rows).any(|row| {
                    let at = offset + row * stride;
                    at < span.offset + span.bytes && span.offset < at + bytes
                });
            assert_eq!(strided_overlap(offset, rows, bytes, stride, span), expected);
        }
    }
}
