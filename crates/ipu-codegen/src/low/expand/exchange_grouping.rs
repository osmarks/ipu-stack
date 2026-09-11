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
        if between.is_empty() {
            return Ok(true);
        }
        // Preserve compute, checkpoints and Repeat boundaries. Moving the whole
        // group preserves dependencies between its copies and fills as well.
        let Some(accesses) = self.local_accesses(between)? else {
            return Ok(false);
        };
        if self.accesses_commute(&accesses, next)? {
            return Ok(true);
        }
        if self.accesses_commute(&accesses, &self.phases[previous.index() as usize].transfers)? {
            region.operations[boundary..].rotate_left(1);
            return Ok(true);
        }
        Ok(false)
    }

    fn local_accesses(
        &self,
        operations: &[BlockOperation],
    ) -> ExpansionResult<Option<Vec<LocalAccess>>> {
        let mut accesses = Vec::new();
        for operation in operations {
            match operation {
                BlockOperation::Copy { copy, .. } => {
                    let copy = &self.local_copies[copy.0 as usize];
                    for write in [false, true] {
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
                                if write {
                                    destination_stride
                                } else {
                                    source_stride
                                },
                            ),
                        };
                        accesses.push(LocalAccess {
                            root: self.storage_root(if write {
                                copy.destination
                            } else {
                                copy.source
                            }),
                            offset: if write {
                                copy.destination_offset
                            } else {
                                copy.source_offset
                            },
                            rows,
                            bytes,
                            stride,
                            write,
                        });
                    }
                }
                BlockOperation::Compute { run, .. } => {
                    let run = &self.kernel_runs[run.0 as usize];
                    let TileKernelSpec::FillZero { offset, bytes, .. } = run.kernel else {
                        return Ok(None);
                    };
                    let view = view_byte_traversal(
                        &self.shards[run.output.shard.index() as usize],
                        &run.output,
                        CopyOrder::Physical,
                    )?;
                    let Some(span) = view.contiguous_span() else {
                        return Ok(None);
                    };
                    accesses.push(LocalAccess {
                        root: self.storage_root(run.output.shard),
                        offset: span
                            .offset
                            .checked_add(offset)
                            .ok_or(StorageError::Overflow)?,
                        rows: 1,
                        bytes,
                        stride: 0,
                        write: true,
                    });
                }
                _ => return Ok(None),
            }
        }
        Ok(Some(accesses))
    }

    fn accesses_commute(
        &self,
        local: &[LocalAccess],
        transfers: &[LogicalExchange],
    ) -> ExpansionResult<bool> {
        // Index only allocations touched locally. Large materializations must
        // not compare every local access with every transfer in the phase.
        let mut accesses = BTreeMap::<BlockValueId, Vec<(&ShardView, CopyOrder, bool)>>::new();
        for access in local {
            accesses.entry(access.root).or_default();
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
        for access in local {
            // Preserve RAW, WAR and WAW; shared reads are harmless.
            for &(view, order, write) in &accesses[&access.root] {
                if !access.write && !write {
                    continue;
                }
                let traversal =
                    view_byte_traversal(&self.shards[view.shard.index() as usize], view, order)?;
                if traversal.spans().any(|span| {
                    strided_overlap(
                        access.offset,
                        access.rows,
                        access.bytes,
                        access.stride,
                        span,
                    )
                }) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}

/// A contiguous or strided read/write, resolved to its allocation root.
struct LocalAccess {
    root: BlockValueId,
    offset: u32,
    rows: u32,
    bytes: u32,
    stride: u32,
    write: bool,
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
