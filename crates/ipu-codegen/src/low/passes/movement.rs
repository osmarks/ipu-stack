//! Local transformations of expanded tile regions, before tile projection.

use crate::kernel::CopyRun;
use crate::low::copy::CopyPattern;
use crate::low::graph::{
    BlockOperation, BlockRegion, BlockValue, BlockValueId, ExchangePhase, ExchangePhaseId,
    LocalCopy, LocalCopyId, LogicalExchange, ShardView, WorkProvenance, WorkReason,
};
use crate::low::storage::{storage_location, storage_root};
use crate::storage::{CopyOrder, StorageResult, StridedSpan};
use std::collections::BTreeMap;

/// Forward a whole-buffer byte copy without changing any consumer's geometry.
/// Count actual writes through aliases; a source must be stable after the copy
/// and the destination must have no other writer. Repeat bindings may supply
/// read-only sources; their destinations and mutable storage remain independent.
pub(in crate::low) fn eliminate_copies(program: &mut crate::low::TileGraph) -> StorageResult<()> {
    let uses = crate::low::uses::StorageUses::analyze(program);
    let roots = &uses.roots;
    let mut alignment = vec![1u32; roots.len()];
    let mut distinct = vec![std::collections::BTreeSet::new(); roots.len()];
    for op in program.body.walk() {
        match op {
            BlockOperation::Compute { run, .. } => {
                let run = &program.kernel_runs[run.0 as usize];
                for group in &run.requirements.distinct_elements {
                    for (i, &a) in group.iter().enumerate() {
                        let a = roots[run.operand_view(a).unwrap().shard.index() as usize];
                        for &b in &group[..i] {
                            let b = roots[run.operand_view(b).unwrap().shard.index() as usize];
                            distinct[a].insert(b);
                            distinct[b].insert(a);
                        }
                    }
                }
                for (shard, access) in run.accesses(&program.shards) {
                    let root = roots[shard.index() as usize];
                    alignment[root] = alignment[root].max(access.alignment);
                }
            }
            BlockOperation::Copy { copy, .. } => {
                let copy = &program.local_copies[copy.0 as usize];
                for (id, access) in copy.accesses() {
                    let root = roots[id.index() as usize];
                    alignment[root] = alignment[root].max(access.alignment);
                }
            }
            BlockOperation::Exchange(_)
            | BlockOperation::Repeat(_)
            | BlockOperation::Checkpoint(..) => {}
        }
    }
    let mut removed = std::collections::BTreeSet::new();
    for (index, op) in program.body.walk().enumerate() {
        let BlockOperation::Copy { copy: id, .. } = op else {
            continue;
        };
        let copy = program.local_copies[id.0 as usize].movement();
        let source = &program.shards[copy.source.index() as usize];
        let destination = &program.shards[copy.destination.index() as usize];
        let source_bytes = crate::low::storage::shard_storage_bytes(source)?;
        let from = roots[copy.source.index() as usize];
        let to = roots[copy.destination.index() as usize];
        let backing = storage_root(&program.shards, copy.source).index() as usize;
        if copy.pattern != CopyPattern::Contiguous
            || copy.destination_offset != 0
            || copy.bytes != crate::low::storage::shard_storage_bytes(destination)?
            || copy
                .source_offset
                .checked_add(copy.bytes)
                .is_none_or(|end| end > source_bytes)
            || source.tile != destination.tile
            || source.tensor_type.format.layout.memory_class
                != destination.tensor_type.format.layout.memory_class
            || from == to
            || to != copy.destination.index() as usize
            || uses.allocations[to].writes != 1
            || uses.allocations[from]
                .last_write
                .is_some_and(|last| last >= index + 1)
            || (uses.allocations[from].boundary && !uses.allocations[from].read_only)
            || uses.allocations[to].boundary
            || distinct[to].contains(&backing)
            || (storage_location(&program.shards, copy.source).1 + i64::from(copy.source_offset))
                .rem_euclid(i64::from(alignment[to]))
                != 0
        {
            continue;
        }
        let Ok(offset) = i32::try_from(copy.source_offset) else {
            continue;
        };
        program.shards[copy.destination.index() as usize].definition = if offset == 0 {
            crate::ShardDefinition::Alias(copy.source)
        } else {
            crate::ShardDefinition::ShiftedAlias {
                source: copy.source,
                offset,
            }
        };
        removed.insert(*id);
        for other in std::mem::take(&mut distinct[to]) {
            distinct[other].remove(&to);
            distinct[other].insert(backing);
            distinct[backing].insert(other);
        }
    }
    program.body.retain(
        &mut |op| !matches!(op, BlockOperation::Copy { copy, .. } if removed.contains(copy)),
    );
    Ok(())
}

/// Move copy-only gaps before or after a shared exchange when they commute
/// with every conflicting access. Scheduling still orders transfers within the
/// merged phase; this pass does not choose their execution order.
#[tracing::instrument(skip_all)]
pub(in crate::low) fn group_exchanges(
    region: &mut BlockRegion,
    phases: &mut Vec<ExchangePhase>,
    copies: &[CopyRun],
    shards: &[BlockValue],
) -> StorageResult<usize> {
    fn group_region(
        region: &mut BlockRegion,
        phases: &mut [ExchangePhase],
        copies: &[CopyRun],
        shards: &[BlockValue],
    ) -> StorageResult<usize> {
        let mut previous: Option<(usize, ExchangePhaseId)> = None;
        let mut merged = 0;
        for mut operation in std::mem::take(&mut region.operations) {
            match &mut operation {
                BlockOperation::Repeat(repeat) => {
                    merged += group_region(&mut repeat.body, phases, copies, shards)?;
                    previous = None;
                }
                BlockOperation::Compute { .. } | BlockOperation::Checkpoint(..) => previous = None,
                _ => {}
            }
            if let BlockOperation::Exchange(next) = operation {
                if let Some((boundary, prior)) = previous {
                    let prior_index = prior.index() as usize;
                    let next_index = next.index() as usize;
                    let provenance = phases[next_index].provenance;
                    let between = &region.operations[boundary + 1..];
                    if phases[prior_index].provenance.operation.is_some()
                        && phases[prior_index].provenance.operation == provenance.operation
                    {
                        let sink =
                            copies_commute(shards, copies, between, &phases[next_index].transfers)?;
                        let hoist = !sink
                            && copies_commute(
                                shards,
                                copies,
                                between,
                                &phases[prior_index].transfers,
                            )?;
                        if sink || hoist {
                            if hoist {
                                region.operations[boundary..].rotate_left(1);
                                previous = Some((region.operations.len() - 1, prior));
                            }
                            let mut transfers = std::mem::take(&mut phases[next_index].transfers);
                            let phase = &mut phases[prior_index];
                            phase.transfers.append(&mut transfers);
                            if phase.provenance != provenance {
                                phase.provenance = WorkProvenance {
                                    operation: provenance.operation,
                                    value: None,
                                    reason: WorkReason::OperatorInputs,
                                };
                            }
                            merged += 1;
                            continue;
                        }
                    }
                }
                previous = Some((region.operations.len(), next));
            }
            region.operations.push(operation);
        }
        Ok(merged)
    }
    fn remap(region: &mut BlockRegion, ids: &[Option<ExchangePhaseId>]) {
        for op in &mut region.operations {
            match op {
                BlockOperation::Exchange(phase) => {
                    *phase =
                        ids[phase.index() as usize].expect("live exchange retains its transfers")
                }
                BlockOperation::Repeat(repeat) => remap(&mut repeat.body, ids),
                _ => {}
            }
        }
    }
    let merged = group_region(region, phases, copies, shards)?;
    if merged != 0 {
        // Exchange consumers index this arena directly. Remove merged entries
        // and update every region reference, including structured Repeat bodies.
        let mut ids = vec![None; phases.len()];
        for mut phase in std::mem::take(phases) {
            if !phase.transfers.is_empty() {
                let id = ExchangePhaseId(phases.len() as u32);
                ids[phase.id.index() as usize] = Some(id);
                phase.id = id;
                phases.push(phase);
            }
        }
        remap(region, &ids);
    }
    Ok(merged)
}

fn copies_commute(
    shards: &[BlockValue],
    local_copies: &[CopyRun],
    operations: &[BlockOperation],
    transfers: &[LogicalExchange],
) -> StorageResult<bool> {
    if operations.is_empty() {
        return Ok(true);
    }
    // Index only allocations touched by the copies. Large materializations
    // must not compare every copy with every transfer in the phase.
    let copies = operations
        .iter()
        .map(|operation| {
            let BlockOperation::Copy { copy, .. } = operation else {
                unreachable!()
            };
            local_copies[copy.0 as usize].movement()
        })
        .collect::<Vec<_>>();
    let mut accesses = BTreeMap::<BlockValueId, Vec<(&ShardView, CopyOrder, bool)>>::new();
    for copy in &copies {
        accesses
            .entry(storage_root(shards, copy.source))
            .or_default();
        accesses
            .entry(storage_root(shards, copy.destination))
            .or_default();
    }
    for transfer in transfers {
        let order = transfer.span_order(shards);
        if let Some(views) = accesses.get_mut(&storage_root(shards, transfer.source.shard)) {
            views.push((&transfer.source, order, false));
        }
        for view in &transfer.destinations {
            if let Some(views) = accesses.get_mut(&storage_root(shards, view.shard)) {
                views.push((view, order, true));
            }
        }
    }
    for copy in copies {
        // Moving in either direction must preserve RAW, WAR and WAW;
        // read/read overlap is harmless. Stream geometry only for aliases.
        for source in [true, false] {
            let root = storage_root(
                shards,
                if source {
                    copy.source
                } else {
                    copy.destination
                },
            );
            for &(view, order, write) in &accesses[&root] {
                if (!source || write) && copy_overlaps_view(shards, copy, source, view, order)? {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

fn copy_overlaps_view(
    shards: &[BlockValue],
    copy: &LocalCopy,
    source: bool,
    view: &ShardView,
    order: CopyOrder,
) -> StorageResult<bool> {
    let (shard, offset) = if source {
        (copy.source, copy.source_offset)
    } else {
        (copy.destination, copy.destination_offset)
    };
    let (root, origin) = storage_location(shards, shard);
    let view = view.bind(shards)?;
    if root != view.backing.0 {
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
    let copy = StridedSpan {
        offset,
        rows,
        bytes,
        stride,
    };
    Ok(view
        .traversal(order)?
        .spans()
        .any(|span| copy.overlaps(origin, span, view.backing.1)))
}

/// Merge contiguous copies between distinct allocations while preserving each
/// tile's compute order and all exchange/repeat/checkpoint boundaries.
/// Rebuild the live copy table as operations are retained, including repeats.
#[tracing::instrument(skip_all)]
pub(super) fn merge_copies(
    region: &mut BlockRegion,
    old: &[CopyRun],
    copies: &mut Vec<CopyRun>,
    roots: &[BlockValueId],
    shards: &[BlockValue],
) -> StorageResult<usize> {
    let mut previous = BTreeMap::<u16, LocalCopyId>::new();
    let mut merged = 0;
    let operations = std::mem::take(&mut region.operations);
    for mut operation in operations {
        match &mut operation {
            BlockOperation::Copy { tile, copy } => {
                let next = old[copy.0 as usize].movement();
                if let Some(first) = previous.get(tile) {
                    let first = copies[first.0 as usize].movement();
                    if first.pattern == CopyPattern::Contiguous
                        && next.pattern == CopyPattern::Contiguous
                        && first.source == next.source
                        && first.destination == next.destination
                        && roots[first.source.index() as usize]
                            != roots[first.destination.index() as usize]
                        && first.source_offset.checked_add(first.bytes) == Some(next.source_offset)
                        && first.destination_offset.checked_add(first.bytes)
                            == Some(next.destination_offset)
                        && let Some(bytes) = first.bytes.checked_add(next.bytes)
                    {
                        let mut combined = first.clone();
                        combined.bytes = bytes;
                        copies[previous[tile].0 as usize] = CopyRun::bind(combined, shards)?;
                        merged += 1;
                        continue;
                    }
                }
                let value = old[copy.0 as usize].clone();
                *copy = LocalCopyId(copies.len() as u32);
                copies.push(value);
                previous.insert(*tile, *copy);
            }
            BlockOperation::Compute { tile, .. } => {
                previous.remove(tile);
            }
            BlockOperation::Repeat(repeat) => {
                previous.clear();
                merged += merge_copies(&mut repeat.body, old, copies, roots, shards)?;
            }
            BlockOperation::Exchange(_) | BlockOperation::Checkpoint(..) => previous.clear(),
        }
        region.operations.push(operation);
    }
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ComputeGraph;
    use crate::low::graph::{BlockRepeat, KernelRunId};

    #[test]
    fn exchange_compaction_preserves_structured_execution_and_boundaries() {
        // Distinct symbolic transfers let us compare the work before and after
        // compaction without relying on the rewritten arena indices.
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [1]).unwrap();
        let first = graph.gelu(input).unwrap();
        graph.gelu(first).unwrap();
        let operation = graph.operations()[0].id;
        let provenance = WorkProvenance {
            operation: Some(operation),
            value: None,
            reason: WorkReason::OperatorInputs,
        };
        let mut phases = (0..12)
            .map(|id| ExchangePhase {
                id: ExchangePhaseId(id),
                provenance,
                transfers: vec![LogicalExchange {
                    source: ShardView {
                        shard: BlockValueId(id),
                        extents: vec![],
                    },
                    destinations: vec![ShardView {
                        shard: BlockValueId(id + 12),
                        extents: vec![],
                    }],
                    order: CopyOrder::Physical,
                }],
            })
            .collect::<Vec<_>>();
        phases[10].provenance.operation = Some(graph.operations()[1].id);
        phases[11].provenance.operation = None;
        let exchange = |id| BlockOperation::Exchange(ExchangePhaseId(id));
        let repeat = |operations| {
            BlockOperation::Repeat(Box::new(BlockRepeat {
                provenance,
                count: 3,
                bindings: vec![],
                body: BlockRegion { operations },
            }))
        };
        let mut region = BlockRegion {
            operations: vec![
                exchange(0),
                exchange(1),
                BlockOperation::Compute {
                    tile: 0,
                    run: KernelRunId(0),
                },
                exchange(2),
                exchange(3),
                BlockOperation::Checkpoint(operation, 0),
                exchange(4),
                repeat(vec![
                    exchange(5),
                    exchange(6),
                    repeat(vec![exchange(7), exchange(8)]),
                ]),
                exchange(9),
                exchange(10),
                exchange(11),
            ],
        };
        fn execution(region: &BlockRegion, phases: &[ExchangePhase]) -> Vec<LogicalExchange> {
            let mut transfers = vec![];
            for operation in &region.operations {
                match operation {
                    BlockOperation::Exchange(id) => {
                        transfers.extend_from_slice(&phases[id.index() as usize].transfers);
                    }
                    BlockOperation::Repeat(repeat) => {
                        for _ in 0..repeat.count {
                            transfers.extend(execution(&repeat.body, phases));
                        }
                    }
                    _ => {}
                }
            }
            transfers
        }
        let expected = execution(&region, &phases);
        assert_eq!(
            group_exchanges(&mut region, &mut phases, &[], &[]).unwrap(),
            4
        );
        assert_eq!(execution(&region, &phases), expected);
        let sizes = region
            .walk()
            .filter_map(|op| match op {
                BlockOperation::Exchange(id) => {
                    let phase = &phases[id.index() as usize];
                    assert_eq!(phase.id, *id);
                    Some(phase.transfers.len())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(sizes, [2, 2, 1, 2, 2, 1, 1, 1]);
        assert_eq!(phases.len(), sizes.len());
        assert_eq!(
            group_exchanges(&mut region, &mut phases, &[], &[]).unwrap(),
            0
        );
    }

    #[test]
    fn merging_respects_tile_dependencies_boundaries_and_aliases() {
        let shards = (0..5)
            .map(|id| BlockValue {
                id: BlockValueId(id),
                tile: if id == 2 || id == 3 { 1 } else { 0 },
                tensor_type: crate::TensorType::new(
                    [64],
                    crate::Precision::F16,
                    crate::Layout::row_sharded(1),
                ),
                extents: vec![crate::ShardExtent {
                    axis: 0,
                    start: 0,
                    logical_end: 64,
                    physical_end: 64,
                }],
                definition: if id == 4 {
                    crate::ShardDefinition::WritableAlias(BlockValueId(0))
                } else {
                    crate::ShardDefinition::Staging
                },
            })
            .collect::<Vec<_>>();
        let mut copies = Vec::new();
        let mut movement = |tile, source, destination, offset, bytes| {
            let copy = LocalCopyId(copies.len() as u32);
            copies.push(
                CopyRun::bind(
                    LocalCopy {
                        source: BlockValueId(source),
                        destination: BlockValueId(destination),
                        source_offset: offset,
                        destination_offset: offset,
                        bytes,
                        pattern: CopyPattern::Contiguous,
                    },
                    &shards,
                )
                .unwrap(),
            );
            BlockOperation::Copy { tile, copy }
        };
        let mut region = BlockRegion {
            operations: vec![
                movement(0, 0, 1, 0, 8),
                movement(1, 2, 3, 0, 8),
                movement(0, 0, 1, 8, 16),
                BlockOperation::Compute {
                    tile: 0,
                    run: KernelRunId(0),
                },
                movement(0, 0, 1, 24, 8),
                BlockOperation::Exchange(ExchangePhaseId(0)),
                movement(0, 0, 1, 32, 8),
                movement(1, 2, 3, 8, 8),
                movement(0, 0, 4, 0, 8),
                movement(0, 0, 4, 8, 8),
                BlockOperation::Repeat(Box::new(BlockRepeat {
                    provenance: WorkProvenance {
                        operation: None,
                        value: None,
                        reason: WorkReason::OperatorInputs,
                    },
                    count: 2,
                    bindings: vec![],
                    body: BlockRegion {
                        operations: vec![movement(0, 0, 1, 0, 8), movement(0, 0, 1, 8, 8)],
                    },
                })),
                movement(0, 0, 1, 16, 8),
            ],
        };
        let roots = [0, 1, 2, 3, 0].map(BlockValueId);
        let mut compact = Vec::new();
        assert_eq!(
            merge_copies(&mut region, &copies, &mut compact, &roots, &shards).unwrap(),
            2
        );
        assert_eq!(compact.len(), 9);
        assert_eq!(compact[0].movement().bytes, 24);
        assert!(compact[1..7].iter().all(|copy| copy.movement().bytes == 8));
        assert_eq!(compact[7].movement().bytes, 16);
        assert_eq!(compact[8].movement().bytes, 8);
        let before = region.clone();
        let mut again = Vec::new();
        assert_eq!(
            merge_copies(&mut region, &compact, &mut again, &roots, &shards).unwrap(),
            0
        );
        assert_eq!(region, before);
        assert_eq!(again, compact);
    }

    #[test]
    fn copy_hazards_translate_both_signed_alias_origins() {
        let root = BlockValue {
            id: BlockValueId(0),
            tile: 0,
            tensor_type: crate::TensorType::new(
                [128],
                crate::Precision::F16,
                crate::Layout::row_sharded(1),
            ),
            extents: vec![crate::ShardExtent {
                axis: 0,
                start: 0,
                logical_end: 128,
                physical_end: 128,
            }],
            definition: crate::ShardDefinition::Staging,
        };
        let mut other = root.clone();
        other.id = BlockValueId(1);
        let mut alias = root.clone();
        alias.id = BlockValueId(2);
        alias.tensor_type.format.precision = crate::Precision::F8F143 { scale_exponent: 0 };
        let view = ShardView {
            shard: alias.id,
            extents: vec![crate::ShardExtent {
                axis: 0,
                start: 4,
                logical_end: 8,
                physical_end: 8,
            }],
        };
        let copy = LocalCopy {
            source: root.id,
            destination: other.id,
            source_offset: 32,
            destination_offset: 0,
            bytes: 8,
            pattern: CopyPattern::Contiguous,
        };
        let mut shards = vec![root, other, alias];
        for (shift, overlap) in [(-8, false), (28, true), (36, false), (1024, false)] {
            shards[2].definition = crate::ShardDefinition::ShiftedAlias {
                source: BlockValueId(0),
                offset: shift,
            };
            assert_eq!(
                copy_overlaps_view(&shards, &copy, true, &view, CopyOrder::Physical).unwrap(),
                overlap
            );
            assert!(
                !copy_overlaps_view(&shards, &copy, false, &view, CopyOrder::Physical).unwrap()
            );
        }
        shards[0].definition = crate::ShardDefinition::ShiftedAlias {
            source: BlockValueId(1),
            offset: -16,
        };
        shards[2].definition = crate::ShardDefinition::ShiftedAlias {
            source: BlockValueId(1),
            offset: 12,
        };
        assert!(copy_overlaps_view(&shards, &copy, true, &view, CopyOrder::Physical).unwrap());
    }
}
