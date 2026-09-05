//! Local transformations of executable mid regions, before tile projection.

use super::*;
use std::collections::BTreeMap;

/// Merge contiguous copies between distinct allocations while preserving each
/// tile's compute order and all exchange/repeat/checkpoint boundaries.
fn merge_copies(
    region: &mut BlockRegion,
    copies: &mut [LocalCopy],
    roots: &[BlockValueId],
) -> usize {
    let mut previous = BTreeMap::<u16, LocalCopyId>::new();
    let mut merged = 0;
    let operations = std::mem::take(&mut region.operations);
    for mut operation in operations {
        match &mut operation {
            BlockOperation::Copy { tile, copy } => {
                let next = &copies[copy.0 as usize];
                if let Some(first) = previous.get(tile) {
                    let first = &copies[first.0 as usize];
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
                        copies[previous[tile].0 as usize].bytes = bytes;
                        merged += 1;
                        continue;
                    }
                }
                previous.insert(*tile, *copy);
            }
            BlockOperation::Compute { tile, .. } => {
                previous.remove(tile);
            }
            BlockOperation::Repeat(repeat) => {
                previous.clear();
                merged += merge_copies(&mut repeat.body, copies, roots);
            }
            BlockOperation::Exchange(_) | BlockOperation::Checkpoint(..) => previous.clear(),
        }
        region.operations.push(operation);
    }
    merged
}

fn compact_copies(region: &mut BlockRegion, old: &[LocalCopy], copies: &mut Vec<LocalCopy>) {
    for operation in &mut region.operations {
        match operation {
            BlockOperation::Copy { copy, .. } => {
                let value = old[copy.0 as usize].clone();
                *copy = LocalCopyId(copies.len() as u32);
                copies.push(value);
            }
            BlockOperation::Repeat(repeat) => compact_copies(&mut repeat.body, old, copies),
            _ => {}
        }
    }
}

pub(super) fn simplify(program: &mut TileGraph) {
    let roots = program
        .shards
        .iter()
        .map(|value| storage_root(&program.shards, value.id))
        .collect::<Vec<_>>();
    let merged = merge_copies(&mut program.body, &mut program.local_copies, &roots);
    if merged != 0 {
        let old = std::mem::take(&mut program.local_copies);
        compact_copies(&mut program.body, &old, &mut program.local_copies);
        tracing::debug!(merged, "merged adjacent mid copies");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merging_respects_tile_dependencies_boundaries_and_aliases() {
        let mut copies = Vec::new();
        let mut movement = |tile, source, destination, offset, bytes| {
            let copy = LocalCopyId(copies.len() as u32);
            copies.push(LocalCopy {
                source: BlockValueId(source),
                destination: BlockValueId(destination),
                source_offset: offset,
                destination_offset: offset,
                bytes,
                pattern: CopyPattern::Contiguous,
            });
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
            ],
        };
        let roots = [0, 1, 2, 3, 0].map(BlockValueId);
        assert_eq!(merge_copies(&mut region, &mut copies, &roots), 1);
        let mut compact = Vec::new();
        compact_copies(&mut region, &copies, &mut compact);
        assert_eq!(compact.len(), 7);
        assert_eq!(compact[0].bytes, 24);
        assert!(compact[1..].iter().all(|copy| copy.bytes == 8));
        assert_eq!(merge_copies(&mut region, &mut compact, &roots), 0);
    }
}
