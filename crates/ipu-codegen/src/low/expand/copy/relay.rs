//! Realize an explicitly requested gather-and-multicast copy.
//! Relays stay in the same exchange phase; scratch uses normal placement.
use crate::low::storage::storage_root;
use crate::low::*;
use crate::{AmpOrder, CopyOrder, ElementOrder, MemoryClass, ShardExtent};
#[cfg(test)]
use ipu_target::Target;
use std::collections::{BTreeMap, BTreeSet};

fn panel_views(shard: &BlockValue) -> Option<Vec<Vec<ShardExtent>>> {
    let rank = shard.extents.len();
    if rank < 2 {
        return None;
    }
    let columns = shard.extents[rank - 1];
    let width = match shard.tensor_type.format.layout.order {
        ElementOrder::RowMajor => columns.physical_end - columns.start,
        ElementOrder::Amp(AmpOrder::Left) => shard
            .tensor_type
            .format
            .layout
            .order
            .retained_linear_column_grain(shard.tensor_type.format.precision)?,
        _ => return None,
    };
    if width == 0
        || !columns.start.is_multiple_of(width)
        || !columns.logical_end.is_multiple_of(width)
    {
        return None;
    }
    let mut result = vec![shard.extents.clone()];
    for axis in (0..rank - 2).chain(std::iter::once(rank - 1)) {
        let step = if axis == rank - 1 { width } else { 1 };
        result = result
            .into_iter()
            .flat_map(|view| {
                (view[axis].start..view[axis].logical_end)
                    .step_by(step as usize)
                    .map(move |start| {
                        let mut part = view.clone();
                        part[axis].start = start;
                        part[axis].logical_end = start + step;
                        part[axis].physical_end = start + step;
                        part
                    })
            })
            .collect();
    }
    for view in &mut result {
        view[rank - 2].physical_end = view[rank - 2].logical_end;
    }
    Some(result)
}

fn intersection(a: &[ShardExtent], b: &[ShardExtent]) -> Option<Vec<ShardExtent>> {
    let mut result = a.to_vec();
    for ((out, a), b) in result.iter_mut().zip(a).zip(b) {
        out.start = a.start.max(b.start);
        out.physical_end = a.physical_end.min(b.physical_end);
        out.logical_end = out.physical_end;
        if out.start >= out.physical_end {
            return None;
        }
    }
    Some(result)
}

pub(super) fn gather(
    shards: &mut Vec<BlockValue>,
    tile_count: u16,
    transfers: &[LogicalExchange],
) -> ExpansionResult<Vec<LogicalExchange>> {
    let roots = |view: &ShardView| storage_root(shards, view.shard);
    let reads: BTreeSet<_> = transfers.iter().map(|t| roots(&t.source)).collect();
    if transfers
        .iter()
        .flat_map(|t| &t.destinations)
        .any(|v| reads.contains(&roots(v)))
    {
        return Err(ExpansionError::InvalidCopyPlan); // Preserve existing receive/forward or alias ordering.
    }
    let mut groups = BTreeMap::<Vec<BlockValueId>, Vec<usize>>::new();
    for (index, transfer) in transfers.iter().enumerate() {
        if transfer.destinations.len() < 2 || transfer.order == CopyOrder::Panels {
            continue;
        }
        let source = &shards[transfer.source.shard.index() as usize];
        if transfer.destinations.iter().any(|view| {
            let target = &shards[view.shard.index() as usize];
            source.tensor_type.format.precision != target.tensor_type.format.precision
                || source.tensor_type.format.layout.order != target.tensor_type.format.layout.order
                || view.extents != transfer.source.extents
                || view.extents.iter().any(|e| e.logical_end != e.physical_end)
        }) {
            continue;
        }
        let mut destinations = transfer
            .destinations
            .iter()
            .map(|v| v.shard)
            .collect::<Vec<_>>();
        destinations.sort_unstable();
        groups.entry(destinations).or_default().push(index);
    }
    let mut plans = Vec::new();
    let mut count = 0usize;
    for (destinations, transfers) in groups {
        let shard = &shards[destinations[0].index() as usize];
        if destinations
            .iter()
            .any(|id| shards[id.index() as usize].extents != shard.extents)
        {
            continue;
        }
        let Some(panels) = panel_views(shard) else {
            continue;
        };
        count += panels.len();
        if count > usize::from(tile_count) {
            return Err(ExpansionError::InvalidCopyPlan);
        }
        plans.push((destinations, transfers, panels));
    }
    if plans.is_empty() {
        return Err(ExpansionError::InvalidCopyPlan);
    }
    let mut used = BTreeSet::new();
    let mut removed = BTreeSet::new();
    let mut gather = Vec::new();
    let mut forward = Vec::new();
    for (destinations, indices, panels) in plans {
        let mut tensor_type = shards[destinations[0].index() as usize].tensor_type.clone();
        tensor_type.format.layout.memory_class = MemoryClass::Ipu21Standard;
        let excluded = destinations
            .iter()
            .map(|id| shards[id.index() as usize].tile)
            .chain(
                indices
                    .iter()
                    .map(|&i| shards[transfers[i].source.shard.index() as usize].tile),
            )
            .collect::<BTreeSet<_>>();
        for extents in panels {
            let Some(tile) = (0..tile_count)
                .rev()
                .find(|t| !used.contains(t) && !excluded.contains(t))
            else {
                return Err(ExpansionError::InvalidCopyPlan);
            };
            let id = BlockValueId(
                shards
                    .len()
                    .try_into()
                    .map_err(|_| ExpansionError::IdOverflow)?,
            );
            let relay = BlockValue {
                id,
                tile,
                tensor_type: tensor_type.clone(),
                extents: extents.clone(),
                definition: ShardDefinition::Staging,
            };
            let expected = u64::from(shard_storage_bytes(&relay)?);
            let mut covered = 0u64;
            let mut parts = Vec::new();
            for &index in &indices {
                let original = &transfers[index];
                if let Some(part) = intersection(&original.source.extents, &extents) {
                    let bytes = part
                        .iter()
                        .fold(relay.tensor_type.format.precision.bytes(), |n, e| {
                            n * u64::from(e.physical_end - e.start)
                        });
                    covered += bytes;
                    parts.push(LogicalExchange {
                        source: ShardView {
                            shard: original.source.shard,
                            extents: part.clone(),
                        },
                        destinations: vec![ShardView {
                            shard: id,
                            extents: part,
                        }],
                        order: original.order,
                    });
                }
            }
            // Incomplete panels (including padding holes) stay direct. Never
            // introduce a zeroing kernel to make this transport option eligible.
            let row = extents.len() - 2;
            let mut spans = parts.iter().map(|p| &p.source.extents).collect::<Vec<_>>();
            spans.sort_unstable_by_key(|v| v[row].start);
            let mut end = extents[row].start;
            let complete = spans.iter().all(|v| {
                let adjacent = v[row].start == end && v[row + 1] == extents[row + 1];
                end = v[row].physical_end;
                adjacent
            }) && end == extents[row].physical_end;
            if covered != expected || !complete {
                return Err(ExpansionError::InvalidCopyPlan);
            }
            shards.push(relay);
            used.insert(tile);
            gather.extend(parts);
            forward.push(LogicalExchange {
                source: ShardView {
                    shard: id,
                    extents: extents.clone(),
                },
                destinations: destinations
                    .iter()
                    .map(|&shard| ShardView {
                        shard,
                        extents: extents.clone(),
                    })
                    .collect(),
                order: CopyOrder::Physical,
            });
        }
        removed.extend(indices);
    }
    let mut transfers = transfers
        .iter()
        .enumerate()
        .filter(|(i, _)| !removed.contains(i))
        .map(|(_, t)| t.clone())
        .collect::<Vec<_>>();
    transfers.extend(gather);
    transfers.extend(forward);
    Ok(transfers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Layout, MidValueId, Precision, TensorTiling, TensorType};

    fn fixture(policy: crate::CopyPolicy) -> TileGraph {
        let mut source = Layout::amp_left(32, 64);
        source.tiling = TensorTiling {
            tile_count: 32,
            replicas: 1,
            axes: vec![crate::AxisTiling::new(
                crate::TensorAxis::FromStart(1),
                32,
                1,
                crate::Padding::Reject,
            )],
        };
        let mut destination = source.clone();
        destination.tiling = TensorTiling::replicated(16);
        let values = [source, destination]
            .into_iter()
            .enumerate()
            .map(|(index, layout)| {
                let id = MidValueId::from_index(index as u32);
                crate::MidValue {
                    id,
                    tensor_type: TensorType::new(
                        [2, 32, 64],
                        Precision::F8F143 { scale_exponent: -4 },
                        layout,
                    ),
                    owners: crate::tensor::OwnerMap::rotated(if index == 0 { 0 } else { 40 }),
                    origin: crate::ValueId::from_index(0),
                    storage_group: id,
                }
            })
            .collect();
        let program = crate::MidGraph {
            tile_count: 64,
            values,
            inputs: vec![crate::MidInput {
                name: "input".into(),
                kind: crate::GraphInputKind::Host,
                value: MidValueId::from_index(0),
            }],
            outputs: vec![MidValueId::from_index(1)],
            operations: vec![crate::MidOperation {
                source: None,
                inputs: vec![MidValueId::from_index(0)],
                results: vec![MidValueId::from_index(1)],
                kind: crate::mid::MidOperationKind::Copy {
                    mapping: crate::CoordinateMapping::default(),
                    packing: crate::PackingPolicy::Direct,
                    policy,
                },
                operands: vec![],
                output_aliases: vec![],
                output_windows: vec![],
            }],
            ..Default::default()
        };
        (*crate::low::expand::expand_tiles(Target::Ipu21, &program, false).unwrap()).clone()
    }

    fn execute(program: &TileGraph) -> BTreeMap<(BlockValueId, u32), (BlockValueId, u32)> {
        let mut memory = BTreeMap::new();
        for transfer in &program.exchange_phases[0].transfers {
            let bytes = |view: &ShardView| {
                view_byte_traversal(
                    &program.shards[view.shard.index() as usize],
                    view,
                    transfer.span_order(&program.shards),
                )
                .unwrap()
                .spans()
                .flat_map(|span| (span.offset..span.offset + span.bytes).collect::<Vec<_>>())
                .collect::<Vec<_>>()
            };
            let source = bytes(&transfer.source);
            for destination in &transfer.destinations {
                let target = bytes(destination);
                assert_eq!(source.len(), target.len());
                for (&a, b) in source.iter().zip(target) {
                    let key = (transfer.source.shard, a);
                    let token = memory.get(&key).copied().unwrap_or(key);
                    memory.insert((destination.shard, b), token);
                }
            }
        }
        memory.retain(|(id, _), _| (32..48).contains(&id.index()));
        memory
    }

    #[test]
    fn relays_preserve_native_bytes_and_enter_normal_costing() {
        let direct = fixture(crate::CopyPolicy::DirectRetile);
        let expected = execute(&direct);
        let program = fixture(crate::CopyPolicy::GatherThenMulticast);
        assert_eq!(direct.shards.len(), 48);
        assert_eq!(program.shards.len(), 52);
        assert_eq!(program.exchange_phases.len(), 1);
        assert_eq!(execute(&program), expected);
        assert!(
            program.shards[48..]
                .iter()
                .all(|s| s.definition == ShardDefinition::Staging)
        );
    }

    #[test]
    fn incomplete_overlapping_or_dependent_materializations_are_rejected() {
        for mode in 0..3 {
            let mut program = fixture(crate::CopyPolicy::DirectRetile);
            match mode {
                0 => {
                    program.exchange_phases[0].transfers.pop();
                }
                1 => {
                    program.exchange_phases[0].transfers[0] =
                        program.exchange_phases[0].transfers[1].clone();
                }
                _ => {
                    program.shards[32].definition = ShardDefinition::Alias(BlockValueId(0));
                }
            }
            assert!(
                gather(
                    &mut program.shards,
                    program.tile_count,
                    &program.exchange_phases[0].transfers,
                )
                .is_err()
            );
        }
    }
}
