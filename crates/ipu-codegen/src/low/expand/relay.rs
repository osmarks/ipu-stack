//! Direct multicast versus gathering native panels once and forwarding them.
//! Both use the same exchange phase; scratch is visible to normal placement.
use super::*;

pub(super) fn select(
    program: &mut TileGraph,
    analysis: &mut crate::estimate::GeometryAnalysis,
) -> ExpansionResult<()> {
    for index in 0..program.exchange_phases.len() {
        if !program.exchange_phases[index]
            .transfers
            .iter()
            .any(|t| t.destinations.len() > 1)
        {
            continue;
        }
        let original = program.exchange_phases[index].clone();
        let start = program.shards.len();
        let Some(candidate) = candidate(program, &original)? else {
            continue;
        };
        let before = crate::estimate::exchange_phase_estimate(program, &original, analysis)?;
        let after = crate::estimate::exchange_phase_estimate(program, &candidate, analysis)?;
        let mut scratch = vec![0u64; usize::from(program.tile_count)];
        for shard in &program.shards[start..] {
            scratch[usize::from(shard.tile)] += u64::from(shard_storage_bytes(shard)?);
        }
        let peak = |bytes: &[u64]| bytes.iter().copied().max().unwrap_or(0);
        let with_scratch = after
            .1
            .iter()
            .zip(&scratch)
            .map(|(row, buffer)| row + buffer)
            .max()
            .unwrap_or(0);
        // Do not trade modeled speed for rows, or hide scratch behind a sum of
        // phase-wide bytes. Complete-plan liveness/placement accounts it again.
        if after.0 <= before.0 && with_scratch < peak(&before.1) {
            tracing::debug!(
                phase = index,
                relays = program.shards.len() - start,
                before_cycles = before.0,
                after_cycles = after.0,
                before_bytes = peak(&before.1),
                after_bytes = with_scratch,
                "selected packed multicast relays"
            );
            program.exchange_phases[index] = candidate;
        } else {
            program.shards.truncate(start);
        }
    }
    Ok(())
}

fn panel_views(shard: &BlockValue) -> Option<Vec<Vec<ShardExtent>>> {
    let rank = shard.extents.len();
    if rank < 2 {
        return None;
    }
    let columns = shard.extents[rank - 1];
    let width = match shard.tensor_type.format.layout.order {
        ElementOrder::RowMajor => columns.physical_end - columns.start,
        ElementOrder::Amp(AmpOrder::Left | AmpOrder::Output) => shard
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

fn candidate(
    program: &mut TileGraph,
    phase: &ExchangePhase,
) -> ExpansionResult<Option<ExchangePhase>> {
    let roots = |view: &ShardView| storage_root(&program.shards, view.shard);
    let reads: BTreeSet<_> = phase.transfers.iter().map(|t| roots(&t.source)).collect();
    if phase
        .transfers
        .iter()
        .flat_map(|t| &t.destinations)
        .any(|v| reads.contains(&roots(v)))
    {
        return Ok(None); // Preserve existing receive/forward or alias ordering.
    }
    let mut groups = BTreeMap::<Vec<BlockValueId>, Vec<usize>>::new();
    for (index, transfer) in phase.transfers.iter().enumerate() {
        if transfer.destinations.len() < 2 || transfer.order == CopyOrder::Panels {
            continue;
        }
        let source = &program.shards[transfer.source.shard.index() as usize];
        if transfer.destinations.iter().any(|view| {
            let target = &program.shards[view.shard.index() as usize];
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
        let shard = &program.shards[destinations[0].index() as usize];
        if destinations
            .iter()
            .any(|id| program.shards[id.index() as usize].extents != shard.extents)
        {
            continue;
        }
        let Some(panels) = panel_views(shard) else {
            continue;
        };
        count += panels.len();
        if count > usize::from(program.tile_count) {
            return Ok(None);
        }
        plans.push((destinations, transfers, panels));
    }
    if plans.is_empty() {
        return Ok(None);
    }
    let start = program.shards.len();
    let mut used = BTreeSet::new();
    let mut removed = BTreeSet::new();
    let mut gather = Vec::new();
    let mut forward = Vec::new();
    for (destinations, transfers, panels) in plans {
        let target = program.shards[destinations[0].index() as usize].clone();
        let excluded =
            destinations
                .iter()
                .map(|id| program.shards[id.index() as usize].tile)
                .chain(transfers.iter().map(|&i| {
                    program.shards[phase.transfers[i].source.shard.index() as usize].tile
                }))
                .collect::<BTreeSet<_>>();
        for extents in panels {
            let Some(tile) = (0..program.tile_count)
                .rev()
                .find(|t| !used.contains(t) && !excluded.contains(t))
            else {
                program.shards.truncate(start);
                return Ok(None);
            };
            let id = BlockValueId(
                program
                    .shards
                    .len()
                    .try_into()
                    .map_err(|_| ExpansionError::IdOverflow)?,
            );
            let mut relay = target.clone();
            relay.id = id;
            relay.tile = tile;
            relay.extents = extents.clone();
            relay.definition = ShardDefinition::ExchangeStaging;
            relay.tensor_type.format.layout.memory_class = MemoryClass::Ipu21Standard;
            let expected = u64::from(shard_storage_bytes(&relay)?);
            let mut covered = 0u64;
            let mut parts = Vec::new();
            for &index in &transfers {
                let original = &phase.transfers[index];
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
                program.shards.truncate(start);
                return Ok(None);
            }
            program.shards.push(relay);
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
        removed.extend(transfers);
    }
    let mut transfers = phase
        .transfers
        .iter()
        .enumerate()
        .filter(|(i, _)| !removed.contains(i))
        .map(|(_, t)| t.clone())
        .collect::<Vec<_>>();
    transfers.extend(gather);
    transfers.extend(forward);
    Ok(Some(ExchangePhase {
        transfers,
        ..phase.clone()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> TileGraph {
        let mut tensor = TensorType {
            shape: crate::TensorShape(vec![2, 32, 64]),
            format: crate::TensorFormat {
                precision: Precision::F8F143 { scale_exponent: -4 },
                layout: Layout::row_major(TensorTiling::replicated(1)),
            },
        };
        tensor.format.layout.order = ElementOrder::Amp(AmpOrder::Left);
        let extents = |row, end| {
            vec![
                ShardExtent {
                    axis: 0,
                    start: 0,
                    logical_end: 2,
                    physical_end: 2,
                },
                ShardExtent {
                    axis: 1,
                    start: row,
                    logical_end: end,
                    physical_end: end,
                },
                ShardExtent {
                    axis: 2,
                    start: 0,
                    logical_end: 64,
                    physical_end: 64,
                },
            ]
        };
        let shards = (0..48u32)
            .map(|i| BlockValue {
                id: BlockValueId(i),
                tile: if i < 32 { i as u16 } else { i as u16 + 8 },
                tensor_type: tensor.clone(),
                extents: if i < 32 {
                    extents(i, i + 1)
                } else {
                    extents(0, 32)
                },
                definition: ShardDefinition::Value(MidValueId::from_index(i)),
            })
            .collect();
        let phase = ExchangePhase {
            id: ExchangePhaseId(0),
            provenance: WorkProvenance {
                operation: None,
                value: None,
                reason: WorkReason::LayoutRearrangement,
            },
            transfers: (0..32)
                .map(|i| LogicalExchange {
                    source: ShardView {
                        shard: BlockValueId(i),
                        extents: extents(i, i + 1),
                    },
                    destinations: (32..48)
                        .map(|d| ShardView {
                            shard: BlockValueId(d),
                            extents: extents(i, i + 1),
                        })
                        .collect(),
                    order: CopyOrder::Physical,
                })
                .collect(),
        };
        TileGraph {
            tile_count: 64,
            shards,
            exchange_phases: vec![phase],
            body: BlockRegion {
                operations: vec![BlockOperation::Exchange(ExchangePhaseId(0))],
            },
            kernel_runs: vec![],
            local_copies: vec![],
            inputs: vec![],
            outputs: vec![],
            value_shards: vec![],
            logical_values: vec![],
            checkpoints: vec![],
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        }
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
        let mut program = fixture();
        let expected = execute(&program);
        select(
            &mut program,
            &mut crate::estimate::GeometryAnalysis::default(),
        )
        .unwrap();
        assert_eq!(program.shards.len(), 52);
        assert_eq!(program.exchange_phases.len(), 1);
        assert_eq!(execute(&program), expected);
        assert!(
            program.shards[48..]
                .iter()
                .all(|s| s.definition == ShardDefinition::ExchangeStaging)
        );
    }

    #[test]
    fn incomplete_overlapping_or_dependent_materializations_stay_direct() {
        for mode in 0..3 {
            let mut program = fixture();
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
            let before = program.clone();
            select(
                &mut program,
                &mut crate::estimate::GeometryAnalysis::default(),
            )
            .unwrap();
            assert_eq!(program, before);
        }
    }
}
