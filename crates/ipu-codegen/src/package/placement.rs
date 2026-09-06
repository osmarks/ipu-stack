//! Bounded SRAM placement search after package support storage is reserved.

use super::*;

/// Apply ownership-preserving placement before projecting per-tile work.
pub(super) fn map_tiles(
    graph: &mut std::sync::Arc<crate::TileGraph>,
    mapping: Option<&[u16]>,
) -> PackageBuildResult<()> {
    let Some(mapping) = mapping else {
        return Ok(());
    };
    let mut sorted = mapping.to_vec();
    sorted.sort_unstable();
    if sorted != (0..graph.tile_count).collect::<Vec<_>>() {
        return Err(invalid(
            "tile mapping must be a bijection over active tiles",
        ));
    }
    fn map_region(body: &mut crate::BlockRegion, mapping: &[u16]) {
        for operation in &mut body.operations {
            match operation {
                crate::BlockOperation::Copy { tile, .. }
                | crate::BlockOperation::Compute { tile, .. } => {
                    *tile = mapping[usize::from(*tile)]
                }
                crate::BlockOperation::Repeat(repeat) => {
                    for binding in &mut repeat.bindings {
                        binding.tile = mapping[usize::from(binding.tile)];
                    }
                    map_region(&mut repeat.body, mapping);
                }
                _ => {}
            }
        }
    }
    let graph = std::sync::Arc::make_mut(graph);
    for shard in &mut graph.shards {
        shard.tile = mapping[usize::from(shard.tile)];
    }
    map_region(&mut graph.body, mapping);
    Ok(())
}

pub(super) fn improve_exchange_placement(
    program: &LowProgram,
    standard_ranges: &[(u32, u32)],
    topology: &Topology,
    baseline: crate::Placement,
    exchanges: crate::exchange::LoweredExchanges,
    row_capacity: u32,
    cache: &crate::exchange::ExchangeScheduleCache,
) -> PackageBuildResult<(crate::Placement, crate::exchange::LoweredExchanges)> {
    if program.exchange_phases.is_empty() {
        return Ok((baseline, exchanges));
    }
    let conflicts = crate::place::ExchangeConflicts::new(program)?;
    let baseline_score = conflicts.score(&baseline);
    if baseline_score == 0 {
        return Ok((baseline, exchanges));
    }
    let mut candidates = Vec::new();
    for offset in (4096..ipu_package::IPU21_INTERLEAVED_ELEMENT_SIZE).step_by(4096) {
        let candidate = match crate::place::place_with_offset(program, standard_ranges, offset) {
            Ok(candidate) => candidate,
            Err(crate::PlacementError::OutOfMemory { .. }) => continue,
            Err(error) => return Err(error.into()),
        };
        let score = conflicts.score(&candidate);
        tracing::debug!(offset, score = %score, "scored exchange placement");
        if score < baseline_score {
            candidates.push((score, offset, candidate));
        }
    }
    candidates.sort_by_key(|(score, offset, _)| (*score, *offset));
    // Scheduling remains expensive. Inspect only the best cheap candidate;
    // keep the existing placement unless complete schedules improve.
    let Some((score, offset, candidate)) = candidates.into_iter().next() else {
        return Ok((baseline, exchanges));
    };
    let mut cache = cache.clone();
    let candidate_exchanges = match crate::exchange::lower_exchanges_cached(
        program,
        &candidate,
        topology,
        crate::ExchangeLoweringOptions::default(),
        &mut cache,
    ) {
        Ok(exchanges) => exchanges,
        Err(error) => {
            tracing::info!(offset, %error, "rejected unschedulable exchange placement");
            return Ok((baseline, exchanges));
        }
    };
    let mut multiplicities = vec![0u64; program.exchange_phases.len()];
    for tile in &program.tiles {
        let mut counts = vec![0u64; multiplicities.len()];
        count_exchanges(program, tile, 1, &mut counts);
        for (maximum, count) in multiplicities.iter_mut().zip(counts) {
            *maximum = (*maximum).max(count);
        }
    }
    let cycles = |lowered: &crate::exchange::LoweredExchanges| -> u64 {
        lowered
            .phases
            .iter()
            .map(|phase| {
                u64::from(phase.event_cycles)
                    .saturating_mul(multiplicities[phase.id.index() as usize])
            })
            .fold(0, u64::saturating_add)
    };
    let baseline_cycles = cycles(&exchanges);
    let candidate_cycles = cycles(&candidate_exchanges);
    let row_bytes = crate::tile::compact_exchange_table_bytes(
        &candidate_exchanges.phases,
        u16::try_from(Topology::c600().tile_count())?,
        program.tile_count,
    )?;
    let accepted = candidate_cycles < baseline_cycles && row_bytes <= row_capacity;
    tracing::info!(offset, baseline_score = %baseline_score, score = %score,
        baseline_cycles, candidate_cycles, row_bytes, row_capacity, accepted,
        "evaluated exchange placement finalist");
    if accepted {
        Ok((candidate, candidate_exchanges))
    } else {
        Ok((baseline, exchanges))
    }
}

fn count_exchanges(
    program: &LowProgram,
    tile: &crate::low::TileWorkList,
    repetitions: u64,
    counts: &mut [u64],
) {
    for work in program.work(tile) {
        match work {
            crate::low::TileWorkRef::Exchange(id) => {
                counts[id.index() as usize] =
                    counts[id.index() as usize].saturating_add(repetitions);
            }
            crate::low::TileWorkRef::Repeat(repeat) => {
                count_exchanges(
                    program,
                    &repeat.body,
                    repetitions.saturating_mul(u64::from(repeat.count)),
                    counts,
                );
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_mapping_preserves_shards_and_projected_work() {
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [32, 16]).unwrap();
        let output = graph.gelu(input).unwrap();
        graph.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(4).with_automatic_input(input, Precision::F16);
        let mid = crate::lower(&graph, &config, &Ipu21CostModel).unwrap();
        let mut expanded = crate::expand_tiles(&mid).unwrap();
        let original = expanded.clone();
        let before = lower_to_tiles(&original, false);
        let mapping = [2, 0, 3, 1];
        map_tiles(&mut expanded, Some(&mapping)).unwrap();
        let after = lower_to_tiles(&expanded, false);
        for (old, new) in original.shards.iter().zip(&expanded.shards) {
            let mut expected = old.clone();
            expected.tile = mapping[usize::from(old.tile)];
            assert_eq!(*new, expected);
        }
        for tile in &before.tiles {
            assert_eq!(
                tile.work,
                after.tiles[usize::from(mapping[usize::from(tile.tile)])].work
            );
        }
        assert_eq!(original.exchange_phases, expanded.exchange_phases);
        let valid = expanded.clone();
        assert!(map_tiles(&mut expanded, Some(&[0, 0, 2, 3])).is_err());
        assert_eq!(expanded, valid);
        assert!(map_tiles(&mut expanded, Some(&[0, 1])).is_err());
    }
}
