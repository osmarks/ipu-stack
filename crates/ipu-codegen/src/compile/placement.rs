//! Bounded SRAM placement search after package support storage is reserved.

#[cfg(test)]
use crate::estimate::Ipu21CostModel;
use crate::low::LowProgram;
use crate::package::{PackageBuildResult, invalid};
#[cfg(test)]
use crate::{ComputeGraph, PipelineConfig, Precision, lower_to_tiles};
use std::collections::BTreeSet;

/// Screen geometry-derived embeddings by resource load, retaining one challenger.
pub(super) fn model_mapping(
    program: &LowProgram,
    baseline: &crate::Placement,
    search: bool,
) -> PackageBuildResult<(u64, Option<Vec<u16>>)> {
    let traffic = crate::exchange::MappingTraffic::new(program, baseline)?;
    let multiplicities = exchange_multiplicities(program);
    let identity = (0..program.tile_count).collect::<Vec<_>>();
    let baseline_score = traffic.score(&identity, &multiplicities);
    let mut blocks = BTreeSet::from([program.tile_count]);
    for shard in &program.shards {
        for axis in &shard.tensor_type.format.layout.tiling.axes {
            if let Some(stride) = axis.tile_stride {
                for block in [stride, stride.saturating_mul(axis.partitions)] {
                    if block > 1 && block <= program.tile_count {
                        blocks.insert(block);
                    }
                }
            }
        }
    }
    let mut best = None;
    let mut best_score = baseline_score;
    let mut candidates = 0;
    for block in blocks.into_iter().filter(|_| search) {
        for width in 2..block {
            if !block.is_multiple_of(width) {
                continue;
            }
            let mapping = (0..program.tile_count)
                .map(|tile| {
                    let base = tile / block * block;
                    let local = tile % block;
                    if u32::from(base) + u32::from(block) <= u32::from(program.tile_count) {
                        base + local % width * (block / width) + local / width
                    } else {
                        tile
                    }
                })
                .collect::<Vec<_>>();
            let score = traffic.score(&mapping, &multiplicities);
            candidates += 1;
            tracing::debug!(block, width, cycles=score.0, pressure=%score.1, "modelled tile mapping");
            if score.0 < baseline_score.0 && score < best_score {
                best_score = score;
                best = Some(mapping);
            }
        }
    }
    tracing::info!(candidates, baseline_cycles=baseline_score.0, candidate_cycles=best_score.0,
        baseline_pressure=%baseline_score.1, candidate_pressure=%best_score.1,
        "screened tile mappings by exchange resource load");
    let selected = best.as_deref().unwrap_or(&identity);
    let cycles =
        crate::estimate::program_cycles(&program.program, Some(&traffic.phase_cycles(selected)))?
            .total;
    Ok((cycles, best))
}

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
    if !sorted.into_iter().eq(0..graph.tile_count) {
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

pub(super) struct AddressProposal {
    pub placement: crate::Placement,
    pub baseline_score: u128,
    pub score: u128,
    pub offset: u32,
}

pub(super) fn propose_exchange_placement(
    program: &LowProgram,
    available_ranges: &[(u32, u32)],
    auxiliary: &[Vec<crate::place::AuxiliaryRequest>],
    baseline: &crate::Placement,
) -> PackageBuildResult<Option<AddressProposal>> {
    if program.exchange_phases.is_empty() {
        return Ok(None);
    }
    let conflicts = crate::place::ExchangeConflicts::new(program)?;
    let baseline_score = conflicts.score(baseline);
    if baseline_score == 0 {
        return Ok(None);
    }
    let mut best: Option<(u128, u32, crate::Placement)> = None;
    for offset in (4096..ipu_package::IPU21_INTERLEAVED_ELEMENT_SIZE).step_by(4096) {
        let candidate = match crate::place::place_with_auxiliary(
            program,
            available_ranges,
            offset,
            auxiliary,
        ) {
            Ok(candidate) => candidate,
            Err(crate::PlacementError::OutOfMemory { .. }) => continue,
            Err(error) => return Err(error.into()),
        };
        let score = conflicts.score(&candidate);
        tracing::debug!(offset, score = %score, "scored exchange placement");
        if score < baseline_score
            && best.as_ref().is_none_or(|(best_score, best_offset, _)| {
                (score, offset) < (*best_score, *best_offset)
            })
        {
            best = Some((score, offset, candidate));
        }
    }
    Ok(best.map(|(score, offset, placement)| AddressProposal {
        placement,
        score,
        offset,
        baseline_score,
    }))
}

pub(super) fn exchange_cycles(
    program: &LowProgram,
    phases: &[crate::PhysicalExchangePhase],
) -> u64 {
    let multiplicities = exchange_multiplicities(program);
    phases
        .iter()
        .map(|phase| {
            u64::from(phase.event_cycles).saturating_mul(multiplicities[phase.id.index() as usize])
        })
        .fold(0, u64::saturating_add)
}

fn exchange_multiplicities(program: &LowProgram) -> Vec<u64> {
    let mut multiplicities = vec![0u64; program.exchange_phases.len()];
    for tile in &program.tiles {
        let mut counts = vec![0u64; multiplicities.len()];
        count_exchanges(program, tile, 1, &mut counts);
        for (maximum, count) in multiplicities.iter_mut().zip(counts) {
            *maximum = (*maximum).max(count);
        }
    }
    multiplicities
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
        let mid = crate::planner::test_support::lower(&graph, &config, &Ipu21CostModel).unwrap();
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
