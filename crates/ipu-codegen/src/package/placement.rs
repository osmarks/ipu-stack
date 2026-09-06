//! Bounded SRAM placement search after package support storage is reserved.

use super::*;

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
