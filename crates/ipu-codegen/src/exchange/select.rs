use super::greedy::ExchangeSchedulingPriority;
use super::{
    ExchangeItemWidth, ExchangeLoweringError, ExchangeScheduleCache, MaterializedSchedule,
    PendingTransfer, SchedulingProblem, encoded_row_storage, endpoint_work_lower_bound,
    materialize_schedule_order,
};
use super::{greedy, matching, packet, repair, reuse, streams};
use crate::ExchangePhaseId;
use ipu_target::ipu21::fabric::Topology;
use std::collections::BTreeMap;

pub(super) fn paired_transfer_alternatives(
    pending: &[PendingTransfer],
    topology: &Topology,
    tile_count: u16,
) -> Result<Vec<Option<PendingTransfer>>, ExchangeLoweringError> {
    let mut alternatives = Vec::with_capacity(pending.len());
    for transfer in pending {
        if transfer.width != ExchangeItemWidth::Word32 || !transfer.supports_paired_width() {
            alternatives.push(None);
            continue;
        }
        let source_pair = topology.paired_logical(transfer.source)?;
        if source_pair >= tile_count {
            alternatives.push(None);
            continue;
        }

        let mut by_pair = BTreeMap::<u16, Vec<(u16, u32)>>::new();
        for &(tile, address) in &transfer.destinations {
            by_pair
                .entry(topology.physical(tile)? & !2)
                .or_default()
                .push((tile, address));
        }
        let mut paired_destinations = Vec::with_capacity(transfer.destinations.len());
        let all_destinations_pairable = by_pair.into_values().all(|destinations| {
            // Each receiver independently programs its SRAM pointer; only
            // membership in a complete receive pair is required here.
            let pairable = destinations.len() == 2
                && topology
                    .paired_logical(destinations[0].0)
                    .is_ok_and(|paired| paired == destinations[1].0);
            if pairable {
                paired_destinations.extend(destinations);
            }
            pairable
        });
        if !all_destinations_pairable || paired_destinations.is_empty() {
            alternatives.push(None);
            continue;
        }

        let paired_tiles = paired_destinations
            .iter()
            .map(|&(tile, _)| tile)
            .collect::<Vec<_>>();
        if crate::exchange::paired_multicast(
            &topology,
            transfer.source,
            &paired_tiles,
            transfer.words / 2,
        )
        .is_err()
        {
            alternatives.push(None);
            continue;
        }
        let mut paired = transfer.clone();
        paired.destinations = paired_destinations;
        paired.width = ExchangeItemWidth::Paired64;
        paired.reserved_source = Some(source_pair);
        alternatives.push(Some(paired));
    }
    Ok(alternatives)
}

pub(super) fn receive_configuration(
    pending: &[PendingTransfer],
    tile_count: u16,
) -> Result<(Vec<usize>, Vec<u32>), ExchangeLoweringError> {
    let mut receive_counts = vec![0usize; usize::from(tile_count)];
    for transfer in pending {
        for &(tile, _) in &transfer.destinations {
            let count = receive_counts
                .get_mut(usize::from(tile))
                .ok_or(ExchangeLoweringError::InvalidDestination)?;
            *count += 1;
        }
    }
    let mut incoming_bases = vec![None::<u32>; usize::from(tile_count)];
    for transfer in pending {
        if let [(tile, address)] = transfer.destinations.as_slice()
            && receive_counts[usize::from(*tile)] == 1
        {
            incoming_bases[usize::from(*tile)] = Some(*address);
        }
    }
    Ok((
        receive_counts,
        incoming_bases
            .into_iter()
            .map(|base| base.unwrap_or(0))
            .collect(),
    ))
}

pub(super) struct OptimizedSchedule {
    pub(super) schedule: MaterializedSchedule,
    pub(super) initial_horizon: u32,
    pub(super) endpoint_lower_bound: u32,
    pub(super) selected_kind: &'static str,
    pub(super) neighborhood_improvements: usize,
}

pub(super) struct ScheduledPending {
    pub(super) pending: Vec<PendingTransfer>,
    pub(super) incoming_bases: Vec<u32>,
    pub(super) optimized: OptimizedSchedule,
}

pub(super) fn optimize_owned_pending(
    topology: &Topology,
    pending: Vec<PendingTransfer>,
    tile_count: u16,
    stream_words: Option<std::num::NonZeroU32>,
) -> Result<ScheduledPending, ExchangeLoweringError> {
    let (receive_counts, incoming_bases) = receive_configuration(&pending, tile_count)?;
    let optimized = optimize_pending_schedule(
        topology,
        &pending,
        &incoming_bases,
        &receive_counts,
        tile_count,
        stream_words,
    )?;
    Ok(ScheduledPending {
        pending,
        incoming_bases,
        optimized,
    })
}

/// Select under the caller's scheduling policy, replaying only compatible work.
/// Policy is an input to compilation; the cache merely records prior choices.
pub(super) fn select_phase(
    phase: ExchangePhaseId,
    topology: &Topology,
    pending: Vec<PendingTransfer>,
    tile_count: u16,
    stream_words: Option<std::num::NonZeroU32>,
    cache: &mut ExchangeScheduleCache,
) -> Result<ScheduledPending, ExchangeLoweringError> {
    let pending = packet::split_self_receive_conflicts(topology, pending)?;
    let structure = reuse::structure_fingerprint(&pending, tile_count);
    if let Some(recipe) = cache.phases.get(&phase)
        && recipe.structure == structure
        && recipe.stream_words == stream_words
    {
        match recipe.replay(topology, &pending, tile_count) {
            Ok(Some(schedule)) => {
                tracing::info!(
                    phase = phase.index(),
                    "reused exchange optimization after validating relocated rows"
                );
                return Ok(schedule);
            }
            Ok(None) => {}
            Err(error) => {
                tracing::debug!(phase = phase.index(), %error, "exchange choices require reoptimization after placement")
            }
        }
    }
    let selected =
        select_transfer_widths(phase.index(), topology, pending, tile_count, stream_words)?;
    cache.phases.insert(
        phase,
        std::sync::Arc::new(reuse::ScheduleRecipe {
            stream_words,
            structure,
            widths: selected
                .pending
                .iter()
                .map(|transfer| transfer.width)
                .collect(),
            order: selected.optimized.schedule.order.clone(),
            rows: reuse::normalized_rows(&selected.optimized.schedule)?,
        }),
    );
    Ok(selected)
}

/// Compare complete width choices. A single width change can leave another
/// path tied at the horizon, so individually profitable transfers are not a
/// useful prerequisite for pairing. This bounds search to two optimizations.
pub(super) fn select_transfer_widths(
    phase: u32,
    topology: &Topology,
    pending: Vec<PendingTransfer>,
    tile_count: u16,
    stream_words: Option<std::num::NonZeroU32>,
) -> Result<ScheduledPending, ExchangeLoweringError> {
    let alternatives = paired_transfer_alternatives(&pending, topology, tile_count)?;
    let candidates = alternatives.iter().flatten().count();
    if candidates == 0 {
        return optimize_owned_pending(topology, pending, tile_count, stream_words);
    }
    let paired = pending
        .iter()
        .zip(alternatives)
        .map(|(ordinary, paired)| paired.unwrap_or_else(|| ordinary.clone()))
        .collect();
    let ordinary = optimize_owned_pending(topology, pending, tile_count, stream_words);
    let paired = optimize_owned_pending(topology, paired, tile_count, stream_words);
    let (ordinary, paired) = match (ordinary, paired) {
        (Ok(ordinary), Ok(paired)) => (ordinary, paired),
        (ordinary, Err(error)) => {
            tracing::debug!(phase, %error, "paired exchange candidate is not encodable");
            return ordinary;
        }
        (Err(error), Ok(paired)) => {
            tracing::debug!(phase, %error, "ordinary exchange candidate is not encodable");
            return Ok(paired);
        }
    };
    let ordinary_horizon = ordinary.optimized.schedule.horizon;
    let paired_horizon = paired.optimized.schedule.horizon;
    let use_paired = if stream_words.is_some() {
        let storage_score = |candidate: &ScheduledPending| -> Result<_, ExchangeLoweringError> {
            let (maximum, total) = encoded_row_storage(&candidate.optimized.schedule)?;
            Ok((maximum, total, candidate.optimized.schedule.horizon))
        };
        storage_score(&paired)? < storage_score(&ordinary)?
    } else {
        paired_horizon < ordinary_horizon
    };
    tracing::info!(
        phase,
        candidates,
        ordinary_horizon,
        paired_horizon,
        use_paired,
        "compared ordinary and paired exchange schedules"
    );
    Ok(if use_paired { paired } else { ordinary })
}

pub(super) fn optimize_pending_schedule(
    topology: &Topology,
    pending: &[PendingTransfer],
    incoming_bases: &[u32],
    receive_counts: &[usize],
    tile_count: u16,
    stream_words: Option<std::num::NonZeroU32>,
) -> Result<OptimizedSchedule, ExchangeLoweringError> {
    let problem = SchedulingProblem::new(pending, tile_count);
    if let Some(words) = stream_words {
        let schedule = streams::schedule(
            topology,
            &problem,
            incoming_bases,
            receive_counts,
            words.get(),
            true,
        )?;
        return Ok(OptimizedSchedule {
            initial_horizon: schedule.horizon,
            endpoint_lower_bound: endpoint_work_lower_bound(pending, tile_count),
            schedule,
            selected_kind: "balanced-compact-streams",
            neighborhood_improvements: 0,
        });
    }
    let schedule = greedy::schedule(
        topology,
        &problem,
        incoming_bases,
        receive_counts,
        ExchangeSchedulingPriority::Automatic,
    )?;
    improve_pending_schedule(
        topology,
        &problem,
        incoming_bases,
        receive_counts,
        schedule,
        "full-duplex",
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn improve_pending_schedule(
    topology: &Topology,
    problem: &SchedulingProblem<'_>,
    incoming_bases: &[u32],
    receive_counts: &[usize],
    mut schedule: MaterializedSchedule,
    initial_kind: &'static str,
) -> Result<OptimizedSchedule, ExchangeLoweringError> {
    let pending = problem.transfers;
    let tile_count = problem.tile_count;
    let initial_horizon = schedule.horizon;
    let endpoint_lower_bound = endpoint_work_lower_bound(pending, tile_count);
    let mut selected_kind = initial_kind;
    let mut neighborhood_improvements = 0usize;
    if let Some(order) = matching::order(problem, &schedule.order) {
        let matching = materialize_schedule_order(
            topology,
            problem,
            incoming_bases,
            receive_counts,
            &order,
            false,
        );
        if let Ok(matching) = matching
            && matching.horizon < schedule.horizon
        {
            schedule = matching;
            selected_kind = "matching-waves";
        }
    }
    loop {
        let repaired_order = repair::order(problem, &schedule, false).unwrap_or_else(|| {
            repair::order(problem, &schedule, true).expect("local repair has no work limit")
        });
        if repaired_order == schedule.order {
            break;
        }
        let repaired = materialize_schedule_order(
            topology,
            problem,
            incoming_bases,
            receive_counts,
            &repaired_order,
            false,
        );
        let Ok(repaired) = repaired else {
            break;
        };
        if repaired.horizon >= schedule.horizon {
            break;
        }
        schedule = repaired;
        selected_kind = "critical-neighborhood";
        neighborhood_improvements += 1;
    }
    Ok(OptimizedSchedule {
        schedule,
        initial_horizon,
        endpoint_lower_bound,
        selected_kind,
        neighborhood_improvements,
    })
}
