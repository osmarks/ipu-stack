//! Offline replay and controlled ordering experiments on captured transfers.
use super::*;

/// Production selection and explicit ordering alternatives for offline replay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExchangeSchedulingPriority {
    #[default]
    Automatic,
    Combined,
    Directional,
    RemainingCombined,
    RemainingDirectional,
    Streams(u32),
    BalancedStreams(u32),
}

pub fn schedule_exchange_problem(
    tile_count: u16,
    problem: &ExchangeScheduleProblem,
) -> Result<ExchangeScheduleRun, ExchangeLoweringError> {
    schedule_exchange_problem_with_priority(
        tile_count,
        problem,
        ExchangeSchedulingPriority::Automatic,
    )
}

pub fn schedule_exchange_problem_with_priority(
    tile_count: u16,
    problem: &ExchangeScheduleProblem,
    priority: ExchangeSchedulingPriority,
) -> Result<ExchangeScheduleRun, ExchangeLoweringError> {
    if tile_count == 0 || usize::from(tile_count) > Topology::c600().tile_count() {
        return Err(ExchangeLoweringError::InvalidSnapshot(format!(
            "tile count {tile_count} is outside the C600 topology"
        )));
    }
    let topology = Topology::new(
        (0..tile_count)
            .map(ipu_exchange::c600_logical_to_physical)
            .collect(),
    )?;
    let pending = pending_from_problem(tile_count, problem)?;
    let (receive_counts, incoming_bases) = receive_configuration(&pending, tile_count)?;
    let scheduling = SchedulingProblem::new(&pending, tile_count);
    let schedule = if let ExchangeSchedulingPriority::Streams(words)
    | ExchangeSchedulingPriority::BalancedStreams(words) = priority
    {
        if words == 0 {
            return Err(ExchangeLoweringError::InvalidSnapshot(
                "stream wave size must be nonzero".into(),
            ));
        }
        materialize_stream_schedule(
            &topology,
            &scheduling,
            &incoming_bases,
            &receive_counts,
            words,
            matches!(priority, ExchangeSchedulingPriority::BalancedStreams(_)),
        )?
    } else {
        materialize_greedy_schedule_with_priority(
            &topology,
            &scheduling,
            &incoming_bases,
            &receive_counts,
            priority,
        )?
    };
    let optimized = if matches!(
        priority,
        ExchangeSchedulingPriority::Streams(_) | ExchangeSchedulingPriority::BalancedStreams(_)
    ) {
        OptimizedSchedule {
            initial_horizon: schedule.horizon,
            endpoint_lower_bound: endpoint_work_lower_bound(&pending, tile_count),
            schedule,
            selected_kind: "compact-streams",
            neighborhood_improvements: 0,
        }
    } else {
        improve_pending_schedule(
            &topology,
            &scheduling,
            &incoming_bases,
            &receive_counts,
            schedule,
            "full-duplex",
        )?
    };
    finish_exchange_run(tile_count, problem.phase, incoming_bases, optimized)
}

pub(super) fn optimize_stream_schedule(
    topology: &Topology,
    problem: &SchedulingProblem<'_>,
    incoming_bases: &[u32],
    receive_counts: &[usize],
    words: u32,
) -> Result<OptimizedSchedule, ExchangeLoweringError> {
    let mut schedule = materialize_stream_schedule(
        topology,
        problem,
        incoming_bases,
        receive_counts,
        words,
        false,
    )?;
    let initial_horizon = schedule.horizon;
    let mut improvements = 0;
    // Larger address-contiguous chunks recover pointer compression lost by
    // endpoint balancing. Keep the incumbent's maximum and total encoded size
    // as hard caps; the compact mode must not trade memory for speed.
    if let Ok(candidate) = materialize_stream_schedule(
        topology,
        problem,
        incoming_bases,
        receive_counts,
        words.saturating_mul(4),
        true,
    ) && candidate.horizon < schedule.horizon
    {
        let (maximum, total) = encoded_row_storage(&schedule)?;
        let (candidate_maximum, candidate_total) = encoded_row_storage(&candidate)?;
        if candidate_maximum <= maximum && candidate_total <= total {
            schedule = candidate;
            improvements = 1;
        }
    }
    Ok(OptimizedSchedule {
        initial_horizon,
        endpoint_lower_bound: endpoint_work_lower_bound(problem.transfers, problem.tile_count),
        schedule,
        selected_kind: if improvements == 0 {
            "compact-streams"
        } else {
            "balanced-compact-streams"
        },
        neighborhood_improvements: improvements,
    })
}

pub(super) fn materialize_stream_schedule(
    topology: &Topology,
    problem: &SchedulingProblem<'_>,
    incoming_bases: &[u32],
    receive_counts: &[usize],
    words: u32,
    balanced: bool,
) -> Result<MaterializedSchedule, ExchangeLoweringError> {
    let order = order::stream_wave_order(problem, words, balanced);
    match materialize_schedule_order(
        topology,
        problem,
        incoming_bases,
        receive_counts,
        &order,
        false,
    ) {
        Ok(schedule) => Ok(schedule),
        Err(ExchangeLoweringError::Exchange(ipu_exchange::ExchangeError::Schedule(
            "SENDPICP instruction alignment",
        ))) => materialize_schedule_order(
            topology,
            problem,
            incoming_bases,
            receive_counts,
            &order,
            true,
        ),
        Err(error) => Err(error),
    }
}
