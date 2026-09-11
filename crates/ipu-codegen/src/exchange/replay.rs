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

pub(super) fn balanced_stream_schedule(
    topology: &Topology,
    problem: &SchedulingProblem<'_>,
    incoming_bases: &[u32],
    receive_counts: &[usize],
    words: u32,
) -> Result<OptimizedSchedule, ExchangeLoweringError> {
    let schedule = materialize_stream_schedule(
        topology,
        problem,
        incoming_bases,
        receive_counts,
        words,
        true,
    )?;
    // Choose one compact policy. Complete-package placement accounts for the
    // per-tile rows, including sharing; aggregate bytes are not a fit test.
    Ok(OptimizedSchedule {
        initial_horizon: schedule.horizon,
        endpoint_lower_bound: endpoint_work_lower_bound(problem.transfers, problem.tile_count),
        schedule,
        selected_kind: "balanced-compact-streams",
        neighborhood_improvements: 0,
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
    materialize_stream_order(topology, problem, incoming_bases, receive_counts, &order)
}

fn materialize_stream_order(
    topology: &Topology,
    problem: &SchedulingProblem<'_>,
    incoming_bases: &[u32],
    receive_counts: &[usize],
    order: &[usize],
) -> Result<MaterializedSchedule, ExchangeLoweringError> {
    match materialize_schedule_order(
        topology,
        problem,
        incoming_bases,
        receive_counts,
        order,
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
            order,
            true,
        ),
        Err(error) => Err(error),
    }
}
