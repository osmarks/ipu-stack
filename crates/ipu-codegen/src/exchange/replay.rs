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

/// Select ordinary/paired transfers through the production path, retaining
/// the recipe for subsequent placement or benchmark replay.
/// `stream_words` chooses compact balanced waves; `None` selects latency search.
pub fn select_exchange_schedule(
    tile_count: u16,
    problem: &ExchangeScheduleProblem,
    stream_words: Option<std::num::NonZeroU32>,
    cache: &mut ExchangeScheduleCache,
) -> Result<(ExchangeScheduleProblem, ExchangeScheduleRun), ExchangeLoweringError> {
    validate_snapshot_tile_count(tile_count)?;
    let topology = Topology::new(
        (0..tile_count)
            .map(ipu_target::c600::logical_to_physical)
            .collect(),
    )?;
    let pending = pending_from_problem(tile_count, problem)?;
    if pending
        .iter()
        .any(|transfer| transfer.width != ExchangeItemWidth::Word32)
    {
        return Err(ExchangeLoweringError::InvalidSnapshot(
            "width selection requires an ordinary-transfer capture".into(),
        ));
    }
    let selected = select_phase(
        ExchangePhaseId::from_index(problem.phase),
        &topology,
        pending,
        tile_count,
        stream_words,
        cache,
    )?;
    let problem = schedule_problem(problem.phase, &selected.pending);
    let run = finish_exchange_run(problem.phase, selected.incoming_bases, selected.optimized)?;
    Ok((problem, run))
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
    validate_snapshot_tile_count(tile_count)?;
    let topology = Topology::new(
        (0..tile_count)
            .map(ipu_target::c600::logical_to_physical)
            .collect(),
    )?;
    let pending = packet::split_self_receive_conflicts(
        &topology,
        pending_from_problem(tile_count, problem)?,
    )?;
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
        streams::schedule(
            &topology,
            &scheduling,
            &incoming_bases,
            &receive_counts,
            words,
            matches!(priority, ExchangeSchedulingPriority::BalancedStreams(_)),
        )?
    } else {
        greedy::schedule(
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
    finish_exchange_run(problem.phase, incoming_bases, optimized)
}
