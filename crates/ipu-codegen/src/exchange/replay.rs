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
    let schedule = if let ExchangeSchedulingPriority::Streams(words) = priority {
        if words == 0 {
            return Err(ExchangeLoweringError::InvalidSnapshot(
                "stream wave size must be nonzero".into(),
            ));
        }
        let order = order::stream_wave_order(&scheduling, words);
        match materialize_schedule_order(
            &topology,
            &scheduling,
            &incoming_bases,
            &receive_counts,
            &order,
            false,
        ) {
            Ok(schedule) => schedule,
            Err(ExchangeLoweringError::Exchange(ipu_exchange::ExchangeError::Schedule(
                "SENDPICP instruction alignment",
            ))) => materialize_schedule_order(
                &topology,
                &scheduling,
                &incoming_bases,
                &receive_counts,
                &order,
                true,
            )?,
            Err(error) => return Err(error),
        }
    } else {
        materialize_greedy_schedule_with_priority(
            &topology,
            &scheduling,
            &incoming_bases,
            &receive_counts,
            priority,
        )?
    };
    let optimized = improve_pending_schedule(
        &topology,
        &scheduling,
        &incoming_bases,
        &receive_counts,
        schedule,
        "full-duplex",
    )?;
    finish_exchange_run(tile_count, problem.phase, incoming_bases, optimized)
}
