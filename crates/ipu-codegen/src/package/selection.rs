//! Expand a bounded shortlist, model placement, then schedule its best candidates.
use super::*;

struct ModelledPlan {
    index: usize,
    cycles: u64,
    low: LowProgram,
    placement: crate::Placement,
    challenger: Option<Vec<u16>>,
}

/// Selected work and its completed provisional allocation/schedule. Package
/// sizing consumes these same artifacts before support storage changes addresses.
pub(super) struct ScheduledPlan {
    pub program: LowProgram,
    pub placement: crate::Placement,
    pub phases: Vec<crate::PhysicalExchangePhase>,
    pub cache: crate::exchange::ExchangeScheduleCache,
}

pub(super) fn select_scheduled_finalist<T>(
    finalists: Vec<crate::MidProgram>,
    planning: &PipelineConfig,
    tile_mapping: Option<&[u16]>,
    mut finalize: impl FnMut(&mut ScheduledPlan) -> PackageBuildResult<T>,
) -> PackageBuildResult<(ScheduledPlan, T)> {
    let topology = active_topology(planning.tile_count)?;
    let screened = finalists
        .into_par_iter()
        .enumerate()
        .map(|(index, mid)| {
            let screen = || -> PackageBuildResult<_> {
                let (low, placement) = expand_and_place(&mid, planning, tile_mapping)?;
                let (cycles, challenger) =
                    placement::model_mapping(&low, &placement, tile_mapping.is_none())?;
                tracing::info!(
                    finalist = index,
                    modelled_cycles = cycles,
                    estimated_cycles = low.estimated_cycles,
                    estimated_exchange_cycles = low.estimated_exchange_cycles,
                    "modelled expanded operator plan"
                );
                Ok(ModelledPlan {
                    index,
                    cycles,
                    low,
                    placement,
                    challenger,
                })
            };
            (index, screen())
        })
        .collect::<Vec<_>>();
    let mut modelled = Vec::new();
    let mut failure = invalid("no operator-plan finalists");
    for (index, result) in screened {
        match result {
            Ok(plan) => modelled.push(plan),
            Err(error) => {
                tracing::info!(finalist=index, %error, "rejected infeasible operator-plan finalist");
                failure = error;
            }
        }
    }
    modelled.sort_by_key(|plan| (plan.cycles, plan.index));
    let mut best = None;
    let mut scheduled = 0;
    for plan in modelled {
        // An infeasible schedule does not consume the bounded finalist budget.
        if scheduled >= planning.exchange_schedule_finalists.max(1) {
            break;
        }
        let ModelledPlan {
            index,
            low,
            placement,
            challenger,
            ..
        } = plan;
        let mapped = challenger.map(|mapping| -> PackageBuildResult<_> {
            let mut graph = low.program.clone();
            placement::map_tiles(&mut graph, Some(&mapping))?;
            let low = lower_to_tiles(&graph, planning.diagnostic_checkpoints);
            let placement = place(&low)?;
            Ok((low, placement))
        });
        let candidates = std::iter::once((false, Ok((low, placement))))
            .chain(mapped.map(|candidate| (true, candidate)))
            .collect::<Vec<_>>();
        let span = tracing::Span::current();
        let results = candidates
            .into_par_iter()
            .map(|(mapped, candidate)| {
                let _entered = span.enter();
                let schedule = || -> PackageBuildResult<_> {
                    let (low, placement) = candidate?;
                    let mut cache = crate::exchange::ExchangeScheduleCache::default();
                    let exchanges = crate::exchange::lower_exchanges_cached(
                        &low, &placement, &topology, false, &mut cache,
                    )?;
                    let refined =
                        crate::estimate::scheduled_program_cycles(&low.program, &exchanges.phases)?;
                    tracing::info!(
                        finalist = index,
                        mapped,
                        analytical_cycles = low.estimated_cycles,
                        analytical_exchange_cycles = low.estimated_exchange_cycles,
                        scheduled_exchange_cycles = refined.exchange,
                        refined_cycles = refined.total,
                        "scheduled operator-plan finalist"
                    );
                    Ok((
                        (refined.total, index, mapped),
                        ScheduledPlan {
                            program: low,
                            placement,
                            phases: exchanges.phases,
                            cache,
                        },
                    ))
                };
                (mapped, schedule())
            })
            .collect::<Vec<_>>();
        let mut feasible = false;
        for (mapped, result) in results {
            match result {
                Ok((score, mut plan)) => {
                    // Exact support placement is part of acceptance. A fast
                    // but unplaceable package must not consume the budget or
                    // prevent another finalist from being attempted.
                    match finalize(&mut plan) {
                        Ok(artifact) => {
                            feasible = true;
                            if best
                                .as_ref()
                                .is_none_or(|(best_score, _, _)| score < *best_score)
                            {
                                best = Some((score, plan, artifact));
                            }
                        }
                        Err(error) => {
                            tracing::info!(finalist=index, mapped, %error, "rejected unbuildable operator-plan finalist");
                            failure = error;
                        }
                    }
                }
                Err(error) => {
                    tracing::info!(finalist=index, mapped, %error, "rejected unschedulable operator-plan finalist");
                    failure = error;
                }
            }
        }
        scheduled += usize::from(feasible);
    }
    let ((_, selected, mapped), plan, artifact) = best.ok_or(failure)?;
    tracing::info!(
        selected,
        mapped,
        "selected physically scheduled operator-plan finalist"
    );
    Ok((plan, artifact))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_rejection_does_not_consume_finalist_budget() {
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [32, 16]).unwrap();
        let output = graph.gelu(input).unwrap();
        graph.set_outputs([output]).unwrap();
        let mut config = PipelineConfig::new(4).with_automatic_input(input, Precision::F16);
        config.exchange_schedule_finalists = 1;
        let mid = lower_finalists(&graph, &config, &Ipu21CostModel, 1)
            .unwrap()
            .remove(0);
        let mut attempts = 0;
        let (_, accepted) =
            select_scheduled_finalist(vec![mid.clone(), mid], &config, None, |_| {
                attempts += 1;
                if attempts == 1 {
                    Err(invalid("exchange row tables do not fit"))
                } else {
                    Ok(attempts)
                }
            })
            .unwrap();
        assert_eq!(accepted, 2);
    }

    #[test]
    fn infeasible_extra_finalist_does_not_discard_a_valid_program() {
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [32, 16]).unwrap();
        let output = graph.gelu(input).unwrap();
        graph.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(4).with_automatic_input(input, Precision::F16);
        let mut finalists = lower_finalists(&graph, &config, &Ipu21CostModel, 1).unwrap();
        let expected =
            crate::low::expand::expand_tiles(&finalists[0], config.diagnostic_checkpoints).unwrap();
        finalists.insert(0, crate::MidProgram::default());
        let (selected, ()) =
            select_scheduled_finalist(finalists, &config, None, |_| Ok(())).unwrap();
        assert_eq!(selected.program, lower_to_tiles(&expected, false));
        assert_eq!(selected.placement, place(&selected.program).unwrap());
        assert!(matches!(
            select_scheduled_finalist(
                vec![crate::MidProgram::default()],
                &config,
                None,
                |_| Ok(())
            ),
            Err(PackageBuildError::Low(
                crate::ExpansionError::EmptyTileGroup
            ))
        ));
        assert!(select_scheduled_finalist(vec![], &config, None, |_| Ok(())).is_err());
    }
}

pub(super) fn expand_and_place(
    mid: &crate::MidProgram,
    planning: &PipelineConfig,
    tile_mapping: Option<&[u16]>,
) -> PackageBuildResult<(LowProgram, crate::Placement)> {
    let mut expanded = crate::low::expand::expand_tiles(mid, planning.diagnostic_checkpoints)?;
    placement::map_tiles(&mut expanded, tile_mapping)?;
    let low = lower_to_tiles(&expanded, planning.diagnostic_checkpoints);
    let placement = place(&low)?;
    Ok((low, placement))
}
