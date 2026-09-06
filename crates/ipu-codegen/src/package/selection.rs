//! Expand a bounded shortlist, model placement, then schedule its best candidates.
use super::*;

struct ModelledPlan {
    index: usize,
    cycles: u64,
    low: LowProgram,
    placement: crate::Placement,
    challenger: Option<Vec<u16>>,
}

pub(super) fn select_scheduled_finalist(
    finalists: Vec<crate::MidProgram>,
    planning: &PipelineConfig,
    tile_mapping: Option<&[u16]>,
) -> PackageBuildResult<(LowProgram, crate::exchange::ExchangeScheduleCache)> {
    let topology = active_topology(planning.tile_count)?;
    let screened = finalists
        .into_par_iter()
        .enumerate()
        .map(|(index, mid)| {
            let screen = || -> PackageBuildResult<_> {
                let mut expanded = crate::low::expand::expand_tiles(&mid)?;
                placement::map_tiles(&mut expanded, tile_mapping)?;
                let low = lower_to_tiles(&expanded, planning.diagnostic_checkpoints);
                let placement = place(&low)?;
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
        let mut feasible = false;
        for (mapped, candidate) in std::iter::once((false, Ok((low, placement))))
            .chain(mapped.map(|candidate| (true, candidate)))
        {
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
                Ok(((refined.total, index, mapped), low, cache))
            };
            match schedule() {
                Ok(candidate) => {
                    feasible = true;
                    if best
                        .as_ref()
                        .is_none_or(|(score, _, _)| candidate.0 < *score)
                    {
                        best = Some(candidate);
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
    let ((_, selected, mapped), low, cache) = best.ok_or(failure)?;
    tracing::info!(
        selected,
        mapped,
        "selected physically scheduled operator-plan finalist"
    );
    Ok((low, cache))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infeasible_extra_finalist_does_not_discard_a_valid_program() {
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [32, 16]).unwrap();
        let output = graph.gelu(input).unwrap();
        graph.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(4).with_automatic_input(input, Precision::F16);
        let mut finalists = lower_finalists(&graph, &config, &Ipu21CostModel, 1).unwrap();
        let expected = crate::low::expand::expand_tiles(&finalists[0]).unwrap();
        finalists.insert(0, crate::MidProgram::default());
        let (selected, _) = select_scheduled_finalist(finalists, &config, None).unwrap();
        assert_eq!(selected, lower_to_tiles(&expected, false));
        assert!(matches!(
            select_scheduled_finalist(vec![crate::MidProgram::default()], &config, None),
            Err(PackageBuildError::Low(
                crate::ExpansionError::EmptyTileGroup
            ))
        ));
        assert!(select_scheduled_finalist(vec![], &config, None).is_err());
    }
}
