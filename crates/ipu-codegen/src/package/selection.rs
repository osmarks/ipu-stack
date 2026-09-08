//! Expand a bounded shortlist, model placement, then schedule its best candidates.
use super::*;

struct ModelledPlan {
    index: usize,
    score: u64,
    row_bytes: u64,
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

pub(super) fn select_graph_finalist<T>(
    graph: &ComputeGraph,
    planning: &PipelineConfig,
    tile_mapping: Option<&[u16]>,
    mut finalize: impl FnMut(&mut ScheduledPlan) -> PackageBuildResult<T>,
) -> PackageBuildResult<(ScheduledPlan, T)> {
    let costs = crate::estimate::MemoizedCostModel::new(&Ipu21CostModel, planning.tile_count);
    let mut search = planning.clone();
    let mut result = Err(invalid("no operator-plan finalists"));
    for penalty in std::iter::once(planning.exchange_table_cost_per_byte).chain(
        [16, 256]
            .into_iter()
            .filter(|p| *p > planning.exchange_table_cost_per_byte),
    ) {
        search.exchange_table_cost_per_byte = penalty;
        let finalists = build_phase("lower_mid", || {
            Ok(lower_finalists(
                graph,
                &search,
                &costs,
                search.exchange_schedule_finalists.max(4),
            )?)
        })?;
        result = select_scheduled_finalist(finalists, &search, tile_mapping, &mut finalize);
        if !matches!(
            result,
            Err(PackageBuildError::ExchangeBudgetExceeded { .. }
                | PackageBuildError::ExchangeTransferLimitExceeded { .. })
        ) {
            break;
        }
        tracing::info!(penalty, "geometry or encoded exchange budget exhausted");
    }
    result
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
                let (low, placement, footprint) = expand_and_place(&mid, planning, tile_mapping)?;
                let (cycles, challenger) =
                    placement::model_mapping(&low, &placement, tile_mapping.is_none())?;
                tracing::info!(
                    finalist = index,
                    modelled_cycles = cycles,
                    estimated_row_bytes = footprint.estimated_row_bytes(),
                    estimated_cycles = low.estimated_cycles,
                    estimated_exchange_cycles = low.estimated_exchange_cycles,
                    "modelled expanded operator plan"
                );
                Ok(ModelledPlan {
                    index,
                    // Preserve the search's storage preference after expansion.
                    // Otherwise a penalty retry simply reselects the fast,
                    // exchange-heavy plans that the retry was meant to avoid.
                    score: cycles.saturating_add(
                        footprint
                            .estimated_row_bytes()
                            .saturating_mul(planning.exchange_table_cost_per_byte),
                    ),
                    row_bytes: footprint.estimated_row_bytes(),
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
    let modelled = admit_scheduling_candidates(modelled, planning);
    let mut best = None;
    let mut scheduled = 0;
    // Preserve compatible phase recipes when a finalist fails late package
    // acceptance. Ordinary and mapped placements keep independent caches.
    let mut caches = [
        crate::ExchangeScheduleCache::default(),
        crate::ExchangeScheduleCache::default(),
    ];
    for plan in modelled {
        // The admission shortlist above bounds failed attempts too.
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
            .zip(caches.par_iter_mut())
            .map(|((mapped, candidate), cache)| {
                let _entered = span.enter();
                let schedule = || -> PackageBuildResult<_> {
                    let (low, placement) = candidate?;
                    let exchanges = crate::exchange::lower_exchanges_cached(
                        &low, &placement, &topology, false, cache,
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
                            cache: cache.clone(),
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
                    let finalized = finalize(&mut plan);
                    caches[usize::from(mapped)] = plan.cache.clone();
                    match finalized {
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

fn admit_scheduling_candidates(
    mut modelled: Vec<ModelledPlan>,
    planning: &PipelineConfig,
) -> Vec<ModelledPlan> {
    modelled.sort_by_key(|plan| {
        (
            plan.row_bytes
                .saturating_sub(planning.exchange_table_budget_bytes),
            plan.score,
            plan.index,
        )
    });
    let limit = planning.exchange_schedule_finalists.max(1);
    // Keep one genuinely smaller alternative outside the performance shortlist.
    // All other candidates are rejected before physical scheduling, including
    // when the admitted candidates fail package acceptance.
    let compact = modelled
        .iter()
        .enumerate()
        .min_by_key(|(_, plan)| (plan.row_bytes, plan.score, plan.index))
        .map(|(position, _)| position);
    let expanded_candidates = modelled.len();
    let mut position = 0;
    modelled.retain(|_| {
        let keep = position < limit || Some(position) == compact;
        position += 1;
        keep
    });
    tracing::info!(
        expanded_candidates,
        admitted_candidates = modelled.len(),
        "bounded physical exchange scheduling shortlist"
    );
    modelled
}

pub(super) fn expand_and_place(
    mid: &crate::MidProgram,
    planning: &PipelineConfig,
    tile_mapping: Option<&[u16]>,
) -> PackageBuildResult<(
    LowProgram,
    crate::Placement,
    crate::estimate::ExchangeFootprint,
)> {
    let mut expanded = crate::low::expand::expand_tiles(mid, planning.diagnostic_checkpoints)?;
    let footprint = crate::estimate::program_footprint(&expanded)?;
    let fragments = footprint.maximum_transfer_chunks_per_tile;
    tracing::info!(
        heuristic_row_bytes = mid.peak_memory.exchange_rows,
        geometry_fragments_per_tile = fragments,
        fragment_limit = planning.exchange_transfer_limit_per_tile,
        "screened exchange table geometry"
    );
    if fragments > planning.exchange_transfer_limit_per_tile {
        return Err(PackageBuildError::ExchangeTransferLimitExceeded {
            transfers: fragments,
            limit: planning.exchange_transfer_limit_per_tile,
        });
    }
    placement::map_tiles(&mut expanded, tile_mapping)?;
    let low = lower_to_tiles(&expanded, planning.diagnostic_checkpoints);
    let placement = place(&low)?;
    Ok((low, placement, footprint))
}

pub(super) fn check_exchange_budget(bytes: u64, config: &PipelineConfig) -> PackageBuildResult<()> {
    if bytes > config.exchange_table_budget_bytes {
        return Err(PackageBuildError::ExchangeBudgetExceeded {
            bytes,
            budget: config.exchange_table_budget_bytes,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exchange_budget_rejects_before_scheduling() {
        let mut graph = ComputeGraph::new();
        let x = graph.host_input("x", [2, 1, 1152]).unwrap();
        let gamma = graph.parameter("gamma", [1152]).unwrap();
        let beta = graph.parameter("beta", [1152]).unwrap();
        let y = graph.layer_norm(x, gamma, beta).unwrap();
        graph.set_outputs([y]).unwrap();
        let mut config = PipelineConfig::new(64)
            .with_automatic_input(x, Precision::F16)
            .with_automatic_input(gamma, Precision::F16)
            .with_automatic_input(beta, Precision::F16);
        let finalists = lower_finalists(&graph, &config, &Ipu21CostModel, 1).unwrap();
        let expanded = crate::expand_tiles(&finalists[0]).unwrap();
        let fragments = crate::estimate::program_footprint(&expanded)
            .unwrap()
            .maximum_transfer_chunks_per_tile;
        assert!(fragments > 0);
        config.exchange_transfer_limit_per_tile = fragments;
        config.exchange_table_budget_bytes = 0;
        // Encoded storage is checked only once encoded; mid heuristics and
        // uncompressed row slots cannot veto this geometry-feasible plan.
        assert!(expand_and_place(&finalists[0], &config, None).is_ok());
        assert!(check_exchange_budget(1, &config).is_err());
        config.exchange_table_budget_bytes = 1;
        assert!(check_exchange_budget(1, &config).is_ok());
        config.exchange_transfer_limit_per_tile = fragments - 1;
        let error =
            select_scheduled_finalist(finalists, &config, None, |_| -> PackageBuildResult<()> {
                panic!("over-limit plan reached package finalization")
            })
            .err()
            .unwrap();
        assert!(matches!(
            error,
            PackageBuildError::ExchangeTransferLimitExceeded { .. }
        ));
        config.exchange_transfer_limit_per_tile = 0;
        let still_planned = lower_finalists(&graph, &config, &Ipu21CostModel, 1).unwrap();
        assert!(expand_and_place(&still_planned[0], &config, None).is_err());
        config.exchange_transfer_limit_per_tile = u64::MAX;
        assert!(expand_and_place(&still_planned[0], &config, None).is_ok());
    }

    #[test]
    fn scheduling_admission_retains_a_compact_alternative() {
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [32, 16]).unwrap();
        let output = graph.gelu(input).unwrap();
        graph.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(4).with_automatic_input(input, Precision::F16);
        let mid = lower_finalists(&graph, &config, &Ipu21CostModel, 1)
            .unwrap()
            .remove(0);
        let (low, placement, _) = expand_and_place(&mid, &config, None).unwrap();
        for (budget, expected) in [(u64::MAX, vec![0, 2]), (2_000, vec![2]), (500, vec![2])] {
            let mut config = config.clone();
            config.exchange_table_budget_bytes = budget;
            let candidates = [(100, 10_000), (101, 10_000), (200, 1_000)]
                .into_iter()
                .enumerate()
                .map(|(index, (score, row_bytes))| ModelledPlan {
                    index,
                    score,
                    row_bytes,
                    low: low.clone(),
                    placement: placement.clone(),
                    challenger: None,
                })
                .collect();
            let admitted = admit_scheduling_candidates(candidates, &config);
            assert_eq!(
                admitted.iter().map(|plan| plan.index).collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn package_rejection_cannot_schedule_the_entire_shortlist() {
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
        let result = select_scheduled_finalist(vec![mid; 20], &config, None, |_| {
            attempts += 1;
            Err::<(), _>(PackageBuildError::ExchangeBudgetExceeded {
                bytes: 100 * 1024,
                budget: 64 * 1024,
            })
        });
        assert!(matches!(
            result,
            Err(PackageBuildError::ExchangeBudgetExceeded { .. })
        ));
        assert_eq!(attempts, 1);
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
