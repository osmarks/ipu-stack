//! Expand a bounded shortlist, model placement, then schedule its best candidates.
use super::*;
use std::sync::Arc;

struct ExpandedPlan {
    index: usize,
    low: LowProgram,
    footprint: crate::estimate::ExchangeFootprint,
}

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
    mut finalize: impl FnMut(&mut ScheduledPlan) -> PackageBuildResult<(u64, T)>,
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
                search
                    .expanded_plan_finalists
                    .max(search.exchange_schedule_finalists),
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
    mut finalize: impl FnMut(&mut ScheduledPlan) -> PackageBuildResult<(u64, T)>,
) -> PackageBuildResult<(ScheduledPlan, T)> {
    let topology = active_topology(planning.tile_count)?;
    let expansion_cache = Arc::new(crate::low::expand::ExpansionCache::default());
    let mut failure = invalid("no operator-plan finalists");
    let mut expanded = Vec::new();
    let mut finalists = finalists.into_iter().enumerate();
    // Retain only the placement shortlist between batches. Keeping all expanded
    // graphs until screening finishes makes memory scale with search breadth.
    loop {
        let batch = finalists
            .by_ref()
            .take(rayon::current_num_threads())
            .collect::<Vec<_>>();
        if batch.is_empty() {
            break;
        }
        let screened = batch
            .into_par_iter()
            .map(|(index, mid)| {
                let span = tracing::info_span!("screen_finalist", finalist = index);
                let _entered = span.enter();
                let result =
                    expand_and_screen(&mid, planning, tile_mapping, Arc::clone(&expansion_cache))
                        .map(|(low, footprint)| ExpandedPlan {
                            index,
                            low,
                            footprint,
                        });
                (index, result)
            })
            .collect();
        expanded.extend(feasible_candidates(screened, &mut failure));
        expanded = admit_candidates(
            expanded,
            planning
                .placement_finalists
                .max(planning.exchange_schedule_finalists),
            planning.exchange_table_budget_bytes,
            |plan| {
                (
                    plan.footprint.estimated_row_bytes(),
                    plan.low.estimated_cycles.saturating_add(
                        plan.footprint
                            .estimated_row_bytes()
                            .saturating_mul(planning.exchange_table_cost_per_byte),
                    ),
                    plan.index,
                )
            },
            "placement",
        );
    }
    tracing::info!(cache_stats = ?expansion_cache.stats(), plan_cache_stats = ?expansion_cache.plan_stats(), "expanded fragment cache");
    drop(expansion_cache);
    let placed = expanded
        .into_par_iter()
        .map(|plan| {
            let ExpandedPlan {
                index,
                low,
                footprint,
            } = plan;
            let result = (|| -> PackageBuildResult<_> {
                let start = Instant::now();
                let placement = place(&low)?;
                let (cycles, challenger) =
                    placement::model_mapping(&low, &placement, tile_mapping.is_none())?;
                tracing::info!(
                    finalist = index,
                    elapsed_ms = start.elapsed().as_millis(),
                    modelled_cycles = cycles,
                    estimated_row_bytes = footprint.estimated_row_bytes(),
                    "modelled expanded operator plan"
                );
                Ok(ModelledPlan {
                    index,
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
            })();
            (index, result)
        })
        .collect();
    let modelled = feasible_candidates(placed, &mut failure);
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
        // Stop after enough buildable packages. Failed attempts can use the
        // remaining placed candidates without rerunning search or expansion.
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
                    Ok(ScheduledPlan {
                        program: low,
                        placement,
                        phases: exchanges.phases,
                        cache: cache.clone(),
                    })
                };
                (mapped, schedule())
            })
            .collect::<Vec<_>>();
        let mut feasible = false;
        for (mapped, result) in results {
            match result {
                Ok(mut plan) => {
                    // Exact support placement is part of acceptance. A fast
                    // but unplaceable package must not consume the budget or
                    // prevent another finalist from being attempted.
                    let finalized = finalize(&mut plan);
                    caches[usize::from(mapped)] = plan.cache.clone();
                    match finalized {
                        Ok((cycles, artifact)) => {
                            let score = (cycles, index, mapped);
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

fn feasible_candidates<T>(
    candidates: Vec<(usize, PackageBuildResult<T>)>,
    failure: &mut PackageBuildError,
) -> Vec<T> {
    candidates.into_iter().filter_map(|(index, result)| match result {
        Ok(plan) => Some(plan),
        Err(error) => {
            tracing::info!(finalist=index, %error, "rejected infeasible operator-plan finalist");
            *failure = error;
            None
        }
    }).collect()
}

fn admit_scheduling_candidates(
    modelled: Vec<ModelledPlan>,
    planning: &PipelineConfig,
) -> Vec<ModelledPlan> {
    let (mut preferred, fallback) = partition_candidates(
        modelled,
        planning.exchange_schedule_finalists,
        planning.exchange_table_budget_bytes,
        |plan| (plan.row_bytes, plan.score, plan.index),
        "scheduling",
    );
    // These candidates have already paid for expansion and placement. Keep
    // them in reserve for late package failures; success still stops the loop.
    preferred.extend(fallback);
    preferred
}

// Placement bounds total expensive attempts. Scheduling prioritizes a smaller
// set (including a compact alternative), but package rejection may use the rest.
fn admit_candidates<T>(
    plans: Vec<T>,
    limit: usize,
    budget: u64,
    metrics: impl Fn(&T) -> (u64, u64, usize),
    stage: &'static str,
) -> Vec<T> {
    partition_candidates(plans, limit, budget, metrics, stage).0
}

fn partition_candidates<T>(
    mut plans: Vec<T>,
    limit: usize,
    budget: u64,
    metrics: impl Fn(&T) -> (u64, u64, usize),
    stage: &'static str,
) -> (Vec<T>, Vec<T>) {
    plans.sort_by_key(|plan| {
        let (bytes, score, index) = metrics(plan);
        (bytes.saturating_sub(budget), score, index)
    });
    let compact = plans
        .iter()
        .enumerate()
        .min_by_key(|(_, plan)| metrics(plan))
        .map(|(index, _)| index);
    let candidates = plans.len();
    let mut preferred = Vec::new();
    let mut fallback = Vec::new();
    for (position, plan) in plans.into_iter().enumerate() {
        if position < limit.max(1) || Some(position) == compact {
            preferred.push(plan);
        } else {
            fallback.push(plan);
        }
    }
    tracing::info!(
        stage,
        candidates,
        admitted_candidates = preferred.len(),
        "bounded finalist shortlist"
    );
    (preferred, fallback)
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
    let (low, footprint) = expand_and_screen(
        mid,
        planning,
        tile_mapping,
        Arc::new(crate::low::expand::ExpansionCache::default()),
    )?;
    let placement = place(&low)?;
    Ok((low, placement, footprint))
}

fn expand_and_screen(
    mid: &crate::MidProgram,
    planning: &PipelineConfig,
    tile_mapping: Option<&[u16]>,
    cache: Arc<crate::low::expand::ExpansionCache>,
) -> PackageBuildResult<(LowProgram, crate::estimate::ExchangeFootprint)> {
    let start = Instant::now();
    let mut analysis = crate::estimate::GeometryAnalysis::default();
    let mut expanded = crate::low::expand::expand_tiles_analyzed(
        mid,
        planning.diagnostic_checkpoints,
        cache,
        &mut analysis,
    )?;
    let expansion_ms = start.elapsed().as_millis();
    let footprint = crate::estimate::program_footprint_analyzed(&expanded, &mut analysis)?;
    let fragments = footprint.maximum_transfer_chunks_per_tile;
    tracing::info!(
        expansion_ms,
        expansion_and_footprint_ms = start.elapsed().as_millis(),
        estimated_row_bytes = footprint.estimated_row_bytes(),
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
    if let Some(copy) = low
        .local_copies
        .iter()
        .find(|copy| crate::tile::local_copy_call(copy).is_none())
    {
        return Err(crate::TileLoweringError::InvalidLocalCopy {
            tile: low.shards[copy.source.index() as usize].tile,
            source_shard: copy.source,
            source_offset: copy.source_offset,
            destination_shard: copy.destination,
            destination_offset: copy.destination_offset,
            bytes: copy.bytes,
        }
        .into());
    }
    Ok((low, footprint))
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
    fn batched_admission_preserves_the_complete_shortlist() {
        let plans = (0..97)
            .map(|index| ((index * 37) % 23, (index * 71) % 31, index as usize))
            .collect::<Vec<_>>();
        for limit in [0, 1, 4, 16] {
            for budget in [0, 10, u64::MAX] {
                let expected = admit_candidates(plans.clone(), limit, budget, |plan| *plan, "test");
                for batch_size in [1, 3, 8, 32] {
                    let mut retained = Vec::new();
                    for batch in plans.chunks(batch_size) {
                        retained.extend_from_slice(batch);
                        retained = admit_candidates(retained, limit, budget, |plan| *plan, "test");
                        assert!(retained.len() <= limit.max(1) + 1);
                    }
                    assert_eq!(retained, expected);
                }
            }
        }
    }

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
        let error = select_scheduled_finalist(
            finalists,
            &config,
            None,
            |_| -> PackageBuildResult<(u64, ())> {
                panic!("over-limit plan reached package finalization")
            },
        )
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
        for (budget, expected) in [
            (u64::MAX, vec![0, 2, 1]),
            (2_000, vec![2, 0, 1]),
            (500, vec![2, 0, 1]),
        ] {
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
            Err::<(u64, ()), _>(PackageBuildError::ExchangeBudgetExceeded {
                bytes: 100 * 1024,
                budget: 64 * 1024,
            })
        });
        assert!(matches!(
            result,
            Err(PackageBuildError::ExchangeBudgetExceeded { .. })
        ));
        assert_eq!(attempts, config.placement_finalists);
    }

    #[test]
    fn selection_uses_final_cost_after_support_placement() {
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [32, 16]).unwrap();
        let output = graph.gelu(input).unwrap();
        graph.set_outputs([output]).unwrap();
        let mut config = PipelineConfig::new(4).with_automatic_input(input, Precision::F16);
        config.exchange_schedule_finalists = 2;
        let mid = lower_finalists(&graph, &config, &Ipu21CostModel, 1)
            .unwrap()
            .remove(0);
        let mut attempts = 0;
        // Equal provisional costs, but the second package has a better final
        // schedule after support reservations change its addresses.
        let (_, selected) = select_scheduled_finalist(vec![mid; 2], &config, None, |_| {
            attempts += 1;
            Ok((100 / attempts, attempts))
        })
        .unwrap();
        assert_eq!(attempts, 2);
        assert_eq!(selected, 2);
    }

    #[test]
    fn package_placement_failure_tries_remaining_placed_candidates() {
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [32, 16]).unwrap();
        let output = graph.gelu(input).unwrap();
        graph.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(4).with_automatic_input(input, Precision::F16);
        let mid = lower_finalists(&graph, &config, &Ipu21CostModel, 1)
            .unwrap()
            .remove(0);
        for rejected in [0, config.placement_finalists - 1] {
            let mut attempts = 0;
            let result = select_scheduled_finalist(vec![mid.clone(); 20], &config, None, |_| {
                attempts += 1;
                if attempts <= rejected {
                    Err(PackageBuildError::Placement(
                        crate::PlacementError::OutOfMemory {
                            tile: 0,
                            class: crate::MemoryClass::Ipu21Interleaved,
                            bytes: 98304,
                        },
                    ))
                } else {
                    Ok((0, ()))
                }
            });
            assert!(result.is_ok());
            assert_eq!(attempts, rejected + 1);
        }
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
            select_scheduled_finalist(finalists, &config, None, |_| Ok((0, ()))).unwrap();
        assert_eq!(selected.program, lower_to_tiles(&expected, false));
        assert_eq!(selected.placement, place(&selected.program).unwrap());
        assert!(matches!(
            select_scheduled_finalist(vec![crate::MidProgram::default()], &config, None, |_| Ok((
                0,
                ()
            ))),
            Err(PackageBuildError::Low(
                crate::ExpansionError::EmptyTileGroup
            ))
        ));
        assert!(select_scheduled_finalist(vec![], &config, None, |_| Ok((0, ()))).is_err());
    }
}
