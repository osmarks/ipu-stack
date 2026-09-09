//! Validate a bounded baseline, then retain only feasible regional improvements.
use super::selection::{ScheduledPlan, select_scheduled_cached};
use super::*;
use crate::mid::regional as search;

fn cycles(plan: &ScheduledPlan) -> PackageBuildResult<u64> {
    Ok(crate::estimate::scheduled_program_cycles(&plan.program.program, &plan.phases)?.total)
}

pub(super) fn select<T>(
    graph: &ComputeGraph,
    planning: &PipelineConfig,
    tile_mapping: Option<&[u16]>,
    mut finalize: impl FnMut(&mut ScheduledPlan) -> PackageBuildResult<T>,
) -> PackageBuildResult<(ScheduledPlan, T)> {
    // Regional search optimizes algorithms/layouts, not topology. Keep the
    // caller's tile map (or identity) fixed across baseline and replacements.
    let identity = (0..planning.tile_count).collect::<Vec<_>>();
    let tile_mapping = Some(tile_mapping.unwrap_or(&identity));
    let options = planning.regional_planning.as_ref().unwrap();
    if options.baseline_attempts == 0
        || options.beam_width == 0
        || options.candidates_per_region == 0
    {
        return Err(invalid(
            "regional planning needs nonzero baseline, beam and candidate budgets",
        ));
    }
    let costs = crate::estimate::MemoizedCostModel::new(&Ipu21CostModel, planning.tile_count);
    let mut validation = planning.clone();
    validation.regional_planning = None;
    validation.expanded_plan_finalists = 1;
    validation.placement_finalists = 1;
    validation.exchange_schedule_finalists = 1;
    let cache = std::sync::Arc::new(crate::low::expand::ExpansionCache::default());
    let mut evaluate = |program: &crate::MidProgram| -> PackageBuildResult<(ScheduledPlan, T)> {
        let resolved = search::resolve(program, planning.diagnostic_checkpoints)?;
        select_scheduled_cached(
            vec![resolved],
            &validation,
            tile_mapping,
            &mut finalize,
            cache.clone(),
        )
    };
    let mut incumbent = None;
    let mut failure = invalid("no feasible bounded regional baseline");
    for attempt in 0..options.baseline_attempts.min(3) {
        let seed = search::baseline_config(planning, attempt);
        let result = (|| -> PackageBuildResult<_> {
            let program = search::baseline(graph, &seed, &costs)?;
            let (physical, artifact) = evaluate(&program)?;
            let score = cycles(&physical)?;
            Ok((program, score, physical, artifact))
        })();
        match result {
            Ok(plan) => {
                tracing::info!(
                    attempt,
                    cycles = plan.1,
                    "accepted feasible regional baseline"
                );
                incumbent = Some(plan);
                break;
            }
            Err(error) => {
                tracing::info!(attempt, %error, "rejected regional baseline");
                failure = error;
            }
        }
    }
    let (mut program, mut score, mut physical, mut artifact) = incumbent.ok_or(failure)?;
    let mut evaluations = 0;
    let mut accepted = 0;
    let mut bounded = 0;
    'passes: for pass in 0..options.passes {
        let mut changed = false;
        for region in search::regions(graph) {
            if evaluations >= options.max_evaluations {
                break 'passes;
            }
            let alternatives = match search::replacements(
                graph,
                &program,
                region.clone(),
                planning,
                options,
                &costs,
            ) {
                Ok(plans) => plans,
                Err(error) => {
                    tracing::info!(pass, region=?region, %error, "kept incumbent after regional search failure");
                    continue;
                }
            };
            let mut ranked = Vec::new();
            for (index, candidate) in alternatives.into_iter().enumerate() {
                if search::compute_lower_bound(&candidate.program) >= score {
                    bounded += 1;
                    continue;
                }
                // Local costing ranks alternatives. It is not a feasibility proof:
                // phase merging and global memory lifetimes still need validation.
                let local_score = candidate
                    .local
                    .as_ref()
                    .and_then(|local| {
                        let resolved =
                            search::resolve(local, planning.diagnostic_checkpoints).ok()?;
                        let mut analysis = crate::estimate::GeometryAnalysis::default();
                        let low = crate::low::expand::expand_tiles_analyzed(
                            &resolved,
                            planning.diagnostic_checkpoints,
                            cache.clone(),
                            &mut analysis,
                        )
                        .ok()?;
                        Some(low.estimated_cycles)
                    })
                    .unwrap_or(u64::MAX);
                ranked.push((local_score, index, candidate.program));
            }
            ranked.sort_by_key(|(cycles, index, _)| (*cycles, *index));
            for (_, _, candidate) in ranked {
                if evaluations >= options.max_evaluations {
                    break 'passes;
                }
                // This is a throughput bound, not a heuristic exchange price.
                if search::compute_lower_bound(&candidate) >= score {
                    bounded += 1;
                    continue;
                }
                evaluations += 1;
                match evaluate(&candidate) {
                    Ok((replacement, replacement_artifact)) => {
                        let replacement_score = match cycles(&replacement) {
                            Ok(score) => score,
                            Err(error) => {
                                tracing::info!(%error, "kept incumbent after replacement scoring failed");
                                continue;
                            }
                        };
                        if replacement_score < score {
                            tracing::info!(pass, region=?region, before=score, after=replacement_score, "accepted feasible regional improvement");
                            program = candidate;
                            score = replacement_score;
                            physical = replacement;
                            artifact = replacement_artifact;
                            accepted += 1;
                            changed = true;
                            // Rebuild alternatives against the new boundary state on the next pass.
                            break;
                        }
                    }
                    Err(error) => {
                        tracing::info!(pass, region=?region, %error, "kept incumbent after replacement failed validation")
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    tracing::info!(
        evaluations,
        accepted,
        bounded,
        cycles = score,
        "completed regional optimization"
    );
    Ok((physical, artifact))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn regional_mlp_accepts_a_faster_complete_plan() {
        let mut graph = ComputeGraph::new();
        let x = graph.host_input("x", [32, 32]).unwrap();
        let up = graph.parameter("up", [32, 128]).unwrap();
        let down = graph.parameter("down", [128, 32]).unwrap();
        let y = graph.gemm(x, up).unwrap();
        let y = graph.gelu(y).unwrap();
        let y = graph.gemm(y, down).unwrap();
        graph.set_outputs([y]).unwrap();
        graph.add_planning_region(0..3).unwrap();
        let mut config = PipelineConfig::new(8)
            .with_automatic_input(x, Precision::F16)
            .with_automatic_input(up, Precision::F16)
            .with_automatic_input(down, Precision::F16);
        config.regional_planning = Some(crate::RegionalPlanning::default());
        let mut baseline = None;
        let (selected, artifact) = select(&graph, &config, None, |plan| {
            let score = cycles(plan)?;
            baseline.get_or_insert(score);
            Ok(score)
        })
        .unwrap();
        assert_eq!(cycles(&selected).unwrap(), artifact);
        assert!(
            artifact < baseline.unwrap(),
            "baseline={baseline:?}, selected={artifact}"
        );
    }

    #[test]
    fn failed_replacements_leave_the_validated_baseline_available() {
        let mut graph = ComputeGraph::new();
        let x = graph.host_input("x", [32, 32]).unwrap();
        let w = graph.parameter("w", [32, 32]).unwrap();
        let y = graph.gemm(x, w).unwrap();
        graph.set_outputs([y]).unwrap();
        graph.add_planning_region(0..1).unwrap();
        let mut config = PipelineConfig::new(8)
            .with_automatic_input(x, Precision::F16)
            .with_automatic_input(w, Precision::F16);
        config.regional_planning = Some(crate::RegionalPlanning {
            max_evaluations: 2,
            ..Default::default()
        });
        let mut validations = 0;
        let (_, artifact) = select(&graph, &config, Some(&(0..8).collect::<Vec<_>>()), |_| {
            validations += 1;
            if validations == 1 {
                Ok(42)
            } else {
                Err(invalid("injected late package failure"))
            }
        })
        .unwrap();
        assert_eq!(artifact, 42);
        assert!(
            validations > 1,
            "the test must exercise replacement rejection"
        );
    }
}
