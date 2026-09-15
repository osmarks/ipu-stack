//! Enumerate explicit recipe changes without building or evaluating packages.

use crate::compile::PipelineConfig;
use crate::graph::ComputeGraph;
use crate::planner::{Candidate, Recipe};
use std::collections::BTreeSet;

/// A proposal contains decisions and, when available, a more specific ranking
/// estimate. This estimate never changes the executable program's own costing.
pub(crate) struct RecipeProposal {
    pub recipe: Recipe,
    pub estimated_cycles: Option<u64>,
}

pub(crate) fn proposals(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    incumbent: &Candidate,
    traffic: Option<&crate::exchange::MappingTraffic>,
) -> Vec<RecipeProposal> {
    let operations = graph
        .walk_operations()
        .filter(|op| !matches!(op.kind, crate::OperationKind::Repeat(_)))
        .collect::<Vec<_>>();
    let cast_sources = incumbent
        .cast_storage_sites
        .iter()
        .map(|site| site.source)
        .collect();
    // Preserve the former joint donation alternative as an ordinary scoped
    // proposal, alongside individual cast choices. The default never toggles.
    let mut candidates = Vec::new();
    // Operator defaults cover casts introduced by a layout change. Pair that
    // change with donation only for the affected families and direct consumers.
    let mut propose = |mut recipe: Recipe, sources: &BTreeSet<crate::OperationId>| {
        candidates.push(recipe.clone());
        if config.diagnostic_checkpoints || sources.is_empty() {
            return;
        }
        let storage = recipe
            .cast_storage
            .as_mut()
            .expect("candidate recipe has effective defaults");
        for &source in sources {
            let opposite = storage.for_operator(Some(source)).opposite();
            storage.operators.insert(source, opposite);
            storage.sites.retain(|site, _| site.source != source);
        }
        candidates.push(recipe);
    };
    let unchanged = BTreeSet::new();
    propose(incumbent.recipe.clone(), &cast_sources);
    if !config.diagnostic_checkpoints {
        for site in &incumbent.cast_storage_sites {
            let mut recipe = incumbent.recipe.clone();
            let storage = recipe.cast_storage.as_mut().unwrap();
            storage
                .sites
                .insert(site.clone(), storage.for_site(site).opposite());
            propose(recipe, &unchanged);
        }
    }
    for rows in [32, 64, 128, 256] {
        if incumbent.recipe.packing_rows != Some(rows) {
            let mut recipe = incumbent.recipe.clone();
            recipe.packing_rows = Some(rows);
            propose(recipe, &cast_sources);
        }
    }
    for limit in 2..=config.max_parallel_reductions {
        if incumbent.recipe.parallel_reductions != limit {
            let mut recipe = incumbent.recipe.clone();
            recipe.parallel_reductions = limit;
            propose(recipe, &cast_sources);
        }
    }
    if !incumbent.recipe.disjoint_copy_sources {
        let mut recipe = incumbent.recipe.clone();
        recipe.disjoint_copy_sources = true;
        propose(recipe, &cast_sources);
    }

    for site in &incumbent.cast_sites {
        let mut recipe = incumbent.recipe.clone();
        if !recipe.cast_before_copies.insert(site.clone()) {
            recipe.cast_before_copies.remove(site);
        }
        propose(recipe, &BTreeSet::from([site.source]));
    }

    for operation in &operations {
        let related = operations
            .iter()
            .filter(|candidate| {
                candidate.id == operation.id
                    || candidate
                        .inputs
                        .iter()
                        .any(|input| operation.results.contains(input))
            })
            .map(|candidate| candidate.id)
            .collect();
        let alternatives = incumbent
            .alternatives
            .get(&operation.id)
            .into_iter()
            .flatten()
            .filter(|plan| incumbent.recipe.plans.get(&operation.id) != Some(plan))
            .take(4);
        for plan in alternatives.clone() {
            let mut recipe = incumbent.recipe.clone();
            recipe.plans.insert(operation.id, plan.clone());
            propose(recipe, &related);
        }
        for &output in &operation.results {
            if graph.outputs().contains(&output)
                || incumbent.recipe.open_boundaries.contains(&output)
            {
                continue;
            }
            let consumers = operations
                .iter()
                .filter(|op| op.inputs.contains(&output))
                .collect::<Vec<_>>();
            if consumers.is_empty() {
                continue;
            }
            let mut recipe = incumbent.recipe.clone();
            recipe.open_boundaries.insert(output);
            // Re-enumerate consumers against the producer's new format. The
            // external boundary after each consumer remains canonical.
            for consumer in consumers {
                recipe.plans.remove(&consumer.id);
            }
            // A store format and its consumer can need to change together;
            // neither isolated change need improve the canonical baseline.
            for plan in alternatives.clone() {
                let mut joint = recipe.clone();
                joint.plans.insert(operation.id, plan.clone());
                propose(joint, &related);
            }
            propose(recipe, &related);
        }
    }
    let mut candidates = candidates
        .into_iter()
        .map(|recipe| RecipeProposal {
            recipe,
            estimated_cycles: None,
        })
        .collect::<Vec<_>>();
    if let Some(traffic) = traffic {
        match owner_mapping(graph, incumbent, traffic) {
            Ok(Some(proposal)) => candidates.push(proposal),
            Ok(None) => {}
            Err(error) => tracing::info!(%error, "could not propose owner mapping"),
        }
    }
    candidates
}

/// Preserve the existing block-transpose neighborhood as an ordinary joint
/// ownership proposal. The fabric model scores relabeling all current endpoints;
/// it does not approximate independent per-operator changes with one permutation.
/// A permutation preserves the coarse mid estimate, so use the model's predicted
/// cycle saving to rank this proposal with layout changes in the same shortlist.
fn owner_mapping(
    graph: &ComputeGraph,
    incumbent: &Candidate,
    traffic: &crate::exchange::MappingTraffic,
) -> Result<Option<RecipeProposal>, crate::mid::ProgramError> {
    let program = &incumbent.program;
    let identity = (0..program.tile_count).collect::<Vec<_>>();
    let baseline = traffic.score(&identity);
    let mut blocks = BTreeSet::from([program.tile_count]);
    for value in &program.values {
        for axis in &value.tensor_type.format.layout.tiling.axes {
            if let Some(stride) = axis.tile_stride {
                blocks.extend(
                    [stride, stride.saturating_mul(axis.partitions)]
                        .into_iter()
                        .filter(|&block| block > 1 && block <= program.tile_count),
                );
            }
        }
    }
    let mut best = None;
    let mut best_score = baseline;
    let mut candidates = 0;
    for block in blocks {
        for width in 2..block {
            if !block.is_multiple_of(width) {
                continue;
            }
            let mapping = (0..program.tile_count)
                .map(|tile| {
                    let base = tile / block * block;
                    let local = tile % block;
                    if u32::from(base) + u32::from(block) <= u32::from(program.tile_count) {
                        base + local % width * (block / width) + local / width
                    } else {
                        tile
                    }
                })
                .collect::<Vec<_>>();
            let score = traffic.score(&mapping);
            candidates += 1;
            tracing::debug!(block, width, cycles=score.0, pressure=%score.1, "modelled owner mapping");
            if score.0 < baseline.0 && score < best_score {
                best_score = score;
                best = Some(mapping);
            }
        }
    }
    tracing::info!(candidates, baseline_cycles=baseline.0, candidate_cycles=best_score.0,
        baseline_pressure=%baseline.1, candidate_pressure=%best_score.1,
        "screened ownership proposal by exchange resource load");
    best.map(|mapping| {
        Ok(RecipeProposal {
            recipe: incumbent
                .recipe
                .remapped(graph, &mapping, program.tile_count)?,
            estimated_cycles: Some(
                program
                    .estimated_cycles
                    .saturating_sub(baseline.0 - best_score.0),
            ),
        })
    })
    .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Precision;
    use crate::estimate::Ipu21CostModel;
    use crate::planner::build;
    fn mlp() -> (ComputeGraph, PipelineConfig) {
        let mut graph = ComputeGraph::new();
        let x = graph.host_input("x", [32, 64]).unwrap();
        let up = graph.parameter("up", [64, 128]).unwrap();
        let down = graph.parameter("down", [128, 64]).unwrap();
        let hidden = graph.gemm(x, up).unwrap();
        let hidden = graph.gelu(hidden).unwrap();
        let y = graph.gemm(hidden, down).unwrap();
        graph.set_outputs([y]).unwrap();
        let config = PipelineConfig::new(8)
            .with_automatic_input(x, Precision::F16)
            .with_automatic_input(up, Precision::F16)
            .with_automatic_input(down, Precision::F16);
        (graph, config)
    }

    #[test]
    fn memoization_does_not_change_baseline_selection() {
        let (graph, mut config) = mlp();
        let costs = crate::estimate::MemoizedCostModel::new(&Ipu21CostModel);
        let fragments = crate::planner::cache::FragmentCache::default();
        for capacity in [false, true] {
            config.capacity_baseline = capacity;
            let recipe = Recipe::default();
            let direct = build::build_candidate(
                &graph,
                &config,
                &Ipu21CostModel,
                &crate::planner::cache::FragmentCache::default(),
                &recipe,
            )
            .unwrap();
            let memoized =
                build::build_candidate(&graph, &config, &costs, &fragments, &recipe).unwrap();
            assert_eq!(direct.program, memoized.program);
            assert!(direct.recipe == memoized.recipe);
        }
    }

    #[test]
    fn layout_donation_proposals_leave_unrelated_families_fixed() {
        let (graph, config) = mlp();
        let incumbent = build::build_candidate(
            &graph,
            &config,
            &Ipu21CostModel,
            &crate::planner::cache::FragmentCache::default(),
            &Recipe::default(),
        )
        .unwrap();
        let first = graph.operations()[0].id;
        let consumer = graph.operations()[1].id;
        let unrelated = graph.operations()[2].id;
        let default = incumbent.recipe.cast_storage.as_ref().unwrap().default;
        let mut joint = 0;
        for proposal in proposals(&graph, &config, &incumbent, None) {
            let recipe = proposal.recipe;
            let policy = recipe.cast_storage.as_ref().unwrap();
            assert_eq!(policy.default, default);
            if recipe.open_boundaries != incumbent.recipe.open_boundaries
                || recipe.plans.get(&first) == incumbent.recipe.plans.get(&first)
                || recipe.plans.get(&consumer) != incumbent.recipe.plans.get(&consumer)
                || recipe.plans.get(&unrelated) != incumbent.recipe.plans.get(&unrelated)
                || policy.operators.is_empty()
            {
                continue;
            }
            assert_eq!(policy.for_operator(Some(unrelated)), default);
            assert_eq!(policy.for_operator(Some(first)), default.opposite());
            assert_eq!(policy.for_operator(Some(consumer)), default.opposite());
            joint += 1;
        }
        assert!(
            joint > 0,
            "layout search must retain joint donation proposals"
        );
    }

    #[test]
    fn implicit_cast_policy_has_one_checkpoint_identity() {
        for capacity_baseline in [false, true] {
            let config = PipelineConfig {
                capacity_baseline,
                ..PipelineConfig::new(8)
            };
            let mut implicit = Recipe::default();
            let mut explicit = Recipe {
                cast_storage: Some(crate::mid::cast::CastStoragePolicy::new(
                    if capacity_baseline {
                        crate::mid::cast::CastStorage::ReuseIfSmaller
                    } else {
                        crate::mid::cast::CastStorage::Separate
                    },
                )),
                ..Recipe::default()
            };
            implicit.normalize(&config);
            explicit.normalize(&config);
            assert!(implicit == explicit);
        }
    }
}
