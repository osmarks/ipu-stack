//! Enumerate explicit recipe changes without building or evaluating packages.

use crate::PipelineConfig;
use crate::graph::ComputeGraph;
use crate::planner::{Candidate, Recipe};
use std::collections::BTreeSet;

pub(super) const PACKING_ROWS: [u16; 4] = [32, 64, 128, 256];

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
    let mut candidates = Vec::new();
    let mut propose = |recipe: Recipe| {
        candidates.push(recipe.clone());
        if !config.diagnostic_checkpoints {
            let mut donated = recipe;
            donated.options.reuse_cast_inputs = !donated.options.reuse_cast_inputs;
            candidates.push(donated);
        }
    };
    propose(incumbent.recipe.clone());
    let mut early = incumbent.recipe.clone();
    early.options.cast_before_copies = !early.options.cast_before_copies;
    propose(early);
    if !config.diagnostic_checkpoints {
        for rows in std::iter::once(0).chain(PACKING_ROWS) {
            let mut recipe = incumbent.recipe.clone();
            recipe.options.packing_rows = rows;
            propose(recipe);
        }
        for limit in 0..=config.max_parallel_reductions {
            if limit == 1 {
                continue;
            }
            let mut recipe = incumbent.recipe.clone();
            recipe.options.parallel_reductions = limit;
            propose(recipe);
        }
        let mut recipe = incumbent.recipe.clone();
        recipe.options.disjoint_copy_sources = !recipe.options.disjoint_copy_sources;
        propose(recipe);
    }
    for operation in &operations {
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
            propose(recipe);
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
                propose(joint);
            }
            propose(recipe);
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
        match owner_mapping(incumbent, traffic) {
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
            recipe: incumbent.recipe.remapped(&mapping, program.tile_count)?,
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
    fn layout_proposals_include_whole_program_donation() {
        let (graph, config) = mlp();
        let incumbent = build::build_candidate(
            &graph,
            &config,
            &Ipu21CostModel,
            &crate::planner::cache::FragmentCache::default(),
            &Recipe::default(),
        )
        .unwrap();
        let choices = proposals(&graph, &config, &incumbent, None);
        let changed = choices
            .iter()
            .find(|p| p.recipe.plans != incumbent.recipe.plans)
            .unwrap();
        assert!(
            choices
                .iter()
                .any(|p| p.recipe.plans == changed.recipe.plans
                    && p.recipe.options.reuse_cast_inputs
                        != changed.recipe.options.reuse_cast_inputs)
        );
        for proposal in choices {
            let encoded = serde_json::to_vec(&proposal.recipe).unwrap();
            let restored: Recipe = serde_json::from_slice(&encoded).unwrap();
            assert!(restored == proposal.recipe);
        }
    }
    #[test]
    fn global_mapping_replays_without_changing_logical_work() {
        let (graph, config) = mlp();
        let cache = crate::planner::cache::FragmentCache::default();
        let baseline = build::build_candidate(
            &graph,
            &config,
            &Ipu21CostModel,
            &cache,
            &Recipe::baseline(&config),
        )
        .unwrap();
        let mapping = (0..config.tile_count).rev().collect::<Vec<_>>();
        let recipe = baseline
            .recipe
            .remapped(&mapping, config.tile_count)
            .unwrap();
        let mapped =
            build::build_candidate(&graph, &config, &Ipu21CostModel, &cache, &recipe).unwrap();
        let mut expected = baseline.program.clone();
        expected.remap_tiles(&mapping).unwrap();
        expected.refresh_estimates().unwrap();
        assert_eq!(mapped.program, expected);
        let inverse = recipe.remapped(&mapping, config.tile_count).unwrap();
        let restored =
            build::build_candidate(&graph, &config, &Ipu21CostModel, &cache, &inverse).unwrap();
        for (old, new) in baseline.program.values.iter().zip(&restored.program.values) {
            for tile in 0..old.tensor_type.format.layout.tiling.tile_count {
                assert_eq!(
                    old.owners.tile(tile, config.tile_count),
                    new.owners.tile(tile, config.tile_count)
                );
            }
        }
    }
}
