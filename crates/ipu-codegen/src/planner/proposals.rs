//! Enumerate explicit recipe changes without building or evaluating packages.
use crate::graph::ComputeGraph;
use crate::mid::PipelineConfig;
use crate::mid::baseline::{Baseline, Recipe};

pub(crate) fn proposals(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    incumbent: &Baseline,
) -> Vec<Recipe> {
    fn visit<'a>(source: &'a [crate::Operation], operations: &mut Vec<&'a crate::Operation>) {
        for op in source {
            if let crate::OperationKind::Repeat(repeat) = &op.kind {
                visit(&repeat.body.operations, operations);
            } else {
                operations.push(op);
            }
        }
    }
    let mut operations = Vec::new();
    visit(graph.operations(), &mut operations);
    // Include the incumbent so its opposite cast-storage policy is considered
    // alongside the paired layout changes below. Visited recipes are filtered.
    let mut candidates = vec![incumbent.recipe.clone()];
    for rows in [32, 64, 128, 256] {
        if incumbent.recipe.packing_rows != Some(rows) {
            let mut recipe = incumbent.recipe.clone();
            recipe.packing_rows = Some(rows);
            candidates.push(recipe);
        }
    }
    for limit in 2..=config.max_parallel_reductions {
        if incumbent.recipe.parallel_reductions != limit {
            let mut recipe = incumbent.recipe.clone();
            recipe.parallel_reductions = limit;
            candidates.push(recipe);
        }
    }
    if !incumbent.recipe.disjoint_copy_sources {
        let mut recipe = incumbent.recipe.clone();
        recipe.disjoint_copy_sources = true;
        candidates.push(recipe);
    }

    for &site in &incumbent.cast_sites {
        let mut recipe = incumbent.recipe.clone();
        if !recipe.cast_before_copies.insert(site) {
            recipe.cast_before_copies.remove(&site);
        }
        candidates.push(recipe);
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
            candidates.push(recipe);
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
                candidates.push(joint);
            }
            candidates.push(recipe);
        }
    }
    // Storage donation can enable a faster layout even though the cast itself
    // is slower. Evaluate it with each layout proposal, not only as an isolated
    // change that the cycle-improvement screen would always discard.
    candidates
        .into_iter()
        .flat_map(|recipe| {
            let mut alternate = recipe.clone();
            alternate.in_place_casts =
                Some(!recipe.in_place_casts.unwrap_or(config.capacity_baseline));
            [recipe, alternate]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mid::baseline;
    use crate::{Ipu21CostModel, Precision};
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
        let fragments = crate::mid::implementation::FragmentCache::default();
        for capacity in [false, true] {
            config.capacity_baseline = capacity;
            let recipe = Recipe::default();
            let direct = baseline::lower(
                &graph,
                &config,
                &Ipu21CostModel,
                &crate::mid::implementation::FragmentCache::default(),
                &recipe,
            )
            .unwrap();
            let memoized = baseline::lower(&graph, &config, &costs, &fragments, &recipe).unwrap();
            assert_eq!(direct.program, memoized.program);
            assert!(direct.recipe == memoized.recipe);
        }
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
                in_place_casts: Some(capacity_baseline),
                ..Recipe::default()
            };
            implicit.normalize(&config);
            explicit.normalize(&config);
            assert!(implicit == explicit);
        }
    }
}
