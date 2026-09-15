//! Operator tests expose intermediate boundaries independently of package policy.
use crate::compile::PipelineConfig;
use crate::estimate::CostModel;
use crate::graph::{ComputeGraph, GraphInputKind, Operation, OperationKind, ValueId};
use crate::mid::MidProgram;
use crate::planner::{FragmentCache, LoweringError, LoweringResult, Recipe, build};
use crate::tensor::{Layout, TensorFormat};
use std::collections::BTreeSet;

pub(crate) fn lower(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
) -> LoweringResult<MidProgram> {
    // Operator/kernel tests exercise a neighborhood with exposed boundaries.
    fn outputs(ops: &[Operation], ids: &mut BTreeSet<ValueId>) {
        for op in ops {
            ids.extend(&op.results);
            if let OperationKind::Repeat(repeat) = &op.kind {
                outputs(&repeat.body.operations, ids);
            }
        }
    }
    // These fixtures test operator lowering with explicit distributed host
    // bindings, independently of the package baseline's coarser boundary policy.
    let mut config = config.clone();
    for input in graph
        .inputs()
        .iter()
        .filter(|input| input.kind == GraphInputKind::Host)
    {
        if let Some(&precision) = config.automatic_inputs.get(&input.value) {
            let rows = input
                .shape
                .0
                .get(input.shape.0.len().saturating_sub(2))
                .copied()
                .unwrap_or(1);
            config.inputs.insert(
                input.value,
                TensorFormat {
                    precision,
                    layout: Layout::row_sharded(u32::from(config.tile_count).min(rows) as u16),
                },
            );
        }
    }
    let mut recipe = Recipe::default();
    outputs(graph.operations(), &mut recipe.open_boundaries);
    recipe.options.cast_before_copies = true;
    let mut program =
        build::select(graph, &config, costs, &FragmentCache::default(), &recipe)?.program;
    program.reorder_casts();
    program.compose_copies();
    program
        .refresh_estimates()
        .ok_or(LoweringError::InvalidImplementation)?;
    Ok(program)
}
