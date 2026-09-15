//! Operator tests expose intermediate boundaries independently of package policy.
use crate::PipelineConfig;
use crate::estimate::CostModel;
use crate::graph::{ComputeGraph, GraphInputKind};
use crate::mid::MidProgram;
use crate::planner::cache::FragmentCache;
use crate::planner::{LoweringError, LoweringResult, build};
use crate::tensor::{Layout, TensorFormat};

pub(crate) fn lower(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
) -> LoweringResult<MidProgram> {
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
    config.cast_before_copies = true;
    let mut program = build::select(graph, &config, costs, &FragmentCache::default(), true)?;
    program.reorder_casts();
    program.compose_copies();
    program
        .refresh_estimates()
        .ok_or(LoweringError::InvalidImplementation)?;
    Ok(program)
}
