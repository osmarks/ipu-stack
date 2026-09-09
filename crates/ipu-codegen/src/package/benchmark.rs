//! Compiler timings which stop before placement and exchange scheduling.
use super::*;

#[derive(serde::Serialize)]
pub struct ExpansionBenchmark {
    pub planning_ms: f64,
    pub finalists: Vec<ExpansionTiming>,
}

#[derive(serde::Serialize)]
pub struct ExpansionTiming {
    pub finalist: usize,
    pub mid_operations: usize,
    pub mid_values: usize,
    pub expand_ms: f64,
    pub tile_lists_ms: f64,
    pub shards: usize,
    pub kernels: usize,
    pub local_copies: usize,
    pub exchange_phases: usize,
    pub logical_transfers: usize,
    pub recipients: usize,
}

/// Search once, then expand each retained mid plan serially. Timings include
/// tile graph simplification and its normal analytical costing, but exclude
/// footprint screening, tile mapping, placement, physical transfer preparation,
/// scheduling, linking, and destruction of each completed low plan.
pub fn benchmark_mid_expansion(
    graph: &ComputeGraph,
    config: &PipelineConfig,
) -> PackageBuildResult<ExpansionBenchmark> {
    let start = Instant::now();
    let plans = lower_finalists(
        graph,
        config,
        &Ipu21CostModel,
        config
            .expanded_plan_finalists
            .max(config.exchange_schedule_finalists),
    )?;
    let planning_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut finalists = Vec::new();
    for (finalist, mid) in plans.iter().enumerate() {
        let start = Instant::now();
        let expanded = crate::low::expand::expand_tiles(mid, config.diagnostic_checkpoints)?;
        let expand_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let low = crate::low::lower_to_tiles(&expanded, config.diagnostic_checkpoints);
        let tile_lists_ms = start.elapsed().as_secs_f64() * 1000.0;
        let timing = ExpansionTiming {
            finalist,
            mid_operations: mid.operations.len(),
            mid_values: mid.values.len(),
            expand_ms,
            tile_lists_ms,
            shards: low.shards.len(),
            kernels: low.kernel_runs.len(),
            local_copies: low.local_copies.len(),
            exchange_phases: low.exchange_phases.len(),
            logical_transfers: low.exchange_phases.iter().map(|p| p.transfers.len()).sum(),
            recipients: low
                .exchange_phases
                .iter()
                .flat_map(|p| &p.transfers)
                .map(|t| t.destinations.len())
                .sum(),
        };
        tracing::info!(
            finalist,
            expand_ms,
            tile_lists_ms,
            "benchmarked mid-to-low expansion"
        );
        finalists.push(timing);
    }
    Ok(ExpansionBenchmark {
        planning_ms,
        finalists,
    })
}
