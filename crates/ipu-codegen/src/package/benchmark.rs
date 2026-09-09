//! Compiler timings which stop before placement and exchange scheduling.
use super::*;
use std::sync::Arc;

#[derive(serde::Serialize)]
pub struct ExpansionBenchmark {
    pub planning_ms: f64,
    pub retained_finalists: usize,
    pub finalists: Vec<ExpansionTiming>,
    /// Cache entries, hits and misses, respectively.
    pub fragment_cache: (usize, u64, u64),
    pub copy_plan_cache: (usize, u64, u64),
    /// Matching selections are opportunities, not validated reusable graph fragments.
    pub selection_reuse: std::collections::BTreeMap<&'static str, SelectionReuse>,
}

#[derive(Default, serde::Serialize)]
pub struct SelectionReuse {
    pub occurrences: usize,
    pub distinct: usize,
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
    pub panel_mappings: usize,
}

/// Search once, then expand at most `limit` retained mid plans serially (zero
/// measures all candidates). Timings include
/// tile graph simplification and its normal analytical costing, but exclude
/// footprint screening, tile mapping, placement, physical transfer preparation,
/// scheduling, linking, and destruction of each completed low plan.
pub fn benchmark_mid_expansion(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    limit: usize,
    cache_enabled: bool,
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
    let cache = Arc::new(if cache_enabled {
        crate::low::expand::ExpansionCache::default()
    } else {
        crate::low::expand::ExpansionCache::disabled()
    });
    let mut selections = std::collections::BTreeMap::<
        &'static str,
        (usize, std::collections::BTreeSet<String>),
    >::new();
    for (finalist, mid) in
        plans
            .iter()
            .enumerate()
            .take(if limit == 0 { usize::MAX } else { limit })
    {
        // Outside the timed expansion. Ignore IDs/provenance but retain layouts,
        // tile ownership and alias-group relationships at the operation boundary.
        fn selections_in(
            graph: &crate::MidProgram,
            operations: &[crate::MidOperation],
            selections: &mut std::collections::BTreeMap<
                &'static str,
                (usize, std::collections::BTreeSet<String>),
            >,
        ) {
            for op in operations {
                if let crate::MidOperationKind::Repeat(repeat) = &op.kind {
                    selections_in(graph, &repeat.body.operations, selections);
                    continue;
                }
                let category = match &op.kind {
                    crate::MidOperationKind::Primitive(crate::Primitive::Compute { .. }) => {
                        "compute"
                    }
                    crate::MidOperationKind::Primitive(crate::Primitive::Copy { .. }) => "copy",
                    crate::MidOperationKind::Primitive(crate::Primitive::Sum { .. }) => "sum",
                    _ => "other",
                };
                let mut groups = Vec::new();
                let boundary = op
                    .inputs
                    .iter()
                    .chain(&op.results)
                    .map(|id| {
                        let value = &graph.values[id.index() as usize];
                        let group = groups
                            .iter()
                            .position(|g| *g == value.storage_group)
                            .unwrap_or_else(|| {
                                groups.push(value.storage_group);
                                groups.len() - 1
                            });
                        (&value.tensor_type, value.tile_offset, group)
                    })
                    .collect::<Vec<_>>();
                let entry = selections.entry(category).or_default();
                entry.0 += 1;
                entry
                    .1
                    .insert(format!("{}|{:?}|{:?}", graph.tile_count, op.kind, boundary));
            }
        }
        selections_in(mid, &mid.operations, &mut selections);
        let start = Instant::now();
        let expanded = crate::low::expand::expand_tiles_cached(
            mid,
            config.diagnostic_checkpoints,
            Arc::clone(&cache),
        )?;
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
            panel_mappings: low
                .exchange_phases
                .iter()
                .flat_map(|p| &p.transfers)
                .filter(|t| t.order == crate::CopyOrder::Panels)
                .count(),
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
        retained_finalists: plans.len(),
        finalists,
        fragment_cache: cache.stats(),
        copy_plan_cache: cache.plan_stats(),
        selection_reuse: selections
            .into_iter()
            .map(|(kind, (occurrences, keys))| {
                (
                    kind,
                    SelectionReuse {
                        occurrences,
                        distinct: keys.len(),
                    },
                )
            })
            .collect(),
    })
}
