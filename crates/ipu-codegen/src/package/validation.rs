//! Expand, screen, and place a concrete whole-device program.
use super::*;
use std::sync::Arc;

/// Selected work and its completed provisional allocation/schedule. Package
/// sizing consumes these same artifacts before support storage changes addresses.
pub(super) struct ScheduledPlan {
    pub program: LowProgram,
    pub placement: crate::Placement,
    pub phases: Vec<crate::PhysicalExchangePhase>,
    pub cache: crate::exchange::ExchangeScheduleCache,
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

pub(super) fn expand_and_screen(
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
        let baseline = lower_baseline(&graph, &config, &Ipu21CostModel).unwrap();
        let expanded = crate::expand_tiles(&baseline).unwrap();
        let fragments = crate::estimate::program_footprint(&expanded)
            .unwrap()
            .maximum_transfer_chunks_per_tile;
        assert!(fragments > 0);
        config.exchange_transfer_limit_per_tile = fragments;
        config.exchange_table_budget_bytes = 0;
        // Encoded storage is checked only once encoded; mid heuristics and
        // uncompressed row slots cannot veto this geometry-feasible plan.
        assert!(expand_and_place(&baseline, &config, None).is_ok());
        assert!(check_exchange_budget(1, &config).is_err());
        config.exchange_table_budget_bytes = 1;
        assert!(check_exchange_budget(1, &config).is_ok());
        config.exchange_transfer_limit_per_tile = fragments - 1;
        let error = super::local::optimize(
            &graph,
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
        let still_planned = [lower_baseline(&graph, &config, &Ipu21CostModel).unwrap()];
        assert!(expand_and_place(&still_planned[0], &config, None).is_err());
        config.exchange_transfer_limit_per_tile = u64::MAX;
        assert!(expand_and_place(&still_planned[0], &config, None).is_ok());
    }
}
