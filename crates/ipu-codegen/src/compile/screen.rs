//! Expand, screen, and place a concrete whole-device program.
use crate::compile::PipelineConfig;
#[cfg(test)]
use crate::estimate::Ipu21CostModel;
use crate::low::{LowProgram, lower_to_tiles};
#[cfg(test)]
use crate::package::check_exchange_budget;
use crate::package::{PackageBuildError, PackageBuildResult};
use crate::place::place;
#[cfg(test)]
use crate::planner::build_baseline;
#[cfg(test)]
use crate::{ComputeGraph, Precision};
use std::sync::Arc;
use std::time::Instant;

pub(super) fn expand_and_place(
    mid: &crate::MidProgram,
    planning: &PipelineConfig,
) -> PackageBuildResult<(
    LowProgram,
    crate::Placement,
    crate::estimate::ExchangeFootprint,
)> {
    let (low, footprint) = expand_and_screen(
        mid,
        planning,
        Arc::new(crate::low::expand::ExpansionCache::default()),
    )?;
    let placement = place(&low)?;
    Ok((low, placement, footprint))
}

pub(super) fn expand_and_screen(
    mid: &crate::MidProgram,
    planning: &PipelineConfig,
    cache: Arc<crate::low::expand::ExpansionCache>,
) -> PackageBuildResult<(LowProgram, crate::estimate::ExchangeFootprint)> {
    let start = Instant::now();
    let mut analysis = crate::estimate::GeometryAnalysis::default();
    let expanded = crate::low::expand::expand_tiles_analyzed(
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
    let low = lower_to_tiles(&expanded, planning.diagnostic_checkpoints);
    if let Some(copy) = low
        .local_copies
        .iter()
        .find(|copy| crate::tile::local_copy_call(copy).is_none())
    {
        return Err(crate::TileLoweringError::InvalidLocalCopy {
            tile: low.shards[copy.source.index() as usize].tile,
            copy: copy.clone(),
        }
        .into());
    }
    Ok((low, footprint))
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
        let baseline = build_baseline(&graph, &config, &Ipu21CostModel).unwrap();
        let expanded = crate::expand_tiles(&baseline).unwrap();
        let fragments = crate::estimate::program_footprint(&expanded)
            .unwrap()
            .maximum_transfer_chunks_per_tile;
        assert!(fragments > 0);
        config.exchange_transfer_limit_per_tile = fragments;
        config.exchange_table_budget_bytes = 0;
        // Encoded storage is checked only once encoded; mid heuristics and
        // uncompressed row slots cannot veto this geometry-feasible plan.
        assert!(expand_and_place(&baseline, &config).is_ok());
        assert!(check_exchange_budget(1, &config).is_err());
        config.exchange_table_budget_bytes = 1;
        assert!(check_exchange_budget(1, &config).is_ok());
        config.exchange_transfer_limit_per_tile = fragments - 1;
        let error = expand_and_place(&baseline, &config).err().unwrap();
        assert!(matches!(
            error,
            PackageBuildError::ExchangeTransferLimitExceeded { .. }
        ));
        config.exchange_transfer_limit_per_tile = 0;
        let still_planned = [build_baseline(&graph, &config, &Ipu21CostModel).unwrap()];
        assert!(expand_and_place(&still_planned[0], &config).is_err());
        config.exchange_transfer_limit_per_tile = u64::MAX;
        assert!(expand_and_place(&still_planned[0], &config).is_ok());
    }
}
