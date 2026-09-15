//! Select implementations and directly construct executable whole-device mid.
//! Graph construction owns enumeration/ranking; family constructors receive a
//! selected choice and actual tensor types. No package evaluation occurs here.
use crate::compile::PipelineConfig;
use crate::estimate::CostModel;
use crate::graph::ComputeGraph;
use crate::mid::MidProgram;
mod attention;
pub(crate) mod bind;
pub(crate) mod build;
pub(crate) mod cache;
pub(crate) mod candidates;
pub(crate) mod catalogue;
pub(crate) mod checkpoint;
mod error;
pub(crate) mod fragments;
mod gemm;
mod layernorm;
pub(crate) mod operator;
pub mod optimistic;
mod parameter_homes;
mod proposals;
mod recipe;
pub(crate) use build::build_candidate;
pub(crate) use cache::FragmentCache;
pub use catalogue::OperatorCandidate;
pub use error::{LoweringError, LoweringResult};
pub use operator::{GemmOrientation, LocalOperandStaging, OperatorFamily};
pub(crate) use proposals::proposals;
pub(crate) use recipe::{Candidate, Recipe};
#[cfg(test)]
pub(crate) mod test_support;

pub(crate) fn build_baseline(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
) -> LoweringResult<MidProgram> {
    Ok(build::build_candidate(
        graph,
        config,
        costs,
        &FragmentCache::default(),
        &Recipe::baseline(config),
    )?
    .program)
}

#[cfg(test)]
mod tests;
