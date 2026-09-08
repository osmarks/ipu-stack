//! Diagnostic regional high-to-mid search, including unavailable implementations.
//!
//! This is deliberately not an executable-program API. Costs are estimates, not
//! hardware bounds; hypotheses must be implemented and physically validated.
//! See `docs/OPTIMISTIC_PLANNING.md` for the search space and its limits.
mod conversion;
mod fusion;
mod report;
mod search;
#[cfg(test)]
mod tests;

use super::*;
pub use conversion::{ConversionPath, enumerate_conversions};
pub use search::{plan_graph, plan_region};
use std::ops::Range;
use std::sync::Arc;
// Expose the existing mid vocabulary, rather than a second layout/algorithm IR.
pub use super::{ElementOrder, MidProgram, OperatorPlan};

/// Region boundary formats are fixed for one invocation. Internal formats are not.
#[derive(Clone, Debug)]
pub struct RegionRequest {
    pub operations: Range<usize>,
    pub inputs: BTreeMap<ValueId, TensorFormat>,
    pub outputs: BTreeMap<ValueId, TensorFormat>,
}

#[derive(Clone, Debug)]
pub struct SearchOptions {
    pub max_operations: usize,
    pub beam_width: usize,
    /// One to four local/distribution edges per conversion.
    pub max_conversion_steps: usize,
    pub conversion_frontier: usize,
    /// Total operator candidates expanded across the region. Exhaustion is reported.
    pub max_expansions: usize,
    /// Maximum adjacent elementwise operations in a hypothetical fused kernel.
    pub max_fusion_operations: usize,
}
impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            max_operations: 8,
            beam_width: 16,
            max_conversion_steps: 3,
            conversion_frontier: 8,
            max_expansions: 2048,
            max_fusion_operations: 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CycleEstimate {
    pub optimistic: u64,
    /// Sensitivity estimate for hypothetical kernels, not a guaranteed upper bound.
    pub conservative: u64,
}
impl CycleEstimate {
    fn plus(self, other: Self) -> Self {
        Self {
            optimistic: self.optimistic.saturating_add(other.optimistic),
            conservative: self.conservative.saturating_add(other.conservative),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Assumption {
    MissingKernel(String),
    MissingEquivalenceRule,
    /// The conversion graph can express this route, but normal conversion insertion excludes it.
    ConversionEligibility,
    MissingCostModel,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransformKind {
    Alias,
    Cast,
    Pack,
    CastAndPack,
    Redistribute,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transform {
    pub from: TensorType,
    pub to: TensorType,
    pub kind: TransformKind,
    pub cycles: CycleEstimate,
    pub assumptions: BTreeSet<Assumption>,
}

/// Steps operate on whole-device values. Existing algorithm fragments are mid
/// programs; unsupported local combinations remain explicit diagnostic steps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepKind {
    Algorithm {
        plan: OperatorPlan,
        implementation: Arc<MidProgram>,
    },
    Transform(Transform),
    /// Original high operators and all their still-live results are retained.
    FusedElementwise {
        operations: Vec<Operation>,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    pub sources: Vec<OperationId>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub kind: StepKind,
    pub cycles: CycleEstimate,
    pub assumptions: BTreeSet<Assumption>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Value {
    pub origin: ValueId,
    pub tensor: TensorType,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiagnosticMidGraph {
    pub tile_count: u16,
    pub values: Vec<Value>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub steps: Vec<Step>,
    pub cycles: CycleEstimate,
    /// Sum of maximum shard sizes at each live interval, plus implementation
    /// scratch. Conservative across unrelated ownership grids; no placement run.
    pub memory: MemoryPeaks,
    /// Device-wide live bytes divided by tile count; excludes unknown scratch.
    pub minimum_live_bytes_per_tile: u64,
    pub assumptions: BTreeSet<Assumption>,
}
#[derive(Clone, Debug, Default)]
pub struct SearchReport {
    pub candidates: Vec<DiagnosticMidGraph>,
    pub expanded: usize,
    pub truncated: bool,
    pub rejected_memory: usize,
    pub rejected_implementations: usize,
}
#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    #[error("invalid optimistic search request: {0}")]
    InvalidRequest(String),
    #[error("unsupported regional operator {0:?}; choose a straight-line region")]
    UnsupportedOperation(OperationId),
    #[error(
        "no regional candidates; expanded {expanded}, memory rejections {memory}, implementation rejections {implementations}"
    )]
    NoCandidates {
        expanded: usize,
        memory: usize,
        implementations: usize,
    },
}

fn invalid(message: impl Into<String>) -> SearchError {
    SearchError::InvalidRequest(message.into())
}
fn valid_tensor(t: &TensorType, tiles: u16) -> bool {
    t.format.layout.tiling.tile_count <= tiles
        && t.format.layout.tiling.tile_count > 0
        && t.format.layout.resolve(&t.shape).is_ok()
        && crate::estimate::maximum_shard_bytes(t) != u64::MAX
}
