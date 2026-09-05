//! Candidate selection and executable, layout-aware block IR.
//!
//! The analytical beam screens coupled implementation choices. Candidate builders
//! produce a [`MidProgram`] with explicit blocks and movement before per-tile
//! projection. Neither low lowering nor placement expands whole operators.

mod block;
mod call;
pub(crate) use call::*;
pub(crate) mod implementation;
pub use block::*;
pub use implementation::{
    BlockBuildError, BlockBuildResult, logical_view_byte_spans, shard_storage_bytes,
    view_byte_spans,
};

mod candidates;
mod copy;
pub use copy::*;
mod catalogue;
mod layout;
mod operator;
mod ownership;
mod passes;
mod planner;
mod resolved;
mod view;
pub use crate::graph::AxisFactorView;

use candidates::*;
use catalogue::*;
pub use catalogue::{OperatorCandidate, OperatorFormatPolicy};
pub use layout::*;
pub use operator::*;
#[cfg(test)]
pub(crate) use planner::lower;
pub(crate) fn lower_finalists(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
    count: usize,
) -> LoweringResult<Vec<std::sync::Arc<MidProgram>>> {
    planner::plan_finalists(graph, config, costs, count)?
        .iter()
        .map(|candidate| implementation::build_blocks(candidate).map_err(LoweringError::from))
        .collect()
}
#[cfg(test)]
pub(crate) use implementation::build_blocks;
use planner::*;

use crate::estimate::MemoizedCostModel;
pub use crate::estimate::{
    CostModel, IPU21_TARGET_COSTS, Ipu21CostModel, MemoryPeaks, MemoryUsage,
};
use crate::estimate::{region_peak_memory, region_peak_memory_with_multiplicity};
use crate::graph::{
    AddOptions, AttentionOptions, ComputeGraph, GemmOptions, GraphInputKind, Operation,
    OperationId, OperationKind, Repeat, TensorShape, ValueId,
};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

/// Exact blocked-GEMM geometry retained for planner diagnosis. Constraints
/// are keyed by the source graph operation and bypass candidate pruning and
/// conservative whole-graph memory rejection. Concrete placement remains the
/// final authority on whether the resulting package fits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmPlanConstraint {
    pub source_operation: u32,
    pub orientation: GemmOrientation,
    pub row_partitions: u16,
    pub column_partitions: u16,
    pub inner_partitions: u16,
    pub result_row_partitions: u16,
    pub result_column_partitions: u16,
    pub output_column_block: u32,
    pub weight_memory_class: MemoryClass,
    pub reduction_staging: ReductionStaging,
    pub local_weight_staging: LocalOperandStaging,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineConfig {
    pub tile_count: u16,
    pub inputs: BTreeMap<ValueId, TensorFormat>,
    /// Graph-boundary tensors whose layout may be selected by their first
    /// consumer. Precision remains fixed, while packaging exposes the chosen
    /// physical layout directly through the host binding.
    pub automatic_inputs: BTreeMap<ValueId, Precision>,
    /// Signatures available independently to each operation. Earlier entries
    /// of the appropriate operation kind win when costs are equal.
    pub operator_candidates: Vec<OperatorCandidate>,
    /// Add near-capacity tile counts derived from graph tensor extents.
    pub shape_aware_active_tile_counts: bool,
    /// Maximum number of partial format assignments retained after each
    /// operation in a straight-line region.
    pub planning_beam_width: usize,
    /// Number of complete beam finalists to materialize and rank with the
    /// physical exchange scheduler. One retains analytical-only selection.
    pub exchange_schedule_finalists: usize,
    /// Diagnostic constraints which retain only one GEMM plan family for the
    /// named source operations.
    pub gemm_plan_constraints: Vec<GemmPlanConstraint>,
    /// Standard-addressed SRAM retained for exchange tables, profiling data,
    /// host commands, and generated tile programs built after planning.
    pub standard_memory_reservation_bytes: u64,
    /// Maximum SRAM per tile available to planned values and the standard
    /// reservation. Lower values emulate a model whose other persistent state
    /// occupies the remainder of SRAM.
    pub tile_memory_budget_bytes: u64,
    pub profiling: bool,
    /// Insert all-tile patched-breakpoint stops after semantic operators.
    pub diagnostic_checkpoints: bool,
    /// Emit exchange-scheduler lower bounds, per-tile role pressure, and
    /// critical dependency chains while constructing the final package.
    pub exchange_diagnostics: bool,
    /// Controls whether one-use layout conversions may be populated as
    /// bounded slices immediately before their consuming dispatch.
    pub conversion_streaming: ConversionStreamingPolicy,
    /// Restricts attention planning to one execution strategy for controlled
    /// benchmarking; automatic planning retains both alternatives.
    pub attention_strategy: AttentionStrategy,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConversionStreamingPolicy {
    /// Require complete converted values.
    Never,
    /// Prefer complete values, retaining streaming when materialization does
    /// not fit the target memory budget.
    #[default]
    WhenRequired,
    /// Stream every eligible conversion, primarily for diagnostics and
    /// memory-constrained deployment experiments.
    Always,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttentionStrategy {
    #[default]
    Automatic,
    Flash,
    Materialized,
}

impl PipelineConfig {
    pub fn new(tile_count: u16) -> Self {
        Self {
            tile_count,
            inputs: BTreeMap::new(),
            automatic_inputs: BTreeMap::new(),
            operator_candidates: default_operator_candidates(tile_count),
            shape_aware_active_tile_counts: true,
            planning_beam_width: 64,
            exchange_schedule_finalists: 1,
            gemm_plan_constraints: Vec::new(),
            standard_memory_reservation_bytes: u64::from(
                crate::memory::IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES,
            ),
            tile_memory_budget_bytes: u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES),
            profiling: false,
            diagnostic_checkpoints: false,
            exchange_diagnostics: false,
            conversion_streaming: ConversionStreamingPolicy::WhenRequired,
            attention_strategy: AttentionStrategy::Automatic,
        }
    }

    pub fn with_input(mut self, value: ValueId, format: TensorFormat) -> Self {
        self.inputs.insert(value, format);
        self.automatic_inputs.remove(&value);
        self
    }

    pub fn with_automatic_input(mut self, value: ValueId, precision: Precision) -> Self {
        self.inputs.remove(&value);
        self.automatic_inputs.insert(value, precision);
        self
    }

    pub fn with_planning_beam_width(mut self, width: usize) -> Self {
        self.planning_beam_width = width.max(1);
        self
    }

    pub fn with_exchange_schedule_finalists(mut self, finalists: usize) -> Self {
        self.exchange_schedule_finalists = finalists.max(1);
        self
    }

    pub fn with_attention_strategy(mut self, strategy: AttentionStrategy) -> Self {
        self.attention_strategy = strategy;
        self
    }

    pub fn with_gemm_plan_constraint(mut self, constraint: GemmPlanConstraint) -> Self {
        self.gemm_plan_constraints
            .retain(|existing| existing.source_operation != constraint.source_operation);
        self.gemm_plan_constraints.push(constraint);
        self
    }

    /// Restrict default operator planning to explicit active tile counts.
    /// This is useful when evaluating a fixed occupancy rather than allowing
    /// the planner to trade occupancy against communication and memory use.
    pub fn with_active_tile_counts(mut self, counts: impl IntoIterator<Item = u16>) -> Self {
        let mut candidates = Vec::new();
        for count in counts {
            if count == 0 || count > self.tile_count {
                continue;
            }
            candidates.extend(operator_candidates_for_tile_count(count));
        }
        candidates.dedup();
        self.operator_candidates = candidates;
        self.shape_aware_active_tile_counts = false;
        self
    }

    pub fn with_standard_memory_reservation(mut self, bytes: u64) -> Self {
        self.standard_memory_reservation_bytes = bytes;
        self
    }

    pub fn with_tile_memory_budget(mut self, bytes: u64) -> Self {
        self.tile_memory_budget_bytes = bytes;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MidValueId(u32);

impl MidValueId {
    pub const fn index(self) -> u32 {
        self.0
    }

    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidValue {
    pub id: MidValueId,
    /// Selected rotation of layout ownership within the graph's tile group.
    pub tile_offset: u16,
    pub tensor_type: TensorType,
    /// Semantic value represented by this value; conversions retain the same
    /// origin. Region arguments also refer to their high-level argument ID.
    pub origin: ValueId,
    /// Values in the same group use the same logical-to-physical tile mapping.
    /// Structured iteration uses this to keep successive parameter blocks
    /// addressable by a single advancing base pointer.
    pub storage_group: MidValueId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[expect(
    clippy::large_enum_variant,
    reason = "keep selected plans inline during beam expansion"
)]
pub enum MidOperationKind {
    Operator {
        plan: OperatorPlan,
        deferred_inputs: Vec<Option<DeferredInputPlan>>,
        implementation: Option<std::sync::Arc<MidProgram>>,
    },
    Convert(ConversionPlan),
    Repeat(MidRepeat),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidOperation {
    pub source: Option<OperationId>,
    pub inputs: Vec<MidValueId>,
    pub results: Vec<MidValueId>,
    pub kind: MidOperationKind,
    pub estimated_cycles: u64,
    pub estimated_exchange_cycles: u64,
}

impl MidOperation {
    pub fn operator_plan(&self) -> Option<&OperatorPlan> {
        match &self.kind {
            MidOperationKind::Operator { plan, .. } => Some(plan),
            _ => None,
        }
    }

    pub fn deferred_inputs(&self) -> &[Option<DeferredInputPlan>] {
        match &self.kind {
            MidOperationKind::Operator {
                deferred_inputs, ..
            } => deferred_inputs,
            _ => &[],
        }
    }

    pub fn conversion_plan(&self) -> Option<&ConversionPlan> {
        match &self.kind {
            MidOperationKind::Convert(plan) => Some(plan),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidRegion {
    pub arguments: Vec<MidValueId>,
    pub operations: Vec<MidOperation>,
    pub yields: Vec<MidValueId>,
    pub estimated_cycles: u64,
    pub peak_memory: MemoryPeaks,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidRepeat {
    pub count: u32,
    pub carried_inputs: usize,
    pub invariant_inputs: usize,
    /// One normalized value list for each iterated body argument. Keeping the
    /// lists on the structured operation avoids unrolling layer parameters.
    pub iterated_inputs: Vec<Vec<MidValueId>>,
    pub body: MidRegion,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidInput {
    pub name: String,
    pub kind: GraphInputKind,
    pub value: MidValueId,
}

/// Transient beam recipe consumed by the executable-block builder.
/// This is never passed to low or retained in a selected MidProgram.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImplementationCandidate {
    pub tile_count: u16,
    pub inputs: Vec<MidInput>,
    pub values: Vec<MidValue>,
    pub operations: Vec<MidOperation>,
    pub outputs: Vec<MidValueId>,
    pub estimated_cycles: u64,
    pub estimated_exchange_cycles: u64,
    pub peak_memory: MemoryPeaks,
}

// Estimation policy is kept in `estimate` so this module remains focused on IR and lowering.

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LoweringError {
    #[error("cannot create planning worker pool: {0}")]
    PlanningThreads(String),
    #[error(transparent)]
    Blocks(#[from] BlockBuildError),
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error(transparent)]
    Storage(#[from] crate::storage::StorageError),
    #[error("mid-level lowering requires a nonzero tile count")]
    EmptyTileGroup,
    #[error("no tensor type was supplied for graph input {0:?}")]
    MissingInputType(ValueId),
    #[error("graph has no stored shape for value {0:?}")]
    MissingShape(ValueId),
    #[error("operation {0:?} has no legal format candidate")]
    NoCandidate(OperationId),
    #[error(
        "operation {operation:?} has no candidate within tile SRAM (smallest rejected peak: standard {standard} bytes plus {standard_reservation} bytes package support, interleaved {interleaved} bytes, simultaneous tensor total {total} bytes, contiguous-standard overflow {standard_contiguous_overflow} bytes)"
    )]
    InsufficientMemory {
        operation: OperationId,
        standard: u64,
        standard_reservation: u64,
        interleaved: u64,
        total: u64,
        standard_contiguous_overflow: u64,
    },
    #[error(
        "GEMM operation {0:?} has per-batch right operands; only weights broadcast across every batch dimension are currently supported"
    )]
    UnsupportedGemmBatching(OperationId),
    #[error("internal lowering error: value {0:?} is unavailable")]
    UnknownValue(ValueId),
}

pub type LoweringResult<T> = std::result::Result<T, LoweringError>;

#[cfg(test)]
mod tests;
