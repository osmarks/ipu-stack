//! Whole-device implementation selection, tensor primitives, and layouts.
//! Tile enumeration occurs only after selection, in low expansion.

mod copy;
mod elementwise;
pub mod optimistic;
mod packing;
pub(crate) use copy::{independent_copy_prefix, independent_sum_prefix};
pub(crate) mod implementation;
mod primitive;
pub use primitive::*;
mod candidates;
mod catalogue;
mod layout;
mod operator;
mod ownership;
mod planner;
mod resolved;
mod view;
pub use crate::graph::AxisFactorView;

use candidates::*;
use catalogue::*;
pub use catalogue::{ConcreteOperatorCandidate, OperatorCandidate, OperatorFormatPolicy};
pub use layout::*;
pub use operator::*;
#[cfg(test)]
pub(crate) use planner::lower;
pub(crate) fn lower_finalists(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
    count: usize,
) -> LoweringResult<Vec<MidProgram>> {
    let mut configurations = vec![config.clone()];
    // New storage choices must not evict the native baseline from a bounded
    // beam before the complete implementations can be compared. This repeats
    // compact planning only; identical candidates share one tile expansion.
    if config.gemm_output_packing == GemmOutputPacking::Automatic
        && graph
            .operations()
            .iter()
            .any(|operation| matches!(operation.kind, OperationKind::View(_)))
    {
        let mut native = config.clone();
        native.gemm_output_packing = GemmOutputPacking::Native;
        configurations.push(native);
    }
    if config.attention_products == AttentionProducts::Automatic
        && config.attention_strategy != AttentionStrategy::Flash
        && graph
            .operations()
            .iter()
            .any(|op| matches!(op.kind, OperationKind::FlashAttention(_)))
    {
        // Preserve the shared-row baseline with both projection-store choices.
        for index in 0..configurations.len() {
            let mut baseline = configurations[index].clone();
            baseline.attention_products = AttentionProducts::SharedRows;
            configurations.push(baseline);
        }
    }
    let mut candidates = Vec::new();
    let mut failure = None;
    let span = tracing::Span::current();
    let searches = configurations
        .into_par_iter()
        .map(|configuration| {
            let _entered = span.enter();
            planner::plan_finalists(graph, &configuration, costs, count)
        })
        .collect::<Vec<_>>();
    for search in searches {
        match search {
            Ok(plans) => {
                for candidate in plans {
                    if !candidates.contains(&candidate) {
                        candidates.push(candidate);
                    }
                }
            }
            Err(error) => {
                tracing::info!(%error, "skipped infeasible planner configuration");
                failure = Some(error);
            }
        }
    }
    if candidates.is_empty() {
        return Err(failure.unwrap_or(LoweringError::InvalidImplementation));
    }
    let resolved = candidates
        .into_iter()
        .map(|program| implementation::resolve(program).ok_or(LoweringError::InvalidImplementation))
        .collect::<LoweringResult<Vec<_>>>()?;
    let mut candidates = Vec::with_capacity(resolved.len() * 2);
    for program in resolved {
        let program = if config.diagnostic_checkpoints {
            program
        } else {
            program.with_elementwise_fusions().unwrap_or(program)
        };
        if !config.diagnostic_checkpoints {
            candidates.extend(program.distributed_packing_candidates());
            for limit in 2..=config.max_parallel_reductions {
                if let Some(overlapped) = program.with_overlapped_reductions(limit)
                    && !candidates.contains(&overlapped)
                {
                    candidates.push(overlapped);
                }
            }
        }
        if let Some(rotated) = program.with_disjoint_copy_sources(config.diagnostic_checkpoints) {
            candidates.push(rotated);
        }
        candidates.push(program);
    }
    Ok(candidates)
}
#[cfg(test)]
pub(crate) fn expand_tiles(
    program: &MidProgram,
) -> crate::ExpansionResult<std::sync::Arc<crate::TileGraph>> {
    let program = implementation::resolve(program.clone())
        .ok_or(crate::ExpansionError::InvalidOperatorPlan)?;
    crate::low::expand::expand_tiles(&program, true)
}
use planner::*;

use crate::estimate::MemoizedCostModel;
pub use crate::estimate::{
    CostModel, IPU21_TARGET_COSTS, Ipu21CostModel, MemoryPeaks, MemoryUsage,
};
use crate::estimate::{region_peak_memory, region_peak_memory_with_multiplicity};
use crate::graph::{
    AttentionOptions, ComputeGraph, GemmOptions, GraphInputKind, Operation, OperationId,
    OperationKind, Repeat, TensorShape, ValueId,
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GemmOutputPacking {
    #[default]
    Automatic,
    Native,
    Packed,
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
    /// Complete plans returned per planning configuration for geometry screening.
    pub expanded_plan_finalists: usize,
    /// Expanded candidates admitted to placement/mapping, plus the smallest
    /// estimated-storage alternative if absent. Independent of scheduling count.
    pub placement_finalists: usize,
    /// Number of buildable finalists compared with physical exchange scheduling.
    /// A compact alternative is prioritized; late failures may try the remaining
    /// placed candidates, bounded by the placement shortlist.
    pub exchange_schedule_finalists: usize,
    /// Hard limit on actual compact encoded exchange tables per tile.
    /// Set to u64::MAX to disable this limit.
    pub exchange_table_budget_bytes: u64,
    /// Static TX/RX fragments per tile, counted from concrete spans before
    /// scheduling. Repeat bodies count once. Independent of encoded bytes.
    /// Set to u64::MAX to disable this complexity limit.
    pub exchange_transfer_limit_per_tile: u64,
    /// Search-only penalty per estimated exchange-table byte. Does not change
    /// reported execution cycles. Budget failures automatically retry with
    /// stronger penalties to preserve simpler prefixes earlier in the graph.
    pub exchange_table_cost_per_byte: u64,
    /// Diagnostic constraints which retain only one GEMM plan family for the
    /// named source operations.
    pub gemm_plan_constraints: Vec<GemmPlanConstraint>,
    /// Compare native output with panel-packed projection output, or force a mode for diagnostics.
    pub gemm_output_packing: GemmOutputPacking,
    /// Maximum independent sums offered as one spatially distributed batch.
    pub max_parallel_reductions: usize,
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
    pub attention_products: AttentionProducts,
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

/// Restrict product-layout choices for controlled attention comparisons.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttentionProducts {
    #[default]
    Automatic,
    SharedRows,
    QkOnly,
    PvOnly,
    Independent,
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
            expanded_plan_finalists: 16,
            placement_finalists: 4,
            exchange_schedule_finalists: 1,
            exchange_table_budget_bytes: 64 * 1024,
            exchange_transfer_limit_per_tile: 16_384,
            exchange_table_cost_per_byte: 0,
            gemm_plan_constraints: Vec::new(),
            gemm_output_packing: GemmOutputPacking::Automatic,
            max_parallel_reductions: 3,
            standard_memory_reservation_bytes: u64::from(
                crate::memory::IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES,
            ),
            tile_memory_budget_bytes: u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES),
            profiling: false,
            diagnostic_checkpoints: false,
            exchange_diagnostics: false,
            conversion_streaming: ConversionStreamingPolicy::WhenRequired,
            attention_strategy: AttentionStrategy::Automatic,
            attention_products: AttentionProducts::Automatic,
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

    pub fn with_attention_products(mut self, products: AttentionProducts) -> Self {
        self.attention_products = products;
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
pub enum MidOperationKind {
    Primitive(Primitive),
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
    /// Values read in the enclosing region, including a repeat's parameter sequences.
    pub(crate) fn read_values(&self) -> impl Iterator<Item = &MidValueId> {
        let sequences = match &self.kind {
            MidOperationKind::Repeat(repeat) => repeat.iterated_inputs.as_slice(),
            _ => &[],
        };
        self.inputs.iter().chain(sequences.iter().flatten())
    }

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

/// Whole-device tensor program. Search recipes retain compact implementations;
/// final selection inlines those primitives before low enumerates tiles.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MidProgram {
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
    #[error("selected operator implementation is invalid")]
    InvalidImplementation,
    #[error("cannot create planning worker pool: {0}")]
    PlanningThreads(String),
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
        "operation {operation:?} has no candidate within tile SRAM (smallest rejected peak: standard {standard} bytes plus {standard_reservation} bytes package support, interleaved {interleaved} bytes, total {total} bytes including {exchange_rows} bytes estimated exchange tables, contiguous-standard overflow {standard_contiguous_overflow} bytes)"
    )]
    InsufficientMemory {
        operation: OperationId,
        standard: u64,
        standard_reservation: u64,
        exchange_rows: u64,
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

#[cfg(test)]
#[test]
#[ignore = "manual full-size planner timing; does not access hardware"]
fn profile_mlp_finalist_expansion() {
    let mut graph = ComputeGraph::new();
    let input = graph.host_input("input", [1, 729, 1152]).unwrap();
    let up = graph.parameter("up", [1, 1152, 4304]).unwrap();
    let down = graph.parameter("down", [1, 4304, 1152]).unwrap();
    let hidden = graph.gemm(input, up).unwrap();
    let hidden = graph.gelu(hidden).unwrap();
    let output = graph.gemm(hidden, down).unwrap();
    graph.set_outputs([output]).unwrap();
    let config = PipelineConfig::new(1472)
        .with_automatic_input(input, Precision::F16)
        .with_automatic_input(up, Precision::F16)
        .with_automatic_input(down, Precision::F16);
    let start = std::time::Instant::now();
    let finalists = planner::plan_finalists(&graph, &config, &crate::Ipu21CostModel, 8).unwrap();
    eprintln!(
        "compact planning: {:?}, {} finalists",
        start.elapsed(),
        finalists.len()
    );
    for (index, mid) in finalists.into_iter().enumerate() {
        for operation in &mid.operations {
            if let Some(plan) = operation.operator_plan()
                && let OperatorDispatch::BlockedGemm { orientation, .. } = plan.dispatch
            {
                eprintln!(
                    "finalist {index}: source {:?}, {:?}, weight memory {:?}",
                    operation.source,
                    plan.dispatch,
                    plan.requirements.inputs[orientation.operand_indices().1]
                        .format
                        .layout
                        .memory_class,
                );
            }
        }
        let mid = implementation::resolve(mid.clone()).unwrap();
        let start = std::time::Instant::now();
        let expanded = crate::low::expand::expand_tiles(&mid, true).unwrap();
        eprintln!(
            "finalist {index}: expansion {:?}, estimated cycles {}, exchange {}",
            start.elapsed(),
            expanded.estimated_cycles,
            expanded.estimated_exchange_cycles
        );
    }
}
