//! Compiler/planner configuration and explicit user overrides.
use crate::graph::{OperationId, ValueId};
use crate::mid::ReductionStaging;
use crate::planner::OperatorCandidate;
use crate::planner::catalogue::{default_operator_candidates, operator_candidates_for_tile_count};
use crate::planner::operator::{GemmOrientation, LocalOperandStaging};
use crate::tensor::{MemoryClass, Precision, TensorFormat};
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
    /// Maximum ordered local search steps after establishing a baseline.
    /// Parallel speculation can validate later proposals that an earlier
    /// improvement invalidates; these do not advance the search.
    pub optimization_steps: usize,
    /// Prefer distributed boundaries and conversion-inclusive memory costs.
    /// Experimental: smaller tensor peaks can still require larger exchange rows.
    pub capacity_baseline: bool,
    /// Compact endpoint-balanced exchange waves; None uses latency-oriented scheduling.
    pub exchange_stream_words: Option<std::num::NonZeroU32>,
    pub inputs: BTreeMap<ValueId, TensorFormat>,
    /// Graph-boundary tensors whose layout may be selected by their first
    /// consumer. Precision remains fixed, while packaging exposes the chosen
    /// physical layout directly through the host binding.
    pub automatic_inputs: BTreeMap<ValueId, Precision>,
    /// Signatures available independently to each operation. Earlier entries
    /// of the appropriate operation kind win when costs are equal.
    pub operator_candidates: Vec<OperatorCandidate>,
    /// Fixed operand precision for individual GEMMs, including operations inside Repeat.
    pub gemm_precisions: BTreeMap<OperationId, Precision>,
    /// Add near-capacity tile counts derived from graph tensor extents.
    pub shape_aware_active_tile_counts: bool,
    /// Per-operator catalogue breadth before local neighborhood evaluation.
    pub operator_candidate_limit: usize,
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
    /// Optional JSON/HTML estimator profiles for the baseline and local proposals.
    /// These explain planner decisions, not concrete placement.
    pub memory_profile_directory: Option<std::path::PathBuf>,
    /// Maximum SRAM per tile available to planned values and the standard
    /// reservation. Lower values emulate a model whose other persistent state
    /// occupies the remainder of SRAM.
    pub tile_memory_budget_bytes: u64,
    pub profiling: bool,
    /// Insert all-tile patched-breakpoint stops after semantic operators.
    pub diagnostic_checkpoints: bool,
    /// Controls whether one-use layout conversions may be populated as
    /// bounded slices immediately before their consuming dispatch.
    pub conversion_streaming: ConversionStreamingPolicy,
    /// Restricts attention planning to one execution strategy for controlled
    /// benchmarking; automatic planning retains both alternatives.
    pub attention_strategy: AttentionStrategy,
    pub attention_products: AttentionProducts,
    /// Experimental materialized QK/PV operand scales; None retains F16/F32.
    pub attention_fp8_scales: [Option<i8>; 2],
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
            memory_profile_directory: None,
            inputs: BTreeMap::new(),
            automatic_inputs: BTreeMap::new(),
            operator_candidates: default_operator_candidates(tile_count),
            gemm_precisions: BTreeMap::new(),
            shape_aware_active_tile_counts: true,
            optimization_steps: 8,
            capacity_baseline: false,
            exchange_stream_words: None,
            operator_candidate_limit: 64,
            exchange_table_budget_bytes: 80 * 1024,
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
            conversion_streaming: ConversionStreamingPolicy::WhenRequired,
            attention_strategy: AttentionStrategy::Automatic,
            attention_products: AttentionProducts::Automatic,
            attention_fp8_scales: [None; 2],
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

    pub fn with_operator_candidate_limit(mut self, width: usize) -> Self {
        self.operator_candidate_limit = width.max(1);
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
        let mut seen = BTreeSet::new();
        self.operator_candidates = counts
            .into_iter()
            .filter(|&count| count > 0 && count <= self.tile_count && seen.insert(count))
            .flat_map(operator_candidates_for_tile_count)
            .collect();
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
