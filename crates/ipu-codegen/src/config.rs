//! Compiler/planner configuration and explicit user overrides.
use crate::graph::ValueId;
use crate::tensor::TensorFormat;
use ipu_target::Target;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineConfig {
    pub target: Target,
    pub tile_count: u16,
    /// Bijection from planned tile indices to execution tile indices.
    pub tile_mapping: Option<Vec<u16>>,
    /// Donate dead FP16 scratch during low optimization, after copy elimination.
    pub reuse_cast_inputs: bool,
    /// Compact endpoint-balanced exchange waves; None uses latency-oriented scheduling.
    pub exchange_stream_words: Option<std::num::NonZeroU32>,
    pub inputs: BTreeMap<ValueId, TensorFormat>,
    /// Hard limit on actual compact encoded exchange tables per tile.
    /// Set to u64::MAX to disable this limit.
    pub exchange_table_budget_bytes: u64,
    /// Static TX/RX fragments per tile, counted from concrete spans before
    /// scheduling. Repeat bodies count once. Independent of encoded bytes.
    /// Set to u64::MAX to disable this complexity limit.
    pub exchange_transfer_limit_per_tile: u64,
    /// Standard-addressed SRAM retained for exchange tables, profiling data,
    /// host commands, and generated tile programs built after planning.
    pub standard_memory_reservation_bytes: u64,
    /// Optional JSON/HTML estimator profiles for the baseline.
    /// These explain planner decisions, not concrete placement.
    pub memory_profile_directory: Option<std::path::PathBuf>,
    /// Maximum SRAM per tile available to planned values and the standard
    /// reservation. Lower values emulate a model whose other persistent state
    /// occupies the remainder of SRAM.
    pub tile_memory_budget_bytes: u64,
    pub profiling: bool,
    /// Insert all-tile patched-breakpoint stops after semantic operators.
    pub diagnostic_checkpoints: bool,
}

impl PipelineConfig {
    pub fn new(tile_count: u16) -> Self {
        Self {
            target: Target::Ipu21,
            tile_count,
            tile_mapping: None,
            memory_profile_directory: None,
            inputs: BTreeMap::new(),
            reuse_cast_inputs: false,
            exchange_stream_words: None,
            exchange_table_budget_bytes: 80 * 1024,
            exchange_transfer_limit_per_tile: 16_384,
            standard_memory_reservation_bytes: Target::Ipu21.support_reservation_bytes(),
            tile_memory_budget_bytes: Target::Ipu21.planned_data_bytes(),
            profiling: false,
            diagnostic_checkpoints: false,
        }
    }

    pub fn with_input(mut self, value: ValueId, format: TensorFormat) -> Self {
        self.inputs.insert(value, format);
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
