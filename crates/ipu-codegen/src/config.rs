//! Compiler target and planner search configuration.

use crate::copy::CopyPlan;
use crate::cost::Ipu21CostModel;
use crate::graph::{TensorShape, ValueId};
use crate::layout::{MemoryClass, TensorFormat};
use crate::operator::{GemmPlanConstraint, Precision};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineConfig {
    pub target: HardwareTarget,
    pub tile_count: u16,
    pub inputs: BTreeMap<ValueId, TensorFormat>,
    /// Graph-boundary tensors whose layout may be selected by their first
    /// consumer. Precision remains fixed, while packaging exposes the chosen
    /// physical layout directly through the host binding.
    pub automatic_inputs: BTreeMap<ValueId, Precision>,
    pub search_domain: PlannerSearchDomain,
    pub planning_beam_width: usize,
    /// Complete beam finalists ranked with the physical exchange scheduler.
    pub exchange_schedule_finalists: usize,
    /// SRAM retained for exchange tables, profiling, host commands, and code.
    pub standard_memory_reservation_bytes: u64,
    /// Maximum SRAM per tile available to planned values and the reservation.
    pub tile_memory_budget_bytes: u64,
    pub profiling: ProfilingConfig,
    pub diagnostic_checkpoints: bool,
    pub exchange_diagnostics: bool,
    pub conversion_streaming: ConversionStreamingPolicy,
    pub(crate) resolved_active_tile_counts: Vec<u16>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConversionStreamingPolicy {
    Never,
    #[default]
    WhenRequired,
    Always,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttentionStrategy {
    #[default]
    Automatic,
    Flash,
    Materialized,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OperatorClass {
    Gemm,
    Gelu,
    Add,
    Attention,
}

/// Search axes shared by semantic operator plan generators.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannerSearchDomain {
    /// `None` derives useful tile counts from graph shapes; `Some` restricts
    /// planning to the explicitly supplied counts.
    pub(crate) active_tiles: Option<Vec<u16>>,
    pub(crate) operator_precisions: BTreeMap<OperatorClass, Vec<Precision>>,
    pub(crate) weight_memory_classes: Vec<MemoryClass>,
    pub(crate) attention_strategy: AttentionStrategy,
    pub(crate) gemm_plan_constraints: Vec<GemmPlanConstraint>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardwareTarget {
    Ipu21,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HardwareMemoryConstraints {
    pub standard_fixed_bytes: u64,
    pub interleaved_bytes: u64,
    pub interleaved_element_bytes: u64,
    pub total_bytes: u64,
    pub default_standard_reservation_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProfilingConfig {
    #[default]
    Disabled,
    Overall,
    Full,
}

impl std::fmt::Display for AttentionStrategy {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        output.write_str(match self {
            Self::Automatic => "auto",
            Self::Flash => "flash",
            Self::Materialized => "materialized",
        })
    }
}

impl std::str::FromStr for AttentionStrategy {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" | "automatic" => Ok(Self::Automatic),
            "flash" => Ok(Self::Flash),
            "materialized" => Ok(Self::Materialized),
            _ => Err("expected auto, flash, or materialized"),
        }
    }
}

impl Default for PlannerSearchDomain {
    fn default() -> Self {
        Self {
            active_tiles: None,
            operator_precisions: BTreeMap::from([
                (OperatorClass::Gemm, vec![Precision::F16, Precision::F32]),
                (OperatorClass::Gelu, vec![Precision::F16, Precision::F32]),
                (OperatorClass::Add, vec![Precision::F16, Precision::F32]),
                (OperatorClass::Attention, vec![Precision::F16]),
            ]),
            weight_memory_classes: vec![MemoryClass::Standard, MemoryClass::Interleaved],
            attention_strategy: AttentionStrategy::Automatic,
            gemm_plan_constraints: Vec::new(),
        }
    }
}

impl PlannerSearchDomain {
    pub(crate) fn precisions(&self, operator: OperatorClass) -> &[Precision] {
        self.operator_precisions
            .get(&operator)
            .map_or(&[], Vec::as_slice)
    }

    pub(crate) fn permits_precision(&self, operator: OperatorClass, precision: Precision) -> bool {
        self.precisions(operator).contains(&precision)
    }

    pub(crate) fn permits_weight_memory(&self, memory_class: MemoryClass) -> bool {
        self.weight_memory_classes.contains(&memory_class)
    }

    pub(crate) fn active_tile_counts<'a>(
        &self,
        capacity: u16,
        shapes: impl IntoIterator<Item = &'a TensorShape>,
    ) -> Vec<u16> {
        match &self.active_tiles {
            None => {
                let mut counts = candidate_active_tile_counts(capacity);
                for count in shape_aware_active_tile_counts(capacity, shapes) {
                    if !counts.contains(&count) {
                        counts.push(count);
                    }
                }
                counts
            }
            Some(counts) => counts
                .iter()
                .copied()
                .filter(|&count| count <= capacity)
                .collect(),
        }
    }

    pub fn with_operator_precisions(
        mut self,
        operator: OperatorClass,
        precisions: impl IntoIterator<Item = Precision>,
    ) -> Self {
        let mut unique = Vec::new();
        for precision in precisions {
            if !unique.contains(&precision) {
                unique.push(precision);
            }
        }
        self.operator_precisions.insert(operator, unique);
        self
    }

    pub fn with_weight_memory_classes(
        mut self,
        classes: impl IntoIterator<Item = MemoryClass>,
    ) -> Self {
        let mut unique = Vec::new();
        for class in classes {
            if !unique.contains(&class) {
                unique.push(class);
            }
        }
        self.weight_memory_classes = unique;
        self
    }

    pub fn with_active_tile_counts(mut self, counts: impl IntoIterator<Item = u16>) -> Self {
        let mut active_tiles = Vec::new();
        for count in counts {
            if count != 0 && !active_tiles.contains(&count) {
                active_tiles.push(count);
            }
        }
        self.active_tiles = Some(active_tiles);
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
}

impl HardwareTarget {
    pub const fn cost_model(self) -> Ipu21CostModel {
        match self {
            Self::Ipu21 => Ipu21CostModel,
        }
    }

    pub const fn memory_constraints(self) -> HardwareMemoryConstraints {
        match self {
            Self::Ipu21 => HardwareMemoryConstraints {
                standard_fixed_bytes: crate::memory::IPU21_STANDARD_FIXED_BYTES as u64,
                interleaved_bytes: crate::memory::IPU21_INTERLEAVED_REGION_BYTES as u64,
                interleaved_element_bytes: ipu_target::memory::IPU21_INTERLEAVED_ELEMENT_SIZE
                    as u64,
                total_bytes: crate::memory::IPU21_PLANNED_DATA_BYTES as u64,
                default_standard_reservation_bytes:
                    crate::memory::IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES as u64,
            },
        }
    }

    pub const fn copy_plan(self) -> CopyPlan {
        match self {
            Self::Ipu21 => CopyPlan {
                // Larger strided regions lose more to worker bank contention
                // than they save in supervisor call overhead on IPU21.
                parallel_strided_max_bytes: 512,
            },
        }
    }
}

impl ProfilingConfig {
    pub const fn records_overall_time(self) -> bool {
        !matches!(self, Self::Disabled)
    }

    pub const fn records_steps(self) -> bool {
        matches!(self, Self::Full)
    }
}

impl std::fmt::Display for ProfilingConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Disabled => "none",
            Self::Overall => "overall",
            Self::Full => "full",
        })
    }
}

impl std::str::FromStr for ProfilingConfig {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" | "disabled" => Ok(Self::Disabled),
            "overall" => Ok(Self::Overall),
            "full" => Ok(Self::Full),
            _ => Err("profiling mode must be one of: none, overall, full"),
        }
    }
}

impl PipelineConfig {
    pub fn new(tile_count: u16) -> Self {
        let target = HardwareTarget::Ipu21;
        let memory = target.memory_constraints();
        Self {
            target,
            tile_count,
            inputs: BTreeMap::new(),
            automatic_inputs: BTreeMap::new(),
            search_domain: PlannerSearchDomain::default(),
            planning_beam_width: 64,
            exchange_schedule_finalists: 1,
            standard_memory_reservation_bytes: memory.default_standard_reservation_bytes,
            tile_memory_budget_bytes: memory.total_bytes,
            profiling: ProfilingConfig::default(),
            diagnostic_checkpoints: false,
            exchange_diagnostics: false,
            conversion_streaming: ConversionStreamingPolicy::WhenRequired,
            resolved_active_tile_counts: Vec::new(),
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

    pub fn with_search_domain(mut self, search_domain: PlannerSearchDomain) -> Self {
        self.search_domain = search_domain;
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

pub(crate) fn candidate_active_tile_counts(capacity: u16) -> Vec<u16> {
    if capacity == 0 {
        return vec![0];
    }
    let mut counts = vec![capacity];
    // Power-of-two subsets provide progressively smaller fallback grids.
    let mut power = 1u16;
    while let Some(next) = power.checked_mul(2) {
        if next > capacity {
            break;
        }
        power = next;
    }
    loop {
        if !counts.contains(&power) {
            counts.push(power);
        }
        if power == 1 {
            break;
        }
        power /= 2;
    }
    counts
}

pub(crate) fn shape_aware_active_tile_counts<'a>(
    capacity: u16,
    shapes: impl IntoIterator<Item = &'a TensorShape>,
) -> Vec<u16> {
    let minimum = capacity.div_ceil(2);
    let mut counts = shapes
        .into_iter()
        .flat_map(|shape| shape.0.iter().copied())
        .filter_map(|extent| {
            let extent = u16::try_from(extent).ok()?;
            (extent > 1 && extent <= capacity).then(|| capacity / extent * extent)
        })
        .filter(|&count| count >= minimum && count < capacity)
        .collect::<Vec<_>>();
    counts.sort_unstable_by(|left, right| right.cmp(left));
    counts.dedup();
    counts
}
