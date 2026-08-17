//! Mid-level, layout-aware representation.
//!
//! This is the boundary between semantic graph operations and scheduling. It
//! records tensor shapes, storage precision, element order, axis tiling, and
//! memory-class requirements, but deliberately does not assign tile addresses
//! or emit exchange rows. [`lower`] tries a set of legal operator plans,
//! prices them with a [`CostModel`], and inserts explicit precision casts and
//! layout rearrangements at format boundaries.

use crate::config::{
    AttentionStrategy, ConversionStreamingPolicy, HardwareMemoryConstraints, OperatorClass,
    PipelineConfig, PlannerSearchDomain,
};
use crate::cost::MemoizedCostModel;
pub use crate::cost::{CostModel, IPU21_TARGET_COSTS, Ipu21CostModel, Ipu21TargetCosts};
use crate::estimate::{
    conversion_memory_estimate, operator_memory_estimate, region_peak_memory,
    region_peak_memory_with_multiplicity,
};
use crate::graph::{
    ComputeGraph, GemmOptions, GraphInputKind, Operation, OperationId, OperationKind, Repeat,
    TensorShape, ValueId,
};
use crate::ir::{
    MidGraph, MidInput, MidOperation, MidOperationKind, MidRegion, MidRepeat, MidValue, MidValueId,
};
use crate::layout::{
    AMP_COLUMN_MICRO, AMP_INNER_BLOCK, AMP_NARROW_OUTPUT_COLUMN_BLOCK, AMP_OUTPUT_COLUMN_BLOCK,
    AMP_WIDE_OUTPUT_COLUMN_BLOCK, AmpOrder, BlockMajorOrder, ElementOrder, Layout, MemoryClass,
    Padding, TensorAxis, TensorFormat, TensorType,
};
pub use crate::metrics::{CostEstimate, ExchangeFootprint};
use crate::metrics::{MemoryEstimate, MemoryPeaks, MemoryUsage, OperationMetrics, RegionMetrics};
use crate::operator::*;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

fn gemm_seed_plans_for_tile_count(
    options: GemmOptions,
    tile_count: u16,
    domain: &PlannerSearchDomain,
) -> Vec<OperatorPlan> {
    let mut candidates = (1..=tile_count)
        .rev()
        .filter(|columns| tile_count.is_multiple_of(*columns))
        .flat_map(|columns| {
            let rows = tile_count / columns;
            let result = GemmResultGrid { rows, columns };
            let order = GridOrder::ColumnsFast;
            let geometry = GemmGeometry {
                block: GemmBlockShape {
                    inner: AMP_INNER_BLOCK,
                    output_columns: AMP_OUTPUT_COLUMN_BLOCK,
                },
                orientation: GemmOrientation::Normal,
                result,
                order,
                distribution: GemmDistribution::OutputStationary,
            };
            let mut grid = Vec::new();
            let mut placements = Vec::new();
            for &precision in domain.precisions(OperatorClass::Gemm) {
                let Some(left_tail) = gemm_left_access_tail(precision) else {
                    continue;
                };
                for &memory_class in &domain.weight_memory_classes {
                    if gemm_supports_weight_memory(precision, memory_class) {
                        placements.push((
                            precision,
                            left_tail,
                            AmpWeightPlacement::resident(memory_class),
                        ));
                    }
                }
            }
            if rows > 1 {
                for &precision in domain.precisions(OperatorClass::Gemm) {
                    let Some(left_tail) = gemm_left_access_tail(precision) else {
                        continue;
                    };
                    for &memory_class in &domain.weight_memory_classes {
                        if gemm_supports_weight_memory(precision, memory_class) {
                            placements.push((
                                precision,
                                left_tail,
                                AmpWeightPlacement::sharded(rows, memory_class),
                            ));
                        }
                    }
                }
            }
            // Two-way F16 interleaving lets each peer retain half of a full
            // kernel-width column shard. Keep the automatic search bounded;
            // explicit layouts may use any divisor of the row grid.
            if rows > 2
                && rows.is_multiple_of(2)
                && domain.permits_precision(OperatorClass::Gemm, Precision::F16)
                && domain.permits_weight_memory(MemoryClass::Interleaved)
            {
                placements.push((
                    Precision::F16,
                    16,
                    AmpWeightPlacement::sharded(2, MemoryClass::Interleaved),
                ));
            }
            for (precision, left_tail, weights) in placements {
                for &output_columns in amp_output_column_blocks(precision) {
                    // A narrow resident interleaved shard can avoid streaming
                    // when a 64-column shard would exceed region capacity.
                    // Narrow streamed panels increase multicast-role pressure
                    // and are not offered until ownership is part of the cost.
                    if output_columns < AMP_OUTPUT_COLUMN_BLOCK
                        && !(weights.inner_partitions == 1
                            && weights.memory_class == MemoryClass::Interleaved)
                    {
                        continue;
                    }
                    let mut geometry = geometry;
                    geometry.block.output_columns = output_columns;
                    let candidate =
                        amp_grid_gemm_plan(options, precision, left_tail, geometry, weights);
                    grid.push(candidate.clone());
                    if precision == Precision::F16 && weights.memory_class == MemoryClass::Standard
                    {
                        let mut staged = candidate;
                        staged.requirements.inputs[1].local_staging =
                            LocalOperandStaging::MatchRemote;
                        grid.push(staged);
                    }
                }
            }
            grid
        })
        .collect::<Vec<_>>();
    if domain.permits_weight_memory(MemoryClass::Standard) {
        for &precision in domain.precisions(OperatorClass::Gemm) {
            let Some(left_tail) = gemm_left_access_tail(precision) else {
                continue;
            };
            for &output_columns in amp_output_column_blocks(precision)
                .iter()
                .filter(|&&columns| columns >= AMP_OUTPUT_COLUMN_BLOCK)
            {
                candidates.push(amp_gemm_plan(
                    options,
                    precision,
                    64,
                    left_tail,
                    output_columns,
                    tile_count,
                ));
            }
        }
    }
    let mut unique = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if !unique.contains(&candidate) {
            unique.push(candidate);
        }
    }
    unique
}

const fn gemm_left_access_tail(precision: Precision) -> Option<u32> {
    match precision {
        Precision::F16 => Some(16),
        Precision::F32 => Some(32),
        Precision::F8F143 { .. } => None,
    }
}

const fn gemm_supports_weight_memory(precision: Precision, memory_class: MemoryClass) -> bool {
    matches!(
        (precision, memory_class),
        (Precision::F16, _) | (Precision::F32, MemoryClass::Standard)
    )
}

fn amp_gemm_plan(
    options: GemmOptions,
    precision: Precision,
    inner: u16,
    left_tail: u32,
    output_columns: u32,
    tile_count: u16,
) -> OperatorPlan {
    let operator = MidOperator::Gemm {
        options,
        multiply: precision,
        accumulate: gemm_accumulation_precision(precision),
    };
    OperatorPlan::candidate(
        operator,
        blocked_gemm_dispatch(
            operator,
            GemmGeometry {
                block: GemmBlockShape {
                    inner: u32::from(inner),
                    output_columns,
                },
                orientation: GemmOrientation::Normal,
                result: GemmResultGrid {
                    rows: tile_count,
                    columns: 1,
                },
                order: GridOrder::ColumnsFast,
                distribution: GemmDistribution::OutputStationary,
            },
        ),
        OperatorRequirements {
            inputs: vec![
                OperandRequirement::new(
                    TensorFormat {
                        precision,
                        layout: Layout::amp_left(inner, tile_count),
                    },
                    32,
                )
                .with_access_tail(left_tail)
                .with_materialization(OperandMaterialization::DispatchSlices),
                OperandRequirement::new(
                    TensorFormat {
                        precision,
                        layout: Layout::block_major_matrix_storage(
                            inner,
                            AMP_OUTPUT_COLUMN_BLOCK,
                            tile_count,
                            1,
                            1,
                            MemoryClass::Standard,
                        ),
                    },
                    32,
                ),
            ],
            output: OperandRequirement::new(
                TensorFormat {
                    precision,
                    layout: if precision == Precision::F16 {
                        Layout::amp_left_result(tile_count)
                    } else {
                        Layout::amp_output(tile_count)
                    },
                },
                32,
            ),
            output_aliasing: OutputAliasing::Fresh,
            memory_space: MemorySpaceRequirements::default()
                .with_distinct_elements([MemoryOperand::Output, MemoryOperand::Input(0)]),
        },
        None,
    )
}

fn amp_grid_gemm_plan(
    options: GemmOptions,
    precision: Precision,
    left_tail: u32,
    geometry: GemmGeometry,
    weights: AmpWeightPlacement,
) -> OperatorPlan {
    let inner = u16::try_from(geometry.block.inner).unwrap_or(0);
    let output_columns = geometry.block.output_columns;
    let grid = geometry.result;
    let right_layout = match (weights.inner_partitions, weights.memory_class) {
        (1, MemoryClass::Standard) => Layout::block_major_matrix_grid(
            inner,
            output_columns,
            grid.tile_count(),
            grid.rows,
            grid.columns,
            geometry.order,
        ),
        (inner_partitions, memory_class) => Layout::block_major_matrix_storage(
            inner,
            output_columns,
            grid.columns,
            inner_partitions,
            grid.rows / inner_partitions,
            memory_class,
        ),
    };
    let operator = MidOperator::Gemm {
        options,
        multiply: precision,
        accumulate: gemm_accumulation_precision(precision),
    };
    OperatorPlan::candidate(
        operator,
        blocked_gemm_dispatch(operator, geometry),
        OperatorRequirements {
            inputs: vec![
                OperandRequirement::new(
                    TensorFormat {
                        precision,
                        layout: Layout::amp_left_grid(
                            inner,
                            grid.tile_count(),
                            grid.rows,
                            grid.columns,
                            geometry.order,
                        ),
                    },
                    32,
                )
                .with_access_tail(left_tail)
                .with_materialization(OperandMaterialization::DispatchSlices),
                OperandRequirement::new(
                    TensorFormat {
                        precision,
                        layout: right_layout,
                    },
                    32,
                ),
            ],
            output: OperandRequirement::new(
                TensorFormat {
                    precision,
                    layout: if precision == Precision::F16 {
                        Layout::amp_left_result_grid(
                            output_columns,
                            grid.tile_count(),
                            grid.rows,
                            grid.columns,
                            geometry.order,
                        )
                    } else {
                        Layout::amp_output_grid(
                            output_columns,
                            grid.tile_count(),
                            grid.rows,
                            grid.columns,
                            geometry.order,
                        )
                    },
                },
                32,
            ),
            output_aliasing: OutputAliasing::Fresh,
            memory_space: MemorySpaceRequirements::default()
                .with_distinct_elements([MemoryOperand::Output, MemoryOperand::Input(0)]),
        },
        None,
    )
}

fn blocked_gemm_dispatch(operator: MidOperator, geometry: GemmGeometry) -> OperatorDispatch {
    let MidOperator::Gemm {
        multiply,
        accumulate,
        ..
    } = operator
    else {
        unreachable!("blocked GEMM dispatch requires a GEMM operator")
    };
    OperatorDispatch::BlockedGemm(BlockedGemmPlan {
        kernel: GemmKernelFamily {
            multiply,
            accumulate,
            weights: GemmWeightLoad::Standard,
        },
        geometry,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct AmpWeightPlacement {
    inner_partitions: u16,
    memory_class: MemoryClass,
}

impl AmpWeightPlacement {
    const fn resident(memory_class: MemoryClass) -> Self {
        Self::sharded(1, memory_class)
    }

    const fn sharded(inner_partitions: u16, memory_class: MemoryClass) -> Self {
        Self {
            inner_partitions,
            memory_class,
        }
    }
}

fn amp_output_column_blocks(precision: Precision) -> &'static [u32] {
    match precision {
        Precision::F16 => &[
            AMP_OUTPUT_COLUMN_BLOCK,
            AMP_WIDE_OUTPUT_COLUMN_BLOCK,
            AMP_NARROW_OUTPUT_COLUMN_BLOCK,
        ],
        Precision::F32 | Precision::F8F143 { .. } => &[AMP_OUTPUT_COLUMN_BLOCK],
    }
}

const fn gemm_accumulation_precision(precision: Precision) -> AccumulationPrecision {
    match precision {
        Precision::F16 | Precision::F8F143 { .. } => AccumulationPrecision::F16,
        Precision::F32 => AccumulationPrecision::F32,
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LoweringError {
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

#[tracing::instrument(
    name = "ipu_codegen.mid.lower",
    skip(graph, config, costs),
    fields(tile_count = config.tile_count, operations = graph.operations().len())
)]
pub fn lower(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
) -> LoweringResult<MidGraph> {
    Ok(lower_finalists(graph, config, costs, 1)?.remove(0))
}

pub(crate) fn lower_finalists(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
    finalist_count: usize,
) -> LoweringResult<Vec<MidGraph>> {
    if config.tile_count == 0 {
        return Err(LoweringError::EmptyTileGroup);
    }
    let active_tile_counts = config
        .search_domain
        .active_tile_counts(config.tile_count, graph.value_shapes().values());
    let mut resolved_config = config.clone();
    resolved_config.resolved_active_tile_counts = active_tile_counts.clone();
    let config = &resolved_config;
    let mut state = LoweringState::default();
    let costs = MemoizedCostModel::new(costs, config.tile_count);
    let mut values = BTreeMap::new();
    let mut inputs = Vec::with_capacity(graph.inputs().len());
    for input in graph.inputs() {
        let (format, automatic) = if let Some(format) = config.inputs.get(&input.value) {
            (format.clone(), false)
        } else if let Some(&precision) = config.automatic_inputs.get(&input.value) {
            (
                TensorFormat {
                    precision,
                    layout: Layout::row_sharded(config.tile_count),
                },
                true,
            )
        } else {
            return Err(LoweringError::MissingInputType(input.value));
        };
        let tensor_type = TensorType {
            shape: input.shape.clone(),
            format,
        };
        let value = state.value(input.value, tensor_type);
        if input.kind == GraphInputKind::Parameter {
            state.parameter_values.insert(value);
        }
        if automatic {
            state.automatic_inputs.insert(value);
        }
        values.insert(input.value, value);
        inputs.push(MidInput {
            name: input.name.clone(),
            kind: input.kind,
            value,
        });
    }
    let branches = plan_region_frontier(
        graph.operations(),
        graph.outputs(),
        &mut values,
        graph.value_shapes(),
        graph,
        config,
        &costs,
        &mut state,
        &RegionPlanningConstraints::default(),
    )?;
    let initial = inputs.iter().map(|input| input.value).collect::<Vec<_>>();
    branches
        .into_iter()
        .take(finalist_count.max(1))
        .enumerate()
        .map(|(finalist, branch)| {
            let outputs = graph
                .outputs()
                .iter()
                .map(|value| lookup(&branch.values, *value))
                .collect::<LoweringResult<Vec<_>>>()?;
            let estimated_cycles = branch
                .operations
                .iter()
                .map(|operation| operation.metrics.cost.cycles)
                .sum::<u64>();
            let estimated_exchange_cycles = branch
                .operations
                .iter()
                .map(|operation| operation.metrics.cost.exchange_cycles)
                .sum::<u64>();
            let cost = branch
                .operations
                .iter()
                .fold(CostEstimate::default(), |cost, operation| {
                    cost.sequence(operation.metrics.cost)
                });
            let peak_memory = region_peak_memory(
                &initial,
                &branch.operations,
                &outputs,
                &branch.state.values,
                config.target.memory_constraints(),
            );
            tracing::info!(
                finalist,
                values = branch.state.values.len(),
                operations = branch.operations.len(),
                estimated_cycles,
                estimated_exchange_cycles,
                exchange_row_bytes = peak_memory.exchange_rows,
                peak_standard_bytes = peak_memory.standard,
                peak_interleaved_bytes = peak_memory.interleaved,
                peak_total_bytes = peak_memory.total,
                maximum_standard_allocation_bytes = peak_memory.maximum_standard_allocation,
                active_tile_counts = ?branch.operations
                    .iter()
                    .filter_map(|operation| operation.results.first())
                    .map(|result| branch.state.values[result.index() as usize]
                        .tensor_type
                        .format
                        .layout
                        .tiling
                        .tile_count)
                    .collect::<BTreeSet<_>>(),
                padding_group_counts = ?branch.operations
                    .iter()
                    .flat_map(|operation| operation.results.iter())
                    .flat_map(|result| branch.state.values[result.index() as usize]
                        .tensor_type
                        .format
                        .layout
                        .tiling
                        .axes
                        .iter()
                        .map(|axis| axis.padding_groups))
                    .filter(|groups| *groups > 1)
                    .collect::<BTreeSet<_>>(),
                conversion_sources = ?branch.operations
                    .iter()
                    .filter(|operation| operation.conversion_plan.is_some())
                    .map(|operation| operation.source)
                    .collect::<Vec<_>>(),
                "retained operator-plan finalist"
            );
            tracing::debug!(
                finalist,
                plans = ?branch.operations
                    .iter()
                    .filter_map(|operation| operation.operator_plan().map(|plan| (
                        operation.source,
                        &plan.dispatch,
                        plan.requirements.inputs.iter().map(|input| &input.format.layout).collect::<Vec<_>>(),
                        &plan.requirements.output.format.layout,
                        operation.metrics.cost.cycles,
                        operation.metrics.cost.exchange_cycles,
                    )))
                    .collect::<Vec<_>>(),
                conversions = ?branch.operations
                    .iter()
                    .filter_map(|operation| operation.conversion_plan.as_ref().map(|plan| (
                        operation.source,
                        &plan.input.format.layout,
                        &plan.output.format.layout,
                        operation.metrics.cost.cycles,
                        operation.metrics.cost.exchange_cycles,
                    )))
                    .collect::<Vec<_>>(),
                "retained operator-plan details"
            );
            Ok(MidGraph {
                inputs: inputs.clone(),
                values: branch.state.values,
                operations: branch.operations,
                outputs,
                metrics: RegionMetrics {
                    cost,
                    memory: peak_memory,
                },
            })
        })
        .collect()
}

#[derive(Clone, Default)]
struct LoweringState {
    values: Vec<MidValue>,
    automatic_inputs: BTreeSet<MidValueId>,
    parameter_values: BTreeSet<MidValueId>,
}

impl LoweringState {
    fn value(&mut self, origin: ValueId, tensor_type: TensorType) -> MidValueId {
        let id = MidValueId::from_index(self.values.len() as u32);
        self.values.push(MidValue {
            id,
            tensor_type,
            origin,
            storage_group: id,
        });
        id
    }

    fn value_in_storage_group(
        &mut self,
        origin: ValueId,
        tensor_type: TensorType,
        storage_group: MidValueId,
    ) -> MidValueId {
        let result = self.value(origin, tensor_type);
        self.values[result.index() as usize].storage_group = storage_group;
        result
    }

    fn get(&self, id: MidValueId) -> &MidValue {
        &self.values[id.index() as usize]
    }

    fn derived_value(&mut self, source: MidValueId, tensor_type: TensorType) -> MidValueId {
        let origin = self.get(source).origin;
        let storage_group = self.get(source).storage_group;
        let result = self.value_in_storage_group(origin, tensor_type, storage_group);
        if self.parameter_values.contains(&source) {
            self.parameter_values.insert(result);
        }
        result
    }

    fn retarget_automatic_input(&mut self, id: MidValueId, layout: Layout) -> bool {
        if !self.automatic_inputs.remove(&id) {
            return false;
        }
        self.values[id.index() as usize].tensor_type.format.layout = layout;
        true
    }
}

#[derive(Clone)]
struct BeamBranch {
    values: BTreeMap<ValueId, MidValueId>,
    state: LoweringState,
    operations: Vec<MidOperation>,
    peak_memory: MemoryPeaks,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FutureValueState {
    origin: ValueId,
    tensor_type: TensorType,
    automatic_input: bool,
    parameter: bool,
    allocation_copies: u32,
    storage_class: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FutureDeferredState {
    origin: ValueId,
    source_type: TensorType,
    source_automatic_input: bool,
    source_parameter: bool,
    source_storage_class: u32,
    transform: DeferredTransform,
    unfused_cost: CostEstimate,
    claimed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FutureBeamState {
    values: Vec<FutureValueState>,
    deferred: Vec<FutureDeferredState>,
    equal_formats_satisfied: Vec<(ValueId, ValueId, bool)>,
}

struct RankedBeamBranch {
    branch: BeamBranch,
    objective: RegionMetrics,
    compatibility: FutureFormatCompatibility,
    order: usize,
}

type FutureFormatCompatibility = Vec<(
    ValueId,
    FutureFormatRole,
    Precision,
    ElementOrderCompatibility,
    MemoryClass,
    Vec<(TensorAxis, u16, u32)>,
)>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum FutureFormatRole {
    Value,
    DeferredSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ElementOrderCompatibility {
    RowMajor,
    BlockMajorMatrix,
    BlockMajorTransposedMatrix,
    Amp(AmpOrder),
}

fn element_order_compatibility(order: ElementOrder) -> ElementOrderCompatibility {
    match order {
        ElementOrder::RowMajor => ElementOrderCompatibility::RowMajor,
        ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. }) => {
            ElementOrderCompatibility::BlockMajorMatrix
        }
        ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix { .. }) => {
            ElementOrderCompatibility::BlockMajorTransposedMatrix
        }
        ElementOrder::Amp(order) => ElementOrderCompatibility::Amp(order),
    }
}

fn future_format_compatibility(
    branch: &BeamBranch,
    future_origins: &BTreeSet<ValueId>,
) -> FutureFormatCompatibility {
    let mut formats = Vec::new();
    for &origin in future_origins {
        let Some(&id) = branch.values.get(&origin) else {
            continue;
        };
        let mut add = |role, format: &TensorFormat| {
            let axes = format
                .layout
                .tiling
                .axes
                .iter()
                .map(|axis| (axis.axis, axis.padding_groups, axis.shard_padding_multiple))
                .collect();
            formats.push((
                origin,
                role,
                format.precision,
                element_order_compatibility(format.layout.order),
                format.layout.memory_class,
                axes,
            ));
        };
        add(
            FutureFormatRole::Value,
            &branch.state.get(id).tensor_type.format,
        );
        let deferred_source = branch
            .operations
            .iter()
            .rev()
            .find(|operation| operation.results.first() == Some(&id))
            .and_then(|operation| {
                let offer = operation.operator_plan()?.deferred_output?;
                operation.inputs.get(offer.source_input).copied()
            });
        if let Some(source) = deferred_source {
            add(
                FutureFormatRole::DeferredSource,
                &branch.state.get(source).tensor_type.format,
            );
        }
    }
    formats
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PlanCacheKey {
    input_shapes: Vec<TensorShape>,
    parameter_inputs: Vec<bool>,
    format_sensitive_inputs: Vec<(usize, TensorFormat)>,
}

#[derive(Default)]
struct RegionPlanningConstraints {
    /// Number of simultaneously resident blocks represented by a region value.
    allocation_copies: BTreeMap<ValueId, u32>,
    /// Value pairs whose formats must agree at a structured-region boundary.
    required_equal_formats: Vec<(ValueId, ValueId)>,
}

#[allow(clippy::too_many_arguments)]
fn plan_region_frontier(
    source: &[Operation],
    required_outputs: &[ValueId],
    values: &mut BTreeMap<ValueId, MidValueId>,
    shapes: &BTreeMap<ValueId, TensorShape>,
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
    state: &mut LoweringState,
    constraints: &RegionPlanningConstraints,
) -> LoweringResult<Vec<BeamBranch>> {
    if source.is_empty() {
        return Ok(vec![BeamBranch {
            values: values.clone(),
            state: state.clone(),
            operations: Vec::new(),
            peak_memory: MemoryPeaks::default(),
        }]);
    }
    let relevant_origins = source
        .iter()
        .flat_map(|operation| operation_graph_inputs(operation, graph))
        .chain(required_outputs.iter().copied())
        .chain(
            constraints
                .required_equal_formats
                .iter()
                .flat_map(|pair| [pair.0, pair.1]),
        )
        .chain(constraints.allocation_copies.keys().copied())
        .collect::<BTreeSet<_>>();
    let initial = relevant_origins
        .iter()
        .filter_map(|origin| values.get(origin).copied())
        .collect::<Vec<_>>();
    let mut value_uses = BTreeMap::<ValueId, usize>::new();
    for value in source
        .iter()
        .flat_map(|operation| operation_graph_inputs(operation, graph))
        .chain(required_outputs.iter().copied())
    {
        *value_uses.entry(value).or_default() += 1;
    }
    let mut beam = vec![BeamBranch {
        values: values.clone(),
        state: state.clone(),
        operations: Vec::new(),
        peak_memory: MemoryPeaks::default(),
    }];
    for (operation_index, operation) in source.iter().enumerate() {
        let distributed_result_is_useful = operation.results.first().is_some_and(|result| {
            required_outputs.contains(result)
                || (value_uses.get(result).copied() == Some(1)
                    && source[operation_index + 1..]
                        .iter()
                        .find(|consumer| consumer.inputs.contains(result))
                        .is_some_and(|consumer| {
                            consumer.inputs.iter().enumerate().any(|(index, input)| {
                                input == result
                                    && operator_accepts_input_layout(&consumer.kind, index, config)
                            })
                        }))
        });
        let mut expanded = Vec::new();
        let mut rejected_memory = Vec::new();
        let mut saw_legal_plan = false;
        let format_sensitive_indices = operation
            .inputs
            .iter()
            .enumerate()
            .filter_map(|(index, _)| {
                operator_accepts_input_layout(&operation.kind, index, config).then_some(index)
            })
            .collect::<BTreeSet<_>>();
        let mut plan_cache = BTreeMap::<PlanCacheKey, Vec<OperatorPlan>>::new();
        let mut plan_cache_hits = 0usize;
        let mut generated_plan_sets = 0usize;
        for branch in beam {
            if let OperationKind::Repeat(repeat) = &operation.kind {
                saw_legal_plan = true;
                let mut next = branch.clone();
                lower_repeat(
                    operation,
                    repeat,
                    &mut next.values,
                    graph,
                    config,
                    costs,
                    &mut next.state,
                    &mut next.operations,
                )?;
                let peak = beam_memory_peak(
                    &next,
                    &initial,
                    source,
                    operation_index,
                    required_outputs,
                    graph,
                    &constraints.allocation_copies,
                    config.target.memory_constraints(),
                );
                if peak.fits_with_budget(
                    config.target.memory_constraints(),
                    config.standard_memory_reservation_bytes,
                    config.tile_memory_budget_bytes,
                ) {
                    next.peak_memory = peak;
                    expanded.push(next);
                } else {
                    tracing::trace!(
                        operation = operation.id.index(),
                        standard = peak.standard,
                        interleaved = peak.interleaved,
                        total = peak.total,
                        contiguous_overflow = peak.standard_contiguous_overflow_with_reservation(
                            config.target.memory_constraints(),
                            config.standard_memory_reservation_bytes,
                        ),
                        plan = ?next.operations.last().and_then(|operation| operation.operator_plan()),
                        "rejected planning branch for memory"
                    );
                    rejected_memory.push(peak);
                }
                continue;
            }
            let input_ids = operation
                .inputs
                .iter()
                .map(|value| lookup(&branch.values, *value))
                .collect::<LoweringResult<Vec<_>>>()?;
            let input_types = input_ids
                .iter()
                .map(|value| branch.state.get(*value).tensor_type.clone())
                .collect::<Vec<_>>();
            if matches!(operation.kind, OperationKind::Gemm(_))
                && input_types.get(1).is_some_and(|right| {
                    right.shape.0[..right.shape.0.len().saturating_sub(2)]
                        .iter()
                        .any(|&extent| extent != 1)
                })
            {
                return Err(LoweringError::UnsupportedGemmBatching(operation.id));
            }
            let output_shape = shapes
                .get(&operation.results[0])
                .cloned()
                .ok_or(LoweringError::MissingShape(operation.results[0]))?;
            let grouped_output = grouped_output_layout(
                source,
                operation_index,
                operation,
                &output_shape,
                &value_uses,
            );
            let direct_consumer_layouts = direct_consumer_layouts(
                source,
                operation_index,
                operation.results[0],
                &output_shape,
                config,
            );
            let parameter_inputs = input_ids
                .iter()
                .map(|id| branch.state.parameter_values.contains(id))
                .collect::<Vec<_>>();
            let cache_key = PlanCacheKey {
                input_shapes: input_types
                    .iter()
                    .map(|input| input.shape.clone())
                    .collect(),
                parameter_inputs: parameter_inputs.clone(),
                format_sensitive_inputs: format_sensitive_indices
                    .iter()
                    .filter_map(|&index| {
                        input_types
                            .get(index)
                            .map(|input| (index, input.format.clone()))
                    })
                    .collect(),
            };
            let cached = if let Some(cached) = plan_cache.get(&cache_key) {
                plan_cache_hits += 1;
                cached
            } else {
                generated_plan_sets += 1;
                let generated = plans_for_operation(
                    operation,
                    &input_types,
                    &parameter_inputs,
                    &output_shape,
                    config,
                    costs,
                    distributed_result_is_useful,
                    grouped_output,
                    &direct_consumer_layouts,
                );
                plan_cache.entry(cache_key).or_insert(generated)
            };
            let available_plans = cached
                .iter()
                .cloned()
                .into_iter()
                .filter(|plan| {
                    input_ids
                        .iter()
                        .zip(&plan.requirements.inputs)
                        .all(|(id, requirement)| {
                            let current = &branch.state.get(*id).tensor_type.format.layout;
                            branch.state.automatic_inputs.contains(id)
                                || current.order == requirement.format.layout.order
                                || !requirement.format.layout.order.requires_direct_population()
                                || (current.order == ElementOrder::RowMajor
                                    && requirement
                                        .format
                                        .layout
                                        .order
                                        .supports_row_major_population())
                        })
                })
                .collect::<Vec<_>>();
            let available_plans = available_plans
                .into_iter()
                .flat_map(|plan| {
                    let mut complete = plan.clone();
                    for requirement in &mut complete.requirements.inputs {
                        requirement.materialization = OperandMaterialization::Complete;
                    }
                    match config.conversion_streaming {
                        ConversionStreamingPolicy::Never => vec![complete],
                        ConversionStreamingPolicy::Always => vec![plan],
                        ConversionStreamingPolicy::WhenRequired if complete == plan => {
                            vec![complete]
                        }
                        ConversionStreamingPolicy::WhenRequired => vec![complete, plan],
                    }
                })
                .collect::<Vec<_>>();
            saw_legal_plan |= !available_plans.is_empty();
            let evaluated = available_plans
                .into_par_iter()
                .map(|plan| {
                    let mut next = branch.clone();
                    apply_selected_plan(
                        operation,
                        output_shape.clone(),
                        plan,
                        &operation
                            .inputs
                            .iter()
                            .map(|value| value_uses.get(value).copied().unwrap_or(0) == 1)
                            .collect::<Vec<_>>(),
                        costs,
                        &mut next.values,
                        &mut next.state,
                        &mut next.operations,
                    );
                    let peak = beam_memory_peak(
                        &next,
                        &initial,
                        source,
                        operation_index,
                        required_outputs,
                        graph,
                        &constraints.allocation_copies,
                        config.target.memory_constraints(),
                    );
                    (next, peak)
                })
                .collect::<Vec<_>>();
            for (mut next, peak) in evaluated {
                let fits = peak.fits_with_budget(
                    config.target.memory_constraints(),
                    config.standard_memory_reservation_bytes,
                    config.tile_memory_budget_bytes,
                );
                if fits || branch_contains_gemm_constraint(&next, config) {
                    if !fits {
                        tracing::debug!(
                            operation = operation.id.index(),
                            standard = peak.standard,
                            interleaved = peak.interleaved,
                            total = peak.total,
                            "retained constrained GEMM past conservative memory estimate"
                        );
                    }
                    next.peak_memory = peak;
                    expanded.push(next);
                } else {
                    tracing::trace!(
                        operation = operation.id.index(),
                        standard = peak.standard,
                        interleaved = peak.interleaved,
                        total = peak.total,
                        contiguous_overflow = peak.standard_contiguous_overflow_with_reservation(
                            config.target.memory_constraints(),
                            config.standard_memory_reservation_bytes,
                        ),
                        plan = ?next.operations.last().and_then(|operation| operation.operator_plan()),
                        "rejected planning branch for memory"
                    );
                    rejected_memory.push(peak);
                }
            }
        }
        if expanded.is_empty() {
            if saw_legal_plan
                && let Some(peak) = rejected_memory
                    .into_iter()
                    .min_by_key(|peak| (peak.total, peak.interleaved, peak.standard))
            {
                return Err(LoweringError::InsufficientMemory {
                    operation: operation.id,
                    standard: peak.standard,
                    standard_reservation: config.standard_memory_reservation_bytes,
                    interleaved: peak.interleaved,
                    total: peak.total,
                    standard_contiguous_overflow: peak
                        .standard_contiguous_overflow_with_reservation(
                            config.target.memory_constraints(),
                            config.standard_memory_reservation_bytes,
                        ),
                });
            }
            return Err(LoweringError::NoCandidate(operation.id));
        }
        let future_origins = source[operation_index + 1..]
            .iter()
            .flat_map(|operation| operation_graph_inputs(operation, graph))
            .chain(required_outputs.iter().copied())
            .chain(
                constraints
                    .required_equal_formats
                    .iter()
                    .flat_map(|pair| [pair.0, pair.1]),
            )
            .chain(constraints.allocation_copies.keys().copied())
            .collect::<BTreeSet<_>>();
        let expanded_count = expanded.len();
        let (expanded, dominated, equivalent, diversity) = retain_pareto_beam(
            expanded,
            &future_origins,
            constraints,
            costs,
            config.planning_beam_width.max(1),
        );
        tracing::debug!(
            operation = operation.id.index(),
            retained = expanded.len(),
            expanded = expanded_count,
            pareto_dominated = dominated,
            equivalent,
            diversity_representatives = diversity,
            best_cycles = deferred_aware_branch_score(&expanded[0], &future_origins),
            generated_plan_sets,
            plan_cache_hits,
            "retained planning beam"
        );
        beam = expanded;
    }
    let final_operation = source.len().saturating_sub(1);
    let beam = beam
        .into_iter()
        .filter_map(|mut branch| {
            let peak = beam_memory_peak(
                &branch,
                &initial,
                source,
                final_operation,
                required_outputs,
                graph,
                &constraints.allocation_copies,
                config.target.memory_constraints(),
            );
            (peak.fits_with_budget(
                config.target.memory_constraints(),
                config.standard_memory_reservation_bytes,
                config.tile_memory_budget_bytes,
            ) || branch_contains_gemm_constraint(&branch, config))
            .then(|| {
                branch.peak_memory = peak;
                branch
            })
        })
        .collect::<Vec<_>>();
    let mut beam = beam;
    beam.sort_by_key(|branch| {
        deferred_aware_branch_score(branch, &BTreeSet::new()).saturating_add(format_equality_cost(
            branch,
            &constraints.required_equal_formats,
            costs,
        ))
    });
    if beam.is_empty() {
        return Err(LoweringError::NoCandidate(source[0].id));
    }
    for branch in &mut beam {
        restore_unclaimed_deferred_costs(&mut branch.operations);
    }
    Ok(beam)
}

fn branch_contains_gemm_constraint(branch: &BeamBranch, config: &PipelineConfig) -> bool {
    branch.operations.iter().any(|operation| {
        operation.source.is_some_and(|source| {
            config
                .search_domain
                .gemm_plan_constraints
                .iter()
                .any(|constraint| constraint.source_operation == source.index())
        })
    })
}

#[allow(clippy::too_many_arguments)]
fn lower_operations(
    source: &[Operation],
    required_outputs: &[ValueId],
    values: &mut BTreeMap<ValueId, MidValueId>,
    shapes: &BTreeMap<ValueId, TensorShape>,
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
    state: &mut LoweringState,
    constraints: &RegionPlanningConstraints,
) -> LoweringResult<Vec<MidOperation>> {
    let mut branches = plan_region_frontier(
        source,
        required_outputs,
        values,
        shapes,
        graph,
        config,
        costs,
        state,
        constraints,
    )?;
    let best = branches.remove(0);
    *values = best.values;
    *state = best.state;
    Ok(best.operations)
}

fn retain_pareto_beam(
    branches: Vec<BeamBranch>,
    future_origins: &BTreeSet<ValueId>,
    constraints: &RegionPlanningConstraints,
    costs: &impl CostModel,
    width: usize,
) -> (Vec<BeamBranch>, usize, usize, usize) {
    let mut groups = BTreeMap::<FutureBeamState, Vec<RankedBeamBranch>>::new();
    for (order, branch) in branches.into_iter().enumerate() {
        let signature = future_beam_state(&branch, future_origins, constraints);
        let objective = RegionMetrics {
            cost: CostEstimate {
                cycles: deferred_aware_branch_score(&branch, future_origins).saturating_add(
                    format_equality_cost(&branch, &constraints.required_equal_formats, costs),
                ),
                ..CostEstimate::default()
            },
            memory: branch.peak_memory,
        };
        groups.entry(signature).or_default().push(RankedBeamBranch {
            compatibility: future_format_compatibility(&branch, future_origins),
            branch,
            objective,
            order,
        });
    }

    let mut frontier = Vec::new();
    let mut dominated = 0usize;
    let mut equivalent = 0usize;
    for (_, candidates) in groups {
        let mut group_frontier = Vec::<RankedBeamBranch>::new();
        for candidate in candidates {
            if group_frontier
                .iter()
                .any(|kept| kept.objective == candidate.objective)
            {
                equivalent += 1;
                continue;
            }
            if group_frontier
                .iter()
                .any(|kept| kept.objective.dominates(candidate.objective))
            {
                dominated += 1;
                continue;
            }
            let before = group_frontier.len();
            group_frontier.retain(|kept| !candidate.objective.dominates(kept.objective));
            dominated += before - group_frontier.len();
            group_frontier.push(candidate);
        }
        frontier.extend(group_frontier);
    }
    frontier.sort_by_key(|candidate| (candidate.objective.cost.cycles, candidate.order));

    let mut selected = BTreeSet::new();
    let mut diversity = 0usize;
    if frontier.len() > width {
        // Preserve the cheapest representative of every live format family
        // before retaining secondary memory tradeoffs. Partition counts are
        // intentionally excluded: they are searched within a family, whereas
        // physical order and ownership axes determine which imminent
        // consumers can use a value without a qualitatively different
        // conversion.
        let mut represented = BTreeSet::new();
        for (index, entry) in frontier.iter().enumerate() {
            if selected.len() == width {
                break;
            }
            if represented.insert(entry.compatibility.clone()) && selected.insert(index) {
                diversity += 1;
            }
        }
        let objectives: [fn(&RankedBeamBranch) -> u64; 6] = [
            |entry: &RankedBeamBranch| entry.objective.memory.standard,
            |entry: &RankedBeamBranch| entry.objective.memory.interleaved,
            |entry: &RankedBeamBranch| entry.objective.memory.total,
            |entry: &RankedBeamBranch| entry.objective.memory.maximum_standard_allocation,
            |entry: &RankedBeamBranch| entry.objective.memory.standard_contiguous_overflow,
            |entry: &RankedBeamBranch| entry.objective.memory.exchange_rows,
        ];
        selected.insert(0);
        for objective in objectives {
            if selected.len() == width {
                break;
            }
            let index = frontier
                .iter()
                .enumerate()
                .min_by_key(|(index, entry)| {
                    (objective(entry), entry.objective.cost.cycles, *index)
                })
                .map(|(index, _)| index)
                .unwrap();
            if selected.insert(index) {
                diversity += 1;
            }
        }
        for index in 0..frontier.len() {
            if selected.len() == width {
                break;
            }
            selected.insert(index);
        }
    } else {
        selected.extend(0..frontier.len());
    }
    let mut retained = frontier
        .into_iter()
        .enumerate()
        .filter_map(|(index, entry)| selected.contains(&index).then_some(entry.branch))
        .collect::<Vec<_>>();
    retained.sort_by_cached_key(|branch| {
        deferred_aware_branch_score(branch, future_origins).saturating_add(format_equality_cost(
            branch,
            &constraints.required_equal_formats,
            costs,
        ))
    });
    (retained, dominated, equivalent, diversity)
}

fn future_beam_state(
    branch: &BeamBranch,
    future_origins: &BTreeSet<ValueId>,
    constraints: &RegionPlanningConstraints,
) -> FutureBeamState {
    let claims = deferred_claims(&branch.operations);
    let mut storage_classes = BTreeMap::<MidValueId, u32>::new();
    let mut next_storage_class = 0u32;
    let mut storage_class = |id: MidValueId| {
        let group = branch.state.get(id).storage_group;
        *storage_classes.entry(group).or_insert_with(|| {
            let class = next_storage_class;
            next_storage_class += 1;
            class
        })
    };
    let mut values = Vec::new();
    let mut deferred_sources = Vec::new();
    for &origin in future_origins {
        let Some(&id) = branch.values.get(&origin) else {
            continue;
        };
        values.push(FutureValueState {
            origin,
            tensor_type: branch.state.get(id).tensor_type.clone(),
            automatic_input: branch.state.automatic_inputs.contains(&id),
            parameter: branch.state.parameter_values.contains(&id),
            allocation_copies: constraints
                .allocation_copies
                .get(&origin)
                .copied()
                .unwrap_or(1),
            storage_class: storage_class(id),
        });
        let offer =
            branch.operations.iter().rev().find_map(|operation| {
                (operation.results.first() == Some(&id)).then_some(operation)
            });
        let Some((operation, offer)) = offer.and_then(|operation| {
            operation
                .operator_plan()
                .and_then(|plan| plan.deferred_output)
                .map(|offer| (operation, offer))
        }) else {
            continue;
        };
        let Some(&source) = operation.inputs.get(offer.source_input) else {
            continue;
        };
        deferred_sources.push((origin, id, source, offer));
    }
    let deferred = deferred_sources
        .into_iter()
        .map(|(origin, result, source, offer)| FutureDeferredState {
            origin,
            source_type: branch.state.get(source).tensor_type.clone(),
            source_automatic_input: branch.state.automatic_inputs.contains(&source),
            source_parameter: branch.state.parameter_values.contains(&source),
            source_storage_class: storage_class(source),
            transform: offer.transform,
            unfused_cost: offer.unfused_cost,
            claimed: claims.contains(&result),
        })
        .collect();
    let equal_formats_satisfied = constraints
        .required_equal_formats
        .iter()
        .map(|&(left, right)| {
            let satisfied = branch
                .values
                .get(&left)
                .zip(branch.values.get(&right))
                .is_some_and(|(&left, &right)| {
                    branch.state.get(left).tensor_type.format
                        == branch.state.get(right).tensor_type.format
                });
            (left, right, satisfied)
        })
        .collect();
    FutureBeamState {
        values,
        deferred,
        equal_formats_satisfied,
    }
}

fn deferred_claims(operations: &[MidOperation]) -> BTreeSet<MidValueId> {
    operations
        .iter()
        .filter_map(|operation| operation.operator_plan())
        .flat_map(|plan| plan.deferred_inputs.iter().flatten())
        .map(|input| input.producer)
        .collect()
}

fn deferred_aware_branch_score(
    branch: &BeamBranch,
    possible_future_consumers: &BTreeSet<ValueId>,
) -> u64 {
    let claims = deferred_claims(&branch.operations);
    branch.operations.iter().fold(0u64, |cycles, operation| {
        let pending = operation
            .operator_plan()
            .and_then(|plan| plan.deferred_output)
            .filter(|_| {
                operation.results.first().is_some_and(|result| {
                    !claims.contains(result)
                        && !possible_future_consumers.contains(&branch.state.get(*result).origin)
                })
            })
            .map_or(0, |offer| offer.unfused_cost.cycles);
        cycles
            .saturating_add(operation.metrics.cost.cycles)
            .saturating_add(pending)
    })
}

fn restore_unclaimed_deferred_costs(operations: &mut [MidOperation]) {
    let claims = deferred_claims(operations);
    for operation in operations {
        let Some(offer) = operation
            .operator_plan()
            .and_then(|plan| plan.deferred_output)
        else {
            continue;
        };
        if operation
            .results
            .first()
            .is_some_and(|result| !claims.contains(result))
        {
            operation.metrics.cost = offer.unfused_cost;
        }
    }
}

fn format_equality_cost(
    branch: &BeamBranch,
    equalities: &[(ValueId, ValueId)],
    costs: &impl CostModel,
) -> u64 {
    equalities.iter().fold(0u64, |total, &(source, target)| {
        let Some((&source, &target)) = branch.values.get(&source).zip(branch.values.get(&target))
        else {
            return total;
        };
        let source = &branch.state.get(source).tensor_type;
        let target = &branch.state.get(target).tensor_type;
        let cast = (source.format.precision != target.format.precision)
            .then(|| costs.cast_cycles(source, target.format.precision))
            .unwrap_or(0);
        let rearrange = (source.format.layout != target.format.layout)
            .then(|| {
                costs
                    .rearrangement_cost(
                        &source.shape,
                        target.format.precision,
                        layout_conversion_strategy(&source.format.layout, &target.format.layout),
                        &source.format.layout,
                        &target.format.layout,
                    )
                    .cycles
            })
            .unwrap_or(0);
        total.saturating_add(cast).saturating_add(rearrange)
    })
}

fn operation_graph_inputs(operation: &Operation, graph: &ComputeGraph) -> Vec<ValueId> {
    let mut inputs = operation.inputs.clone();
    if let OperationKind::Repeat(repeat) = &operation.kind {
        for sequence in &repeat.iterated_inputs {
            inputs.extend(&graph.sequences()[sequence.index() as usize].values);
        }
    }
    inputs
}

fn apply_selected_plan(
    operation: &Operation,
    output_shape: TensorShape,
    mut plan: OperatorPlan,
    single_use_inputs: &[bool],
    costs: &impl CostModel,
    values: &mut BTreeMap<ValueId, MidValueId>,
    state: &mut LoweringState,
    operations: &mut Vec<MidOperation>,
) {
    let input_ids = operation
        .inputs
        .iter()
        .map(|value| values[value])
        .collect::<Vec<_>>();
    let original_input_ids = input_ids.clone();
    let mut source_types = Vec::with_capacity(input_ids.len());
    let mut converted = Vec::with_capacity(input_ids.len());
    for (value, requirement) in input_ids.into_iter().zip(&plan.requirements.inputs) {
        let conversion_start = operations.len();
        let converted_value = ensure_format(
            value,
            requirement.format.clone(),
            requirement.materialization,
            operation.id,
            costs,
            state,
            operations,
        );
        let streamed_source = operations[conversion_start..]
            .last_mut()
            .and_then(|conversion| {
                let streamed = conversion.conversion_plan.as_ref().is_some_and(|plan| {
                    plan.output.materialization == OperandMaterialization::DispatchSlices
                });
                if streamed {
                    conversion.metrics.cost.cycles = 0;
                    conversion.metrics.cost.exchange_cycles = 0;
                    conversion.inputs.first().copied()
                } else {
                    None
                }
            });
        let source_value = streamed_source.unwrap_or(converted_value);
        source_types.push(state.get(source_value).tensor_type.clone());
        converted.push(converted_value);
    }
    let result = state.value(
        operation.results[0],
        TensorType {
            shape: output_shape,
            format: plan.requirements.output.format.clone(),
        },
    );
    let converted_types = converted
        .iter()
        .map(|value| state.get(*value).tensor_type.clone())
        .collect::<Vec<_>>();
    let mut operator_cycles = costs.operator_transition_cycles(
        plan.operator,
        &plan.dispatch,
        &plan.requirements,
        &source_types,
        &converted_types,
        &state.get(result).tensor_type,
    );
    let mut operator_exchange_cycles = costs.operator_transition_exchange_cycles(
        plan.operator,
        &plan.dispatch,
        &plan.requirements,
        &source_types,
        &converted_types,
        &state.get(result).tensor_type,
    );
    let mut deferred_inputs = vec![None; converted.len()];
    for (input_index, ((&original, &converted), requirement)) in original_input_ids
        .iter()
        .zip(&converted)
        .zip(&plan.requirements.inputs)
        .enumerate()
    {
        let conversion_is_streamed = original == converted
            || operations.iter().any(|candidate| {
                candidate.inputs.as_slice() == [original]
                    && candidate.results.as_slice() == [converted]
                    && candidate
                        .conversion_plan
                        .as_ref()
                        .is_some_and(|conversion| {
                            conversion.output.materialization
                                == OperandMaterialization::DispatchSlices
                        })
            });
        if !conversion_is_streamed
            || !single_use_inputs.get(input_index).copied().unwrap_or(false)
            || requirement.materialization != OperandMaterialization::DispatchSlices
        {
            continue;
        }
        let Some(producer_index) = operations
            .iter()
            .position(|candidate| candidate.results.as_slice() == [original])
        else {
            continue;
        };
        let Some(offered) = operations[producer_index]
            .operator_plan()
            .and_then(|producer| producer.deferred_output)
        else {
            continue;
        };
        let Some(&source) = operations[producer_index].inputs.get(offered.source_input) else {
            continue;
        };
        let producer_cycles = offered.unfused_cost.cycles;
        let fused_cycles = costs.deferred_input_cycles(
            offered.transform,
            &state.get(source).tensor_type,
            &state.get(original).tensor_type,
            &converted_types[input_index],
            &plan.dispatch,
            producer_cycles,
        );
        operator_cycles = operator_cycles.saturating_add(fused_cycles);
        operator_exchange_cycles =
            operator_exchange_cycles.saturating_add(costs.deferred_input_exchange_cycles(
                offered.transform,
                &state.get(source).tensor_type,
                &state.get(original).tensor_type,
                &converted_types[input_index],
                &plan.dispatch,
                producer_cycles,
            ));
        deferred_inputs[input_index] = Some(DeferredInputPlan {
            producer: original,
            source,
            transform: offered.transform,
        });
    }
    tracing::trace!(
        source = operation.id.index(),
        cycles = operator_cycles,
        dispatch = ?plan.dispatch,
        input_layouts = ?converted_types
            .iter()
            .map(|input| &input.format.layout)
            .collect::<Vec<_>>(),
        output_layout = ?state.get(result).tensor_type.format.layout,
        "costed operator plan"
    );
    let exchange = costs.operator_exchange_footprint(
        plan.operator,
        &plan.dispatch,
        &plan.requirements,
        &converted_types,
        &state.get(result).tensor_type,
    );
    let memory = operator_memory_estimate(
        &plan.dispatch,
        &plan.requirements,
        &converted_types,
        &state.get(result).tensor_type,
    );
    let mut deferred_output = plan.deferred_output;
    if let Some(offer) = &mut deferred_output {
        offer.unfused_cost = CostEstimate {
            cycles: operator_cycles,
            exchange_cycles: operator_exchange_cycles,
            exchange_footprint: exchange,
        };
        operator_cycles = 0;
        operator_exchange_cycles = 0;
    }
    plan.deferred_output = deferred_output;
    plan.deferred_inputs = deferred_inputs;
    operations.push(MidOperation {
        source: Some(operation.id),
        inputs: converted,
        results: vec![result],
        kind: MidOperationKind::Operator(plan),
        conversion_plan: None,
        metrics: OperationMetrics {
            cost: CostEstimate {
                cycles: operator_cycles,
                exchange_cycles: operator_exchange_cycles,
                exchange_footprint: exchange,
            },
            memory,
        },
    });
    values.insert(operation.results[0], result);
}

fn beam_memory_peak(
    branch: &BeamBranch,
    initial: &[MidValueId],
    source: &[Operation],
    operation_index: usize,
    required_outputs: &[ValueId],
    graph: &ComputeGraph,
    allocation_multiplicity: &BTreeMap<ValueId, u32>,
    memory_constraints: HardwareMemoryConstraints,
) -> MemoryPeaks {
    let live_origins = source[operation_index + 1..]
        .iter()
        .flat_map(|operation| operation_graph_inputs(operation, graph))
        .chain(required_outputs.iter().copied())
        .collect::<BTreeSet<_>>();
    let live = live_origins
        .iter()
        .filter_map(|origin| branch.values.get(origin).copied())
        .collect::<Vec<_>>();
    let multiplicity = branch
        .state
        .values
        .iter()
        .filter_map(|value| {
            allocation_multiplicity
                .get(&value.origin)
                .map(|copies| (value.id, *copies))
        })
        .collect::<BTreeMap<_, _>>();
    region_peak_memory_with_multiplicity(
        initial,
        &branch.operations,
        &live,
        &branch.state.values,
        &multiplicity,
        memory_constraints,
    )
}

fn plan_fits_operator_memory(
    plan: &OperatorPlan,
    inputs: &[TensorType],
    output: &TensorShape,
    config: &PipelineConfig,
) -> bool {
    let planned_inputs = inputs
        .iter()
        .zip(&plan.requirements.inputs)
        .map(|(input, requirement)| TensorType {
            shape: input.shape.clone(),
            format: requirement.format.clone(),
        })
        .collect::<Vec<_>>();
    let planned_output = TensorType {
        shape: output.clone(),
        format: plan.requirements.output.format.clone(),
    };
    let peak = operator_memory_estimate(
        &plan.dispatch,
        &plan.requirements,
        &planned_inputs,
        &planned_output,
    )
    .peak;
    let constraints = config.target.memory_constraints();
    peak.interleaved <= constraints.interleaved_bytes
        && peak
            .total()
            .saturating_add(config.standard_memory_reservation_bytes)
            <= config.tile_memory_budget_bytes.min(constraints.total_bytes)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GroupedOutputLayout {
    groups: u16,
    physical_lane_multiple: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ParallelGridCandidate {
    metrics: RegionMetrics,
    grid: GemmGrid,
    physical_column_groups: u16,
    grouped: bool,
}

fn grouped_output_layout(
    source: &[Operation],
    operation_index: usize,
    operation: &Operation,
    output: &TensorShape,
    value_uses: &BTreeMap<ValueId, usize>,
) -> Option<GroupedOutputLayout> {
    let result = *operation.results.first()?;
    if value_uses.get(&result).copied() != Some(1) {
        return None;
    }
    let consumer = source[operation_index + 1..]
        .iter()
        .find(|candidate| candidate.inputs.contains(&result))?;
    let OperationKind::SplitHeads(options) = consumer.kind else {
        return None;
    };
    let groups = u16::try_from(options.heads).ok()?;
    let width = *output.0.last()?;
    (groups != 0 && width.is_multiple_of(u32::from(groups))).then_some(GroupedOutputLayout {
        groups,
        physical_lane_multiple: AMP_COLUMN_MICRO,
    })
}

fn direct_consumer_layouts(
    source: &[Operation],
    operation_index: usize,
    result: ValueId,
    output: &TensorShape,
    config: &PipelineConfig,
) -> Vec<Layout> {
    if !config
        .search_domain
        .permits_precision(OperatorClass::Attention, Precision::F16)
    {
        return Vec::new();
    }
    let Ok(streams) = u16::try_from(output.0.first().copied().unwrap_or(0)) else {
        return Vec::new();
    };
    if streams == 0 {
        return Vec::new();
    }
    let Some(&rows) = output.0.get(1) else {
        return Vec::new();
    };
    let query_partitions = u16::try_from(rows)
        .unwrap_or(u16::MAX)
        .min(config.tile_count / streams);
    let key_partitions = u16::try_from(rows.div_ceil(AMP_INNER_BLOCK))
        .unwrap_or(u16::MAX)
        .min(config.tile_count / streams);
    let mut layouts = Vec::new();
    for consumer in &source[operation_index + 1..] {
        for input_index in consumer
            .inputs
            .iter()
            .enumerate()
            .filter_map(|(index, &input)| (input == result).then_some(index))
        {
            let layout = match (&consumer.kind, input_index) {
                (OperationKind::FlashAttention(_), 0) if query_partitions != 0 => {
                    Some(Layout::attention_query(streams, query_partitions))
                }
                (OperationKind::FlashAttention(_), 1) if key_partitions != 0 => {
                    Some(Layout::attention_key(streams, key_partitions))
                }
                (OperationKind::FlashAttention(_), 2) if key_partitions != 0 => Some(
                    Layout::attention_block_major_key_value(streams, key_partitions),
                ),
                _ => None,
            };
            if let Some(layout) = layout
                && !layouts.contains(&layout)
            {
                layouts.push(layout);
            }
        }
    }
    layouts
}

fn operator_accepts_input_layout(
    operation: &OperationKind,
    input_index: usize,
    config: &PipelineConfig,
) -> bool {
    match operation {
        OperationKind::Gelu => {
            input_index == 0
                && !config
                    .search_domain
                    .precisions(OperatorClass::Gelu)
                    .is_empty()
        }
        OperationKind::Add(_) => {
            input_index < 2
                && !config
                    .search_domain
                    .precisions(OperatorClass::Add)
                    .is_empty()
        }
        OperationKind::SplitHeads(_) => input_index == 0,
        OperationKind::Gemm(_) | OperationKind::FlashAttention(_) | OperationKind::Repeat(_) => {
            false
        }
    }
}

fn plans_for_operation(
    operation: &Operation,
    inputs: &[TensorType],
    parameter_inputs: &[bool],
    output: &TensorShape,
    config: &PipelineConfig,
    costs: &impl CostModel,
    distributed_result_is_useful: bool,
    grouped_output: Option<GroupedOutputLayout>,
    direct_consumer_layouts: &[Layout],
) -> Vec<OperatorPlan> {
    let mut plans = Vec::new();
    let gemm_constraint = config
        .search_domain
        .gemm_plan_constraints
        .iter()
        .find(|constraint| constraint.source_operation == operation.id.index());
    if let OperationKind::SplitHeads(options) = operation.kind
        && let [input] = inputs
        && output.0.len() == 3
        && let (Ok(streams), Ok(rows)) = (u16::try_from(output.0[0]), u16::try_from(output.0[1]))
        && streams != 0
    {
        let query_partitions = rows.min(config.tile_count / streams);
        let key_partitions = u16::try_from(output.0[1].div_ceil(AMP_INNER_BLOCK))
            .unwrap_or(u16::MAX)
            .min(config.tile_count / streams);
        let layouts = if direct_consumer_layouts.is_empty() {
            [
                (query_partitions != 0).then(|| Layout::attention_query(streams, query_partitions)),
                (key_partitions != 0).then(|| Layout::attention_key(streams, key_partitions)),
                (key_partitions != 0)
                    .then(|| Layout::attention_block_major_key_value(streams, key_partitions)),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
        } else {
            direct_consumer_layouts.to_vec()
        };
        for layout in layouts {
            let plan = OperatorPlan {
                operator: MidOperator::SplitHeads(options),
                dispatch: OperatorDispatch::SplitHeads,
                requirements: OperatorRequirements {
                    inputs: vec![OperandRequirement::new(input.format.clone(), 8)],
                    output: OperandRequirement::new(
                        TensorFormat {
                            precision: input.format.precision,
                            layout,
                        },
                        8,
                    ),
                    output_aliasing: OutputAliasing::Fresh,
                    memory_space: MemorySpaceRequirements::default(),
                },
                deferred_output: Some(DeferredOutputPlan {
                    source_input: 0,
                    transform: DeferredTransform::SplitLastAxisIntoLeading {
                        parts: options.heads,
                    },
                    unfused_cost: CostEstimate::default(),
                }),
                deferred_inputs: vec![None],
            };
            if !plans.contains(&plan) {
                plans.push(plan);
            }
        }
    }
    if let OperationKind::FlashAttention(options) = operation.kind
        && config
            .search_domain
            .permits_precision(OperatorClass::Attention, Precision::F16)
        && !options.causal
        && let [query, key, value] = inputs
        && query.shape.0.len() == 3
        && key.shape.0.len() == 3
        && value.shape.0.len() == 3
        && query.shape.0[0] == key.shape.0[0]
        && query.shape.0[0] == value.shape.0[0]
        && let Ok(heads) = u16::try_from(query.shape.0[0])
        && heads != 0
    {
        let query_rows = query.shape.0[1];
        let query_partitions = u16::try_from(query_rows)
            .unwrap_or(u16::MAX)
            .min(config.tile_count / heads);
        if query_partitions != 0 {
            let key_partitions =
                u16::try_from(key.shape.0[1].div_ceil(AMP_INNER_BLOCK)).unwrap_or(u16::MAX);
            if key_partitions == 0 || heads.saturating_mul(key_partitions) > config.tile_count {
                return plans;
            }
            let padded_query_dimension =
                query.shape.0[2].div_ceil(AMP_COLUMN_MICRO) * AMP_COLUMN_MICRO;
            let padded_value_dimension =
                value.shape.0[2].div_ceil(AMP_COLUMN_MICRO) * AMP_COLUMN_MICRO;
            let padded_key_rows = key.shape.0[1].div_ceil(AMP_INNER_BLOCK) * AMP_INNER_BLOCK;
            let query_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::attention_query(heads, query_partitions),
            };
            let key_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::attention_key(heads, key_partitions),
            };
            let value_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::attention_block_major_key_value(heads, key_partitions),
            };
            let output_format = TensorFormat {
                precision: Precision::F32,
                layout: Layout::attention_output(heads, query_partitions),
            };
            if config.search_domain.attention_strategy != AttentionStrategy::Materialized {
                plans.push(OperatorPlan {
                    operator: MidOperator::FlashAttention {
                        options,
                        accumulate: AccumulationPrecision::F32,
                    },
                    dispatch: OperatorDispatch::Attention(AttentionPlan {
                        kernel: GemmKernelFamily {
                            multiply: Precision::F16,
                            accumulate: AccumulationPrecision::F32,
                            weights: GemmWeightLoad::Standard,
                        },
                        blocking: AttentionBlocking::Flash {
                            query_rows: query_rows.div_ceil(u32::from(query_partitions)),
                            key_rows: AMP_INNER_BLOCK,
                        },
                        padding: AttentionPadding {
                            query_dimension: padded_query_dimension,
                            value_dimension: padded_value_dimension,
                        },
                    }),
                    requirements: OperatorRequirements {
                        inputs: vec![
                            OperandRequirement::new(query_format.clone(), 8)
                                .with_materialization(OperandMaterialization::DispatchSlices),
                            OperandRequirement::new(key_format.clone(), 8)
                                .with_materialization(OperandMaterialization::DispatchSlices),
                            OperandRequirement::new(value_format.clone(), 8)
                                .with_materialization(OperandMaterialization::DispatchSlices),
                        ],
                        output: OperandRequirement::new(output_format.clone(), 8),
                        output_aliasing: OutputAliasing::Fresh,
                        memory_space: MemorySpaceRequirements::default(),
                    },
                    deferred_output: None,
                    deferred_inputs: vec![None; 3],
                });
            }
            if config.search_domain.attention_strategy != AttentionStrategy::Flash {
                plans.push(OperatorPlan {
                    operator: MidOperator::FlashAttention {
                        options,
                        accumulate: AccumulationPrecision::F32,
                    },
                    dispatch: OperatorDispatch::Attention(AttentionPlan {
                        kernel: GemmKernelFamily {
                            multiply: Precision::F16,
                            accumulate: AccumulationPrecision::F32,
                            weights: GemmWeightLoad::Standard,
                        },
                        blocking: AttentionBlocking::Materialized {
                            query_rows: query_rows.div_ceil(u32::from(query_partitions)),
                            padded_key_rows,
                        },
                        padding: AttentionPadding {
                            query_dimension: padded_query_dimension,
                            value_dimension: padded_value_dimension,
                        },
                    }),
                    requirements: OperatorRequirements {
                        inputs: vec![
                            OperandRequirement::new(query_format, 8)
                                .with_materialization(OperandMaterialization::DispatchSlices),
                            OperandRequirement::new(key_format, 8)
                                .with_materialization(OperandMaterialization::DispatchSlices),
                            OperandRequirement::new(value_format, 8)
                                .with_materialization(OperandMaterialization::DispatchSlices),
                        ],
                        output: OperandRequirement::new(output_format, 8),
                        output_aliasing: OutputAliasing::Fresh,
                        memory_space: MemorySpaceRequirements::default(),
                    },
                    deferred_output: None,
                    deferred_inputs: vec![None; 3],
                });
            }
        }
    }
    match operation.kind {
        OperationKind::Gemm(options) => plans.extend(gemm_plans(
            options,
            inputs,
            parameter_inputs,
            output,
            config,
            costs,
            distributed_result_is_useful,
            gemm_constraint,
            grouped_output,
        )),
        OperationKind::Gelu => plans.extend(pointwise_plans(
            MidOperator::Gelu,
            OperatorClass::Gelu,
            inputs,
            output,
            config,
        )),
        OperationKind::Add(options) => plans.extend(pointwise_plans(
            MidOperator::Add(options),
            OperatorClass::Add,
            inputs,
            output,
            config,
        )),
        OperationKind::SplitHeads(_)
        | OperationKind::FlashAttention(_)
        | OperationKind::Repeat(_) => {}
    }
    plans.retain(|plan| {
        plan.supports(inputs, output) && plan_fits_operator_memory(plan, inputs, output, config)
    });
    if let Some(constraint) = gemm_constraint {
        plans.retain(|plan| {
            gemm_plan_matches(constraint, &plan.dispatch, &plan.requirements.inputs)
        });
        tracing::info!(
            source_operation = constraint.source_operation,
            matching_plans = plans.len(),
            ?constraint,
            "applied GEMM plan constraint"
        );
    }
    plans
}

fn pointwise_plans(
    operator: MidOperator,
    class: OperatorClass,
    inputs: &[TensorType],
    output: &TensorShape,
    config: &PipelineConfig,
) -> Vec<OperatorPlan> {
    let mapping = match operator {
        MidOperator::Gelu => PointwiseInputMapping::TileLocal,
        MidOperator::Add(_) => PointwiseInputMapping::BroadcastToOutput,
        _ => return Vec::new(),
    };
    let kernel = match operator {
        MidOperator::Gelu => TileKernelSpec::Gelu,
        MidOperator::Add(_) => TileKernelSpec::Add,
        _ => unreachable!(),
    };
    let mut plans = Vec::new();
    for (anchor, input) in inputs.iter().enumerate() {
        if input.shape != *output
            || !config
                .search_domain
                .permits_precision(class, input.format.precision)
            || (matches!(operator, MidOperator::Gelu) && anchor != 0)
        {
            continue;
        }
        let mut formats = Vec::new();
        if config.conversion_streaming != ConversionStreamingPolicy::Always {
            formats.extend(pointwise_flat_formats(input, output, config));
        }
        formats.push(input.format.clone());
        for format in formats {
            let aliasing = OutputAliasing::MayAliasInputs(
                inputs
                    .iter()
                    .enumerate()
                    .filter_map(|(index, input)| (input.shape == *output).then_some(index as u16))
                    .collect(),
            );
            let plan = OperatorPlan {
                operator,
                dispatch: OperatorDispatch::Pointwise {
                    kernel: kernel.clone(),
                    input_mapping: mapping,
                },
                requirements: OperatorRequirements {
                    inputs: (0..inputs.len())
                        .map(|_| OperandRequirement::new(format.clone(), 8))
                        .collect(),
                    output: OperandRequirement::new(format, 8),
                    output_aliasing: aliasing,
                    memory_space: MemorySpaceRequirements::default(),
                },
                deferred_output: None,
                deferred_inputs: vec![None; inputs.len()],
            };
            if !plans.contains(&plan) {
                plans.push(plan);
            }
        }
    }
    plans
}

fn pointwise_flat_formats(
    input: &TensorType,
    output: &TensorShape,
    config: &PipelineConfig,
) -> Vec<TensorFormat> {
    let grain = 8_u32.div_ceil(input.format.precision.bytes() as u32);
    let Some(&width) = output.0.last() else {
        return Vec::new();
    };
    if grain == 0 || !output.elements().is_multiple_of(u64::from(grain)) {
        return Vec::new();
    }
    let width = u64::from(width);
    let grains = output.elements() / u64::from(grain);
    let mut candidates = BTreeMap::new();
    for &tiles in &config.resolved_active_tile_counts {
        let splits = (1..tiles)
            .filter(|&tile| {
                let tile = u64::from(tile);
                let offset = (tile * (grains / u64::from(tiles))
                    + tile.min(grains % u64::from(tiles)))
                    * u64::from(grain);
                !offset.is_multiple_of(width)
            })
            .count();
        candidates.entry(tiles).or_insert(splits);
    }
    candidates
        .iter()
        .filter(|(tiles, splits)| {
            !candidates.iter().any(|(other_tiles, other_splits)| {
                other_tiles >= tiles
                    && other_splits <= splits
                    && (other_tiles > tiles || other_splits < splits)
            })
        })
        .flat_map(|(&tiles, _)| {
            let mut layouts = vec![Layout::logical_linear(tiles, grain)];
            if let Some(retained_grain) = input
                .format
                .layout
                .order
                .retained_linear_column_grain(input.format.precision)
                .filter(|retained_grain| retained_grain.is_multiple_of(grain))
                .filter(|retained_grain| {
                    output.elements().is_multiple_of(u64::from(*retained_grain))
                        && output
                            .0
                            .last()
                            .is_some_and(|width| width.is_multiple_of(*retained_grain))
                })
            {
                layouts.push(
                    input
                        .format
                        .layout
                        .with_retained_order_linear_ownership(tiles, retained_grain),
                );
            }
            layouts.into_iter().map(|layout| TensorFormat {
                precision: input.format.precision,
                layout,
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn gemm_plans(
    options: GemmOptions,
    inputs: &[TensorType],
    parameter_inputs: &[bool],
    output: &TensorShape,
    config: &PipelineConfig,
    costs: &impl CostModel,
    distributed_result_is_useful: bool,
    constraint: Option<&GemmPlanConstraint>,
    grouped_output: Option<GroupedOutputLayout>,
) -> Vec<OperatorPlan> {
    if options != GemmOptions::default() {
        return Vec::new();
    }
    let mut plans = Vec::new();
    for &tile_count in &config.resolved_active_tile_counts {
        for seed in gemm_seed_plans_for_tile_count(options, tile_count, &config.search_domain) {
            let mut variants = vec![seed.clone()];
            variants.extend(parallel_reduction_plans(
                &seed,
                inputs,
                output,
                config,
                costs,
                distributed_result_is_useful,
                constraint,
                grouped_output,
            ));
            for (input_index, _) in parameter_inputs
                .iter()
                .enumerate()
                .filter(|(_, parameter)| **parameter)
            {
                let additions = variants
                    .iter()
                    .flat_map(|variant| {
                        independent_parameter_storage(variant, inputs, input_index, config)
                    })
                    .filter(|independent| !variants.contains(independent))
                    .collect::<Vec<_>>();
                variants.extend(additions);
            }
            for plan in variants {
                if !plan.supports(inputs, output) {
                    continue;
                }
                if !plans.contains(&plan) {
                    plans.push(plan);
                }
            }
        }
    }
    plans
}

fn gemm_plan_matches(
    constraint: &GemmPlanConstraint,
    dispatch: &OperatorDispatch,
    inputs: &[OperandRequirement],
) -> bool {
    let OperatorDispatch::BlockedGemm(plan) = dispatch else {
        return false;
    };
    if !matches!(
        plan.geometry.distribution,
        GemmDistribution::ParallelReduction(_)
    ) {
        return false;
    }
    let weight_index = match plan.geometry.orientation {
        GemmOrientation::Normal => 1,
        GemmOrientation::Swapped => 0,
    };
    let Some(weight) = inputs.get(weight_index) else {
        return false;
    };
    plan.geometry == constraint.geometry
        && weight.format.layout.memory_class == constraint.weight_memory_class
        && weight.local_staging == constraint.local_weight_staging
}

fn independent_parameter_storage(
    candidate: &OperatorPlan,
    inputs: &[TensorType],
    input_index: usize,
    config: &PipelineConfig,
) -> Vec<OperatorPlan> {
    if !matches!(candidate.dispatch, OperatorDispatch::BlockedGemm(_)) {
        return Vec::new();
    }
    let Some(requirement) = candidate.requirements.inputs.get(input_index) else {
        return Vec::new();
    };
    let ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
        row_block: inner_block,
        column_block: _,
    }) = requirement.format.layout.order
    else {
        return Vec::new();
    };
    let Some(input) = inputs.get(input_index) else {
        return Vec::new();
    };
    let rank = input.shape.0.len();
    let Some(inner_axis) = rank.checked_sub(2) else {
        return Vec::new();
    };
    let Some(&inner) = input.shape.0.get(inner_axis) else {
        return Vec::new();
    };
    let Some(&columns) = input.shape.0.last() else {
        return Vec::new();
    };
    let inner_blocks = inner.div_ceil(u32::from(inner_block));
    let output_column_block = match candidate.dispatch {
        OperatorDispatch::BlockedGemm(plan) => plan.geometry.block.output_columns,
        OperatorDispatch::Pointwise { .. }
        | OperatorDispatch::Attention(_)
        | OperatorDispatch::SplitHeads => {
            return Vec::new();
        }
    };
    if output_column_block < AMP_OUTPUT_COLUMN_BLOCK {
        return Vec::new();
    }
    let column_blocks = columns.div_ceil(output_column_block);
    let mut storage_grids = (1..=inner_blocks.min(u32::from(config.tile_count)))
        .flat_map(|inner_partitions| {
            let maximum_columns = (u32::from(config.tile_count) / inner_partitions)
                .min(column_blocks)
                .min(u32::from(u16::MAX));
            (1..=maximum_columns).map(move |column_partitions| {
                let panels_per_shard = inner_blocks
                    .div_ceil(inner_partitions)
                    .saturating_mul(column_blocks.div_ceil(column_partitions));
                let used = inner_partitions.saturating_mul(column_partitions);
                (
                    panels_per_shard,
                    u32::MAX - used,
                    column_partitions,
                    inner_partitions,
                )
            })
        })
        .collect::<Vec<_>>();
    storage_grids.sort_unstable();
    storage_grids
        .first()
        .and_then(|&(_, _, column_partitions, inner_partitions)| {
            Some((
                u16::try_from(column_partitions).ok()?,
                u16::try_from(inner_partitions).ok()?,
            ))
        })
        .into_iter()
        .map(|(column_partitions, inner_partitions)| {
            let mut independent = candidate.clone();
            independent.requirements.inputs[input_index].format.layout =
                Layout::block_major_matrix_storage(
                    inner_block,
                    output_column_block,
                    column_partitions,
                    inner_partitions,
                    1,
                    requirement.format.layout.memory_class,
                );
            independent
        })
        .collect()
}

fn parallel_reduction_plans(
    candidate: &OperatorPlan,
    inputs: &[TensorType],
    output: &TensorShape,
    config: &PipelineConfig,
    costs: &impl CostModel,
    distributed_result_is_useful: bool,
    constraint: Option<&GemmPlanConstraint>,
    grouped_output: Option<GroupedOutputLayout>,
) -> Vec<OperatorPlan> {
    [GemmOrientation::Normal, GemmOrientation::Swapped]
        .into_iter()
        .flat_map(|orientation| {
            parallel_reduction_plans_for_orientation(
                candidate,
                inputs,
                output,
                config,
                costs,
                orientation,
                distributed_result_is_useful,
                constraint,
                grouped_output,
            )
        })
        .collect()
}

fn parallel_reduction_plans_for_orientation(
    candidate: &OperatorPlan,
    inputs: &[TensorType],
    output: &TensorShape,
    config: &PipelineConfig,
    costs: &impl CostModel,
    orientation: GemmOrientation,
    distributed_result_is_useful: bool,
    constraint: Option<&GemmPlanConstraint>,
    grouped_output: Option<GroupedOutputLayout>,
) -> Vec<OperatorPlan> {
    // Residual supervisor, weight-feed, and worker setup cost after retained
    // state, measured on IPU21 independently of the four issue cycles per row.
    const AMP_F16_MICROBLOCK_FIXED_CYCLES: u64 = 160;
    let OperatorDispatch::BlockedGemm(plan) = candidate.dispatch else {
        return Vec::new();
    };
    if !matches!(
        plan.geometry.distribution,
        GemmDistribution::OutputStationary
    ) {
        return Vec::new();
    }
    let output_column_block = plan.geometry.block.output_columns;
    if !matches!(
        candidate.operator,
        MidOperator::Gemm {
            multiply: Precision::F16,
            ..
        }
    ) || output_column_block != AMP_OUTPUT_COLUMN_BLOCK
    {
        return Vec::new();
    }
    let [left, right] = inputs else {
        return Vec::new();
    };
    let rank = left.shape.0.len();
    if rank < 2 || right.shape.0.len() < 2 {
        return Vec::new();
    }
    let Some(&inner) = left.shape.0.last() else {
        return Vec::new();
    };
    let Some(&normal_columns) = right.shape.0.last() else {
        return Vec::new();
    };
    let normal_rows = left.shape.0[left.shape.0.len() - 2];
    let (rows, columns) = match orientation {
        GemmOrientation::Normal => (normal_rows, normal_columns),
        GemmOrientation::Swapped => (normal_columns, normal_rows),
    };
    // Generate the shape-specialized family once from the ordinary C64 seed.
    // Each grid chooses the exact padded local K and C extents, so one tile
    // call traverses all of its AMP micro-groups without fixed K64/C64
    // boundaries.
    let tile_count = candidate
        .requirements
        .output
        .format
        .layout
        .tiling
        .tile_count;
    let column_groups = columns.div_ceil(AMP_COLUMN_MICRO);
    let inner_groups = inner.div_ceil(AMP_COLUMN_MICRO);
    let Ok(inner_groups) = u16::try_from(inner_groups) else {
        return Vec::new();
    };
    let Ok(column_groups) = u16::try_from(column_groups) else {
        return Vec::new();
    };
    let grouped_column_groups = grouped_output.and_then(|grouping| {
        let groups = u32::from(grouping.groups);
        (groups != 0 && columns.is_multiple_of(groups)).then(|| {
            let columns_per_group = columns / groups;
            columns_per_group
                .div_ceil(grouping.physical_lane_multiple)
                .saturating_mul(groups)
        })
    });
    let grouped_column_groups = grouped_column_groups
        .and_then(|groups| u16::try_from(groups).ok())
        .filter(|groups| *groups >= column_groups);
    let output_seed_partitions = candidate
        .requirements
        .output
        .format
        .layout
        .tiling
        .axes
        .iter()
        .find(|axis| axis.axis == TensorAxis::FromEnd(1))
        .map(|axis| axis.partitions);
    if output_seed_partitions != Some(tile_count)
        || candidate.requirements.inputs[1].format.layout.memory_class != MemoryClass::Standard
        || candidate.requirements.inputs[1].local_staging != LocalOperandStaging::Direct
    {
        return Vec::new();
    }
    let outer_rows = left.shape.0[..left.shape.0.len() - 2]
        .iter()
        .copied()
        .fold(1u64, |product, extent| {
            product.saturating_mul(u64::from(extent))
        });
    let mut grids = Vec::new();
    for inner_partitions in 2..=inner_groups.min(tile_count) {
        let maximum_columns = grouped_column_groups
            .unwrap_or(column_groups)
            .min(tile_count / inner_partitions);
        for column_partitions in 1..=maximum_columns {
            let grouped_options = [
                (column_partitions <= column_groups).then_some((false, column_groups)),
                grouped_output.and_then(|grouping| {
                    let physical = grouped_column_groups?;
                    column_partitions
                        .is_multiple_of(grouping.groups)
                        .then_some((true, physical))
                }),
            ];
            for (grouped, physical_column_groups) in grouped_options.into_iter().flatten() {
                let row_partitions = (tile_count / inner_partitions / column_partitions)
                    .min(u16::try_from(rows).unwrap_or(u16::MAX));
                let used_tiles = row_partitions
                    .saturating_mul(column_partitions)
                    .saturating_mul(inner_partitions);
                if used_tiles < tile_count.div_ceil(2) || u32::from(row_partitions) > rows {
                    continue;
                }
                let local_rows = rows.div_ceil(u32::from(row_partitions));
                let local_columns =
                    u32::from(physical_column_groups).div_ceil(u32::from(column_partitions));
                let local_inner = u32::from(inner_groups).div_ceil(u32::from(inner_partitions));
                if u32::from(inner_partitions - 1).saturating_mul(local_inner)
                    >= u32::from(inner_groups)
                {
                    continue;
                }
                // Retain grids by the generated kernel's actual K16 x C16
                // invocation structure, including its fixed weight-feed and
                // worker/supervisor cost. Pure arithmetic work is almost
                // constant across grids and incorrectly favors tiny row runs.
                let row_run_cycles = outer_rows
                    .saturating_mul(u64::from(local_rows))
                    .saturating_mul(4)
                    .saturating_add(AMP_F16_MICROBLOCK_FIXED_CYCLES);
                let compute = u64::from(local_columns)
                    .saturating_mul(u64::from(local_inner))
                    .saturating_mul(row_run_cycles);
                let communication = u64::from(local_columns)
                    .saturating_mul(u64::from(local_inner))
                    .saturating_add(u64::from(local_rows).saturating_mul(u64::from(local_inner)))
                    .saturating_add(
                        u64::from(local_rows)
                            .saturating_mul(u64::from(local_columns))
                            .saturating_mul(u64::from(inner_partitions - 1)),
                    );
                let left_bytes = outer_rows
                    .saturating_mul(u64::from(local_rows))
                    .saturating_mul(u64::from(local_inner))
                    .saturating_mul(u64::from(AMP_COLUMN_MICRO))
                    .saturating_mul(
                        candidate.requirements.inputs[match orientation {
                            GemmOrientation::Normal => 0,
                            GemmOrientation::Swapped => 1,
                        }]
                        .format
                        .precision
                        .bytes(),
                    );
                let right_bytes = u64::from(local_columns)
                    .saturating_mul(u64::from(AMP_COLUMN_MICRO))
                    .saturating_mul(u64::from(local_inner))
                    .saturating_mul(u64::from(AMP_COLUMN_MICRO))
                    .saturating_mul(
                        candidate.requirements.inputs[match orientation {
                            GemmOrientation::Normal => 1,
                            GemmOrientation::Swapped => 0,
                        }]
                        .format
                        .precision
                        .bytes(),
                    );
                let partial_bytes = outer_rows
                    .saturating_mul(u64::from(local_rows))
                    .saturating_mul(u64::from(local_columns))
                    .saturating_mul(u64::from(AMP_COLUMN_MICRO))
                    .saturating_mul(candidate.requirements.output.format.precision.bytes());
                // Operand staging and the local partial coexist during
                // convolution. Complete staging is evaluated later by the
                // ordinary operator-memory model.
                let convolution_bytes = left_bytes
                    .saturating_add(right_bytes)
                    .saturating_add(partial_bytes);
                let reduction_bytes = partial_bytes.saturating_mul(4);
                let temporary_bytes = convolution_bytes.max(reduction_bytes);
                let constraints = config.target.memory_constraints();
                if temporary_bytes > constraints.total_bytes
                    || right_bytes.saturating_add(partial_bytes) > constraints.interleaved_bytes
                {
                    continue;
                }
                grids.push(ParallelGridCandidate {
                    metrics: RegionMetrics {
                        cost: CostEstimate {
                            cycles: compute.saturating_add(communication),
                            exchange_cycles: communication,
                            ..CostEstimate::default()
                        },
                        memory: MemoryPeaks {
                            standard: temporary_bytes,
                            total: temporary_bytes,
                            ..MemoryPeaks::default()
                        },
                    },
                    grid: GemmGrid {
                        rows: row_partitions,
                        columns: column_partitions,
                        inner: inner_partitions,
                    },
                    physical_column_groups,
                    grouped,
                });
            }
        }
    }
    let generated_grids = grids.len();
    let grids = if let Some(constraint) = constraint {
        grids
            .into_iter()
            .filter(|grid| {
                let GemmDistribution::ParallelReduction(reduction) =
                    constraint.geometry.distribution
                else {
                    return false;
                };
                orientation == constraint.geometry.orientation && grid.grid == reduction.compute
            })
            .collect::<Vec<_>>()
    } else {
        // Inner partitioning and grouped outputs select different lowering
        // families. Within each family the shared metrics vocabulary retains
        // cycle, exchange, and memory tradeoffs for precise evaluation below.
        let dominates = |left: &ParallelGridCandidate, right: &ParallelGridCandidate| {
            left.grid.inner == right.grid.inner
                && left.grouped == right.grouped
                && left.metrics.dominates(right.metrics)
        };
        let mut frontier = Vec::new();
        for grid in grids {
            if frontier.iter().any(|kept| dominates(kept, &grid)) {
                continue;
            }
            frontier.retain(|kept| !dominates(&grid, kept));
            frontier.push(grid);
        }
        frontier.sort_by_key(|grid| {
            (
                grid.metrics.cost.cycles,
                grid.metrics.cost.exchange_cycles,
                grid.metrics.memory.total,
                grid.grid,
                grid.grouped,
            )
        });
        frontier
    };
    let proxy_frontier_grids = grids.len();
    let mut variants = Vec::new();
    for grid in grids {
        let ParallelGridCandidate {
            grid:
                GemmGrid {
                    rows: row_partitions,
                    columns: column_partitions,
                    inner: inner_partitions,
                },
            physical_column_groups,
            grouped,
            ..
        } = grid;
        let used_tiles = row_partitions
            .saturating_mul(column_partitions)
            .saturating_mul(inner_partitions);
        let kernel_inner_block = u32::from(inner_groups)
            .div_ceil(u32::from(inner_partitions))
            .saturating_mul(AMP_COLUMN_MICRO);
        let kernel_output_columns = u32::from(physical_column_groups)
            .div_ceil(u32::from(column_partitions))
            .saturating_mul(AMP_COLUMN_MICRO);
        let Ok(kernel_inner_block_u16) = u16::try_from(kernel_inner_block) else {
            continue;
        };
        for &memory_class in &config.search_domain.weight_memory_classes {
            let mut variant = candidate.clone();
            match orientation {
                GemmOrientation::Normal => {
                    variant.requirements.inputs[0].format.layout = Layout::amp_left_parallel_grid(
                        kernel_inner_block_u16,
                        used_tiles,
                        row_partitions,
                        column_partitions,
                        inner_partitions,
                    );
                    variant.requirements.inputs[1].format.layout =
                        Layout::block_major_matrix_storage(
                            kernel_inner_block_u16,
                            kernel_output_columns,
                            column_partitions,
                            inner_partitions,
                            1,
                            memory_class,
                        );
                    balance_parallel_gemm_columns(
                        &mut variant.requirements.inputs[1].format.layout,
                        TensorAxis::FromEnd(1),
                    );
                    variant.requirements.output.format.layout = Layout::amp_left_result_grid(
                        kernel_output_columns,
                        row_partitions.saturating_mul(column_partitions),
                        row_partitions,
                        column_partitions,
                        GridOrder::ColumnsFast,
                    );
                    balance_parallel_gemm_columns(
                        &mut variant.requirements.output.format.layout,
                        TensorAxis::FromEnd(1),
                    );
                }
                GemmOrientation::Swapped => {
                    let mut physical_left = variant.requirements.inputs[1].clone();
                    physical_left.format.layout = Layout::amp_transposed_left_parallel_grid(
                        kernel_inner_block_u16,
                        used_tiles,
                        row_partitions,
                        column_partitions,
                        inner_partitions,
                    );
                    physical_left.materialization = OperandMaterialization::DispatchSlices;
                    let mut physical_right = variant.requirements.inputs[0].clone();
                    physical_right.format.layout = Layout::transposed_block_major_matrix_storage(
                        kernel_inner_block_u16,
                        kernel_output_columns,
                        column_partitions,
                        inner_partitions,
                        row_partitions,
                        memory_class,
                    );
                    balance_parallel_gemm_columns(
                        &mut physical_right.format.layout,
                        TensorAxis::FromEnd(2),
                    );
                    physical_right.materialization = OperandMaterialization::Complete;
                    variant.requirements.inputs = vec![physical_right, physical_left];
                    variant.requirements.output.format.layout =
                        Layout::amp_transposed_left_result_grid(
                            kernel_output_columns,
                            row_partitions.saturating_mul(column_partitions),
                            row_partitions,
                            column_partitions,
                            GridOrder::ColumnsFast,
                        );
                    balance_parallel_gemm_columns(
                        &mut variant.requirements.output.format.layout,
                        TensorAxis::FromEnd(2),
                    );
                    variant.requirements.memory_space = MemorySpaceRequirements::default()
                        .with_distinct_elements([MemoryOperand::Output, MemoryOperand::Input(1)]);
                }
            }
            if let OperatorDispatch::BlockedGemm(plan) = &mut variant.dispatch {
                plan.kernel.weights = if memory_class == MemoryClass::Interleaved {
                    GemmWeightLoad::Interleaved
                } else {
                    GemmWeightLoad::Standard
                };
                plan.geometry = GemmGeometry {
                    block: GemmBlockShape {
                        inner: kernel_inner_block,
                        output_columns: kernel_output_columns,
                    },
                    orientation,
                    result: GemmResultGrid {
                        rows: row_partitions,
                        columns: column_partitions,
                    },
                    order: GridOrder::ColumnsFast,
                    distribution: GemmDistribution::ParallelReduction(ParallelReductionPlan {
                        compute: GemmGrid {
                            rows: row_partitions,
                            columns: column_partitions,
                            inner: inner_partitions,
                        },
                        staging: ReductionStaging::Complete,
                    }),
                };
            }
            let physical_right_index = match orientation {
                GemmOrientation::Normal => 1,
                GemmOrientation::Swapped => 0,
            };
            let local_staging_options: &[_] = match orientation {
                GemmOrientation::Normal => &[LocalOperandStaging::Direct],
                GemmOrientation::Swapped => &[
                    LocalOperandStaging::Direct,
                    LocalOperandStaging::MatchRemote,
                ],
            };
            let mut result_layout_variants = Vec::new();
            let maximum_result_rows = u16::try_from(rows / u32::from(row_partitions))
                .unwrap_or(u16::MAX)
                .min(inner_partitions);
            let maximum_result_columns = u16::try_from(
                columns
                    .div_ceil(AMP_COLUMN_MICRO)
                    .checked_div(u32::from(column_partitions))
                    .unwrap_or(0),
            )
            .unwrap_or(u16::MAX)
            .min(inner_partitions);
            let mut result_partition_options = vec![(1, 1)];
            if distributed_result_is_useful {
                if inner_partitions <= maximum_result_rows {
                    result_partition_options.push((inner_partitions, 1));
                } else if inner_partitions <= maximum_result_columns {
                    result_partition_options.push((1, inner_partitions));
                }
            }
            for (result_row_partitions, result_column_partitions) in result_partition_options {
                let result_rows = row_partitions.saturating_mul(result_row_partitions);
                let result_columns = column_partitions.saturating_mul(result_column_partitions);
                let result_column_block = if result_column_partitions > 1 {
                    AMP_COLUMN_MICRO
                } else {
                    kernel_output_columns
                };
                for grid_order in [GridOrder::ColumnsFast, GridOrder::RowsFast] {
                    if grid_order == GridOrder::RowsFast
                        && ((result_row_partitions, result_column_partitions) != (1, 1)
                            || result_rows == 1
                            || result_columns == 1)
                    {
                        continue;
                    }
                    let mut result_variant = variant.clone();
                    if let OperatorDispatch::BlockedGemm(plan) = &mut result_variant.dispatch
                        && matches!(
                            plan.geometry.distribution,
                            GemmDistribution::ParallelReduction(_)
                        )
                    {
                        plan.geometry.result.rows = result_rows;
                        plan.geometry.result.columns = result_columns;
                        plan.geometry.order = grid_order;
                    }
                    let mut result_layout = match orientation {
                        GemmOrientation::Normal => Layout::amp_left_result_grid(
                            result_column_block,
                            result_rows.saturating_mul(result_columns),
                            result_rows,
                            result_columns,
                            grid_order,
                        ),
                        GemmOrientation::Swapped => Layout::amp_transposed_left_result_grid(
                            result_column_block,
                            result_rows.saturating_mul(result_columns),
                            result_rows,
                            result_columns,
                            grid_order,
                        ),
                    };
                    let physical_column_axis = match orientation {
                        GemmOrientation::Normal => TensorAxis::FromEnd(1),
                        GemmOrientation::Swapped => TensorAxis::FromEnd(2),
                    };
                    balance_parallel_gemm_columns(&mut result_layout, physical_column_axis);
                    result_variant.requirements.output.format.layout = result_layout;
                    result_layout_variants.push(result_variant);
                }
            }
            let mut layout_variants = Vec::new();
            for mut result_layout in result_layout_variants {
                let (physical_row_axis, physical_rows, physical_left_index) = match orientation {
                    GemmOrientation::Normal => (TensorAxis::FromEnd(2), normal_rows, 0),
                    GemmOrientation::Swapped => (TensorAxis::FromEnd(1), normal_columns, 1),
                };
                let result_rows = result_layout
                    .requirements
                    .output
                    .format
                    .layout
                    .tiling
                    .axes
                    .iter()
                    .find(|axis| axis.axis == physical_row_axis)
                    .map_or(row_partitions, |axis| axis.partitions);
                // Exchange moves whole 32-bit words. Give every F16 shard an
                // even physical-row grain rather than allowing a later layout
                // conversion to discover an unsendable two-byte tail.
                if u32::from(result_rows) > physical_rows.div_ceil(2) {
                    continue;
                }
                pad_axis_to_f16_exchange_word(
                    &mut result_layout.requirements.inputs[physical_left_index]
                        .format
                        .layout,
                    physical_row_axis,
                );
                pad_axis_to_f16_exchange_word(
                    &mut result_layout.requirements.output.format.layout,
                    physical_row_axis,
                );
                layout_variants.push(result_layout);
            }
            for mut layout_variant in layout_variants {
                if grouped
                    && !grouped_output.is_some_and(|grouping| {
                        apply_grouped_output_layout(&mut layout_variant, grouping)
                    })
                {
                    continue;
                }
                for &local_staging in local_staging_options {
                    let mut staged = layout_variant.clone();
                    staged.requirements.inputs[physical_right_index].local_staging = local_staging;
                    variants.push(staged.clone());
                    if let OperatorDispatch::BlockedGemm(plan) = &mut staged.dispatch
                        && let GemmDistribution::ParallelReduction(reduction) =
                            &mut plan.geometry.distribution
                    {
                        reduction.staging = ReductionStaging::Streamed;
                    }
                    variants.push(staged);
                }
            }
        }
    }
    let generated_variants = variants.len();
    let generated_grouped_variants = variants
        .iter()
        .filter(|candidate| {
            candidate
                .requirements
                .output
                .format
                .layout
                .tiling
                .axes
                .iter()
                .any(|axis| axis.padding_groups > 1)
        })
        .count();
    let retained = if let Some(constraint) = constraint {
        variants
            .into_iter()
            .filter(|candidate| {
                gemm_plan_matches(
                    constraint,
                    &candidate.dispatch,
                    &candidate.requirements.inputs,
                )
            })
            .collect::<Vec<_>>()
    } else {
        retain_precise_gemm_plans(
            variants,
            inputs,
            output,
            costs,
            config.planning_beam_width.max(1),
        )
    };
    tracing::debug!(
        ?orientation,
        generated_grids,
        proxy_frontier_grids,
        generated_variants,
        generated_grouped_variants,
        retained_grouped_variants = retained
            .iter()
            .filter(|candidate| candidate
                .requirements
                .output
                .format
                .layout
                .tiling
                .axes
                .iter()
                .any(|axis| axis.padding_groups > 1))
            .count(),
        retained_variants = retained.len(),
        "retained parallel GEMM candidates"
    );
    retained
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct GemmPlanCompatibility {
    orientation: Option<GemmOrientation>,
    reduction_staging: Option<ReductionStaging>,
    inputs: Vec<(ElementOrderCompatibility, MemoryClass, LocalOperandStaging)>,
    output: (
        ElementOrderCompatibility,
        MemoryClass,
        Vec<(TensorAxis, u16, u32)>,
    ),
}

fn gemm_plan_compatibility(candidate: &OperatorPlan) -> GemmPlanCompatibility {
    let (orientation, reduction_staging) = match candidate.dispatch {
        OperatorDispatch::BlockedGemm(plan) => (
            Some(plan.geometry.orientation),
            match plan.geometry.distribution {
                GemmDistribution::ParallelReduction(reduction) => Some(reduction.staging),
                GemmDistribution::OutputStationary => None,
            },
        ),
        _ => (None, None),
    };
    GemmPlanCompatibility {
        orientation,
        reduction_staging,
        inputs: candidate
            .requirements
            .inputs
            .iter()
            .map(|input| {
                (
                    element_order_compatibility(input.format.layout.order),
                    input.format.layout.memory_class,
                    input.local_staging,
                )
            })
            .collect(),
        output: (
            element_order_compatibility(candidate.requirements.output.format.layout.order),
            candidate.requirements.output.format.layout.memory_class,
            candidate
                .requirements
                .output
                .format
                .layout
                .tiling
                .axes
                .iter()
                .map(|axis| (axis.axis, axis.padding_groups, axis.shard_padding_multiple))
                .collect(),
        ),
    }
}

fn retain_precise_gemm_plans(
    candidates: Vec<OperatorPlan>,
    inputs: &[TensorType],
    output: &TensorShape,
    costs: &impl CostModel,
    width: usize,
) -> Vec<OperatorPlan> {
    let ranked = candidates
        .into_iter()
        .map(|candidate| {
            let planned_inputs = inputs
                .iter()
                .zip(&candidate.requirements.inputs)
                .map(|(input, requirement)| TensorType {
                    shape: input.shape.clone(),
                    format: requirement.format.clone(),
                })
                .collect::<Vec<_>>();
            let planned_output = TensorType {
                shape: output.clone(),
                format: candidate.requirements.output.format.clone(),
            };
            let memory = operator_memory_estimate(
                &candidate.dispatch,
                &candidate.requirements,
                &planned_inputs,
                &planned_output,
            );
            let exchange = costs.operator_exchange_footprint(
                candidate.operator,
                &candidate.dispatch,
                &candidate.requirements,
                &planned_inputs,
                &planned_output,
            );
            let objective = RegionMetrics {
                cost: CostEstimate {
                    cycles: costs.operator_cycles(
                        candidate.operator,
                        &candidate.dispatch,
                        &candidate.requirements,
                        &planned_inputs,
                        &planned_output,
                    ),
                    exchange_footprint: exchange,
                    ..CostEstimate::default()
                },
                memory: memory.peaks(exchange.estimated_row_bytes()),
            };
            let compatibility = gemm_plan_compatibility(&candidate);
            (candidate, objective, compatibility)
        })
        .collect::<Vec<_>>();
    let mut frontier = Vec::<(OperatorPlan, RegionMetrics, GemmPlanCompatibility)>::new();
    for entry in ranked {
        if frontier
            .iter()
            .any(|(_, kept, compatibility)| *compatibility == entry.2 && kept.dominates(entry.1))
        {
            continue;
        }
        frontier.retain(|(_, kept, compatibility)| {
            *compatibility != entry.2 || !entry.1.dominates(*kept)
        });
        frontier.push(entry);
    }
    let mut ranked = frontier;
    ranked.sort_by_key(|(_, objective, _)| {
        (
            objective.cost.cycles,
            objective.memory.total,
            objective.memory.interleaved,
            objective.memory.exchange_rows,
        )
    });
    let mut selected = BTreeSet::new();
    let mut represented = BTreeSet::new();
    for (index, (_, _, compatibility)) in ranked.iter().enumerate() {
        if represented.insert(compatibility.clone()) {
            selected.insert(index);
        }
    }
    for index in 0..ranked.len() {
        if selected.len() == width {
            break;
        }
        selected.insert(index);
    }
    ranked
        .into_iter()
        .enumerate()
        .filter_map(|(index, (candidate, _, _))| selected.contains(&index).then_some(candidate))
        .collect()
}

fn balance_parallel_gemm_columns(layout: &mut Layout, axis: TensorAxis) {
    if let Some(columns) = layout
        .tiling
        .axes
        .iter_mut()
        .find(|tiling| tiling.axis == axis)
    {
        columns.block_size = AMP_COLUMN_MICRO;
        columns.padding_multiple = AMP_COLUMN_MICRO;
        columns.padding = Padding::Zero;
    }
}

fn apply_grouped_output_layout(
    candidate: &mut OperatorPlan,
    grouping: GroupedOutputLayout,
) -> bool {
    if candidate.requirements.output.format.precision != Precision::F16
        || grouping.groups == 0
        || grouping.physical_lane_multiple == 0
    {
        return false;
    }
    let configure = |layout: &mut Layout| {
        let Some(axis) = layout
            .tiling
            .axes
            .iter_mut()
            .find(|axis| axis.axis == TensorAxis::FromEnd(1))
        else {
            return false;
        };
        if !axis.partitions.is_multiple_of(grouping.groups) {
            return false;
        }
        // Subdivide every semantic group independently. This permits several
        // shards per group while keeping padding at the group boundary rather
        // than inserting it at unrelated grid boundaries.
        axis.block_size = grouping.physical_lane_multiple;
        axis.padding_multiple = grouping.physical_lane_multiple;
        axis.padding_groups = grouping.groups;
        axis.shard_padding_multiple = 1;
        axis.padding = Padding::Zero;
        true
    };
    configure(&mut candidate.requirements.inputs[1].format.layout)
        && configure(&mut candidate.requirements.output.format.layout)
}

fn pad_axis_to_f16_exchange_word(layout: &mut Layout, axis: TensorAxis) {
    if let Some(tiling) = layout
        .tiling
        .axes
        .iter_mut()
        .find(|tiling| tiling.axis == axis)
    {
        tiling.block_size = tiling.block_size.div_ceil(2) * 2;
        tiling.padding_multiple = tiling.padding_multiple.div_ceil(2) * 2;
        tiling.padding = Padding::Zero;
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_repeat(
    operation: &Operation,
    repeat: &Repeat,
    values: &mut BTreeMap<ValueId, MidValueId>,
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
    state: &mut LoweringState,
    operations: &mut Vec<MidOperation>,
) -> LoweringResult<()> {
    let inputs = operation
        .inputs
        .iter()
        .map(|value| lookup(values, *value))
        .collect::<LoweringResult<Vec<_>>>()?;
    let mut argument_types = inputs
        .iter()
        .map(|value| state.get(*value).tensor_type.clone())
        .collect::<Vec<_>>();
    let mut raw_iterated_inputs = Vec::with_capacity(repeat.iterated_inputs.len());
    let mut iterated_parameters = Vec::with_capacity(repeat.iterated_inputs.len());
    let mut iterated_automatic = Vec::with_capacity(repeat.iterated_inputs.len());
    for sequence_id in &repeat.iterated_inputs {
        let sequence = &graph.sequences()[sequence_id.index() as usize];
        let first = lookup(values, sequence.values[0])?;
        let first_type = state.get(first).tensor_type.clone();
        let sequence_values: Vec<_> = sequence
            .values
            .iter()
            .map(|value| lookup(values, *value))
            .collect::<LoweringResult<Vec<_>>>()?;
        let storage_group = state.get(first).storage_group;
        for &value in &sequence_values {
            state.values[value.index() as usize].storage_group = storage_group;
        }
        iterated_parameters.push(
            sequence_values
                .iter()
                .all(|value| state.parameter_values.contains(value)),
        );
        iterated_automatic.push(
            sequence_values
                .iter()
                .all(|value| state.automatic_inputs.contains(value)),
        );
        raw_iterated_inputs.push(sequence_values);
        argument_types.push(first_type);
    }
    let mut body_values = BTreeMap::new();
    let mut arguments = Vec::new();
    for (argument_index, (&origin, tensor_type)) in
        repeat.body.arguments.iter().zip(argument_types).enumerate()
    {
        let storage_group = if argument_index < inputs.len() {
            state.get(inputs[argument_index]).storage_group
        } else {
            state
                .get(raw_iterated_inputs[argument_index - inputs.len()][0])
                .storage_group
        };
        let value = state.value_in_storage_group(origin, tensor_type, storage_group);
        if argument_index < inputs.len() {
            if state.automatic_inputs.contains(&inputs[argument_index]) {
                state.automatic_inputs.insert(value);
            }
            if state.parameter_values.contains(&inputs[argument_index]) {
                state.parameter_values.insert(value);
            }
        } else {
            let iterated_index = argument_index - inputs.len();
            if iterated_automatic[iterated_index] {
                state.automatic_inputs.insert(value);
            }
            if iterated_parameters[iterated_index] {
                state.parameter_values.insert(value);
            }
        }
        body_values.insert(origin, value);
        arguments.push(value);
    }
    let body_allocation_copies = repeat
        .body
        .arguments
        .iter()
        .skip(inputs.len())
        .copied()
        .map(|argument| (argument, repeat.count))
        .collect::<BTreeMap<_, _>>();
    let required_equal_formats = repeat
        .body
        .yields
        .iter()
        .copied()
        .zip(repeat.body.arguments.iter().copied())
        .take(repeat.carried_inputs)
        .collect();
    let body_constraints = RegionPlanningConstraints {
        allocation_copies: body_allocation_copies,
        required_equal_formats,
    };
    let mut body_operations = lower_operations(
        &repeat.body.operations,
        &repeat.body.yields,
        &mut body_values,
        &repeat.body.value_shapes,
        graph,
        config,
        costs,
        state,
        &body_constraints,
    )?;
    for index in 0..repeat.carried_inputs {
        let body_layout = state
            .get(arguments[index])
            .tensor_type
            .format
            .layout
            .clone();
        state.retarget_automatic_input(inputs[index], body_layout);
    }
    let iterated_inputs = raw_iterated_inputs
        .into_iter()
        .enumerate()
        .map(|(index, sequence)| {
            let target = state
                .get(arguments[inputs.len() + index])
                .tensor_type
                .format
                .clone();
            sequence
                .into_iter()
                .map(|value| {
                    ensure_format(
                        value,
                        target.clone(),
                        OperandMaterialization::Complete,
                        operation.id,
                        costs,
                        state,
                        operations,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut yields = Vec::new();
    for (index, high_yield) in repeat.body.yields.iter().enumerate() {
        let value = lookup(&body_values, *high_yield)?;
        let target = state.get(inputs[index]).tensor_type.format.clone();
        yields.push(ensure_format(
            value,
            target,
            OperandMaterialization::Complete,
            operation.id,
            costs,
            state,
            &mut body_operations,
        ));
    }
    let body_metrics = body_operations
        .iter()
        .fold(CostEstimate::default(), |cost, operation| {
            cost.sequence(operation.metrics.cost)
        });
    let body_allocation_multiplicity = arguments
        .iter()
        .skip(inputs.len())
        .copied()
        .map(|argument| (argument, repeat.count))
        .collect::<BTreeMap<_, _>>();
    let body_peak = region_peak_memory_with_multiplicity(
        &arguments,
        &body_operations,
        &yields,
        &state.values,
        &body_allocation_multiplicity,
        config.target.memory_constraints(),
    );
    let mut results = Vec::new();
    for (origin, input) in operation.results.iter().zip(&inputs) {
        let tensor_type = state.get(*input).tensor_type.clone();
        let storage_group = state.get(*input).storage_group;
        let result = state.value_in_storage_group(*origin, tensor_type, storage_group);
        values.insert(*origin, result);
        results.push(result);
    }
    operations.push(MidOperation {
        source: Some(operation.id),
        inputs,
        results,
        kind: MidOperationKind::Repeat(MidRepeat {
            count: repeat.count,
            carried_inputs: repeat.carried_inputs,
            invariant_inputs: repeat.invariant_inputs,
            iterated_inputs,
            body: MidRegion {
                arguments,
                operations: body_operations,
                yields,
                metrics: RegionMetrics {
                    cost: body_metrics,
                    memory: body_peak,
                },
            },
        }),
        conversion_plan: None,
        metrics: OperationMetrics {
            cost: body_metrics.repeated(repeat.count),
            memory: MemoryEstimate {
                live: body_peak.conservative_tensor_usage(),
                temporary: MemoryUsage::default(),
                peak: body_peak.conservative_tensor_usage(),
                maximum_standard_temporary_allocation: 0,
            },
        },
    });
    Ok(())
}

fn ensure_format(
    mut value: MidValueId,
    target: TensorFormat,
    materialization: OperandMaterialization,
    source: OperationId,
    costs: &impl CostModel,
    state: &mut LoweringState,
    operations: &mut Vec<MidOperation>,
) -> MidValueId {
    if state.retarget_automatic_input(value, target.layout.clone())
        && state.get(value).tensor_type.format.precision == target.precision
    {
        return value;
    }
    let original = state.get(value).clone();
    if original.tensor_type.format.precision != target.precision {
        let mut tensor_type = original.tensor_type.clone();
        let from = tensor_type.format.precision;
        tensor_type.format.precision = target.precision;
        let result = state.derived_value(value, tensor_type.clone());
        let memory = conversion_memory_estimate(&original.tensor_type, &tensor_type);
        operations.push(MidOperation {
            source: Some(source),
            inputs: vec![value],
            results: vec![result],
            kind: MidOperationKind::CastPrecision {
                from,
                to: target.precision,
            },
            conversion_plan: Some(ConversionPlan {
                kernel: TileKernelSpec::Cast {
                    from,
                    to: target.precision,
                },
                input: OperandRequirement::new(original.tensor_type.format.clone(), 8),
                output: OperandRequirement::new(tensor_type.format.clone(), 8),
                strategy: ConversionStrategy::LocalKernel,
            }),
            metrics: OperationMetrics {
                cost: CostEstimate {
                    cycles: costs.cast_cycles(&original.tensor_type, target.precision),
                    ..CostEstimate::default()
                },
                memory,
            },
        });
        value = result;
    }
    let current = state.get(value).clone();
    if current.tensor_type.format.layout != target.layout {
        let mut tensor_type = current.tensor_type.clone();
        let from = tensor_type.format.layout.clone();
        tensor_type.format.layout = target.layout.clone();
        let result = state.derived_value(value, tensor_type.clone());
        let strategy = layout_conversion_strategy(&from, &target.layout);
        let rearrangement = costs.rearrangement_cost(
            &tensor_type.shape,
            tensor_type.format.precision,
            strategy,
            &from,
            &target.layout,
        );
        let memory = conversion_memory_estimate(&current.tensor_type, &tensor_type);
        operations.push(MidOperation {
            source: Some(source),
            inputs: vec![value],
            results: vec![result],
            kind: MidOperationKind::Rearrange {
                from: from.clone(),
                to: target.layout.clone(),
            },
            conversion_plan: Some(ConversionPlan {
                kernel: TileKernelSpec::Rearrange {
                    from: from.clone(),
                    to: target.layout.clone(),
                },
                input: OperandRequirement::new(current.tensor_type.format.clone(), 8),
                output: OperandRequirement::new(tensor_type.format.clone(), 8)
                    .with_materialization(materialization),
                strategy,
            }),
            metrics: OperationMetrics {
                cost: rearrangement,
                memory,
            },
        });
        value = result;
    }
    value
}

fn lookup(values: &BTreeMap<ValueId, MidValueId>, value: ValueId) -> LoweringResult<MidValueId> {
    values
        .get(&value)
        .copied()
        .ok_or(LoweringError::UnknownValue(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{AddOptions, AttentionOptions};
    use crate::{AxisTiling, HardwareTarget, LayoutError, TensorTiling};

    const RANDOM_CASES: usize = 128;

    #[test]
    fn randomized_memory_peaks_reserve_disjoint_class_arenas() {
        let mut random = fastrand::Rng::with_seed(0x636c_6173_735f_7372);
        let constraints = HardwareTarget::Ipu21.memory_constraints();
        let capacity = constraints.total_bytes;
        let interleaved_capacity = constraints.interleaved_bytes;
        let element = u64::from(ipu_target::memory::IPU21_INTERLEAVED_ELEMENT_SIZE);
        let mut rejected_noncoincident_peaks = 0;
        for _ in 0..RANDOM_CASES * 16 {
            let standard = random.u64(0..=capacity);
            let interleaved = random.u64(0..=interleaved_capacity);
            let reservation = random.u64(0..=capacity / 4);
            let simultaneous =
                random.u64(standard.max(interleaved)..=standard.saturating_add(interleaved));
            let peaks = MemoryPeaks {
                standard,
                interleaved,
                total: simultaneous,
                maximum_standard_allocation: 0,
                ..MemoryPeaks::default()
            };
            let aligned_interleaved = interleaved.div_ceil(element) * element;
            let static_partition = standard
                .saturating_add(aligned_interleaved)
                .saturating_add(reservation);
            let fits = peaks.fits_with_budget(
                HardwareTarget::Ipu21.memory_constraints(),
                reservation,
                capacity,
            );
            assert_eq!(fits, static_partition <= capacity);
            if simultaneous.saturating_add(reservation) <= capacity && static_partition > capacity {
                rejected_noncoincident_peaks += 1;
                assert!(!fits);
            }
        }
        assert!(rejected_noncoincident_peaks > 0);
    }

    fn dimension(random: &mut fastrand::Rng) -> u32 {
        random.u32(1..=128)
    }

    fn small_dimension(random: &mut fastrand::Rng) -> u32 {
        random.u32(1..=4)
    }

    fn precision(random: &mut fastrand::Rng) -> Precision {
        if random.bool() {
            Precision::F16
        } else {
            Precision::F32
        }
    }

    fn format(precision: Precision, layout: Layout) -> TensorFormat {
        TensorFormat { precision, layout }
    }

    fn random_format(random: &mut fastrand::Rng, tiles: u16) -> TensorFormat {
        let tiling = if random.bool() {
            TensorTiling::replicated(tiles)
        } else {
            TensorTiling::sharded(TensorAxis::FromEnd(2), tiles)
        };
        let mut layout = Layout::row_major(tiling);
        if random.bool() {
            layout.memory_class = MemoryClass::Interleaved;
        }
        format(precision(random), layout)
    }

    #[test]
    fn randomized_future_state_is_id_independent_but_preserves_aliasing() {
        let mut random = fastrand::Rng::with_seed(0x616c_6961_7365_7321);
        for _ in 0..RANDOM_CASES {
            let mut graph = ComputeGraph::new();
            let first = graph.host_input("first", [1]).unwrap();
            let second = graph.host_input("second", [1]).unwrap();
            let dummy = graph.host_input("dummy", [1]).unwrap();
            let tiles = random.u16(1..=64);
            let tensor_type = TensorType {
                shape: TensorShape::new([random.u32(1..=128)]),
                format: random_format(&mut random, tiles),
            };
            let aliases = random.bool();
            let automatic = random.bool();
            let parameter = random.bool();

            let make_branch = |prepend_dummy: bool, aliases: bool| {
                let mut state = LoweringState::default();
                if prepend_dummy {
                    state.value(dummy, tensor_type.clone());
                }
                let first_id = state.value(first, tensor_type.clone());
                let second_id = if aliases {
                    state.value_in_storage_group(second, tensor_type.clone(), first_id)
                } else {
                    state.value(second, tensor_type.clone())
                };
                if automatic {
                    state.automatic_inputs.extend([first_id, second_id]);
                }
                if parameter {
                    state.parameter_values.extend([first_id, second_id]);
                }
                BeamBranch {
                    values: [(first, first_id), (second, second_id)]
                        .into_iter()
                        .collect(),
                    state,
                    operations: Vec::new(),
                    peak_memory: MemoryPeaks::default(),
                }
            };
            let future = [first, second].into_iter().collect();
            let constraints = RegionPlanningConstraints {
                allocation_copies: [(first, random.u32(1..=8))].into_iter().collect(),
                required_equal_formats: vec![(first, second)],
            };
            let baseline = future_beam_state(&make_branch(false, aliases), &future, &constraints);
            let renumbered = future_beam_state(&make_branch(true, aliases), &future, &constraints);
            let changed_aliasing =
                future_beam_state(&make_branch(true, !aliases), &future, &constraints);
            assert_eq!(baseline, renumbered);
            assert_ne!(baseline, changed_aliasing);
        }
    }

    #[test]
    fn randomized_active_tile_candidates_bound_idle_capacity() {
        let mut random = fastrand::Rng::with_seed(0x7469_6c65);
        for _ in 0..RANDOM_CASES {
            let capacity = random.u16(1..=1472);
            let counts = crate::config::candidate_active_tile_counts(capacity);
            assert_eq!(counts[0], capacity);
            assert!(counts.windows(2).all(|pair| pair[0] > pair[1]));
            assert!(counts.iter().all(|&count| count <= capacity));
            assert!(counts[1..].iter().all(|count| count.is_power_of_two()));
            assert_eq!(counts.last(), Some(&1));
        }
        for exponent in 1..=10 {
            let capacity = 1_u16 << exponent;
            assert_eq!(
                crate::config::candidate_active_tile_counts(capacity).len(),
                exponent + 1
            );
        }
    }

    #[test]
    fn randomized_shape_aware_tile_candidates_follow_graph_extents() {
        let mut random = fastrand::Rng::with_seed(0x7368_6170_655f_6772);
        for case in 0..RANDOM_CASES {
            let capacity = random.u16(16..=1472);
            let extent = random.u16(2..=capacity);
            let shape = TensorShape(vec![u32::from(extent), random.u32(1..=4096)]);
            let counts = crate::config::shape_aware_active_tile_counts(capacity, [&shape]);
            let expected = capacity / extent * extent;
            if expected >= capacity.div_ceil(2) && expected < capacity {
                assert!(counts.contains(&expected), "case {case}");
            }
            assert!(counts.iter().all(|&count| {
                count < capacity
                    && count >= capacity.div_ceil(2)
                    && shape.0.iter().any(|&axis| u32::from(count) % axis == 0)
            }));
        }
    }

    #[test]
    fn randomized_search_domains_filter_operator_plan_generation() {
        let mut random = fastrand::Rng::with_seed(0x646f_6d61_696e_2121);
        for case in 0..RANDOM_CASES {
            let capacity = random.u16(2..=64);
            let first = random.u16(1..=capacity);
            let second = random.u16(1..=capacity);
            let precisions = match random.u8(0..3) {
                0 => vec![Precision::F16],
                1 => vec![Precision::F32],
                _ => vec![Precision::F16, Precision::F32],
            };
            let memory_classes = if random.bool() {
                vec![MemoryClass::Standard]
            } else {
                vec![MemoryClass::Interleaved]
            };
            let domain = PlannerSearchDomain::default()
                .with_active_tile_counts([first, second, first, 0, capacity.saturating_add(1)])
                .with_operator_precisions(OperatorClass::Gemm, precisions.clone())
                .with_operator_precisions(OperatorClass::Gelu, precisions.clone())
                .with_operator_precisions(OperatorClass::Add, precisions.clone())
                .with_weight_memory_classes(memory_classes.clone());
            let shape = TensorShape::new([random.u32(1..=64), random.u32(1..=64)]);
            let active = domain.active_tile_counts(capacity, [&shape]);
            let mut expected = vec![first, second];
            expected.dedup();
            assert_eq!(active, expected, "case {case}");

            let seeds = active
                .iter()
                .flat_map(|&tiles| {
                    gemm_seed_plans_for_tile_count(GemmOptions::default(), tiles, &domain)
                })
                .collect::<Vec<_>>();
            for seed in seeds {
                assert!(
                    precisions.contains(&seed.requirements.inputs[0].format.precision),
                    "case {case}"
                );
                assert!(
                    memory_classes
                        .contains(&seed.requirements.inputs[1].format.layout.memory_class),
                    "case {case}"
                );
                assert!(active.contains(&seed.requirements.output.format.layout.tiling.tile_count));
            }

            let selected_precision = precisions[0];
            let tensor = TensorType::new(
                [u32::from(capacity), random.u32(1..=64)],
                selected_precision,
                Layout::row_sharded(capacity),
            );
            let mut config = PipelineConfig::new(capacity).with_search_domain(domain);
            config.resolved_active_tile_counts = active;
            for (operator, class, arity) in [
                (MidOperator::Gelu, OperatorClass::Gelu, 1),
                (
                    MidOperator::Add(AddOptions::default()),
                    OperatorClass::Add,
                    2,
                ),
            ] {
                let inputs = vec![tensor.clone(); arity];
                let plans = pointwise_plans(operator, class, &inputs, &tensor.shape, &config);
                assert!(!plans.is_empty(), "case {case}");
                assert!(plans.iter().all(|plan| {
                    plan.operator == operator
                        && plan.requirements.output.format.precision == selected_precision
                }));
            }
        }
    }

    fn value(lowered: &MidGraph, id: MidValueId) -> &MidValue {
        &lowered.values[id.index() as usize]
    }

    #[test]
    fn randomized_parallel_reduction_plans_cover_uneven_three_axis_grids() {
        let mut random = fastrand::Rng::with_seed(0x7061_7274_6961_6c73);
        let mut distributed_result_cases = 0;
        for _ in 0..RANDOM_CASES {
            let output_columns = AMP_OUTPUT_COLUMN_BLOCK;
            let inner_partitions = random.u16(2..=4);
            let column_partitions = random.u16(1..=4);
            let row_partitions = random.u16(1..=8);
            let tiles = row_partitions * column_partitions * inner_partitions;
            let k = u32::from(inner_partitions) * 64 + random.u32(0..64);
            let n = u32::from(column_partitions) * output_columns + random.u32(0..output_columns);
            let m = u32::from(row_partitions) + random.u32(0..=16);
            let base = amp_grid_gemm_plan(
                GemmOptions::default(),
                Precision::F16,
                16,
                GemmGeometry {
                    block: GemmBlockShape {
                        inner: 64,
                        output_columns,
                    },
                    orientation: GemmOrientation::Normal,
                    result: GemmResultGrid {
                        rows: 1,
                        columns: tiles,
                    },
                    order: GridOrder::ColumnsFast,
                    distribution: GemmDistribution::OutputStationary,
                },
                AmpWeightPlacement::resident(MemoryClass::Standard),
            );
            let inputs = [
                TensorType::new([m, k], Precision::F16, Layout::row_sharded(tiles)),
                TensorType::new([k, n], Precision::F16, Layout::row_sharded(tiles)),
            ];
            let config = PipelineConfig::new(tiles).with_planning_beam_width(16);
            let candidates = parallel_reduction_plans(
                &base,
                &inputs,
                &TensorShape(vec![m, n]),
                &config,
                &Ipu21CostModel,
                true,
                None,
                None,
            );
            assert!(
                !candidates.is_empty(),
                "shape={m}x{k}x{n} tiles={tiles} output_columns={output_columns}"
            );
            distributed_result_cases += usize::from(candidates.iter().any(|candidate| {
                let OperatorDispatch::BlockedGemm(plan) = candidate.dispatch else {
                    return false;
                };
                let GemmDistribution::ParallelReduction(reduction) = plan.geometry.distribution
                else {
                    return false;
                };
                plan.geometry.result
                    != GemmResultGrid {
                        rows: reduction.compute.rows,
                        columns: reduction.compute.columns,
                    }
            }));
            for candidate in candidates {
                assert!(
                    candidate.supports(&inputs, &TensorShape(vec![m, n])),
                    "unsupported candidate: {candidate:?}; shape={m}x{k}x{n}"
                );
                assert!(matches!(candidate.dispatch,
                    OperatorDispatch::BlockedGemm(BlockedGemmPlan {
                        geometry: GemmGeometry {
                            block: GemmBlockShape { inner: inner_block, output_columns: output_column_block },
                            orientation,
                            distribution: GemmDistribution::ParallelReduction(ParallelReductionPlan {
                                compute: GemmGrid { rows: actual_rows, columns: actual_columns, inner: actual },
                                ..
                            }),
                            ..
                        },
                        ..
                    }) if actual_rows * actual_columns * actual <= tiles
                        && actual_rows * actual_columns * actual >= tiles.div_ceil(2)
                        && u32::from(actual_rows) <= match orientation {
                            GemmOrientation::Normal => m,
                            GemmOrientation::Swapped => n,
                        }
                        && u32::from(actual_columns) * output_column_block >= match orientation {
                            GemmOrientation::Normal => n,
                            GemmOrientation::Swapped => m,
                        }
                        && u32::from(actual) * inner_block >= k
                ));
            }
        }
        assert!(distributed_result_cases > 0);
    }

    #[test]
    fn randomized_cycle_model_rewards_direct_interleaved_weight_loads() {
        let mut random = fastrand::Rng::with_seed(0x6379_636c);
        for _ in 0..RANDOM_CASES {
            let rows = 1_u16 << random.u32(0..=2);
            let columns = 1_u16 << random.u32(0..=2);
            let tiles = rows * columns;
            let m = u32::from(rows) * random.u32(1..=4);
            let k = 64 * random.u32(2..=4);
            let n = u32::from(columns) * 64;
            let left = TensorType::new(
                [m, k],
                Precision::F16,
                Layout::amp_left_grid(64, tiles, rows, columns, GridOrder::ColumnsFast),
            );
            let mut standard_layout = Layout::block_major_matrix_grid(
                64,
                64,
                tiles,
                rows,
                columns,
                GridOrder::ColumnsFast,
            );
            let mut direct_layout = standard_layout.clone();
            direct_layout.memory_class = MemoryClass::Interleaved;
            standard_layout.memory_class = MemoryClass::Standard;
            let standard = TensorType::new([k, n], Precision::F16, standard_layout);
            let direct = TensorType::new([k, n], Precision::F16, direct_layout);
            let output = TensorType::new(
                [m, n],
                Precision::F16,
                Layout::amp_output_grid(64, tiles, rows, columns, GridOrder::ColumnsFast),
            );
            let operator = MidOperator::Gemm {
                options: GemmOptions::default(),
                multiply: Precision::F16,
                accumulate: AccumulationPrecision::F32,
            };
            let dispatch = blocked_gemm_dispatch(
                operator,
                GemmGeometry {
                    block: GemmBlockShape {
                        inner: AMP_INNER_BLOCK,
                        output_columns: AMP_OUTPUT_COLUMN_BLOCK,
                    },
                    orientation: GemmOrientation::Normal,
                    result: GemmResultGrid { rows, columns },
                    order: GridOrder::ColumnsFast,
                    distribution: GemmDistribution::OutputStationary,
                },
            );
            let requirements = OperatorRequirements {
                inputs: Vec::new(),
                output: OperandRequirement::new(output.format.clone(), 8),
                output_aliasing: OutputAliasing::Fresh,
                memory_space: MemorySpaceRequirements::default(),
            };
            let standard_cost = Ipu21CostModel.operator_cycles(
                operator,
                &dispatch,
                &requirements,
                &[left.clone(), standard],
                &output,
            );
            let direct_cost = Ipu21CostModel.operator_cycles(
                operator,
                &dispatch,
                &requirements,
                &[left, direct],
                &output,
            );
            assert!(direct_cost < standard_cost);
        }
    }

    #[test]
    fn randomized_parameter_storage_balances_one_copy_independently_of_compute_grids() {
        let mut random = fastrand::Rng::with_seed(0x6f77_6e65_7273);
        for case in 0..RANDOM_CASES {
            let row_partitions = 1_u16 << random.u32(1..=4);
            let column_partitions = 1_u16 << random.u32(0..=4);
            let tiles = row_partitions * column_partitions;
            let inner_blocks = u32::from(row_partitions) * random.u32(1..=4);
            let inner = inner_blocks * AMP_INNER_BLOCK;
            let columns = u32::from(column_partitions) * AMP_OUTPUT_COLUMN_BLOCK;
            let geometry = GemmGeometry {
                block: GemmBlockShape {
                    inner: 64,
                    output_columns: AMP_OUTPUT_COLUMN_BLOCK,
                },
                orientation: GemmOrientation::Normal,
                result: GemmResultGrid {
                    rows: row_partitions,
                    columns: column_partitions,
                },
                order: GridOrder::ColumnsFast,
                distribution: GemmDistribution::OutputStationary,
            };
            let candidate = amp_grid_gemm_plan(
                GemmOptions::default(),
                Precision::F16,
                16,
                geometry,
                AmpWeightPlacement::resident(MemoryClass::Interleaved),
            );
            let inputs = [
                TensorType::new(
                    [u32::from(row_partitions), inner],
                    Precision::F16,
                    candidate.requirements.inputs[0].format.layout.clone(),
                ),
                TensorType::new(
                    [inner, columns],
                    Precision::F16,
                    candidate.requirements.inputs[1].format.layout.clone(),
                ),
            ];
            let variants =
                independent_parameter_storage(&candidate, &inputs, 1, &PipelineConfig::new(tiles));
            assert!(!variants.is_empty(), "case {case}");
            for variant in variants {
                let tiling = &variant.requirements.inputs[1].format.layout.tiling;
                assert_eq!(tiling.replicas, 1, "case {case}");
                assert!(tiling.tile_count <= tiles, "case {case}");
                assert_eq!(
                    tiling.tile_count,
                    tiling.replicas
                        * tiling
                            .axes
                            .iter()
                            .map(|axis| axis.partitions)
                            .product::<u16>(),
                    "case {case}"
                );
                assert!(
                    variant.requirements.inputs[1]
                        .format
                        .layout
                        .resolve(&inputs[1].shape)
                        .is_ok(),
                    "case {case}"
                );
            }
        }
    }

    fn assert_conversions_are_explicit(lowered: &MidGraph, operations: &[MidOperation]) {
        for operation in operations {
            let [input] = operation.inputs.as_slice() else {
                continue;
            };
            let [result] = operation.results.as_slice() else {
                continue;
            };
            let before = &value(lowered, *input).tensor_type;
            let after = &value(lowered, *result).tensor_type;
            match &operation.kind {
                MidOperationKind::CastPrecision { from, to } => {
                    assert_eq!(*from, before.format.precision);
                    assert_eq!(*to, after.format.precision);
                    assert_eq!(before.shape, after.shape);
                    assert_eq!(before.format.layout, after.format.layout);
                }
                MidOperationKind::Rearrange { from, to } => {
                    assert_eq!(from, &before.format.layout);
                    assert_eq!(to, &after.format.layout);
                    assert_eq!(before.shape, after.shape);
                    assert_eq!(before.format.precision, after.format.precision);
                }
                MidOperationKind::Operator(_) | MidOperationKind::Repeat(_) => {}
            }
        }
    }

    struct ColumnParityCost;

    impl CostModel for ColumnParityCost {
        fn operator_cycles(
            &self,
            operator: MidOperator,
            _dispatch: &OperatorDispatch,
            _requirements: &OperatorRequirements,
            _inputs: &[TensorType],
            output: &TensorType,
        ) -> u64 {
            let preferred = if output.shape.0.last().unwrap().is_multiple_of(2) {
                Precision::F16
            } else {
                Precision::F32
            };
            match operator {
                MidOperator::Gemm { multiply, .. } if multiply == preferred => 0,
                MidOperator::Gemm { .. } => 1,
                _ => 0,
            }
        }

        fn cast_cycles(&self, _input: &TensorType, _to: Precision) -> u64 {
            0
        }

        fn rearrangement_cost(
            &self,
            _shape: &TensorShape,
            _precision: Precision,
            _strategy: ConversionStrategy,
            _from: &Layout,
            _to: &Layout,
        ) -> crate::CostEstimate {
            crate::CostEstimate::default()
        }
    }

    #[test]
    fn randomized_axis_tiling_applies_or_rejects_padding() {
        let mut random = fastrand::Rng::with_seed(0x7469_6c65);
        for case in 0..RANDOM_CASES {
            let rank = random.usize(1..=6);
            let axis = random.usize(0..rank);
            let extent = dimension(&mut random);
            let block_size = random.u32(1..=32);
            let partitions = random.u16(1..=16);
            let replicas = random.u16(1..=4);
            let padding = if random.bool() {
                Padding::Reject
            } else {
                Padding::Zero
            };
            let mut shape = (0..rank)
                .map(|_| dimension(&mut random))
                .collect::<Vec<_>>();
            shape[axis] = extent;
            let layout = Layout::row_major(TensorTiling {
                tile_count: partitions * replicas,
                replicas,
                axes: vec![AxisTiling::new(
                    TensorAxis::FromStart(axis as u16),
                    partitions,
                    block_size,
                    padding,
                )],
            });

            let result = layout.resolve(&TensorShape(shape.clone()));
            if padding == Padding::Reject && !extent.is_multiple_of(block_size) {
                assert!(
                    matches!(result, Err(LayoutError::IndivisibleAxis { .. })),
                    "random case {case}"
                );
            } else {
                let resolved = result.unwrap();
                let padded = resolved.padded_shape();
                let expected = extent.div_ceil(block_size) * block_size;
                assert_eq!(padded.0[axis], expected, "random case {case}");
                for (other, original) in shape.iter().enumerate() {
                    if other != axis {
                        assert_eq!(padded.0[other], *original, "random case {case}");
                    }
                }
            }
        }
    }

    #[test]
    fn randomized_gemm_lowering_makes_every_format_boundary_explicit() {
        let mut random = fastrand::Rng::with_seed(0x6d69_6467);
        for case in 0..RANDOM_CASES {
            let tiles = [1, 2, 4, 8, 16][random.usize(0..5)];
            let (rows, inner, columns) = (
                u32::from(tiles) * random.u32(1..=2),
                random.u32(1..=2) * 64,
                random.u32(1..=2) * 64,
            );
            let batches = (0..random.usize(0..=2)).map(|_| 1).collect::<Vec<_>>();
            let multiply = precision(&mut random);
            let mut left_shape = batches.clone();
            left_shape.extend([rows, inner]);
            let mut right_shape = vec![1; batches.len()];
            right_shape.extend([inner, columns]);

            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", left_shape).unwrap();
            let right = graph.parameter("right", right_shape).unwrap();
            let product = graph.gemm(left, right).unwrap();
            graph.set_outputs([product]).unwrap();
            let config = PipelineConfig::new(tiles)
                .with_search_domain(
                    PlannerSearchDomain::default()
                        .with_operator_precisions(OperatorClass::Gemm, [multiply]),
                )
                .with_automatic_input(left, precision(&mut random))
                .with_automatic_input(right, precision(&mut random));

            let lowered = lower(&graph, &config, &Ipu21CostModel).unwrap_or_else(|error| {
                panic!(
                    "random case {case}: tiles={tiles} rows={rows} inner={inner} columns={columns} batches={batches:?}: {error:?}"
                )
            });
            let operator = lowered
                .operations
                .iter()
                .find(|operation| matches!(operation.kind, MidOperationKind::Operator(_)))
                .unwrap();
            let Some(MidOperator::Gemm {
                multiply: selected_multiply,
                accumulate: selected_accumulate,
                ..
            }) = operator.operator()
            else {
                panic!("random case {case}: expected GEMM");
            };
            assert_eq!(selected_multiply, multiply, "random case {case}");
            assert_eq!(
                selected_accumulate,
                gemm_accumulation_precision(multiply),
                "random case {case}"
            );
            let output = value(&lowered, lowered.outputs[0]);
            let expected_shape = graph.value_shape(product).unwrap().clone();
            assert_eq!(
                output.tensor_type.shape, expected_shape,
                "random case {case}"
            );
            assert_eq!(
                output.tensor_type.format.precision, multiply,
                "random case {case}"
            );
            assert_conversions_are_explicit(&lowered, &lowered.operations);
        }
    }

    #[test]
    fn randomized_beam_finalists_have_consistent_costs_and_memory() {
        let mut random = fastrand::Rng::with_seed(0x6265_616d);
        for case in 0..RANDOM_CASES {
            let tiles = [1, 2, 4, 8][random.usize(0..4)];
            let rows = u32::from(tiles) * random.u32(1..=8);
            let inner = random.u32(1..=4) * 64;
            let columns = random.u32(1..=4) * 64;
            let row = format(Precision::F16, Layout::row_sharded(tiles));
            let right = format(
                Precision::F16,
                Layout::block_major_matrix_storage(
                    64,
                    AMP_OUTPUT_COLUMN_BLOCK,
                    tiles,
                    1,
                    1,
                    MemoryClass::Standard,
                ),
            );
            let mut graph = ComputeGraph::new();
            let activation = graph.host_input("activation", [rows, inner]).unwrap();
            let weights = graph.parameter("weights", [inner, columns]).unwrap();
            let activated = graph.gelu(activation).unwrap();
            let product = graph.gemm(activated, weights).unwrap();
            graph.set_outputs([product]).unwrap();

            let make_config = |beam_width| {
                PipelineConfig::new(tiles)
                    .with_search_domain(
                        PlannerSearchDomain::default().with_active_tile_counts([tiles]),
                    )
                    .with_input(activation, row.clone())
                    .with_input(weights, right.clone())
                    .with_planning_beam_width(beam_width)
            };
            let searched_config = make_config(2);
            let finalists = lower_finalists(&graph, &searched_config, &Ipu21CostModel, 2).unwrap();
            assert!(
                !finalists.is_empty() && finalists.len() <= 2,
                "random case {case}"
            );
            for finalist in &finalists {
                assert_eq!(
                    finalist.metrics.cost.cycles,
                    finalist
                        .operations
                        .iter()
                        .map(|operation| operation.metrics.cost.cycles)
                        .sum::<u64>(),
                    "random case {case}"
                );
                assert_eq!(
                    finalist.metrics.cost.exchange_cycles,
                    finalist
                        .operations
                        .iter()
                        .map(|operation| operation.metrics.cost.exchange_cycles)
                        .sum::<u64>(),
                    "random case {case}"
                );
                assert!(
                    finalist.metrics.cost.exchange_cycles <= finalist.metrics.cost.cycles,
                    "random case {case}"
                );
            }
            let searched = &finalists[0];
            assert!(
                searched.metrics.memory.fits_with_budget(
                    searched_config.target.memory_constraints(),
                    searched_config.standard_memory_reservation_bytes,
                    searched_config.tile_memory_budget_bytes,
                ),
                "random case {case}"
            );
        }
    }

    #[test]
    fn randomized_gemm_lowering_rejects_per_batch_weights() {
        let mut random = fastrand::Rng::with_seed(0x6261_7463);
        for _ in 0..RANDOM_CASES {
            let batch = random.u32(2..=8);
            let rows = random.u32(1..=8);
            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", [batch, rows, 64]).unwrap();
            let right = graph.parameter("right", [batch, 64, 64]).unwrap();
            let output = graph.gemm(left, right).unwrap();
            graph.set_outputs([output]).unwrap();
            let config = PipelineConfig::new(1)
                .with_automatic_input(left, Precision::F16)
                .with_automatic_input(right, Precision::F16);
            assert!(matches!(
                lower(&graph, &config, &Ipu21CostModel),
                Err(LoweringError::UnsupportedGemmBatching(_))
            ));
        }
    }

    #[test]
    fn randomized_gemms_choose_precision_independently_within_one_graph() {
        let mut random = fastrand::Rng::with_seed(0x6d75_6c74);
        for case in 0..RANDOM_CASES / 4 {
            let tiles = random.u16(1..=64);
            let rows = u32::from(tiles) * small_dimension(&mut random);
            let inner = random.u32(1..=64);
            let even_columns = random.u32(1..=16) * 2;
            let odd_columns = random.u32(1..=16) * 2 - 1;
            let layout = Layout::row_sharded(tiles);
            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", [rows, inner]).unwrap();
            let even_right = graph.parameter("even", [inner, even_columns]).unwrap();
            let odd_right = graph.parameter("odd", [inner, odd_columns]).unwrap();
            let even = graph.gemm(left, even_right).unwrap();
            let odd = graph.gemm(left, odd_right).unwrap();
            graph.set_outputs([even, odd]).unwrap();
            let input_format = format(precision(&mut random), layout);
            let config = PipelineConfig::new(tiles)
                .with_input(left, input_format.clone())
                .with_input(even_right, input_format.clone())
                .with_input(odd_right, input_format);

            let lowered = lower(&graph, &config, &ColumnParityCost).unwrap();
            let chosen = lowered
                .operations
                .iter()
                .filter_map(|operation| match operation.operator() {
                    Some(MidOperator::Gemm { multiply, .. }) => Some(multiply),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                chosen,
                vec![Precision::F16, Precision::F32],
                "random case {case}"
            );
            for operation in lowered
                .operations
                .iter()
                .filter(|operation| matches!(operation.operator(), Some(MidOperator::Gemm { .. })))
            {
                let requirements = &operation.operator_plan().unwrap().requirements;
                assert!(
                    requirements
                        .inputs
                        .iter()
                        .chain([&requirements.output])
                        .all(|requirement| requirement.allocation.alignment == 32)
                );
                assert_eq!(
                    requirements.output.format.layout.memory_class,
                    MemoryClass::Interleaved
                );
                let orientation = match operation.operator_plan().map(|plan| &plan.dispatch) {
                    Some(OperatorDispatch::BlockedGemm(plan)) => plan.geometry.orientation,
                    _ => unreachable!(),
                };
                let physical_left = match orientation {
                    GemmOrientation::Normal => 0usize,
                    GemmOrientation::Swapped => 1usize,
                };
                assert_eq!(
                    requirements.memory_space.distinct_element_groups,
                    [vec![
                        MemoryOperand::Output,
                        MemoryOperand::Input(physical_left as u16),
                    ]]
                );
                let expected_tail = match operation.operator() {
                    Some(MidOperator::Gemm {
                        multiply: Precision::F16,
                        ..
                    }) => 16,
                    Some(MidOperator::Gemm {
                        multiply: Precision::F32,
                        ..
                    }) => 32,
                    _ => unreachable!(),
                };
                assert_eq!(
                    requirements.inputs[0].allocation.access_tail_bytes,
                    expected_tail
                );
            }
        }
    }

    #[test]
    fn randomized_non_gemm_lowering_honors_operator_plans() {
        let mut random = fastrand::Rng::with_seed(0x6164_642b);
        for case in 0..RANDOM_CASES {
            let tiles = random.u16(1..=64);
            let batch = random.u32(1..=2);
            let query_rows = u32::from(tiles) * random.u32(1..=2);
            let key_rows = random.u32(1..=8);
            let channels = random.u32(1..=8);
            let value_channels = random.u32(1..=8);
            let mut graph = ComputeGraph::new();
            let activation = graph
                .host_input("activation", [batch, query_rows, channels])
                .unwrap();
            let residual = graph
                .host_input("residual", [batch, query_rows, channels])
                .unwrap();
            let query = graph
                .host_input("query", [batch, query_rows, channels])
                .unwrap();
            let key = graph
                .host_input("key", [batch, key_rows, channels])
                .unwrap();
            let attention_value = graph
                .host_input("value", [batch, key_rows, value_channels])
                .unwrap();
            let activated = graph.gelu(activation).unwrap();
            let sum = graph.add(activated, residual).unwrap();
            let attended = graph.flash_attention(query, key, attention_value).unwrap();
            graph.set_outputs([sum, attended]).unwrap();

            let config = PipelineConfig::new(tiles)
                .with_automatic_input(activation, Precision::F16)
                .with_automatic_input(residual, Precision::F16)
                .with_automatic_input(query, Precision::F16)
                .with_automatic_input(key, Precision::F16)
                .with_automatic_input(attention_value, Precision::F16);

            let lowered = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let operators = lowered
                .operations
                .iter()
                .filter(|operation| matches!(operation.kind, MidOperationKind::Operator(_)))
                .collect::<Vec<_>>();
            assert_eq!(operators.len(), 3, "random case {case}");
            let gelu = operators
                .iter()
                .copied()
                .find(|operation| matches!(operation.operator(), Some(MidOperator::Gelu)))
                .expect("random graph retains its GeLU");
            let add = operators
                .iter()
                .copied()
                .find(|operation| matches!(operation.operator(), Some(MidOperator::Add(_))))
                .expect("random graph retains its add");
            let attention = operators
                .iter()
                .copied()
                .find(|operation| {
                    matches!(
                        operation.operator(),
                        Some(MidOperator::FlashAttention { .. })
                    )
                })
                .expect("random graph retains its attention");
            assert_eq!(
                value(&lowered, gelu.results[0])
                    .tensor_type
                    .format
                    .precision,
                Precision::F16
            );
            assert_eq!(
                gelu.operator_plan().unwrap().requirements.output_aliasing,
                OutputAliasing::MayAliasInputs(vec![0])
            );
            assert_eq!(
                value(&lowered, add.results[0]).tensor_type.format.precision,
                Precision::F16
            );
            assert_eq!(
                add.operator_plan().unwrap().requirements.output_aliasing,
                OutputAliasing::MayAliasInputs(vec![0, 1])
            );
            assert!(matches!(
                attention.operator(),
                Some(MidOperator::FlashAttention { options, .. })
                    if options == AttentionOptions::default()
            ));
            assert_eq!(
                value(&lowered, attention.results[0]).tensor_type.shape.0,
                vec![batch, query_rows, value_channels],
                "random case {case}"
            );
            assert_conversions_are_explicit(&lowered, &lowered.operations);
        }
    }

    #[test]
    fn randomized_repeat_lowering_retains_sequences_without_unrolling() {
        let mut random = fastrand::Rng::with_seed(0x7265_7065);
        for case in 0..RANDOM_CASES {
            let tiles = random.u16(1..=64);
            let size = u32::from(tiles);
            let count = random.u32(1..=12);
            let layout = Layout::row_sharded(tiles);
            let carried_format = format(precision(&mut random), layout.clone());
            let mut graph = ComputeGraph::new();
            let carried = graph.host_input("state", [size, size]).unwrap();
            let weights = (0..count)
                .map(|index| graph.parameter(format!("weight.{index}"), [size, size]))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let sequence = graph.value_sequence("weights", weights.clone()).unwrap();
            let output = graph
                .repeat(count, [carried], [], [sequence], |body, arguments| {
                    Ok(vec![
                        body.gemm(arguments.carried[0], arguments.iterated[0])?,
                    ])
                })
                .unwrap()[0];
            graph.set_outputs([output]).unwrap();
            let mut config = PipelineConfig::new(tiles).with_input(carried, carried_format.clone());
            for weight in weights {
                config
                    .inputs
                    .insert(weight, format(precision(&mut random), layout.clone()));
            }

            let lowered = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let repeat = lowered
                .operations
                .iter()
                .find_map(|operation| match &operation.kind {
                    MidOperationKind::Repeat(repeat) => Some(repeat),
                    _ => None,
                })
                .unwrap();
            assert_eq!(repeat.count, count, "random case {case}");
            assert_eq!(repeat.iterated_inputs.len(), 1, "random case {case}");
            assert_eq!(
                repeat.iterated_inputs[0].len(),
                count as usize,
                "random case {case}"
            );
            let sequence_format = &value(&lowered, repeat.iterated_inputs[0][0])
                .tensor_type
                .format;
            assert!(repeat.iterated_inputs[0].iter().all(|value_id| {
                &value(&lowered, *value_id).tensor_type.format == sequence_format
            }));
            assert_eq!(
                &value(&lowered, repeat.body.yields[0]).tensor_type.format,
                &carried_format,
                "random case {case}"
            );
            assert_eq!(
                &value(&lowered, lowered.outputs[0]).tensor_type.format,
                &carried_format,
                "random case {case}"
            );
            assert_conversions_are_explicit(&lowered, &lowered.operations);
            assert_conversions_are_explicit(&lowered, &repeat.body.operations);
        }
    }

    #[test]
    fn randomized_single_use_views_are_claimed_by_slice_consumers() {
        let mut random = fastrand::Rng::with_seed(0x6465_6665_7272_6564);
        for case in 0..RANDOM_CASES / 32 {
            let heads = random.u32(2..=6);
            let head_width = random.u32(4..=40) * 2;
            let tokens = random.u32(1..=3) * AMP_INNER_BLOCK;
            let model_width = heads * head_width;
            let tiles = u16::try_from(heads * tokens.div_ceil(AMP_INNER_BLOCK)).unwrap();
            let mut graph = ComputeGraph::new();
            let input = graph.host_input("input", [1, tokens, model_width]).unwrap();
            let mut projected = Vec::new();
            let mut parameters = Vec::new();
            for index in 0..3 {
                let weights = graph
                    .parameter(format!("projection.{index}"), [model_width, model_width])
                    .unwrap();
                parameters.push(weights);
                projected.push(graph.gemm(input, weights).unwrap());
            }
            let split = projected
                .iter()
                .map(|&value| graph.split_heads(value, heads).unwrap())
                .collect::<Vec<_>>();
            let output = graph.flash_attention(split[0], split[1], split[2]).unwrap();
            graph.set_outputs([output]).unwrap();
            let mut config = PipelineConfig::new(tiles).with_automatic_input(input, Precision::F16);
            for parameter in parameters {
                config = config.with_automatic_input(parameter, Precision::F16);
            }
            config.conversion_streaming = ConversionStreamingPolicy::Always;

            let lowered = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let producers = lowered
                .operations
                .iter()
                .filter(|operation| {
                    matches!(operation.operator(), Some(MidOperator::SplitHeads(_)))
                })
                .collect::<Vec<_>>();
            assert_eq!(producers.len(), split.len(), "random case {case}");
            assert!(
                producers.iter().all(|operation| {
                    operation.metrics.cost.cycles == 0
                        && operation
                            .operator_plan()
                            .is_some_and(|plan| plan.deferred_output.is_some())
                }),
                "random case {case}"
            );
            let consumer = lowered
                .operations
                .iter()
                .find(|operation| {
                    matches!(
                        operation.operator(),
                        Some(MidOperator::FlashAttention { .. })
                    )
                })
                .unwrap();
            let claims = &consumer.operator_plan().unwrap().deferred_inputs;
            assert_eq!(claims.len(), split.len(), "random case {case}");
            assert!(claims.iter().all(Option::is_some), "random case {case}");
            assert!(
                consumer.metrics.cost.exchange_footprint.phases >= 2,
                "random case {case}"
            );
            assert!(
                consumer.metrics.cost.exchange_row_bytes() != 0,
                "random case {case}"
            );
            assert_eq!(
                lowered.metrics.cost.cycles,
                lowered
                    .operations
                    .iter()
                    .map(|operation| operation.metrics.cost.cycles)
                    .sum::<u64>(),
                "random case {case}"
            );
            let tiled = crate::low::lower_to_tiles(&lowered, &config)
                .unwrap_or_else(|error| {
                    panic!(
                        "random case {case}, heads {heads}, width {head_width}, tokens {tokens}: {error}"
                    )
                });
            crate::KernelBuildPlan::from_program(&tiled)
                .unwrap_or_else(|error| panic!("random case {case}: {error}"));
            let attention_phases = tiled
                .exchange_phases
                .iter()
                .filter(|phase| phase.provenance.operation == consumer.source)
                .count();
            assert!(
                attention_phases <= tokens.div_ceil(AMP_INNER_BLOCK) as usize + 2,
                "random case {case}: {attention_phases} attention exchange phases"
            );
        }
    }

    #[test]
    fn randomized_unclaimed_deferred_offers_restore_materialization_cost() {
        let mut random = fastrand::Rng::with_seed(0x756e_636c_6169_6d65);
        for case in 0..RANDOM_CASES / 8 {
            let batch = random.u32(1..=4);
            let heads = random.u32(1..=8);
            let rows = random.u32(1..=4) * AMP_INNER_BLOCK;
            let head_width = random.u32(1..=4) * AMP_COLUMN_MICRO;
            let mut graph = ComputeGraph::new();
            let input = graph
                .host_input("input", [batch, rows, heads * head_width])
                .unwrap();
            let output = graph.split_heads(input, heads).unwrap();
            graph.set_outputs([output]).unwrap();
            let tiles = u16::try_from(batch * heads).unwrap();
            let config = PipelineConfig::new(tiles).with_automatic_input(input, Precision::F16);

            let lowered = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let operation = lowered
                .operations
                .iter()
                .find(|operation| matches!(operation.operator(), Some(MidOperator::SplitHeads(_))))
                .unwrap();
            let offer = operation
                .operator_plan()
                .and_then(|plan| plan.deferred_output)
                .unwrap();
            assert_eq!(
                operation.metrics.cost.cycles, offer.unfused_cost.cycles,
                "random case {case}"
            );
            assert!(operation.metrics.cost.cycles != 0, "random case {case}");
        }
    }
}
