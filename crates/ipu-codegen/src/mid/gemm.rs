use super::{StorageOrderCompatibility, storage_order_compatibility};
use crate::OperatorSchedule;
use crate::config::{OperatorClass, PipelineConfig, PlannerSearchDomain};
use crate::cost::{CostModel, operator_memory_estimate, parallel_reduction_preselection_metrics};
use crate::graph::{GemmOptions, Operation, OperationKind, TensorShape, ValueId};
use crate::layout::{
    AMP_COLUMN_MICRO, AMP_INNER_BLOCK, AMP_NARROW_OUTPUT_COLUMN_BLOCK, AMP_OUTPUT_COLUMN_BLOCK,
    AMP_WIDE_OUTPUT_COLUMN_BLOCK, Layout, MemoryClass, Padding, StorageOrder, TensorAxis,
    TensorFormat, TensorType,
};
use crate::metrics::{CostEstimate, RegionMetrics, pareto_frontier};
use crate::operator::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GroupedOutputLayout {
    pub groups: u16,
    pub physical_lane_multiple: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ParallelGridCandidate {
    metrics: RegionMetrics,
    grid: GemmGrid,
    physical_column_groups: u16,
    grouped: bool,
}

pub(super) fn gemm_seed_plans_for_tile_count(
    options: GemmOptions,
    tile_count: u16,
    domain: &PlannerSearchDomain,
) -> Vec<OperatorPlan> {
    let candidates = (1..=tile_count)
        .rev()
        .filter(|columns| tile_count.is_multiple_of(*columns))
        .flat_map(|columns| {
            let rows = tile_count / columns;
            let geometry = GemmGeometry {
                block: GemmBlockShape {
                    inner: AMP_INNER_BLOCK,
                    output_columns: AMP_OUTPUT_COLUMN_BLOCK,
                },
                orientation: GemmOrientation::Normal,
                compute: GemmGrid {
                    rows,
                    columns,
                    inner: 1,
                },
                result: GemmResultGrid { rows, columns },
                order: GridOrder::ColumnsFast,
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
                    if output_columns < AMP_OUTPUT_COLUMN_BLOCK
                        && !(weights.inner_partitions == 1
                            && weights.memory_class == MemoryClass::Interleaved)
                    {
                        continue;
                    }
                    let mut geometry = geometry;
                    geometry.block.output_columns = output_columns;
                    let candidate = amp_grid_gemm_plan(
                        options,
                        precision,
                        left_tail,
                        geometry,
                        output_columns,
                        weights,
                    );
                    grid.push(candidate.clone());
                    if columns == 1
                        && rows == tile_count
                        && output_columns > AMP_OUTPUT_COLUMN_BLOCK
                        && weights == AmpWeightPlacement::resident(MemoryClass::Standard)
                    {
                        grid.push(amp_grid_gemm_plan(
                            options,
                            precision,
                            left_tail,
                            geometry,
                            AMP_OUTPUT_COLUMN_BLOCK,
                            weights,
                        ));
                    }
                    if precision == Precision::F16 && weights.memory_class == MemoryClass::Standard
                    {
                        let mut staged = candidate;
                        staged.requirements.inputs[1].local_staging =
                            LocalOperandStaging::Staged(MemoryClass::Interleaved);
                        grid.push(staged);
                    }
                }
            }
            grid
        })
        .collect::<Vec<_>>();
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

pub(super) fn amp_grid_gemm_plan(
    options: GemmOptions,
    precision: Precision,
    left_tail: u32,
    geometry: GemmGeometry,
    storage_column_block: u32,
    weights: AmpWeightPlacement,
) -> OperatorPlan {
    let inner = u16::try_from(geometry.block.inner).unwrap_or(0);
    let grid = geometry.result;
    let right_layout = match (weights.inner_partitions, weights.memory_class) {
        (1, MemoryClass::Standard) => Layout::block_major_matrix_grid(
            inner,
            storage_column_block,
            grid.tile_count(),
            grid.rows,
            grid.columns,
            geometry.order,
        ),
        (inner_partitions, memory_class) => Layout::block_major_matrix_storage(
            GemmOrientation::Normal,
            inner,
            storage_column_block,
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
        blocked_gemm_schedule(operator, geometry),
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
                )
                .with_local_staging(LocalOperandStaging::Direct(
                    if precision == Precision::F16 && weights.inner_partitions > 1 {
                        MemoryClass::Interleaved
                    } else {
                        weights.memory_class
                    },
                )),
            ],
            output: OperandRequirement::new(
                TensorFormat {
                    precision,
                    layout: if precision == Precision::F16 {
                        Layout::amp_left_result_grid(
                            GemmOrientation::Normal,
                            storage_column_block,
                            grid.tile_count(),
                            grid.rows,
                            grid.columns,
                            geometry.order,
                        )
                    } else {
                        Layout::amp_output_grid(
                            GemmOrientation::Normal,
                            storage_column_block,
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
    )
}

pub(super) fn grouped_output_layout(
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

pub(super) fn blocked_gemm_schedule(
    operator: MidOperator,
    geometry: GemmGeometry,
) -> OperatorSchedule {
    let MidOperator::Gemm {
        multiply,
        accumulate,
        ..
    } = operator
    else {
        unreachable!("blocked GEMM schedule requires a GEMM operator")
    };
    OperatorSchedule::blocked_gemm(BlockedGemmPlan {
        kernel: GemmKernelFamily {
            multiply,
            accumulate,
            weights: GemmWeightLoad::Standard,
        },
        geometry,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct AmpWeightPlacement {
    pub inner_partitions: u16,
    pub memory_class: MemoryClass,
}

impl AmpWeightPlacement {
    pub const fn resident(memory_class: MemoryClass) -> Self {
        Self::sharded(1, memory_class)
    }

    pub const fn sharded(inner_partitions: u16, memory_class: MemoryClass) -> Self {
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

pub(super) const fn gemm_accumulation_precision(precision: Precision) -> AccumulationPrecision {
    match precision {
        Precision::F16 | Precision::F8F143 { .. } => AccumulationPrecision::F16,
        Precision::F32 => AccumulationPrecision::F32,
    }
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
pub(super) fn gemm_plans(
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

pub(super) fn gemm_plan_matches(
    constraint: &GemmPlanConstraint,
    schedule: &OperatorSchedule,
    inputs: &[OperandRequirement],
) -> bool {
    let Some(plan) = schedule.gemm_plan() else {
        return false;
    };
    if plan.geometry.compute.inner < 2 {
        return false;
    }
    let weight_index = plan.geometry.orientation.physical_right_input();
    let Some(weight) = inputs.get(weight_index) else {
        return false;
    };
    plan.geometry == constraint.geometry
        && schedule.reduction_staging() == constraint.reduction_staging
        && weight.format.layout.memory_class == constraint.weight_memory_class
        && weight.local_staging == constraint.local_weight_staging
}

pub(super) fn independent_parameter_storage(
    candidate: &OperatorPlan,
    inputs: &[TensorType],
    input_index: usize,
    config: &PipelineConfig,
) -> Vec<OperatorPlan> {
    if candidate.schedule.gemm_plan().is_none() {
        return Vec::new();
    }
    let Some(requirement) = candidate.requirements.inputs.get(input_index) else {
        return Vec::new();
    };
    let StorageOrder::Blocked(order) = requirement.format.layout.order else {
        return Vec::new();
    };
    if !order.is_matrix() {
        return Vec::new();
    }
    let inner_block = order.block_shape[0];
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
    let output_column_block = candidate
        .schedule
        .gemm_plan()
        .map(|plan| plan.geometry.block.output_columns)
        .unwrap_or(0);
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
                    GemmOrientation::Normal,
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

pub(super) fn parallel_reduction_plans(
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
    let Some(plan) = candidate.schedule.gemm_plan().copied() else {
        return Vec::new();
    };
    if plan.geometry.compute.inner != 1 {
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
    let [rows, columns] = orientation.physical_order([normal_rows, normal_columns]);
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
        || candidate.requirements.inputs[1]
            .local_staging
            .stages_local()
    {
        return Vec::new();
    }
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
                let local_columns =
                    u32::from(physical_column_groups).div_ceil(u32::from(column_partitions));
                let local_inner = u32::from(inner_groups).div_ceil(u32::from(inner_partitions));
                if u32::from(inner_partitions - 1).saturating_mul(local_inner)
                    >= u32::from(inner_groups)
                {
                    continue;
                }
                let grid = GemmGrid {
                    rows: row_partitions,
                    columns: column_partitions,
                    inner: inner_partitions,
                };
                let block = GemmBlockShape {
                    inner: local_inner.saturating_mul(AMP_COLUMN_MICRO),
                    output_columns: local_columns.saturating_mul(AMP_COLUMN_MICRO),
                };
                let Some(metrics) = parallel_reduction_preselection_metrics(
                    config.target,
                    block,
                    grid,
                    orientation,
                    inputs,
                    candidate.requirements.output.format.precision,
                ) else {
                    continue;
                };
                let constraints = config.target.memory_constraints();
                if !metrics.memory.fits(constraints) {
                    continue;
                }
                grids.push(ParallelGridCandidate {
                    metrics,
                    grid,
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
                orientation == constraint.geometry.orientation
                    && grid.grid == constraint.geometry.compute
            })
            .collect::<Vec<_>>()
    } else {
        // Inner partitioning and grouped outputs select different lowering
        // families. Within each family the shared metrics vocabulary retains
        // cycle, exchange, and memory tradeoffs for precise evaluation below.
        let (mut frontier, _) = pareto_frontier(
            grids,
            |grid| (grid.grid.inner, grid.grouped),
            |grid| grid.metrics,
        );
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
                        orientation,
                        kernel_inner_block_u16,
                        used_tiles,
                        row_partitions,
                        column_partitions,
                        inner_partitions,
                    );
                    variant.requirements.inputs[1].format.layout =
                        Layout::block_major_matrix_storage(
                            orientation,
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
                        orientation,
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
                    physical_left.format.layout = Layout::amp_left_parallel_grid(
                        orientation,
                        kernel_inner_block_u16,
                        used_tiles,
                        row_partitions,
                        column_partitions,
                        inner_partitions,
                    );
                    physical_left.materialization = OperandMaterialization::DispatchSlices;
                    let mut physical_right = variant.requirements.inputs[0].clone();
                    physical_right.format.layout = Layout::block_major_matrix_storage(
                        orientation,
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
                    variant.requirements.output.format.layout = Layout::amp_left_result_grid(
                        orientation,
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
            if let Some(plan) = variant.schedule.gemm_plan_mut() {
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
                    compute: GemmGrid {
                        rows: row_partitions,
                        columns: column_partitions,
                        inner: inner_partitions,
                    },
                    result: GemmResultGrid {
                        rows: row_partitions,
                        columns: column_partitions,
                    },
                    order: GridOrder::ColumnsFast,
                };
            }
            variant
                .schedule
                .set_reduction_staging(ReductionStaging::Complete);
            let physical_right_index = orientation.physical_right_input();
            let local_staging_options: &[_] = match orientation {
                GemmOrientation::Normal => &[LocalOperandStaging::Direct(MemoryClass::Interleaved)],
                GemmOrientation::Swapped => &[
                    LocalOperandStaging::Direct(MemoryClass::Interleaved),
                    LocalOperandStaging::Staged(MemoryClass::Interleaved),
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
                    if let Some(plan) = result_variant.schedule.gemm_plan_mut()
                        && plan.geometry.compute.inner > 1
                    {
                        plan.geometry.result.rows = result_rows;
                        plan.geometry.result.columns = result_columns;
                        plan.geometry.order = grid_order;
                    }
                    let mut result_layout = Layout::amp_left_result_grid(
                        orientation,
                        result_column_block,
                        result_rows.saturating_mul(result_columns),
                        result_rows,
                        result_columns,
                        grid_order,
                    );
                    let physical_column_axis = orientation.column_axis();
                    balance_parallel_gemm_columns(&mut result_layout, physical_column_axis);
                    result_variant.requirements.output.format.layout = result_layout;
                    result_layout_variants.push(result_variant);
                }
            }
            let mut layout_variants = Vec::new();
            for mut result_layout in result_layout_variants {
                let physical_row_axis = orientation.row_axis();
                let physical_rows =
                    [normal_rows, normal_columns][orientation.physical_left_input()];
                let physical_left_index = orientation.physical_left_input();
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
                    staged
                        .schedule
                        .set_reduction_staging(ReductionStaging::Streamed);
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
                    &candidate.schedule,
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
    inputs: Vec<(StorageOrderCompatibility, MemoryClass, LocalOperandStaging)>,
    output: (
        StorageOrderCompatibility,
        MemoryClass,
        Vec<(TensorAxis, u16, u32)>,
    ),
}

fn gemm_plan_compatibility(candidate: &OperatorPlan) -> GemmPlanCompatibility {
    let (orientation, reduction_staging) = match candidate.schedule.gemm_plan() {
        Some(plan) => (
            Some(plan.geometry.orientation),
            candidate.schedule.reduction_staging(),
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
                    storage_order_compatibility(input.format.layout.order),
                    input.format.layout.memory_class,
                    input.local_staging,
                )
            })
            .collect(),
        output: (
            storage_order_compatibility(candidate.requirements.output.format.layout.order),
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
                &candidate.schedule,
                &candidate.requirements,
                &planned_inputs,
                &planned_output,
            );
            let exchange = costs.operator_exchange_footprint(
                candidate.operator,
                &candidate.schedule,
                &candidate.requirements,
                &planned_inputs,
                &planned_output,
            );
            let objective = RegionMetrics {
                cost: CostEstimate {
                    cycles: costs.operator_cycles(
                        candidate.operator,
                        &candidate.schedule,
                        &candidate.requirements,
                        &planned_inputs,
                        &planned_output,
                    ),
                    exchange_footprint: exchange,
                    ..CostEstimate::default()
                },
                memory: memory.peaks(exchange.estimated_row_bytes(costs.target())),
            };
            let compatibility = gemm_plan_compatibility(&candidate);
            (candidate, objective, compatibility)
        })
        .collect::<Vec<_>>();
    let (mut ranked, _) = pareto_frontier(
        ranked,
        |(_, _, compatibility)| compatibility.clone(),
        |(_, objective, _)| *objective,
    );
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
