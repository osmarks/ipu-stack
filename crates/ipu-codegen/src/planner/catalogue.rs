//! Concrete implementations and explicitly enabled shape-dependent families.

use crate::graph::{AttentionOptions, GemmOptions};
use crate::kernel::AccumulationPrecision;
use crate::tensor::MemoryClass;
use crate::tensor::Precision;

use crate::planner::operator::{
    GemmDistribution, GemmOrientation, OperandMaterialization, OperandRequirement,
    OperatorDispatch, OperatorFamily, OperatorPlan, default_dispatch,
};
use crate::tensor::{
    AMP_INNER_BLOCK, AMP_OUTPUT_COLUMN_BLOCK, GridOrder, Layout, TensorFormat, TensorShape,
};

/// An explicitly enabled concrete implementation or shape-dependent family.
// Most entries are concrete; keep their storage inline as in the original catalogue.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OperatorCandidate {
    Concrete(ConcreteOperatorCandidate),
    /// GEMMs with independently distributed K partials and a reduction.
    ParallelGemm {
        tile_count: u16,
        multiply: Precision,
        options: GemmOptions,
        accumulate: AccumulationPrecision,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConcreteOperatorCandidate {
    pub plan: OperatorPlan,
    pub format_policy: OperatorFormatPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorFormatPolicy {
    /// Use the candidate's concrete input and output formats.
    Concrete,
    /// Fill a row-major row/column grid, using rows before splitting columns.
    RowMajorGrid,
    /// Distribute complete rows over all leading axes, including batches.
    RowMajorRows,
    /// Resolve both the selected input and output to the input value's full
    /// layout. This is the normal policy for layout-transparent unary work.
    PreserveInputLayout(u16),
}

impl OperatorCandidate {
    pub fn parallel_gemm(tile_count: u16) -> Self {
        Self::ParallelGemm {
            tile_count,
            multiply: Precision::F16,
            options: GemmOptions::default(),
            accumulate: gemm_accumulation_precision(Precision::F16),
        }
    }

    pub fn fp8_gemm(tile_count: u16, scale_exponent: i8) -> Self {
        Self::ParallelGemm {
            tile_count,
            multiply: Precision::F8F143 { scale_exponent },
            options: GemmOptions::default(),
            accumulate: AccumulationPrecision::F16,
        }
    }

    pub fn concrete(&self) -> Option<&ConcreteOperatorCandidate> {
        match self {
            Self::Concrete(candidate) => Some(candidate),
            _ => None,
        }
    }

    pub fn operator(&self) -> OperatorFamily {
        match self {
            Self::Concrete(candidate) => candidate.plan.operator,
            Self::ParallelGemm {
                options,
                multiply,
                accumulate,
                ..
            } => OperatorFamily::Gemm {
                options: *options,
                multiply: *multiply,
                accumulate: *accumulate,
            },
        }
    }

    pub fn format_policy(&self) -> OperatorFormatPolicy {
        self.concrete()
            .map_or(OperatorFormatPolicy::Concrete, |candidate| {
                candidate.format_policy
            })
    }
}

impl ConcreteOperatorCandidate {
    pub fn new(
        operator: OperatorFamily,
        inputs: impl IntoIterator<Item = OperandRequirement>,
        output: TensorFormat,
    ) -> Self {
        Self {
            plan: OperatorPlan {
                operator,
                dispatch: default_dispatch(operator),
                inputs: inputs.into_iter().collect(),
                output,
                reuse_inputs: None,
            },
            format_policy: OperatorFormatPolicy::Concrete,
        }
    }

    pub fn with_dispatch(mut self, dispatch: OperatorDispatch) -> Self {
        self.plan.dispatch = dispatch;
        self
    }

    pub fn with_preserved_input_layout(mut self, input: u16) -> Self {
        self.format_policy = OperatorFormatPolicy::PreserveInputLayout(input);
        self
    }

    pub fn with_reusable_inputs(mut self, inputs: impl IntoIterator<Item = u16>) -> Self {
        self.plan.reuse_inputs = Some(inputs.into_iter().collect());
        self
    }
}

pub(crate) fn default_operator_candidates(tile_count: u16) -> Vec<OperatorCandidate> {
    // Each group is unique, and candidates from different groups retain
    // different tile counts in their layouts or family parameters.
    candidate_active_tile_counts(tile_count)
        .into_iter()
        .flat_map(operator_candidates_for_tile_count)
        .collect()
}

pub(super) fn candidate_active_tile_counts(capacity: u16) -> Vec<u16> {
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

pub(super) fn shape_aware_active_tile_counts<'a>(
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

pub(crate) fn operator_candidates_for_tile_count(tile_count: u16) -> Vec<OperatorCandidate> {
    let amp_left_result_f16 = TensorFormat {
        precision: Precision::F16,
        layout: Layout::amp_left_result(tile_count),
    };
    let rows_f16 = TensorFormat {
        precision: Precision::F16,
        layout: Layout::row_sharded(tile_count),
    };
    let heads_f16 = TensorFormat {
        precision: Precision::F16,
        layout: Layout::head_sharded(tile_count),
    };
    let mut candidates = (1..=tile_count)
        .rev()
        .filter(|columns| tile_count.is_multiple_of(*columns))
        .flat_map(|columns| {
            let rows = tile_count / columns;
            let grid_shape = AmpGridShape {
                tile_count,
                row_partitions: rows,
                column_partitions: columns,
                order: GridOrder::ColumnsFast,
            };
            let mut grid = Vec::new();
            let mut placements = vec![
                AmpWeightPlacement::resident(MemoryClass::Ipu21Standard),
                AmpWeightPlacement::resident(MemoryClass::Ipu21Interleaved),
            ];
            if rows > 1 {
                placements.extend([
                    AmpWeightPlacement::sharded(rows, MemoryClass::Ipu21Standard),
                    AmpWeightPlacement::sharded(rows, MemoryClass::Ipu21Interleaved),
                ]);
            }
            // Two-way F16 interleaving lets each peer retain half of a full
            // kernel-width column shard. Keep the automatic search bounded;
            // explicit layouts may use any divisor of the row grid.
            if rows > 2 && rows.is_multiple_of(2) {
                placements.push(AmpWeightPlacement::sharded(
                    2,
                    MemoryClass::Ipu21Interleaved,
                ));
            }
            for weights in placements {
                for output_columns in [AMP_OUTPUT_COLUMN_BLOCK, 128, 32] {
                    // A narrow resident interleaved shard can avoid streaming
                    // when a 64-column shard would exceed region capacity.
                    // Narrow streamed panels increase multicast-role pressure
                    // and are not offered until ownership is part of the cost.
                    if output_columns < AMP_OUTPUT_COLUMN_BLOCK
                        && !(weights.inner_partitions == 1
                            && weights.memory_class == MemoryClass::Ipu21Interleaved)
                    {
                        continue;
                    }
                    let candidate = amp_grid_gemm_operator_candidate(
                        Precision::F16,
                        64,
                        output_columns,
                        grid_shape,
                        weights,
                    );
                    grid.push(candidate);
                }
            }
            grid
        })
        .collect::<Vec<_>>();
    for output_columns in [AMP_OUTPUT_COLUMN_BLOCK, 128] {
        candidates.push(amp_gemm_operator_candidate(
            Precision::F16,
            64,
            output_columns,
            tile_count,
        ));
    }
    for input in [0, 1] {
        candidates.push(
            pointwise_operator_candidate(
                OperatorFamily::Add,
                [rows_f16.clone(), rows_f16.clone()],
                rows_f16.clone(),
            )
            .with_preserved_input_layout(input)
            .with_reusable_inputs([0, 1]),
        );
    }
    let mut grid_add = pointwise_operator_candidate(
        OperatorFamily::Add,
        [rows_f16.clone(), rows_f16.clone()],
        rows_f16.clone(),
    );
    grid_add.format_policy = OperatorFormatPolicy::RowMajorGrid;
    candidates.push(grid_add.with_reusable_inputs([0, 1]));
    let mut norm = pointwise_operator_candidate(
        OperatorFamily::LayerNorm,
        [rows_f16.clone(), rows_f16.clone(), rows_f16.clone()],
        rows_f16.clone(),
    );
    norm.format_policy = OperatorFormatPolicy::RowMajorRows;
    candidates.push(norm);
    candidates.extend([
        pointwise_operator_candidate(
            OperatorFamily::LayerNorm,
            [rows_f16.clone(), rows_f16.clone(), rows_f16.clone()],
            rows_f16.clone(),
        ),
        format_preserving_unary_candidate(OperatorFamily::Gelu, amp_left_result_f16),
        format_preserving_unary_candidate(OperatorFamily::Gelu, rows_f16.clone()),
        pointwise_operator_candidate(
            OperatorFamily::Add,
            [rows_f16.clone(), rows_f16.clone()],
            rows_f16,
        )
        .with_reusable_inputs([0, 1]),
        pointwise_operator_candidate(
            OperatorFamily::FlashAttention {
                options: AttentionOptions::default(),
                accumulate: AccumulationPrecision::F32,
            },
            [heads_f16.clone(), heads_f16.clone(), heads_f16.clone()],
            heads_f16,
        ),
    ]);
    let mut unique = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if !unique.contains(&candidate) {
            unique.push(candidate);
        }
    }
    unique
        .into_iter()
        .map(OperatorCandidate::Concrete)
        .chain(std::iter::once(OperatorCandidate::parallel_gemm(
            tile_count,
        )))
        .collect()
}

pub(super) fn pointwise_operator_candidate(
    operator: OperatorFamily,
    inputs: impl IntoIterator<Item = TensorFormat>,
    output: TensorFormat,
) -> ConcreteOperatorCandidate {
    ConcreteOperatorCandidate::new(
        operator,
        inputs.into_iter().map(OperandRequirement::new),
        output,
    )
}

pub(super) fn format_preserving_unary_candidate(
    operator: OperatorFamily,
    format: TensorFormat,
) -> ConcreteOperatorCandidate {
    pointwise_operator_candidate(operator, [format.clone()], format)
        .with_preserved_input_layout(0)
        .with_reusable_inputs([0])
}

pub(super) fn gemm_plan(
    operator: OperatorFamily,
    layouts: [Layout; 3],
    dispatch: OperatorDispatch,
) -> OperatorPlan {
    let OperatorFamily::Gemm {
        multiply: precision,
        ..
    } = operator
    else {
        unreachable!("GEMM operand contract requires a GEMM operator");
    };
    let [left, right, output] = layouts;
    let operand = |layout| OperandRequirement::new(TensorFormat { precision, layout });
    let orientation = match dispatch {
        OperatorDispatch::BlockedGemm { orientation, .. } => orientation,
        _ => unreachable!("GEMM operand contract requires a GEMM dispatch"),
    };
    let left_index = orientation.operand_indices().0;
    let mut inputs = vec![operand(left), operand(right)];
    inputs[left_index].materialization = OperandMaterialization::DispatchSlices;
    OperatorPlan {
        operator,
        dispatch,
        inputs,
        output: TensorFormat {
            precision: if matches!(precision, Precision::F8F143 { .. }) {
                Precision::F16
            } else {
                precision
            },
            layout: output,
        },
        reuse_inputs: None,
    }
}

pub(super) fn amp_gemm_operator_candidate(
    precision: Precision,
    inner: u16,
    output_columns: u32,
    tile_count: u16,
) -> ConcreteOperatorCandidate {
    let operator = OperatorFamily::Gemm {
        options: GemmOptions::default(),
        multiply: precision,
        accumulate: gemm_accumulation_precision(precision),
    };
    ConcreteOperatorCandidate {
        plan: gemm_plan(
            operator,
            [
                Layout::amp_left(inner, tile_count),
                Layout::block_major_matrix_storage(
                    inner,
                    AMP_OUTPUT_COLUMN_BLOCK,
                    tile_count,
                    1,
                    1,
                    MemoryClass::Ipu21Standard,
                ),
                if precision == Precision::F16 {
                    Layout::amp_left_result(tile_count)
                } else {
                    Layout::amp_output(tile_count)
                },
            ],
            blocked_gemm_dispatch(output_columns),
        ),
        format_policy: OperatorFormatPolicy::Concrete,
    }
}

pub(super) fn amp_grid_gemm_operator_candidate(
    precision: Precision,
    inner: u16,
    output_columns: u32,
    grid: AmpGridShape,
    weights: AmpWeightPlacement,
) -> ConcreteOperatorCandidate {
    let right_layout = match (weights.inner_partitions, weights.memory_class) {
        (1, MemoryClass::Ipu21Standard) => Layout::block_major_matrix_grid(
            inner,
            output_columns,
            grid.tile_count,
            grid.row_partitions,
            grid.column_partitions,
            grid.order,
        ),
        (inner_partitions, memory_class) => Layout::block_major_matrix_storage(
            inner,
            output_columns,
            grid.column_partitions,
            inner_partitions,
            grid.row_partitions / inner_partitions,
            memory_class,
        ),
    };
    let operator = OperatorFamily::Gemm {
        options: GemmOptions::default(),
        multiply: precision,
        accumulate: gemm_accumulation_precision(precision),
    };
    ConcreteOperatorCandidate {
        plan: gemm_plan(
            operator,
            [
                Layout::amp_left_grid(
                    inner,
                    grid.tile_count,
                    grid.row_partitions,
                    grid.column_partitions,
                    grid.order,
                ),
                right_layout,
                if precision == Precision::F16 {
                    Layout::amp_left_result_grid(
                        output_columns,
                        grid.tile_count,
                        grid.row_partitions,
                        grid.column_partitions,
                        grid.order,
                    )
                } else {
                    Layout::amp_output_grid(
                        output_columns,
                        grid.tile_count,
                        grid.row_partitions,
                        grid.column_partitions,
                        grid.order,
                    )
                },
            ],
            blocked_gemm_dispatch(output_columns),
        ),
        format_policy: OperatorFormatPolicy::Concrete,
    }
}

pub(super) fn blocked_gemm_dispatch(output_columns: u32) -> OperatorDispatch {
    OperatorDispatch::BlockedGemm {
        inner_block: AMP_INNER_BLOCK,
        output_column_block: output_columns,
        orientation: GemmOrientation::Normal,
        distribution: GemmDistribution::OutputStationary,
    }
}

#[derive(Clone, Copy)]
pub(super) struct AmpGridShape {
    pub(super) tile_count: u16,
    pub(super) row_partitions: u16,
    pub(super) column_partitions: u16,
    pub(super) order: GridOrder,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct AmpWeightPlacement {
    pub(super) inner_partitions: u16,
    pub(super) memory_class: MemoryClass,
}

impl AmpWeightPlacement {
    pub(super) const fn resident(memory_class: MemoryClass) -> Self {
        Self::sharded(1, memory_class)
    }

    const fn sharded(inner_partitions: u16, memory_class: MemoryClass) -> Self {
        Self {
            inner_partitions,
            memory_class,
        }
    }
}

pub(super) const fn gemm_accumulation_precision(precision: Precision) -> AccumulationPrecision {
    match precision {
        Precision::F16 | Precision::F8F143 { .. } => AccumulationPrecision::F16,
        Precision::F32 => AccumulationPrecision::F32,
    }
}
