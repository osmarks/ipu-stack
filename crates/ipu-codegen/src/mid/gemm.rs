use crate::GemmGeometry;
use crate::graph::{Operation, OperationKind, TensorShape, ValueId};
use crate::layout::{
    AMP_COLUMN_MICRO, AMP_NARROW_OUTPUT_COLUMN_BLOCK, AMP_OUTPUT_COLUMN_BLOCK,
    AMP_WIDE_OUTPUT_COLUMN_BLOCK, Layout, MemoryClass, Padding, TensorAxis,
};
use crate::operator::{
    AccumulationPrecision, BlockedGemmPlan, GemmKernelFamily, GemmWeightLoad, MidOperator,
    OperatorDispatch, OperatorPlan, Precision,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GroupedOutputLayout {
    pub groups: u16,
    pub physical_lane_multiple: u32,
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

pub(super) fn blocked_gemm_dispatch(
    operator: MidOperator,
    geometry: GemmGeometry,
) -> OperatorDispatch {
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

pub(super) fn amp_output_column_blocks(precision: Precision) -> &'static [u32] {
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

pub(super) fn balance_parallel_gemm_columns(layout: &mut Layout, axis: TensorAxis) {
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

pub(super) fn apply_grouped_output_layout(
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

pub(super) fn pad_axis_to_f16_exchange_word(layout: &mut Layout, axis: TensorAxis) {
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
