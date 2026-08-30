use crate::config::{OperatorClass, PipelineConfig};
use crate::graph::{Operation, OperationKind, TensorShape, ValueId};
use crate::layout::{AMP_INNER_BLOCK, Layout};
use crate::operator::Precision;

pub(super) fn direct_consumer_layouts(
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

pub(super) fn operator_accepts_input_layout(
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
