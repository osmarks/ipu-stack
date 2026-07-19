use crate::{
    Allocation, AllocationKind, CompileError, KernelCommand, OpId, Phase, Schedule,
    SpecializationKey, TensorId,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashAttentionConfig {
    pub batch_size: u16,
    pub sequence_length: u16,
    pub hidden_size: u16,
    pub attention_heads: u16,
    pub tile_count: u16,
    pub data_base: u32,
    pub data_limit: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionTaskPlacement {
    pub batch: u16,
    pub head: u16,
    pub tile: u16,
    pub query: TensorId,
    pub key_value: TensorId,
    pub accumulator: TensorId,
    pub output: TensorId,
    pub query_address: u32,
    pub key_value_address: u32,
    pub accumulator_address: u32,
    pub output_address: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlashAttentionPlan {
    pub schedule: Schedule,
    pub tasks: Vec<AttentionTaskPlacement>,
    pub head_dimension: u16,
}

pub fn plan_flash_attention(
    config: FlashAttentionConfig,
) -> Result<FlashAttentionPlan, CompileError> {
    if config.batch_size == 0
        || config.sequence_length == 0
        || config.hidden_size == 0
        || config.attention_heads == 0
        || config.tile_count == 0
        || !config.hidden_size.is_multiple_of(config.attention_heads)
        || config.data_base >= config.data_limit
        || config.data_base & 7 != 0
    {
        return Err(CompileError::Graph(
            "flash attention requires non-zero dimensions, integral heads, and aligned memory"
                .into(),
        ));
    }
    let head_dimension = config.hidden_size / config.attention_heads;
    if !head_dimension.is_multiple_of(2) {
        return Err(CompileError::Graph(
            "FP16 attention finalization requires an even head dimension".into(),
        ));
    }
    let task_count = usize::from(config.batch_size) * usize::from(config.attention_heads);
    let elements = u32::from(config.sequence_length) * u32::from(head_dimension);
    let query_bytes = elements * 2;
    let key_value_bytes = query_bytes * 2;
    let accumulator_bytes = elements * 4;
    let output_bytes = query_bytes;
    let tensors_per_kind = task_count;
    let mut cursors = vec![config.data_base; usize::from(config.tile_count)];
    let mut allocations = Vec::with_capacity(task_count * 4);
    let mut tasks = Vec::with_capacity(task_count);

    for index in 0..task_count {
        let tile = u16::try_from(index % usize::from(config.tile_count))
            .map_err(|_| CompileError::Graph("attention tile index overflow".into()))?;
        let cursor = &mut cursors[usize::from(tile)];
        let query_address = allocate(cursor, query_bytes, 8)?;
        let key_value_address = allocate(cursor, key_value_bytes, 8)?;
        let accumulator_address = allocate(cursor, accumulator_bytes, 8)?;
        let output_address = allocate(cursor, output_bytes, 8)?;
        if *cursor > config.data_limit {
            return Err(CompileError::Memory(format!(
                "attention task {index} exhausts tile {tile} SRAM"
            )));
        }
        let query = TensorId(index);
        let key_value = TensorId(tensors_per_kind + index);
        let accumulator = TensorId(tensors_per_kind * 2 + index);
        let output = TensorId(tensors_per_kind * 3 + index);
        allocations.extend([
            home(query, tile, query_address, query_bytes, 0, 2),
            home(key_value, tile, key_value_address, key_value_bytes, 0, 1),
            home(
                accumulator,
                tile,
                accumulator_address,
                accumulator_bytes,
                0,
                2,
            ),
            home(output, tile, output_address, output_bytes, 1, 3),
        ]);
        tasks.push(AttentionTaskPlacement {
            batch: u16::try_from(index / usize::from(config.attention_heads))
                .map_err(|_| CompileError::Graph("attention batch index overflow".into()))?,
            head: u16::try_from(index % usize::from(config.attention_heads))
                .map_err(|_| CompileError::Graph("attention head index overflow".into()))?,
            tile,
            query,
            key_value,
            accumulator,
            output,
            query_address,
            key_value_address,
            accumulator_address,
            output_address,
        });
    }

    let attention_commands = tasks
        .iter()
        .map(|task| KernelCommand {
            tile: task.tile,
            output: task.accumulator,
            inputs: vec![task.query, task.key_value],
            arguments: Vec::new(),
            specialization: SpecializationKey {
                operation: "flash_attention_f16".into(),
                shape: vec![
                    usize::from(config.sequence_length),
                    usize::from(head_dimension),
                ],
                worker_count: 6,
                role: format!("batch-{}-head-{}", task.batch, task.head),
                alignment: 8,
            },
            metadata: task_metadata(task, config, "FlashAttention online softmax"),
        })
        .collect();
    let output_pairs = elements.div_ceil(2);
    let finalize_commands = tasks
        .iter()
        .map(|task| KernelCommand {
            tile: task.tile,
            output: task.output,
            inputs: vec![task.accumulator, task.query],
            arguments: vec![output_pairs],
            specialization: SpecializationKey {
                operation: "attention_f32_to_f16".into(),
                shape: vec![
                    usize::from(config.sequence_length),
                    usize::from(head_dimension),
                ],
                worker_count: 6,
                role: format!("batch-{}-head-{}", task.batch, task.head),
                alignment: 8,
            },
            metadata: task_metadata(task, config, "Attention FP16 output"),
        })
        .collect();
    let peak_sram = cursors
        .into_iter()
        .enumerate()
        .filter(|(_, cursor)| *cursor > config.data_base)
        .map(|(tile, cursor)| (tile as u16, cursor - config.data_base))
        .collect();

    Ok(FlashAttentionPlan {
        schedule: Schedule {
            layouts: Vec::new(),
            phases: vec![
                Phase::Compute {
                    op: OpId(0),
                    commands: attention_commands,
                },
                Phase::Compute {
                    op: OpId(1),
                    commands: finalize_commands,
                },
            ],
            allocations,
            tile_count: config.tile_count,
            peak_sram,
        },
        tasks,
        head_dimension,
    })
}

fn allocate(cursor: &mut u32, size: u32, alignment: u32) -> Result<u32, CompileError> {
    *cursor = cursor.div_ceil(alignment) * alignment;
    let address = *cursor;
    *cursor = cursor
        .checked_add(size)
        .ok_or_else(|| CompileError::Memory("attention allocation overflow".into()))?;
    Ok(address)
}

fn home(
    tensor: TensorId,
    tile: u16,
    address: u32,
    size: u32,
    live_from: usize,
    live_until: usize,
) -> Allocation {
    Allocation {
        tensor,
        tile,
        address,
        size,
        live_from,
        live_until,
        kind: AllocationKind::Home,
    }
}

fn task_metadata(
    task: &AttentionTaskPlacement,
    config: FlashAttentionConfig,
    label: &str,
) -> BTreeMap<String, String> {
    BTreeMap::from([
        ("label".into(), label.into()),
        ("batch".into(), task.batch.to_string()),
        ("head".into(), task.head.to_string()),
        ("sequence_length".into(), config.sequence_length.to_string()),
        (
            "head_dimension".into(),
            (config.hidden_size / config.attention_heads).to_string(),
        ),
        ("causal".into(), "false".into()),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_plan_covers_each_batch_head_without_quadratic_storage() {
        for hidden_size in [768, 1024, 1152] {
            for batch_size in [1, 2, 5] {
                let config = FlashAttentionConfig {
                    batch_size,
                    sequence_length: 128,
                    hidden_size,
                    attention_heads: 16,
                    tile_count: 1472,
                    data_base: 0xa0000,
                    data_limit: 0xe8000,
                };
                let plan = plan_flash_attention(config).unwrap();
                assert_eq!(plan.tasks.len(), usize::from(batch_size) * 16);
                assert_eq!(plan.head_dimension, hidden_size / 16);
                assert!(plan.tasks.iter().enumerate().all(|(index, task)| {
                    usize::from(task.batch) * 16 + usize::from(task.head) == index
                }));
                let per_task_bytes = plan
                    .schedule
                    .allocations
                    .iter()
                    .map(|allocation| allocation.size)
                    .sum::<u32>()
                    / u32::try_from(plan.tasks.len()).unwrap();
                assert_eq!(
                    per_task_bytes,
                    u32::from(config.sequence_length) * u32::from(plan.head_dimension) * 12
                );
            }
        }
    }
}
