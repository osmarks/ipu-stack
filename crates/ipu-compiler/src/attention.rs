use crate::{
    Allocation, AllocationKind, CompileError, KernelCommand, OpId, Phase, Schedule,
    SpecializationKey, TensorId, Transfer,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashAttentionConfig {
    pub batch_size: u16,
    pub sequence_length: u16,
    pub hidden_size: u16,
    pub attention_heads: u16,
    /// Zero selects as many rows as fit while using at most one task per tile.
    pub query_block_rows: u16,
    /// Zero selects the largest exchange block supported by the fabric window.
    pub key_block_rows: u16,
    pub tile_count: u16,
    pub data_base: u32,
    pub data_limit: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionTaskPlacement {
    pub batch: u16,
    pub head: u16,
    pub query_row_start: u16,
    pub query_rows: u16,
    pub tile: u16,
    pub query: TensorId,
    pub accumulator: TensorId,
    pub output: TensorId,
    pub query_address: u32,
    pub accumulator_address: u32,
    pub output_address: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionKeyValuePlacement {
    pub batch: u16,
    pub head: u16,
    pub key_row_start: u16,
    pub key_rows: u16,
    pub tile: u16,
    pub tensor: TensorId,
    pub address: u32,
    pub size: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlashAttentionPlan {
    pub schedule: Schedule,
    pub tasks: Vec<AttentionTaskPlacement>,
    pub key_values: Vec<AttentionKeyValuePlacement>,
    pub head_dimension: u16,
    pub query_block_rows: u16,
    pub key_block_rows: u16,
}

pub fn plan_flash_attention(
    config: FlashAttentionConfig,
) -> Result<FlashAttentionPlan, CompileError> {
    validate(config)?;
    let head_dimension = config.hidden_size / config.attention_heads;
    if !head_dimension.is_multiple_of(2) {
        return Err(CompileError::Graph(
            "FP16 attention finalization requires an even head dimension".into(),
        ));
    }

    let head_count = usize::from(config.batch_size) * usize::from(config.attention_heads);
    let query_block_rows = if config.query_block_rows == 0 {
        (1..=config.sequence_length)
            .find(|&rows| {
                head_count * usize::from(config.sequence_length.div_ceil(rows))
                    <= usize::from(config.tile_count)
            })
            .ok_or_else(|| {
                CompileError::Graph(format!(
                    "attention needs at least {head_count} tiles, one for each batch/head"
                ))
            })?
    } else {
        config.query_block_rows
    }
    .min(config.sequence_length);
    let query_blocks = config.sequence_length.div_ceil(query_block_rows);
    let task_count = head_count * usize::from(query_blocks);
    if task_count > usize::from(config.tile_count) {
        return Err(CompileError::Graph(format!(
            "attention needs {task_count} query tasks but only {} tiles are available; increase query_block_rows",
            config.tile_count
        )));
    }

    let maximum_key_rows =
        u16::try_from((ipu_exchange::MAX_TRANSFER_WORDS * 4) / (u32::from(head_dimension) * 4))
            .map_err(|_| CompileError::Graph("attention key block overflow".into()))?;
    let key_block_rows = if config.key_block_rows == 0 {
        maximum_key_rows
    } else {
        config.key_block_rows
    }
    .min(config.sequence_length);
    if key_block_rows == 0 || key_block_rows > maximum_key_rows {
        return Err(CompileError::Graph(format!(
            "attention key blocks may contain at most {maximum_key_rows} rows for head dimension {head_dimension}"
        )));
    }
    let key_blocks = config.sequence_length.div_ceil(key_block_rows);

    let mut cursors = vec![config.data_base; usize::from(config.tile_count)];
    let mut allocations = Vec::new();
    let mut tasks = Vec::with_capacity(task_count);
    let mut next_tensor = 0usize;
    for batch in 0..config.batch_size {
        for head in 0..config.attention_heads {
            for query_block in 0..query_blocks {
                let query_row_start = query_block * query_block_rows;
                let query_rows = query_block_rows.min(config.sequence_length - query_row_start);
                let tile = u16::try_from(tasks.len())
                    .map_err(|_| CompileError::Graph("attention tile index overflow".into()))?;
                let elements = u32::from(query_rows) * u32::from(head_dimension);
                let query_bytes = elements * 2;
                // The online state is [accumulator rows][maximum per row][denominator per row].
                let accumulator_bytes = elements * 4 + u32::from(query_rows) * 8;
                let output_bytes = query_bytes;
                let cursor = &mut cursors[usize::from(tile)];
                let query_address = allocate(cursor, query_bytes, 8)?;
                let accumulator_address = allocate(cursor, accumulator_bytes, 8)?;
                let output_address = allocate(cursor, output_bytes, 8)?;
                let query = fresh_tensor(&mut next_tensor);
                let accumulator = fresh_tensor(&mut next_tensor);
                let output = fresh_tensor(&mut next_tensor);
                tasks.push(AttentionTaskPlacement {
                    batch,
                    head,
                    query_row_start,
                    query_rows,
                    tile,
                    query,
                    accumulator,
                    output,
                    query_address,
                    accumulator_address,
                    output_address,
                });
            }
        }
    }

    let mut key_values = Vec::with_capacity(head_count * usize::from(key_blocks));
    for batch in 0..config.batch_size {
        for head in 0..config.attention_heads {
            let owner = tasks
                .iter()
                .find(|task| task.batch == batch && task.head == head)
                .expect("each head has a query task")
                .tile;
            for key_block in 0..key_blocks {
                let key_row_start = key_block * key_block_rows;
                let key_rows = key_block_rows.min(config.sequence_length - key_row_start);
                let size = u32::from(key_rows) * u32::from(head_dimension) * 4;
                let address = allocate(&mut cursors[usize::from(owner)], size, 8)?;
                key_values.push(AttentionKeyValuePlacement {
                    batch,
                    head,
                    key_row_start,
                    key_rows,
                    tile: owner,
                    tensor: fresh_tensor(&mut next_tensor),
                    address,
                    size,
                });
            }
        }
    }
    for (tile, &cursor) in cursors.iter().enumerate() {
        if cursor > config.data_limit {
            return Err(CompileError::Memory(format!(
                "attention data exhausts tile {tile}: 0x{cursor:x} exceeds 0x{:x}",
                config.data_limit
            )));
        }
    }

    let final_phase = usize::from(key_blocks) * 2 + 1;
    for task in &tasks {
        allocations.extend([
            home(
                task.query,
                task.tile,
                task.query_address,
                u32::from(task.query_rows) * u32::from(head_dimension) * 2,
                0,
                final_phase,
            ),
            home(
                task.accumulator,
                task.tile,
                task.accumulator_address,
                u32::from(task.query_rows) * (u32::from(head_dimension) * 4 + 8),
                0,
                final_phase,
            ),
            home(
                task.output,
                task.tile,
                task.output_address,
                u32::from(task.query_rows) * u32::from(head_dimension) * 2,
                final_phase - 1,
                final_phase,
            ),
        ]);
    }
    for block in &key_values {
        allocations.push(home(
            block.tensor,
            block.tile,
            block.address,
            block.size,
            0,
            final_phase,
        ));
    }

    let mut phases = Vec::with_capacity(final_phase);
    for key_block in 0..key_blocks {
        let exchange_phase = phases.len();
        let compute_phase = exchange_phase + 1;
        let initial = u32::from(key_block == 0);
        let final_block = u32::from(key_block + 1 == key_blocks);
        let mut transfers = Vec::new();
        let mut commands = Vec::with_capacity(tasks.len());
        for task in &tasks {
            let block = key_values
                .iter()
                .find(|block| {
                    block.batch == task.batch
                        && block.head == task.head
                        && block.key_row_start == key_block * key_block_rows
                })
                .expect("each head has every key block");
            if task.tile != block.tile {
                transfers.push(Transfer {
                    source_tile: block.tile,
                    destination_tile: task.tile,
                    tensor: block.tensor,
                    bytes: block.size,
                });
                allocations.push(Allocation {
                    tensor: block.tensor,
                    tile: task.tile,
                    address: ipu_exchange::EXCHANGE_WINDOW_BASE,
                    size: block.size,
                    live_from: exchange_phase,
                    live_until: compute_phase,
                    kind: AllocationKind::ExchangeStaging {
                        phase: exchange_phase,
                    },
                });
            }
            commands.push(KernelCommand {
                tile: task.tile,
                output: task.accumulator,
                inputs: vec![task.query, block.tensor],
                arguments: vec![
                    u32::from(task.query_rows),
                    u32::from(block.key_rows),
                    initial,
                    final_block,
                ],
                specialization: SpecializationKey {
                    operation: "flash_attention_f16".into(),
                    shape: vec![
                        usize::from(task.query_rows),
                        usize::from(block.key_rows),
                        usize::from(head_dimension),
                    ],
                    worker_count: 6,
                    role: format!(
                        "batch-{}-head-{}-queries-{}-{}-keys-{}-{}",
                        task.batch,
                        task.head,
                        task.query_row_start,
                        task.query_row_start + task.query_rows,
                        block.key_row_start,
                        block.key_row_start + block.key_rows
                    ),
                    alignment: 8,
                },
                metadata: task_metadata(task, block, config, head_dimension),
            });
        }
        phases.push(Phase::Exchange { transfers });
        phases.push(Phase::Compute {
            op: OpId(compute_phase),
            commands,
        });
    }

    let finalize_commands = tasks
        .iter()
        .map(|task| KernelCommand {
            tile: task.tile,
            output: task.output,
            inputs: vec![task.accumulator, task.query],
            arguments: vec![(u32::from(task.query_rows) * u32::from(head_dimension)).div_ceil(2)],
            specialization: SpecializationKey {
                operation: "attention_f32_to_f16".into(),
                shape: vec![usize::from(task.query_rows), usize::from(head_dimension)],
                worker_count: 6,
                role: format!(
                    "batch-{}-head-{}-queries-{}-{}",
                    task.batch,
                    task.head,
                    task.query_row_start,
                    task.query_row_start + task.query_rows
                ),
                alignment: 8,
            },
            metadata: BTreeMap::from([
                ("label".into(), "Attention FP16 output".into()),
                ("batch".into(), task.batch.to_string()),
                ("head".into(), task.head.to_string()),
                ("query_row_start".into(), task.query_row_start.to_string()),
                ("query_rows".into(), task.query_rows.to_string()),
            ]),
        })
        .collect();
    phases.push(Phase::Compute {
        op: OpId(phases.len()),
        commands: finalize_commands,
    });

    let peak_sram = cursors
        .into_iter()
        .enumerate()
        .filter(|(_, cursor)| *cursor > config.data_base)
        .map(|(tile, cursor)| (tile as u16, cursor - config.data_base))
        .collect();
    Ok(FlashAttentionPlan {
        schedule: Schedule {
            layouts: Vec::new(),
            phases,
            allocations,
            tile_count: config.tile_count,
            peak_sram,
        },
        tasks,
        key_values,
        head_dimension,
        query_block_rows,
        key_block_rows,
    })
}

fn validate(config: FlashAttentionConfig) -> Result<(), CompileError> {
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
    Ok(())
}

fn fresh_tensor(next: &mut usize) -> TensorId {
    let tensor = TensorId(*next);
    *next += 1;
    tensor
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
    block: &AttentionKeyValuePlacement,
    config: FlashAttentionConfig,
    head_dimension: u16,
) -> BTreeMap<String, String> {
    BTreeMap::from([
        ("label".into(), "FlashAttention online softmax".into()),
        ("batch".into(), task.batch.to_string()),
        ("head".into(), task.head.to_string()),
        ("sequence_length".into(), config.sequence_length.to_string()),
        ("head_dimension".into(), head_dimension.to_string()),
        ("query_row_start".into(), task.query_row_start.to_string()),
        ("query_rows".into(), task.query_rows.to_string()),
        ("key_row_start".into(), block.key_row_start.to_string()),
        ("key_rows".into(), block.key_rows.to_string()),
        ("causal".into(), "false".into()),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_plan_shards_queries_and_bounds_exchange_blocks() {
        for hidden_size in [768, 1024, 1152] {
            for batch_size in [1, 2, 5] {
                let config = FlashAttentionConfig {
                    batch_size,
                    sequence_length: 128,
                    hidden_size,
                    attention_heads: 16,
                    query_block_rows: 16,
                    key_block_rows: 0,
                    tile_count: 1472,
                    data_base: 0xa0000,
                    data_limit: 0xe8000,
                };
                let plan = plan_flash_attention(config).unwrap();
                assert_eq!(plan.head_dimension, hidden_size / 16);
                assert_eq!(plan.tasks.len(), usize::from(batch_size) * 16 * 8);
                assert!(
                    plan.key_values
                        .iter()
                        .all(|block| { block.size <= ipu_exchange::MAX_TRANSFER_WORDS * 4 })
                );
                for batch in 0..batch_size {
                    for head in 0..16 {
                        assert_eq!(
                            plan.tasks
                                .iter()
                                .filter(|task| task.batch == batch && task.head == head)
                                .map(|task| u32::from(task.query_rows))
                                .sum::<u32>(),
                            128
                        );
                        assert_eq!(
                            plan.key_values
                                .iter()
                                .filter(|block| block.batch == batch && block.head == head)
                                .map(|block| u32::from(block.key_rows))
                                .sum::<u32>(),
                            128
                        );
                    }
                }
                plan.schedule
                    .lower_tile_programs(&ipu_exchange::Topology::c600())
                    .unwrap();
            }
        }
    }

    #[test]
    fn automatic_query_blocking_never_oversubscribes_tiles() {
        let plan = plan_flash_attention(FlashAttentionConfig {
            batch_size: 8,
            sequence_length: 256,
            hidden_size: 1024,
            attention_heads: 16,
            query_block_rows: 0,
            key_block_rows: 0,
            tile_count: 1472,
            data_base: 0xa0000,
            data_limit: 0xe8000,
        })
        .unwrap();
        assert!(plan.tasks.len() <= 1472);
        assert!(plan.query_block_rows >= 23);
    }
}
