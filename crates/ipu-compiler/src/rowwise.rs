use crate::{
    Allocation, AllocationKind, BlockPlacement, CompileError, KernelCommand, OpId, Phase, Schedule,
    SpecializationKey, TensorId, Transfer,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineLayerNormConfig {
    pub rows: u16,
    pub columns: u16,
    pub row_block_dimension: u16,
    pub tile_count: u16,
    pub data_base: u32,
    pub data_limit: u32,
    pub epsilon_bits: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowShardPlacement {
    pub tile: u16,
    pub row_start: u16,
    pub rows: u16,
    pub columns: u16,
    pub tensor: TensorId,
    pub address: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AffineLayerNormPlan {
    pub schedule: Schedule,
    pub input: Vec<RowShardPlacement>,
    pub affine: Vec<RowShardPlacement>,
    pub output: Vec<RowShardPlacement>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowShardTransitionConfig {
    pub columns: u16,
    pub data_base: u32,
    pub data_limit: u32,
}

pub fn append_c16_to_a16_row_shards(
    schedule: &mut Schedule,
    source: &[BlockPlacement],
    config: RowShardTransitionConfig,
) -> Result<Vec<RowShardPlacement>, CompileError> {
    if source.is_empty()
        || config.columns == 0
        || !config.columns.is_multiple_of(16)
        || config.data_base & 7 != 0
        || config.data_base >= config.data_limit
    {
        return Err(CompileError::Graph(
            "C16 to A16 transition requires blocks, columns divisible by 16, and aligned SRAM"
                .into(),
        ));
    }
    let mut rows = BTreeMap::<u16, Vec<&BlockPlacement>>::new();
    for block in source {
        rows.entry(block.block_row).or_default().push(block);
    }
    let mut next_tensor = schedule
        .allocations
        .iter()
        .map(|allocation| allocation.tensor.0)
        .max()
        .unwrap_or(0)
        + 1;
    let exchange_phase = schedule.phases.len();
    let compute_phase = exchange_phase + 1;
    let mut transfers = Vec::new();
    let mut commands = Vec::new();
    let mut destinations = Vec::with_capacity(rows.len());
    let mut cursors = vec![config.data_base; usize::from(schedule.tile_count)];

    for (destination_tile, (_block_row, mut blocks)) in rows.into_iter().enumerate() {
        let destination_tile = u16::try_from(destination_tile)
            .map_err(|_| CompileError::Graph("row-shard destination tile overflow".into()))?;
        blocks.sort_by_key(|block| block.column_start);
        let first = blocks[0];
        let mut next_column = 0u16;
        for block in &blocks {
            if block.row_start != first.row_start
                || block.rows != first.rows
                || block.column_start != next_column
            {
                return Err(CompileError::Graph(
                    "C16 source blocks do not form complete aligned row shards".into(),
                ));
            }
            next_column = next_column
                .checked_add(block.columns)
                .ok_or_else(|| CompileError::Graph("row-shard column overflow".into()))?;
        }
        if next_column != config.columns {
            return Err(CompileError::Graph(format!(
                "C16 row shard covers {next_column} columns, expected {}",
                config.columns
            )));
        }
        let bytes = u32::from(first.rows) * u32::from(config.columns) * 2;
        let address = allocate(&mut cursors[usize::from(destination_tile)], bytes, 8)?;
        if cursors[usize::from(destination_tile)] > config.data_limit {
            return Err(CompileError::Memory(format!(
                "A16 row shard on tile {destination_tile} exceeds its SRAM arena"
            )));
        }
        let destination_tensor = TensorId(next_tensor);
        next_tensor += 1;
        schedule.allocations.push(Allocation {
            tensor: destination_tensor,
            tile: destination_tile,
            address,
            size: bytes,
            live_from: compute_phase,
            live_until: usize::MAX,
            kind: AllocationKind::Home,
        });
        destinations.push(RowShardPlacement {
            tile: destination_tile,
            row_start: first.row_start,
            rows: first.rows,
            columns: config.columns,
            tensor: destination_tensor,
            address,
        });

        for block in blocks {
            let block_bytes = u32::from(block.rows) * u32::from(block.columns) * 2;
            if block.tile != destination_tile {
                let staging_address = ipu_exchange::EXCHANGE_WINDOW_BASE
                    + u32::from(block.column_start) * u32::from(block.rows) * 2;
                transfers.push(Transfer {
                    source_tile: block.tile,
                    destination_tile,
                    tensor: block.tensor,
                    bytes: block_bytes,
                });
                schedule.allocations.push(Allocation {
                    tensor: block.tensor,
                    tile: destination_tile,
                    address: staging_address,
                    size: block_bytes,
                    live_from: exchange_phase,
                    live_until: compute_phase,
                    kind: AllocationKind::ExchangeStaging {
                        phase: exchange_phase,
                    },
                });
            }
            let output_alias = TensorId(next_tensor);
            next_tensor += 1;
            let output_address =
                address + u32::from(block.column_start) * u32::from(block.rows) * 2;
            schedule.allocations.push(Allocation {
                tensor: output_alias,
                tile: destination_tile,
                address: output_address,
                size: block_bytes,
                live_from: compute_phase,
                live_until: compute_phase,
                kind: AllocationKind::Home,
            });
            commands.push(KernelCommand {
                tile: destination_tile,
                output: output_alias,
                inputs: vec![block.tensor, block.tensor],
                arguments: vec![u32::from(block.rows)],
                specialization: SpecializationKey {
                    operation: "relayout_f16_c16_to_a16".into(),
                    shape: vec![usize::from(block.rows), usize::from(block.columns)],
                    worker_count: 6,
                    role: "blocked-to-row-sharded".into(),
                    alignment: 8,
                },
                metadata: BTreeMap::from([
                    ("label".into(), "gather blocked activation".into()),
                    ("row_start".into(), block.row_start.to_string()),
                    ("column_start".into(), block.column_start.to_string()),
                ]),
            });
        }
    }
    schedule.phases.push(Phase::Exchange { transfers });
    schedule.phases.push(Phase::Compute {
        op: OpId(compute_phase),
        commands,
    });
    Ok(destinations)
}

pub fn plan_affine_layer_norm_f16(
    config: AffineLayerNormConfig,
) -> Result<AffineLayerNormPlan, CompileError> {
    let epsilon = f32::from_bits(config.epsilon_bits);
    let epsilon_q30 = (epsilon * (1u64 << 30) as f32).round() as u32;
    if config.rows == 0
        || config.columns == 0
        || !config.columns.is_multiple_of(16)
        || config.row_block_dimension == 0
        || config.tile_count == 0
        || config.data_base >= config.data_limit
        || config.data_base & 7 != 0
        || !epsilon.is_finite()
        || epsilon_q30 == 0
        || epsilon_q30 >= 1 << 20
    {
        return Err(CompileError::Graph(
            "FP16 affine layer norm requires nonzero dimensions, columns divisible by 16, tiles, and aligned SRAM".into(),
        ));
    }

    let shard_count = config.rows.div_ceil(config.row_block_dimension);
    if shard_count > config.tile_count {
        return Err(CompileError::Graph(format!(
            "FP16 affine layer norm needs {shard_count} tiles for its row shards but only {} are available",
            config.tile_count
        )));
    }
    let affine_bytes = u32::from(config.columns) * 4;
    let mut cursors = vec![config.data_base; usize::from(config.tile_count)];
    let mut allocations = Vec::with_capacity(usize::from(shard_count) * 3);
    let mut input = Vec::with_capacity(usize::from(shard_count));
    let mut affine = Vec::with_capacity(usize::from(shard_count));
    let mut output = Vec::with_capacity(usize::from(shard_count));
    let mut commands = Vec::with_capacity(usize::from(shard_count));

    for shard in 0..shard_count {
        let tile = shard;
        let row_start = shard * config.row_block_dimension;
        let rows = config.row_block_dimension.min(config.rows - row_start);
        let activation_bytes = u32::from(rows) * u32::from(config.columns) * 2;
        let cursor = &mut cursors[usize::from(tile)];
        let input_address = allocate(cursor, activation_bytes, 8)?;
        let affine_address = allocate(cursor, affine_bytes, 8)?;
        let output_address = allocate(cursor, activation_bytes, 8)?;
        if *cursor > config.data_limit {
            return Err(CompileError::Memory(format!(
                "FP16 affine layer norm shard {shard} ends at 0x{cursor:x}, beyond 0x{:x}",
                config.data_limit
            )));
        }

        let input_tensor = TensorId(usize::from(shard));
        let affine_tensor = TensorId(usize::from(shard_count + shard));
        let output_tensor = TensorId(usize::from(shard_count * 2 + shard));
        let activation_placement = |tensor, address| RowShardPlacement {
            tile,
            row_start,
            rows,
            columns: config.columns,
            tensor,
            address,
        };
        input.push(activation_placement(input_tensor, input_address));
        affine.push(RowShardPlacement {
            tile,
            row_start: 0,
            rows: 2,
            columns: config.columns,
            tensor: affine_tensor,
            address: affine_address,
        });
        output.push(activation_placement(output_tensor, output_address));
        for (tensor, address, size) in [
            (input_tensor, input_address, activation_bytes),
            (affine_tensor, affine_address, affine_bytes),
            (output_tensor, output_address, activation_bytes),
        ] {
            allocations.push(Allocation {
                tensor,
                tile,
                address,
                size,
                live_from: 0,
                live_until: usize::MAX,
                kind: AllocationKind::Home,
            });
        }
        commands.push(KernelCommand {
            tile,
            output: output_tensor,
            inputs: vec![input_tensor, affine_tensor],
            arguments: vec![u32::from(rows), u32::from(config.columns), epsilon_q30],
            specialization: SpecializationKey {
                operation: "layer_norm_affine_f16".into(),
                shape: vec![usize::from(rows), usize::from(config.columns)],
                worker_count: 6,
                role: "normalize".into(),
                alignment: 8,
            },
            metadata: BTreeMap::from([
                ("row_start".into(), row_start.to_string()),
                ("rows".into(), rows.to_string()),
                ("columns".into(), config.columns.to_string()),
            ]),
        });
    }

    let peak_sram = cursors
        .into_iter()
        .enumerate()
        .filter(|(_, cursor)| *cursor > config.data_base)
        .map(|(tile, cursor)| (tile as u16, cursor - config.data_base))
        .collect();
    Ok(AffineLayerNormPlan {
        schedule: Schedule {
            layouts: Vec::new(),
            tile_count: config.tile_count,
            allocations,
            phases: vec![Phase::Compute {
                op: OpId(0),
                commands,
            }],
            peak_sram,
        },
        input,
        affine,
        output,
    })
}

fn allocate(cursor: &mut u32, bytes: u32, alignment: u32) -> Result<u32, CompileError> {
    let address = cursor
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| CompileError::Memory("row-sharded allocation overflow".into()))?;
    *cursor = address
        .checked_add(bytes)
        .ok_or_else(|| CompileError::Memory("row-sharded allocation overflow".into()))?;
    Ok(address)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_layer_norm_balances_non_divisible_rows() {
        let plan = plan_affine_layer_norm_f16(AffineLayerNormConfig {
            rows: 729,
            columns: 1152,
            row_block_dimension: 12,
            tile_count: 1472,
            data_base: 0xa0000,
            data_limit: 0xe8000,
            epsilon_bits: 1e-6f32.to_bits(),
        })
        .unwrap();
        assert_eq!(plan.input.len(), 61);
        assert_eq!(plan.input.iter().map(|shard| shard.rows).sum::<u16>(), 729);
        assert_eq!(plan.input.last().unwrap().rows, 9);
        assert!(plan.output.iter().all(|shard| shard.address & 7 == 0));
    }
}
