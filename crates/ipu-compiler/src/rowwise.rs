use crate::{
    Allocation, AllocationKind, CompileError, KernelCommand, OpId, Phase, Schedule,
    SpecializationKey, TensorId,
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
