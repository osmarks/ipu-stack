use crate::{
    Allocation, AllocationKind, BlockPlacement, CompileError, KernelCommand, MemoryConstraint,
    MemoryPlacement, OpId, Phase, Schedule, SpecializationKey, TensorId, Transfer,
    find_free_region,
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
pub struct AppendAffineLayerNormConfig {
    pub data_base: u32,
    pub data_limit: u32,
    pub epsilon_bits: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppendedAffineLayerNorm {
    pub affine: Vec<RowShardPlacement>,
    pub output: Vec<RowShardPlacement>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowShardTransitionConfig {
    pub columns: u16,
    pub data_base: u32,
    pub data_limit: u32,
}

pub fn end_tensor_lifetimes(
    schedule: &mut Schedule,
    tensors: impl IntoIterator<Item = TensorId>,
) -> Result<(), CompileError> {
    let phase = schedule.phases.len();
    for tensor in tensors {
        let mut found = false;
        for allocation in schedule
            .allocations
            .iter_mut()
            .filter(|allocation| allocation.tensor == tensor)
        {
            found = true;
            if allocation.live_until == usize::MAX {
                if allocation.live_from >= phase {
                    return Err(CompileError::Graph(format!(
                        "tensor {} cannot end before it becomes live",
                        tensor.0
                    )));
                }
                allocation.live_until = phase;
            }
        }
        if !found {
            return Err(CompileError::Graph(format!(
                "cannot end unknown tensor {}",
                tensor.0
            )));
        }
    }
    Ok(())
}

pub fn append_add_f16_row_shards_in_place(
    schedule: &mut Schedule,
    destination: &[RowShardPlacement],
    source: &[RowShardPlacement],
) -> Result<Vec<RowShardPlacement>, CompileError> {
    if destination.is_empty() || destination.len() != source.len() {
        return Err(CompileError::Graph(
            "row-sharded add requires matching non-empty inputs".into(),
        ));
    }
    let exchange_phase = schedule.phases.len();
    let compute_phase = exchange_phase + 1;
    let mut transfers = Vec::new();
    let mut commands = Vec::with_capacity(destination.len());
    for destination in destination {
        let source = source
            .iter()
            .find(|source| {
                source.row_start == destination.row_start && source.rows == destination.rows
            })
            .ok_or_else(|| CompileError::Graph("row-sharded add source is missing".into()))?;
        if source.columns != destination.columns || destination.columns % 2 != 0 {
            return Err(CompileError::Graph(
                "row-sharded add inputs have incompatible columns".into(),
            ));
        }
        let bytes = u32::from(destination.rows) * u32::from(destination.columns) * 2;
        if source.tile != destination.tile {
            if bytes > ipu_exchange::EXCHANGE_WINDOW_BYTES {
                return Err(CompileError::Graph(
                    "row-sharded add source exceeds the exchange window".into(),
                ));
            }
            transfers.push(Transfer {
                source_tile: source.tile,
                destination_tile: destination.tile,
                tensor: source.tensor,
                bytes,
            });
            schedule.allocations.push(Allocation {
                tensor: source.tensor,
                tile: destination.tile,
                address: ipu_exchange::EXCHANGE_WINDOW_BASE,
                size: bytes,
                live_from: exchange_phase,
                live_until: compute_phase,
                kind: AllocationKind::ExchangeStaging {
                    phase: exchange_phase,
                },
            });
        }
        let units = bytes / 4;
        commands.push(KernelCommand {
            tile: destination.tile,
            output: destination.tensor,
            inputs: vec![destination.tensor, source.tensor],
            arguments: vec![units, units / 6, units % 6],
            specialization: SpecializationKey {
                operation: "add_f16".into(),
                shape: vec![
                    usize::from(destination.rows),
                    usize::from(destination.columns),
                ],
                worker_count: 6,
                role: "row-sharded-residual".into(),
                alignment: 4,
            },
            metadata: BTreeMap::from([
                ("label".into(), "row-sharded residual add".into()),
                ("row_start".into(), destination.row_start.to_string()),
                ("rows".into(), destination.rows.to_string()),
                ("columns".into(), destination.columns.to_string()),
            ]),
        });
    }
    schedule.phases.push(Phase::Exchange { transfers });
    schedule.phases.push(Phase::Compute {
        op: OpId(compute_phase),
        commands,
    });
    Ok(destination.to_vec())
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
    if config.rows == 0
        || config.columns == 0
        || !config.columns.is_multiple_of(16)
        || config.row_block_dimension == 0
        || config.tile_count == 0
        || config.data_base >= config.data_limit
        || config.data_base & 7 != 0
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
    let mut input = Vec::with_capacity(usize::from(shard_count));
    let mut allocations = Vec::with_capacity(usize::from(shard_count) * 3);
    for shard in 0..shard_count {
        let tile = shard;
        let row_start = shard * config.row_block_dimension;
        let rows = config.row_block_dimension.min(config.rows - row_start);
        let activation_bytes = u32::from(rows) * u32::from(config.columns) * 2;
        let input_address = config.data_base;
        if input_address + activation_bytes > config.data_limit {
            return Err(CompileError::Memory(format!(
                "FP16 affine layer norm shard {shard} exceeds 0x{:x}",
                config.data_limit
            )));
        }
        let input_tensor = TensorId(usize::from(shard));
        input.push(RowShardPlacement {
            tile,
            row_start,
            rows,
            columns: config.columns,
            tensor: input_tensor,
            address: input_address,
        });
        allocations.push(Allocation {
            tensor: input_tensor,
            tile,
            address: input_address,
            size: activation_bytes,
            live_from: 0,
            live_until: usize::MAX,
            kind: AllocationKind::Home,
        });
    }
    let mut schedule = Schedule {
        layouts: Vec::new(),
        tile_count: config.tile_count,
        allocations,
        phases: Vec::new(),
        peak_sram: BTreeMap::new(),
    };
    let appended = append_affine_layer_norm_f16(
        &mut schedule,
        &input,
        AppendAffineLayerNormConfig {
            data_base: config.data_base,
            data_limit: config.data_limit,
            epsilon_bits: config.epsilon_bits,
        },
    )?;
    Ok(AffineLayerNormPlan {
        schedule,
        input,
        affine: appended.affine,
        output: appended.output,
    })
}

pub fn append_affine_layer_norm_f16(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    config: AppendAffineLayerNormConfig,
) -> Result<AppendedAffineLayerNorm, CompileError> {
    let epsilon = f32::from_bits(config.epsilon_bits);
    let epsilon_q30 = (epsilon * (1u64 << 30) as f32).round() as u32;
    let columns = input.first().map(|shard| shard.columns).unwrap_or(0);
    if input.is_empty()
        || columns == 0
        || !columns.is_multiple_of(16)
        || input.iter().any(|shard| {
            shard.columns != columns || shard.tile >= schedule.tile_count || shard.address & 7 != 0
        })
        || config.data_base & 7 != 0
        || config.data_base >= config.data_limit
        || !epsilon.is_finite()
        || epsilon_q30 == 0
        || epsilon_q30 >= 1 << 20
    {
        return Err(CompileError::Graph(
            "appended FP16 affine layer norm has incompatible row shards, SRAM, or epsilon".into(),
        ));
    }
    let mut next_tensor = schedule
        .allocations
        .iter()
        .map(|allocation| allocation.tensor.0)
        .max()
        .unwrap_or(0)
        + 1;
    let phase = schedule.phases.len();
    let affine_bytes = u32::from(columns) * 4;
    let mut affine = Vec::with_capacity(input.len());
    let mut output = Vec::with_capacity(input.len());
    let mut commands = Vec::with_capacity(input.len());
    for shard in input {
        let activation_bytes = u32::from(shard.rows) * u32::from(columns) * 2;
        let constraint = MemoryConstraint {
            base: config.data_base,
            limit: config.data_limit,
            alignment: 8,
            placement: MemoryPlacement::Low,
        };
        let affine_address = find_free_region(
            &schedule.allocations,
            shard.tile,
            affine_bytes,
            0,
            usize::MAX,
            constraint,
        )?;
        let affine_tensor = TensorId(next_tensor);
        let output_tensor = TensorId(next_tensor + 1);
        next_tensor += 2;
        schedule.allocations.push(Allocation {
            tensor: affine_tensor,
            tile: shard.tile,
            address: affine_address,
            size: affine_bytes,
            live_from: 0,
            live_until: usize::MAX,
            kind: AllocationKind::Home,
        });
        let output_address = find_free_region(
            &schedule.allocations,
            shard.tile,
            activation_bytes,
            phase,
            usize::MAX,
            constraint,
        )?;
        affine.push(RowShardPlacement {
            tile: shard.tile,
            row_start: 0,
            rows: 2,
            columns,
            tensor: affine_tensor,
            address: affine_address,
        });
        output.push(RowShardPlacement {
            tensor: output_tensor,
            address: output_address,
            ..*shard
        });
        schedule.allocations.push(Allocation {
            tensor: output_tensor,
            tile: shard.tile,
            address: output_address,
            size: activation_bytes,
            live_from: phase,
            live_until: usize::MAX,
            kind: AllocationKind::Home,
        });
        commands.push(KernelCommand {
            tile: shard.tile,
            output: output_tensor,
            inputs: vec![shard.tensor, affine_tensor],
            arguments: vec![u32::from(shard.rows), u32::from(columns), epsilon_q30],
            specialization: SpecializationKey {
                operation: "layer_norm_affine_f16".into(),
                shape: vec![usize::from(shard.rows), usize::from(columns)],
                worker_count: 6,
                role: "normalize".into(),
                alignment: 8,
            },
            metadata: BTreeMap::from([
                ("row_start".into(), shard.row_start.to_string()),
                ("rows".into(), shard.rows.to_string()),
                ("columns".into(), columns.to_string()),
            ]),
        });
    }
    schedule.phases.push(Phase::Compute {
        op: OpId(phase),
        commands,
    });
    Ok(AppendedAffineLayerNorm { affine, output })
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

    #[test]
    fn row_sharded_add_stages_only_remote_sources() {
        let shard = |tile, row_start, tensor| RowShardPlacement {
            tile,
            row_start,
            rows: 4,
            columns: 16,
            tensor: TensorId(tensor),
            address: 0x90000,
        };
        let destination = vec![shard(0, 0, 0), shard(1, 4, 1)];
        let source = vec![shard(0, 0, 2), shard(2, 4, 3)];
        let mut schedule = Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations: Vec::new(),
            tile_count: 3,
            peak_sram: BTreeMap::new(),
        };

        let output =
            append_add_f16_row_shards_in_place(&mut schedule, &destination, &source).unwrap();

        assert_eq!(output, destination);
        assert!(matches!(
            &schedule.phases[0],
            Phase::Exchange { transfers } if transfers.len() == 1
        ));
        assert!(matches!(
            &schedule.phases[1],
            Phase::Compute { commands, .. } if commands.len() == destination.len()
        ));
    }

    #[test]
    fn appended_norm_reuses_storage_after_tensor_lifetime_ends() {
        let dead = TensorId(0);
        let input = RowShardPlacement {
            tile: 0,
            row_start: 0,
            rows: 4,
            columns: 16,
            tensor: TensorId(1),
            address: 0xb0000,
        };
        let mut schedule = Schedule {
            layouts: Vec::new(),
            phases: vec![Phase::Compute {
                op: OpId(0),
                commands: Vec::new(),
            }],
            allocations: vec![
                Allocation {
                    tensor: dead,
                    tile: 0,
                    address: 0xa0000,
                    size: 128,
                    live_from: 0,
                    live_until: usize::MAX,
                    kind: AllocationKind::Home,
                },
                Allocation {
                    tensor: input.tensor,
                    tile: 0,
                    address: input.address,
                    size: 128,
                    live_from: 0,
                    live_until: usize::MAX,
                    kind: AllocationKind::Home,
                },
            ],
            tile_count: 1,
            peak_sram: BTreeMap::new(),
        };
        end_tensor_lifetimes(&mut schedule, [dead]).unwrap();

        let appended = append_affine_layer_norm_f16(
            &mut schedule,
            &[input],
            AppendAffineLayerNormConfig {
                data_base: 0xa0000,
                data_limit: 0xa00c0,
                epsilon_bits: 1e-6f32.to_bits(),
            },
        )
        .unwrap();

        assert_eq!(appended.output[0].address, 0xa0000);
        assert_ne!(appended.affine[0].address, appended.output[0].address);
    }
}
