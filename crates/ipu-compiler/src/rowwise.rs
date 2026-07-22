use crate::{
    Allocation, AllocationKind, BlockPlacement, CompileError, KernelCommand, MemoryArena,
    MemoryConstraint, MemoryPlacement, MemoryPolicy, OpId, Phase, Schedule, SpecializationKey,
    TensorId, Transfer, allocate_from_occupied, allocate_from_occupied_arenas,
    find_free_region_in_arenas, occupied_intervals_by_tile,
};
use rustc_hash::FxHashSet as HashSet;
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

pub fn choose_row_shard_rows_for_copies_in_arenas(
    schedule: &Schedule,
    rows: u16,
    columns: u16,
    maximum_rows: u16,
    copies: usize,
    arenas: &[MemoryArena],
) -> Option<u16> {
    if rows == 0 || columns == 0 || maximum_rows == 0 || copies == 0 || arenas.is_empty() {
        return None;
    }
    let phase = schedule.phases.len() + 1;
    let data_base = arenas.iter().map(|arena| arena.base).min()?;
    let data_limit = arenas.iter().map(|arena| arena.limit).max()?;
    let occupied = occupied_intervals_by_tile(
        &schedule.allocations,
        schedule.tile_count,
        phase,
        usize::MAX,
        data_base,
        data_limit,
    );
    let mut previous_grid = None;
    for target_rows in (1..=maximum_rows.min(rows)).rev() {
        let row_grid = rows.div_ceil(target_rows);
        if row_grid > schedule.tile_count || previous_grid == Some(row_grid) {
            continue;
        }
        previous_grid = Some(row_grid);
        let base_rows = rows / row_grid;
        let larger_shards = rows % row_grid;
        let fits = (0..row_grid).all(|index| {
            let shard_rows = base_rows + u16::from(index < larger_shards);
            let mut tile_occupied = occupied[usize::from(index)].clone();
            u32::from(shard_rows)
                .checked_mul(u32::from(columns))
                .and_then(|elements| elements.checked_mul(2))
                .is_some_and(|bytes| {
                    (0..copies).all(|_| {
                        allocate_from_occupied_arenas(&mut tile_occupied, bytes, arenas, 8).is_ok()
                    })
                })
        });
        if fits {
            return Some(base_rows + u16::from(larger_shards != 0));
        }
    }
    None
}

pub fn end_tensor_lifetimes(
    schedule: &mut Schedule,
    tensors: impl IntoIterator<Item = TensorId>,
) -> Result<(), CompileError> {
    let phase = schedule.phases.len();
    let tensors = tensors.into_iter().collect::<HashSet<_>>();
    let mut found = HashSet::default();
    let mut regions = vec![Vec::<(u32, u32)>::new(); usize::from(schedule.tile_count)];
    for allocation in &schedule.allocations {
        if tensors.contains(&allocation.tensor) {
            found.insert(allocation.tensor);
            if allocation.kind == AllocationKind::Home {
                regions[usize::from(allocation.tile)].push((
                    allocation.address,
                    allocation.address.saturating_add(allocation.size),
                ));
            }
        }
    }
    if let Some(tensor) = tensors.difference(&found).next() {
        return Err(CompileError::Graph(format!(
            "cannot end unknown tensor {}",
            tensor.0
        )));
    }
    for tile_regions in &mut regions {
        tile_regions.sort_unstable();
        let mut merged = Vec::<(u32, u32)>::with_capacity(tile_regions.len());
        for &(start, end) in tile_regions.iter() {
            if let Some(previous) = merged.last_mut()
                && start <= previous.1
            {
                previous.1 = previous.1.max(end);
            } else {
                merged.push((start, end));
            }
        }
        *tile_regions = merged;
    }
    for allocation in &mut schedule.allocations {
        if allocation.kind != AllocationKind::Home || allocation.live_until != usize::MAX {
            continue;
        }
        let allocation_end = allocation.address.saturating_add(allocation.size);
        let aliases_target = regions[usize::from(allocation.tile)]
            .iter()
            .any(|&(start, end)| allocation.address >= start && allocation_end <= end);
        if aliases_target {
            if allocation.live_from >= phase {
                return Err(CompileError::Graph(
                    "tensor cannot end before an alias becomes live".into(),
                ));
            }
            allocation.live_until = phase;
        }
    }
    Ok(())
}

pub fn make_tensors_resident(
    schedule: &mut Schedule,
    tensors: impl IntoIterator<Item = TensorId>,
) -> Result<(), CompileError> {
    make_tensors_resident_since(schedule, 0, tensors)
}

pub fn make_tensors_resident_since(
    schedule: &mut Schedule,
    allocation_start: usize,
    tensors: impl IntoIterator<Item = TensorId>,
) -> Result<(), CompileError> {
    let tensors = tensors.into_iter().collect::<HashSet<_>>();
    debug_assert!(
        schedule.allocations[..allocation_start]
            .iter()
            .all(|allocation| !tensors.contains(&allocation.tensor)),
        "newly appended tensors must not have allocations before allocation_start"
    );
    let mut found = HashSet::default();
    let allocations = schedule
        .allocations
        .get_mut(allocation_start..)
        .ok_or_else(|| CompileError::Graph("resident allocation start is out of range".into()))?;
    for allocation in allocations {
        if allocation.kind == AllocationKind::Home && tensors.contains(&allocation.tensor) {
            found.insert(allocation.tensor);
            allocation.live_from = 0;
            allocation.live_until = usize::MAX;
        }
    }
    if let Some(tensor) = tensors.difference(&found).next() {
        return Err(CompileError::Graph(format!(
            "cannot retain unknown tensor {}",
            tensor.0
        )));
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
                staging_address: Some(ipu_exchange::EXCHANGE_WINDOW_BASE),
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
    append_c16_to_a16_row_shards_impl(schedule, source, config, false)
}

pub fn append_c16_to_a16_row_shards_gelu_f16(
    schedule: &mut Schedule,
    source: &[BlockPlacement],
    config: RowShardTransitionConfig,
) -> Result<Vec<RowShardPlacement>, CompileError> {
    append_c16_to_a16_row_shards_impl(schedule, source, config, true)
}

/// Applies GeLU while converting independently placed C16 blocks to A16.
/// Keeping the 64-column blocks distributed avoids requiring a contiguous
/// full-width row shard for wide MLP intermediates.
pub fn append_c16_to_a16_blocks_gelu_f16(
    schedule: &mut Schedule,
    source: &[BlockPlacement],
    data_base: u32,
    data_limit: u32,
) -> Result<Vec<BlockPlacement>, CompileError> {
    append_c16_to_a16_blocks_gelu_f16_in_arenas(
        schedule,
        source,
        &[MemoryArena {
            base: data_base,
            limit: data_limit,
            placement: MemoryPlacement::Low,
        }],
    )
}

pub fn append_c16_to_a16_blocks_gelu_f16_in_arenas(
    schedule: &mut Schedule,
    source: &[BlockPlacement],
    arenas: &[MemoryArena],
) -> Result<Vec<BlockPlacement>, CompileError> {
    if source.is_empty()
        || arenas.is_empty()
        || arenas
            .iter()
            .any(|arena| arena.base & 7 != 0 || arena.base >= arena.limit)
        || source
            .iter()
            .any(|block| block.columns == 0 || !block.columns.is_multiple_of(16))
    {
        return Err(CompileError::Graph(
            "blocked C16-to-A16 GeLU requires 16-column-aligned blocks and aligned SRAM".into(),
        ));
    }
    let phase = schedule.phases.len();
    let mut next_tensor = schedule
        .allocations
        .iter()
        .map(|allocation| allocation.tensor.0)
        .max()
        .unwrap_or(0)
        + 1;
    let mut output = Vec::with_capacity(source.len());
    let mut commands = Vec::with_capacity(source.len());
    let data_base = arenas.iter().map(|arena| arena.base).min().unwrap();
    let data_limit = arenas.iter().map(|arena| arena.limit).max().unwrap();
    let mut occupied = occupied_intervals_by_tile(
        &schedule.allocations,
        schedule.tile_count,
        phase,
        usize::MAX,
        data_base,
        data_limit,
    );
    for block in source {
        let bytes = u32::from(block.rows) * u32::from(block.columns) * 2;
        let address = allocate_from_occupied_arenas(
            &mut occupied[usize::from(block.tile)],
            bytes,
            arenas,
            8,
        )?;
        let tensor = TensorId(next_tensor);
        next_tensor += 1;
        let placement = BlockPlacement {
            tensor,
            address,
            ..*block
        };
        schedule.allocations.push(Allocation {
            tensor,
            tile: block.tile,
            address,
            size: bytes,
            live_from: phase,
            live_until: usize::MAX,
            kind: AllocationKind::Home,
        });
        commands.push(KernelCommand {
            tile: block.tile,
            output: tensor,
            inputs: vec![block.tensor, block.tensor],
            arguments: vec![
                u32::from(block.rows),
                u32::from(block.rows / 6) | (u32::from(block.columns / 16) << 16),
                u32::from(block.rows % 6),
            ],
            specialization: SpecializationKey {
                operation: "gelu_f16_c16_to_a16".into(),
                shape: vec![usize::from(block.rows), usize::from(block.columns)],
                worker_count: 6,
                role: "blocked-gelu".into(),
                alignment: 8,
            },
            metadata: BTreeMap::from([
                ("label".into(), "blocked GeLU".into()),
                ("row_start".into(), block.row_start.to_string()),
                ("column_start".into(), block.column_start.to_string()),
            ]),
        });
        output.push(placement);
    }
    schedule.phases.push(Phase::Compute {
        op: OpId(phase),
        commands,
    });
    Ok(output)
}

fn append_c16_to_a16_row_shards_impl(
    schedule: &mut Schedule,
    source: &[BlockPlacement],
    config: RowShardTransitionConfig,
    gelu: bool,
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
    let first_compute_phase = schedule.phases.len() + 1;
    let mut destinations = Vec::with_capacity(rows.len());
    let mut destination_blocks = Vec::with_capacity(rows.len());
    let mut occupied = occupied_intervals_by_tile(
        &schedule.allocations,
        schedule.tile_count,
        first_compute_phase,
        usize::MAX,
        config.data_base,
        config.data_limit,
    );
    for (destination_tile, (_block_row, mut blocks)) in rows.into_iter().enumerate() {
        let destination_tile = u16::try_from(destination_tile)
            .map_err(|_| CompileError::Graph("row-shard destination tile overflow".into()))?;
        blocks.sort_by_key(|block| block.column_start);
        let first = blocks[0];
        let mut next_column = 0u16;
        for block in &blocks {
            if block.row_start != first.row_start
                || block.rows != first.rows
                || block.columns == 0
                || !block.columns.is_multiple_of(16)
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
        let address = allocate_from_occupied(
            &mut occupied[usize::from(destination_tile)],
            bytes,
            MemoryConstraint {
                base: config.data_base,
                limit: config.data_limit,
                alignment: 8,
                placement: MemoryPlacement::Low,
            },
        )?;
        let destination_tensor = TensorId(next_tensor);
        next_tensor += 1;
        schedule.allocations.push(Allocation {
            tensor: destination_tensor,
            tile: destination_tile,
            address,
            size: bytes,
            live_from: first_compute_phase,
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
        destination_blocks.push((destination_tile, address, blocks));
    }

    let maximum_rows = destination_blocks
        .iter()
        .flat_map(|(_, _, blocks)| blocks.iter().map(|block| block.rows))
        .max()
        .unwrap();
    let maximum_columns = destination_blocks
        .iter()
        .flat_map(|(_, _, blocks)| blocks.iter().map(|block| block.columns))
        .max()
        .unwrap();
    let maximum_block_bytes = u32::from(maximum_rows) * u32::from(maximum_columns) * 2;
    let blocks_per_pass =
        usize::try_from(ipu_exchange::EXCHANGE_WINDOW_BYTES / maximum_block_bytes)
            .map_err(|_| CompileError::Graph("row-shard pass size overflow".into()))?;
    if blocks_per_pass == 0 {
        return Err(CompileError::Graph(
            "one row-shard block exceeds the exchange window".into(),
        ));
    }
    let block_count = destination_blocks[0].2.len();
    for first_block in (0..block_count).step_by(blocks_per_pass) {
        let exchange_phase = schedule.phases.len();
        let compute_phase = exchange_phase + 1;
        let mut transfers = Vec::new();
        let mut commands = Vec::new();
        for &(destination_tile, address, ref blocks) in &destination_blocks {
            for block in blocks.iter().skip(first_block).take(blocks_per_pass) {
                let block_bytes = u32::from(block.rows) * u32::from(block.columns) * 2;
                if block.tile != destination_tile {
                    let staging_address = ipu_exchange::EXCHANGE_WINDOW_BASE
                        + u32::from(block.block_column - first_block as u16) * block_bytes;
                    transfers.push(Transfer {
                        source_tile: block.tile,
                        destination_tile,
                        tensor: block.tensor,
                        bytes: block_bytes,
                        staging_address: Some(staging_address),
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
                let units_per_worker = u32::from(block.rows / 6);
                let extra_workers = u32::from(block.rows % 6);
                commands.push(KernelCommand {
                    tile: destination_tile,
                    output: output_alias,
                    inputs: vec![block.tensor, block.tensor],
                    arguments: if gelu {
                        vec![
                            u32::from(block.rows),
                            units_per_worker | (u32::from(block.columns / 16) << 16),
                            extra_workers,
                        ]
                    } else {
                        vec![u32::from(block.rows), u32::from(block.columns / 16)]
                    },
                    specialization: SpecializationKey {
                        operation: if gelu {
                            "gelu_f16_c16_to_a16"
                        } else {
                            "relayout_f16_c16_to_a16"
                        }
                        .into(),
                        shape: vec![usize::from(block.rows), usize::from(block.columns)],
                        worker_count: 6,
                        role: if gelu {
                            "blocked-to-row-sharded-gelu"
                        } else {
                            "blocked-to-row-sharded"
                        }
                        .into(),
                        alignment: 8,
                    },
                    metadata: BTreeMap::from([
                        (
                            "label".into(),
                            if gelu {
                                "gather blocked activation with GeLU"
                            } else {
                                "gather blocked activation"
                            }
                            .into(),
                        ),
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
    }
    Ok(destinations)
}

pub fn append_c16_to_a16_row_shards_reblocked_in_arenas(
    schedule: &mut Schedule,
    source: &[BlockPlacement],
    columns: u16,
    row_block_dimension: u16,
    arenas: &[MemoryArena],
) -> Result<Vec<RowShardPlacement>, CompileError> {
    if source.is_empty()
        || columns == 0
        || !columns.is_multiple_of(64)
        || row_block_dimension == 0
        || arenas.is_empty()
        || arenas
            .iter()
            .any(|arena| arena.base & 7 != 0 || arena.base >= arena.limit)
    {
        return Err(CompileError::Graph(
            "C16 row reblocking requires 64-column blocks, a row block size, and aligned SRAM"
                .into(),
        ));
    }
    let source = source
        .iter()
        .map(|block| ReblockSource {
            tensor: block.tensor,
            tile: block.tile,
            address: block.address,
            row_start: block.row_start,
            rows: block.rows,
            column_start: block.column_start,
            columns: block.columns,
            layout: ReblockSourceLayout::C16,
        })
        .collect::<Vec<_>>();
    append_to_a16_row_shards_reblocked_in_arenas(
        schedule,
        &source,
        columns,
        row_block_dimension,
        arenas,
    )
}

pub fn append_a16_to_a16_row_shards_reblocked_in_arenas(
    schedule: &mut Schedule,
    source: &[RowShardPlacement],
    row_block_dimension: u16,
    arenas: &[MemoryArena],
) -> Result<Vec<RowShardPlacement>, CompileError> {
    let columns = source.first().map(|shard| shard.columns).unwrap_or(0);
    if source.is_empty()
        || columns == 0
        || !columns.is_multiple_of(16)
        || source
            .iter()
            .any(|shard| shard.rows == 0 || shard.columns != columns)
        || row_block_dimension == 0
        || arenas.is_empty()
        || arenas
            .iter()
            .any(|arena| arena.base & 7 != 0 || arena.base >= arena.limit)
    {
        return Err(CompileError::Graph(
            "A16 row reblocking requires consistent 16-column panels, a row block size, and aligned SRAM"
                .into(),
        ));
    }
    let source = source
        .iter()
        .flat_map(|shard| {
            let panel_stride = u32::from(shard.rows) * 32;
            let transfer_limit =
                ipu_exchange::EXCHANGE_WINDOW_BYTES.min(ipu_exchange::MAX_TRANSFER_WORDS * 4);
            let panels_per_fragment = (transfer_limit / panel_stride)
                .max(1)
                .min(u32::from(shard.columns / 16)) as u16;
            let columns_per_fragment = panels_per_fragment * 16;
            (0..shard.columns)
                .step_by(usize::from(columns_per_fragment))
                .map(move |column_start| {
                    let columns = columns_per_fragment.min(shard.columns - column_start);
                    ReblockSource {
                        tensor: shard.tensor,
                        tile: shard.tile,
                        address: shard.address
                            + u32::from(column_start) * u32::from(shard.rows) * 2,
                        row_start: shard.row_start,
                        rows: shard.rows,
                        column_start,
                        columns,
                        layout: ReblockSourceLayout::A16,
                    }
                })
        })
        .collect::<Vec<_>>();
    append_to_a16_row_shards_reblocked_in_arenas(
        schedule,
        &source,
        columns,
        row_block_dimension,
        arenas,
    )
}

#[derive(Clone, Copy)]
enum ReblockSourceLayout {
    A16,
    C16,
}

#[derive(Clone, Copy)]
struct ReblockSource {
    tensor: TensorId,
    tile: u16,
    address: u32,
    row_start: u16,
    rows: u16,
    column_start: u16,
    columns: u16,
    layout: ReblockSourceLayout,
}

fn append_to_a16_row_shards_reblocked_in_arenas(
    schedule: &mut Schedule,
    source: &[ReblockSource],
    columns: u16,
    row_block_dimension: u16,
    arenas: &[MemoryArena],
) -> Result<Vec<RowShardPlacement>, CompileError> {
    let rows = source
        .iter()
        .map(|block| block.row_start + block.rows)
        .max()
        .unwrap();
    let row_grid = rows.div_ceil(row_block_dimension);
    let base_rows = rows / row_grid;
    let larger_shards = rows % row_grid;
    let phase = schedule.phases.len();
    let first_compute_phase = phase + 1;
    let data_base = arenas.iter().map(|arena| arena.base).min().unwrap();
    let data_limit = arenas.iter().map(|arena| arena.limit).max().unwrap();
    let mut occupied = occupied_intervals_by_tile(
        &schedule.allocations,
        schedule.tile_count,
        first_compute_phase,
        usize::MAX,
        data_base,
        data_limit,
    );
    let mut next_tensor = schedule
        .allocations
        .iter()
        .map(|allocation| allocation.tensor.0)
        .max()
        .unwrap_or(0)
        + 1;
    let mut destinations = Vec::with_capacity(usize::from(row_grid));
    let mut groups = Vec::with_capacity(usize::from(row_grid));
    let mut row_start = 0u16;
    for index in 0..row_grid {
        let destination_rows = base_rows + u16::from(index < larger_shards);
        let destination_tile = index;
        if destination_tile >= schedule.tile_count {
            return Err(CompileError::Graph(
                "C16 row reblocking needs more destination tiles".into(),
            ));
        }
        let bytes = u32::from(destination_rows) * u32::from(columns) * 2;
        let address = allocate_from_occupied_arenas(
            &mut occupied[usize::from(destination_tile)],
            bytes,
            arenas,
            8,
        )?;
        let tensor = TensorId(next_tensor);
        next_tensor += 1;
        schedule.allocations.push(Allocation {
            tensor,
            tile: destination_tile,
            address,
            size: bytes,
            live_from: first_compute_phase,
            live_until: usize::MAX,
            kind: AllocationKind::Home,
        });
        destinations.push(RowShardPlacement {
            tile: destination_tile,
            row_start,
            rows: destination_rows,
            columns,
            tensor,
            address,
        });
        let row_end = row_start + destination_rows;
        let mut fragments = source
            .iter()
            .filter_map(|block| {
                let overlap_start = row_start.max(block.row_start);
                let overlap_end = row_end.min(block.row_start + block.rows);
                (overlap_start < overlap_end).then(|| {
                    (
                        block,
                        overlap_start - block.row_start,
                        overlap_start - row_start,
                        overlap_end - overlap_start,
                    )
                })
            })
            .collect::<Vec<_>>();
        fragments.sort_unstable_by_key(|(block, source_row, _, _)| {
            (block.column_start, block.row_start + *source_row)
        });
        let covered = fragments
            .iter()
            .map(|(block, _, _, copy_rows)| u32::from(*copy_rows) * u32::from(block.columns))
            .sum::<u32>();
        if covered != u32::from(destination_rows) * u32::from(columns) {
            return Err(CompileError::Graph(format!(
                "C16 row shard {row_start}..{row_end} has incomplete source coverage"
            )));
        }
        groups.push((
            destination_tile,
            address,
            destination_rows,
            tensor,
            fragments,
        ));
        row_start = row_end;
    }

    let mut cursors = vec![0usize; groups.len()];
    while cursors
        .iter()
        .zip(&groups)
        .any(|(&fragment, group)| fragment < group.4.len())
    {
        let exchange_phase = schedule.phases.len();
        let compute_phase = exchange_phase + 1;
        let mut transfers = Vec::new();
        let mut commands = Vec::new();
        for (group_index, (tile, address, destination_rows, destination_tensor, fragments)) in
            groups.iter().enumerate()
        {
            let mut staging_cursor = ipu_exchange::EXCHANGE_WINDOW_BASE;
            while let Some(&(block, source_row_start, destination_row_start, copy_rows)) =
                fragments.get(cursors[group_index])
            {
                if block.columns == 0 || !block.columns.is_multiple_of(16) {
                    return Err(CompileError::Graph(
                        "C16 reblocking requires source blocks divisible into 16-column panels"
                            .into(),
                    ));
                }
                let panel_count = block.columns / 16;
                let panel_stride = u32::from(block.rows) * 32;
                let source_bytes =
                    u32::from(panel_count - 1) * panel_stride + u32::from(copy_rows) * 32;
                let transfer_limit =
                    ipu_exchange::EXCHANGE_WINDOW_BYTES.min(ipu_exchange::MAX_TRANSFER_WORDS * 4);
                if block.tile != *tile && source_bytes > transfer_limit {
                    return Err(CompileError::Memory(format!(
                        "row-shard fragment requires {source_bytes} exchange bytes"
                    )));
                }
                if block.tile != *tile
                    && staging_cursor + source_bytes
                        > ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES
                {
                    break;
                }
                let source_alias = TensorId(next_tensor);
                next_tensor += 1;
                let source_address = block.address + u32::from(source_row_start) * 32;
                schedule.allocations.push(Allocation {
                    tensor: source_alias,
                    tile: block.tile,
                    address: source_address,
                    size: source_bytes,
                    live_from: exchange_phase,
                    live_until: compute_phase + 1,
                    kind: AllocationKind::HomeAlias {
                        source: block.tensor,
                    },
                });
                if block.tile != *tile {
                    transfers.push(Transfer {
                        source_tile: block.tile,
                        destination_tile: *tile,
                        tensor: source_alias,
                        bytes: source_bytes,
                        staging_address: Some(staging_cursor),
                    });
                    staging_cursor = crate::align_u32(staging_cursor + source_bytes, 32);
                }
                let output_alias = TensorId(next_tensor);
                next_tensor += 1;
                let output_address =
                    *address + u32::from(block.column_start) * u32::from(*destination_rows) * 2;
                schedule.allocations.push(Allocation {
                    tensor: output_alias,
                    tile: *tile,
                    address: output_address,
                    size: u32::from(*destination_rows) * u32::from(block.columns) * 2,
                    live_from: compute_phase,
                    live_until: compute_phase + 1,
                    kind: AllocationKind::HomeAlias {
                        source: *destination_tensor,
                    },
                });
                commands.push(KernelCommand {
                    tile: *tile,
                    output: output_alias,
                    inputs: vec![source_alias, source_alias],
                    arguments: vec![
                        crate::pack_reblock_row_pair(block.rows, *destination_rows)?,
                        crate::pack_reblock_row_pair(0, destination_row_start)?,
                        u32::from(copy_rows) | (u32::from(panel_count) << 16),
                    ],
                    specialization: SpecializationKey {
                        operation: match block.layout {
                            ReblockSourceLayout::A16 => "reblock_f16_a16_to_a16",
                            ReblockSourceLayout::C16 => "reblock_f16_c16_to_a16",
                        }
                        .into(),
                        shape: vec![
                            usize::from(block.rows),
                            usize::from(*destination_rows),
                            usize::from(copy_rows),
                        ],
                        worker_count: 1,
                        role: match block.layout {
                            ReblockSourceLayout::A16 => "A16 row-shard transition",
                            ReblockSourceLayout::C16 => "C16 to A16 row reblocking",
                        }
                        .into(),
                        alignment: 8,
                    },
                    metadata: BTreeMap::from([
                        ("label".into(), "reblock GEMM output rows".into()),
                        ("row_start".into(), block.row_start.to_string()),
                        ("column_start".into(), block.column_start.to_string()),
                        ("copy_rows".into(), copy_rows.to_string()),
                    ]),
                });
                cursors[group_index] += 1;
            }
        }
        schedule.phases.push(Phase::Exchange { transfers });
        schedule.phases.push(Phase::Compute {
            op: OpId(compute_phase),
            commands,
        });
    }
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
    append_affine_layer_norm_f16_in_arenas(
        schedule,
        input,
        config,
        &[MemoryArena {
            base: config.data_base,
            limit: config.data_limit,
            placement: MemoryPlacement::Low,
        }],
    )
}

pub fn append_affine_layer_norm_f16_in_arenas(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    config: AppendAffineLayerNormConfig,
    affine_arenas: &[MemoryArena],
) -> Result<AppendedAffineLayerNorm, CompileError> {
    append_affine_layer_norm_f16_with_memory_policy(
        schedule,
        input,
        config,
        &MemoryPolicy {
            resident: affine_arenas.to_vec(),
            transient: vec![MemoryArena {
                base: config.data_base,
                limit: config.data_limit,
                placement: MemoryPlacement::Low,
            }],
            resident_tile_assignment: crate::ResidentTileAssignment::Balanced,
            allocation_occupancy: crate::AllocationOccupancyCache::default(),
        },
    )
}

pub fn append_affine_layer_norm_f16_with_memory_policy(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    config: AppendAffineLayerNormConfig,
    memory: &MemoryPolicy,
) -> Result<AppendedAffineLayerNorm, CompileError> {
    append_affine_layer_norm_f16_impl(schedule, input, None, config, memory)
}

pub fn append_add_affine_layer_norm_f16_with_memory_policy(
    schedule: &mut Schedule,
    residual: &[RowShardPlacement],
    right: &[RowShardPlacement],
    config: AppendAffineLayerNormConfig,
    memory: &MemoryPolicy,
) -> Result<AppendedAffineLayerNorm, CompileError> {
    if residual.is_empty()
        || residual.len() != right.len()
        || residual.iter().any(|destination| {
            !right.iter().any(|source| {
                source.tile == destination.tile
                    && source.row_start == destination.row_start
                    && source.rows == destination.rows
                    && source.columns == destination.columns
            })
        })
    {
        return Err(CompileError::Graph(
            "fused residual LayerNorm requires matching colocated row shards".into(),
        ));
    }
    append_affine_layer_norm_f16_impl(schedule, residual, Some(right), config, memory)
}

fn append_affine_layer_norm_f16_impl(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    residual_right: Option<&[RowShardPlacement]>,
    config: AppendAffineLayerNormConfig,
    memory: &MemoryPolicy,
) -> Result<AppendedAffineLayerNorm, CompileError> {
    memory.validate()?;
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
    let exchange_phase = schedule.phases.len();
    let compute_phase = exchange_phase + 1;
    let affine_row_bytes = u32::from(columns) * 2;
    let affine_bytes = affine_row_bytes * 2;
    let mut resident_pressure = vec![0u64; usize::from(schedule.tile_count)];
    for allocation in &schedule.allocations {
        if allocation.live_from < compute_phase
            && allocation.live_until > 0
            && memory
                .resident
                .iter()
                .any(|arena| allocation.address >= arena.base && allocation.address < arena.limit)
        {
            resident_pressure[usize::from(allocation.tile)] += u64::from(allocation.size);
        }
    }
    let owner = input
        .iter()
        .min_by_key(|candidate| {
            (
                resident_pressure[usize::from(candidate.tile)],
                candidate.tile,
            )
        })
        .unwrap();
    let constraint = MemoryConstraint {
        base: config.data_base,
        limit: config.data_limit,
        alignment: 8,
        placement: MemoryPlacement::High,
    };
    let affine_address = find_free_region_in_arenas(
        &schedule.allocations,
        owner.tile,
        affine_bytes,
        0,
        usize::MAX,
        &memory.resident,
        8,
    )?;
    let affine_tensors = [TensorId(next_tensor), TensorId(next_tensor + 1)];
    next_tensor += 2;
    let affine = affine_tensors
        .iter()
        .enumerate()
        .map(|(row, &tensor)| RowShardPlacement {
            tile: owner.tile,
            row_start: row as u16,
            rows: 1,
            columns,
            tensor,
            address: affine_address + row as u32 * affine_row_bytes,
        })
        .collect::<Vec<_>>();
    for placement in &affine {
        schedule.allocations.push(Allocation {
            tensor: placement.tensor,
            tile: placement.tile,
            address: placement.address,
            size: affine_row_bytes,
            live_from: 0,
            live_until: usize::MAX,
            kind: AllocationKind::Home,
        });
    }

    let mut transfers = Vec::with_capacity(input.len().saturating_sub(1) * 2);
    let mut output = Vec::with_capacity(input.len());
    let mut commands = Vec::with_capacity(input.len());
    let transient_base = memory
        .transient
        .iter()
        .map(|arena| arena.base)
        .min()
        .unwrap();
    let transient_limit = memory
        .transient
        .iter()
        .map(|arena| arena.limit)
        .max()
        .unwrap();
    let mut output_occupied = occupied_intervals_by_tile(
        &schedule.allocations,
        schedule.tile_count,
        compute_phase,
        usize::MAX,
        transient_base,
        transient_limit,
    );
    for shard in input {
        let right = residual_right.map(|right| {
            right
                .iter()
                .find(|source| {
                    source.tile == shard.tile
                        && source.row_start == shard.row_start
                        && source.rows == shard.rows
                })
                .expect("fused residual shards were validated")
        });
        let activation_bytes = u32::from(shard.rows) * u32::from(columns) * 2;
        let output_tensor = TensorId(next_tensor);
        next_tensor += 1;
        let output_address = allocate_from_occupied_arenas(
            &mut output_occupied[usize::from(shard.tile)],
            activation_bytes,
            &memory.transient,
            constraint.alignment,
        )?;
        if shard.tile != owner.tile {
            for (row, &tensor) in affine_tensors.iter().enumerate() {
                transfers.push(Transfer {
                    source_tile: owner.tile,
                    destination_tile: shard.tile,
                    tensor,
                    bytes: affine_row_bytes,
                    staging_address: Some(
                        ipu_exchange::EXCHANGE_WINDOW_BASE + row as u32 * affine_row_bytes,
                    ),
                });
            }
        }
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
            live_from: compute_phase,
            live_until: usize::MAX,
            kind: AllocationKind::Home,
        });
        let (inputs, operation, role, label) = if let Some(right) = right {
            (
                vec![shard.tensor, right.tensor, affine_tensors[0]],
                "add_layer_norm_affine_f16",
                "add-and-normalize",
                "fused residual add and LayerNorm",
            )
        } else {
            (
                vec![shard.tensor, affine_tensors[0]],
                "layer_norm_affine_f16",
                "normalize",
                "affine LayerNorm",
            )
        };
        commands.push(KernelCommand {
            tile: shard.tile,
            output: output_tensor,
            inputs,
            arguments: vec![u32::from(shard.rows), u32::from(columns), epsilon_q30],
            specialization: SpecializationKey {
                operation: operation.into(),
                shape: vec![usize::from(shard.rows), usize::from(columns)],
                worker_count: 6,
                role: role.into(),
                alignment: 8,
            },
            metadata: BTreeMap::from([
                ("label".into(), label.into()),
                ("row_start".into(), shard.row_start.to_string()),
                ("rows".into(), shard.rows.to_string()),
                ("columns".into(), columns.to_string()),
            ]),
        });
    }
    schedule.phases.push(Phase::Exchange { transfers });
    schedule.phases.push(Phase::Compute {
        op: OpId(compute_phase),
        commands,
    });
    Ok(AppendedAffineLayerNorm { affine, output })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_tensors_cover_the_full_execution_lifetime() {
        let tensor = TensorId(4);
        let mut schedule = Schedule {
            layouts: Vec::new(),
            phases: vec![Phase::Exchange {
                transfers: Vec::new(),
            }],
            allocations: vec![Allocation {
                tensor,
                tile: 0,
                address: 0x80000,
                size: 128,
                live_from: 1,
                live_until: 2,
                kind: AllocationKind::Home,
            }],
            tile_count: 1,
            peak_sram: BTreeMap::new(),
        };

        make_tensors_resident(&mut schedule, [tensor]).unwrap();

        let allocation = &schedule.allocations[0];
        assert_eq!(allocation.live_from, 0);
        assert_eq!(allocation.live_until, usize::MAX);
    }

    #[test]
    fn colocated_residual_layer_norm_uses_one_fused_compute_phase() {
        let residual = RowShardPlacement {
            tile: 0,
            row_start: 0,
            rows: 8,
            columns: 64,
            tensor: TensorId(0),
            address: 0xa0000,
        };
        let right = RowShardPlacement {
            tensor: TensorId(1),
            address: 0xa1000,
            ..residual
        };
        let bytes = u32::from(residual.rows) * u32::from(residual.columns) * 2;
        let mut schedule = Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations: [residual, right]
                .into_iter()
                .map(|shard| Allocation {
                    tensor: shard.tensor,
                    tile: shard.tile,
                    address: shard.address,
                    size: bytes,
                    live_from: 0,
                    live_until: usize::MAX,
                    kind: AllocationKind::Home,
                })
                .collect(),
            tile_count: 1,
            peak_sram: BTreeMap::new(),
        };
        let memory = MemoryPolicy {
            resident: vec![MemoryArena::high(0xc0000, 0xe8000)],
            transient: vec![MemoryArena::low(0xb0000, 0xc0000)],
            resident_tile_assignment: crate::ResidentTileAssignment::Balanced,
            allocation_occupancy: crate::AllocationOccupancyCache::default(),
        };

        let appended = append_add_affine_layer_norm_f16_with_memory_policy(
            &mut schedule,
            &[residual],
            &[right],
            AppendAffineLayerNormConfig {
                data_base: 0xb0000,
                data_limit: 0xe8000,
                epsilon_bits: 1e-6f32.to_bits(),
            },
            &memory,
        )
        .unwrap();

        assert_eq!(appended.output.len(), 1);
        assert!(
            matches!(&schedule.phases[0], Phase::Exchange { transfers } if transfers.is_empty())
        );
        let Phase::Compute { commands, .. } = &schedule.phases[1] else {
            unreachable!()
        };
        assert_eq!(commands.len(), 1);
        assert_eq!(
            commands[0].specialization.operation,
            "add_layer_norm_affine_f16"
        );
        assert_eq!(commands[0].inputs.len(), 3);
        assert_eq!(commands[0].inputs[0], residual.tensor);
        schedule.validate_allocations().unwrap();
    }

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
    fn blocked_gelu_keeps_blocks_distributed_and_allocates_by_lifetime() {
        let source = vec![
            BlockPlacement {
                tensor: TensorId(0),
                tile: 0,
                address: 0xa0000,
                block_row: 0,
                block_column: 0,
                row_start: 0,
                column_start: 0,
                rows: 13,
                columns: 64,
            },
            BlockPlacement {
                tensor: TensorId(1),
                tile: 1,
                address: 0xa0000,
                block_row: 0,
                block_column: 1,
                row_start: 0,
                column_start: 64,
                rows: 13,
                columns: 64,
            },
        ];
        let bytes = 13 * 64 * 2;
        let mut schedule = Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations: source
                .iter()
                .map(|block| Allocation {
                    tensor: block.tensor,
                    tile: block.tile,
                    address: block.address,
                    size: bytes,
                    live_from: 0,
                    live_until: usize::MAX,
                    kind: AllocationKind::Home,
                })
                .collect(),
            tile_count: 2,
            peak_sram: BTreeMap::new(),
        };

        let output =
            append_c16_to_a16_blocks_gelu_f16(&mut schedule, &source, 0xa0000, 0xe8000).unwrap();

        assert_eq!(output.len(), source.len());
        assert!(output.iter().zip(&source).all(|(output, source)| {
            output.tile == source.tile
                && output.row_start == source.row_start
                && output.column_start == source.column_start
                && output.address >= source.address + bytes
        }));
        assert!(matches!(
            &schedule.phases[0],
            Phase::Compute { commands, .. } if commands.len() == source.len()
        ));
    }

    #[test]
    fn c16_outputs_reblock_to_complete_balanced_row_shards() {
        let mut source = Vec::new();
        let mut allocations = Vec::new();
        for block_row in 0..2u16 {
            for block_column in 0..2u16 {
                let tensor = TensorId(usize::from(block_row * 2 + block_column));
                let block = BlockPlacement {
                    tensor,
                    tile: block_row * 2 + block_column,
                    address: 0xa0000,
                    block_row,
                    block_column,
                    row_start: block_row * 13,
                    rows: 13,
                    column_start: block_column * 64,
                    columns: 64,
                };
                allocations.push(Allocation {
                    tensor,
                    tile: block.tile,
                    address: block.address,
                    size: u32::from(block.rows) * u32::from(block.columns) * 2,
                    live_from: 0,
                    live_until: usize::MAX,
                    kind: AllocationKind::Home,
                });
                source.push(block);
            }
        }
        let mut schedule = Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations,
            tile_count: 8,
            peak_sram: BTreeMap::new(),
        };

        let output = append_c16_to_a16_row_shards_reblocked_in_arenas(
            &mut schedule,
            &source,
            128,
            20,
            &[MemoryArena::low(0xb0000, 0xe8000)],
        )
        .unwrap();

        assert_eq!(output.iter().map(|shard| shard.rows).sum::<u16>(), 26);
        assert!(output.iter().all(|shard| shard.columns == 128));
        assert!(schedule.phases.iter().any(|phase| matches!(
            phase,
            Phase::Compute { commands, .. }
                if commands.iter().any(|command| {
                    command.specialization.operation == "reblock_f16_c16_to_a16"
                        && command.arguments.len() == 3
                })
        )));
        assert!(schedule.phases.iter().all(|phase| {
            match phase {
                Phase::Exchange { transfers } => transfers
                    .iter()
                    .all(|transfer| transfer.bytes <= 13 * 64 * 2),
                Phase::Compute { .. } => true,
            }
        }));
        schedule.validate_allocations().unwrap();
        schedule
            .lower_tile_programs(&ipu_exchange::Topology::c600())
            .unwrap();
    }

    #[test]
    fn a16_row_shards_transition_to_a_balanced_grid() {
        let source = (0..2u16)
            .map(|index| RowShardPlacement {
                tensor: TensorId(usize::from(index)),
                tile: index,
                address: 0xa0000,
                row_start: index * 40,
                rows: 40,
                columns: 1024,
            })
            .collect::<Vec<_>>();
        let mut schedule = Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations: source
                .iter()
                .map(|shard| Allocation {
                    tensor: shard.tensor,
                    tile: shard.tile,
                    address: shard.address,
                    size: u32::from(shard.rows) * u32::from(shard.columns) * 2,
                    live_from: 0,
                    live_until: usize::MAX,
                    kind: AllocationKind::Home,
                })
                .collect(),
            tile_count: 8,
            peak_sram: BTreeMap::new(),
        };

        let output = append_a16_to_a16_row_shards_reblocked_in_arenas(
            &mut schedule,
            &source,
            20,
            &[MemoryArena::low(0xb0000, 0xe8000)],
        )
        .unwrap();

        assert_eq!(output.iter().map(|shard| shard.rows).sum::<u16>(), 80);
        assert_eq!(output.len(), 4);
        assert!(schedule.phases.iter().any(|phase| matches!(
            phase,
            Phase::Compute { commands, .. }
                if commands.iter().any(|command| {
                    command.specialization.operation == "reblock_f16_a16_to_a16"
                })
        )));
        assert!(schedule.phases.iter().all(|phase| {
            match phase {
                Phase::Exchange { transfers } => transfers
                    .iter()
                    .all(|transfer| transfer.bytes <= ipu_exchange::MAX_TRANSFER_WORDS * 4),
                Phase::Compute { .. } => true,
            }
        }));
        schedule.validate_allocations().unwrap();
        schedule
            .lower_tile_programs(&ipu_exchange::Topology::c600())
            .unwrap();
    }

    #[test]
    fn row_shard_choice_accounts_for_simultaneously_live_copies() {
        let schedule = Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations: Vec::new(),
            tile_count: 2,
            peak_sram: BTreeMap::new(),
        };
        let arena = MemoryArena::low(0x1000, 0x1180);

        assert_eq!(
            choose_row_shard_rows_for_copies_in_arenas(&schedule, 12, 16, 12, 2, &[arena]),
            Some(6)
        );
        assert_eq!(
            choose_row_shard_rows_for_copies_in_arenas(&schedule, 12, 16, 12, 3, &[arena]),
            None
        );
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
