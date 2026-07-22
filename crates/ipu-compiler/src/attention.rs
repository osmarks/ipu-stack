use crate::{
    Allocation, AllocationKind, CompileError, KernelCommand, MemoryArena, MemoryPlacement, OpId,
    Phase, RowShardPlacement, Schedule, SpecializationKey, TensorId, Transfer,
    allocate_from_occupied_arenas, find_free_region_in_arenas, occupied_intervals_by_tile,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashAttentionConfig {
    pub batch_size: u16,
    /// Zero uses `sequence_length`, preserving self-attention behavior.
    pub query_sequence_length: u16,
    pub sequence_length: u16,
    pub hidden_size: u16,
    pub attention_heads: u16,
    /// Zero selects the smallest worker-saturating block that fits on the tiles.
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
    pub scores: TensorId,
    pub weights: TensorId,
    pub output: TensorId,
    pub query_address: u32,
    pub accumulator_address: u32,
    pub scores_address: u32,
    pub weights_address: u32,
    pub output_address: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionKeyValuePlacement {
    pub batch: u16,
    pub head: u16,
    pub key_row_start: u16,
    pub key_rows: u16,
    pub tile: u16,
    pub key_tensor: TensorId,
    pub value_tensor: TensorId,
    pub key_address: u32,
    pub value_address: u32,
    pub matrix_size: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlashAttentionPlan {
    pub schedule: Schedule,
    pub tasks: Vec<AttentionTaskPlacement>,
    pub key_values: Vec<AttentionKeyValuePlacement>,
    pub head_dimension: u16,
    pub padded_head_dimension: u16,
    pub query_block_rows: u16,
    pub key_block_rows: u16,
    pub key_block_columns: u16,
}

pub fn append_flash_attention_from_a16_qkv(
    schedule: &mut Schedule,
    query: &[RowShardPlacement],
    key: &[RowShardPlacement],
    value: &[RowShardPlacement],
    config: FlashAttentionConfig,
) -> Result<FlashAttentionPlan, CompileError> {
    append_flash_attention_from_a16_qkv_in_arenas(
        schedule,
        query,
        key,
        value,
        config,
        &[MemoryArena {
            base: config.data_base,
            limit: config.data_limit,
        }],
    )
}

pub fn append_flash_attention_from_a16_qkv_in_arenas(
    schedule: &mut Schedule,
    query: &[RowShardPlacement],
    key: &[RowShardPlacement],
    value: &[RowShardPlacement],
    config: FlashAttentionConfig,
    arenas: &[MemoryArena],
) -> Result<FlashAttentionPlan, CompileError> {
    let query_sequence_length = if config.query_sequence_length == 0 {
        config.sequence_length
    } else {
        config.query_sequence_length
    };
    if config.batch_size != 1 || query.is_empty() || key.is_empty() || key.len() != value.len() {
        return Err(CompileError::Graph(
            "row-sharded FlashAttention composition currently requires one batch and matching K/V shards"
                .into(),
        ));
    }
    if query.iter().map(|shard| shard.rows).sum::<u16>() != query_sequence_length
        || key.iter().map(|shard| shard.rows).sum::<u16>() != config.sequence_length
        || query
            .iter()
            .any(|shard| shard.columns != config.hidden_size)
    {
        return Err(CompileError::Graph(
            "row-sharded query layout does not match the attention configuration".into(),
        ));
    }
    for (key, value) in key.iter().zip(value) {
        if key.row_start != value.row_start
            || key.rows != value.rows
            || key.columns != config.hidden_size
            || value.columns != config.hidden_size
        {
            return Err(CompileError::Graph(
                "row-sharded K/V layouts do not match the attention configuration".into(),
            ));
        }
    }
    // Preserve the row-shard contract consumed by downstream composed operators.
    // K/V packing below can independently assemble the larger blocks selected by
    // the attention cost model from any number of input shards.
    let query_block_rows = if config.query_block_rows == 0 {
        query.iter().map(|shard| shard.rows).max().unwrap()
    } else {
        config.query_block_rows
    };
    let mut plan = plan_flash_attention(FlashAttentionConfig {
        query_block_rows,
        ..config
    })?;
    relocate_appended_attention(schedule, &mut plan, arenas)?;
    let tensor_base = schedule
        .allocations
        .iter()
        .map(|allocation| allocation.tensor.0)
        .max()
        .unwrap_or(0)
        + 1;
    remap_attention_tensors(&mut plan, tensor_base)?;
    let mut next_tensor = schedule
        .allocations
        .iter()
        .chain(&plan.schedule.allocations)
        .map(|allocation| allocation.tensor.0)
        .max()
        .unwrap_or(0)
        + 1;

    append_attention_pack_phase(
        schedule,
        &mut next_tensor,
        query,
        &plan,
        AttentionPackKind::Query,
    )?;
    append_attention_pack_phase(
        schedule,
        &mut next_tensor,
        key,
        &plan,
        AttentionPackKind::Key,
    )?;
    append_attention_pack_phase(
        schedule,
        &mut next_tensor,
        value,
        &plan,
        AttentionPackKind::Value,
    )?;
    for allocation in &mut plan.schedule.allocations {
        if allocation.kind == AllocationKind::Home && allocation.live_from == 0 {
            allocation.live_from = 1;
        }
    }
    crate::append_child_schedule(schedule, &mut plan.schedule)?;
    Ok(plan)
}

fn relocate_appended_attention(
    parent: &Schedule,
    plan: &mut FlashAttentionPlan,
    arenas: &[MemoryArena],
) -> Result<(), CompileError> {
    if arenas.is_empty() {
        return Err(CompileError::Memory(
            "attention relocation requires an SRAM arena".into(),
        ));
    }
    let data_base = arenas.iter().map(|arena| arena.base).min().unwrap();
    let data_limit = arenas.iter().map(|arena| arena.limit).max().unwrap();
    let live_from = parent.phases.len();
    let mut occupied = occupied_intervals_by_tile(
        &parent.allocations,
        parent.tile_count,
        live_from,
        usize::MAX,
        data_base,
        data_limit,
    );
    let mut score_arenas = vec![MemoryArena {
        base: ipu_package::IPU21_INTERLEAVED_MEMORY_BASE,
        limit: ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT,
    }];
    for arena in arenas {
        if !score_arenas.contains(arena) {
            score_arenas.push(*arena);
        }
    }
    let mut relocated = BTreeMap::<usize, u32>::new();
    for task in &plan.tasks {
        let size = attention_scratch_bytes(
            task.query_rows,
            plan.key_block_columns,
            plan.padded_head_dimension,
        );
        let address = allocate_from_occupied_arenas(
            &mut occupied[usize::from(task.tile)],
            size,
            &score_arenas,
            8,
            MemoryPlacement::Low,
        )?;
        relocated.insert(task.scores.0, address);
    }
    let mut regions = Vec::<(TensorId, u16, u32, u32)>::new();
    for task in &plan.tasks {
        regions.extend([
            (
                task.query,
                task.tile,
                task.query_address,
                u32::from(task.query_rows) * u32::from(plan.padded_head_dimension) * 2,
            ),
            (
                task.accumulator,
                task.tile,
                task.accumulator_address,
                u32::from(task.query_rows) * (u32::from(plan.head_dimension) * 4 + 8),
            ),
            (
                task.weights,
                task.tile,
                task.weights_address,
                u32::from(task.query_rows) * (u32::from(plan.key_block_columns) * 2 + 8),
            ),
            (
                task.output,
                task.tile,
                task.output_address,
                u32::from(task.query_rows) * u32::from(plan.head_dimension) * 2,
            ),
        ]);
    }
    for key_value in &plan.key_values {
        regions.extend([
            (
                key_value.key_tensor,
                key_value.tile,
                key_value.key_address,
                key_value.matrix_size,
            ),
            (
                key_value.value_tensor,
                key_value.tile,
                key_value.value_address,
                key_value.matrix_size,
            ),
        ]);
    }
    regions.sort_unstable_by_key(|&(_, _, _, size)| std::cmp::Reverse(size));
    for &(tensor, tile, _address, size) in &regions {
        let address = allocate_from_occupied_arenas(
            &mut occupied[usize::from(tile)],
            size,
            arenas,
            8,
            MemoryPlacement::Low,
        )?;
        relocated.insert(tensor.0, address);
    }
    for task in &mut plan.tasks {
        task.query_address = relocated[&task.query.0];
        task.accumulator_address = relocated[&task.accumulator.0];
        task.scores_address = relocated[&task.scores.0];
        task.weights_address = relocated[&task.weights.0];
        task.output_address = relocated[&task.output.0];
    }
    for key_value in &mut plan.key_values {
        key_value.key_address = relocated[&key_value.key_tensor.0];
        key_value.value_address = relocated[&key_value.value_tensor.0];
    }
    for allocation in &mut plan.schedule.allocations {
        if allocation.kind == AllocationKind::Home
            && let Some(&address) = relocated.get(&allocation.tensor.0)
        {
            allocation.address = address;
        }
    }
    plan.schedule.peak_sram.clear();
    Ok(())
}

pub fn append_flash_attention_to_a16_row_shards(
    schedule: &mut Schedule,
    plan: &FlashAttentionPlan,
    data_base: u32,
    data_limit: u32,
) -> Result<Vec<RowShardPlacement>, CompileError> {
    append_flash_attention_to_a16_row_shards_in_arenas(
        schedule,
        plan,
        &[MemoryArena {
            base: data_base,
            limit: data_limit,
        }],
    )
}

pub fn append_flash_attention_to_a16_row_shards_in_arenas(
    schedule: &mut Schedule,
    plan: &FlashAttentionPlan,
    arenas: &[MemoryArena],
) -> Result<Vec<RowShardPlacement>, CompileError> {
    if plan.tasks.is_empty()
        || arenas.is_empty()
        || arenas
            .iter()
            .any(|arena| arena.base & 7 != 0 || arena.base >= arena.limit)
    {
        return Err(CompileError::Graph(
            "attention output gather requires tasks and an aligned SRAM arena".into(),
        ));
    }
    if plan.tasks.iter().any(|task| task.batch != 0) {
        return Err(CompileError::Graph(
            "row-sharded attention output currently requires one batch".into(),
        ));
    }
    let head_count = plan.tasks.iter().map(|task| task.head).max().unwrap() + 1;
    let hidden_size = head_count
        .checked_mul(plan.head_dimension)
        .ok_or_else(|| CompileError::Graph("attention hidden size overflow".into()))?;
    let mut groups = BTreeMap::<(u16, u16), Vec<&AttentionTaskPlacement>>::new();
    for task in &plan.tasks {
        groups
            .entry((task.query_row_start, task.query_rows))
            .or_default()
            .push(task);
    }
    let mut next_tensor = schedule
        .allocations
        .iter()
        .map(|allocation| allocation.tensor.0)
        .max()
        .unwrap_or(0)
        + 1;
    let first_compute_phase = schedule.phases.len() + 1;
    let mut destinations = Vec::with_capacity(groups.len());
    let mut output_groups = Vec::with_capacity(groups.len());
    for (destination_index, ((row_start, rows), mut tasks)) in groups.into_iter().enumerate() {
        tasks.sort_by_key(|task| task.head);
        if tasks.len() != usize::from(head_count)
            || tasks
                .iter()
                .enumerate()
                .any(|(head, task)| usize::from(task.head) != head)
        {
            return Err(CompileError::Graph(
                "attention output has incomplete head coverage".into(),
            ));
        }
        let destination_tile = u16::try_from(destination_index)
            .map_err(|_| CompileError::Graph("attention destination tile overflow".into()))?;
        if destination_tile >= schedule.tile_count {
            return Err(CompileError::Graph(
                "attention output needs more row-shard destination tiles".into(),
            ));
        }
        let activation_bytes = u32::from(rows) * u32::from(hidden_size) * 2;
        let address = find_free_region_in_arenas(
            &schedule.allocations,
            destination_tile,
            activation_bytes,
            first_compute_phase,
            usize::MAX,
            arenas,
            8,
            MemoryPlacement::Low,
        )?;
        let destination_tensor = fresh_tensor(&mut next_tensor);
        schedule.allocations.push(Allocation {
            tensor: destination_tensor,
            tile: destination_tile,
            address,
            size: activation_bytes,
            live_from: first_compute_phase,
            live_until: usize::MAX,
            kind: AllocationKind::Home,
        });
        destinations.push(RowShardPlacement {
            tile: destination_tile,
            row_start,
            rows,
            columns: hidden_size,
            tensor: destination_tensor,
            address,
        });
        output_groups.push((destination_tile, row_start, rows, address, tasks));
    }

    let maximum_head_bytes = output_groups
        .iter()
        .map(|(_, _, rows, _, _)| u32::from(*rows) * u32::from(plan.head_dimension) * 2)
        .max()
        .unwrap();
    let heads_per_pass = u16::try_from(ipu_exchange::EXCHANGE_WINDOW_BYTES / maximum_head_bytes)
        .unwrap_or(u16::MAX)
        .min(head_count);
    if heads_per_pass == 0 {
        return Err(CompileError::Graph(format!(
            "one attention head row shard is {maximum_head_bytes} bytes, larger than the exchange window"
        )));
    }
    for first_head in (0..head_count).step_by(usize::from(heads_per_pass)) {
        let exchange_phase = schedule.phases.len();
        let compute_phase = exchange_phase + 1;
        let pass_heads = (head_count - first_head).min(heads_per_pass);
        let mut transfers = Vec::new();
        let mut commands = Vec::new();
        for &(destination_tile, row_start, rows, address, ref tasks) in &output_groups {
            let activation_bytes = u32::from(rows) * u32::from(hidden_size) * 2;
            let head_bytes = u32::from(rows) * u32::from(plan.head_dimension) * 2;
            for task in &tasks[usize::from(first_head)..usize::from(first_head + pass_heads)] {
                let staging_offset = u32::from(task.head - first_head) * head_bytes;
                if task.tile != destination_tile {
                    transfers.push(Transfer {
                        source_tile: task.tile,
                        destination_tile,
                        tensor: task.output,
                        bytes: head_bytes,
                    });
                    schedule.allocations.push(Allocation {
                        tensor: task.output,
                        tile: destination_tile,
                        address: ipu_exchange::EXCHANGE_WINDOW_BASE + staging_offset,
                        size: head_bytes,
                        live_from: exchange_phase,
                        live_until: compute_phase,
                        kind: AllocationKind::ExchangeStaging {
                            phase: exchange_phase,
                        },
                    });
                }
                let output_alias = fresh_tensor(&mut next_tensor);
                schedule.allocations.push(Allocation {
                    tensor: output_alias,
                    tile: destination_tile,
                    address,
                    size: activation_bytes,
                    live_from: compute_phase,
                    live_until: compute_phase,
                    kind: AllocationKind::Home,
                });
                commands.push(KernelCommand {
                    tile: destination_tile,
                    output: output_alias,
                    inputs: vec![task.output, task.output],
                    arguments: vec![u32::from(rows), u32::from(task.head * plan.head_dimension)],
                    specialization: SpecializationKey {
                        operation: "attention_unpack_head_f16".into(),
                        shape: vec![usize::from(rows), usize::from(plan.head_dimension)],
                        worker_count: 6,
                        role: format!("head-{}-rows-{row_start}", task.head).into(),
                        alignment: 8,
                    },
                    metadata: BTreeMap::from([
                        ("label".into(), "gather attention heads".into()),
                        ("head".into(), task.head.to_string()),
                        ("row_start".into(), row_start.to_string()),
                        ("rows".into(), rows.to_string()),
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

#[derive(Clone, Copy)]
enum AttentionPackKind {
    Query,
    Key,
    Value,
}

fn append_attention_pack_phase(
    schedule: &mut Schedule,
    next_tensor: &mut usize,
    source: &[RowShardPlacement],
    plan: &FlashAttentionPlan,
    kind: AttentionPackKind,
) -> Result<(), CompileError> {
    let exchange_phase = schedule.phases.len();
    let compute_phase = exchange_phase + 1;
    let mut transfers = Vec::new();
    let mut commands = Vec::new();
    let mut staging_cursors =
        vec![ipu_exchange::EXCHANGE_WINDOW_BASE; usize::from(schedule.tile_count)];
    let mut append = |shard: &RowShardPlacement,
                      head: u16,
                      destination_tile: u16,
                      output: TensorId,
                      operation: &'static str,
                      source_row_start: u16,
                      destination_row_start: u16,
                      copy_rows: u16,
                      destination_rows: u16|
     -> Result<(), CompileError> {
        const ROW_OFFSET_BITS: u32 = 10;
        const ROW_OFFSET_LIMIT: u16 = 1 << ROW_OFFSET_BITS;
        if source_row_start >= ROW_OFFSET_LIMIT || destination_row_start >= ROW_OFFSET_LIMIT {
            return Err(CompileError::Graph(
                "attention pack block-local row offset exceeds 10-bit ABI".into(),
            ));
        }
        let source_panel = head * plan.head_dimension / 16;
        let source_offset = head * plan.head_dimension % 16;
        let source_columns = (source_offset + plan.head_dimension).div_ceil(16) * 16;
        let bytes = u32::from(shard.rows) * u32::from(source_columns) * 2;
        if bytes > ipu_exchange::MAX_TRANSFER_WORDS * 4 {
            return Err(CompileError::Graph(format!(
                "attention source shard is {bytes} bytes, larger than one exchange transfer"
            )));
        }
        let alias = TensorId(*next_tensor);
        *next_tensor += 1;
        let source_address = shard
            .address
            .checked_add(u32::from(source_panel) * u32::from(shard.rows) * 32)
            .ok_or_else(|| CompileError::Memory("attention source address overflow".into()))?;
        schedule.allocations.push(Allocation {
            tensor: alias,
            tile: shard.tile,
            address: source_address,
            size: bytes,
            live_from: exchange_phase,
            live_until: compute_phase + 1,
            kind: AllocationKind::HomeAlias {
                source: shard.tensor,
            },
        });
        if shard.tile != destination_tile {
            let cursor = &mut staging_cursors[usize::from(destination_tile)];
            let staging_address = *cursor;
            *cursor = crate::align_u32(
                cursor.checked_add(bytes).ok_or_else(|| {
                    CompileError::Memory("attention staging address overflow".into())
                })?,
                32,
            );
            if *cursor > ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES {
                return Err(CompileError::Memory(format!(
                    "attention packing exhausts tile {destination_tile} exchange window"
                )));
            }
            transfers.push(Transfer {
                source_tile: shard.tile,
                destination_tile,
                tensor: alias,
                bytes,
            });
            schedule.allocations.push(Allocation {
                tensor: alias,
                tile: destination_tile,
                address: staging_address,
                size: bytes,
                live_from: exchange_phase,
                live_until: compute_phase,
                kind: AllocationKind::ExchangeStaging {
                    phase: exchange_phase,
                },
            });
        }
        commands.push(KernelCommand {
            tile: destination_tile,
            output,
            inputs: vec![alias, alias],
            arguments: vec![
                u32::from(shard.rows),
                u32::from(source_offset),
                u32::from(source_row_start) | (u32::from(destination_row_start) << ROW_OFFSET_BITS),
                u32::from(copy_rows),
                u32::from(destination_rows),
            ],
            specialization: SpecializationKey {
                operation: operation.into(),
                shape: vec![
                    usize::from(shard.rows),
                    usize::from(source_row_start),
                    usize::from(copy_rows),
                    usize::from(destination_rows),
                    usize::from(plan.head_dimension),
                ],
                worker_count: 6,
                role: format!(
                    "head-{head}-rows-{}..{}",
                    shard.row_start + source_row_start,
                    shard.row_start + source_row_start + copy_rows
                )
                .into(),
                alignment: 8,
            },
            metadata: BTreeMap::from([
                (
                    "row_start".into(),
                    (shard.row_start + source_row_start).to_string(),
                ),
                ("rows".into(), copy_rows.to_string()),
                (
                    "destination_row_start".into(),
                    destination_row_start.to_string(),
                ),
                ("head".into(), head.to_string()),
            ]),
        });
        Ok(())
    };
    let mut append_range = |row_start: u16,
                            rows: u16,
                            head: u16,
                            destination_tile: u16,
                            output: TensorId,
                            operation: &'static str|
     -> Result<(), CompileError> {
        let row_end = row_start + rows;
        let mut covered = 0u16;
        for shard in source {
            let overlap_start = row_start.max(shard.row_start);
            let overlap_end = row_end.min(shard.row_start + shard.rows);
            if overlap_start >= overlap_end {
                continue;
            }
            let copy_rows = overlap_end - overlap_start;
            append(
                shard,
                head,
                destination_tile,
                output,
                operation,
                overlap_start - shard.row_start,
                overlap_start - row_start,
                copy_rows,
                rows,
            )?;
            covered = covered.checked_add(copy_rows).ok_or_else(|| {
                CompileError::Graph("attention packed row coverage overflow".into())
            })?;
        }
        if covered != rows {
            return Err(CompileError::Graph(format!(
                "attention rows {row_start}..{row_end} have only {covered} source rows"
            )));
        }
        Ok(())
    };
    match kind {
        AttentionPackKind::Query => {
            for task in &plan.tasks {
                append_range(
                    task.query_row_start,
                    task.query_rows,
                    task.head,
                    task.tile,
                    task.query,
                    "attention_pack_query_f16",
                )?;
            }
        }
        AttentionPackKind::Key | AttentionPackKind::Value => {
            for block in &plan.key_values {
                let (output, operation) = match kind {
                    AttentionPackKind::Key => (block.key_tensor, "attention_pack_key_f16"),
                    AttentionPackKind::Value => (block.value_tensor, "attention_pack_value_f16"),
                    AttentionPackKind::Query => unreachable!(),
                };
                append_range(
                    block.key_row_start,
                    block.key_rows,
                    block.head,
                    block.tile,
                    output,
                    operation,
                )?;
            }
        }
    }
    schedule.phases.push(Phase::Exchange { transfers });
    schedule.phases.push(Phase::Compute {
        op: OpId(compute_phase),
        commands,
    });
    Ok(())
}

fn remap_attention_tensors(plan: &mut FlashAttentionPlan, base: usize) -> Result<(), CompileError> {
    let remap = |tensor: &mut TensorId| -> Result<(), CompileError> {
        tensor.0 = tensor
            .0
            .checked_add(base)
            .ok_or_else(|| CompileError::Graph("attention tensor ID overflow".into()))?;
        Ok(())
    };
    for task in &mut plan.tasks {
        for tensor in [
            &mut task.query,
            &mut task.accumulator,
            &mut task.scores,
            &mut task.weights,
            &mut task.output,
        ] {
            remap(tensor)?;
        }
    }
    for block in &mut plan.key_values {
        remap(&mut block.key_tensor)?;
        remap(&mut block.value_tensor)?;
    }
    for allocation in &mut plan.schedule.allocations {
        remap(&mut allocation.tensor)?;
        if let AllocationKind::HomeAlias { source } = &mut allocation.kind {
            remap(source)?;
        }
    }
    for phase in &mut plan.schedule.phases {
        match phase {
            Phase::Exchange { transfers } => {
                for transfer in transfers {
                    remap(&mut transfer.tensor)?;
                }
            }
            Phase::Compute { commands, .. } => {
                for command in commands {
                    remap(&mut command.output)?;
                    for input in &mut command.inputs {
                        remap(input)?;
                    }
                }
            }
        }
    }
    Ok(())
}

pub fn plan_flash_attention(
    config: FlashAttentionConfig,
) -> Result<FlashAttentionPlan, CompileError> {
    validate(config)?;
    let query_sequence_length = if config.query_sequence_length == 0 {
        config.sequence_length
    } else {
        config.query_sequence_length
    };
    let head_dimension = config.hidden_size / config.attention_heads;
    if !head_dimension.is_multiple_of(2) {
        return Err(CompileError::Graph(
            "FP16 attention finalization requires an even head dimension".into(),
        ));
    }

    let head_count = usize::from(config.batch_size) * usize::from(config.attention_heads);
    const WORKER_COUNT: u16 = 6;
    let query_block_rows = if config.query_block_rows == 0 {
        (WORKER_COUNT.min(query_sequence_length)..=query_sequence_length)
            .find(|&rows| {
                head_count * usize::from(query_sequence_length.div_ceil(rows))
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
    .min(query_sequence_length);
    let query_blocks = query_sequence_length.div_ceil(query_block_rows);
    let task_count = head_count * usize::from(query_blocks);
    if task_count > usize::from(config.tile_count) {
        return Err(CompileError::Graph(format!(
            "attention needs {task_count} query tasks but only {} tiles are available; increase query_block_rows",
            config.tile_count
        )));
    }

    let padded_head_dimension = head_dimension.div_ceil(16) * 16;
    let transfer_limit = ipu_exchange::MAX_TRANSFER_WORDS * 4;
    let maximum_key_rows = (1..=config.sequence_length)
        .take_while(|&rows| key_value_block_bytes(rows, padded_head_dimension) <= transfer_limit)
        .last()
        .ok_or_else(|| {
            CompileError::Graph("one attention key row exceeds the exchange limit".into())
        })?;
    let key_block_rows = if config.key_block_rows == 0 {
        select_key_block_rows(config.sequence_length, maximum_key_rows)
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
    let key_block_columns = key_block_rows.div_ceil(16) * 16;

    let mut cursors = vec![config.data_base; usize::from(config.tile_count)];
    let mut allocations = Vec::new();
    let mut tasks = Vec::with_capacity(task_count);
    let mut next_tensor = 0usize;
    for batch in 0..config.batch_size {
        for head in 0..config.attention_heads {
            for query_block in 0..query_blocks {
                let (query_row_start, query_rows) =
                    balanced_partition(query_sequence_length, query_blocks, query_block);
                let tile = u16::try_from(tasks.len())
                    .map_err(|_| CompileError::Graph("attention tile index overflow".into()))?;
                let elements = u32::from(query_rows) * u32::from(head_dimension);
                let query_bytes = u32::from(query_rows) * u32::from(padded_head_dimension) * 2;
                // The online state is [accumulator rows][maximum per row][denominator per row].
                let accumulator_bytes = elements * 4 + u32::from(query_rows) * 8;
                let qk_scores_bytes = u32::from(query_rows) * u32::from(key_block_columns) * 2;
                let scores_bytes =
                    attention_scratch_bytes(query_rows, key_block_columns, padded_head_dimension);
                let weights_bytes = qk_scores_bytes + u32::from(query_rows) * 8;
                let output_bytes = elements * 2;
                let cursor = &mut cursors[usize::from(tile)];
                let query_address = allocate(cursor, query_bytes, 8)?;
                let accumulator_address = allocate(cursor, accumulator_bytes, 8)?;
                let scores_address = ipu_package::IPU21_INTERLEAVED_MEMORY_BASE;
                if scores_address + scores_bytes > ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT {
                    return Err(CompileError::Memory(format!(
                        "attention score block for tile {tile} exceeds interleaved SRAM"
                    )));
                }
                let weights_address = allocate(cursor, weights_bytes, 8)?;
                let output_address = allocate(cursor, output_bytes, 8)?;
                let query = fresh_tensor(&mut next_tensor);
                let accumulator = fresh_tensor(&mut next_tensor);
                let scores = fresh_tensor(&mut next_tensor);
                let weights = fresh_tensor(&mut next_tensor);
                let output = fresh_tensor(&mut next_tensor);
                tasks.push(AttentionTaskPlacement {
                    batch,
                    head,
                    query_row_start,
                    query_rows,
                    tile,
                    query,
                    accumulator,
                    scores,
                    weights,
                    output,
                    query_address,
                    accumulator_address,
                    scores_address,
                    weights_address,
                    output_address,
                });
            }
        }
    }

    let mut key_values = Vec::with_capacity(head_count * usize::from(key_blocks));
    for batch in 0..config.batch_size {
        for head in 0..config.attention_heads {
            let head_tiles = tasks
                .iter()
                .filter(|task| task.batch == batch && task.head == head)
                .map(|task| task.tile)
                .collect::<Vec<_>>();
            debug_assert!(!head_tiles.is_empty());
            for key_block in 0..key_blocks {
                let owner = if head_tiles.len() == 1 {
                    let global_head = usize::from(batch) * usize::from(config.attention_heads)
                        + usize::from(head);
                    u16::try_from(
                        (global_head * usize::from(key_blocks) + usize::from(key_block))
                            % usize::from(config.tile_count),
                    )
                    .map_err(|_| CompileError::Graph("attention owner tile overflow".into()))?
                } else {
                    head_tiles[usize::from(key_block) % head_tiles.len()]
                };
                let (key_row_start, key_rows) =
                    balanced_partition(config.sequence_length, key_blocks, key_block);
                let matrix_size = matrix_storage_bytes(key_block_columns, padded_head_dimension);
                let key_address = allocate(&mut cursors[usize::from(owner)], matrix_size, 8)?;
                let value_address = allocate(&mut cursors[usize::from(owner)], matrix_size, 8)?;
                key_values.push(AttentionKeyValuePlacement {
                    batch,
                    head,
                    key_row_start,
                    key_rows,
                    tile: owner,
                    key_tensor: fresh_tensor(&mut next_tensor),
                    value_tensor: fresh_tensor(&mut next_tensor),
                    key_address,
                    value_address,
                    matrix_size,
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

    let final_phase = usize::from(key_blocks) * 5 + 1;
    for task in &tasks {
        allocations.extend([
            home(
                task.query,
                task.tile,
                task.query_address,
                u32::from(task.query_rows) * u32::from(padded_head_dimension) * 2,
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
                task.scores,
                task.tile,
                task.scores_address,
                attention_scratch_bytes(task.query_rows, key_block_columns, padded_head_dimension),
                0,
                final_phase,
            ),
            home(
                task.weights,
                task.tile,
                task.weights_address,
                u32::from(task.query_rows) * (u32::from(key_block_columns) * 2 + 8),
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
        allocations.extend([
            home(
                block.key_tensor,
                block.tile,
                block.key_address,
                block.matrix_size,
                0,
                final_phase,
            ),
            home(
                block.value_tensor,
                block.tile,
                block.value_address,
                block.matrix_size,
                0,
                final_phase,
            ),
        ]);
    }

    let mut phases = Vec::with_capacity(final_phase);
    for key_block in 0..key_blocks {
        let exchange_phase = phases.len();
        let qk_phase = exchange_phase + 1;
        let softmax_phase = exchange_phase + 2;
        let pv_phase = exchange_phase + 3;
        let merge_phase = exchange_phase + 4;
        let initial = u32::from(key_block == 0);
        let final_block = u32::from(key_block + 1 == key_blocks);
        let merge_role = match (initial != 0, final_block != 0) {
            (true, true) => "single",
            (true, false) => "initial",
            (false, true) => "final",
            (false, false) => "middle",
        };
        let mut transfers = Vec::new();
        let mut qk_commands = Vec::with_capacity(tasks.len());
        let mut softmax_commands = Vec::with_capacity(tasks.len());
        let mut pv_commands = Vec::with_capacity(tasks.len());
        let mut merge_commands = Vec::with_capacity(tasks.len());
        for task in &tasks {
            let (key_row_start, _) =
                balanced_partition(config.sequence_length, key_blocks, key_block);
            let block = key_values
                .iter()
                .find(|block| {
                    block.batch == task.batch
                        && block.head == task.head
                        && block.key_row_start == key_row_start
                })
                .expect("each head has every key block");
            let query_size = if task.query_rows == query_block_rows {
                "large"
            } else {
                "small"
            };
            let key_size = if block.key_rows == key_block_rows {
                "large"
            } else {
                "small"
            };
            if task.tile != block.tile {
                transfers.extend([
                    Transfer {
                        source_tile: block.tile,
                        destination_tile: task.tile,
                        tensor: block.key_tensor,
                        bytes: block.matrix_size,
                    },
                    Transfer {
                        source_tile: block.tile,
                        destination_tile: task.tile,
                        tensor: block.value_tensor,
                        bytes: block.matrix_size,
                    },
                ]);
                allocations.extend([
                    Allocation {
                        tensor: block.key_tensor,
                        tile: task.tile,
                        address: ipu_exchange::EXCHANGE_WINDOW_BASE,
                        size: block.matrix_size,
                        live_from: exchange_phase,
                        live_until: qk_phase,
                        kind: AllocationKind::ExchangeStaging {
                            phase: exchange_phase,
                        },
                    },
                    Allocation {
                        tensor: block.value_tensor,
                        tile: task.tile,
                        address: ipu_exchange::EXCHANGE_WINDOW_BASE + block.matrix_size,
                        size: block.matrix_size,
                        live_from: exchange_phase,
                        live_until: pv_phase,
                        kind: AllocationKind::ExchangeStaging {
                            phase: exchange_phase,
                        },
                    },
                ]);
            }
            qk_commands.push(KernelCommand {
                tile: task.tile,
                output: task.scores,
                inputs: vec![task.query, block.key_tensor],
                arguments: Vec::new(),
                specialization: SpecializationKey {
                    operation: format!("attention_qk_init_{}_rows", query_size).into(),
                    shape: vec![
                        usize::from(task.query_rows),
                        usize::from(padded_head_dimension),
                        usize::from(key_block_columns),
                    ],
                    worker_count: 6,
                    role: format!(
                        "attention-qk-batch-{}-head-{}-queries-{}-{}-keys-{}-{}",
                        task.batch,
                        task.head,
                        task.query_row_start,
                        task.query_row_start + task.query_rows,
                        block.key_row_start,
                        block.key_row_start + block.key_rows
                    )
                    .into(),
                    alignment: 8,
                },
                metadata: BTreeMap::from([
                    ("label".into(), "FlashAttention QK AMP".into()),
                    ("batch".into(), task.batch.to_string()),
                    ("head".into(), task.head.to_string()),
                    ("query_rows".into(), task.query_rows.to_string()),
                    ("key_rows".into(), block.key_rows.to_string()),
                ]),
            });
            softmax_commands.push(KernelCommand {
                tile: task.tile,
                output: task.weights,
                inputs: vec![task.scores, task.scores],
                arguments: Vec::new(),
                specialization: SpecializationKey {
                    operation: format!("attention_softmax_{query_size}_query_{key_size}_key_f16")
                        .into(),
                    shape: vec![usize::from(task.query_rows), usize::from(block.key_rows)],
                    worker_count: 6,
                    role: format!(
                        "attention-softmax-batch-{}-head-{}-queries-{}-{}-keys-{}-{}",
                        task.batch,
                        task.head,
                        task.query_row_start,
                        task.query_row_start + task.query_rows,
                        block.key_row_start,
                        block.key_row_start + block.key_rows
                    )
                    .into(),
                    alignment: 8,
                },
                metadata: task_metadata(
                    task,
                    block,
                    config,
                    head_dimension,
                    "FlashAttention block softmax",
                )
                .into(),
            });
            pv_commands.push(KernelCommand {
                tile: task.tile,
                output: task.scores,
                inputs: vec![task.weights, block.value_tensor],
                arguments: Vec::new(),
                specialization: SpecializationKey {
                    operation: format!("attention_pv_init_{}_rows", query_size).into(),
                    shape: vec![
                        usize::from(task.query_rows),
                        usize::from(key_block_columns),
                        usize::from(padded_head_dimension),
                    ],
                    worker_count: 6,
                    role: format!("attention-pv-batch-{}-head-{}", task.batch, task.head).into(),
                    alignment: 8,
                },
                metadata: task_metadata(
                    task,
                    block,
                    config,
                    head_dimension,
                    "FlashAttention PV AMP",
                ),
            });
            merge_commands.push(KernelCommand {
                tile: task.tile,
                output: task.accumulator,
                inputs: vec![task.scores, task.weights],
                arguments: Vec::new(),
                specialization: SpecializationKey {
                    operation: format!("attention_merge_{query_size}_query_{merge_role}_block_f16")
                        .into(),
                    shape: vec![usize::from(task.query_rows), usize::from(head_dimension)],
                    worker_count: 6,
                    role: format!("attention-merge-batch-{}-head-{}", task.batch, task.head).into(),
                    alignment: 8,
                },
                metadata: task_metadata(
                    task,
                    block,
                    config,
                    head_dimension,
                    "FlashAttention FP32 block merge",
                )
                .into(),
            });
        }
        phases.push(Phase::Exchange { transfers });
        phases.push(Phase::Compute {
            op: OpId(qk_phase),
            commands: qk_commands,
        });
        phases.push(Phase::Compute {
            op: OpId(softmax_phase),
            commands: softmax_commands,
        });
        phases.push(Phase::Compute {
            op: OpId(pv_phase),
            commands: pv_commands,
        });
        phases.push(Phase::Compute {
            op: OpId(merge_phase),
            commands: merge_commands,
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
                )
                .into(),
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
        padded_head_dimension,
        query_block_rows,
        key_block_rows,
        key_block_columns,
    })
}

fn attention_scratch_bytes(
    query_rows: u16,
    key_block_columns: u16,
    padded_head_dimension: u16,
) -> u32 {
    u32::from(query_rows) * u32::from(key_block_columns.max(padded_head_dimension)) * 2
}

fn balanced_partition(total: u16, parts: u16, index: u16) -> (u16, u16) {
    let base = total / parts;
    let larger = total % parts;
    let start = index * base + index.min(larger);
    (start, base + u16::from(index < larger))
}

fn key_value_block_bytes(rows: u16, padded_dimension: u16) -> u32 {
    let padded_rows = rows.div_ceil(16) * 16;
    matrix_storage_bytes(padded_rows, padded_dimension)
}

fn matrix_storage_bytes(storage_rows: u16, padded_dimension: u16) -> u32 {
    u32::from(padded_dimension) * u32::from(storage_rows) * 2
}

fn select_key_block_rows(sequence_length: u16, maximum_rows: u16) -> u16 {
    // The critical path trades fixed synchronization per block against packing
    // and computing one padded block on a single owner tile. Two rows of block
    // work per synchronization reflects the IPU21 launch/compute crossover;
    // padding and the final tie-breakers keep the choice stable.
    const SYNC_ROW_EQUIVALENT: u32 = 2;
    (1..=maximum_rows)
        .min_by_key(|&rows| {
            let blocks = u32::from(sequence_length.div_ceil(rows));
            let storage_rows = u32::from(rows.div_ceil(16) * 16);
            (
                storage_rows + blocks * SYNC_ROW_EQUIVALENT,
                blocks * storage_rows,
                maximum_rows - rows,
            )
        })
        .expect("maximum key rows is non-zero")
}

fn validate(config: FlashAttentionConfig) -> Result<(), CompileError> {
    if config.batch_size == 0
        || config.sequence_length == 0
        || (config.query_sequence_length != 0 && config.query_sequence_length < 6)
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
    label: &str,
) -> BTreeMap<String, String> {
    BTreeMap::from([
        ("label".into(), label.into()),
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
    use std::collections::BTreeSet;

    #[test]
    fn attention_plan_shards_queries_and_bounds_exchange_blocks() {
        for hidden_size in [768, 1024, 1152] {
            for batch_size in [1, 2, 5] {
                let config = FlashAttentionConfig {
                    batch_size,
                    query_sequence_length: 0,
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
                        .all(|block| { block.matrix_size <= ipu_exchange::MAX_TRANSFER_WORDS * 4 })
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
            query_sequence_length: 0,
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

    #[test]
    fn long_sequence_splits_large_key_value_transfers() {
        let plan = plan_flash_attention(FlashAttentionConfig {
            batch_size: 1,
            query_sequence_length: 0,
            sequence_length: 1024,
            hidden_size: 1152,
            attention_heads: 16,
            query_block_rows: 0,
            key_block_rows: 92,
            tile_count: 1472,
            data_base: 0xa0000,
            data_limit: 0xe8000,
        })
        .unwrap();
        assert!(
            plan.key_values
                .iter()
                .any(|block| { block.matrix_size * 2 > ipu_exchange::MAX_TRANSFER_WORDS * 4 })
        );
        for head in 0..16 {
            let blocks = plan
                .key_values
                .iter()
                .filter(|block| block.head == head)
                .collect::<Vec<_>>();
            let owners = blocks
                .iter()
                .map(|block| block.tile)
                .collect::<BTreeSet<_>>();
            assert_eq!(owners.len(), blocks.len());
        }
    }

    #[test]
    fn short_queries_attend_to_a_long_key_value_sequence() {
        let plan = plan_flash_attention(FlashAttentionConfig {
            batch_size: 1,
            query_sequence_length: 12,
            sequence_length: 729,
            hidden_size: 1152,
            attention_heads: 16,
            query_block_rows: 12,
            key_block_rows: 13,
            tile_count: 1472,
            data_base: 0xa0000,
            data_limit: 0xe8000,
        })
        .unwrap();

        assert_eq!(plan.tasks.len(), 16);
        assert!(plan.tasks.iter().all(|task| task.query_rows == 12));
        for task in &plan.tasks {
            let scratch = plan
                .schedule
                .allocations
                .iter()
                .find(|allocation| allocation.tensor == task.scores && allocation.tile == task.tile)
                .unwrap();
            let qk_bytes = u32::from(task.query_rows) * u32::from(plan.key_block_columns) * 2;
            let pv_bytes = u32::from(task.query_rows) * u32::from(plan.padded_head_dimension) * 2;
            assert!(scratch.size >= qk_bytes.max(pv_bytes));
        }
        for head in 0..16 {
            assert_eq!(
                plan.key_values
                    .iter()
                    .filter(|block| block.head == head)
                    .map(|block| block.key_rows)
                    .sum::<u16>(),
                729
            );
        }
    }

    #[test]
    fn row_sharded_qkv_composes_into_static_attention_programs() {
        let rows = [6, 6, 5];
        let mut row_start = 0u16;
        let mut allocations = Vec::new();
        let shards = rows
            .into_iter()
            .enumerate()
            .map(|(index, shard_rows)| {
                let shard = RowShardPlacement {
                    tile: index as u16,
                    row_start,
                    rows: shard_rows,
                    columns: 88,
                    tensor: TensorId(index),
                    address: 0x90000,
                };
                row_start += shard_rows;
                allocations.push(Allocation {
                    tensor: shard.tensor,
                    tile: shard.tile,
                    address: shard.address,
                    size: u32::from(shard.rows) * u32::from(shard.columns) * 2,
                    live_from: 0,
                    live_until: usize::MAX,
                    kind: AllocationKind::Home,
                });
                shard
            })
            .collect::<Vec<_>>();
        let mut schedule = Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations,
            tile_count: 32,
            peak_sram: BTreeMap::new(),
        };

        let plan = append_flash_attention_from_a16_qkv(
            &mut schedule,
            &shards,
            &shards,
            &shards,
            FlashAttentionConfig {
                batch_size: 1,
                query_sequence_length: 0,
                sequence_length: row_start,
                hidden_size: 88,
                attention_heads: 4,
                query_block_rows: 0,
                key_block_rows: 0,
                tile_count: 32,
                data_base: 0xa0000,
                data_limit: 0xe8000,
            },
        )
        .unwrap();
        let output =
            append_flash_attention_to_a16_row_shards(&mut schedule, &plan, 0xc0000, 0xe8000)
                .unwrap();

        assert_eq!(plan.tasks.iter().map(|task| task.query_rows).max(), Some(6));
        assert!(
            plan.key_values.iter().any(|block| block.key_rows > 6),
            "K/V blocks should be reassembled across source row shards"
        );
        assert_eq!(
            output
                .iter()
                .map(|shard| u32::from(shard.rows))
                .sum::<u32>(),
            u32::from(row_start)
        );
        assert!(output.iter().all(|shard| shard.columns == 88));
        assert!(schedule.phases.iter().any(|phase| matches!(
            phase,
            Phase::Compute { commands, .. }
                if commands.iter().any(|command| command.specialization.operation == "attention_pack_query_f16")
        )));
        schedule
            .lower_tile_programs(&ipu_exchange::Topology::c600())
            .unwrap();
    }

    #[test]
    fn attention_relocation_respects_live_allocations_in_every_arena() {
        let shard = RowShardPlacement {
            tile: 0,
            row_start: 0,
            rows: 8,
            columns: 64,
            tensor: TensorId(0),
            address: 0x6c000,
        };
        let mut schedule = Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations: vec![Allocation {
                tensor: shard.tensor,
                tile: shard.tile,
                address: shard.address,
                size: u32::from(shard.rows) * u32::from(shard.columns) * 2,
                live_from: 0,
                live_until: usize::MAX,
                kind: AllocationKind::Home,
            }],
            tile_count: 16,
            peak_sram: BTreeMap::new(),
        };
        let arenas = [
            MemoryArena {
                base: 0x6c000,
                limit: 0x80000,
            },
            MemoryArena {
                base: ipu_package::IPU21_INTERLEAVED_MEMORY_BASE,
                limit: ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT,
            },
        ];

        append_flash_attention_from_a16_qkv_in_arenas(
            &mut schedule,
            &[shard],
            &[shard],
            &[shard],
            FlashAttentionConfig {
                batch_size: 1,
                query_sequence_length: 0,
                sequence_length: shard.rows,
                hidden_size: shard.columns,
                attention_heads: 1,
                query_block_rows: shard.rows,
                key_block_rows: shard.rows,
                tile_count: 16,
                data_base: arenas[0].base,
                data_limit: arenas[1].limit,
            },
            &arenas,
        )
        .unwrap();

        schedule.validate_allocations().unwrap();
    }

    #[test]
    fn attention_output_gather_splits_shards_across_exchange_passes() {
        let rows = 48u16;
        let shard_rows = 24u16;
        let hidden_size = 1152u16;
        let mut schedule = Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations: (0..rows / shard_rows)
                .map(|index| Allocation {
                    tensor: TensorId(usize::from(index)),
                    tile: index,
                    address: 0x90000,
                    size: u32::from(shard_rows) * u32::from(hidden_size) * 2,
                    live_from: 0,
                    live_until: usize::MAX,
                    kind: AllocationKind::Home,
                })
                .collect(),
            tile_count: 1472,
            peak_sram: BTreeMap::new(),
        };
        let shards = (0..rows / shard_rows)
            .map(|index| RowShardPlacement {
                tile: index,
                row_start: index * shard_rows,
                rows: shard_rows,
                columns: hidden_size,
                tensor: TensorId(usize::from(index)),
                address: 0x90000,
            })
            .collect::<Vec<_>>();
        let plan = append_flash_attention_from_a16_qkv(
            &mut schedule,
            &shards,
            &shards,
            &shards,
            FlashAttentionConfig {
                batch_size: 1,
                query_sequence_length: 0,
                sequence_length: rows,
                hidden_size,
                attention_heads: 16,
                query_block_rows: shard_rows,
                key_block_rows: 0,
                tile_count: 1472,
                data_base: 0xa0000,
                data_limit: 0xe8000,
            },
        )
        .unwrap();
        let output =
            append_flash_attention_to_a16_row_shards(&mut schedule, &plan, 0xc0000, 0xe8000)
                .unwrap();

        assert!(output.iter().all(|shard| {
            u32::from(shard.rows) * u32::from(shard.columns) * 2
                > ipu_exchange::EXCHANGE_WINDOW_BYTES
        }));
        let gather_phases = schedule
            .phases
            .iter()
            .filter(|phase| matches!(
                phase,
                Phase::Compute { commands, .. }
                    if commands.iter().any(|command| command.specialization.operation == "attention_unpack_head_f16")
            ))
            .count();
        assert!(gather_phases > 1);
        assert!(schedule.allocations.iter().all(|allocation| {
            !matches!(allocation.kind, AllocationKind::ExchangeStaging { .. })
                || allocation.address + allocation.size
                    <= ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES
        }));
        schedule
            .lower_tile_programs(&ipu_exchange::Topology::c600())
            .unwrap();
    }
}
