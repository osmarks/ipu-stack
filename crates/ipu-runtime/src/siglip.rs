use crate::{
    BlockLayout, Result, block_binding_typed, blocked_matrix_f8_f143_by_block, blocked_matrix_f16,
    f143_block_scales, f143_scale,
};
use half::f16;
use ipu_compiler::{
    Allocation, AllocationKind, AppendAffineLayerNormConfig, BlockPlacement, BlockedGemmConfig,
    FlashAttentionConfig, FlashAttentionPlan, GemmDataType, MemoryConstraint, MemoryPlacement,
    MemoryPolicy, RowShardPlacement, RowShardTransitionConfig, Schedule, TensorId,
    allocate_from_occupied, append_add_f16_row_shards_in_place, append_affine_layer_norm_f16,
    append_affine_layer_norm_f16_with_memory_policy, append_bias_f16_c16,
    append_bias_f16_c16_in_arenas, append_blocked_gemm_f16_with_a16_blocks,
    append_blocked_gemm_f16_with_a16_blocks_with_memory_policy,
    append_blocked_gemm_f16_with_a16_input,
    append_blocked_gemm_f16_with_a16_input_with_memory_policy, append_c16_to_a16_blocks_gelu_f16,
    append_c16_to_a16_row_shards, append_flash_attention_from_a16_qkv,
    append_flash_attention_to_a16_row_shards, end_tensor_lifetimes, make_tensors_resident,
    make_tensors_resident_since, occupied_intervals_by_tile, set_f8_weight_block_scales_in_phases,
};
use ipu_models::SiglipWeights;
use ipu_package::{Binding, RegionSlice};
use tracing::{info, info_span};

#[derive(Default)]
pub struct HostTensorSet {
    pub bindings: Vec<Binding>,
    pub bytes: Vec<u8>,
    pub resident_bindings: Vec<Binding>,
    pub resident_bytes: Vec<u8>,
}

impl HostTensorSet {
    pub fn push_input(&mut self, binding: Binding, bytes: Vec<u8>) -> Result<()> {
        push_host_tensor(&mut self.bindings, &mut self.bytes, binding, bytes)
    }

    pub fn push(&mut self, binding: Binding, bytes: Vec<u8>) -> Result<()> {
        push_host_tensor(
            &mut self.resident_bindings,
            &mut self.resident_bytes,
            binding,
            bytes,
        )
    }
}

fn push_host_tensor(
    bindings: &mut Vec<Binding>,
    destination: &mut Vec<u8>,
    binding: Binding,
    bytes: Vec<u8>,
) -> Result<()> {
    let binding_bytes = binding.slices.iter().map(|slice| slice.size).sum::<u64>();
    if binding_bytes != bytes.len() as u64 {
        return Err(format!(
            "host tensor {} has {} binding bytes but {} data bytes",
            binding.name,
            binding_bytes,
            bytes.len()
        )
        .into());
    }
    bindings.push(binding);
    destination.extend(bytes);
    Ok(())
}

pub struct SiglipEncoderLayer {
    pub output: Vec<RowShardPlacement>,
    pub norm2: Vec<RowShardPlacement>,
    pub mlp_gelu: Vec<BlockPlacement>,
    pub attention: FlashAttentionPlan,
}

pub struct SiglipMapHead {
    pub output: Vec<RowShardPlacement>,
    pub attention: FlashAttentionPlan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AttentionKernelVariant {
    pub small_query_rows: u16,
    pub large_query_rows: u16,
    pub small_key_rows: u16,
    pub large_key_rows: u16,
}

impl AttentionKernelVariant {
    pub fn suffix(self) -> String {
        format!(
            "q{}_{}_k{}_{}",
            self.small_query_rows, self.large_query_rows, self.small_key_rows, self.large_key_rows
        )
    }
}

pub fn attention_kernel_variant(plan: &FlashAttentionPlan) -> AttentionKernelVariant {
    let query_rows = plan.tasks.iter().map(|task| task.query_rows);
    let key_rows = plan.key_values.iter().map(|block| block.key_rows);
    AttentionKernelVariant {
        small_query_rows: query_rows.clone().min().unwrap(),
        large_query_rows: query_rows.max().unwrap(),
        small_key_rows: key_rows.clone().min().unwrap(),
        large_key_rows: key_rows.max().unwrap(),
    }
}

pub fn consolidate_attention_kernel_variants(
    schedule: &mut Schedule,
    plans: &[FlashAttentionPlan],
) -> AttentionKernelVariant {
    let variants = plans
        .iter()
        .map(attention_kernel_variant)
        .collect::<Vec<_>>();
    let domain = AttentionKernelVariant {
        small_query_rows: variants
            .iter()
            .map(|variant| variant.small_query_rows)
            .min()
            .unwrap(),
        large_query_rows: variants
            .iter()
            .map(|variant| variant.large_query_rows)
            .max()
            .unwrap(),
        small_key_rows: variants
            .iter()
            .map(|variant| variant.small_key_rows)
            .min()
            .unwrap(),
        large_key_rows: variants
            .iter()
            .map(|variant| variant.large_key_rows)
            .max()
            .unwrap(),
    };
    assert!(variants.iter().all(|variant| {
        [variant.small_query_rows, variant.large_query_rows]
            .into_iter()
            .all(|rows| rows == domain.small_query_rows || rows == domain.large_query_rows)
            && [variant.small_key_rows, variant.large_key_rows]
                .into_iter()
                .all(|rows| rows == domain.small_key_rows || rows == domain.large_key_rows)
    }));
    let local_suffixes = variants
        .iter()
        .map(|variant| (format!("_{}", variant.suffix()), *variant))
        .collect::<Vec<_>>();

    for phase in &mut schedule.phases {
        let ipu_compiler::Phase::Compute { commands, .. } = phase else {
            continue;
        };
        for command in commands {
            let operation = command.specialization.operation.as_ref();
            if !operation.starts_with("attention_") {
                continue;
            }
            let Some((suffix, local)) = local_suffixes
                .iter()
                .find(|(suffix, _)| operation.ends_with(suffix))
            else {
                continue;
            };
            let base = operation.strip_suffix(suffix).unwrap();
            let query_small = row_role(local.small_query_rows, domain.small_query_rows);
            let query_large = row_role(local.large_query_rows, domain.small_query_rows);
            let key_small = row_role(local.small_key_rows, domain.small_key_rows);
            let key_large = row_role(local.large_key_rows, domain.small_key_rows);
            let base = base
                .replace("small_query", "@QUERY_SMALL@")
                .replace("large_query", "@QUERY_LARGE@")
                .replace("small_key", "@KEY_SMALL@")
                .replace("large_key", "@KEY_LARGE@")
                .replace("small_rows", "@QUERY_SMALL_ROWS@")
                .replace("large_rows", "@QUERY_LARGE_ROWS@")
                .replace("@QUERY_SMALL@", &format!("{query_small}_query"))
                .replace("@QUERY_LARGE@", &format!("{query_large}_query"))
                .replace("@KEY_SMALL@", &format!("{key_small}_key"))
                .replace("@KEY_LARGE@", &format!("{key_large}_key"))
                .replace("@QUERY_SMALL_ROWS@", &format!("{query_small}_rows"))
                .replace("@QUERY_LARGE_ROWS@", &format!("{query_large}_rows"));
            command.specialization.operation = format!("{base}_{}", domain.suffix()).into();
        }
    }
    domain
}

fn row_role(rows: u16, small_rows: u16) -> &'static str {
    if rows == small_rows { "small" } else { "large" }
}

fn specialize_attention_phases(
    schedule: &mut Schedule,
    phase_start: usize,
    plan: &FlashAttentionPlan,
) {
    let suffix = attention_kernel_variant(plan).suffix();
    for phase in &mut schedule.phases[phase_start..] {
        let ipu_compiler::Phase::Compute { commands, .. } = phase else {
            continue;
        };
        for command in commands {
            let operation = command.specialization.operation.as_ref();
            if operation.starts_with("attention_qk_")
                || operation.starts_with("attention_pv_")
                || operation.starts_with("attention_softmax_")
                || operation.starts_with("attention_merge_")
                || operation == "attention_f32_to_f16"
            {
                command.specialization.operation = format!("{operation}_{suffix}").into();
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiglipWeightStorage {
    F16,
    F143,
}

impl SiglipWeightStorage {
    fn gemm_data_type(self, values: impl IntoIterator<Item = f32>) -> GemmDataType {
        match self {
            Self::F16 => GemmDataType::F16,
            Self::F143 => GemmDataType::F16F8Weights {
                scale: f143_scale(values),
            },
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn append_host_a16_matrix(
    schedule: &mut Schedule,
    name: &str,
    values: &[f32],
    rows: u16,
    columns: u16,
    row_block_dimension: u16,
    data_base: u32,
    data_limit: u32,
    host: &mut HostTensorSet,
) -> Result<Vec<RowShardPlacement>> {
    if rows == 0
        || columns == 0
        || !columns.is_multiple_of(16)
        || row_block_dimension == 0
        || values.len() != usize::from(rows) * usize::from(columns)
    {
        return Err("host A16 matrix has incompatible dimensions or data".into());
    }
    let mut next_tensor = schedule
        .allocations
        .iter()
        .map(|allocation| allocation.tensor.0)
        .max()
        .unwrap_or(0)
        + 1;
    let row_grid = rows.div_ceil(row_block_dimension);
    let base_rows = rows / row_grid;
    let larger_shards = rows % row_grid;
    let mut shards = Vec::with_capacity(usize::from(row_grid));
    let mut resident_pressure = vec![0u64; usize::from(schedule.tile_count)];
    for allocation in &schedule.allocations {
        if allocation.kind == AllocationKind::Home
            && allocation.live_from == 0
            && allocation.live_until == usize::MAX
        {
            resident_pressure[usize::from(allocation.tile)] += u64::from(allocation.size);
        }
    }
    let mut occupied = occupied_intervals_by_tile(
        &schedule.allocations,
        schedule.tile_count,
        0,
        usize::MAX,
        data_base,
        data_limit,
    );
    let constraint = MemoryConstraint {
        base: data_base,
        limit: data_limit,
        alignment: 8,
        placement: MemoryPlacement::High,
    };
    let mut row_start = 0;
    for shard_index in 0..row_grid {
        let shard_rows = base_rows + u16::from(shard_index < larger_shards);
        let bytes = u32::from(shard_rows) * u32::from(columns) * 2;
        let (tile, address) = (0..schedule.tile_count)
            .filter_map(|tile| {
                let mut candidate = occupied[usize::from(tile)].clone();
                let address = allocate_from_occupied(&mut candidate, bytes, constraint).ok()?;
                Some((resident_pressure[usize::from(tile)], tile, address))
            })
            .min()
            .map(|(_, tile, address)| (tile, address))
            .ok_or_else(|| format!("no tile can hold {bytes} bytes for host matrix {name}"))?;
        let allocated =
            allocate_from_occupied(&mut occupied[usize::from(tile)], bytes, constraint)?;
        debug_assert_eq!(allocated, address);
        resident_pressure[usize::from(tile)] += u64::from(bytes);
        let tensor = TensorId(next_tensor);
        next_tensor += 1;
        shards.push(RowShardPlacement {
            tile,
            row_start,
            rows: shard_rows,
            columns,
            tensor,
            address,
        });
        schedule.allocations.push(Allocation {
            tensor,
            tile,
            address,
            size: bytes,
            live_from: 0,
            live_until: usize::MAX,
            kind: AllocationKind::Home,
        });
        row_start += shard_rows;
    }
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for shard in &shards {
        for panel in 0..columns / 16 {
            for row in shard.row_start..shard.row_start + shard.rows {
                for column in panel * 16..panel * 16 + 16 {
                    let value =
                        values[usize::from(row) * usize::from(columns) + usize::from(column)];
                    bytes.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
                }
            }
        }
    }
    host.push_input(row_shard_binding(name, rows, columns, &shards), bytes)?;
    Ok(shards)
}

#[allow(clippy::too_many_arguments)]
pub fn append_siglip_post_layer_norm(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    model: &SiglipWeights,
    data_base: u32,
    data_limit: u32,
    host: &mut HostTensorSet,
) -> Result<Vec<RowShardPlacement>> {
    let columns = u16::try_from(model.config.hidden_size)?;
    let norm = append_affine_layer_norm_f16(
        schedule,
        input,
        AppendAffineLayerNormConfig {
            data_base,
            data_limit,
            epsilon_bits: model.config.layer_norm_eps.to_bits(),
        },
    )?;
    push_named_layer_norm_affine(
        schedule,
        host,
        "post_layernorm.affine",
        columns,
        &norm.affine,
        &model.tensor_f32("vision_model.post_layernorm.weight")?,
        &model.tensor_f32("vision_model.post_layernorm.bias")?,
        0,
    )?;
    end_tensor_lifetimes(schedule, input.iter().map(|shard| shard.tensor))?;
    Ok(norm.output)
}

#[allow(clippy::too_many_arguments)]
pub fn append_siglip_map_head(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    model: &SiglipWeights,
    rows: u16,
    row_block_dimension: u16,
    tile_count: u16,
    data_base: u32,
    data_limit: u32,
    host: &mut HostTensorSet,
) -> Result<SiglipMapHead> {
    const PROBE_ROWS: u16 = 12;
    let config = &model.config;
    let columns = u16::try_from(config.hidden_size)?;
    let intermediate_columns = u16::try_from(config.intermediate_size.div_ceil(64) * 64)?;
    let probe = model.tensor_f32("vision_model.head.probe")?;
    let mut repeated_probe = Vec::with_capacity(usize::from(PROBE_ROWS) * probe.len());
    for _ in 0..PROBE_ROWS {
        repeated_probe.extend_from_slice(&probe);
    }
    let probe = append_host_a16_matrix(
        schedule,
        "map.probe",
        &repeated_probe,
        PROBE_ROWS,
        columns,
        PROBE_ROWS,
        data_base,
        data_limit,
        host,
    )?;
    let in_weight = model.tensor_f32("vision_model.head.attention.in_proj_weight")?;
    let in_bias = model.tensor_f32("vision_model.head.attention.in_proj_bias")?;
    let query = append_a16_linear_c16(
        schedule,
        &probe,
        PROBE_ROWS,
        columns,
        columns,
        columns,
        columns,
        0,
        &in_weight,
        &in_bias,
        "map.query",
        PROBE_ROWS,
        tile_count,
        data_base,
        data_limit,
        host,
    )?;
    end_tensor_lifetimes(schedule, probe.iter().map(|shard| shard.tensor))?;
    let query = append_c16_to_a16_row_shards(
        schedule,
        &query,
        RowShardTransitionConfig {
            columns,
            data_base,
            data_limit,
        },
    )?;

    let key_value = append_a16_linear_c16(
        schedule,
        input,
        rows,
        columns,
        columns * 2,
        columns,
        columns * 2,
        columns,
        &in_weight,
        &in_bias,
        "map.key_value",
        row_block_dimension,
        tile_count,
        data_base,
        data_limit,
        host,
    )?;
    let key = append_c16_to_a16_row_shards(
        schedule,
        &projection_blocks(&key_value, 0, columns),
        RowShardTransitionConfig {
            columns,
            data_base,
            data_limit,
        },
    )?;
    let value = append_c16_to_a16_row_shards(
        schedule,
        &projection_blocks(&key_value, 1, columns),
        RowShardTransitionConfig {
            columns,
            data_base,
            data_limit,
        },
    )?;
    end_tensor_lifetimes(schedule, key_value.iter().map(|block| block.tensor))?;
    end_tensor_lifetimes(schedule, input.iter().map(|shard| shard.tensor))?;

    let attention_phase_start = schedule.phases.len();
    let attention = append_flash_attention_from_a16_qkv(
        schedule,
        &query,
        &key,
        &value,
        FlashAttentionConfig {
            batch_size: 1,
            query_sequence_length: PROBE_ROWS,
            sequence_length: rows,
            hidden_size: columns,
            attention_heads: u16::try_from(config.num_attention_heads)?,
            query_block_rows: PROBE_ROWS,
            key_block_rows: 0,
            tile_count,
            data_base,
            data_limit,
        },
    )?;
    specialize_attention_phases(schedule, attention_phase_start, &attention);
    end_tensor_lifetimes(
        schedule,
        query
            .iter()
            .chain(&key)
            .chain(&value)
            .map(|shard| shard.tensor),
    )?;
    let attention_shards =
        append_flash_attention_to_a16_row_shards(schedule, &attention, data_base, data_limit)?;
    let projected = append_a16_linear_c16(
        schedule,
        &attention_shards,
        PROBE_ROWS,
        columns,
        columns,
        columns,
        columns,
        0,
        &model.tensor_f32("vision_model.head.attention.out_proj.weight")?,
        &model.tensor_f32("vision_model.head.attention.out_proj.bias")?,
        "map.attention_output",
        PROBE_ROWS,
        tile_count,
        data_base,
        data_limit,
        host,
    )?;
    end_tensor_lifetimes(schedule, attention_shards.iter().map(|shard| shard.tensor))?;
    let residual = append_c16_to_a16_row_shards(
        schedule,
        &projected,
        RowShardTransitionConfig {
            columns,
            data_base,
            data_limit,
        },
    )?;
    end_tensor_lifetimes(schedule, projected.iter().map(|block| block.tensor))?;

    let norm = append_affine_layer_norm_f16(
        schedule,
        &residual,
        AppendAffineLayerNormConfig {
            data_base,
            data_limit,
            epsilon_bits: config.layer_norm_eps.to_bits(),
        },
    )?;
    push_named_layer_norm_affine(
        schedule,
        host,
        "map.layernorm",
        columns,
        &norm.affine,
        &model.tensor_f32("vision_model.head.layernorm.weight")?,
        &model.tensor_f32("vision_model.head.layernorm.bias")?,
        0,
    )?;
    let up = append_a16_linear_c16(
        schedule,
        &norm.output,
        PROBE_ROWS,
        columns,
        intermediate_columns,
        columns,
        u16::try_from(config.intermediate_size)?,
        0,
        &model.tensor_f32("vision_model.head.mlp.fc1.weight")?,
        &model.tensor_f32("vision_model.head.mlp.fc1.bias")?,
        "map.mlp_up",
        PROBE_ROWS,
        tile_count,
        data_base,
        data_limit,
        host,
    )?;
    end_tensor_lifetimes(schedule, norm.output.iter().map(|shard| shard.tensor))?;
    let gelu = append_c16_to_a16_blocks_gelu_f16(schedule, &up, data_base, data_limit)?;
    end_tensor_lifetimes(schedule, up.iter().map(|block| block.tensor))?;
    let down = append_blocked_gemm_f16_with_a16_blocks(
        schedule,
        &gelu,
        gemm_config(
            PROBE_ROWS,
            intermediate_columns,
            columns,
            PROBE_ROWS,
            tile_count,
            data_base,
            data_limit,
            GemmDataType::F16,
            true,
        ),
    )?;
    let down_weight = model.tensor_f32("vision_model.head.mlp.fc2.weight")?;
    host.push(
        block_binding_typed(
            "map.mlp_down.weight",
            intermediate_columns,
            columns,
            &down.right,
            "f16",
            2,
        ),
        blocked_matrix_f16(&down.right, BlockLayout::AmpB16x16, |row, column| {
            if usize::from(row) < config.intermediate_size {
                down_weight[usize::from(column) * config.intermediate_size + usize::from(row)]
            } else {
                0.0
            }
        }),
    )?;
    make_tensors_resident(schedule, down.right.iter().map(|block| block.tensor))?;
    end_tensor_lifetimes(schedule, gelu.iter().map(|block| block.tensor))?;
    let down_bias = append_bias_f16_c16(schedule, &down.output, data_base, data_limit)?;
    let bias = model.tensor_f32("vision_model.head.mlp.fc2.bias")?;
    host.push(
        block_binding_typed(
            "map.mlp_down.bias",
            u16::try_from(down_bias.len())?,
            64,
            &down_bias,
            "f16",
            2,
        ),
        blocked_matrix_f16(&down_bias, BlockLayout::AmpC16F16, |_row, column| {
            bias[usize::from(column)]
        }),
    )?;
    make_tensors_resident(schedule, down_bias.iter().map(|block| block.tensor))?;
    let output = append_c16_to_a16_row_shards(
        schedule,
        &down.output,
        RowShardTransitionConfig {
            columns,
            data_base,
            data_limit,
        },
    )?;
    end_tensor_lifetimes(schedule, down.output.iter().map(|block| block.tensor))?;
    let output = append_add_f16_row_shards_in_place(schedule, &output, &residual)?;
    end_tensor_lifetimes(schedule, residual.iter().map(|shard| shard.tensor))?;
    Ok(SiglipMapHead { output, attention })
}

#[allow(clippy::too_many_arguments)]
pub fn append_siglip_encoder_layer(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    model: &SiglipWeights,
    layer: usize,
    rows: u16,
    columns: u16,
    row_block_dimension: u16,
    tile_count: u16,
    memory: &MemoryPolicy,
    weight_storage: SiglipWeightStorage,
    retain_profile_metadata: bool,
    retain_diagnostics: bool,
    host: &mut HostTensorSet,
) -> Result<SiglipEncoderLayer> {
    memory.validate()?;
    // The standalone kernel planners use this ordinary-memory window before
    // policy-aware composition relocates movable operands into their arenas.
    let data_base = ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT;
    let data_limit = ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE;
    let span = info_span!("siglip_encoder_layer", layer);
    let _guard = span.enter();
    let config = &model.config;
    let prefix = format!("encoder_layer_{layer:02}");
    info!(stage = "norm1_qkv", "planning SigLIP encoder stage");
    let norm_allocation_start = schedule.allocations.len();
    let norm = append_affine_layer_norm_f16_with_memory_policy(
        schedule,
        input,
        AppendAffineLayerNormConfig {
            data_base,
            data_limit,
            epsilon_bits: config.layer_norm_eps.to_bits(),
        },
        memory,
    )?;
    push_layer_norm_affine(
        schedule,
        host,
        model,
        layer,
        "layer_norm1",
        columns,
        &norm.affine,
        norm_allocation_start,
    )?;

    let projection_names = ["q_proj", "k_proj", "v_proj"];
    let projection_weights = projection_names.map(|projection| {
        model
            .tensor_f32(
                &model
                    .layer_name(layer, &format!("self_attn.{projection}.weight"))
                    .unwrap(),
            )
            .unwrap()
    });
    let qkv_data_type = weight_storage.gemm_data_type(
        projection_weights
            .iter()
            .flat_map(|weights| weights.iter().copied()),
    );
    let qkv_phase_start = schedule.phases.len();
    let qkv_allocation_start = schedule.allocations.len();
    let qkv = append_blocked_gemm_f16_with_a16_input_with_memory_policy(
        schedule,
        &norm.output,
        gemm_config(
            rows,
            columns,
            columns * 3,
            row_block_dimension,
            tile_count,
            data_base,
            data_limit,
            qkv_data_type,
            retain_profile_metadata,
        ),
        memory,
    )?;
    push_gemm_weight(
        schedule,
        host,
        &format!("{prefix}.qkv.weight"),
        columns,
        columns * 3,
        &qkv.right,
        qkv_data_type,
        qkv_phase_start,
        |row, column| {
            let projection = usize::from(column / columns);
            let output = usize::from(column % columns);
            projection_weights[projection][output * usize::from(columns) + usize::from(row)]
        },
    )?;
    make_tensors_resident_since(
        schedule,
        qkv_allocation_start,
        qkv.right.iter().map(|block| block.tensor),
    )?;
    end_tensor_lifetimes(schedule, norm.output.iter().map(|shard| shard.tensor))?;

    let projection_biases = projection_names.map(|projection| {
        model
            .tensor_f32(
                &model
                    .layer_name(layer, &format!("self_attn.{projection}.bias"))
                    .unwrap(),
            )
            .unwrap()
    });
    let qkv_bias_allocation_start = schedule.allocations.len();
    let qkv_bias = append_bias_f16_c16_in_arenas(schedule, &qkv.output, &memory.resident)?;
    host.push(
        block_binding_typed(
            &format!("{prefix}.qkv.bias"),
            u16::try_from(qkv_bias.len())?,
            64,
            &qkv_bias,
            "f16",
            2,
        ),
        blocked_matrix_f16(&qkv_bias, BlockLayout::AmpC16F16, |_row, column| {
            let projection = usize::from(column / columns);
            projection_biases[projection][usize::from(column % columns)]
        }),
    )?;
    make_tensors_resident_since(
        schedule,
        qkv_bias_allocation_start,
        qkv_bias.iter().map(|block| block.tensor),
    )?;
    let qkv_shards = (0..3)
        .map(|projection| {
            append_c16_to_a16_row_shards(
                schedule,
                &projection_blocks(&qkv.output, projection, columns),
                RowShardTransitionConfig {
                    columns,
                    data_base,
                    data_limit,
                },
            )
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    end_tensor_lifetimes(schedule, qkv.output.iter().map(|block| block.tensor))?;

    info!(stage = "attention", "planning SigLIP encoder stage");
    let attention_phase_start = schedule.phases.len();
    let attention = append_flash_attention_from_a16_qkv(
        schedule,
        &qkv_shards[0],
        &qkv_shards[1],
        &qkv_shards[2],
        FlashAttentionConfig {
            batch_size: 1,
            query_sequence_length: 0,
            sequence_length: rows,
            hidden_size: columns,
            attention_heads: u16::try_from(config.num_attention_heads)?,
            query_block_rows: 0,
            key_block_rows: 0,
            tile_count,
            data_base,
            data_limit,
        },
    )?;
    specialize_attention_phases(schedule, attention_phase_start, &attention);
    end_tensor_lifetimes(
        schedule,
        qkv_shards
            .iter()
            .flat_map(|shards| shards.iter().map(|shard| shard.tensor)),
    )?;
    let attention_shards =
        append_flash_attention_to_a16_row_shards(schedule, &attention, data_base, data_limit)?;
    info!(stage = "attention_output", "planning SigLIP encoder stage");
    let output_weight = model.tensor_f32(&model.layer_name(layer, "self_attn.out_proj.weight")?)?;
    let output_data_type = weight_storage.gemm_data_type(output_weight.iter().copied());
    let output_projection_phase_start = schedule.phases.len();
    let output_projection_allocation_start = schedule.allocations.len();
    let output_projection = append_blocked_gemm_f16_with_a16_input_with_memory_policy(
        schedule,
        &attention_shards,
        gemm_config(
            rows,
            columns,
            columns,
            row_block_dimension,
            tile_count,
            data_base,
            data_limit,
            output_data_type,
            retain_profile_metadata,
        ),
        memory,
    )?;
    push_gemm_weight(
        schedule,
        host,
        &format!("{prefix}.attention_output.weight"),
        columns,
        columns,
        &output_projection.right,
        output_data_type,
        output_projection_phase_start,
        |row, column| output_weight[usize::from(column) * usize::from(columns) + usize::from(row)],
    )?;
    make_tensors_resident_since(
        schedule,
        output_projection_allocation_start,
        output_projection.right.iter().map(|block| block.tensor),
    )?;
    end_tensor_lifetimes(schedule, attention_shards.iter().map(|shard| shard.tensor))?;
    let output_bias = model.tensor_f32(&model.layer_name(layer, "self_attn.out_proj.bias")?)?;
    let output_adjustment_allocation_start = schedule.allocations.len();
    let output_adjustment =
        append_bias_f16_c16_in_arenas(schedule, &output_projection.output, &memory.resident)?;
    host.push(
        block_binding_typed(
            &format!("{prefix}.attention_output.bias"),
            u16::try_from(output_adjustment.len())?,
            64,
            &output_adjustment,
            "f16",
            2,
        ),
        blocked_matrix_f16(
            &output_adjustment,
            BlockLayout::AmpC16F16,
            |_row, column| output_bias[usize::from(column)],
        ),
    )?;
    make_tensors_resident_since(
        schedule,
        output_adjustment_allocation_start,
        output_adjustment.iter().map(|block| block.tensor),
    )?;
    let projected_shards = append_c16_to_a16_row_shards(
        schedule,
        &output_projection.output,
        RowShardTransitionConfig {
            columns,
            data_base,
            data_limit,
        },
    )?;
    end_tensor_lifetimes(
        schedule,
        output_projection.output.iter().map(|block| block.tensor),
    )?;
    let attention_residual =
        append_add_f16_row_shards_in_place(schedule, &projected_shards, input)?;
    end_tensor_lifetimes(schedule, input.iter().map(|shard| shard.tensor))?;

    info!(stage = "norm2_mlp", "planning SigLIP encoder stage");
    let norm2_allocation_start = schedule.allocations.len();
    let norm2 = append_affine_layer_norm_f16_with_memory_policy(
        schedule,
        &attention_residual,
        AppendAffineLayerNormConfig {
            data_base,
            data_limit,
            epsilon_bits: config.layer_norm_eps.to_bits(),
        },
        memory,
    )?;
    push_layer_norm_affine(
        schedule,
        host,
        model,
        layer,
        "layer_norm2",
        columns,
        &norm2.affine,
        norm2_allocation_start,
    )?;
    let intermediate_columns = u16::try_from(config.intermediate_size.div_ceil(64) * 64)?;
    let mlp_up_weight = model.tensor_f32(&model.layer_name(layer, "mlp.fc1.weight")?)?;
    let mlp_up_data_type = weight_storage.gemm_data_type(mlp_up_weight.iter().copied());
    let mlp_up_phase_start = schedule.phases.len();
    let mlp_up_allocation_start = schedule.allocations.len();
    let mlp_up = append_blocked_gemm_f16_with_a16_input_with_memory_policy(
        schedule,
        &norm2.output,
        gemm_config(
            rows,
            columns,
            intermediate_columns,
            row_block_dimension,
            tile_count,
            data_base,
            data_limit,
            mlp_up_data_type,
            retain_profile_metadata,
        ),
        memory,
    )?;
    push_gemm_weight(
        schedule,
        host,
        &format!("{prefix}.mlp_up.weight"),
        columns,
        intermediate_columns,
        &mlp_up.right,
        mlp_up_data_type,
        mlp_up_phase_start,
        |row, column| {
            if usize::from(column) < config.intermediate_size {
                mlp_up_weight[usize::from(column) * usize::from(columns) + usize::from(row)]
            } else {
                0.0
            }
        },
    )?;
    make_tensors_resident_since(
        schedule,
        mlp_up_allocation_start,
        mlp_up.right.iter().map(|block| block.tensor),
    )?;
    if !retain_diagnostics {
        end_tensor_lifetimes(schedule, norm2.output.iter().map(|shard| shard.tensor))?;
    }
    let mlp_up_bias = model.tensor_f32(&model.layer_name(layer, "mlp.fc1.bias")?)?;
    let mlp_up_adjustment_allocation_start = schedule.allocations.len();
    let mlp_up_adjustment =
        append_bias_f16_c16_in_arenas(schedule, &mlp_up.output, &memory.resident)?;
    host.push(
        block_binding_typed(
            &format!("{prefix}.mlp_up.bias"),
            u16::try_from(mlp_up_adjustment.len())?,
            64,
            &mlp_up_adjustment,
            "f16",
            2,
        ),
        blocked_matrix_f16(
            &mlp_up_adjustment,
            BlockLayout::AmpC16F16,
            |_row, column| mlp_up_bias.get(usize::from(column)).copied().unwrap_or(0.0),
        ),
    )?;
    make_tensors_resident_since(
        schedule,
        mlp_up_adjustment_allocation_start,
        mlp_up_adjustment.iter().map(|block| block.tensor),
    )?;
    let mlp_gelu =
        append_c16_to_a16_blocks_gelu_f16(schedule, &mlp_up.output, data_base, data_limit)?;
    end_tensor_lifetimes(schedule, mlp_up.output.iter().map(|block| block.tensor))?;
    let mlp_down_weight = model.tensor_f32(&model.layer_name(layer, "mlp.fc2.weight")?)?;
    let mlp_down_data_type = weight_storage.gemm_data_type(mlp_down_weight.iter().copied());
    let mlp_down_phase_start = schedule.phases.len();
    let mlp_down_allocation_start = schedule.allocations.len();
    let mlp_down = append_blocked_gemm_f16_with_a16_blocks_with_memory_policy(
        schedule,
        &mlp_gelu,
        gemm_config(
            rows,
            intermediate_columns,
            columns,
            row_block_dimension,
            tile_count,
            data_base,
            data_limit,
            mlp_down_data_type,
            retain_profile_metadata,
        ),
        memory,
    )?;
    push_gemm_weight(
        schedule,
        host,
        &format!("{prefix}.mlp_down.weight"),
        intermediate_columns,
        columns,
        &mlp_down.right,
        mlp_down_data_type,
        mlp_down_phase_start,
        |row, column| {
            if usize::from(row) < config.intermediate_size {
                mlp_down_weight[usize::from(column) * config.intermediate_size + usize::from(row)]
            } else {
                0.0
            }
        },
    )?;
    make_tensors_resident_since(
        schedule,
        mlp_down_allocation_start,
        mlp_down.right.iter().map(|block| block.tensor),
    )?;
    if !retain_diagnostics {
        end_tensor_lifetimes(schedule, mlp_gelu.iter().map(|block| block.tensor))?;
    }
    let mlp_down_bias = model.tensor_f32(&model.layer_name(layer, "mlp.fc2.bias")?)?;
    let mlp_down_adjustment_allocation_start = schedule.allocations.len();
    let mlp_down_adjustment =
        append_bias_f16_c16_in_arenas(schedule, &mlp_down.output, &memory.resident)?;
    host.push(
        block_binding_typed(
            &format!("{prefix}.mlp_down.bias"),
            u16::try_from(mlp_down_adjustment.len())?,
            64,
            &mlp_down_adjustment,
            "f16",
            2,
        ),
        blocked_matrix_f16(
            &mlp_down_adjustment,
            BlockLayout::AmpC16F16,
            |_row, column| mlp_down_bias[usize::from(column)],
        ),
    )?;
    make_tensors_resident_since(
        schedule,
        mlp_down_adjustment_allocation_start,
        mlp_down_adjustment.iter().map(|block| block.tensor),
    )?;
    let output = append_c16_to_a16_row_shards(
        schedule,
        &mlp_down.output,
        RowShardTransitionConfig {
            columns,
            data_base,
            data_limit,
        },
    )?;
    end_tensor_lifetimes(schedule, mlp_down.output.iter().map(|block| block.tensor))?;
    let output = append_add_f16_row_shards_in_place(schedule, &output, &attention_residual)?;
    end_tensor_lifetimes(
        schedule,
        attention_residual.iter().map(|shard| shard.tensor),
    )?;
    info!(stage = "complete", "planned SigLIP encoder layer");
    Ok(SiglipEncoderLayer {
        output,
        norm2: norm2.output,
        mlp_gelu,
        attention,
    })
}

fn gemm_config(
    rows: u16,
    inner_dimension: u16,
    columns: u16,
    row_block_dimension: u16,
    tile_count: u16,
    data_base: u32,
    data_limit: u32,
    data_type: GemmDataType,
    retain_profile_metadata: bool,
) -> BlockedGemmConfig {
    BlockedGemmConfig {
        rows,
        inner_dimension,
        columns,
        block_dimension: 64,
        inner_block_dimension: 64,
        row_block_dimension,
        tile_count,
        data_base,
        data_limit,
        data_type,
        retain_profile_metadata,
    }
}

fn push_gemm_weight(
    schedule: &mut Schedule,
    host: &mut HostTensorSet,
    name: &str,
    rows: u16,
    columns: u16,
    placements: &[BlockPlacement],
    data_type: GemmDataType,
    phase_start: usize,
    value: impl Fn(u16, u16) -> f32,
) -> Result<()> {
    let (dtype, bytes) = match data_type {
        GemmDataType::F16 => (
            "f16",
            blocked_matrix_f16(placements, BlockLayout::AmpB16x16, value),
        ),
        GemmDataType::F16F8Weights { .. } => {
            let scales = f143_block_scales(placements, &value);
            let phase_end = schedule.phases.len();
            set_f8_weight_block_scales_in_phases(
                schedule,
                phase_start..phase_end,
                placements,
                &scales,
            )?;
            (
                "f8-f143-block-scaled",
                blocked_matrix_f8_f143_by_block(placements, BlockLayout::AmpB16x16, &scales, value),
            )
        }
        GemmDataType::F8F143 { .. } | GemmDataType::F32 => {
            return Err("SigLIP weight serialization does not support this GEMM data type".into());
        }
    };
    host.push(
        block_binding_typed(
            name,
            rows,
            columns,
            placements,
            dtype,
            u64::from(data_type.weight_element_bytes()),
        ),
        bytes,
    )
}

#[allow(clippy::too_many_arguments)]
fn append_a16_linear_c16(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    rows: u16,
    padded_inner: u16,
    padded_output: u16,
    actual_inner: u16,
    actual_output: u16,
    output_offset: u16,
    weight: &[f32],
    bias: &[f32],
    name: &str,
    row_block_dimension: u16,
    tile_count: u16,
    data_base: u32,
    data_limit: u32,
    host: &mut HostTensorSet,
) -> Result<Vec<BlockPlacement>> {
    if weight.len() < usize::from(output_offset + actual_output) * usize::from(actual_inner)
        || bias.len() < usize::from(output_offset + actual_output)
    {
        return Err(format!("{name} weight or bias is smaller than its declared slice").into());
    }
    let gemm = append_blocked_gemm_f16_with_a16_input(
        schedule,
        input,
        gemm_config(
            rows,
            padded_inner,
            padded_output,
            row_block_dimension,
            tile_count,
            data_base,
            data_limit,
            GemmDataType::F16,
            true,
        ),
    )?;
    host.push(
        block_binding_typed(
            &format!("{name}.weight"),
            padded_inner,
            padded_output,
            &gemm.right,
            "f16",
            2,
        ),
        blocked_matrix_f16(&gemm.right, BlockLayout::AmpB16x16, |row, column| {
            if row < actual_inner && column < actual_output {
                weight[usize::from(output_offset + column) * usize::from(actual_inner)
                    + usize::from(row)]
            } else {
                0.0
            }
        }),
    )?;
    make_tensors_resident(schedule, gemm.right.iter().map(|block| block.tensor))?;
    let adjustment = append_bias_f16_c16(schedule, &gemm.output, data_base, data_limit)?;
    host.push(
        block_binding_typed(
            &format!("{name}.bias"),
            u16::try_from(adjustment.len())?,
            64,
            &adjustment,
            "f16",
            2,
        ),
        blocked_matrix_f16(&adjustment, BlockLayout::AmpC16F16, |_row, column| {
            if column < actual_output {
                bias[usize::from(output_offset + column)]
            } else {
                0.0
            }
        }),
    )?;
    make_tensors_resident(schedule, adjustment.iter().map(|block| block.tensor))?;
    Ok(gemm.output)
}

fn push_named_layer_norm_affine(
    schedule: &mut Schedule,
    host: &mut HostTensorSet,
    name: &str,
    columns: u16,
    placements: &[RowShardPlacement],
    weight: &[f32],
    bias: &[f32],
    allocation_start: usize,
) -> Result<()> {
    if weight.len() != usize::from(columns) || bias.len() != usize::from(columns) {
        return Err(format!("{name} affine dimensions do not match {columns} columns").into());
    }
    let mut bytes = Vec::with_capacity(placements.len() * usize::from(columns) * 2);
    for placement in placements {
        for row in placement.row_start..placement.row_start + placement.rows {
            let values = match row {
                0 => weight,
                1 => bias,
                _ => {
                    return Err(format!("layer norm affine row {row} is outside scale/bias").into());
                }
            };
            bytes.extend(
                values
                    .iter()
                    .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes()),
            );
        }
    }
    host.push(row_shard_binding(name, 2, columns, placements), bytes)?;
    make_tensors_resident_since(
        schedule,
        allocation_start,
        placements.iter().map(|placement| placement.tensor),
    )?;
    Ok(())
}

fn push_layer_norm_affine(
    schedule: &mut Schedule,
    host: &mut HostTensorSet,
    model: &SiglipWeights,
    layer: usize,
    norm: &str,
    columns: u16,
    placements: &[RowShardPlacement],
    allocation_start: usize,
) -> Result<()> {
    let weight = model.tensor_f32(&model.layer_name(layer, &format!("{norm}.weight"))?)?;
    let bias = model.tensor_f32(&model.layer_name(layer, &format!("{norm}.bias"))?)?;
    push_named_layer_norm_affine(
        schedule,
        host,
        &format!("encoder_layer_{layer:02}.{norm}.affine"),
        columns,
        placements,
        &weight,
        &bias,
        allocation_start,
    )
}

fn row_shard_binding(name: &str, rows: u16, columns: u16, shards: &[RowShardPlacement]) -> Binding {
    let topology = ipu_exchange::Topology::c600();
    let mut file_offset = 0u64;
    let slices = shards
        .iter()
        .map(|shard| {
            let size = u64::from(shard.rows) * u64::from(shard.columns) * 2;
            let slice = RegionSlice {
                tile: u32::from(topology.physical(shard.tile).unwrap()),
                tile_address: shard.address,
                file_offset,
                size,
            };
            file_offset += size;
            slice
        })
        .collect();
    Binding {
        name: name.into(),
        dtype: "f16".into(),
        shape: vec![u32::from(rows), u32::from(columns)],
        slices,
    }
}

fn projection_blocks(
    output: &[BlockPlacement],
    projection: u16,
    hidden: u16,
) -> Vec<BlockPlacement> {
    let first_column = projection * hidden;
    output
        .iter()
        .filter(|block| {
            block.column_start >= first_column && block.column_start < first_column + hidden
        })
        .map(|block| BlockPlacement {
            block_column: block.block_column - first_column / block.columns,
            column_start: block.column_start - first_column,
            ..*block
        })
        .collect()
}
