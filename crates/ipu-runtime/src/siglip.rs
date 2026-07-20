use crate::{BlockLayout, Result, block_binding_typed, blocked_matrix_f16};
use half::f16;
use ipu_compiler::{
    Allocation, AllocationKind, AppendAffineLayerNormConfig, BlockPlacement, BlockedGemmConfig,
    FlashAttentionConfig, FlashAttentionPlan, GemmDataType, MemoryConstraint, MemoryPlacement,
    RowShardPlacement, RowShardTransitionConfig, Schedule, TensorId,
    append_add_f16_row_shards_in_place, append_affine_layer_norm_f16, append_bias_f16_c16,
    append_blocked_gemm_f16_with_a16_blocks, append_blocked_gemm_f16_with_a16_input,
    append_c16_to_a16_blocks_gelu_f16, append_c16_to_a16_row_shards,
    append_flash_attention_from_a16_qkv, append_flash_attention_to_a16_row_shards,
    end_tensor_lifetimes, find_free_region, make_tensors_resident,
};
use ipu_models::SiglipWeights;
use ipu_package::{Binding, RegionSlice};
use tracing::{info, info_span};

#[derive(Default)]
pub struct HostTensorSet {
    pub bindings: Vec<Binding>,
    pub bytes: Vec<u8>,
}

impl HostTensorSet {
    pub fn push(&mut self, binding: Binding, bytes: Vec<u8>) -> Result<()> {
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
        self.bindings.push(binding);
        self.bytes.extend(bytes);
        Ok(())
    }
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
    let mut row_start = 0;
    for shard_index in 0..row_grid {
        let shard_rows = base_rows + u16::from(shard_index < larger_shards);
        let bytes = u32::from(shard_rows) * u32::from(columns) * 2;
        let (tile, address) = (0..schedule.tile_count)
            .filter_map(|tile| {
                let address = find_free_region(
                    &schedule.allocations,
                    tile,
                    bytes,
                    0,
                    usize::MAX,
                    MemoryConstraint {
                        base: data_base,
                        limit: data_limit,
                        alignment: 8,
                        placement: MemoryPlacement::High,
                    },
                )
                .ok()?;
                let pressure = schedule
                    .allocations
                    .iter()
                    .filter(|allocation| {
                        allocation.tile == tile
                            && allocation.kind == AllocationKind::Home
                            && allocation.live_from == 0
                            && allocation.live_until == usize::MAX
                    })
                    .map(|allocation| u64::from(allocation.size))
                    .sum::<u64>();
                Some((pressure, tile, address))
            })
            .min()
            .map(|(_, tile, address)| (tile, address))
            .ok_or_else(|| format!("no tile can hold {bytes} bytes for host matrix {name}"))?;
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
    host.push(row_shard_binding(name, rows, columns, &shards), bytes)?;
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
    data_base: u32,
    data_limit: u32,
    retain_diagnostics: bool,
    host: &mut HostTensorSet,
) -> Result<SiglipEncoderLayer> {
    let span = info_span!("siglip_encoder_layer", layer);
    let _guard = span.enter();
    let config = &model.config;
    let prefix = format!("encoder_layer_{layer:02}");
    info!(stage = "norm1_qkv", "planning SigLIP encoder stage");
    let norm = append_affine_layer_norm_f16(
        schedule,
        input,
        AppendAffineLayerNormConfig {
            data_base,
            data_limit,
            epsilon_bits: config.layer_norm_eps.to_bits(),
        },
    )?;
    push_layer_norm_affine(
        schedule,
        host,
        model,
        layer,
        "layer_norm1",
        columns,
        &norm.affine,
    )?;

    let qkv = append_blocked_gemm_f16_with_a16_input(
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
        ),
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
    host.push(
        block_binding_typed(
            &format!("{prefix}.qkv.weight"),
            columns,
            columns * 3,
            &qkv.right,
            "f16",
            2,
        ),
        blocked_matrix_f16(&qkv.right, BlockLayout::AmpB16x16, |row, column| {
            let projection = usize::from(column / columns);
            let output = usize::from(column % columns);
            projection_weights[projection][output * usize::from(columns) + usize::from(row)]
        }),
    )?;
    make_tensors_resident(schedule, qkv.right.iter().map(|block| block.tensor))?;
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
    let qkv_bias = append_bias_f16_c16(schedule, &qkv.output, data_base, data_limit)?;
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
    make_tensors_resident(schedule, qkv_bias.iter().map(|block| block.tensor))?;
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
    end_tensor_lifetimes(
        schedule,
        qkv_shards
            .iter()
            .flat_map(|shards| shards.iter().map(|shard| shard.tensor)),
    )?;
    let attention_shards =
        append_flash_attention_to_a16_row_shards(schedule, &attention, data_base, data_limit)?;
    info!(stage = "attention_output", "planning SigLIP encoder stage");
    let output_projection = append_blocked_gemm_f16_with_a16_input(
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
        ),
    )?;
    let output_weight = model.tensor_f32(&model.layer_name(layer, "self_attn.out_proj.weight")?)?;
    host.push(
        block_binding_typed(
            &format!("{prefix}.attention_output.weight"),
            columns,
            columns,
            &output_projection.right,
            "f16",
            2,
        ),
        blocked_matrix_f16(
            &output_projection.right,
            BlockLayout::AmpB16x16,
            |row, column| {
                output_weight[usize::from(column) * usize::from(columns) + usize::from(row)]
            },
        ),
    )?;
    make_tensors_resident(
        schedule,
        output_projection.right.iter().map(|block| block.tensor),
    )?;
    end_tensor_lifetimes(schedule, attention_shards.iter().map(|shard| shard.tensor))?;
    let output_bias = model.tensor_f32(&model.layer_name(layer, "self_attn.out_proj.bias")?)?;
    let output_adjustment =
        append_bias_f16_c16(schedule, &output_projection.output, data_base, data_limit)?;
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
    make_tensors_resident(schedule, output_adjustment.iter().map(|block| block.tensor))?;
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
    let norm2 = append_affine_layer_norm_f16(
        schedule,
        &attention_residual,
        AppendAffineLayerNormConfig {
            data_base,
            data_limit,
            epsilon_bits: config.layer_norm_eps.to_bits(),
        },
    )?;
    push_layer_norm_affine(
        schedule,
        host,
        model,
        layer,
        "layer_norm2",
        columns,
        &norm2.affine,
    )?;
    let intermediate_columns = u16::try_from(config.intermediate_size.div_ceil(64) * 64)?;
    let mlp_up = append_blocked_gemm_f16_with_a16_input(
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
        ),
    )?;
    let mlp_up_weight = model.tensor_f32(&model.layer_name(layer, "mlp.fc1.weight")?)?;
    host.push(
        block_binding_typed(
            &format!("{prefix}.mlp_up.weight"),
            columns,
            intermediate_columns,
            &mlp_up.right,
            "f16",
            2,
        ),
        blocked_matrix_f16(&mlp_up.right, BlockLayout::AmpB16x16, |row, column| {
            if usize::from(column) < config.intermediate_size {
                mlp_up_weight[usize::from(column) * usize::from(columns) + usize::from(row)]
            } else {
                0.0
            }
        }),
    )?;
    make_tensors_resident(schedule, mlp_up.right.iter().map(|block| block.tensor))?;
    if !retain_diagnostics {
        end_tensor_lifetimes(schedule, norm2.output.iter().map(|shard| shard.tensor))?;
    }
    let mlp_up_bias = model.tensor_f32(&model.layer_name(layer, "mlp.fc1.bias")?)?;
    let mlp_up_adjustment = append_bias_f16_c16(schedule, &mlp_up.output, data_base, data_limit)?;
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
    make_tensors_resident(schedule, mlp_up_adjustment.iter().map(|block| block.tensor))?;
    let mlp_gelu =
        append_c16_to_a16_blocks_gelu_f16(schedule, &mlp_up.output, data_base, data_limit)?;
    end_tensor_lifetimes(schedule, mlp_up.output.iter().map(|block| block.tensor))?;
    let mlp_down = append_blocked_gemm_f16_with_a16_blocks(
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
        ),
    )?;
    let mlp_down_weight = model.tensor_f32(&model.layer_name(layer, "mlp.fc2.weight")?)?;
    host.push(
        block_binding_typed(
            &format!("{prefix}.mlp_down.weight"),
            intermediate_columns,
            columns,
            &mlp_down.right,
            "f16",
            2,
        ),
        blocked_matrix_f16(&mlp_down.right, BlockLayout::AmpB16x16, |row, column| {
            if usize::from(row) < config.intermediate_size {
                mlp_down_weight[usize::from(column) * config.intermediate_size + usize::from(row)]
            } else {
                0.0
            }
        }),
    )?;
    make_tensors_resident(schedule, mlp_down.right.iter().map(|block| block.tensor))?;
    if !retain_diagnostics {
        end_tensor_lifetimes(schedule, mlp_gelu.iter().map(|block| block.tensor))?;
    }
    let mlp_down_bias = model.tensor_f32(&model.layer_name(layer, "mlp.fc2.bias")?)?;
    let mlp_down_adjustment =
        append_bias_f16_c16(schedule, &mlp_down.output, data_base, data_limit)?;
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
    make_tensors_resident(
        schedule,
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
        data_type: GemmDataType::F16,
    }
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
    make_tensors_resident(
        schedule,
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
