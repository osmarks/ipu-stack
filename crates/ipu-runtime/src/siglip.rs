use crate::{BlockLayout, Result, block_binding_typed, blocked_matrix_f16};
use half::f16;
use ipu_compiler::{
    AppendAffineLayerNormConfig, BlockPlacement, BlockedGemmConfig, FlashAttentionConfig,
    FlashAttentionPlan, GemmDataType, RowShardPlacement, RowShardTransitionConfig, Schedule,
    append_add_f16_row_shards_in_place, append_affine_layer_norm_f16, append_bias_f16_c16,
    append_blocked_gemm_f16_with_a16_blocks, append_blocked_gemm_f16_with_a16_input,
    append_c16_to_a16_blocks_gelu_f16, append_c16_to_a16_row_shards,
    append_flash_attention_from_a16_qkv, append_flash_attention_to_a16_row_shards,
    end_tensor_lifetimes,
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
    push_layer_norm_affine(host, model, layer, "layer_norm1", columns, &norm.affine)?;

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
    end_tensor_lifetimes(schedule, norm.output.iter().map(|shard| shard.tensor))?;
    end_tensor_lifetimes(schedule, qkv.right.iter().map(|block| block.tensor))?;

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
    end_tensor_lifetimes(schedule, attention_shards.iter().map(|shard| shard.tensor))?;
    end_tensor_lifetimes(
        schedule,
        output_projection.right.iter().map(|block| block.tensor),
    )?;
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
    push_layer_norm_affine(host, model, layer, "layer_norm2", columns, &norm2.affine)?;
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
    if !retain_diagnostics {
        end_tensor_lifetimes(schedule, norm2.output.iter().map(|shard| shard.tensor))?;
    }
    end_tensor_lifetimes(schedule, mlp_up.right.iter().map(|block| block.tensor))?;
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
    if !retain_diagnostics {
        end_tensor_lifetimes(schedule, mlp_gelu.iter().map(|block| block.tensor))?;
    }
    end_tensor_lifetimes(schedule, mlp_down.right.iter().map(|block| block.tensor))?;
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

fn push_layer_norm_affine(
    host: &mut HostTensorSet,
    model: &SiglipWeights,
    layer: usize,
    norm: &str,
    columns: u16,
    placements: &[RowShardPlacement],
) -> Result<()> {
    let weight = model.tensor_f32(&model.layer_name(layer, &format!("{norm}.weight"))?)?;
    let bias = model.tensor_f32(&model.layer_name(layer, &format!("{norm}.bias"))?)?;
    let mut bytes = Vec::with_capacity(placements.len() * usize::from(columns) * 4);
    for placement in placements {
        for row in placement.row_start..placement.row_start + placement.rows {
            let values = match row {
                0 => weight.as_slice(),
                1 => bias.as_slice(),
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
    host.push(
        row_shard_binding(
            &format!("encoder_layer_{layer:02}.{norm}.affine"),
            2,
            columns,
            placements,
        ),
        bytes,
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
