use half::f16;
use ipu_compiler::{
    Allocation, AllocationKind, AppendAffineLayerNormConfig, BlockPlacement, BlockedGemmConfig,
    FlashAttentionConfig, FlashAttentionPlan, GemmDataType, KernelCommand, OpId, Phase,
    RowShardPlacement, RowShardTransitionConfig, SpecializationKey, TensorId,
    append_affine_layer_norm_f16, append_blocked_gemm_f16_with_a16_input,
    append_c16_to_a16_row_shards, append_flash_attention_from_a16_qkv,
    append_flash_attention_to_a16_row_shards, choose_gemm_row_block_for, plan_blocked_gemm,
};
use ipu_elf::Toolchain;
use ipu_models::{SiglipWeights, TensorArchive};
use ipu_package::{Binding, RegionSlice};
use ipu_runtime::{
    BlockLayout, ExecutableGraph, HostRunOptions, block_binding_typed, blocked_matrix_f16,
    package_graph, run_host_with_options,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use tracing::info;

const TILE_COUNT: u16 = 1472;
const BLOCK_DIMENSION: u16 = 64;
const INNER_BLOCK_DIMENSION: u16 = 64;
const DATA_BASE: u32 = 0xa0000;

fn main() {
    ipu_runtime::init_tracing();
    let model_directory = PathBuf::from(
        std::env::var_os("IPU_SIGLIP_MODEL")
            .unwrap_or_else(|| "/srv/home/gc-sdk/siglip-so400m-patch14-384".into()),
    );
    let reference_path = PathBuf::from(
        std::env::var_os("IPU_SIGLIP_REFERENCE")
            .unwrap_or_else(|| "profiles/siglip/reference-b1.safetensors".into()),
    );
    let model = SiglipWeights::open(&model_directory).unwrap();
    let reference = TensorArchive::open(&reference_path).unwrap();
    let config = &model.config;
    let rows = u16::try_from(model.sequence_length()).unwrap();
    let patch_elements = config.num_channels * config.patch_size.pow(2);
    let inner = u16::try_from(patch_elements.div_ceil(64) * 64).unwrap();
    let columns = u16::try_from(config.hidden_size).unwrap();
    let row_block_dimension = choose_gemm_row_block_for(
        rows,
        INNER_BLOCK_DIMENSION,
        columns,
        BLOCK_DIMENSION,
        TILE_COUNT,
        GemmDataType::F16,
    )
    .unwrap();
    let mut plan = plan_blocked_gemm(BlockedGemmConfig {
        rows,
        inner_dimension: inner,
        columns,
        block_dimension: BLOCK_DIMENSION,
        inner_block_dimension: INNER_BLOCK_DIMENSION,
        row_block_dimension,
        tile_count: TILE_COUNT,
        data_base: DATA_BASE,
        data_limit: ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
        data_type: GemmDataType::F16,
    })
    .unwrap();

    let pixels = reference.tensor_f32("pixel_values").unwrap();
    let weights = model
        .tensor_f32("vision_model.embeddings.patch_embedding.weight")
        .unwrap();
    let bias = model
        .tensor_f32("vision_model.embeddings.patch_embedding.bias")
        .unwrap();
    let position = model
        .tensor_f32("vision_model.embeddings.position_embedding.weight")
        .unwrap();
    let mut host_input = blocked_matrix_f16(&plan.left, BlockLayout::AmpA16, |row, column| {
        patch_value(&pixels, config, row, column)
    });
    host_input.extend(blocked_matrix_f16(
        &plan.right,
        BlockLayout::AmpB16x16,
        |row, column| {
            if usize::from(row) < patch_elements {
                weights[usize::from(column) * patch_elements + usize::from(row)]
            } else {
                0.0
            }
        },
    ));

    let adjustment = append_adjustment_phase(&mut plan.schedule, &plan.output).unwrap();
    host_input.extend(blocked_matrix_f16(
        &adjustment,
        BlockLayout::AmpC16F16,
        |row, column| {
            position[usize::from(row) * config.hidden_size + usize::from(column)]
                + bias[usize::from(column)]
        },
    ));
    let transition_base = plan
        .schedule
        .allocations
        .iter()
        .filter(|allocation| {
            allocation.kind == AllocationKind::Home && allocation.address >= DATA_BASE
        })
        .map(|allocation| allocation.address + allocation.size)
        .max()
        .map(|address| (address + 31) & !31)
        .unwrap_or(DATA_BASE);
    let row_shards = append_c16_to_a16_row_shards(
        &mut plan.schedule,
        &plan.output,
        RowShardTransitionConfig {
            columns,
            data_base: transition_base,
            data_limit: ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
        },
    )
    .unwrap();
    let norm = append_affine_layer_norm_f16(
        &mut plan.schedule,
        &row_shards,
        AppendAffineLayerNormConfig {
            data_base: transition_base,
            data_limit: ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
            epsilon_bits: config.layer_norm_eps.to_bits(),
        },
    )
    .unwrap();
    let norm_weight = model
        .tensor_f32("vision_model.encoder.layers.0.layer_norm1.weight")
        .unwrap();
    let norm_bias = model
        .tensor_f32("vision_model.encoder.layers.0.layer_norm1.bias")
        .unwrap();
    for _ in &norm.affine {
        host_input.extend(
            norm_weight
                .iter()
                .chain(&norm_bias)
                .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes()),
        );
    }
    let qkv_data_base = resident_data_end(&plan.schedule);
    let qkv = append_blocked_gemm_f16_with_a16_input(
        &mut plan.schedule,
        &norm.output,
        BlockedGemmConfig {
            rows,
            inner_dimension: columns,
            columns: columns * 3,
            block_dimension: BLOCK_DIMENSION,
            inner_block_dimension: INNER_BLOCK_DIMENSION,
            row_block_dimension,
            tile_count: TILE_COUNT,
            data_base: qkv_data_base,
            data_limit: ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
            data_type: GemmDataType::F16,
        },
    )
    .unwrap();
    let qkv_weight_names = ["q_proj", "k_proj", "v_proj"];
    let qkv_weights = qkv_weight_names.map(|projection| {
        model
            .tensor_f32(&format!(
                "vision_model.encoder.layers.0.self_attn.{projection}.weight"
            ))
            .unwrap()
    });
    let qkv_biases = qkv_weight_names.map(|projection| {
        model
            .tensor_f32(&format!(
                "vision_model.encoder.layers.0.self_attn.{projection}.bias"
            ))
            .unwrap()
    });
    host_input.extend(blocked_matrix_f16(
        &qkv.right,
        BlockLayout::AmpB16x16,
        |row, column| {
            let projection = usize::from(column / columns);
            let output = usize::from(column % columns);
            qkv_weights[projection][output * usize::from(columns) + usize::from(row)]
        },
    ));
    let qkv_bias = append_adjustment_phase(&mut plan.schedule, &qkv.output).unwrap();
    host_input.extend(blocked_matrix_f16(
        &qkv_bias,
        BlockLayout::AmpC16F16,
        |_row, column| {
            let projection = usize::from(column / columns);
            qkv_biases[projection][usize::from(column % columns)]
        },
    ));
    let qkv_shards = (0..3)
        .map(|projection| {
            let blocks = projection_blocks(&qkv.output, projection, columns);
            let data_base = resident_data_end(&plan.schedule);
            append_c16_to_a16_row_shards(
                &mut plan.schedule,
                &blocks,
                RowShardTransitionConfig {
                    columns,
                    data_base,
                    data_limit: ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let attention_data_base = resident_data_end(&plan.schedule);
    let attention = append_flash_attention_from_a16_qkv(
        &mut plan.schedule,
        &qkv_shards[0],
        &qkv_shards[1],
        &qkv_shards[2],
        FlashAttentionConfig {
            batch_size: 1,
            sequence_length: rows,
            hidden_size: columns,
            attention_heads: u16::try_from(config.num_attention_heads).unwrap(),
            query_block_rows: 0,
            key_block_rows: 0,
            tile_count: TILE_COUNT,
            data_base: attention_data_base,
            data_limit: ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
        },
    )
    .unwrap();
    let attention_output_base = resident_data_end(&plan.schedule);
    let attention_shards = append_flash_attention_to_a16_row_shards(
        &mut plan.schedule,
        &attention,
        attention_output_base,
        ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
    )
    .unwrap();
    let objects = compile_objects(&plan, &attention).unwrap();
    let graph = ExecutableGraph {
        schedule: plan.schedule,
        initial_buffers: Vec::new(),
        outputs: Vec::new(),
        host_inputs: vec![
            block_binding_typed("patches", rows, inner, &plan.left, "f16", 2),
            block_binding_typed("patch_weight", inner, columns, &plan.right, "f16", 2),
            block_binding_typed("position_bias", rows, columns, &adjustment, "f16", 2),
            row_shard_binding("layer_norm1_affine", 2, columns, &norm.affine),
            block_binding_typed("qkv_weight", columns, columns * 3, &qkv.right, "f16", 2),
            block_binding_typed("qkv_bias", rows, columns * 3, &qkv_bias, "f16", 2),
        ],
        host_outputs: vec![row_shard_binding(
            "attention_hidden",
            rows,
            columns,
            &attention_shards,
        )],
    };
    let app = package_graph(&graph, &objects).unwrap();
    let sdk = PathBuf::from(required_env("POPLAR_SDK_ENABLED"));
    let configuration = fs::read(required_env("IPU_CONFIG")).unwrap();
    let bootloader = fs::read(
        std::env::var_os("IPU_BOOTLOADER")
            .map(PathBuf::from)
            .unwrap_or_else(|| sdk.join("bin/ipu/tile_bootloader_ipu2.elf")),
    )
    .unwrap();
    let actual = run_host_with_options(
        &app,
        &bootloader,
        &configuration,
        &std::env::var("IPU_DEVICE").unwrap_or_else(|_| "/dev/ipu0".into()),
        &host_input,
        HostRunOptions::from_environment().unwrap(),
    )
    .unwrap();
    let expected = reference
        .tensor_f32("encoder_layer_00_attention_heads")
        .unwrap();
    let expected = attention_heads_to_hidden(
        &expected,
        usize::from(rows),
        usize::from(attention.head_dimension),
        usize::try_from(config.num_attention_heads).unwrap(),
    );
    let expected = serialize_a16_row_shards(
        &expected,
        usize::from(rows),
        usize::from(columns),
        &attention_shards,
    );
    let max_error = verify_linear_f16(&actual, &expected);
    let limit = env_f32("IPU_F16_MAX_ERROR", 0.2);
    assert!(
        max_error <= limit,
        "attention max error {max_error} exceeds {limit}"
    );
    info!(
        image_size = config.image_size,
        patch_size = config.patch_size,
        patch_grid = model.patch_grid(),
        discarded_pixels = config.image_size % config.patch_size,
        rows,
        inner,
        columns,
        row_block_dimension,
        max_error,
        "SigLIP embedding, QKV, and FlashAttention passed against Hugging Face"
    );
}

fn patch_value(
    pixels: &[f32],
    config: &ipu_models::SiglipVisionConfig,
    patch: u16,
    inner: u16,
) -> f32 {
    let patch_elements = config.num_channels * config.patch_size.pow(2);
    let inner = usize::from(inner);
    if inner >= patch_elements {
        return 0.0;
    }
    let patch_grid = config.image_size / config.patch_size;
    let patch = usize::from(patch);
    let patch_y = patch / patch_grid;
    let patch_x = patch % patch_grid;
    let channel = inner / config.patch_size.pow(2);
    let within_channel = inner % config.patch_size.pow(2);
    let y = patch_y * config.patch_size + within_channel / config.patch_size;
    let x = patch_x * config.patch_size + within_channel % config.patch_size;
    pixels[(channel * config.image_size + y) * config.image_size + x]
}

fn append_adjustment_phase(
    schedule: &mut ipu_compiler::Schedule,
    output: &[BlockPlacement],
) -> ipu_runtime::Result<Vec<BlockPlacement>> {
    let mut cursors = vec![DATA_BASE; usize::from(TILE_COUNT)];
    let mut maximum_tensor = 0usize;
    for allocation in &schedule.allocations {
        maximum_tensor = maximum_tensor.max(allocation.tensor.0);
        if allocation.kind == AllocationKind::Home && allocation.address >= DATA_BASE {
            let end = allocation
                .address
                .checked_add(allocation.size)
                .ok_or("SRAM overflow")?;
            cursors[usize::from(allocation.tile)] = cursors[usize::from(allocation.tile)].max(end);
        }
    }
    let phase = schedule.phases.len();
    let mut placements = Vec::with_capacity(output.len());
    let mut commands = Vec::with_capacity(output.len());
    for (index, output) in output.iter().enumerate() {
        let tensor = TensorId(maximum_tensor + 1 + index);
        let bytes = u32::from(output.rows) * u32::from(output.columns) * 2;
        let cursor = &mut cursors[usize::from(output.tile)];
        *cursor = (*cursor + 7) & !7;
        let address = *cursor;
        *cursor = cursor
            .checked_add(bytes)
            .ok_or("adjustment allocation overflow")?;
        if *cursor > ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE {
            return Err(format!("position adjustment exhausts tile {} SRAM", output.tile).into());
        }
        let placement = BlockPlacement {
            tensor,
            address,
            ..*output
        };
        placements.push(placement);
        schedule.allocations.push(Allocation {
            tensor,
            tile: output.tile,
            address,
            size: bytes,
            live_from: 0,
            live_until: phase,
            kind: AllocationKind::Home,
        });
        let units = bytes / 4;
        commands.push(KernelCommand {
            tile: output.tile,
            output: output.tensor,
            inputs: vec![output.tensor, tensor],
            arguments: vec![units, units / 6, units % 6],
            specialization: SpecializationKey {
                operation: "add_f16".into(),
                shape: vec![usize::from(output.rows), usize::from(output.columns)],
                worker_count: 6,
                role: "patch-position-bias".into(),
                alignment: 4,
            },
            metadata: BTreeMap::from([
                ("label".into(), "patch bias and learned position".into()),
                ("row_start".into(), output.row_start.to_string()),
                ("column_start".into(), output.column_start.to_string()),
            ]),
        });
    }
    schedule.phases.push(Phase::Compute {
        op: OpId(phase),
        commands,
    });
    Ok(placements)
}

fn compile_objects(
    plan: &ipu_compiler::BlockedGemmPlan,
    attention: &FlashAttentionPlan,
) -> ipu_runtime::Result<Vec<Vec<u8>>> {
    let sdk = PathBuf::from(required_env("POPLAR_SDK_ENABLED"));
    let artifacts = std::env::temp_dir().join(format!("ipu-siglip-patch-{}", std::process::id()));
    let source = |name: &str| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../device")
            .join(name)
    };
    let minimum_rows = plan.left.iter().map(|block| block.rows).min().unwrap();
    let maximum_rows = plan.left.iter().map(|block| block.rows).max().unwrap();
    let toolchain = Toolchain::from_sdk(sdk);
    let runtime = toolchain.compile(source("static_runtime.S"), &artifacts, "runtime", &[])?;
    let gemm = toolchain.compile(
        source("gemm_f16_64_amp.S"),
        &artifacts,
        "patch-gemm",
        &[
            format!("-DGEMM_INNER_BLOCK_DIMENSION={INNER_BLOCK_DIMENSION}"),
            format!("-DGEMM_OUTPUT_COLUMNS={BLOCK_DIMENSION}"),
            format!("-DGEMM_SMALL_ROWS={minimum_rows}"),
            format!("-DGEMM_LARGE_ROWS={maximum_rows}"),
        ],
    )?;
    let add = toolchain.compile(source("add_f16.S"), &artifacts, "add-f16", &[])?;
    let relayout = toolchain.compile(source("relayout_f16.S"), &artifacts, "relayout-f16", &[])?;
    let norm_codelet = toolchain.compile(
        source("layer_norm_f16.cpp"),
        &artifacts,
        "layer-norm-codelet",
        &["-O1".into()],
    )?;
    let norm_wrapper = toolchain.compile(
        source("layer_norm_f16.S"),
        &artifacts,
        "layer-norm-wrapper",
        &[],
    )?;
    let attention_rows = attention.tasks.iter().map(|task| task.query_rows);
    let minimum_attention_rows = attention_rows.clone().min().unwrap();
    let maximum_attention_rows = attention_rows.max().unwrap();
    let key_rows = attention.key_values.iter().map(|block| block.key_rows);
    let minimum_key_rows = key_rows.clone().min().unwrap();
    let maximum_key_rows = key_rows.max().unwrap();
    let pack_codelet = toolchain.compile(
        source("attention_pack_f16.cpp"),
        &artifacts,
        "attention-pack-codelet",
        &[
            "-O1".into(),
            format!("-DATTENTION_HEAD_DIMENSION={}", attention.head_dimension),
            format!(
                "-DATTENTION_PADDED_HEAD_DIMENSION={}",
                attention.padded_head_dimension
            ),
        ],
    )?;
    let pack_wrapper = toolchain.compile(
        source("attention_pack_f16.S"),
        &artifacts,
        "attention-pack-wrapper",
        &[],
    )?;
    let unpack_codelet = toolchain.compile(
        source("attention_unpack_f16.cpp"),
        &artifacts,
        "attention-unpack-codelet",
        &[
            "-O1".into(),
            format!("-DATTENTION_HEAD_DIMENSION={}", attention.head_dimension),
        ],
    )?;
    let unpack_wrapper = toolchain.compile(
        source("attention_unpack_f16.S"),
        &artifacts,
        "attention-unpack-wrapper",
        &[],
    )?;
    let attention_codelet = toolchain.compile(
        source("flash_attention_f16.cpp"),
        &artifacts,
        "flash-attention-codelet",
        &[
            format!("-DATTENTION_HEAD_DIMENSION={}", attention.head_dimension),
            format!(
                "-DATTENTION_PADDED_HEAD_DIMENSION={}",
                attention.padded_head_dimension
            ),
            format!(
                "-DATTENTION_KEY_BLOCK_COLUMNS={}",
                attention.key_block_columns
            ),
            format!("-DATTENTION_SMALL_QUERY_ROWS={minimum_attention_rows}"),
            format!("-DATTENTION_LARGE_QUERY_ROWS={maximum_attention_rows}"),
            format!("-DATTENTION_SMALL_KEY_ROWS={minimum_key_rows}"),
            format!("-DATTENTION_LARGE_KEY_ROWS={maximum_key_rows}"),
        ],
    )?;
    let attention_wrapper = toolchain.compile(
        source("flash_attention_f16.S"),
        &artifacts,
        "flash-attention-wrapper",
        &[],
    )?;
    let attention_qk = toolchain.compile(
        source("gemm_f16_64_amp.S"),
        &artifacts,
        "attention-qk",
        &[
            format!(
                "-DGEMM_INNER_BLOCK_DIMENSION={}",
                attention.padded_head_dimension
            ),
            format!("-DGEMM_OUTPUT_COLUMNS={}", attention.key_block_columns),
            format!("-DGEMM_SMALL_ROWS={minimum_attention_rows}"),
            format!("-DGEMM_LARGE_ROWS={maximum_attention_rows}"),
            "-DGEMM_INIT_SMALL_SYMBOL=ipu_stack_attention_qk_init_small_rows".into(),
            "-DGEMM_INIT_LARGE_SYMBOL=ipu_stack_attention_qk_init_large_rows".into(),
            "-DGEMM_ACCUMULATE_SMALL_SYMBOL=ipu_stack_attention_qk_accumulate_small_rows".into(),
            "-DGEMM_ACCUMULATE_LARGE_SYMBOL=ipu_stack_attention_qk_accumulate_large_rows".into(),
        ],
    )?;
    let attention_pv = toolchain.compile(
        source("gemm_f16_64_amp.S"),
        &artifacts,
        "attention-pv",
        &[
            format!(
                "-DGEMM_INNER_BLOCK_DIMENSION={}",
                attention.key_block_columns
            ),
            format!("-DGEMM_OUTPUT_COLUMNS={}", attention.padded_head_dimension),
            format!("-DGEMM_SMALL_ROWS={minimum_attention_rows}"),
            format!("-DGEMM_LARGE_ROWS={maximum_attention_rows}"),
            "-DGEMM_INIT_SMALL_SYMBOL=ipu_stack_attention_pv_init_small_rows".into(),
            "-DGEMM_INIT_LARGE_SYMBOL=ipu_stack_attention_pv_init_large_rows".into(),
            "-DGEMM_ACCUMULATE_SMALL_SYMBOL=ipu_stack_attention_pv_accumulate_small_rows".into(),
            "-DGEMM_ACCUMULATE_LARGE_SYMBOL=ipu_stack_attention_pv_accumulate_large_rows".into(),
        ],
    )?;
    let worker_support = toolchain.compile(
        source("worker_support.S"),
        &artifacts,
        "worker-support",
        &[],
    )?;
    Ok(vec![
        fs::read(runtime.object)?,
        fs::read(gemm.object)?,
        fs::read(add.object)?,
        fs::read(relayout.object)?,
        fs::read(norm_codelet.object)?,
        fs::read(norm_wrapper.object)?,
        fs::read(pack_codelet.object)?,
        fs::read(pack_wrapper.object)?,
        fs::read(unpack_codelet.object)?,
        fs::read(unpack_wrapper.object)?,
        fs::read(attention_codelet.object)?,
        fs::read(attention_wrapper.object)?,
        fs::read(attention_qk.object)?,
        fs::read(attention_pv.object)?,
        fs::read(worker_support.object)?,
    ])
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

fn verify_linear_f16(actual: &[u8], expected: &[f32]) -> f32 {
    assert!(actual.len() >= expected.len() * 2);
    let mut max_error = 0.0f32;
    for (index, &expected) in expected.iter().enumerate() {
        let observed = f16::from_bits(u16::from_le_bytes(
            actual[index * 2..index * 2 + 2].try_into().unwrap(),
        ))
        .to_f32();
        assert!(observed.is_finite());
        max_error = max_error.max((observed - expected).abs());
    }
    max_error
}

fn attention_heads_to_hidden(
    heads: &[f32],
    rows: usize,
    head_dimension: usize,
    head_count: usize,
) -> Vec<f32> {
    assert_eq!(heads.len(), rows * head_dimension * head_count);
    let mut hidden = vec![0.0; heads.len()];
    for head in 0..head_count {
        for row in 0..rows {
            let source = (head * rows + row) * head_dimension;
            let destination = row * head_count * head_dimension + head * head_dimension;
            hidden[destination..destination + head_dimension]
                .copy_from_slice(&heads[source..source + head_dimension]);
        }
    }
    hidden
}

fn serialize_a16_row_shards(
    values: &[f32],
    rows: usize,
    columns: usize,
    shards: &[RowShardPlacement],
) -> Vec<f32> {
    assert_eq!(values.len(), rows * columns);
    let mut serialized = Vec::with_capacity(values.len());
    for shard in shards {
        for panel in 0..usize::from(shard.columns) / 16 {
            for row in 0..usize::from(shard.rows) {
                for column in 0..16 {
                    serialized.push(
                        values
                            [(usize::from(shard.row_start) + row) * columns + panel * 16 + column],
                    );
                }
            }
        }
    }
    serialized
}

fn resident_data_end(schedule: &ipu_compiler::Schedule) -> u32 {
    schedule
        .allocations
        .iter()
        .filter(|allocation| {
            allocation.kind == AllocationKind::Home && allocation.address >= DATA_BASE
        })
        .map(|allocation| allocation.address + allocation.size)
        .max()
        .map(|address| (address + 31) & !31)
        .unwrap_or(DATA_BASE)
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .map(|value| value.parse().unwrap())
        .unwrap_or(default)
}
