use half::f16;
use ipu_compiler::{
    Allocation, AllocationKind, BlockPlacement, BlockedGemmConfig, FlashAttentionConfig,
    FlashAttentionPlan, GemmDataType, KernelCommand, MemoryConstraint, MemoryPlacement, OpId,
    Phase, RowShardPlacement, RowShardTransitionConfig, SpecializationKey, TensorId,
    append_c16_to_a16_row_shards, choose_gemm_row_block_for, plan_blocked_gemm,
    plan_flash_attention,
};
use ipu_elf::Toolchain;
use ipu_models::{SiglipWeights, TensorArchive};
use ipu_package::{Binding, RegionSlice};
use ipu_runtime::{
    BlockLayout, ExecutableGraph, HostRunOptions, HostTensorSet, append_host_a16_matrix,
    append_siglip_encoder_layer, append_siglip_map_head, append_siglip_post_layer_norm,
    block_binding_typed, block_coordinates, blocked_matrix_f16, package_graph,
    run_host_with_options,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use tracing::info;

const TILE_COUNT: u16 = 1472;
const BLOCK_DIMENSION: u16 = 64;
const INNER_BLOCK_DIMENSION: u16 = 64;
const DATA_BASE: u32 = ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT;

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
    if std::env::var_os("IPU_SIGLIP_MAP_ONLY").is_some() {
        run_map_only(&model, &reference);
        return;
    }
    let config = &model.config;
    let rows = u16::try_from(model.sequence_length()).unwrap();
    let patch_elements = config.num_channels * config.patch_size.pow(2);
    let inner = u16::try_from(patch_elements.div_ceil(64) * 64).unwrap();
    let columns = u16::try_from(config.hidden_size).unwrap();
    let automatic_row_block_dimension = choose_gemm_row_block_for(
        rows,
        INNER_BLOCK_DIMENSION,
        columns,
        BLOCK_DIMENSION,
        TILE_COUNT,
        GemmDataType::F16,
    )
    .unwrap();
    let row_block_dimension = std::env::var("IPU_SIGLIP_ROW_BLOCK_ROWS")
        .map(|value| value.parse().unwrap())
        .unwrap_or(automatic_row_block_dimension);
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
    let mut host = HostTensorSet::default();
    let patch_bytes = blocked_matrix_f16(&plan.left, BlockLayout::AmpA16, |row, column| {
        patch_value(&pixels, config, row, column)
    });
    host.push(
        block_binding_typed("patches", rows, inner, &plan.left, "f16", 2),
        patch_bytes,
    )
    .unwrap();
    let patch_weight_bytes =
        blocked_matrix_f16(&plan.right, BlockLayout::AmpB16x16, |row, column| {
            if usize::from(row) < patch_elements {
                weights[usize::from(column) * patch_elements + usize::from(row)]
            } else {
                0.0
            }
        });
    host.push(
        block_binding_typed("patch_weight", inner, columns, &plan.right, "f16", 2),
        patch_weight_bytes,
    )
    .unwrap();

    let adjustment = append_adjustment_phase(&mut plan.schedule, &plan.output).unwrap();
    let adjustment_bytes =
        blocked_matrix_f16(&adjustment, BlockLayout::AmpC16F16, |row, column| {
            position[usize::from(row) * config.hidden_size + usize::from(column)]
                + bias[usize::from(column)]
        });
    host.push(
        block_binding_typed("position_bias", rows, columns, &adjustment, "f16", 2),
        adjustment_bytes,
    )
    .unwrap();
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
    let layer_count = std::env::var("IPU_SIGLIP_LAYER_COUNT")
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(1);
    assert!((1..=config.num_hidden_layers).contains(&layer_count));
    let data_limit = ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE;
    let detailed_diagnostics = layer_count == 1;
    let mut current = row_shards;
    let mut last_layer = None;
    for layer in 0..layer_count {
        let appended = append_siglip_encoder_layer(
            &mut plan.schedule,
            &current,
            &model,
            layer,
            rows,
            columns,
            row_block_dimension,
            TILE_COUNT,
            DATA_BASE,
            data_limit,
            detailed_diagnostics && layer + 1 == layer_count,
            &mut host,
        )
        .unwrap();
        current = appended.output.clone();
        last_layer = Some(appended);
    }
    let last_layer = last_layer.unwrap();
    let norm2 = last_layer.norm2;
    let mlp_gelu = last_layer.mlp_gelu;
    let layer_output = last_layer.output;
    let attention = last_layer.attention;
    let intermediate_columns = u16::try_from(config.intermediate_size.div_ceil(64) * 64).unwrap();
    let objects = compile_objects(&plan, &attention).unwrap();
    let HostTensorSet {
        bindings: host_inputs,
        bytes: host_input,
    } = host;
    let mut host_outputs = Vec::new();
    if detailed_diagnostics {
        host_outputs.push(row_shard_binding("layer_norm2", rows, columns, &norm2));
        host_outputs.push(block_binding_typed(
            "mlp_gelu",
            rows,
            intermediate_columns,
            &mlp_gelu,
            "f16",
            2,
        ));
    }
    host_outputs.push(row_shard_binding(
        &format!("encoder_layer_{:02}", layer_count - 1),
        rows,
        columns,
        &layer_output,
    ));
    let graph = ExecutableGraph {
        schedule: plan.schedule,
        initial_buffers: Vec::new(),
        outputs: Vec::new(),
        host_inputs,
        host_outputs,
    };
    let app = package_graph(&graph, &objects).unwrap();
    if let Some(path) = std::env::var_os("IPU_SIGLIP_PACKAGE_OUTPUT") {
        app.write(fs::File::create(path).unwrap()).unwrap();
    }
    if let Some(path) = std::env::var_os("IPU_SIGLIP_INPUT_OUTPUT") {
        fs::write(path, &host_input).unwrap();
    }
    if std::env::var_os("IPU_SIGLIP_BUILD_ONLY").is_some() {
        info!(
            layer_count,
            "SigLIP executable built without device execution"
        );
        return;
    }
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
    let (diagnostic_bytes, norm2_error, mlp_gelu_error) = if detailed_diagnostics {
        let norm2_bytes = usize::from(rows) * usize::from(columns) * 2;
        let expected_norm2 = reference.tensor_f32("encoder_layer_00_norm2").unwrap();
        let expected_norm2 = serialize_a16_row_shards(
            &expected_norm2,
            usize::from(rows),
            usize::from(columns),
            &norm2,
        );
        let norm2_error = verify_linear_f16(&actual[..norm2_bytes], &expected_norm2);
        let expected = reference.tensor_f32("encoder_layer_00_mlp_gelu").unwrap();
        let expected = pad_columns(
            &expected,
            usize::from(rows),
            config.intermediate_size,
            usize::from(intermediate_columns),
        );
        let expected = serialize_a16_blocks(
            &expected,
            usize::from(rows),
            usize::from(intermediate_columns),
            &mlp_gelu,
        );
        let mlp_gelu_error = verify_linear_f16(&actual[norm2_bytes..], &expected);
        let gelu_bytes = usize::from(rows) * usize::from(intermediate_columns) * 2;
        (
            norm2_bytes + gelu_bytes,
            Some(norm2_error),
            Some(mlp_gelu_error),
        )
    } else {
        (0, None, None)
    };
    let expected_layer = reference
        .tensor_f32(&format!("encoder_layer_{:02}", layer_count - 1))
        .unwrap();
    let expected_layer = serialize_a16_row_shards(
        &expected_layer,
        usize::from(rows),
        usize::from(columns),
        &layer_output,
    );
    let layer_error = verify_linear_f16(&actual[diagnostic_bytes..], &expected_layer);
    let limit = env_f32("IPU_F16_MAX_ERROR", 0.2);
    info!(
        ?norm2_error,
        ?mlp_gelu_error,
        layer_error,
        limit,
        "SigLIP verification results"
    );
    if let Some(error) = norm2_error {
        assert!(error <= limit, "norm2 max error {error} exceeds {limit}");
    }
    if let Some(error) = mlp_gelu_error {
        assert!(error <= limit, "MLP GeLU max error {error} exceeds {limit}");
    }
    assert!(
        layer_error <= limit,
        "encoder layer max error {layer_error} exceeds {limit}"
    );
    info!(
        image_size = config.image_size,
        patch_size = config.patch_size,
        patch_grid = model.patch_grid(),
        discarded_pixels = config.image_size % config.patch_size,
        rows,
        inner,
        columns,
        intermediate_columns,
        row_block_dimension,
        layer_count,
        ?norm2_error,
        ?mlp_gelu_error,
        layer_error,
        "SigLIP encoder prefix passed against Hugging Face"
    );
}

fn run_map_only(model: &SiglipWeights, reference: &TensorArchive) {
    let config = &model.config;
    let rows = u16::try_from(model.sequence_length()).unwrap();
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
    let compilation_plan = plan_blocked_gemm(BlockedGemmConfig {
        rows,
        inner_dimension: columns,
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
    let mut schedule = ipu_compiler::Schedule {
        layouts: Vec::new(),
        phases: Vec::new(),
        allocations: Vec::new(),
        tile_count: TILE_COUNT,
        peak_sram: BTreeMap::new(),
    };
    let mut host = HostTensorSet::default();
    let input_values = reference.tensor_f32("encoder_layer_26").unwrap();
    let input = append_host_a16_matrix(
        &mut schedule,
        "map.input",
        &input_values,
        rows,
        columns,
        row_block_dimension,
        DATA_BASE,
        ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
        &mut host,
    )
    .unwrap();
    let input = append_siglip_post_layer_norm(
        &mut schedule,
        &input,
        model,
        DATA_BASE,
        ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
        &mut host,
    )
    .unwrap();
    let post_norm_only = std::env::var_os("IPU_SIGLIP_POST_NORM_ONLY").is_some();
    let (output, output_rows, attention, expected) = if post_norm_only {
        let attention = plan_flash_attention(FlashAttentionConfig {
            batch_size: 1,
            query_sequence_length: 12,
            sequence_length: rows,
            hidden_size: columns,
            attention_heads: u16::try_from(config.num_attention_heads).unwrap(),
            query_block_rows: 12,
            key_block_rows: 0,
            tile_count: TILE_COUNT,
            data_base: DATA_BASE,
            data_limit: ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
        })
        .unwrap();
        (
            input,
            rows,
            attention,
            reference.tensor_f32("post_layernorm").unwrap(),
        )
    } else {
        let map = append_siglip_map_head(
            &mut schedule,
            &input,
            model,
            rows,
            row_block_dimension,
            TILE_COUNT,
            DATA_BASE,
            ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
            &mut host,
        )
        .unwrap();
        let expected = reference.tensor_f32("pooler_output").unwrap();
        let expected = (0..12)
            .flat_map(|_| expected.iter().copied())
            .collect::<Vec<_>>();
        (map.output, 12, map.attention, expected)
    };
    let objects = compile_objects(&compilation_plan, &attention).unwrap();
    let HostTensorSet {
        bindings: host_inputs,
        bytes: host_input,
    } = host;
    let graph = ExecutableGraph {
        schedule,
        initial_buffers: Vec::new(),
        outputs: Vec::new(),
        host_inputs,
        host_outputs: vec![row_shard_binding(
            if post_norm_only {
                "post_layernorm"
            } else {
                "pooler_output"
            },
            output_rows,
            columns,
            &output,
        )],
    };
    let app = package_graph(&graph, &objects).unwrap();
    if std::env::var_os("IPU_SIGLIP_BUILD_ONLY").is_some() {
        info!("SigLIP MAP executable built without device execution");
        return;
    }
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
    let expected = serialize_a16_row_shards(
        &expected,
        usize::from(output_rows),
        usize::from(columns),
        &output,
    );
    let error = verify_linear_f16(&actual, &expected);
    let limit = env_f32("IPU_F16_MAX_ERROR", 0.2);
    info!(error, limit, "SigLIP MAP verification result");
    assert!(error <= limit, "MAP max error {error} exceeds {limit}");
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
    let mut maximum_tensor = 0usize;
    for allocation in &schedule.allocations {
        maximum_tensor = maximum_tensor.max(allocation.tensor.0);
    }
    let phase = schedule.phases.len();
    let mut placements = Vec::with_capacity(output.len());
    let mut commands = Vec::with_capacity(output.len());
    for (index, output) in output.iter().enumerate() {
        let tensor = TensorId(maximum_tensor + 1 + index);
        let bytes = u32::from(output.rows) * u32::from(output.columns) * 2;
        let address = ipu_compiler::find_free_region(
            &schedule.allocations,
            output.tile,
            bytes,
            0,
            phase,
            MemoryConstraint {
                base: DATA_BASE,
                limit: ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
                alignment: 8,
                placement: MemoryPlacement::Low,
            },
        )?;
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
    let add_bias = toolchain.compile(source("add_bias_f16.S"), &artifacts, "add-bias-f16", &[])?;
    let relayout = toolchain.compile(source("relayout_f16.S"), &artifacts, "relayout-f16", &[])?;
    let gelu_relayout = toolchain.compile(
        source("gelu_relayout_f16.S"),
        &artifacts,
        "gelu-relayout-f16",
        &[],
    )?;
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
        fs::read(add_bias.object)?,
        fs::read(relayout.object)?,
        fs::read(gelu_relayout.object)?,
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

fn verify_linear_f16(actual: &[u8], expected: &[f32]) -> f32 {
    assert!(actual.len() >= expected.len() * 2);
    let mut max_error = 0.0f32;
    let mut worst = (0usize, 0.0f32, 0.0f32);
    let mut observed_min = f32::INFINITY;
    let mut observed_max = f32::NEG_INFINITY;
    for (index, &expected) in expected.iter().enumerate() {
        let observed = f16::from_bits(u16::from_le_bytes(
            actual[index * 2..index * 2 + 2].try_into().unwrap(),
        ))
        .to_f32();
        assert!(observed.is_finite());
        observed_min = observed_min.min(observed);
        observed_max = observed_max.max(observed);
        let error = (observed - expected).abs();
        if error > max_error {
            max_error = error;
            worst = (index, observed, expected);
        }
    }
    info!(
        worst_index = worst.0,
        worst_observed = worst.1,
        worst_expected = worst.2,
        observed_min,
        observed_max,
        "FP16 comparison diagnostics"
    );
    max_error
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

fn serialize_a16_blocks(
    values: &[f32],
    rows: usize,
    columns: usize,
    blocks: &[BlockPlacement],
) -> Vec<f32> {
    assert_eq!(values.len(), rows * columns);
    let mut serialized = Vec::with_capacity(values.len());
    for block in blocks {
        for linear in 0..block.rows * block.columns {
            let (row, column) =
                block_coordinates(BlockLayout::AmpA16, block.rows, block.columns, linear);
            serialized.push(
                values[(usize::from(block.row_start + row)) * columns
                    + usize::from(block.column_start + column)],
            );
        }
    }
    serialized
}

fn pad_columns(values: &[f32], rows: usize, columns: usize, padded_columns: usize) -> Vec<f32> {
    assert_eq!(values.len(), rows * columns);
    assert!(padded_columns >= columns);
    let mut padded = vec![0.0; rows * padded_columns];
    for row in 0..rows {
        padded[row * padded_columns..row * padded_columns + columns]
            .copy_from_slice(&values[row * columns..(row + 1) * columns]);
    }
    padded
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .map(|value| value.parse().unwrap())
        .unwrap_or(default)
}
