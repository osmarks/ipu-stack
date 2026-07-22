use half::f16;
use ipu_compiler::{
    Allocation, AllocationKind, BlockPlacement, BlockedGemmConfig, FlashAttentionConfig,
    FlashAttentionPlan, GemmDataType, Ipu21MemoryRegion, KernelCommand, MemoryConstraint,
    MemoryPlacement, MemoryPolicy, OpId, Phase, RepeatedRegion, RowShardPlacement,
    RowShardTransitionConfig, SpecializationKey, TensorId, append_c16_to_a16_row_shards,
    choose_gemm_row_block_for_shape, end_tensor_lifetimes, make_tensors_resident,
    plan_blocked_gemm, plan_flash_attention,
};
use ipu_elf::{KernelArtifact, Toolchain};
use ipu_models::{SiglipWeights, TensorArchive};
use ipu_package::{Binding, RegionSlice};
use ipu_runtime::{
    BlockLayout, ExecutableGraph, HostRunOptions, HostTensorSet, ProfileGranularity,
    SiglipEncoderPrecision, SiglipEncoderTuning, SiglipLinearPrecision, SiglipWeightStorage,
    StaticProfileRegion, StaticTemplateRegion, allocator_memory_profile, append_host_a16_matrix,
    append_siglip_encoder_layer_batched_with_precision, append_siglip_map_head,
    append_siglip_map_head_with_memory_policy, append_siglip_post_layer_norm,
    append_siglip_post_layer_norm_with_memory_policy, block_binding_typed, block_coordinates,
    blocked_matrix_f16, consolidate_attention_kernel_variants, defer_terminal_residual_add,
    fuse_deferred_residual_into_layer_norm, materialize_deferred_residual_add,
    package_graph_repeated, package_graph_repeated_with_templates,
    package_graph_repeated_with_templates_profiled_regions,
    package_graph_repeated_with_templates_profiled_with_regions, run_host_with_options,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

const TILE_COUNT: u16 = 1472;
const BLOCK_DIMENSION: u16 = 64;
const INNER_BLOCK_DIMENSION: u16 = 64;
const DATA_BASE: u32 = ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT;
const ORDINARY_LOW_BASE: u32 =
    ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES;

fn main() {
    ipu_runtime::init_tracing();
    let profile_output = std::env::var_os("IPU_PROFILE_OUTPUT").map(PathBuf::from);
    let profile_granularity = profile_output
        .as_ref()
        .map(|_| ProfileGranularity::from_environment().unwrap());
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
    let sequence_length = u16::try_from(model.sequence_length()).unwrap();
    let batch_size =
        u16::try_from(env_u32("IPU_SIGLIP_BATCH_SIZE", 1)).expect("SigLIP batch size exceeds u16");
    let rows = sequence_length
        .checked_mul(batch_size)
        .expect("SigLIP flattened row count exceeds u16");
    let patch_elements = config.num_channels * config.patch_size.pow(2);
    let inner = u16::try_from(patch_elements.div_ceil(64) * 64).unwrap();
    let columns = u16::try_from(config.hidden_size).unwrap();
    let data_limit = ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE;
    let memory = encoder_memory_policy(data_limit);
    let automatic_row_block_dimension = choose_gemm_row_block_for_shape(
        rows,
        inner,
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
        retain_profile_metadata: true,
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
        patch_value(&pixels, config, row % sequence_length, column)
    });
    host.push_input(
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
    make_tensors_resident(
        &mut plan.schedule,
        plan.right.iter().map(|block| block.tensor),
    )
    .unwrap();

    let adjustment = append_adjustment_phase(&mut plan.schedule, &plan.output).unwrap();
    let adjustment_bytes =
        blocked_matrix_f16(&adjustment, BlockLayout::AmpC16F16, |row, column| {
            position[usize::from(row % sequence_length) * config.hidden_size + usize::from(column)]
                + bias[usize::from(column)]
        });
    host.push(
        block_binding_typed("position_bias", rows, columns, &adjustment, "f16", 2),
        adjustment_bytes,
    )
    .unwrap();
    make_tensors_resident(
        &mut plan.schedule,
        adjustment.iter().map(|block| block.tensor),
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
    end_tensor_lifetimes(
        &mut plan.schedule,
        plan.left
            .iter()
            .chain(&plan.output)
            .map(|block| block.tensor),
    )
    .unwrap();
    let layer_count = std::env::var("IPU_SIGLIP_LAYER_COUNT")
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(1);
    assert!((1..=config.num_hidden_layers).contains(&layer_count));
    let weight_storage = match std::env::var("IPU_SIGLIP_WEIGHT_STORAGE").as_deref() {
        Ok("f16") => SiglipWeightStorage::F16,
        Ok("f143") | Err(_) => SiglipWeightStorage::F143,
        Ok(value) => panic!("unsupported SigLIP weight storage {value}"),
    };
    let precision = siglip_encoder_precision(weight_storage);
    let resident_f16_layers = usize::try_from(env_u32(
        "IPU_SIGLIP_RESIDENT_F16_PREFIX_LAYERS",
        u32::try_from(layer_count).unwrap(),
    ))
    .unwrap();
    assert!(resident_f16_layers <= layer_count);
    let tuning = SiglipEncoderTuning {
        gemm_row_block_rows: u16::try_from(env_u32("IPU_SIGLIP_GEMM_ROW_BLOCK_ROWS", 0))
            .expect("GEMM row block rows exceed u16"),
        automatic_gemm_row_blocks: env_u32("IPU_SIGLIP_AUTOMATIC_GEMM_ROW_BLOCKS", 1) != 0,
        row_gemm_inner_block_columns: u16::try_from(env_u32(
            "IPU_SIGLIP_ROW_GEMM_INNER_BLOCK_COLUMNS",
            0,
        ))
        .expect("row-sharded GEMM inner block columns exceed u16"),
        qkv_inner_block_columns: u16::try_from(env_u32("IPU_SIGLIP_QKV_INNER_BLOCK_COLUMNS", 0))
            .expect("QKV inner block columns exceed u16"),
        attention_output_inner_block_columns: u16::try_from(env_u32(
            "IPU_SIGLIP_ATTENTION_OUTPUT_INNER_BLOCK_COLUMNS",
            0,
        ))
        .expect("attention-output inner block columns exceed u16"),
        mlp_up_inner_block_columns: u16::try_from(env_u32(
            "IPU_SIGLIP_MLP_UP_INNER_BLOCK_COLUMNS",
            0,
        ))
        .expect("MLP-up inner block columns exceed u16"),
        gemm_output_block_columns: u16::try_from(env_u32(
            "IPU_SIGLIP_GEMM_OUTPUT_BLOCK_COLUMNS",
            0,
        ))
        .expect("GEMM output block columns exceed u16"),
        qkv_output_block_columns: u16::try_from(env_u32("IPU_SIGLIP_QKV_OUTPUT_BLOCK_COLUMNS", 0))
            .expect("QKV output block columns exceed u16"),
        attention_output_block_columns: u16::try_from(env_u32(
            "IPU_SIGLIP_ATTENTION_OUTPUT_BLOCK_COLUMNS",
            0,
        ))
        .expect("attention-output block columns exceed u16"),
        mlp_up_output_block_columns: u16::try_from(env_u32(
            "IPU_SIGLIP_MLP_UP_OUTPUT_BLOCK_COLUMNS",
            0,
        ))
        .expect("MLP-up output block columns exceed u16"),
        mlp_down_output_block_columns: u16::try_from(env_u32(
            "IPU_SIGLIP_MLP_DOWN_OUTPUT_BLOCK_COLUMNS",
            0,
        ))
        .expect("MLP-down output block columns exceed u16"),
        attention_query_block_rows: u16::try_from(env_u32(
            "IPU_SIGLIP_ATTENTION_QUERY_BLOCK_ROWS",
            0,
        ))
        .expect("attention query block rows exceed u16"),
        attention_key_block_rows: u16::try_from(env_u32("IPU_SIGLIP_ATTENTION_KEY_BLOCK_ROWS", 0))
            .expect("attention key block rows exceed u16"),
    };
    info!(
        transient = ?memory.transient,
        resident = ?memory.resident,
        ?weight_storage,
        ?precision,
        resident_f16_layers,
        ?tuning,
        "configured encoder tile-memory policy"
    );
    let detailed_diagnostics = layer_count == 1 && batch_size == 1;
    let retain_profile_metadata = std::env::var_os("IPU_SIGLIP_RETAIN_PROFILE_METADATA").is_some()
        || matches!(
            profile_granularity,
            Some(ProfileGranularity::Phase | ProfileGranularity::Step)
        );
    let mut current = row_shards;
    let mut last_layer = None;
    let mut deferred_residual = None;
    let mut layer_template_groups = Vec::<(SiglipEncoderPrecision, RepeatedRegion)>::new();
    let mut profile_regions = vec![StaticProfileRegion {
        name: "embedding".into(),
        phases: 0..plan.schedule.phases.len(),
    }];
    let (mut compute_command_count, mut transfer_count) = phase_entry_counts(&plan.schedule.phases);
    for layer in 0..layer_count {
        let phase_start = plan.schedule.phases.len();
        let layer_precision = if layer < resident_f16_layers {
            precision
        } else {
            expanded_storage_fallback(precision)
        };
        let mut appended = append_siglip_encoder_layer_batched_with_precision(
            &mut plan.schedule,
            &current,
            &model,
            layer,
            batch_size,
            rows,
            columns,
            row_block_dimension,
            TILE_COUNT,
            &memory,
            layer_precision,
            tuning,
            retain_profile_metadata,
            detailed_diagnostics && layer + 1 == layer_count,
            &mut host,
        )
        .unwrap();
        if let Some(deferred) = deferred_residual.take() {
            fuse_deferred_residual_into_layer_norm(&mut plan.schedule, phase_start, deferred)
                .unwrap();
        }
        if let Some(deferred) = defer_terminal_residual_add(&mut plan.schedule).unwrap() {
            appended
                .profile_stages
                .last_mut()
                .expect("encoder layer has profile stages")
                .phases
                .end -= 2;
            deferred_residual = Some(deferred);
        }
        profile_regions.extend(
            appended
                .profile_stages
                .iter()
                .map(|stage| StaticProfileRegion {
                    name: stage.name.clone(),
                    phases: stage.phases.clone(),
                }),
        );
        current = appended.output.clone();
        last_layer = Some(appended);
        let phase_range = phase_start..plan.schedule.phases.len();
        specialize_gemm_row_operations(&mut plan.schedule, phase_range.clone());
        if !retain_profile_metadata {
            plan.schedule.discard_profile_metadata(phase_range.clone());
        }
        if let Some((group_precision, region)) = layer_template_groups.last_mut()
            && *group_precision == layer_precision
            && region.is_compatible(&plan.schedule, phase_range.clone())
        {
            region.push_instance(&plan.schedule, phase_range).unwrap();
        } else {
            let name = format!("siglip_encoder_layer_{}", layer_template_groups.len());
            layer_template_groups.push((
                layer_precision,
                RepeatedRegion::new(name, &plan.schedule, phase_range).unwrap(),
            ));
        }
        let (new_compute_commands, new_transfers) =
            phase_entry_counts(&plan.schedule.phases[phase_start..]);
        compute_command_count += new_compute_commands;
        transfer_count += new_transfers;
        info!(
            layer,
            phases = plan.schedule.phases.len(),
            compute_commands = compute_command_count,
            transfers = transfer_count,
            allocations = plan.schedule.allocations.len(),
            host_bindings = host.bindings.len(),
            host_bytes = host.bytes.len(),
            "accumulated SigLIP compiler state"
        );
    }
    materialize_deferred_residual_add(
        &mut plan.schedule,
        deferred_residual
            .take()
            .expect("encoder layer has a terminal residual add"),
    )
    .unwrap();
    profile_regions
        .last_mut()
        .expect("encoder layer has a profile region")
        .phases
        .end = plan.schedule.phases.len();
    let last_layer = last_layer.unwrap();
    let norm2 = last_layer.norm2;
    let mlp_gelu = last_layer.mlp_gelu;
    let layer_output = last_layer.output;
    let attention = last_layer.attention;
    let intermediate_columns = u16::try_from(config.intermediate_size.div_ceil(64) * 64).unwrap();
    let full_model = std::env::var_os("IPU_SIGLIP_FULL_MODEL").is_some();
    assert!(
        !full_model || layer_count == config.num_hidden_layers,
        "full-model execution requires every encoder layer"
    );
    assert!(
        !full_model || batch_size == 1,
        "batched MAP-head execution is not implemented"
    );
    let (output, output_rows, output_name, attentions, post_norm_output) = if full_model {
        let post_norm = append_siglip_post_layer_norm_with_memory_policy(
            &mut plan.schedule,
            &layer_output,
            &model,
            &memory,
            &mut host,
        )
        .unwrap();
        let map = append_siglip_map_head_with_memory_policy(
            &mut plan.schedule,
            &post_norm,
            &model,
            rows,
            row_block_dimension,
            TILE_COUNT,
            DATA_BASE,
            data_limit,
            &memory,
            &mut host,
        )
        .unwrap();
        (
            map.output,
            12,
            "pooler_output".to_string(),
            vec![attention, map.attention],
            Some(post_norm),
        )
    } else {
        (
            layer_output,
            rows,
            format!("encoder_layer_{:02}", layer_count - 1),
            vec![attention],
            None,
        )
    };
    let attention_variant = consolidate_attention_kernel_variants(&mut plan.schedule, &attentions);
    let profiled_end = profile_regions.last().unwrap().phases.end;
    if profiled_end < plan.schedule.phases.len() {
        profile_regions.push(StaticProfileRegion {
            name: if full_model {
                "post_norm_map_head".into()
            } else {
                "output".into()
            },
            phases: profiled_end..plan.schedule.phases.len(),
        });
    }
    let phase_count = plan.schedule.phases.len();
    specialize_gemm_row_operations(&mut plan.schedule, 0..phase_count);
    let objects = compile_objects(&plan.schedule, &attentions, attention_variant).unwrap();
    let HostTensorSet {
        bindings: host_inputs,
        bytes: host_input,
        resident_bindings: host_weights,
        resident_bytes: host_weight_bytes,
    } = host;
    let invocations = env_u32("IPU_SIGLIP_INVOCATIONS", 1);
    let mut host_input = host_input.repeat(invocations as usize);
    host_input.extend(host_weight_bytes);
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
    if std::env::var_os("IPU_SIGLIP_DIAGNOSTIC_BOUNDARIES").is_some() {
        if let Some(post_norm) = &post_norm_output {
            host_outputs.push(row_shard_binding(
                "post_layernorm",
                rows,
                columns,
                post_norm,
            ));
        }
    }
    host_outputs.push(row_shard_binding(
        &output_name,
        output_rows,
        columns,
        &output,
    ));
    let graph = ExecutableGraph {
        memory_policy: Some(memory.clone()),
        host_weights,
        schedule: plan.schedule,
        initial_buffers: Vec::new(),
        outputs: Vec::new(),
        host_inputs,
        host_outputs,
    };
    write_memory_profile(&graph);
    let templates = layer_template_groups
        .into_iter()
        .map(|(_, region)| StaticTemplateRegion::from(region))
        .collect::<Vec<_>>();
    assert!(
        profile_output.is_none() || invocations == 1,
        "semantic profiling currently requires one invocation"
    );
    let (app, profile_layout) = if profile_output.is_some() {
        let granularity = profile_granularity.expect("profile output has a granularity");
        let (app, layout) = if granularity == ProfileGranularity::Graph {
            package_graph_repeated_with_templates_profiled_regions(
                &graph,
                &objects,
                &templates,
                &profile_regions,
                invocations,
            )
        } else {
            package_graph_repeated_with_templates_profiled_with_regions(
                &graph,
                &objects,
                &templates,
                &profile_regions,
                granularity,
                invocations,
            )
        }
        .unwrap();
        info!(
            ?granularity,
            regions = profile_regions.len(),
            "enabled SigLIP profiling"
        );
        (app, Some(layout))
    } else {
        (
            package_graph_repeated_with_templates(&graph, &objects, &templates, invocations)
                .unwrap(),
            None,
        )
    };
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
    let mut actual = run_host_with_options(
        &app,
        &bootloader,
        &configuration,
        &std::env::var("IPU_DEVICE").unwrap_or_else(|_| "/dev/ipu0".into()),
        &host_input,
        HostRunOptions::from_environment().unwrap(),
    )
    .unwrap();
    if let (Some(path), Some(layout)) = (&profile_output, &profile_layout) {
        let clock_hz = std::env::var("IPU_CLOCK_HZ")
            .map(|value| value.parse().expect("IPU_CLOCK_HZ must be an integer"))
            .unwrap_or(1_500_000_000);
        let report = layout.decode(&actual, clock_hz).unwrap();
        report.write(fs::File::create(path).unwrap()).unwrap();
        actual.truncate(layout.output_offset);
        info!(path = %path.display(), clock_hz, "wrote semantic SigLIP cycle profile");
    }
    let invocation_output_bytes = actual.len() / invocations as usize;
    assert_eq!(
        actual.len(),
        invocation_output_bytes * invocations as usize,
        "runtime returned a partial invocation output"
    );
    let invocation_outputs = actual
        .chunks_exact(invocation_output_bytes)
        .collect::<Vec<_>>();
    let (diagnostic_bytes, norm2_error, mlp_gelu_error) = if detailed_diagnostics {
        let norm2_bytes = usize::from(rows) * usize::from(columns) * 2;
        let expected_norm2 = reference.tensor_f32("encoder_layer_00_norm2").unwrap();
        let expected_norm2 = serialize_a16_row_shards(
            &expected_norm2,
            usize::from(rows),
            usize::from(columns),
            &norm2,
        );
        let norm2_error = invocation_outputs
            .iter()
            .enumerate()
            .map(|(invocation, actual)| {
                let error = verify_linear_f16(&actual[..norm2_bytes], &expected_norm2);
                info!(
                    invocation,
                    error,
                    stage = "norm2",
                    "SigLIP invocation error"
                );
                error
            })
            .fold(0.0, f32::max);
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
        let mlp_gelu_error = invocation_outputs
            .iter()
            .enumerate()
            .map(|(invocation, actual)| {
                let error = verify_linear_f16(&actual[norm2_bytes..], &expected);
                info!(
                    invocation,
                    error,
                    stage = "mlp_gelu",
                    "SigLIP invocation error"
                );
                error
            })
            .fold(0.0, f32::max);
        let gelu_bytes = usize::from(rows) * usize::from(intermediate_columns) * 2;
        (
            norm2_bytes + gelu_bytes,
            Some(norm2_error),
            Some(mlp_gelu_error),
        )
    } else {
        (0, None, None)
    };
    let expected_layer = if full_model {
        let expected = reference.tensor_f32("pooler_output").unwrap();
        (0..usize::from(output_rows))
            .flat_map(|_| expected.iter().copied())
            .collect::<Vec<_>>()
    } else {
        let expected = reference
            .tensor_f32(&format!("encoder_layer_{:02}", layer_count - 1))
            .unwrap();
        expected.repeat(usize::from(batch_size))
    };
    let expected_layer = serialize_a16_row_shards(
        &expected_layer,
        usize::from(output_rows),
        usize::from(columns),
        &output,
    );
    let layer_error = invocation_outputs
        .iter()
        .enumerate()
        .map(|(invocation, actual)| {
            let error = verify_linear_f16(&actual[diagnostic_bytes..], &expected_layer);
            info!(
                invocation,
                error,
                stage = "output",
                "SigLIP invocation error"
            );
            error
        })
        .fold(0.0, f32::max);
    let cosine_similarity = full_model.then(|| {
        invocation_outputs
            .iter()
            .map(|actual| cosine_similarity_f16(&actual[diagnostic_bytes..], &expected_layer))
            .fold(f64::INFINITY, f64::min)
    });
    let limit = env_f32("IPU_F16_MAX_ERROR", 0.2);
    info!(
        ?norm2_error,
        ?mlp_gelu_error,
        layer_error,
        ?cosine_similarity,
        limit,
        "SigLIP verification results"
    );
    if let Some(error) = norm2_error {
        assert!(error <= limit, "norm2 max error {error} exceeds {limit}");
    }
    if let Some(error) = mlp_gelu_error {
        assert!(error <= limit, "MLP GeLU max error {error} exceeds {limit}");
    }
    if let Some(cosine_similarity) = cosine_similarity {
        let minimum = env_f32("IPU_SIGLIP_MIN_COSINE", 0.995);
        assert!(
            cosine_similarity >= f64::from(minimum),
            "pooler output cosine {cosine_similarity} is below {minimum}"
        );
    } else {
        assert!(
            layer_error <= limit,
            "encoder layer max error {layer_error} exceeds {limit}"
        );
    }
    info!(
        image_size = config.image_size,
        patch_size = config.patch_size,
        patch_grid = model.patch_grid(),
        discarded_pixels = config.image_size % config.patch_size,
        batch_size,
        sequence_length,
        rows,
        inner,
        columns,
        intermediate_columns,
        row_block_dimension,
        layer_count,
        invocations,
        ?norm2_error,
        ?mlp_gelu_error,
        layer_error,
        ?cosine_similarity,
        full_model,
        "SigLIP encoder prefix passed against Hugging Face"
    );
}

fn phase_entry_counts(phases: &[Phase]) -> (usize, usize) {
    phases
        .iter()
        .fold((0usize, 0usize), |(compute, exchange), phase| match phase {
            Phase::Compute { commands, .. } => (compute + commands.len(), exchange),
            Phase::Exchange { transfers } => (compute, exchange + transfers.len()),
        })
}

fn run_map_only(model: &SiglipWeights, reference: &TensorArchive) {
    let config = &model.config;
    let rows = u16::try_from(model.sequence_length()).unwrap();
    let columns = u16::try_from(config.hidden_size).unwrap();
    let row_block_dimension = choose_gemm_row_block_for_shape(
        rows,
        columns,
        INNER_BLOCK_DIMENSION,
        columns,
        BLOCK_DIMENSION,
        TILE_COUNT,
        GemmDataType::F16,
    )
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
    let attention_variant =
        consolidate_attention_kernel_variants(&mut schedule, std::slice::from_ref(&attention));
    let phase_count = schedule.phases.len();
    specialize_gemm_row_operations(&mut schedule, 0..phase_count);
    let objects = compile_objects(&schedule, &[attention], attention_variant).unwrap();
    let HostTensorSet {
        bindings: host_inputs,
        bytes: host_input,
        resident_bindings: host_weights,
        resident_bytes: host_weight_bytes,
    } = host;
    let invocations = env_u32("IPU_SIGLIP_INVOCATIONS", 1);
    let mut host_input = host_input.repeat(invocations as usize);
    host_input.extend(host_weight_bytes);
    let graph = ExecutableGraph {
        memory_policy: None,
        host_weights,
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
    write_memory_profile(&graph);
    let app = package_graph_repeated(&graph, &objects, invocations).unwrap();
    if let Some(path) = std::env::var_os("IPU_SIGLIP_PACKAGE_OUTPUT") {
        app.write(fs::File::create(path).unwrap()).unwrap();
    }
    if let Some(path) = std::env::var_os("IPU_SIGLIP_INPUT_OUTPUT") {
        fs::write(path, &host_input).unwrap();
    }
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
    let invocation_output_bytes = actual.len() / invocations as usize;
    assert_eq!(
        actual.len(),
        invocation_output_bytes * invocations as usize,
        "runtime returned a partial invocation output"
    );
    let invocation_outputs = actual
        .chunks_exact(invocation_output_bytes)
        .collect::<Vec<_>>();
    let expected = serialize_a16_row_shards(
        &expected,
        usize::from(output_rows),
        usize::from(columns),
        &output,
    );
    let error = invocation_outputs
        .iter()
        .map(|actual| verify_linear_f16(actual, &expected))
        .fold(0.0, f32::max);
    let limit = env_f32("IPU_F16_MAX_ERROR", 0.2);
    info!(error, limit, "SigLIP MAP verification result");
    assert!(error <= limit, "MAP max error {error} exceeds {limit}");
}

fn write_memory_profile(graph: &ExecutableGraph) {
    let Some(path) = std::env::var_os("IPU_MEMORY_PROFILE_OUTPUT") else {
        return;
    };
    let profile = allocator_memory_profile(graph).unwrap();
    profile.write(fs::File::create(&path).unwrap()).unwrap();
    info!(
        path = %PathBuf::from(path).display(),
        tiles = profile.tiles.len(),
        "wrote SigLIP allocator memory profile"
    );
}

fn encoder_memory_policy(data_limit: u32) -> MemoryPolicy {
    let resident = memory_region_order(
        "IPU_SIGLIP_RESIDENT_ORDER",
        &[
            Ipu21MemoryRegion::OrdinaryHigh,
            Ipu21MemoryRegion::OrdinaryLow,
            Ipu21MemoryRegion::Interleaved,
        ],
    );
    let transient = memory_region_order(
        "IPU_SIGLIP_TRANSIENT_ORDER",
        &[
            Ipu21MemoryRegion::OrdinaryLow,
            Ipu21MemoryRegion::Interleaved,
            Ipu21MemoryRegion::OrdinaryHigh,
        ],
    );
    MemoryPolicy::ipu21(ORDINARY_LOW_BASE, data_limit, &resident, &transient).unwrap()
}

fn memory_region_order(name: &str, default: &[Ipu21MemoryRegion]) -> Vec<Ipu21MemoryRegion> {
    let Ok(value) = std::env::var(name) else {
        return default.to_vec();
    };
    let regions = value
        .split(',')
        .map(|region| match region.trim() {
            "ordinary-low" => Ipu21MemoryRegion::OrdinaryLow,
            "interleaved" => Ipu21MemoryRegion::Interleaved,
            "ordinary-high" => Ipu21MemoryRegion::OrdinaryHigh,
            region => panic!("invalid IPU21 SRAM region {region}"),
        })
        .collect::<Vec<_>>();
    assert!(!regions.is_empty(), "IPU21 SRAM region order is empty");
    regions
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
            specialization: Arc::new(SpecializationKey {
                operation: "add_f16".into(),
                shape: vec![usize::from(output.rows), usize::from(output.columns)],
                worker_count: 6,
                role: "patch-position-bias".into(),
                alignment: 4,
            }),
            metadata: BTreeMap::from([
                ("label".into(), "patch bias and learned position".into()),
                ("row_start".into(), output.row_start.to_string()),
                ("column_start".into(), output.column_start.to_string()),
            ]),
        });
    }
    schedule.phases.push(Phase::Compute {
        op: OpId(phase),
        commands: commands.into_iter().map(Arc::new).collect(),
    });
    Ok(placements)
}

fn compile_objects(
    schedule: &ipu_compiler::Schedule,
    attentions: &[FlashAttentionPlan],
    attention_variant: ipu_runtime::AttentionKernelVariant,
) -> ipu_runtime::Result<Vec<Vec<u8>>> {
    let sdk = PathBuf::from(required_env("POPLAR_SDK_ENABLED"));
    let artifacts = std::env::temp_dir().join(format!("ipu-siglip-patch-{}", std::process::id()));
    let source = |name: &str| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../device")
            .join(name)
    };
    let toolchain = Toolchain::from_sdk(sdk);
    let runtime = toolchain.compile(source("static_runtime.S"), &artifacts, "runtime", &[])?;
    let gemm_variants = compile_gemm_row_variants(&toolchain, &artifacts, schedule)?;
    let uses_fp8_weights = schedule.phases.iter().any(|phase| {
        matches!(phase, Phase::Compute { commands, .. } if commands.iter().any(|command| {
            command.specialization.operation == "expand_f8_f143_to_f16"
        }))
    });
    let fp8_expander = uses_fp8_weights
        .then(|| {
            Ok::<_, ipu_elf::ElfError>((
                toolchain.compile(
                    source("expand_f8_f143_to_f16.cpp"),
                    &artifacts,
                    "expand-f8-codelet",
                    &[],
                )?,
                toolchain.compile(
                    source("expand_f8_f143_to_f16.S"),
                    &artifacts,
                    "expand-f8-wrapper",
                    &[],
                )?,
            ))
        })
        .transpose()?;
    let uses_native_fp8 = schedule.phases.iter().any(|phase| {
        matches!(phase, Phase::Compute { commands, .. } if commands.iter().any(|command| {
            command.specialization.operation.starts_with("gemm_f8_")
        }))
    });
    let native_fp8 = uses_native_fp8
        .then(|| {
            Ok::<_, ipu_elf::ElfError>((
                toolchain.compile(
                    source("quantize_a16_to_a32_f143.cpp"),
                    &artifacts,
                    "quantize-a16-to-a32-f143-codelet",
                    &[],
                )?,
                toolchain.compile(
                    source("quantize_a16_to_a32_f143.S"),
                    &artifacts,
                    "quantize-a16-to-a32-f143-wrapper",
                    &[],
                )?,
            ))
        })
        .transpose()?;
    let add = toolchain.compile(source("add_f16.S"), &artifacts, "add-f16", &[])?;
    let add_bias = toolchain.compile(source("add_bias_f16.S"), &artifacts, "add-bias-f16", &[])?;
    let relayout = toolchain.compile(source("relayout_f16.S"), &artifacts, "relayout-f16", &[])?;
    let reblock = toolchain.compile(source("reblock_f16.S"), &artifacts, "reblock-f16", &[])?;
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
    let attention = &attentions[0];
    assert!(attentions.iter().all(|candidate| {
        candidate.head_dimension == attention.head_dimension
            && candidate.padded_head_dimension == attention.padded_head_dimension
            && candidate.key_block_columns == attention.key_block_columns
    }));
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
            format!(
                "-DATTENTION_KEY_BLOCK_COLUMNS={}",
                attention.key_block_columns
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
    let attention_objects =
        compile_attention_variant(&toolchain, &artifacts, attention, attention_variant)?;
    let worker_support = toolchain.compile(
        source("worker_support.S"),
        &artifacts,
        "worker-support",
        &[],
    )?;
    let mut objects = vec![
        fs::read(runtime.object)?,
        fs::read(add.object)?,
        fs::read(add_bias.object)?,
        fs::read(relayout.object)?,
        fs::read(reblock.object)?,
        fs::read(gelu_relayout.object)?,
        fs::read(norm_codelet.object)?,
        fs::read(norm_wrapper.object)?,
        fs::read(pack_codelet.object)?,
        fs::read(pack_wrapper.object)?,
        fs::read(unpack_codelet.object)?,
        fs::read(unpack_wrapper.object)?,
        fs::read(worker_support.object)?,
    ];
    for object in gemm_variants {
        objects.push(fs::read(object.object)?);
    }
    for object in attention_objects {
        objects.push(fs::read(object.object)?);
    }
    if let Some((codelet, wrapper)) = fp8_expander {
        objects.push(fs::read(codelet.object)?);
        objects.push(fs::read(wrapper.object)?);
    }
    if let Some((codelet, wrapper)) = native_fp8 {
        objects.push(fs::read(codelet.object)?);
        objects.push(fs::read(wrapper.object)?);
    }
    Ok(objects)
}

fn specialize_gemm_row_operations(
    schedule: &mut ipu_compiler::Schedule,
    phases: std::ops::Range<usize>,
) {
    for phase in &mut schedule.phases[phases] {
        let Phase::Compute { commands, .. } = phase else {
            continue;
        };
        for command in commands {
            let command = Arc::make_mut(command);
            let operation = command.specialization.operation.as_ref();
            let Some(base) = operation
                .strip_suffix("_small_rows")
                .or_else(|| operation.strip_suffix("_large_rows"))
            else {
                continue;
            };
            if !base.starts_with("gemm_") {
                continue;
            }
            let rows = command
                .specialization
                .shape
                .first()
                .copied()
                .expect("GEMM row specialization requires its block shape");
            let inner = command
                .specialization
                .shape
                .get(1)
                .copied()
                .expect("GEMM specialization requires its inner block shape");
            let output = command
                .specialization
                .shape
                .get(2)
                .copied()
                .expect("GEMM specialization requires its output block shape");
            let specialized = format!("{base}_rows_{rows}_inner_{inner}_output_{output}");
            Arc::make_mut(&mut command.specialization).operation = specialized.into();
        }
    }
}

fn compile_gemm_row_variants(
    toolchain: &Toolchain,
    artifacts: &std::path::Path,
    schedule: &ipu_compiler::Schedule,
) -> Result<Vec<KernelArtifact>, ipu_elf::ElfError> {
    let mut variants = BTreeSet::<(String, u16, u16, u16)>::new();
    for phase in &schedule.phases {
        let Phase::Compute { commands, .. } = phase else {
            continue;
        };
        for command in commands {
            let operation = command.specialization.operation.as_ref();
            let Some((inner_operation, output)) = operation.rsplit_once("_output_") else {
                continue;
            };
            let Some((row_operation, inner)) = inner_operation.rsplit_once("_inner_") else {
                continue;
            };
            let Some((base, rows)) = row_operation.rsplit_once("_rows_") else {
                continue;
            };
            let family = if base.starts_with("gemm_f16_f8w_") {
                "gemm_f16_f8w"
            } else if base.starts_with("gemm_f16_") {
                "gemm_f16"
            } else if base.starts_with("gemm_f8_") {
                "gemm_f8"
            } else {
                continue;
            };
            let rows = rows.parse::<u16>().map_err(|_| {
                ipu_elf::ElfError::Link(format!("invalid GEMM row operation {operation}"))
            })?;
            let inner = inner.parse::<u16>().map_err(|_| {
                ipu_elf::ElfError::Link(format!("invalid GEMM inner operation {operation}"))
            })?;
            let output = output.parse::<u16>().map_err(|_| {
                ipu_elf::ElfError::Link(format!("invalid GEMM output operation {operation}"))
            })?;
            variants.insert((family.into(), rows, inner, output));
        }
    }
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../device/gemm_f16_64_amp.S");
    variants
        .into_iter()
        .map(|(family, rows, inner, output)| {
            let mut flags = vec![
                format!("-DGEMM_INNER_BLOCK_DIMENSION={inner}"),
                format!("-DGEMM_OUTPUT_COLUMNS={output}"),
                format!("-DGEMM_SMALL_ROWS={rows}"),
                "-DGEMM_SINGLE_ROWS=1".into(),
                format!(
                    "-DGEMM_INIT_SMALL_SYMBOL=ipu_stack_{family}_init_rows_{rows}_inner_{inner}_output_{output}"
                ),
                format!(
                    "-DGEMM_ACCUMULATE_SMALL_SYMBOL=ipu_stack_{family}_accumulate_rows_{rows}_inner_{inner}_output_{output}"
                ),
            ];
            match family.as_str() {
                "gemm_f16" => {}
                "gemm_f16_f8w" => flags.push("-DGEMM_INTERLEAVED_WEIGHTS=1".into()),
                "gemm_f8" => flags.push("-DGEMM_NATIVE_FP8=1".into()),
                _ => unreachable!(),
            }
            toolchain.compile(
                &source,
                artifacts,
                &format!("{family}-rows-{rows}-inner-{inner}-output-{output}"),
                &flags,
            )
        })
        .collect()
}

fn compile_attention_variant(
    toolchain: &Toolchain,
    artifacts: &std::path::Path,
    shape: &FlashAttentionPlan,
    variant: ipu_runtime::AttentionKernelVariant,
) -> Result<Vec<KernelArtifact>, ipu_elf::ElfError> {
    let source = |name: &str| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../device")
            .join(name)
    };
    let softmax = [
        (
            "AttentionSoftmaxSmallQuerySmallKeyF16",
            "attention_softmax_small_query_small_key_f16",
        ),
        (
            "AttentionSoftmaxSmallQueryLargeKeyF16",
            "attention_softmax_small_query_large_key_f16",
        ),
        (
            "AttentionSoftmaxLargeQuerySmallKeyF16",
            "attention_softmax_large_query_small_key_f16",
        ),
        (
            "AttentionSoftmaxLargeQueryLargeKeyF16",
            "attention_softmax_large_query_large_key_f16",
        ),
    ];
    let merge = [
        (
            "AttentionMergeSmallQuerySingleBlockF16",
            "attention_merge_small_query_single_block_f16",
        ),
        (
            "AttentionMergeSmallQueryInitialBlockF16",
            "attention_merge_small_query_initial_block_f16",
        ),
        (
            "AttentionMergeSmallQueryMiddleBlockF16",
            "attention_merge_small_query_middle_block_f16",
        ),
        (
            "AttentionMergeSmallQueryFinalBlockF16",
            "attention_merge_small_query_final_block_f16",
        ),
        (
            "AttentionMergeLargeQuerySingleBlockF16",
            "attention_merge_large_query_single_block_f16",
        ),
        (
            "AttentionMergeLargeQueryInitialBlockF16",
            "attention_merge_large_query_initial_block_f16",
        ),
        (
            "AttentionMergeLargeQueryMiddleBlockF16",
            "attention_merge_large_query_middle_block_f16",
        ),
        (
            "AttentionMergeLargeQueryFinalBlockF16",
            "attention_merge_large_query_final_block_f16",
        ),
    ];
    let mut objects = Vec::with_capacity(4);
    {
        let suffix = variant.suffix();
        let class_suffix = format!(
            "Q{}_{}K{}_{}",
            variant.small_query_rows,
            variant.large_query_rows,
            variant.small_key_rows,
            variant.large_key_rows
        );
        let mut codelet_flags = vec![
            format!("-DATTENTION_HEAD_DIMENSION={}", shape.head_dimension),
            format!(
                "-DATTENTION_PADDED_HEAD_DIMENSION={}",
                shape.padded_head_dimension
            ),
            format!("-DATTENTION_KEY_BLOCK_COLUMNS={}", shape.key_block_columns),
            format!("-DATTENTION_SMALL_QUERY_ROWS={}", variant.small_query_rows),
            format!("-DATTENTION_LARGE_QUERY_ROWS={}", variant.large_query_rows),
            format!("-DATTENTION_SMALL_KEY_ROWS={}", variant.small_key_rows),
            format!("-DATTENTION_LARGE_KEY_ROWS={}", variant.large_key_rows),
        ];
        let mut wrapper_flags = Vec::new();
        for &(class, symbol) in softmax.iter().chain(&merge) {
            let renamed = format!("{class}{class_suffix}");
            codelet_flags.push(format!("-D{class}={renamed}"));
            wrapper_flags.push(format!("-D__runCodelet_{class}=__runCodelet_{renamed}"));
            wrapper_flags.push(format!("-Dipu_stack_{symbol}=ipu_stack_{symbol}_{suffix}"));
        }
        wrapper_flags.push(format!(
            "-Dipu_stack_attention_f32_to_f16=ipu_stack_attention_f32_to_f16_{suffix}"
        ));
        objects.push(toolchain.compile(
            source("flash_attention_f16.cpp"),
            artifacts,
            &format!("flash-attention-codelet-{suffix}"),
            &codelet_flags,
        )?);
        objects.push(toolchain.compile(
            source("flash_attention_f16.S"),
            artifacts,
            &format!("flash-attention-wrapper-{suffix}"),
            &wrapper_flags,
        )?);
        for (kind, inner, output) in [
            ("qk", shape.padded_head_dimension, shape.key_block_columns),
            ("pv", shape.key_block_columns, shape.padded_head_dimension),
        ] {
            objects.push(toolchain.compile(
                source("gemm_f16_64_amp.S"),
                artifacts,
                &format!("attention-{kind}-{suffix}"),
                &[
                    format!("-DGEMM_INNER_BLOCK_DIMENSION={inner}"),
                    format!("-DGEMM_OUTPUT_COLUMNS={output}"),
                    format!("-DGEMM_SMALL_ROWS={}", variant.small_query_rows),
                    format!("-DGEMM_LARGE_ROWS={}", variant.large_query_rows),
                    format!("-DGEMM_INIT_SMALL_SYMBOL=ipu_stack_attention_{kind}_init_small_rows_{suffix}"),
                    format!("-DGEMM_INIT_LARGE_SYMBOL=ipu_stack_attention_{kind}_init_large_rows_{suffix}"),
                    format!("-DGEMM_ACCUMULATE_SMALL_SYMBOL=ipu_stack_attention_{kind}_accumulate_small_rows_{suffix}"),
                    format!("-DGEMM_ACCUMULATE_LARGE_SYMBOL=ipu_stack_attention_{kind}_accumulate_large_rows_{suffix}"),
                ],
            )?);
        }
    }
    Ok(objects)
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

fn cosine_similarity_f16(actual: &[u8], expected: &[f32]) -> f64 {
    assert!(actual.len() >= expected.len() * 2);
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    for (index, &expected) in expected.iter().enumerate() {
        let actual = f16::from_bits(u16::from_le_bytes(
            actual[index * 2..index * 2 + 2].try_into().unwrap(),
        ))
        .to_f32();
        assert!(actual.is_finite());
        let actual = f64::from(actual);
        let expected = f64::from(expected);
        dot += actual * expected;
        actual_norm += actual * actual;
        expected_norm += expected * expected;
    }
    assert!(actual_norm > 0.0 && expected_norm > 0.0);
    dot / (actual_norm * expected_norm).sqrt()
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

fn siglip_encoder_precision(storage: SiglipWeightStorage) -> SiglipEncoderPrecision {
    let fallback = match storage {
        SiglipWeightStorage::F16 => SiglipLinearPrecision::F16,
        SiglipWeightStorage::F143 => SiglipLinearPrecision::F143Expanded,
    };
    SiglipEncoderPrecision {
        qkv: linear_precision("IPU_SIGLIP_QKV", fallback),
        attention_output: linear_precision("IPU_SIGLIP_ATTENTION_OUTPUT", fallback),
        mlp_up: linear_precision("IPU_SIGLIP_MLP_UP", fallback),
        mlp_down: linear_precision("IPU_SIGLIP_MLP_DOWN", fallback),
    }
}

fn expanded_storage_fallback(precision: SiglipEncoderPrecision) -> SiglipEncoderPrecision {
    let fallback = |linear| match linear {
        SiglipLinearPrecision::F16 => SiglipLinearPrecision::F143Expanded,
        other => other,
    };
    SiglipEncoderPrecision {
        qkv: fallback(precision.qkv),
        attention_output: fallback(precision.attention_output),
        mlp_up: fallback(precision.mlp_up),
        mlp_down: fallback(precision.mlp_down),
    }
}

fn linear_precision(prefix: &str, fallback: SiglipLinearPrecision) -> SiglipLinearPrecision {
    let name = format!("{prefix}_PRECISION");
    match std::env::var(&name).as_deref() {
        Err(_) => fallback,
        Ok("f16") => SiglipLinearPrecision::F16,
        Ok("f143-expanded") => SiglipLinearPrecision::F143Expanded,
        Ok("f143-native") => {
            let scale_name = format!("{prefix}_ACTIVATION_SCALE");
            let activation_scale = std::env::var(&scale_name)
                .unwrap_or_else(|_| panic!("{scale_name} is required for native FP8"))
                .parse()
                .unwrap_or_else(|_| panic!("{scale_name} must be an i8"));
            SiglipLinearPrecision::F143Native { activation_scale }
        }
        Ok(value) => panic!("unsupported precision {value:?} in {name}"),
    }
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .map(|value| value.parse().unwrap())
        .unwrap_or(default)
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .map(|value| value.parse().unwrap())
        .unwrap_or(default)
}
