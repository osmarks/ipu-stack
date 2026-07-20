use half::f16;
use ipu_compiler::{
    Allocation, AllocationKind, AppendAffineLayerNormConfig, BlockPlacement, BlockedGemmConfig,
    GemmDataType, KernelCommand, OpId, Phase, RowShardPlacement, RowShardTransitionConfig,
    SpecializationKey, TensorId, append_affine_layer_norm_f16, append_c16_to_a16_row_shards,
    choose_gemm_row_block_for, plan_blocked_gemm,
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

    let adjustment = append_adjustment_phase(&mut plan).unwrap();
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
    let objects = compile_objects(&plan).unwrap();
    let graph = ExecutableGraph {
        schedule: plan.schedule,
        initial_buffers: Vec::new(),
        outputs: Vec::new(),
        host_inputs: vec![
            block_binding_typed("patches", rows, inner, &plan.left, "f16", 2),
            block_binding_typed("patch_weight", inner, columns, &plan.right, "f16", 2),
            block_binding_typed("position_bias", rows, columns, &adjustment, "f16", 2),
            row_shard_binding("layer_norm1_affine", 2, columns, &norm.affine),
        ],
        host_outputs: vec![row_shard_binding(
            "patch_and_position",
            rows,
            columns,
            &norm.output,
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
    let embedding = reference.tensor_f32("patch_and_position").unwrap();
    let expected = layer_norm_reference(
        &embedding,
        usize::from(rows),
        usize::from(columns),
        &norm_weight,
        &norm_bias,
        config.layer_norm_eps,
    );
    let max_error = verify(&actual, &norm.output, columns, &expected);
    let limit = env_f32("IPU_F16_MAX_ERROR", 0.1);
    assert!(
        max_error <= limit,
        "patch embedding max error {max_error} exceeds {limit}"
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
        "SigLIP patch embedding and encoder layer norm passed against Hugging Face"
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
    plan: &mut ipu_compiler::BlockedGemmPlan,
) -> ipu_runtime::Result<Vec<BlockPlacement>> {
    let mut cursors = vec![DATA_BASE; usize::from(TILE_COUNT)];
    let mut maximum_tensor = 0usize;
    for allocation in &plan.schedule.allocations {
        maximum_tensor = maximum_tensor.max(allocation.tensor.0);
        if allocation.kind == AllocationKind::Home && allocation.address >= DATA_BASE {
            let end = allocation
                .address
                .checked_add(allocation.size)
                .ok_or("SRAM overflow")?;
            cursors[usize::from(allocation.tile)] = cursors[usize::from(allocation.tile)].max(end);
        }
    }
    let phase = plan.schedule.phases.len();
    let mut placements = Vec::with_capacity(plan.output.len());
    let mut commands = Vec::with_capacity(plan.output.len());
    for (index, output) in plan.output.iter().enumerate() {
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
        plan.schedule.allocations.push(Allocation {
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
    plan.schedule.phases.push(Phase::Compute {
        op: OpId(phase),
        commands,
    });
    Ok(placements)
}

fn compile_objects(plan: &ipu_compiler::BlockedGemmPlan) -> ipu_runtime::Result<Vec<Vec<u8>>> {
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
        &[],
    )?;
    let norm_wrapper = toolchain.compile(
        source("layer_norm_f16.S"),
        &artifacts,
        "layer-norm-wrapper",
        &[],
    )?;
    Ok(vec![
        fs::read(runtime.object)?,
        fs::read(gemm.object)?,
        fs::read(add.object)?,
        fs::read(relayout.object)?,
        fs::read(norm_codelet.object)?,
        fs::read(norm_wrapper.object)?,
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

fn verify(actual: &[u8], output: &[RowShardPlacement], columns: u16, expected: &[f32]) -> f32 {
    let mut offset = 0usize;
    let mut max_error = 0.0f32;
    for block in output {
        for linear in 0..block.rows * block.columns {
            let panel_elements = block.rows * 16;
            let panel = linear / panel_elements;
            let panel_offset = linear % panel_elements;
            let row = panel_offset / 16;
            let column = panel * 16 + panel_offset % 16;
            let observed = f16::from_bits(u16::from_le_bytes(
                actual[offset..offset + 2].try_into().unwrap(),
            ))
            .to_f32();
            let expected = expected
                [usize::from(block.row_start + row) * usize::from(columns) + usize::from(column)];
            assert!(observed.is_finite());
            max_error = max_error.max((observed - expected).abs());
            offset += 2;
        }
    }
    max_error
}

fn layer_norm_reference(
    input: &[f32],
    rows: usize,
    columns: usize,
    weight: &[f32],
    bias: &[f32],
    epsilon: f32,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(columns).take(rows) {
        let mean = row.iter().sum::<f32>() / columns as f32;
        let variance = row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / columns as f32;
        let scale = (variance + epsilon).sqrt().recip();
        output.extend(
            row.iter()
                .zip(weight)
                .zip(bias)
                .map(|((&value, &weight), &bias)| (value - mean) * scale * weight + bias),
        );
    }
    output
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .map(|value| value.parse().unwrap())
        .unwrap_or(default)
}
