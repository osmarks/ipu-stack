use ipu_compiler::{BlockedMlpConfig, choose_gemm_row_block, plan_blocked_mlp};
use ipu_elf::Toolchain;
use ipu_runtime::{
    BlockLayout, ExecutableGraph, HostRunOptions, ProfileGranularity, allocator_memory_profile,
    block_binding, block_coordinates, blocked_matrix, package_graph, package_graph_profiled_with,
    run_host_with_options,
};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

const TILE_COUNT: u16 = 1472;
const BLOCK_DIMENSION: u16 = 64;
const INNER_BLOCK_DIMENSION: u16 = 64;
const MLP_DATA_BASE: u32 = 0xa0000;

fn main() {
    ipu_runtime::init_tracing();
    let batch = env_u16("IPU_MLP_BATCH", 512);
    let width = env_u16("IPU_MLP_WIDTH", 2048);
    let layers = env_u16("IPU_MLP_LAYERS", 8);
    let row_block_dimension = std::env::var("IPU_MLP_ROW_BLOCK")
        .map(|value| {
            value
                .parse::<u16>()
                .expect("IPU_MLP_ROW_BLOCK must be a u16")
        })
        .unwrap_or_else(|_| {
            choose_gemm_row_block(
                batch,
                INNER_BLOCK_DIMENSION,
                width,
                BLOCK_DIMENSION,
                TILE_COUNT,
            )
            .expect("MLP shape has no feasible row blocking")
        });
    let sdk = PathBuf::from(required_env("POPLAR_SDK_ENABLED"));
    let configuration = fs::read(required_env("IPU_CONFIG")).unwrap();
    let bootloader = fs::read(
        std::env::var_os("IPU_BOOTLOADER")
            .map(PathBuf::from)
            .unwrap_or_else(|| sdk.join("bin/ipu/tile_bootloader_ipu2.elf")),
    )
    .unwrap();
    let device = std::env::var("IPU_DEVICE").unwrap_or_else(|_| "/dev/ipu0".into());
    let artifact_dir = std::env::temp_dir().join(format!("ipu-stack-mlp-{}", std::process::id()));
    let toolchain = Toolchain::from_sdk(sdk);
    let source = |name: &str| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../device")
            .join(name)
    };
    let runtime = toolchain
        .compile(
            source("static_runtime.S"),
            &artifact_dir,
            "mlp-runtime",
            &[],
        )
        .unwrap();
    let gemm = toolchain
        .compile(
            source("gemm_f32_64_amp.S"),
            &artifact_dir,
            "mlp-gemm",
            &[format!(
                "-DGEMM_INNER_BLOCK_DIMENSION={INNER_BLOCK_DIMENSION}"
            )],
        )
        .unwrap();
    let gelu = toolchain
        .compile(
            source("gelu_relayout_f32.S"),
            &artifact_dir,
            "mlp-gelu",
            &[],
        )
        .unwrap();

    let compile_start = Instant::now();
    let plan = plan_blocked_mlp(BlockedMlpConfig {
        batch,
        width,
        layers,
        block_dimension: BLOCK_DIMENSION,
        inner_block_dimension: INNER_BLOCK_DIMENSION,
        row_block_dimension,
        tile_count: TILE_COUNT,
        data_base: MLP_DATA_BASE,
        data_limit: ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
    })
    .unwrap();
    let output_placements = plan.output.clone();
    let input_binding = block_binding("input", batch, width, &plan.input);
    let mut host_inputs = vec![input_binding];
    let mut input = blocked_matrix(&plan.input, BlockLayout::AmpA8, input_value);
    for (layer, placements) in plan.weights.iter().enumerate() {
        host_inputs.push(block_binding(
            &format!("layer_{layer}_weight"),
            width,
            width,
            placements,
        ));
        input.extend(blocked_matrix(
            placements,
            BlockLayout::AmpB8x16,
            |row, column| weight_value(layer, row, column),
        ));
    }
    let output_binding = block_binding("output", batch, width, &plan.output);
    let graph = ExecutableGraph {
        schedule: plan.schedule,
        initial_buffers: Vec::new(),
        outputs: Vec::new(),
        host_inputs,
        host_outputs: vec![output_binding],
    };
    if let Some(path) = std::env::var_os("IPU_MEMORY_PROFILE_OUTPUT") {
        let profile = allocator_memory_profile(&graph).unwrap();
        profile.write(fs::File::create(&path).unwrap()).unwrap();
        info!(path = %PathBuf::from(path).display(), tiles = profile.tiles.len(), "wrote MLP allocator profile");
    }
    let objects = [
        fs::read(runtime.object).unwrap(),
        fs::read(gemm.object).unwrap(),
        fs::read(gelu.object).unwrap(),
    ];
    let profile_output = std::env::var_os("IPU_PROFILE_OUTPUT").map(PathBuf::from);
    let (app, profile_layout) = if profile_output.is_some() {
        let granularity = ProfileGranularity::from_environment().unwrap();
        let (app, layout) = package_graph_profiled_with(&graph, &objects, granularity).unwrap();
        info!(?granularity, "enabled cycle profiling");
        (app, Some(layout))
    } else {
        (package_graph(&graph, &objects).unwrap(), None)
    };
    info!(
        batch,
        width,
        layers,
        row_block_dimension,
        phases = graph.schedule.phases.len(),
        compile_ms = compile_start.elapsed().as_millis(),
        input_bytes = input.len(),
        "packaged FP32 MLP"
    );
    if let Some(path) = std::env::var_os("IPU_MLP_PACKAGE") {
        app.write(fs::File::create(path).unwrap()).unwrap();
    }
    if std::env::var_os("IPU_MLP_PACKAGE_ONLY").is_some() {
        let _ = fs::remove_dir_all(artifact_dir);
        return;
    }

    let start = Instant::now();
    let actual = run_host_with_options(
        &app,
        &bootloader,
        &configuration,
        &device,
        &input,
        HostRunOptions::from_environment().unwrap(),
    )
    .unwrap();
    verify_output(batch, width, layers, &actual, &output_placements).unwrap();
    if let (Some(path), Some(layout)) = (&profile_output, &profile_layout) {
        let clock_hz = std::env::var("IPU_CLOCK_HZ")
            .map(|value| value.parse().expect("IPU_CLOCK_HZ must be an integer"))
            .unwrap_or(1_500_000_000);
        let report = layout.decode(&actual, clock_hz).unwrap();
        let graph_cycles = report
            .tiles
            .iter()
            .filter_map(|tile| {
                Some(
                    tile.samples
                        .last()?
                        .end_cycle
                        .wrapping_sub(tile.samples.first()?.start_cycle),
                )
            })
            .max()
            .unwrap_or(0);
        let graph_seconds = f64::from(graph_cycles) / clock_hz as f64;
        let gemm_flops = 2.0 * f64::from(batch) * f64::from(width).powi(2) * f64::from(layers);
        report.write(fs::File::create(path).unwrap()).unwrap();
        info!(
            path = %path.display(),
            graph_cycles,
            graph_ms = graph_seconds * 1.0e3,
            gemm_tflops = (graph_seconds != 0.0).then_some(gemm_flops / graph_seconds / 1.0e12),
            "wrote MLP cycle profile"
        );
    }
    info!(
        batch,
        width,
        layers,
        elapsed_ms = start.elapsed().as_millis(),
        output_bytes = actual.len(),
        "FP32 MLP passed"
    );
    let _ = fs::remove_dir_all(artifact_dir);
}

fn verify_output(
    batch: u16,
    width: u16,
    layers: u16,
    actual: &[u8],
    placements: &[ipu_compiler::BlockPlacement],
) -> ipu_runtime::Result<()> {
    let expected_bytes = usize::from(batch) * usize::from(width) * 4;
    let actual = actual
        .get(..expected_bytes)
        .ok_or("MLP output is shorter than the activation matrix")?;
    let mut offset = 0usize;
    for placement in placements {
        for linear in 0..placement.rows * placement.columns {
            let (row, column) = block_coordinates(
                BlockLayout::AmpA8,
                placement.rows,
                placement.columns,
                linear,
            );
            let row = placement.row_start + row;
            let column = placement.column_start + column;
            let observed = f32::from_le_bytes(actual[offset..offset + 4].try_into().unwrap());
            let expected = expected_value(layers, row, column);
            let tolerance = 2.0e-5_f32.max(expected.abs() * 2.0e-5);
            if !observed.is_finite() || (observed - expected).abs() > tolerance {
                return Err(format!(
                    "MLP mismatch at [{row}, {column}]: observed={observed:?}, expected={expected:?}, tolerance={tolerance:?}"
                )
                .into());
            }
            offset += 4;
        }
    }
    Ok(())
}

fn input_value(row: u16, column: u16) -> f32 {
    f32::from(((3 * row + column) % 17) as i8 - 8) / 16.0
}

fn layer_scale(layer: usize) -> f32 {
    0.75 + layer as f32 / 32.0
}

fn weight_value(layer: usize, row: u16, column: u16) -> f32 {
    if row == column {
        layer_scale(layer)
    } else {
        0.0
    }
}

fn expected_value(layers: u16, row: u16, column: u16) -> f32 {
    (0..usize::from(layers)).fold(input_value(row, column), |value, layer| {
        gelu(value * layer_scale(layer))
    })
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044715 * value * value * value)).tanh())
}

fn env_u16(name: &str, default: u16) -> u16 {
    std::env::var(name)
        .map(|value| {
            value
                .parse::<u16>()
                .unwrap_or_else(|_| panic!("{name} must be a u16"))
        })
        .unwrap_or(default)
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"))
}
