use ipu_compiler::{BlockedGemmConfig, BlockedGemmPlan, plan_blocked_gemm};
use ipu_elf::Toolchain;
use ipu_package::{Binding, RegionSlice};
use ipu_runtime::{ExecutableGraph, package_graph, package_graph_profiled, run_host};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

const TILE_COUNT: u16 = 1472;
const BLOCK_DIMENSION: u16 = 64;
// The 12,288 static host attachment phases occupy the lower runtime arena.
const GEMM_DATA_BASE: u32 = 0xa0000;

fn main() {
    ipu_runtime::init_tracing();
    let dimension = std::env::var("IPU_GEMM_DIMENSION")
        .map(|value| {
            value
                .parse::<u16>()
                .expect("IPU_GEMM_DIMENSION must be a u16")
        })
        .unwrap_or(2048);
    let sdk = PathBuf::from(required_env("POPLAR_SDK_ENABLED"));
    let configuration = fs::read(required_env("IPU_CONFIG")).unwrap();
    let bootloader = fs::read(
        std::env::var_os("IPU_BOOTLOADER")
            .map(PathBuf::from)
            .unwrap_or_else(|| sdk.join("bin/ipu/tile_bootloader_ipu2.elf")),
    )
    .unwrap();
    let device = std::env::var("IPU_DEVICE").unwrap_or_else(|_| "/dev/ipu0".into());
    let output = std::env::temp_dir().join(format!("ipu-stack-gemm-{}", std::process::id()));
    let toolchain = Toolchain::from_sdk(sdk);
    let runtime_source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../device/static_runtime.S");
    let runtime = toolchain
        .compile(&runtime_source, &output, "gemm-runtime", &[])
        .unwrap();
    let kernel_source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../device/gemm_f32_64.S");
    let kernel = toolchain
        .compile(&kernel_source, &output, "gemm-f32-64", &[])
        .unwrap();

    let compile_start = Instant::now();
    let plan = plan_blocked_gemm(BlockedGemmConfig {
        dimension,
        block_dimension: BLOCK_DIMENSION,
        tile_count: TILE_COUNT,
        data_base: GEMM_DATA_BASE,
        data_limit: ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
    })
    .unwrap();
    let (graph, input) = gemm_graph_and_input(dimension, plan);
    let objects = [
        fs::read(runtime.object).unwrap(),
        fs::read(kernel.object).unwrap(),
    ];
    let profile_output = std::env::var_os("IPU_PROFILE_OUTPUT").map(PathBuf::from);
    let load_package = std::env::var_os("IPU_GEMM_LOAD_PACKAGE").map(PathBuf::from);
    assert!(
        profile_output.is_none() || load_package.is_none(),
        "a cached package does not carry this run's profile layout"
    );
    let (app, profile_layout) = if let Some(path) = load_package {
        (
            ipu_package::Application::read(fs::File::open(path).unwrap()).unwrap(),
            None,
        )
    } else if profile_output.is_some() {
        let (app, layout) = package_graph_profiled(&graph, &objects).unwrap();
        (app, Some(layout))
    } else {
        (package_graph(&graph, &objects).unwrap(), None)
    };
    info!(
        dimension,
        compile_ms = compile_start.elapsed().as_millis(),
        input_bytes = input.len(),
        "packaged blocked FP32 GEMM"
    );
    if let Some(path) = std::env::var_os("IPU_GEMM_PACKAGE") {
        app.write(fs::File::create(path).unwrap()).unwrap();
    }
    if std::env::var_os("IPU_GEMM_PACKAGE_ONLY").is_some() {
        let _ = fs::remove_dir_all(output);
        return;
    }

    let run_start = Instant::now();
    let actual = run_host(&app, &bootloader, &configuration, &device, &input).unwrap();
    let matrix_bytes = usize::from(dimension) * usize::from(dimension) * 4;
    verify_output(dimension, &actual[..matrix_bytes]);
    if let (Some(path), Some(layout)) = (profile_output, profile_layout) {
        let clock_hz = std::env::var("IPU_CLOCK_HZ")
            .map(|value| value.parse().expect("IPU_CLOCK_HZ must be an integer"))
            .unwrap_or(1_500_000_000);
        let report = layout.decode(&actual, clock_hz).unwrap();
        report.write(fs::File::create(&path).unwrap()).unwrap();
        info!(path = %path.display(), tiles = report.tiles.len(), "wrote cycle profile");
    }
    info!(
        dimension,
        run_ms = run_start.elapsed().as_millis(),
        output_bytes = actual.len(),
        "blocked FP32 GEMM passed"
    );
    let _ = fs::remove_dir_all(output);
}

fn gemm_graph_and_input(dimension: u16, plan: BlockedGemmPlan) -> (ExecutableGraph, Vec<u8>) {
    let block_bytes = u64::from(BLOCK_DIMENSION) * u64::from(BLOCK_DIMENSION) * 4;
    let left_binding = block_binding("left", dimension, block_bytes, &plan.left);
    let right_binding = block_binding("right", dimension, block_bytes, &plan.right);
    let output_binding = block_binding("output", dimension, block_bytes, &plan.output);
    let mut input = blocked_matrix(dimension, left_value);
    input.extend(blocked_matrix(dimension, right_value));
    (
        ExecutableGraph {
            schedule: plan.schedule,
            initial_buffers: Vec::new(),
            outputs: Vec::new(),
            host_inputs: vec![left_binding, right_binding],
            host_outputs: vec![output_binding],
        },
        input,
    )
}

fn block_binding(
    name: &str,
    dimension: u16,
    block_bytes: u64,
    placements: &[ipu_compiler::BlockPlacement],
) -> Binding {
    let topology = ipu_exchange::Topology::c600();
    Binding {
        name: name.into(),
        dtype: "f32".into(),
        shape: vec![u32::from(dimension), u32::from(dimension)],
        slices: placements
            .iter()
            .enumerate()
            .map(|(index, placement)| RegionSlice {
                tile: u32::from(topology.physical(placement.tile).unwrap()),
                tile_address: placement.address,
                file_offset: index as u64 * block_bytes,
                size: block_bytes,
            })
            .collect(),
    }
}

fn blocked_matrix(dimension: u16, value: impl Fn(u16, u16) -> f32) -> Vec<u8> {
    let grid = dimension / BLOCK_DIMENSION;
    let mut bytes = Vec::with_capacity(usize::from(dimension) * usize::from(dimension) * 4);
    for block_row in 0..grid {
        for block_column in 0..grid {
            for row in 0..BLOCK_DIMENSION {
                for column in 0..BLOCK_DIMENSION {
                    bytes.extend_from_slice(
                        &value(
                            block_row * BLOCK_DIMENSION + row,
                            block_column * BLOCK_DIMENSION + column,
                        )
                        .to_le_bytes(),
                    );
                }
            }
        }
    }
    debug_assert_eq!(
        bytes.len(),
        usize::from(dimension) * usize::from(dimension) * 4
    );
    bytes
}

fn verify_output(dimension: u16, actual: &[u8]) {
    let expected_bytes = usize::from(dimension) * usize::from(dimension) * 4;
    assert_eq!(actual.len(), expected_bytes, "GEMM output byte count");
    let grid = dimension / BLOCK_DIMENSION;
    let expected = (0..7)
        .map(|row| {
            (0..5)
                .map(|column| expected_value(dimension, row, column))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut offset = 0;
    for block_row in 0..grid {
        for block_column in 0..grid {
            for row in 0..BLOCK_DIMENSION {
                for column in 0..BLOCK_DIMENSION {
                    let global_row = block_row * BLOCK_DIMENSION + row;
                    let global_column = block_column * BLOCK_DIMENSION + column;
                    let actual_value =
                        f32::from_le_bytes(actual[offset..offset + 4].try_into().unwrap());
                    let expected =
                        expected[usize::from(global_row % 7)][usize::from(global_column % 5)];
                    assert_eq!(
                        actual_value.to_bits(),
                        expected.to_bits(),
                        "GEMM mismatch at [{global_row}, {global_column}]"
                    );
                    offset += 4;
                }
            }
        }
    }
}

fn left_value(row: u16, inner: u16) -> f32 {
    f32::from(((row + 3 * inner) % 7) as i8 - 3)
}

fn right_value(inner: u16, column: u16) -> f32 {
    f32::from(((2 * inner + column) % 5) as i8 - 2)
}

fn expected_value(dimension: u16, row: u16, column: u16) -> f32 {
    (0..dimension)
        .map(|inner| left_value(row, inner) * right_value(inner, column))
        .sum()
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"))
}
