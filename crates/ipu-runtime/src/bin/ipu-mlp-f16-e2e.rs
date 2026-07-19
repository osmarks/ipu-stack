use half::f16;
use ipu_compiler::{BlockedMlpConfig, GemmDataType, choose_gemm_row_block_for, plan_blocked_mlp};
use ipu_elf::Toolchain;
use ipu_runtime::{
    BlockLayout, ExecutableGraph, HostRunOptions, block_binding_typed, block_coordinates,
    blocked_matrix_f16, normal_f16, package_graph, run_host_with_options,
};
use std::fs;
use std::path::PathBuf;
use tracing::info;

const TILE_COUNT: u16 = 1472;
const BLOCK_DIMENSION: u16 = 64;
const INNER_BLOCK_DIMENSION: u16 = 64;
const MLP_DATA_BASE: u32 = 0xa0000;

fn main() {
    ipu_runtime::init_tracing();
    let batch = env_u16("IPU_MLP_BATCH", 64);
    let width = env_u16("IPU_MLP_WIDTH", 256);
    let layers = env_u16("IPU_MLP_LAYERS", 4);
    let seed = env_u64("IPU_MLP_SEED", 0x5eed_1616);
    let max_error_limit = env_f32("IPU_F16_MAX_ERROR", 0.005);
    assert!(width.is_multiple_of(BLOCK_DIMENSION));
    let row_block_dimension = std::env::var("IPU_MLP_ROW_BLOCK")
        .map(|value| value.parse().expect("IPU_MLP_ROW_BLOCK must be a u16"))
        .unwrap_or_else(|_| {
            choose_gemm_row_block_for(
                batch,
                INNER_BLOCK_DIMENSION,
                width,
                BLOCK_DIMENSION,
                TILE_COUNT,
                GemmDataType::F16,
            )
            .expect("FP16 MLP shape has no feasible row blocking")
        });
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
        data_type: GemmDataType::F16,
    })
    .unwrap();
    let output_placements = plan.output.clone();
    let minimum_rows = output_placements
        .iter()
        .map(|block| block.rows)
        .min()
        .unwrap();
    let maximum_rows = output_placements
        .iter()
        .map(|block| block.rows)
        .max()
        .unwrap();

    let input_values = normal_f16(usize::from(batch) * usize::from(width), seed, 0.5);
    let weights = (0..usize::from(layers))
        .map(|layer| {
            normal_f16(
                usize::from(width).pow(2),
                seed ^ (0x9e37_79b9_7f4a_7c15u64.wrapping_mul(layer as u64 + 1)),
                f32::from(width).sqrt().recip(),
            )
        })
        .collect::<Vec<_>>();
    let mut host_inputs = vec![block_binding_typed(
        "input",
        batch,
        width,
        &plan.input,
        "f16",
        2,
    )];
    let mut input = blocked_matrix_f16(&plan.input, BlockLayout::AmpA16, |row, column| {
        input_values[matrix_index(width, row, column)].to_f32()
    });
    for (layer, placements) in plan.weights.iter().enumerate() {
        host_inputs.push(block_binding_typed(
            &format!("layer_{layer}_weight"),
            width,
            width,
            placements,
            "f16",
            2,
        ));
        input.extend(blocked_matrix_f16(
            placements,
            BlockLayout::AmpB16x16,
            |row, column| weights[layer][matrix_index(width, row, column)].to_f32(),
        ));
    }
    let graph = ExecutableGraph {
        schedule: plan.schedule,
        initial_buffers: Vec::new(),
        outputs: Vec::new(),
        host_inputs,
        host_outputs: vec![block_binding_typed(
            "output",
            batch,
            width,
            &output_placements,
            "f16",
            2,
        )],
    };

    let sdk = PathBuf::from(required_env("POPLAR_SDK_ENABLED"));
    let artifact_dir = std::env::temp_dir().join(format!("ipu-mlp-f16-{}", std::process::id()));
    let toolchain = Toolchain::from_sdk(&sdk);
    let source = |name: &str| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../device")
            .join(name)
    };
    let runtime = toolchain
        .compile(source("static_runtime.S"), &artifact_dir, "runtime", &[])
        .unwrap();
    let gemm = toolchain
        .compile(
            source("gemm_f16_64_amp.S"),
            &artifact_dir,
            "gemm-f16",
            &[
                format!("-DGEMM_INNER_BLOCK_DIMENSION={INNER_BLOCK_DIMENSION}"),
                format!("-DGEMM_SMALL_ROWS={minimum_rows}"),
                format!("-DGEMM_LARGE_ROWS={maximum_rows}"),
            ],
        )
        .unwrap();
    let gelu = toolchain
        .compile(
            source("gelu_relayout_f16.S"),
            &artifact_dir,
            "gelu-f16",
            &[],
        )
        .unwrap();
    let app = package_graph(
        &graph,
        &[
            fs::read(runtime.object).unwrap(),
            fs::read(gemm.object).unwrap(),
            fs::read(gelu.object).unwrap(),
        ],
    )
    .unwrap();
    info!(
        batch,
        width,
        layers,
        seed,
        row_block_dimension,
        phases = graph.schedule.phases.len(),
        input_bytes = input.len(),
        "packaged randomized FP16/16 MLP"
    );

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
        &input,
        HostRunOptions::from_environment().unwrap(),
    )
    .unwrap();
    let reference = reference_mlp(batch, width, &input_values, &weights);
    let errors = verify_output(
        batch,
        width,
        &actual,
        &output_placements,
        &reference,
        max_error_limit,
    )
    .unwrap();
    info!(
        batch,
        width,
        layers,
        max_abs_error = errors.max_abs,
        rmse = errors.rmse,
        max_error_limit,
        "randomized FP16/16 MLP passed"
    );
    let _ = fs::remove_dir_all(artifact_dir);
}

fn reference_mlp(batch: u16, width: u16, input: &[f16], weights: &[Vec<f16>]) -> Vec<f16> {
    let mut activation = input.to_vec();
    for weight in weights {
        let mut output = vec![f16::ZERO; usize::from(batch) * usize::from(width)];
        for row in 0..batch {
            for column in 0..width {
                let value = (0..width)
                    .map(|inner| {
                        activation[matrix_index(width, row, inner)].to_f32()
                            * weight[matrix_index(width, inner, column)].to_f32()
                    })
                    .sum::<f32>();
                output[matrix_index(width, row, column)] = f16::from_f32(gelu(value));
            }
        }
        activation = output;
    }
    activation
}

struct ErrorStatistics {
    max_abs: f32,
    rmse: f32,
}

fn verify_output(
    batch: u16,
    width: u16,
    actual: &[u8],
    placements: &[ipu_compiler::BlockPlacement],
    reference: &[f16],
    max_error_limit: f32,
) -> ipu_runtime::Result<ErrorStatistics> {
    let actual = actual
        .get(..usize::from(batch) * usize::from(width) * 2)
        .ok_or("FP16 MLP output is shorter than the activation matrix")?;
    let mut offset = 0usize;
    let mut max_abs = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut worst = (0, 0, 0.0, 0.0);
    for placement in placements {
        for linear in 0..placement.rows * placement.columns {
            let (row, column) = block_coordinates(
                BlockLayout::AmpA16,
                placement.rows,
                placement.columns,
                linear,
            );
            let row = placement.row_start + row;
            let column = placement.column_start + column;
            let observed = f16::from_bits(u16::from_le_bytes(
                actual[offset..offset + 2].try_into().unwrap(),
            ))
            .to_f32();
            let expected = reference[matrix_index(width, row, column)].to_f32();
            if !observed.is_finite() {
                return Err(format!("non-finite FP16 MLP output at [{row}, {column}]").into());
            }
            let error = (observed - expected).abs();
            if error > max_abs {
                max_abs = error;
                worst = (row, column, observed, expected);
            }
            squared_error += f64::from(error).powi(2);
            offset += 2;
        }
    }
    if max_abs > max_error_limit {
        return Err(format!(
            "FP16 MLP max error {max_abs:.7} exceeds {max_error_limit:.7}; worst at [{}, {}]: observed={:.7}, expected={:.7}",
            worst.0, worst.1, worst.2, worst.3
        )
        .into());
    }
    Ok(ErrorStatistics {
        max_abs,
        rmse: (squared_error / (f64::from(batch) * f64::from(width))).sqrt() as f32,
    })
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044715 * value.powi(3))).tanh())
}

fn matrix_index(columns: u16, row: u16, column: u16) -> usize {
    usize::from(row) * usize::from(columns) + usize::from(column)
}

fn env_u16(name: &str, default: u16) -> u16 {
    std::env::var(name)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be a u16"))
        })
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be a u64"))
        })
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be an f32"))
        })
        .unwrap_or(default)
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"))
}
