use half::f16;
use ipu_compiler::{
    BlockedGemmConfig, GemmDataType, choose_gemm_row_block_for_shape, plan_blocked_gemm,
};
use ipu_elf::Toolchain;
use ipu_runtime::{
    BlockLayout, ExecutableGraph, HostRunOptions, ProfileGranularity, block_binding_typed,
    block_coordinates, blocked_matrix_f8_f143, blocked_matrix_f16, f143_from_f32, f143_scale,
    f143_to_f32, normal_f16, package_graph, package_graph_profiled_with, run_host_with_options,
};
use std::fs;
use std::path::PathBuf;
use tracing::info;

const TILE_COUNT: u16 = 1472;
const DEFAULT_BLOCK_DIMENSION: u16 = 64;
const DEFAULT_INNER_BLOCK_DIMENSION: u16 = 64;
const GEMM_DATA_BASE: u32 = 0xa0000;
const DEFAULT_CLOCK_HZ: u64 = 1_500_000_000;
const FP16_FLOPS_PER_TILE_CYCLE: f64 = 128.0;
const FP8_FLOPS_PER_TILE_CYCLE: f64 = 256.0;

fn main() {
    ipu_runtime::init_tracing();
    let dimension = env_u16("IPU_GEMM_DIMENSION", 128);
    let rows = env_u16("IPU_GEMM_ROWS", dimension);
    let inner_dimension = env_u16("IPU_GEMM_INNER_DIMENSION", dimension);
    let columns = env_u16("IPU_GEMM_COLUMNS", dimension);
    let block_dimension = env_u16("IPU_GEMM_BLOCK", DEFAULT_BLOCK_DIMENSION);
    let inner_block_dimension = env_u16("IPU_GEMM_INNER_BLOCK", DEFAULT_INNER_BLOCK_DIMENSION);
    assert!(columns.is_multiple_of(block_dimension));
    assert!(inner_dimension.is_multiple_of(inner_block_dimension));
    let seed = env_u64("IPU_GEMM_SEED", 0x05ee_df16);
    let max_error_limit = env_f32("IPU_F16_MAX_ERROR", 0.005);
    let fp8_weights = env_bool("IPU_GEMM_FP8_WEIGHTS", false);
    let native_fp8 = env_bool("IPU_GEMM_NATIVE_FP8", false);
    assert!(!(fp8_weights && native_fp8));
    let left = normal_f16(usize::from(rows) * usize::from(inner_dimension), seed, 0.5);
    let right = normal_f16(
        usize::from(inner_dimension) * usize::from(columns),
        seed ^ 0x9e37_79b9_7f4a_7c15,
        f32::from(inner_dimension).sqrt().recip(),
    );
    let weight_scale = f143_scale(right.iter().map(|value| value.to_f32()));
    let input_scale = f143_scale(left.iter().map(|value| value.to_f32()));
    let data_type = if native_fp8 {
        GemmDataType::F8F143 {
            input_scale,
            weight_scale,
        }
    } else if fp8_weights {
        GemmDataType::F16F8Weights {
            scale: weight_scale,
        }
    } else {
        GemmDataType::F16
    };
    let row_block_dimension = std::env::var("IPU_GEMM_ROW_BLOCK")
        .map(|value| value.parse().expect("IPU_GEMM_ROW_BLOCK must be a u16"))
        .unwrap_or_else(|_| {
            choose_gemm_row_block_for_shape(
                rows,
                inner_dimension,
                inner_block_dimension,
                columns,
                block_dimension,
                TILE_COUNT,
                data_type,
            )
            .expect("FP16 GEMM shape has no feasible row blocking")
        });
    let plan = plan_blocked_gemm(BlockedGemmConfig {
        rows,
        inner_dimension,
        columns,
        block_dimension,
        inner_block_dimension,
        row_block_dimension,
        tile_count: TILE_COUNT,
        data_base: GEMM_DATA_BASE,
        data_limit: ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
        data_type,
        retain_profile_metadata: true,
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

    let mut input = if native_fp8 {
        blocked_matrix_f8_f143(
            &plan.left,
            BlockLayout::AmpA32,
            input_scale,
            |row, column| left[matrix_index(inner_dimension, row, column)].to_f32(),
        )
    } else {
        blocked_matrix_f16(&plan.left, BlockLayout::AmpA16, |row, column| {
            left[matrix_index(inner_dimension, row, column)].to_f32()
        })
    };
    if fp8_weights || native_fp8 {
        input.extend(blocked_matrix_f8_f143(
            &plan.right,
            if native_fp8 {
                BlockLayout::AmpB32x16
            } else {
                BlockLayout::AmpB16x16
            },
            weight_scale,
            |row, column| right[matrix_index(columns, row, column)].to_f32(),
        ));
    } else {
        input.extend(blocked_matrix_f16(
            &plan.right,
            BlockLayout::AmpB16x16,
            |row, column| right[matrix_index(columns, row, column)].to_f32(),
        ));
    }
    let reference_left = left
        .iter()
        .map(|value| {
            let value = value.to_f32();
            if native_fp8 {
                f143_to_f32(f143_from_f32(value, input_scale), input_scale)
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    let reference_right = right
        .iter()
        .map(|value| {
            let value = value.to_f32();
            if fp8_weights || native_fp8 {
                f143_to_f32(f143_from_f32(value, weight_scale), weight_scale)
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    let graph = ExecutableGraph {
        memory_policy: None,
        host_weights: Vec::new(),
        schedule: plan.schedule,
        initial_buffers: Vec::new(),
        outputs: Vec::new(),
        host_inputs: vec![
            block_binding_typed(
                "left",
                rows,
                inner_dimension,
                &plan.left,
                if native_fp8 { "f8-f143" } else { "f16" },
                u64::from(data_type.input_element_bytes()),
            ),
            block_binding_typed(
                "right",
                inner_dimension,
                columns,
                &plan.right,
                if fp8_weights || native_fp8 {
                    "f8-f143"
                } else {
                    "f16"
                },
                u64::from(data_type.weight_element_bytes()),
            ),
        ],
        host_outputs: vec![block_binding_typed(
            "output",
            rows,
            columns,
            &output_placements,
            "f16",
            2,
        )],
    };

    let sdk = PathBuf::from(required_env("POPLAR_SDK_ENABLED"));
    let artifact_dir = std::env::temp_dir().join(format!("ipu-gemm-f16-{}", std::process::id()));
    let toolchain = Toolchain::from_sdk(&sdk);
    let source = |name: &str| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../device")
            .join(name)
    };
    let runtime = toolchain
        .compile(source("static_runtime.S"), &artifact_dir, "runtime", &[])
        .unwrap();
    let mut kernel_defines = vec![
        format!("-DGEMM_INNER_BLOCK_DIMENSION={inner_block_dimension}"),
        format!("-DGEMM_OUTPUT_COLUMNS={block_dimension}"),
        format!("-DGEMM_SMALL_ROWS={minimum_rows}"),
        format!("-DGEMM_LARGE_ROWS={maximum_rows}"),
    ];
    if native_fp8 {
        kernel_defines.extend([
            "-DGEMM_NATIVE_FP8=1".into(),
            "-DGEMM_INIT_SMALL_SYMBOL=ipu_stack_gemm_f8_init_small_rows".into(),
            "-DGEMM_INIT_LARGE_SYMBOL=ipu_stack_gemm_f8_init_large_rows".into(),
            "-DGEMM_ACCUMULATE_SMALL_SYMBOL=ipu_stack_gemm_f8_accumulate_small_rows".into(),
            "-DGEMM_ACCUMULATE_LARGE_SYMBOL=ipu_stack_gemm_f8_accumulate_large_rows".into(),
        ]);
    } else if fp8_weights {
        kernel_defines.extend([
            "-DGEMM_INIT_SMALL_SYMBOL=ipu_stack_gemm_f16_f8w_init_small_rows".into(),
            "-DGEMM_INIT_LARGE_SYMBOL=ipu_stack_gemm_f16_f8w_init_large_rows".into(),
            "-DGEMM_ACCUMULATE_SMALL_SYMBOL=ipu_stack_gemm_f16_f8w_accumulate_small_rows".into(),
            "-DGEMM_ACCUMULATE_LARGE_SYMBOL=ipu_stack_gemm_f16_f8w_accumulate_large_rows".into(),
        ]);
        kernel_defines.push("-DGEMM_INTERLEAVED_WEIGHTS=1".into());
    }
    let kernel = toolchain
        .compile(
            source("gemm_f16_64_amp.S"),
            &artifact_dir,
            if native_fp8 {
                "gemm-f8"
            } else if fp8_weights {
                "gemm-f16-f8w"
            } else {
                "gemm-f16"
            },
            &kernel_defines,
        )
        .unwrap();
    let mut objects = vec![
        fs::read(runtime.object).unwrap(),
        fs::read(kernel.object).unwrap(),
    ];
    if fp8_weights {
        let expand_codelet = toolchain
            .compile(
                source("expand_f8_f143_to_f16.cpp"),
                &artifact_dir,
                "expand-f8-codelet",
                &[],
            )
            .unwrap();
        let expand_wrapper = toolchain
            .compile(
                source("expand_f8_f143_to_f16.S"),
                &artifact_dir,
                "expand-f8-wrapper",
                &[],
            )
            .unwrap();
        let worker_support = toolchain
            .compile(
                source("worker_support.S"),
                &artifact_dir,
                "worker-support",
                &[],
            )
            .unwrap();
        objects.push(fs::read(expand_codelet.object).unwrap());
        objects.push(fs::read(expand_wrapper.object).unwrap());
        objects.push(fs::read(worker_support.object).unwrap());
    }
    let profile_output = std::env::var_os("IPU_PROFILE_OUTPUT").map(PathBuf::from);
    let (app, profile_layout) = if profile_output.is_some() {
        let granularity = ProfileGranularity::from_environment().unwrap();
        let (app, layout) = package_graph_profiled_with(&graph, &objects, granularity).unwrap();
        info!(?granularity, "enabled FP16 GEMM cycle profiling");
        (app, Some(layout))
    } else {
        (package_graph(&graph, &objects).unwrap(), None)
    };
    info!(
        rows,
        inner_dimension,
        columns,
        block_dimension,
        inner_block_dimension,
        fp8_weights,
        native_fp8,
        input_scale,
        weight_scale,
        seed,
        row_block_dimension,
        input_bytes = input.len(),
        "packaged randomized FP16/16 GEMM"
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
    if let (Some(path), Some(layout)) = (&profile_output, &profile_layout) {
        let clock_hz = env_u64("IPU_CLOCK_HZ", DEFAULT_CLOCK_HZ);
        let report = layout.decode(&actual, clock_hz).unwrap();
        let graph_cycles = graph_cycles(&report);
        let graph_seconds = f64::from(graph_cycles) / clock_hz as f64;
        let flops = 2.0 * f64::from(rows) * f64::from(inner_dimension) * f64::from(columns);
        let tflops = flops / graph_seconds / 1.0e12;
        let flops_per_tile_cycle = if native_fp8 {
            FP8_FLOPS_PER_TILE_CYCLE
        } else {
            FP16_FLOPS_PER_TILE_CYCLE
        };
        let peak_tflops = clock_hz as f64 * f64::from(TILE_COUNT) * flops_per_tile_cycle / 1.0e12;
        report.write(fs::File::create(path).unwrap()).unwrap();
        info!(
            path = %path.display(),
            graph_cycles,
            graph_ms = graph_seconds * 1.0e3,
            tflops,
            peak_tflops,
            efficiency_percent = tflops / peak_tflops * 100.0,
            "wrote FP16 GEMM cycle profile"
        );
    }
    if env_bool("IPU_F16_VALIDATE", true) {
        let errors = verify_output(
            rows,
            inner_dimension,
            columns,
            &actual,
            &output_placements,
            &reference_left,
            &reference_right,
            max_error_limit,
        )
        .unwrap();
        info!(
            rows,
            inner_dimension,
            columns,
            max_abs_error = errors.max_abs,
            rmse = errors.rmse,
            max_error_limit,
            "randomized FP16/16 GEMM passed"
        );
    } else {
        info!(
            rows,
            inner_dimension, columns, "skipped FP16 GEMM host reference"
        );
    }
    let _ = fs::remove_dir_all(artifact_dir);
}

fn graph_cycles(report: &ipu_package::ProfileReport) -> u32 {
    report
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
        .unwrap_or(0)
}

struct ErrorStatistics {
    max_abs: f32,
    rmse: f32,
}

fn verify_output(
    rows: u16,
    inner_dimension: u16,
    columns: u16,
    actual: &[u8],
    placements: &[ipu_compiler::BlockPlacement],
    left: &[f32],
    right: &[f32],
    max_error_limit: f32,
) -> ipu_runtime::Result<ErrorStatistics> {
    let actual = actual
        .get(..usize::from(rows) * usize::from(columns) * 2)
        .ok_or("FP16 GEMM output is shorter than the matrix")?;
    let mut offset = 0usize;
    let mut max_abs = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut worst = (0, 0, 0.0, 0.0);
    for placement in placements {
        for linear in 0..placement.rows * placement.columns {
            let (row, column) = block_coordinates(
                BlockLayout::AmpC16F16,
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
            let expected = (0..inner_dimension)
                .map(|inner| {
                    left[matrix_index(inner_dimension, row, inner)]
                        * right[matrix_index(columns, inner, column)]
                })
                .sum::<f32>();
            if !observed.is_finite() {
                return Err(format!("non-finite FP16 GEMM output at [{row}, {column}]").into());
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
            "FP16 GEMM max error {max_abs:.7} exceeds {max_error_limit:.7}; worst at [{}, {}]: observed={:.7}, expected={:.7}",
            worst.0, worst.1, worst.2, worst.3,
        )
        .into());
    }
    Ok(ErrorStatistics {
        max_abs,
        rmse: (squared_error / (f64::from(rows) * f64::from(columns))).sqrt() as f32,
    })
}

fn matrix_index(dimension: u16, row: u16, column: u16) -> usize {
    usize::from(row) * usize::from(dimension) + usize::from(column)
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

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| match value.as_str() {
            "0" | "false" => false,
            "1" | "true" => true,
            _ => panic!("{name} must be 0, 1, false, or true"),
        })
        .unwrap_or(default)
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"))
}
