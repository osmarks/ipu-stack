use half::f16;
use ipu_compiler::{
    AttentionKeyValuePlacement, AttentionTaskPlacement, FlashAttentionConfig, FlashAttentionPlan,
    plan_flash_attention,
};
use ipu_elf::Toolchain;
use ipu_package::{Binding, RegionSlice};
use ipu_runtime::{
    BlockLayout, ExecutableGraph, HostRunOptions, ProfileGranularity, block_coordinates,
    normal_f16, package_graph, package_graph_profiled_with, run_host_with_options,
};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;

const TILE_COUNT: u16 = 1472;
const ATTENTION_HEADS: u16 = 16;
const ATTENTION_DATA_BASE: u32 = 0xa0000;
const DEFAULT_CLOCK_HZ: u64 = 1_500_000_000;
const FP16_FLOPS_PER_TILE_CYCLE: f64 = 128.0;

fn main() {
    ipu_runtime::init_tracing();
    let hidden_sizes = env_u16_list("IPU_ATTENTION_HIDDEN_SIZES", &[768, 1024, 1152]);
    let batch_sizes = env_u16_list("IPU_ATTENTION_BATCH_SIZES", &[1, 3]);
    let sequence_length = env_u16("IPU_ATTENTION_SEQUENCE_LENGTH", 64);
    let query_block_rows = env_u16("IPU_ATTENTION_QUERY_BLOCK_ROWS", 0);
    let key_block_rows = env_u16("IPU_ATTENTION_KEY_BLOCK_ROWS", 0);
    let seed = env_u64("IPU_ATTENTION_SEED", 0xfa57_a77e_1616);
    let max_error_limit = env_f32("IPU_F16_MAX_ERROR", 0.001);
    let sdk = PathBuf::from(required_env("POPLAR_SDK_ENABLED"));
    let configuration = fs::read(required_env("IPU_CONFIG")).unwrap();
    let bootloader = fs::read(
        std::env::var_os("IPU_BOOTLOADER")
            .map(PathBuf::from)
            .unwrap_or_else(|| sdk.join("bin/ipu/tile_bootloader_ipu2.elf")),
    )
    .unwrap();
    let device = std::env::var("IPU_DEVICE").unwrap_or_else(|_| "/dev/ipu0".into());

    for (hidden_index, &hidden_size) in hidden_sizes.iter().enumerate() {
        for (batch_index, &batch_size) in batch_sizes.iter().enumerate() {
            run_case(Case {
                batch_size,
                sequence_length,
                hidden_size,
                query_block_rows,
                key_block_rows,
                seed: seed
                    ^ (hidden_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    ^ (batch_index as u64).wrapping_mul(0xd1b5_4a32_d192_ed03),
                max_error_limit,
                sdk: &sdk,
                configuration: &configuration,
                bootloader: &bootloader,
                device: &device,
            });
        }
    }
}

struct Case<'a> {
    batch_size: u16,
    sequence_length: u16,
    hidden_size: u16,
    query_block_rows: u16,
    key_block_rows: u16,
    seed: u64,
    max_error_limit: f32,
    sdk: &'a Path,
    configuration: &'a [u8],
    bootloader: &'a [u8],
    device: &'a str,
}

fn run_case(case: Case<'_>) {
    let config = FlashAttentionConfig {
        batch_size: case.batch_size,
        sequence_length: case.sequence_length,
        hidden_size: case.hidden_size,
        attention_heads: ATTENTION_HEADS,
        query_block_rows: case.query_block_rows,
        key_block_rows: case.key_block_rows,
        tile_count: TILE_COUNT,
        data_base: ATTENTION_DATA_BASE,
        data_limit: ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
    };
    let plan = plan_flash_attention(config).unwrap();
    let elements = usize::from(case.batch_size)
        * usize::from(case.sequence_length)
        * usize::from(case.hidden_size);
    let query = normal_f16(elements, case.seed, 0.5);
    let key = normal_f16(elements, case.seed ^ 0xa076_1d64_78bd_642f, 0.5);
    let value = normal_f16(elements, case.seed ^ 0xe703_7ed1_a0b4_28db, 0.5);
    let (query_bytes, key_value_bytes) = pack_inputs(&plan, config, &query, &key, &value);
    let mut input = query_bytes;
    input.extend(key_value_bytes);
    let graph = ExecutableGraph {
        schedule: plan.schedule.clone(),
        initial_buffers: Vec::new(),
        outputs: Vec::new(),
        host_inputs: vec![
            task_binding(
                "query",
                "f16",
                vec![
                    u32::from(case.batch_size),
                    u32::from(ATTENTION_HEADS),
                    u32::from(case.sequence_length),
                    u32::from(plan.head_dimension),
                ],
                &plan.tasks,
                |task| task.query_address,
                |task| u64::from(task.query_rows) * u64::from(plan.padded_head_dimension) * 2,
            ),
            key_value_binding(
                "key_value",
                "f16",
                vec![
                    2,
                    u32::from(case.batch_size),
                    u32::from(ATTENTION_HEADS),
                    u32::from(case.sequence_length),
                    u32::from(plan.head_dimension),
                ],
                &plan.key_values,
            ),
        ],
        host_outputs: vec![task_binding(
            "attention",
            "f16",
            vec![
                u32::from(case.batch_size),
                u32::from(ATTENTION_HEADS),
                u32::from(case.sequence_length),
                u32::from(plan.head_dimension),
            ],
            &plan.tasks,
            |task| task.output_address,
            |task| u64::from(task.query_rows) * u64::from(plan.head_dimension) * 2,
        )],
    };

    let artifact_dir = std::env::temp_dir().join(format!(
        "ipu-attention-f16-{}-{}-{}",
        std::process::id(),
        case.hidden_size,
        case.batch_size
    ));
    let toolchain = Toolchain::from_sdk(case.sdk);
    let source = |name: &str| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../device")
            .join(name)
    };
    let runtime = toolchain
        .compile(source("static_runtime.S"), &artifact_dir, "runtime", &[])
        .unwrap();
    let codelet = toolchain
        .compile(
            source("flash_attention_f16.cpp"),
            &artifact_dir,
            "flash-attention-codelet",
            &[
                format!("-DATTENTION_HEAD_DIMENSION={}", plan.head_dimension),
                format!(
                    "-DATTENTION_PADDED_HEAD_DIMENSION={}",
                    plan.padded_head_dimension
                ),
                format!("-DATTENTION_KEY_BLOCK_COLUMNS={}", plan.key_block_columns),
            ],
        )
        .unwrap();
    let wrapper = toolchain
        .compile(
            source("flash_attention_f16.S"),
            &artifact_dir,
            "flash-attention-wrapper",
            &[],
        )
        .unwrap();
    let minimum_rows = plan.tasks.iter().map(|task| task.query_rows).min().unwrap();
    let maximum_rows = plan.tasks.iter().map(|task| task.query_rows).max().unwrap();
    let qk = toolchain
        .compile(
            source("gemm_f16_64_amp.S"),
            &artifact_dir,
            "attention-qk",
            &[
                format!(
                    "-DGEMM_INNER_BLOCK_DIMENSION={}",
                    plan.padded_head_dimension
                ),
                format!("-DGEMM_OUTPUT_COLUMNS={}", plan.key_block_columns),
                format!("-DGEMM_SMALL_ROWS={minimum_rows}"),
                format!("-DGEMM_LARGE_ROWS={maximum_rows}"),
            ],
        )
        .unwrap();
    let value_offset =
        u32::from(plan.padded_head_dimension) * u32::from(plan.key_block_columns) * 2;
    let pv = toolchain
        .compile(
            source("gemm_f16_64_amp.S"),
            &artifact_dir,
            "attention-pv",
            &[
                format!("-DGEMM_INNER_BLOCK_DIMENSION={}", plan.key_block_columns),
                format!("-DGEMM_OUTPUT_COLUMNS={}", plan.padded_head_dimension),
                format!("-DGEMM_SMALL_ROWS={minimum_rows}"),
                format!("-DGEMM_LARGE_ROWS={maximum_rows}"),
                format!("-DGEMM_RIGHT_BYTE_OFFSET={value_offset}"),
                "-DGEMM_INIT_SMALL_SYMBOL=ipu_stack_attention_pv_init_small_rows".into(),
                "-DGEMM_INIT_LARGE_SYMBOL=ipu_stack_attention_pv_init_large_rows".into(),
                "-DGEMM_ACCUMULATE_SMALL_SYMBOL=ipu_stack_attention_pv_accumulate_small_rows"
                    .into(),
                "-DGEMM_ACCUMULATE_LARGE_SYMBOL=ipu_stack_attention_pv_accumulate_large_rows"
                    .into(),
            ],
        )
        .unwrap();
    let objects = [
        fs::read(runtime.object).unwrap(),
        fs::read(codelet.object).unwrap(),
        fs::read(wrapper.object).unwrap(),
        fs::read(qk.object).unwrap(),
        fs::read(pv.object).unwrap(),
    ];
    let profile_output = profile_path(&case);
    let (app, profile_layout) = if profile_output.is_some() {
        let granularity = ProfileGranularity::from_environment().unwrap();
        let (app, layout) = package_graph_profiled_with(&graph, &objects, granularity).unwrap();
        info!(?granularity, "enabled FlashAttention cycle profiling");
        (app, Some(layout))
    } else {
        (package_graph(&graph, &objects).unwrap(), None)
    };
    info!(
        batch_size = case.batch_size,
        sequence_length = case.sequence_length,
        hidden_size = case.hidden_size,
        heads = ATTENTION_HEADS,
        head_dimension = plan.head_dimension,
        tasks = plan.tasks.len(),
        query_block_rows = plan.query_block_rows,
        key_block_rows = plan.key_block_rows,
        padded_head_dimension = plan.padded_head_dimension,
        key_block_columns = plan.key_block_columns,
        exchange_phases =
            plan.key_values.len() / (usize::from(case.batch_size) * usize::from(ATTENTION_HEADS)),
        input_bytes = input.len(),
        "packaged FP16 FlashAttention"
    );
    let actual = run_host_with_options(
        &app,
        case.bootloader,
        case.configuration,
        case.device,
        &input,
        HostRunOptions::from_environment().unwrap(),
    )
    .unwrap();
    if let (Some(path), Some(layout)) = (&profile_output, &profile_layout) {
        let clock_hz = env_u64("IPU_CLOCK_HZ", DEFAULT_CLOCK_HZ);
        let report = layout.decode(&actual, clock_hz).unwrap();
        let graph_cycles = graph_cycles(&report);
        let graph_seconds = f64::from(graph_cycles) / clock_hz as f64;
        // QK and PV each perform one multiply and one add per attention pair.
        let flops = 4.0
            * f64::from(case.batch_size)
            * f64::from(ATTENTION_HEADS)
            * f64::from(case.sequence_length).powi(2)
            * f64::from(plan.head_dimension);
        let tflops = flops / graph_seconds / 1.0e12;
        let peak_tflops =
            clock_hz as f64 * f64::from(TILE_COUNT) * FP16_FLOPS_PER_TILE_CYCLE / 1.0e12;
        report.write(fs::File::create(path).unwrap()).unwrap();
        info!(
            path = %path.display(),
            graph_cycles,
            graph_ms = graph_seconds * 1.0e3,
            attention_tflops = tflops,
            peak_tflops,
            efficiency_percent = tflops / peak_tflops * 100.0,
            "wrote FlashAttention cycle profile"
        );
    }
    let expected = attention_reference(config, &query, &key, &value);
    let errors = verify_output(&actual, &expected, case.max_error_limit).unwrap();
    info!(
        batch_size = case.batch_size,
        sequence_length = case.sequence_length,
        hidden_size = case.hidden_size,
        heads = ATTENTION_HEADS,
        max_abs_error = errors.max_abs,
        rmse = errors.rmse,
        max_error_limit = case.max_error_limit,
        "FP16 FlashAttention passed host softmax reference"
    );
    let _ = fs::remove_dir_all(artifact_dir);
}

fn profile_path(case: &Case<'_>) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("IPU_PROFILE_OUTPUT")?);
    let multiple_cases = env_u16_list("IPU_ATTENTION_HIDDEN_SIZES", &[768, 1024, 1152]).len()
        * env_u16_list("IPU_ATTENTION_BATCH_SIZES", &[1, 3]).len()
        > 1;
    if !multiple_cases {
        return Some(path);
    }
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("capnp");
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("attention");
    Some(path.with_file_name(format!(
        "{stem}-h{}-b{}.{}",
        case.hidden_size, case.batch_size, extension
    )))
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

fn task_binding(
    name: &str,
    dtype: &str,
    shape: Vec<u32>,
    tasks: &[AttentionTaskPlacement],
    address: impl Fn(&AttentionTaskPlacement) -> u32,
    size: impl Fn(&AttentionTaskPlacement) -> u64,
) -> Binding {
    let topology = ipu_exchange::Topology::c600();
    let mut file_offset = 0u64;
    Binding {
        name: name.into(),
        dtype: dtype.into(),
        shape,
        slices: tasks
            .iter()
            .map(|task| {
                let size = size(task);
                let slice = RegionSlice {
                    tile: u32::from(topology.physical(task.tile).unwrap()),
                    tile_address: address(task),
                    file_offset,
                    size,
                };
                file_offset += size;
                slice
            })
            .collect(),
    }
}

fn key_value_binding(
    name: &str,
    dtype: &str,
    shape: Vec<u32>,
    blocks: &[AttentionKeyValuePlacement],
) -> Binding {
    let topology = ipu_exchange::Topology::c600();
    let mut file_offset = 0u64;
    Binding {
        name: name.into(),
        dtype: dtype.into(),
        shape,
        slices: blocks
            .iter()
            .map(|block| {
                let slice = RegionSlice {
                    tile: u32::from(topology.physical(block.tile).unwrap()),
                    tile_address: block.address,
                    file_offset,
                    size: u64::from(block.size),
                };
                file_offset += u64::from(block.size);
                slice
            })
            .collect(),
    }
}

fn pack_inputs(
    plan: &FlashAttentionPlan,
    config: FlashAttentionConfig,
    query: &[f16],
    key: &[f16],
    value: &[f16],
) -> (Vec<u8>, Vec<u8>) {
    let mut query_bytes = Vec::new();
    let mut key_value_bytes = Vec::new();
    for task in &plan.tasks {
        for linear in 0..task.query_rows * plan.padded_head_dimension {
            let (row, column) = block_coordinates(
                BlockLayout::AmpA16,
                task.query_rows,
                plan.padded_head_dimension,
                linear,
            );
            let bits = if column < plan.head_dimension {
                query[tensor_index(
                    config,
                    task.batch,
                    task.head,
                    task.query_row_start + row,
                    column,
                )]
                .to_bits()
            } else {
                0
            };
            query_bytes.extend_from_slice(&bits.to_le_bytes());
        }
    }
    for block in &plan.key_values {
        for linear in 0..plan.padded_head_dimension * plan.key_block_columns {
            let (inner, key_column) = block_coordinates(
                BlockLayout::AmpB16x16,
                plan.padded_head_dimension,
                plan.key_block_columns,
                linear,
            );
            let bits = if inner < plan.head_dimension && key_column < block.key_rows {
                key[tensor_index(
                    config,
                    block.batch,
                    block.head,
                    block.key_row_start + key_column,
                    inner,
                )]
                .to_bits()
            } else {
                0
            };
            key_value_bytes.extend_from_slice(&bits.to_le_bytes());
        }
        for linear in 0..plan.padded_head_dimension * plan.key_block_columns {
            let (key_row, column) = block_coordinates(
                BlockLayout::AmpB16x16,
                plan.key_block_columns,
                plan.padded_head_dimension,
                linear,
            );
            let bits = if key_row < block.key_rows && column < plan.head_dimension {
                value[tensor_index(
                    config,
                    block.batch,
                    block.head,
                    block.key_row_start + key_row,
                    column,
                )]
                .to_bits()
            } else {
                0
            };
            key_value_bytes.extend_from_slice(&bits.to_le_bytes());
        }
    }
    (query_bytes, key_value_bytes)
}

fn attention_reference(
    config: FlashAttentionConfig,
    query: &[f16],
    key: &[f16],
    value: &[f16],
) -> Vec<f32> {
    let head_dimension = config.hidden_size / config.attention_heads;
    let scale = f32::from(head_dimension).sqrt().recip();
    let mut output = Vec::with_capacity(query.len());
    for batch in 0..config.batch_size {
        for head in 0..config.attention_heads {
            for query_row in 0..config.sequence_length {
                let mut scores = Vec::with_capacity(usize::from(config.sequence_length));
                for key_row in 0..config.sequence_length {
                    let score = (0..head_dimension)
                        .map(|column| {
                            query[tensor_index(config, batch, head, query_row, column)].to_f32()
                                * key[tensor_index(config, batch, head, key_row, column)].to_f32()
                        })
                        .sum::<f32>()
                        * scale;
                    scores.push(score);
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator = scores
                    .iter_mut()
                    .map(|score| {
                        *score = (*score - maximum).exp();
                        *score
                    })
                    .sum::<f32>();
                for column in 0..head_dimension {
                    output.push(
                        scores
                            .iter()
                            .enumerate()
                            .map(|(key_row, weight)| {
                                *weight
                                    * value
                                        [tensor_index(config, batch, head, key_row as u16, column)]
                                    .to_f32()
                            })
                            .sum::<f32>()
                            / denominator,
                    );
                }
            }
        }
    }
    output
}

fn tensor_index(
    config: FlashAttentionConfig,
    batch: u16,
    head: u16,
    row: u16,
    column: u16,
) -> usize {
    ((usize::from(batch) * usize::from(config.sequence_length) + usize::from(row))
        * usize::from(config.hidden_size))
        + usize::from(head) * usize::from(config.hidden_size / config.attention_heads)
        + usize::from(column)
}

struct ErrorStatistics {
    max_abs: f32,
    rmse: f32,
}

fn verify_output(
    actual: &[u8],
    expected: &[f32],
    max_error_limit: f32,
) -> ipu_runtime::Result<ErrorStatistics> {
    let actual = actual
        .get(..expected.len() * 2)
        .ok_or("FlashAttention output is shorter than expected")?;
    let mut max_abs = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut worst = (0usize, 0.0f32, 0.0f32);
    for (index, &expected) in expected.iter().enumerate() {
        let observed = f16::from_bits(u16::from_le_bytes(
            actual[index * 2..index * 2 + 2].try_into().unwrap(),
        ))
        .to_f32();
        if !observed.is_finite() {
            return Err(format!("non-finite FlashAttention output at element {index}").into());
        }
        let error = (observed - expected).abs();
        if error > max_abs {
            max_abs = error;
            worst = (index, observed, expected);
        }
        squared_error += f64::from(error).powi(2);
    }
    if max_abs > max_error_limit {
        return Err(format!(
            "FlashAttention max error {max_abs:.7} exceeds {max_error_limit:.7}; worst element {} observed={:.7} expected={:.7}",
            worst.0, worst.1, worst.2
        )
        .into());
    }
    Ok(ErrorStatistics {
        max_abs,
        rmse: (squared_error / expected.len() as f64).sqrt() as f32,
    })
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

fn env_u16_list(name: &str, default: &[u16]) -> Vec<u16> {
    std::env::var(name)
        .map(|value| {
            value
                .split(',')
                .map(|item| {
                    item.parse()
                        .unwrap_or_else(|_| panic!("{name} must be a comma-separated u16 list"))
                })
                .collect()
        })
        .unwrap_or_else(|_| default.to_vec())
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .map(|value| {
            let value = value.strip_prefix("0x").unwrap_or(&value);
            u64::from_str_radix(value, 16).unwrap_or_else(|_| panic!("{name} must be hexadecimal"))
        })
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be f32"))
        })
        .unwrap_or(default)
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"))
}
