use half::f16;
use ipu_compiler::{AffineLayerNormConfig, RowShardPlacement, plan_affine_layer_norm_f16};
use ipu_elf::Toolchain;
use ipu_package::{Binding, RegionSlice};
use ipu_runtime::{
    ExecutableGraph, HostRunOptions, normal_f16, package_graph, run_host_with_options,
};
use std::fs;
use std::path::PathBuf;
use tracing::info;

const TILE_COUNT: u16 = 1472;
const DATA_BASE: u32 = 0xa0000;

fn main() {
    ipu_runtime::init_tracing();
    let rows = env_u16("IPU_LAYER_NORM_ROWS", 729);
    let columns = env_u16("IPU_LAYER_NORM_COLUMNS", 1152);
    let row_block_dimension = env_u16("IPU_LAYER_NORM_ROW_BLOCK", 12);
    let epsilon = env_f32("IPU_LAYER_NORM_EPSILON", 1e-6);
    let seed = env_u64("IPU_LAYER_NORM_SEED", 0x51_61_1f_16);
    let max_error_limit = env_f32("IPU_F16_MAX_ERROR", 0.01);
    let plan = plan_affine_layer_norm_f16(AffineLayerNormConfig {
        rows,
        columns,
        row_block_dimension,
        tile_count: TILE_COUNT,
        data_base: DATA_BASE,
        data_limit: ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
        epsilon_bits: epsilon.to_bits(),
    })
    .unwrap();

    let elements = usize::from(rows) * usize::from(columns);
    let input = normal_f16(elements, seed, 0.5);
    let gamma_noise = normal_f16(usize::from(columns), seed ^ 0xa076_1d64, 0.05);
    let beta = normal_f16(usize::from(columns), seed ^ 0xe703_7ed1, 0.05);
    let gamma = gamma_noise
        .into_iter()
        .map(|value| f16::from_f32(1.0 + value.to_f32()))
        .collect::<Vec<_>>();
    let affine = gamma
        .iter()
        .chain(&beta)
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    let mut host_input = pack_rows(&input, columns, &plan.input);
    for _ in &plan.affine {
        host_input.extend_from_slice(&affine);
    }

    let graph = ExecutableGraph {
        schedule: plan.schedule,
        initial_buffers: Vec::new(),
        outputs: Vec::new(),
        host_inputs: vec![
            row_binding("input", "f16", rows, columns, &plan.input, 2),
            row_binding("affine", "f16", 2, columns, &plan.affine, 2),
        ],
        host_outputs: vec![row_binding("output", "f16", rows, columns, &plan.output, 2)],
    };

    let sdk = PathBuf::from(required_env("POPLAR_SDK_ENABLED"));
    let artifact_dir =
        std::env::temp_dir().join(format!("ipu-layer-norm-f16-{}", std::process::id()));
    let source = |name: &str| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../device")
            .join(name)
    };
    let toolchain = Toolchain::from_sdk(&sdk);
    let runtime = toolchain
        .compile(source("static_runtime.S"), &artifact_dir, "runtime", &[])
        .unwrap();
    let codelet = toolchain
        .compile(
            source("layer_norm_f16.cpp"),
            &artifact_dir,
            "layer-norm-codelet",
            &[],
        )
        .unwrap();
    let wrapper = toolchain
        .compile(
            source("layer_norm_f16.S"),
            &artifact_dir,
            "layer-norm-wrapper",
            &[],
        )
        .unwrap();
    let app = package_graph(
        &graph,
        &[
            fs::read(runtime.object).unwrap(),
            fs::read(codelet.object).unwrap(),
            fs::read(wrapper.object).unwrap(),
        ],
    )
    .unwrap();
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
    let max_error = verify(
        rows,
        columns,
        epsilon,
        &input,
        &gamma,
        &beta,
        &actual,
        &plan.output,
    );
    assert!(
        max_error <= max_error_limit,
        "FP16 layer norm max error {max_error:.7} exceeds {max_error_limit:.7}"
    );
    info!(
        rows,
        columns,
        row_block_dimension,
        shards = plan.output.len(),
        max_error,
        "randomized FP16 affine layer norm passed"
    );
}

fn row_binding(
    name: &str,
    dtype: &str,
    rows: u16,
    columns: u16,
    shards: &[RowShardPlacement],
    element_bytes: u64,
) -> Binding {
    let topology = ipu_exchange::Topology::c600();
    let mut file_offset = 0;
    let slices = shards
        .iter()
        .map(|shard| {
            let size = u64::from(shard.rows) * u64::from(shard.columns) * element_bytes;
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
        dtype: dtype.into(),
        shape: vec![u32::from(rows), u32::from(columns)],
        slices,
    }
}

fn pack_rows(values: &[f16], columns: u16, shards: &[RowShardPlacement]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for shard in shards {
        for linear in 0..shard.rows * columns {
            let panel_elements = shard.rows * 16;
            let panel = linear / panel_elements;
            let panel_offset = linear % panel_elements;
            let row = panel_offset / 16;
            let column = panel * 16 + panel_offset % 16;
            let index =
                usize::from(shard.row_start + row) * usize::from(columns) + usize::from(column);
            bytes.extend_from_slice(&values[index].to_bits().to_le_bytes());
        }
    }
    bytes
}

fn verify(
    rows: u16,
    columns: u16,
    epsilon: f32,
    input: &[f16],
    gamma: &[f16],
    beta: &[f16],
    actual: &[u8],
    shards: &[RowShardPlacement],
) -> f32 {
    let mut max_error = 0.0f32;
    let mut byte_offset = 0usize;
    for shard in shards {
        for linear in 0..shard.rows * columns {
            let panel_elements = shard.rows * 16;
            let panel = linear / panel_elements;
            let panel_offset = linear % panel_elements;
            let local_row = panel_offset / 16;
            let column = panel * 16 + panel_offset % 16;
            let row = shard.row_start + local_row;
            let values = &input[usize::from(row) * usize::from(columns)
                ..usize::from(row + 1) * usize::from(columns)];
            let mean = values.iter().map(|value| value.to_f32()).sum::<f32>() / f32::from(columns);
            let variance = values
                .iter()
                .map(|value| (value.to_f32() - mean).powi(2))
                .sum::<f32>()
                / f32::from(columns);
            let expected = (values[usize::from(column)].to_f32() - mean)
                * (variance + epsilon).sqrt().recip()
                * gamma[usize::from(column)].to_f32()
                + beta[usize::from(column)].to_f32();
            let observed = f16::from_bits(u16::from_le_bytes(
                actual[byte_offset..byte_offset + 2].try_into().unwrap(),
            ))
            .to_f32();
            assert!(
                observed.is_finite(),
                "non-finite output at [{row}, {column}]"
            );
            max_error = max_error.max((observed - expected).abs());
            byte_offset += 2;
        }
    }
    assert_eq!(byte_offset, usize::from(rows) * usize::from(columns) * 2);
    max_error
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

fn env_u16(name: &str, default: u16) -> u16 {
    std::env::var(name)
        .map(|value| value.parse().unwrap())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .map(|value| value.parse().unwrap())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .map(|value| value.parse().unwrap())
        .unwrap_or(default)
}
