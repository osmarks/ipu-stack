use ipu_compiler::{
    BlockedGemmConfig, BlockedGemmPlan, GemmDataType, choose_gemm_row_block, plan_blocked_gemm,
};
use ipu_elf::Toolchain;
use ipu_runtime::{
    BlockLayout, ExecutableGraph, HostRunOptions, ProfileGranularity, allocator_memory_profile,
    block_binding, block_coordinates, blocked_matrix, package_graph, package_graph_profiled_with,
    run_host_with_inspector,
};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

const TILE_COUNT: u16 = 1472;
const BLOCK_DIMENSION: u16 = 64;
const DEFAULT_INNER_BLOCK_DIMENSION: u16 = 64;
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
    let inner_block_dimension = std::env::var("IPU_GEMM_INNER_BLOCK")
        .map(|value| {
            value
                .parse::<u16>()
                .expect("IPU_GEMM_INNER_BLOCK must be a u16")
        })
        .unwrap_or(DEFAULT_INNER_BLOCK_DIMENSION);
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
    let compile_start = Instant::now();
    let row_block_dimension = std::env::var("IPU_GEMM_ROW_BLOCK")
        .map(|value| {
            value
                .parse::<u16>()
                .expect("IPU_GEMM_ROW_BLOCK must be a u16")
        })
        .unwrap_or_else(|_| {
            choose_gemm_row_block(
                dimension,
                inner_block_dimension,
                dimension,
                BLOCK_DIMENSION,
                TILE_COUNT,
            )
            .expect("GEMM shape has no feasible row blocking")
        });
    let plan = plan_blocked_gemm(BlockedGemmConfig {
        rows: dimension,
        inner_dimension: dimension,
        columns: dimension,
        block_dimension: BLOCK_DIMENSION,
        inner_block_dimension,
        row_block_dimension,
        tile_count: TILE_COUNT,
        data_base: GEMM_DATA_BASE,
        data_limit: ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
        data_type: GemmDataType::F32,
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
    let kernel_source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../device/gemm_f32_64_amp.S");
    let kernel = toolchain
        .compile(
            &kernel_source,
            &output,
            "gemm-f32-64",
            &[
                format!("-DGEMM_INNER_BLOCK_DIMENSION={inner_block_dimension}"),
                format!("-DGEMM_SMALL_ROWS={minimum_rows}"),
                format!("-DGEMM_LARGE_ROWS={maximum_rows}"),
            ],
        )
        .unwrap();
    let (graph, input) = gemm_graph_and_input(dimension, plan);
    if let Some(path) = std::env::var_os("IPU_MEMORY_PROFILE_OUTPUT") {
        let profile = allocator_memory_profile(&graph).unwrap();
        profile.write(fs::File::create(&path).unwrap()).unwrap();
        info!(path = %PathBuf::from(path).display(), tiles = profile.tiles.len(), "wrote allocator memory profile");
    }
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
        let granularity = ProfileGranularity::from_environment().unwrap();
        let (app, layout) = package_graph_profiled_with(&graph, &objects, granularity).unwrap();
        info!(?granularity, "enabled cycle profiling");
        (app, Some(layout))
    } else {
        (package_graph(&graph, &objects).unwrap(), None)
    };
    info!(
        dimension,
        inner_block_dimension,
        row_block_dimension,
        row_shards = output_placements
            .iter()
            .map(|block| block.block_row)
            .max()
            .map_or(0, |row| row + 1),
        minimum_rows,
        maximum_rows,
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

    let runs = std::env::var("IPU_GEMM_RUNS")
        .map(|value| {
            value
                .parse::<usize>()
                .expect("IPU_GEMM_RUNS must be an integer")
        })
        .unwrap_or(1);
    assert!(runs != 0, "IPU_GEMM_RUNS must be nonzero");
    assert!(
        profile_output.is_none() || runs == 1,
        "cycle profiling currently requires exactly one GEMM run"
    );
    let diagnostic_block = diagnostic_block(dimension);
    let options = HostRunOptions::from_environment().unwrap();
    let all_runs_start = Instant::now();
    for run_index in 0..runs {
        let run_start = Instant::now();
        let actual = run_host_with_inspector(
            &app,
            &bootloader,
            &configuration,
            &device,
            &input,
            options,
            |device, output| {
                inspect_output(
                    device,
                    dimension,
                    output,
                    &output_placements,
                    diagnostic_block,
                )
            },
        )
        .unwrap_or_else(|error| panic!("GEMM run {}/{} failed: {error}", run_index + 1, runs));
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
            let floating_point_operations = 2.0 * f64::from(dimension).powi(3);
            let tflops = (graph_seconds != 0.0)
                .then_some(floating_point_operations / graph_seconds / 1.0e12);
            report.write(fs::File::create(path).unwrap()).unwrap();
            info!(
                path = %path.display(),
                tiles = report.tiles.len(),
                graph_cycles,
                graph_ms = graph_seconds * 1.0e3,
                tflops,
                "wrote cycle profile"
            );
        }
        info!(
            dimension,
            completed = run_index + 1,
            runs,
            run_ms = run_start.elapsed().as_millis(),
            output_bytes = actual.len(),
            "blocked FP32 GEMM iteration passed"
        );
    }
    info!(
        dimension,
        runs,
        elapsed_ms = all_runs_start.elapsed().as_millis(),
        "all blocked FP32 GEMM iterations passed"
    );
    let _ = fs::remove_dir_all(output);
}

fn gemm_graph_and_input(dimension: u16, plan: BlockedGemmPlan) -> (ExecutableGraph, Vec<u8>) {
    let left_binding = block_binding("left", dimension, dimension, &plan.left);
    let right_binding = block_binding("right", dimension, dimension, &plan.right);
    let output_binding = block_binding("output", dimension, dimension, &plan.output);
    let mut input = blocked_matrix(&plan.left, BlockLayout::AmpA8, left_value);
    input.extend(blocked_matrix(
        &plan.right,
        BlockLayout::AmpB8x16,
        right_value,
    ));
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

#[derive(Clone, Copy)]
struct GemmMismatch {
    row: u16,
    column: u16,
    actual: u32,
    expected: u32,
}

struct BlockComparison {
    transport_differences: usize,
    sram_expected_differences: usize,
}

fn inspect_output(
    device: &ipu_driver::Device,
    dimension: u16,
    actual: &[u8],
    placements: &[ipu_compiler::BlockPlacement],
    requested_block: Option<(u16, u16)>,
) -> ipu_runtime::Result<()> {
    let expected_bytes = usize::from(dimension) * usize::from(dimension) * 4;
    let actual = actual
        .get(..expected_bytes)
        .ok_or("GEMM output is shorter than the matrix")?;
    let mismatch = find_mismatch(dimension, actual, placements);
    let mismatch_block = mismatch.and_then(|item| {
        placements
            .iter()
            .find(|placement| {
                (placement.row_start..placement.row_start + placement.rows).contains(&item.row)
                    && placement.block_column == item.column / BLOCK_DIMENSION
            })
            .map(|placement| (placement.block_row, placement.block_column))
    });
    let comparison = if let Some((block_row, block_column)) = mismatch_block.or(requested_block) {
        Some(compare_sram_block(
            device,
            dimension,
            actual,
            placements,
            block_row,
            block_column,
            mismatch,
        )?)
    } else {
        None
    };
    if let Some(mismatch) = mismatch {
        let comparison = comparison.expect("a mismatched output always selects its SRAM block");
        return Err(format!(
            "GEMM mismatch at [{}, {}]: D2H={:#010x} expected={:#010x}; block SRAM-vs-D2H differences={}, SRAM-vs-expected differences={}",
            mismatch.row,
            mismatch.column,
            mismatch.actual,
            mismatch.expected,
            comparison.transport_differences,
            comparison.sram_expected_differences
        )
        .into());
    }
    Ok(())
}

fn find_mismatch(
    dimension: u16,
    actual: &[u8],
    placements: &[ipu_compiler::BlockPlacement],
) -> Option<GemmMismatch> {
    let expected = (0..7)
        .map(|row| {
            (0..5)
                .map(|column| expected_value(dimension, row, column))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut offset = 0;
    for placement in placements {
        for linear in 0..placement.rows * placement.columns {
            let (row, column) = block_coordinates(
                BlockLayout::AmpC16,
                placement.rows,
                placement.columns,
                linear,
            );
            let global_row = placement.row_start + row;
            let global_column = placement.column_start + column;
            let actual_value = f32::from_le_bytes(actual[offset..offset + 4].try_into().unwrap());
            let expected = expected[usize::from(global_row % 7)][usize::from(global_column % 5)];
            if actual_value.to_bits() != expected.to_bits() {
                return Some(GemmMismatch {
                    row: global_row,
                    column: global_column,
                    actual: actual_value.to_bits(),
                    expected: expected.to_bits(),
                });
            }
            offset += 4;
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn compare_sram_block(
    device: &ipu_driver::Device,
    dimension: u16,
    d2h: &[u8],
    placements: &[ipu_compiler::BlockPlacement],
    block_row: u16,
    block_column: u16,
    mismatch: Option<GemmMismatch>,
) -> ipu_runtime::Result<BlockComparison> {
    let block_index = placements
        .iter()
        .position(|placement| {
            placement.block_row == block_row && placement.block_column == block_column
        })
        .ok_or("diagnostic GEMM block is outside the placement table")?;
    let placement = &placements[block_index];
    let physical_tile = ipu_exchange::Topology::c600().physical(placement.tile)?;
    let block_words = u32::from(placement.rows) * u32::from(placement.columns);
    let sram = device.read_tile_words_from_inactive_context(
        physical_tile,
        1,
        placement.address,
        block_words,
    )?;
    let d2h_start = placements[..block_index]
        .iter()
        .map(|placement| usize::from(placement.rows) * usize::from(placement.columns) * 4)
        .sum::<usize>();
    let block_bytes = usize::try_from(block_words)? * 4;
    let d2h = &d2h[d2h_start..d2h_start + block_bytes];
    let mut transport_differences = 0usize;
    let mut sram_expected_differences = 0usize;
    let mut first_transport_difference = None;
    for (index, &sram_word) in sram.iter().enumerate() {
        let d2h_word = u32::from_le_bytes(d2h[index * 4..index * 4 + 4].try_into().unwrap());
        if d2h_word != sram_word {
            transport_differences += 1;
            first_transport_difference.get_or_insert((index, d2h_word, sram_word));
        }
        let (row, column) = block_coordinates(
            BlockLayout::AmpC16,
            placement.rows,
            placement.columns,
            u16::try_from(index)?,
        );
        let row = placement.row_start + row;
        let column = placement.column_start + column;
        if sram_word != expected_value(dimension, row, column).to_bits() {
            sram_expected_differences += 1;
        }
    }
    info!(
        block_row,
        block_column,
        logical_tile = placement.tile,
        physical_tile,
        address = format_args!("0x{:x}", placement.address),
        transport_differences,
        sram_expected_differences,
        "compared GEMM output block in SRAM with D2H"
    );
    if mismatch.is_none() && transport_differences != 0 {
        return Err(format!(
            "D2H differs from SRAM in {transport_differences} words of block ({block_row}, {block_column}); first difference {first_transport_difference:?}"
        )
        .into());
    }
    Ok(BlockComparison {
        transport_differences,
        sram_expected_differences,
    })
}

fn diagnostic_block(dimension: u16) -> Option<(u16, u16)> {
    let value = std::env::var("IPU_GEMM_SRAM_CHECK_BLOCK").ok()?;
    let (row, column) = value
        .split_once(',')
        .expect("IPU_GEMM_SRAM_CHECK_BLOCK must be ROW,COLUMN");
    let block = (
        row.parse::<u16>()
            .expect("diagnostic block row must be a u16"),
        column
            .parse::<u16>()
            .expect("diagnostic block column must be a u16"),
    );
    let grid = dimension / BLOCK_DIMENSION;
    assert!(
        block.0 < grid && block.1 < grid,
        "diagnostic block is outside the GEMM grid"
    );
    Some(block)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn amp_block_layouts_are_bijections() {
        for (layout, rows, columns) in [
            (BlockLayout::AmpA8, 128, 32),
            (BlockLayout::AmpA16, 128, 32),
            (BlockLayout::AmpB8x16, 32, 64),
            (BlockLayout::AmpB16x16, 32, 64),
            (BlockLayout::AmpC16, 128, 64),
            (BlockLayout::AmpC16F16, 128, 64),
        ] {
            let coordinates = (0..rows * columns)
                .map(|linear| block_coordinates(layout, rows, columns, linear))
                .collect::<BTreeSet<_>>();
            assert_eq!(coordinates.len(), usize::from(rows) * usize::from(columns));
            assert!(
                coordinates
                    .iter()
                    .all(|&(row, column)| { row < rows && column < columns })
            );
        }
    }
}
