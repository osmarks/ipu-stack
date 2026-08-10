use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use ipu_codegen::{
    AmpOrder, ComputeGraph, Layout, PackageConfig, PipelineConfig, Precision, TensorFormat,
    amp_matrix_coordinates, build_package,
};
use ipu_elf::Toolchain;
use ipu_package::{Application, Binding};
use ipu_runtime::Runtime;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(version, about = "Build, load, and diagnose the trivial IPU21 package")]
struct Arguments {
    /// Captured IPUCFG1 device configuration.
    configuration: PathBuf,
    #[arg(long, env = "POPLAR_SDK_ENABLED")]
    sdk: PathBuf,
    #[arg(long)]
    bootloader: Option<PathBuf>,
    #[arg(long, default_value = "/dev/ipu0")]
    device: String,
    #[arg(long, default_value = "/tmp/ipu-trivial.ipuexe")]
    package: PathBuf,
    /// Load an existing --package without rebuilding it.
    #[arg(long)]
    reuse_package: bool,
    /// Write per-tile kernel and exchange cycle samples for benchmark runs.
    #[arg(long, conflicts_with = "no_profile")]
    profile_output: Option<PathBuf>,
    /// Build benchmark programs without cycle-counter or per-step profiling.
    #[arg(long)]
    no_profile: bool,
    /// Log exchange scheduling lower bounds and critical dependency chains.
    #[arg(long)]
    exchange_diagnostics: bool,
    /// Force eligible one-use layout conversions to stream into consumer slices.
    #[arg(long)]
    stream_conversions: bool,
    /// Constrain planning as though only this much SRAM per tile were free.
    #[arg(long, conflicts_with = "reuse_package")]
    tile_memory_budget_kib: Option<u64>,
    #[arg(long, default_value_t = c600_tile_count())]
    tiles: u32,
    #[arg(long)]
    runtime_source: Option<PathBuf>,
    #[arg(long, default_value_t = 10)]
    timeout_seconds: u64,
    /// Program and validation workload to build and run.
    #[arg(long, value_enum, default_value_t = Workload::Diagnostic)]
    workload: Workload,
    #[arg(long, default_value_t = SIGLIP_MLP_BATCH)]
    mlp_batch: u32,
    #[arg(long, default_value_t = SIGLIP_MLP_TOKENS)]
    mlp_tokens: u32,
    #[arg(long, default_value_t = SIGLIP_MLP_DIMENSION)]
    mlp_dim: u32,
    /// Defaults to four times --mlp-dim.
    #[arg(long)]
    mlp_hidden_dim: Option<u32>,
    /// Number of sequential MLP blocks, represented by one structured repeat.
    #[arg(long, default_value_t = 1)]
    mlp_blocks: u32,
    /// Global GEMM row count, independent of the configured tile count.
    #[arg(long, default_value_t = GEMM_BENCHMARK_ROWS)]
    benchmark_rows: u32,
    #[arg(long, default_value_t = GEMM_BENCHMARK_INNER)]
    benchmark_inner: u32,
    #[arg(long, default_value_t = GEMM_BENCHMARK_COLUMNS)]
    benchmark_columns: u32,
    #[arg(long, default_value_t = 1_500_000_000)]
    clock_hz: u64,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Workload {
    /// Load the minimal runtime and report tile completion diagnostics.
    Diagnostic,
    /// Numerically verify one blocked F16 GEMM.
    GemmSmoke,
    /// Numerically verify an F16 GEMM with batched activations.
    BatchedGemmSmoke,
    /// Numerically verify GEMM-GeLU-GEMM-GeLU with Gaussian data.
    MlpSmoke,
    /// Profile one compute-dense F16 GEMM.
    GemmBenchmark,
    /// Profile the canonical batched SigLIP Dense-GeLU-Dense workload.
    SiglipMlpBenchmark,
}

impl Workload {
    fn is_benchmark(self) -> bool {
        matches!(self, Self::GemmBenchmark | Self::SiglipMlpBenchmark)
    }
}

const SIGLIP_MLP_BATCH: u32 = 4;
const SIGLIP_MLP_TOKENS: u32 = 256;
const SIGLIP_MLP_DIMENSION: u32 = 1024;
const GEMM_BENCHMARK_ROWS: u32 = 131_072;
const GEMM_BENCHMARK_INNER: u32 = 64;
const GEMM_BENCHMARK_COLUMNS: u32 = 64;

fn mlp_weight_name(blocks: u32, block: u32, projection: u32) -> String {
    if blocks == 1 {
        format!("right.{projection}")
    } else {
        format!("right.{block}.{projection}")
    }
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    ipu_runtime::init_tracing();
    let arguments = Arguments::parse();
    if !arguments.workload.is_benchmark()
        && (arguments.profile_output.is_some() || arguments.no_profile)
    {
        bail!("--profile-output and --no-profile require a benchmark workload");
    }
    if !matches!(arguments.workload, Workload::SiglipMlpBenchmark)
        && (arguments.mlp_batch != SIGLIP_MLP_BATCH
            || arguments.mlp_tokens != SIGLIP_MLP_TOKENS
            || arguments.mlp_dim != SIGLIP_MLP_DIMENSION
            || arguments.mlp_hidden_dim.is_some()
            || arguments.mlp_blocks != 1)
    {
        bail!("--mlp-* shape options require --workload siglip-mlp-benchmark");
    }
    if !matches!(arguments.workload, Workload::GemmBenchmark)
        && (arguments.benchmark_rows != GEMM_BENCHMARK_ROWS
            || arguments.benchmark_inner != GEMM_BENCHMARK_INNER
            || arguments.benchmark_columns != GEMM_BENCHMARK_COLUMNS)
    {
        bail!("--benchmark-* shape options require --workload gemm-benchmark");
    }
    let active_tiles = u16::try_from(arguments.tiles).context("tile count exceeds u16")?;
    let benchmark_rows = arguments.benchmark_rows;
    let mlp_hidden_dim = arguments.mlp_hidden_dim.map_or_else(
        || {
            arguments
                .mlp_dim
                .checked_mul(4)
                .context("MLP dimension overflow")
        },
        Ok,
    )?;
    let runtime_source = arguments.runtime_source.unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../device/static_runtime.S")
    });
    let bootloader = arguments
        .bootloader
        .unwrap_or_else(|| arguments.sdk.join("bin/ipu/tile_bootloader_cc_ipu21.elf"));
    let mut graph = ComputeGraph::default();
    let mut pipeline = PipelineConfig::new(active_tiles);
    if let Some(kib) = arguments.tile_memory_budget_kib {
        let bytes = kib.checked_mul(1024).context("tile SRAM budget overflow")?;
        pipeline = pipeline.with_tile_memory_budget(bytes);
    }
    pipeline.exchange_diagnostics = arguments.exchange_diagnostics;
    if arguments.stream_conversions {
        pipeline.conversion_streaming = ipu_codegen::ConversionStreamingPolicy::Always;
    }
    if matches!(
        arguments.workload,
        Workload::GemmSmoke | Workload::BatchedGemmSmoke
    ) {
        let batch = if matches!(arguments.workload, Workload::BatchedGemmSmoke) {
            3
        } else {
            1
        };
        let left = graph.host_input("left", [batch, u32::from(active_tiles), 64])?;
        let right = graph.parameter("right", [1, 64, u32::from(active_tiles) * 64])?;
        let output = graph.gemm(left, right)?;
        graph.set_outputs([output])?;
        pipeline = pipeline
            .with_input(
                left,
                TensorFormat {
                    precision: Precision::F16,
                    layout: Layout::amp_left(64, active_tiles),
                },
            )
            .with_input(
                right,
                TensorFormat {
                    precision: Precision::F16,
                    layout: Layout::amp_right(64, active_tiles),
                },
            );
    } else if matches!(arguments.workload, Workload::MlpSmoke) {
        let left = graph.host_input("left", [u32::from(active_tiles), 64])?;
        let right0 = graph.parameter("right.0", [64, 64])?;
        let right1 = graph.parameter("right.1", [64, 64])?;
        let hidden = graph.gemm(left, right0)?;
        let hidden = graph.gelu(hidden)?;
        let output = graph.gemm(hidden, right1)?;
        let output = graph.gelu(output)?;
        graph.set_outputs([output])?;
        let left_format = TensorFormat {
            precision: Precision::F16,
            layout: Layout::amp_left(64, active_tiles),
        };
        let right_format = TensorFormat {
            precision: Precision::F16,
            layout: Layout::amp_right(64, active_tiles),
        };
        pipeline = pipeline
            .with_input(left, left_format)
            .with_input(right0, right_format.clone())
            .with_input(right1, right_format);
    } else if matches!(arguments.workload, Workload::SiglipMlpBenchmark) {
        validate_mlp_benchmark_shape(
            arguments.mlp_batch,
            arguments.mlp_tokens,
            arguments.mlp_dim,
            mlp_hidden_dim,
        )?;
        if arguments.mlp_blocks == 0 {
            bail!("--mlp-blocks must be nonzero");
        }
        let left = graph.host_input(
            "left",
            [arguments.mlp_batch, arguments.mlp_tokens, arguments.mlp_dim],
        )?;
        let mut right0 = Vec::with_capacity(arguments.mlp_blocks as usize);
        let mut right1 = Vec::with_capacity(arguments.mlp_blocks as usize);
        for block in 0..arguments.mlp_blocks {
            right0.push(graph.parameter(
                mlp_weight_name(arguments.mlp_blocks, block, 0),
                [1, arguments.mlp_dim, mlp_hidden_dim],
            )?);
            right1.push(graph.parameter(
                mlp_weight_name(arguments.mlp_blocks, block, 1),
                [1, mlp_hidden_dim, arguments.mlp_dim],
            )?);
        }
        let right0_sequence = graph.value_sequence("MLP up weights", right0.clone())?;
        let right1_sequence = graph.value_sequence("MLP down weights", right1.clone())?;
        let output = graph.repeat(
            arguments.mlp_blocks,
            [left],
            [],
            [right0_sequence, right1_sequence],
            |body, arguments| {
                let hidden = body.gemm(arguments.carried[0], arguments.iterated[0])?;
                let hidden = body.gelu(hidden)?;
                Ok(vec![body.gemm(hidden, arguments.iterated[1])?])
            },
        )?[0];
        graph.set_outputs([output])?;
        pipeline.profiling.enabled = !arguments.no_profile;
        pipeline = pipeline.with_automatic_input(left, Precision::F16);
        for weight in right0.into_iter().chain(right1) {
            pipeline = pipeline.with_automatic_input(weight, Precision::F16);
        }
    } else if matches!(arguments.workload, Workload::GemmBenchmark) {
        validate_benchmark_shape(
            benchmark_rows,
            arguments.benchmark_inner,
            arguments.benchmark_columns,
        )?;
        let left = graph.host_input("left", [benchmark_rows, arguments.benchmark_inner])?;
        let right = graph.parameter(
            "right",
            [arguments.benchmark_inner, arguments.benchmark_columns],
        )?;
        let output = graph.gemm(left, right)?;
        graph.set_outputs([output])?;
        pipeline.profiling.enabled = !arguments.no_profile;
        pipeline = pipeline
            .with_automatic_input(left, Precision::F16)
            .with_automatic_input(right, Precision::F16);
    }
    if !arguments.reuse_package {
        let application = build_package(
            &graph,
            &PackageConfig {
                toolchain: Toolchain::from_sdk(&arguments.sdk),
                kernel_source_directory: runtime_source
                    .parent()
                    .expect("runtime source has no parent directory")
                    .to_owned(),
                runtime_source,
                pipeline,
            },
        )?;
        write_package(&application, &arguments.package)?;
    }
    let application = Application::read(
        fs::File::open(&arguments.package)
            .with_context(|| format!("open {}", arguments.package.display()))?,
    )?;

    let configuration = fs::read(&arguments.configuration)
        .with_context(|| format!("read {}", arguments.configuration.display()))?;
    let bootloader_bytes =
        fs::read(&bootloader).with_context(|| format!("read {}", bootloader.display()))?;
    let runtime = Runtime::open(&arguments.device, &configuration)?;
    if !matches!(arguments.workload, Workload::Diagnostic) {
        runtime.load(
            &application,
            &bootloader_bytes,
            application.host_exchange.startup_mark,
        )?;
        if matches!(
            arguments.workload,
            Workload::GemmSmoke | Workload::BatchedGemmSmoke
        ) {
            run_gemm(
                &runtime,
                &application,
                active_tiles,
                if matches!(arguments.workload, Workload::BatchedGemmSmoke) {
                    3
                } else {
                    1
                },
                arguments.timeout_seconds,
            )?;
        } else if matches!(arguments.workload, Workload::MlpSmoke) {
            run_mlp_chain(
                &runtime,
                &application,
                active_tiles,
                arguments.timeout_seconds,
            )?;
        } else if matches!(arguments.workload, Workload::SiglipMlpBenchmark) {
            run_siglip_mlp_benchmark(
                &runtime,
                &application,
                active_tiles,
                arguments.mlp_batch,
                arguments.mlp_tokens,
                arguments.mlp_dim,
                mlp_hidden_dim,
                arguments.mlp_blocks,
                arguments.clock_hz,
                arguments.timeout_seconds,
                arguments.profile_output.as_deref(),
                !arguments.no_profile,
            )?;
        } else {
            run_gemm_benchmark(
                &runtime,
                &application,
                active_tiles,
                benchmark_rows,
                arguments.benchmark_inner,
                arguments.benchmark_columns,
                arguments.clock_hz,
                arguments.timeout_seconds,
                arguments.profile_output.as_deref(),
                !arguments.no_profile,
            )?;
        }
    } else {
        runtime.load(&application, &bootloader_bytes, 0)?;
        diagnose_completion(
            &runtime,
            &application,
            Duration::from_secs(arguments.timeout_seconds),
        )?;
    }
    println!(
        "package={} tiles={} hardwareTest=PASS",
        arguments.package.display(),
        application.tiles.len()
    );
    Ok(())
}

fn run_gemm(
    runtime: &Runtime,
    application: &Application,
    active_tiles: u16,
    batch: u32,
    timeout_seconds: u64,
) -> Result<()> {
    let left = application
        .inputs
        .iter()
        .find(|binding| binding.name == "left")
        .cloned()
        .context("GEMM package has no left input binding")?;
    let right = application
        .weights
        .iter()
        .find(|binding| binding.name == "right")
        .cloned()
        .context("GEMM package has no right weight binding")?;
    let left_bytes = packed_binding(&left, |logical_tile, linear, elements| {
        let (batch_index, inner) = amp_matrix_coordinates(
            AmpOrder::Left,
            Precision::F16,
            batch,
            elements / batch,
            linear,
        )?;
        let selected_inner = (batch_index * 7 + u32::from(logical_tile)) % 64;
        Ok(if selected_inner == inner { 0x3c00 } else { 0 })
    })?;
    let right_bytes = packed_binding(&right, |logical_tile, linear, elements| {
        let (inner, column) =
            amp_matrix_coordinates(AmpOrder::Right, Precision::F16, 64, elements / 64, linear)?;
        Ok(gemm_right_value(
            inner,
            u32::from(logical_tile) * 64 + column,
        ))
    })?;
    if left.slices.len() != usize::from(active_tiles)
        || right.slices.len() != usize::from(active_tiles)
    {
        bail!("GEMM bindings do not cover every active tile");
    }
    let output = run_initialized_program(
        runtime,
        application,
        &right_bytes,
        &left_bytes,
        timeout_seconds,
    )?;
    verify_gemm_output(application, active_tiles, batch, &output)
}

fn run_mlp_chain(
    runtime: &Runtime,
    application: &Application,
    active_tiles: u16,
    timeout_seconds: u64,
) -> Result<()> {
    let binding = |name: &str, bindings: &[Binding]| {
        bindings
            .iter()
            .find(|binding| binding.name == name)
            .cloned()
            .with_context(|| format!("MLP package has no {name} binding"))
    };
    let left = binding("left", &application.inputs)?;
    let right0 = binding("right.0", &application.weights)?;
    let right1 = binding("right.1", &application.weights)?;
    let left_bytes = packed_binding(&left, |logical_tile, linear, elements| {
        let (_, inner) =
            amp_matrix_coordinates(AmpOrder::Left, Precision::F16, 1, elements, linear)?;
        Ok(mlp_smoke_value(
            MLP_INPUT_SEED,
            u64::from(logical_tile) * u64::from(MLP_SMOKE_WIDTH) + u64::from(inner),
            MLP_INPUT_STANDARD_DEVIATION,
        ))
    })?;
    let right0_bytes = packed_binding(&right0, |logical_tile, linear, elements| {
        let (inner, column) =
            amp_matrix_coordinates(AmpOrder::Right, Precision::F16, 64, elements / 64, linear)?;
        let column = u32::from(logical_tile) * 64 + column;
        Ok(if column < 64 {
            mlp_smoke_value(
                MLP_FIRST_WEIGHT_SEED,
                u64::from(inner) * u64::from(MLP_SMOKE_WIDTH) + u64::from(column),
                MLP_WEIGHT_STANDARD_DEVIATION,
            )
        } else {
            0
        })
    })?;
    let right1_bytes = packed_binding(&right1, |logical_tile, linear, elements| {
        let (inner, column) =
            amp_matrix_coordinates(AmpOrder::Right, Precision::F16, 64, elements / 64, linear)?;
        let column = u32::from(logical_tile) * 64 + column;
        Ok(if column < 64 {
            mlp_smoke_value(
                MLP_SECOND_WEIGHT_SEED,
                u64::from(inner) * u64::from(MLP_SMOKE_WIDTH) + u64::from(column),
                MLP_WEIGHT_STANDARD_DEVIATION,
            )
        } else {
            0
        })
    })?;
    let mut weights = Vec::with_capacity(right0_bytes.len() + right1_bytes.len());
    weights.extend_from_slice(&right0_bytes);
    weights.extend_from_slice(&right1_bytes);

    let output =
        run_initialized_program(runtime, application, &weights, &left_bytes, timeout_seconds)?;
    verify_mlp_output(application, active_tiles, &output)
}

fn run_initialized_program(
    runtime: &Runtime,
    application: &Application,
    weights: &[u8],
    input: &[u8],
    timeout_seconds: u64,
) -> Result<Vec<u8>> {
    let mut session = runtime.host_session(application)?;
    session.start().inspect_err(|_| {
        eprintln!(
            "startFailureDiagnostics={}",
            device_failure_diagnostics(runtime, application)
        );
    })?;
    let initialized = session.invoke_streaming_deferred("initialize", weights)?;
    session.collect(&initialized)?;
    let executed = session
        .invoke_streaming_deferred("run", input)
        .inspect_err(|_| {
            eprintln!(
                "runFailureDiagnostics={}",
                device_failure_diagnostics(runtime, application)
            );
        })?;
    runtime
        .device()
        .write_sync_mark(ipu_driver::pci::HSP_GS2_CONTROL, 1)?;
    diagnose_completion(runtime, application, Duration::from_secs(timeout_seconds))?;
    Ok(session.collect(&executed)?)
}

#[allow(clippy::too_many_arguments)]
fn run_gemm_benchmark(
    runtime: &Runtime,
    application: &Application,
    execution_tiles: u16,
    rows: u32,
    inner: u32,
    columns: u32,
    clock_hz: u64,
    timeout_seconds: u64,
    profile_output: Option<&Path>,
    profiling_enabled: bool,
) -> Result<()> {
    validate_benchmark_shape(rows, inner, columns)?;
    if clock_hz == 0 {
        bail!("benchmark clock must be nonzero");
    }
    let left = application
        .inputs
        .iter()
        .find(|binding| binding.name == "left")
        .context("benchmark package has no left binding")?;
    let right = application
        .weights
        .iter()
        .find(|binding| binding.name == "right")
        .context("benchmark package has no right binding")?;
    let left_bytes = filled_f16_binding(left, 0x3c00)?;
    let right_bytes = filled_f16_binding(right, 0x3400)?;
    let output = run_initialized_program(
        runtime,
        application,
        &right_bytes,
        &left_bytes,
        timeout_seconds,
    )?;
    let maximum_absolute_error = verify_benchmark_output(application, &output, inner)?;
    if !profiling_enabled {
        println!(
            "workload=gemm-f16-r{rows}-k{inner}-c{columns} benchmark=gemm-f16 rows={rows} inner={inner} columns={columns} profiling=false maximumAbsoluteError={maximum_absolute_error:.6}"
        );
        return Ok(());
    }
    let (cycles, minimum_cycles) = benchmark_cycles(application, &output, execution_tiles)?;
    write_profile(application, &output, clock_hz, profile_output)?;
    let active_tiles = binding_tile_count(application, "output.0")?;
    let rows = u64::from(rows);
    let flops = 2.0 * rows as f64 * f64::from(inner) * f64::from(columns);
    let seconds = f64::from(cycles) / clock_hz as f64;
    let tflops = flops / seconds / 1.0e12;
    let peak_tflops = clock_hz as f64 * f64::from(active_tiles) * 128.0 / 1.0e12;
    println!(
        "workload=gemm-f16-r{rows}-k{inner}-c{columns} benchmark=gemm-f16 rows={rows} inner={inner} columns={columns} activeTiles={active_tiles} executionTiles={execution_tiles} inputBytes={} weightBytes={} cycles={cycles} minimumTileCycles={minimum_cycles} deviceMicroseconds={:.3} tflops={tflops:.3} peakTflops={peak_tflops:.3} efficiencyPercent={:.2} maximumAbsoluteError={maximum_absolute_error:.6}",
        left_bytes.len(),
        right_bytes.len(),
        seconds * 1.0e6,
        tflops / peak_tflops * 100.0,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_siglip_mlp_benchmark(
    runtime: &Runtime,
    application: &Application,
    execution_tiles: u16,
    batch: u32,
    tokens: u32,
    dimension: u32,
    hidden_dimension: u32,
    blocks: u32,
    clock_hz: u64,
    timeout_seconds: u64,
    profile_output: Option<&Path>,
    profiling_enabled: bool,
) -> Result<()> {
    validate_mlp_benchmark_shape(batch, tokens, dimension, hidden_dimension)?;
    if clock_hz == 0 {
        bail!("benchmark clock must be nonzero");
    }
    let left = application
        .inputs
        .iter()
        .find(|binding| binding.name == "left")
        .context("MLP benchmark package has no left binding")?;
    let left_bytes = filled_f16_binding(left, 0x3c00)?;
    let up_bits = f32_to_half(1.0 / dimension as f32);
    let down_bits = f32_to_half(1.0 / hidden_dimension as f32);
    let mut weights = Vec::new();
    for block in 0..blocks {
        for (projection, bits) in [(0, up_bits), (1, down_bits)] {
            let name = mlp_weight_name(blocks, block, projection);
            let binding = application
                .weights
                .iter()
                .find(|binding| binding.name == name)
                .with_context(|| format!("MLP benchmark package has no {name} binding"))?;
            weights.extend_from_slice(&filled_f16_binding(binding, bits)?);
        }
    }

    let output =
        run_initialized_program(runtime, application, &weights, &left_bytes, timeout_seconds)?;

    let expected = (0..blocks).fold(1.0, |value, _| gelu_reference(value));
    let maximum_absolute_error = verify_constant_output(application, &output, expected)?;
    if !profiling_enabled {
        println!(
            "workload=siglip-mlp-f16-b{batch}-t{tokens}-d{dimension}-h{hidden_dimension}-n{blocks} benchmark=siglip-mlp-f16 batch={batch} tokens={tokens} dimension={dimension} hiddenDimension={hidden_dimension} blocks={blocks} biases=false profiling=false maximumAbsoluteError={maximum_absolute_error:.6}"
        );
        return Ok(());
    }
    let (cycles, minimum_cycles) = benchmark_cycles(application, &output, execution_tiles)?;
    write_profile(application, &output, clock_hz, profile_output)?;
    let active_tiles = binding_tile_count(application, "output.0")?;
    let rows = u64::from(batch) * u64::from(tokens);
    let flops =
        4.0 * rows as f64 * f64::from(dimension) * f64::from(hidden_dimension) * f64::from(blocks);
    let seconds = f64::from(cycles) / clock_hz as f64;
    let tflops = flops / seconds / 1.0e12;
    let peak_tflops = clock_hz as f64 * f64::from(active_tiles) * 128.0 / 1.0e12;
    println!(
        "workload=siglip-mlp-f16-b{batch}-t{tokens}-d{dimension}-h{hidden_dimension}-n{blocks} benchmark=siglip-mlp-f16 batch={batch} tokens={tokens} rows={rows} dimension={dimension} hiddenDimension={hidden_dimension} blocks={blocks} biases=false activeTiles={active_tiles} executionTiles={execution_tiles} inputBytes={} weightBytes={} cycles={cycles} minimumTileCycles={minimum_cycles} deviceMicroseconds={:.3} effectiveGemmTflops={tflops:.3} peakTflops={peak_tflops:.3} efficiencyPercent={:.2} maximumAbsoluteError={maximum_absolute_error:.6}",
        left_bytes.len(),
        weights.len(),
        seconds * 1.0e6,
        tflops / peak_tflops * 100.0,
    );
    Ok(())
}

fn write_profile(
    application: &Application,
    output: &[u8],
    clock_hz: u64,
    path: Option<&Path>,
) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    let report = application.profile_report(output, clock_hz)?;
    report.write(fs::File::create(path).with_context(|| format!("create {}", path.display()))?)?;
    println!(
        "profile={} tiles={} samples={}",
        path.display(),
        report.tiles.len(),
        report
            .tiles
            .iter()
            .map(|tile| tile.samples.len())
            .sum::<usize>()
    );
    Ok(())
}

fn benchmark_cycles(
    application: &Application,
    output: &[u8],
    active_tiles: u16,
) -> Result<(u32, u32)> {
    let starts = binding_u32_values(application, output, "profile.start-cycle")?;
    let ends = binding_u32_values(application, output, "profile.end-cycle")?;
    if starts.len() != usize::from(active_tiles) || ends.len() != starts.len() {
        bail!(
            "benchmark profile bindings have {} start and {} end samples for {active_tiles} tiles",
            starts.len(),
            ends.len()
        );
    }
    let durations = starts
        .into_iter()
        .zip(ends)
        .map(|(start, end)| end.wrapping_sub(start))
        .collect::<Vec<_>>();
    let maximum = durations.iter().copied().max().unwrap_or(0);
    if maximum == 0 {
        bail!("benchmark cycle interval is zero");
    }
    Ok((maximum, durations.iter().copied().min().unwrap_or(maximum)))
}

fn validate_benchmark_shape(rows: u32, inner: u32, columns: u32) -> Result<()> {
    if rows == 0
        || inner == 0
        || columns == 0
        || !inner.is_multiple_of(64)
        || !columns.is_multiple_of(64)
    {
        bail!("benchmark rows must be nonzero and inner/columns must be nonzero multiples of 64");
    }
    Ok(())
}

fn validate_mlp_benchmark_shape(
    batch: u32,
    tokens: u32,
    dimension: u32,
    hidden_dimension: u32,
) -> Result<()> {
    if batch == 0
        || tokens == 0
        || dimension == 0
        || hidden_dimension == 0
        || !dimension.is_multiple_of(64)
        || !hidden_dimension.is_multiple_of(64)
    {
        bail!(
            "MLP batch/tokens must be nonzero and feature dimensions must be nonzero multiples of 64"
        );
    }
    Ok(())
}

fn verify_benchmark_output(application: &Application, bytes: &[u8], inner: u32) -> Result<f32> {
    let (binding, base) = output_binding(application, "output.0")?;
    let size = binding_size(binding);
    if size == 0 || !size.is_multiple_of(2) {
        bail!("benchmark graph output is not a nonempty F16 binding");
    }
    let start = usize::try_from(base)?;
    let end = usize::try_from(
        base.checked_add(size)
            .context("benchmark output overflow")?,
    )?;
    let output = bytes
        .get(start..end)
        .context("benchmark graph output exceeds host output")?;
    let expected = inner as f32 * 0.25;
    let mut maximum = 0.0f32;
    for raw in output.chunks_exact(2) {
        let actual = half_to_f32(u16::from_le_bytes(raw.try_into().unwrap()));
        maximum = maximum.max((actual - expected).abs());
    }
    if maximum > expected.abs() * 0.002 + 0.01 {
        bail!(
            "benchmark numerical output differs from {expected}: maximum absolute error {maximum}"
        );
    }
    Ok(maximum)
}

fn verify_constant_output(application: &Application, bytes: &[u8], expected: f32) -> Result<f32> {
    let (binding, base) = output_binding(application, "output.0")?;
    let size = binding_size(binding);
    if size == 0 || !size.is_multiple_of(2) {
        bail!("MLP benchmark output is not a nonempty F16 binding");
    }
    let start = usize::try_from(base)?;
    let end = usize::try_from(base.checked_add(size).context("MLP output overflow")?)?;
    let output = bytes
        .get(start..end)
        .context("MLP benchmark output exceeds host output")?;
    let mut maximum = 0.0f32;
    let mut minimum_value = f32::INFINITY;
    let mut maximum_value = f32::NEG_INFINITY;
    let mut zero_values = 0usize;
    let mut unchanged_values = 0usize;
    for raw in output.chunks_exact(2) {
        let actual = half_to_f32(u16::from_le_bytes(raw.try_into().unwrap()));
        minimum_value = minimum_value.min(actual);
        maximum_value = maximum_value.max(actual);
        zero_values += usize::from(actual == 0.0);
        unchanged_values += usize::from(actual == 1.0);
        maximum = maximum.max((actual - expected).abs());
    }
    if maximum > expected.abs() * 0.02 + 0.05 {
        bail!(
            "MLP benchmark numerical output differs from {expected}: maximum absolute error {maximum}, observed range {minimum_value}..={maximum_value}, zeros={zero_values}, unchanged={unchanged_values}"
        );
    }
    Ok(maximum)
}

fn binding_u32_values(application: &Application, bytes: &[u8], name: &str) -> Result<Vec<u32>> {
    let (binding, base) = output_binding(application, name)?;
    let mut values = Vec::with_capacity(binding.slices.len());
    for slice in &binding.slices {
        if slice.size != 4 {
            bail!("profile binding {name} contains a non-u32 slice");
        }
        let start = usize::try_from(
            base.checked_add(slice.file_offset)
                .context("profile offset overflow")?,
        )?;
        let end = start.checked_add(4).context("profile offset overflow")?;
        let raw = bytes
            .get(start..end)
            .with_context(|| format!("profile binding {name} exceeds host output"))?;
        values.push(u32::from_le_bytes(raw.try_into().unwrap()));
    }
    Ok(values)
}

fn output_binding<'a>(application: &'a Application, name: &str) -> Result<(&'a Binding, u64)> {
    let mut base = 0u64;
    for binding in &application.outputs {
        if binding.name == name {
            return Ok((binding, base));
        }
        base = base
            .checked_add(binding_size(binding))
            .context("host output binding offset overflow")?;
    }
    bail!("package has no {name} output binding")
}

fn binding_tile_count(application: &Application, name: &str) -> Result<u16> {
    let (binding, _) = output_binding(application, name)?;
    let tiles = binding
        .slices
        .iter()
        .map(|slice| slice.tile)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    u16::try_from(tiles).context("binding tile count exceeds u16")
}

fn binding_size(binding: &Binding) -> u64 {
    binding
        .slices
        .iter()
        .map(|slice| slice.file_offset + slice.size)
        .max()
        .unwrap_or(0)
}

fn filled_f16_binding(binding: &Binding, bits: u16) -> Result<Vec<u8>> {
    let size = binding
        .slices
        .iter()
        .map(|slice| slice.file_offset + slice.size)
        .max()
        .context("binding has no slices")?;
    if size & 1 != 0 {
        bail!("binding {} has an odd byte count", binding.name);
    }
    let pair = bits.to_le_bytes();
    let mut bytes = vec![0; usize::try_from(size)?];
    for chunk in bytes.chunks_exact_mut(2) {
        chunk.copy_from_slice(&pair);
    }
    Ok(bytes)
}

fn verify_mlp_output(application: &Application, active_tiles: u16, bytes: &[u8]) -> Result<()> {
    let output = application
        .outputs
        .iter()
        .find(|binding| binding.name == "output.0")
        .context("MLP package has no output binding")?;
    let expected_bytes = output
        .slices
        .iter()
        .map(|slice| slice.file_offset + slice.size)
        .max()
        .context("MLP output has no slices")?;
    if bytes.len() != usize::try_from(expected_bytes)? {
        bail!(
            "MLP returned {} bytes, expected {expected_bytes}",
            bytes.len()
        );
    }
    let mut maximum_error = 0.0f32;
    let mut mismatches = Vec::new();
    let mut checked = 0usize;
    if output.slices.len() != usize::from(active_tiles) {
        bail!("MLP output does not cover every logical tile");
    }
    for (row, slice) in output.slices.iter().enumerate() {
        let row = u16::try_from(row)?;
        let expected_row = mlp_smoke_reference(row);
        let elements = u32::try_from(slice.size / 2)?;
        for linear in 0..elements {
            let (_, column) =
                amp_matrix_coordinates(AmpOrder::Left, Precision::F16, 1, elements, linear)?;
            if column >= 64 {
                continue;
            }
            let offset = usize::try_from(slice.file_offset + u64::from(linear) * 2)?;
            let actual = half_to_f32(u16::from_le_bytes(
                bytes[offset..offset + 2].try_into().unwrap(),
            ));
            let expected = expected_row[column as usize];
            let error = (actual - expected).abs();
            maximum_error = maximum_error.max(error);
            checked += 1;
            if error > 0.02 && mismatches.len() < 16 {
                mismatches.push((row, column, expected, actual, error));
            }
        }
    }
    if !mismatches.is_empty() {
        bail!("MLP numerical verification failed after {checked} checks: {mismatches:?}");
    }
    println!(
        "mlpNumericalChecks={checked} maximumAbsoluteError={maximum_error:.6} numericalTest=PASS"
    );
    Ok(())
}

fn gelu_reference(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value.powi(3))).tanh())
}

const MLP_SMOKE_WIDTH: u32 = 64;
const MLP_INPUT_SEED: u64 = 0x6d6c_705f_696e_7075;
const MLP_FIRST_WEIGHT_SEED: u64 = 0x6d6c_705f_7730_5f5f;
const MLP_SECOND_WEIGHT_SEED: u64 = 0x6d6c_705f_7731_5f5f;
const MLP_INPUT_STANDARD_DEVIATION: f32 = 0.5;
const MLP_WEIGHT_STANDARD_DEVIATION: f32 = 0.125;

fn mlp_smoke_value(seed: u64, index: u64, standard_deviation: f32) -> u16 {
    f32_to_half(gaussian(seed, index) * standard_deviation)
}

fn mlp_smoke_reference(row: u16) -> [f32; MLP_SMOKE_WIDTH as usize] {
    let mut hidden = [0.0; MLP_SMOKE_WIDTH as usize];
    for column in 0..MLP_SMOKE_WIDTH {
        let mut sum = 0.0;
        for inner in 0..MLP_SMOKE_WIDTH {
            let input = half_to_f32(mlp_smoke_value(
                MLP_INPUT_SEED,
                u64::from(row) * u64::from(MLP_SMOKE_WIDTH) + u64::from(inner),
                MLP_INPUT_STANDARD_DEVIATION,
            ));
            let weight = half_to_f32(mlp_smoke_value(
                MLP_FIRST_WEIGHT_SEED,
                u64::from(inner) * u64::from(MLP_SMOKE_WIDTH) + u64::from(column),
                MLP_WEIGHT_STANDARD_DEVIATION,
            ));
            sum += input * weight;
        }
        hidden[column as usize] = gelu_reference(half_to_f32(f32_to_half(sum)));
    }
    let mut output = [0.0; MLP_SMOKE_WIDTH as usize];
    for column in 0..MLP_SMOKE_WIDTH {
        let mut sum = 0.0;
        for inner in 0..MLP_SMOKE_WIDTH {
            let weight = half_to_f32(mlp_smoke_value(
                MLP_SECOND_WEIGHT_SEED,
                u64::from(inner) * u64::from(MLP_SMOKE_WIDTH) + u64::from(column),
                MLP_WEIGHT_STANDARD_DEVIATION,
            ));
            sum += half_to_f32(f32_to_half(hidden[inner as usize])) * weight;
        }
        output[column as usize] = gelu_reference(half_to_f32(f32_to_half(sum)));
    }
    output
}

fn gaussian(seed: u64, index: u64) -> f32 {
    let first = splitmix64(seed ^ index.wrapping_mul(2));
    let second = splitmix64(seed ^ index.wrapping_mul(2).wrapping_add(1));
    let unit = |bits: u64| ((bits >> 40) as f32 + 0.5) / 16_777_216.0;
    let radius = (-2.0 * unit(first).ln()).sqrt();
    radius * (std::f32::consts::TAU * unit(second)).cos()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn f32_to_half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | u16::from(fraction != 0);
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa = fraction | 0x80_0000;
        let shift = u32::try_from(14 - half_exponent).unwrap();
        let truncated = mantissa >> shift;
        let remainder = mantissa & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        return sign
            | (truncated
                + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0)))
                as u16;
    }
    let truncated = ((half_exponent as u32) << 10) | (fraction >> 13);
    let remainder = fraction & 0x1fff;
    sign | (truncated
        + u32::from(remainder > 0x1000 || (remainder == 0x1000 && truncated & 1 != 0)))
        as u16
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    let value = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            let normalized = fraction << shift;
            sign | ((113 - shift) << 23) | ((normalized & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(value)
}

fn packed_binding(
    binding: &Binding,
    mut value: impl FnMut(u16, u32, u32) -> Result<u16>,
) -> Result<Vec<u8>> {
    let total = binding
        .slices
        .iter()
        .map(|slice| slice.file_offset + slice.size)
        .max()
        .context("binding has no slices")?;
    let mut bytes = vec![0; usize::try_from(total)?];
    for (logical_tile, slice) in binding.slices.iter().enumerate() {
        if slice.size == 0 {
            continue;
        }
        if slice.size & 1 != 0 {
            bail!("binding {} has a non-F16 slice", binding.name);
        }
        let elements = u32::try_from(slice.size / 2)?;
        let logical_tile = u16::try_from(logical_tile)?;
        for linear in 0..elements {
            let bits = value(logical_tile, linear, elements)?;
            let offset = usize::try_from(slice.file_offset + u64::from(linear) * 2)?;
            bytes[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    Ok(bytes)
}

fn verify_gemm_output(
    application: &Application,
    active_tiles: u16,
    batch: u32,
    bytes: &[u8],
) -> Result<()> {
    let output = application
        .outputs
        .iter()
        .find(|binding| binding.name == "output.0")
        .context("GEMM package has no output binding")?;
    let expected_bytes = output
        .slices
        .iter()
        .map(|slice| slice.file_offset + slice.size)
        .max()
        .context("GEMM output has no slices")?;
    if bytes.len() != usize::try_from(expected_bytes)? {
        bail!(
            "GEMM returned {} bytes, expected {expected_bytes}",
            bytes.len()
        );
    }
    let mut mismatches = Vec::new();
    let mut checked = 0usize;
    if output.slices.len() != usize::from(active_tiles) {
        bail!("GEMM output does not cover every logical tile");
    }
    for (row, slice) in output.slices.iter().enumerate() {
        let row = u16::try_from(row)?;
        let elements = u32::try_from(slice.size / 2)?;
        for linear in 0..elements {
            let (batch_index, column) = amp_matrix_coordinates(
                AmpOrder::Output,
                Precision::F16,
                batch,
                u32::from(active_tiles) * 64,
                linear,
            )?;
            let offset = usize::try_from(slice.file_offset + u64::from(linear) * 2)?;
            let actual = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            let selected_inner = (batch_index * 7 + u32::from(row)) % 64;
            let expected = gemm_right_value(selected_inner, column);
            checked += 1;
            if actual != expected && mismatches.len() < 16 {
                mismatches.push((row, batch_index, column, expected, actual));
            }
        }
    }
    if !mismatches.is_empty() {
        bail!("GEMM numerical verification failed after {checked} checks: {mismatches:?}");
    }
    println!("gemmNumericalChecks={checked} numericalTest=PASS");
    Ok(())
}

fn gemm_right_value(inner: u32, column: u32) -> u16 {
    const VALUES: [u16; 5] = [0xbc00, 0xb800, 0x0000, 0x3800, 0x3c00];
    VALUES[((inner * 3 + column) % VALUES.len() as u32) as usize]
}

fn c600_tile_count() -> u32 {
    ipu_exchange::Topology::c600().tile_count() as u32
}

fn write_package(application: &Application, path: &Path) -> Result<()> {
    let mut output =
        fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    application.write(&mut output)?;
    Ok(())
}

fn diagnose_completion(
    runtime: &Runtime,
    application: &Application,
    timeout: Duration,
) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        let states = supervisor_states(runtime, application)?;
        if states.iter().all(|&(_, state)| state == 0) {
            break;
        }
        if Instant::now() >= deadline {
            bail!("supervisors did not halt: {}", summarize_states(&states));
        }
        std::thread::sleep(Duration::from_micros(100));
    }

    let mut active_workers = Vec::new();
    for tile in &application.tiles {
        let physical = u16::try_from(tile.physical_tile)?;
        for context in 1..=6 {
            let state = runtime.device().tile_context_state(physical, context)?;
            if state != 0 && active_workers.len() < 16 {
                active_workers.push((physical, context, state));
            }
        }
    }
    if !active_workers.is_empty() {
        bail!("workers did not halt: {active_workers:?}");
    }
    Ok(())
}

fn supervisor_states(runtime: &Runtime, application: &Application) -> Result<Vec<(u16, u32)>> {
    application
        .tiles
        .iter()
        .map(|tile| {
            let physical = u16::try_from(tile.physical_tile)?;
            Ok((physical, runtime.device().tile_context_state(physical, 0)?))
        })
        .collect()
}

fn summarize_states(states: &[(u16, u32)]) -> String {
    let mut counts = [0usize; 4];
    let mut unexpected = Vec::new();
    for &(tile, state) in states {
        if let Some(count) = counts.get_mut(state as usize) {
            *count += 1;
        }
        if state != 0 && unexpected.len() < 16 {
            unexpected.push((tile, state));
        }
    }
    format!("counts={counts:?} firstUnexpected={unexpected:?}")
}

fn device_failure_diagnostics(runtime: &Runtime, application: &Application) -> String {
    let states = match supervisor_states(runtime, application) {
        Ok(states) => states,
        Err(error) => return format!("supervisor state read failed: {error}"),
    };
    let mut contexts = Vec::new();
    for &(physical, state) in states.iter().filter(|(_, state)| *state != 0).take(16) {
        let program_counter = runtime.device().read_tile_program_counter(physical, 0);
        let segment = program_counter.as_ref().ok().and_then(|&pc| {
            application
                .tiles
                .iter()
                .find(|tile| tile.physical_tile == u32::from(physical))?
                .segments
                .iter()
                .find(|segment| {
                    (segment.address..segment.address.saturating_add(segment.memory_size))
                        .contains(&pc)
                })
                .map(|segment| {
                    let offset = usize::try_from(pc - segment.address).ok()?;
                    let start = offset.saturating_sub(8) & !3;
                    let end = (offset + 12).min(segment.data.len()) & !3;
                    let words = segment
                        .data
                        .get(start..end)?
                        .chunks_exact(4)
                        .map(|word| {
                            u32::from_le_bytes(word.try_into().expect("four-byte instruction"))
                        })
                        .collect::<Vec<_>>();
                    Some((segment.address, segment.memory_size, start, words))
                })?
        });
        let row_readback =
            segment
                .as_ref()
                .and_then(|&(address, memory_size, byte_offset, ref expected)| {
                    let words = memory_size / 4;
                    let actual = runtime
                        .device()
                        .read_tile_words_from_inactive_context(physical, 1, address, words)
                        .ok()?;
                    let actual_near =
                        actual.get(byte_offset / 4..byte_offset / 4 + expected.len())?;
                    let differences = expected
                        .iter()
                        .zip(actual_near)
                        .enumerate()
                        .filter(|(_, (expected, actual))| expected != actual)
                        .map(|(offset, (&expected, &actual))| {
                            (byte_offset / 4 + offset, expected, actual)
                        })
                        .collect::<Vec<_>>();
                    Some((actual, differences))
                });
        let supervisor_registers = (0..16)
            .map(|register| runtime.device().read_tile_m_register(physical, 0, register))
            .collect::<Result<Vec<_>, _>>();
        let exchange_state = runtime.device().tile_exchange_state(physical);
        let workers = (1..=6)
            .map(|context| runtime.device().tile_context_state(physical, context))
            .collect::<Result<Vec<_>, _>>();
        contexts.push((
            physical,
            state,
            program_counter,
            segment,
            row_readback,
            supervisor_registers,
            exchange_state,
            workers,
        ));
    }
    format!("{} contexts={contexts:?}", summarize_states(&states))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn randomized_binding_values_follow_logical_slice_order() -> Result<()> {
        let mut random = fastrand::Rng::with_seed(0x6c6f_6769_6361_6c5f);
        for _ in 0..256 {
            let count = random.usize(1..=128);
            let mut physical = (0..count as u32).collect::<Vec<_>>();
            random.shuffle(&mut physical);
            let binding = Binding {
                name: "mapped".into(),
                dtype: "f16".into(),
                shape: vec![count as u32],
                slices: physical
                    .into_iter()
                    .enumerate()
                    .map(|(logical, tile)| ipu_package::RegionSlice {
                        tile,
                        tile_address: 0x10_0000,
                        file_offset: (logical * 2) as u64,
                        size: 2,
                    })
                    .collect(),
            };

            let bytes = packed_binding(&binding, |logical, _, _| Ok(logical))?;
            for (logical, raw) in bytes.chunks_exact(2).enumerate() {
                assert_eq!(u16::from_le_bytes(raw.try_into().unwrap()), logical as u16);
            }
        }
        Ok(())
    }
}
