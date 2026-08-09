use anyhow::{Context, Result, bail};
use clap::Parser;
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
    #[arg(long, default_value_t = c600_tile_count())]
    tiles: u32,
    #[arg(long)]
    runtime_source: Option<PathBuf>,
    #[arg(long, default_value_t = 10)]
    timeout_seconds: u64,
    /// Execute one blocked F16 GEMM as a kernel/exchange smoke test.
    #[arg(long, conflicts_with_all = ["batched_gemm_smoke", "mlp_smoke", "gemm_benchmark", "siglip_mlp_benchmark"])]
    gemm_smoke: bool,
    /// Execute an F16 GEMM with a batched activation and broadcast weights.
    #[arg(long, conflicts_with_all = ["gemm_smoke", "mlp_smoke", "gemm_benchmark", "siglip_mlp_benchmark"])]
    batched_gemm_smoke: bool,
    /// Execute and numerically verify GEMM-GeLU-GEMM-GeLU.
    #[arg(long, conflicts_with_all = ["gemm_smoke", "batched_gemm_smoke", "gemm_benchmark", "siglip_mlp_benchmark"])]
    mlp_smoke: bool,
    /// Measure a compute-dense blocked F16 GEMM using device cycle counters.
    #[arg(long, conflicts_with_all = ["gemm_smoke", "batched_gemm_smoke", "mlp_smoke", "siglip_mlp_benchmark"])]
    gemm_benchmark: bool,
    /// Benchmark batched Dense-GeLU-Dense with broadcast weights and no biases.
    #[arg(long, conflicts_with_all = ["gemm_smoke", "batched_gemm_smoke", "mlp_smoke", "gemm_benchmark"])]
    siglip_mlp_benchmark: bool,
    #[arg(long, default_value_t = 4)]
    mlp_batch: u32,
    #[arg(long, default_value_t = 256)]
    mlp_tokens: u32,
    #[arg(long, default_value_t = 1024)]
    mlp_dim: u32,
    /// Defaults to four times --mlp-dim.
    #[arg(long)]
    mlp_hidden_dim: Option<u32>,
    #[arg(long, default_value_t = 256)]
    benchmark_rows_per_tile: u32,
    /// Global row count. Overrides --benchmark-rows-per-tile and allows the
    /// planner to distribute a smaller matrix over a multi-axis tile grid.
    #[arg(long)]
    benchmark_rows: Option<u32>,
    #[arg(long, default_value_t = 64)]
    benchmark_inner: u32,
    #[arg(long, default_value_t = 64)]
    benchmark_columns: u32,
    #[arg(long, default_value_t = 1_500_000_000)]
    clock_hz: u64,
}

fn main() -> Result<()> {
    ipu_runtime::init_tracing();
    let arguments = Arguments::parse();
    let active_tiles = u16::try_from(arguments.tiles).context("tile count exceeds u16")?;
    let benchmark_rows = arguments.benchmark_rows.map_or_else(
        || {
            u32::from(active_tiles)
                .checked_mul(arguments.benchmark_rows_per_tile)
                .context("benchmark row count overflow")
        },
        Ok,
    )?;
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
    if arguments.gemm_smoke || arguments.batched_gemm_smoke {
        let batch = if arguments.batched_gemm_smoke { 3 } else { 1 };
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
    } else if arguments.mlp_smoke {
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
    } else if arguments.siglip_mlp_benchmark {
        validate_mlp_benchmark_shape(
            arguments.mlp_batch,
            arguments.mlp_tokens,
            arguments.mlp_dim,
            mlp_hidden_dim,
        )?;
        let left = graph.host_input(
            "left",
            [arguments.mlp_batch, arguments.mlp_tokens, arguments.mlp_dim],
        )?;
        let right0 = graph.parameter("right.0", [1, arguments.mlp_dim, mlp_hidden_dim])?;
        let right1 = graph.parameter("right.1", [1, mlp_hidden_dim, arguments.mlp_dim])?;
        let hidden = graph.gemm(left, right0)?;
        let hidden = graph.gelu(hidden)?;
        let output = graph.gemm(hidden, right1)?;
        graph.set_outputs([output])?;
        pipeline.profiling.enabled = true;
        pipeline = pipeline
            .with_automatic_input(left, Precision::F16)
            .with_automatic_input(right0, Precision::F16)
            .with_automatic_input(right1, Precision::F16);
    } else if arguments.gemm_benchmark {
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
        pipeline.profiling.enabled = true;
        pipeline = pipeline
            .with_automatic_input(left, Precision::F16)
            .with_automatic_input(right, Precision::F16);
    }
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
    let application = Application::read(
        fs::File::open(&arguments.package)
            .with_context(|| format!("open {}", arguments.package.display()))?,
    )?;

    let configuration = fs::read(&arguments.configuration)
        .with_context(|| format!("read {}", arguments.configuration.display()))?;
    let bootloader_bytes =
        fs::read(&bootloader).with_context(|| format!("read {}", bootloader.display()))?;
    let runtime = Runtime::open(&arguments.device, &configuration)?;
    if arguments.gemm_smoke
        || arguments.batched_gemm_smoke
        || arguments.mlp_smoke
        || arguments.siglip_mlp_benchmark
        || arguments.gemm_benchmark
    {
        runtime.load(
            &application,
            &bootloader_bytes,
            application.host_exchange.startup_mark,
        )?;
        if arguments.gemm_smoke || arguments.batched_gemm_smoke {
            run_gemm(
                &runtime,
                &application,
                active_tiles,
                if arguments.batched_gemm_smoke { 3 } else { 1 },
                arguments.timeout_seconds,
            )?;
        } else if arguments.mlp_smoke {
            run_mlp_chain(
                &runtime,
                &application,
                active_tiles,
                arguments.timeout_seconds,
            )?;
        } else if arguments.siglip_mlp_benchmark {
            run_siglip_mlp_benchmark(
                &runtime,
                &application,
                active_tiles,
                arguments.mlp_batch,
                arguments.mlp_tokens,
                arguments.mlp_dim,
                mlp_hidden_dim,
                arguments.clock_hz,
                arguments.timeout_seconds,
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
        Ok(if u32::from(logical_tile) % 64 == inner {
            0x3c00
        } else {
            0
        })
    })?;
    let right0_bytes = packed_binding(&right0, |logical_tile, linear, elements| {
        let (inner, column) =
            amp_matrix_coordinates(AmpOrder::Right, Precision::F16, 64, elements / 64, linear)?;
        let column = u32::from(logical_tile) * 64 + column;
        Ok(if column < 64 {
            gemm_right_value(inner, column)
        } else {
            0
        })
    })?;
    let right1_bytes = packed_binding(&right1, |logical_tile, linear, elements| {
        let (inner, column) =
            amp_matrix_coordinates(AmpOrder::Right, Precision::F16, 64, elements / 64, linear)?;
        let column = u32::from(logical_tile) * 64 + column;
        Ok(if inner == column { 0x3c00 } else { 0 })
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
    session.start().map_err(|error| {
        eprintln!(
            "startFailureDiagnostics={}",
            device_failure_diagnostics(runtime, application)
        );
        error
    })?;
    let initialized = session.invoke_streaming_deferred("initialize", weights)?;
    session.collect(&initialized)?;
    let executed = session
        .invoke_streaming_deferred("run", input)
        .map_err(|error| {
            eprintln!(
                "runFailureDiagnostics={}",
                device_failure_diagnostics(runtime, application)
            );
            error
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
    let (cycles, minimum_cycles) = benchmark_cycles(application, &output, execution_tiles)?;
    let active_tiles = binding_tile_count(application, "output.0")?;
    let rows = u64::from(rows);
    let flops = 2.0 * rows as f64 * f64::from(inner) * f64::from(columns);
    let seconds = f64::from(cycles) / clock_hz as f64;
    let tflops = flops / seconds / 1.0e12;
    let peak_tflops = clock_hz as f64 * f64::from(active_tiles) * 128.0 / 1.0e12;
    println!(
        "benchmark=gemm-f16 rows={rows} inner={inner} columns={columns} activeTiles={active_tiles} executionTiles={execution_tiles} inputBytes={} weightBytes={} cycles={cycles} minimumTileCycles={minimum_cycles} deviceMicroseconds={:.3} tflops={tflops:.3} peakTflops={peak_tflops:.3} efficiencyPercent={:.2} maximumAbsoluteError={maximum_absolute_error:.6}",
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
    clock_hz: u64,
    timeout_seconds: u64,
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
    let right0 = application
        .weights
        .iter()
        .find(|binding| binding.name == "right.0")
        .context("MLP benchmark package has no first weight binding")?;
    let right1 = application
        .weights
        .iter()
        .find(|binding| binding.name == "right.1")
        .context("MLP benchmark package has no second weight binding")?;
    let left_bytes = filled_f16_binding(left, 0x3c00)?;
    let right0_bytes = filled_f16_binding(right0, 0x2000)?;
    let right1_bytes = filled_f16_binding(right1, 0x2000)?;
    let mut weights = Vec::with_capacity(right0_bytes.len() + right1_bytes.len());
    weights.extend_from_slice(&right0_bytes);
    weights.extend_from_slice(&right1_bytes);

    let output =
        run_initialized_program(runtime, application, &weights, &left_bytes, timeout_seconds)?;

    let first_dense = dimension as f32 / 128.0;
    let expected = gelu_reference(first_dense) * hidden_dimension as f32 / 128.0;
    let maximum_absolute_error = verify_constant_output(application, &output, expected)?;
    let (cycles, minimum_cycles) = benchmark_cycles(application, &output, execution_tiles)?;
    let active_tiles = binding_tile_count(application, "output.0")?;
    let rows = u64::from(batch) * u64::from(tokens);
    let flops = 4.0 * rows as f64 * f64::from(dimension) * f64::from(hidden_dimension);
    let seconds = f64::from(cycles) / clock_hz as f64;
    let tflops = flops / seconds / 1.0e12;
    let peak_tflops = clock_hz as f64 * f64::from(active_tiles) * 128.0 / 1.0e12;
    println!(
        "benchmark=siglip-mlp-f16 batch={batch} tokens={tokens} rows={rows} dimension={dimension} hiddenDimension={hidden_dimension} biases=false activeTiles={active_tiles} executionTiles={execution_tiles} inputBytes={} weightBytes={} cycles={cycles} minimumTileCycles={minimum_cycles} deviceMicroseconds={:.3} effectiveGemmTflops={tflops:.3} peakTflops={peak_tflops:.3} efficiencyPercent={:.2} maximumAbsoluteError={maximum_absolute_error:.6}",
        left_bytes.len(),
        weights.len(),
        seconds * 1.0e6,
        tflops / peak_tflops * 100.0,
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
    for raw in output.chunks_exact(2) {
        let actual = half_to_f32(u16::from_le_bytes(raw.try_into().unwrap()));
        maximum = maximum.max((actual - expected).abs());
    }
    if maximum > expected.abs() * 0.02 + 0.05 {
        bail!(
            "MLP benchmark numerical output differs from {expected}: maximum absolute error {maximum}"
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
            let input = half_to_f32(gemm_right_value(u32::from(row) % 64, column));
            let expected = gelu_reference(gelu_reference(input));
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
        let workers = (1..=6)
            .map(|context| runtime.device().tile_context_state(physical, context))
            .collect::<Result<Vec<_>, _>>();
        contexts.push((physical, state, workers));
    }
    format!("{} contexts={contexts:?}", summarize_states(&states))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn randomized_benchmark_shapes_enforce_blocks_and_tile_capacity() {
        let mut random = fastrand::Rng::with_seed(0x6265_6e63_686d_6172);
        for _ in 0..512 {
            let rows = random.u32(1..=512);
            let inner = random.u32(1..=16) * 64;
            let columns = random.u32(1..=8) * 64;
            assert!(validate_benchmark_shape(rows, inner, columns).is_ok());
            assert!(validate_benchmark_shape(rows, inner + 1, columns).is_err());
            assert!(validate_benchmark_shape(rows, inner, columns + 1).is_err());
        }
    }

    #[test]
    fn randomized_mlp_benchmark_shapes_require_blocked_feature_dimensions() {
        let mut random = fastrand::Rng::with_seed(0x6d6c_705f_7368_6170);
        for _ in 0..512 {
            let batch = random.u32(1..=8);
            let tokens = random.u32(1..=512);
            let dimension = random.u32(1..=32) * 64;
            let hidden = random.u32(1..=128) * 64;
            assert!(validate_mlp_benchmark_shape(batch, tokens, dimension, hidden).is_ok());
            assert!(validate_mlp_benchmark_shape(0, tokens, dimension, hidden).is_err());
            assert!(validate_mlp_benchmark_shape(batch, tokens, dimension + 1, hidden).is_err());
            assert!(validate_mlp_benchmark_shape(batch, tokens, dimension, hidden + 1).is_err());
        }
    }

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
