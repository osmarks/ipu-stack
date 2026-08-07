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
    #[arg(long, conflicts_with_all = ["mlp_smoke", "gemm_benchmark"])]
    gemm_smoke: bool,
    /// Execute and numerically verify GEMM-GeLU-GEMM-GeLU.
    #[arg(long, conflicts_with_all = ["gemm_smoke", "gemm_benchmark"])]
    mlp_smoke: bool,
    /// Measure a compute-dense blocked F16 GEMM using device cycle counters.
    #[arg(long, conflicts_with_all = ["gemm_smoke", "mlp_smoke"])]
    gemm_benchmark: bool,
    #[arg(long, default_value_t = 256)]
    benchmark_rows_per_tile: u32,
    #[arg(long, default_value_t = 64)]
    benchmark_inner: u32,
    #[arg(long, default_value_t = 1_500_000_000)]
    clock_hz: u64,
}

fn main() -> Result<()> {
    ipu_runtime::init_tracing();
    let arguments = Arguments::parse();
    let runtime_source = arguments.runtime_source.unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../device/static_runtime.S")
    });
    let bootloader = arguments
        .bootloader
        .unwrap_or_else(|| arguments.sdk.join("bin/ipu/tile_bootloader_cc_ipu21.elf"));
    let active_tiles = u16::try_from(arguments.tiles).context("tile count exceeds u16")?;
    let mut graph = ComputeGraph::default();
    let mut pipeline = PipelineConfig::new(active_tiles);
    if arguments.gemm_smoke {
        let left = graph.host_input("left", [u32::from(active_tiles), 64])?;
        let right = graph.parameter("right", [64, u32::from(active_tiles) * 64])?;
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
    } else if arguments.gemm_benchmark {
        let rows = u32::from(active_tiles)
            .checked_mul(arguments.benchmark_rows_per_tile)
            .context("benchmark row count overflow")?;
        let left = graph.host_input("left", [rows, arguments.benchmark_inner])?;
        let right = graph.parameter("right", [arguments.benchmark_inner, 64])?;
        let output = graph.gemm(left, right)?;
        graph.set_outputs([output])?;
        pipeline.profiling.enabled = true;
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
            build_directory: std::env::temp_dir().join("ipu-stack-package"),
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
    if arguments.gemm_smoke || arguments.mlp_smoke || arguments.gemm_benchmark {
        runtime.load(
            &application,
            &bootloader_bytes,
            application.host_exchange.startup_mark,
        )?;
        if arguments.gemm_smoke {
            run_gemm(
                &runtime,
                &application,
                active_tiles,
                arguments.timeout_seconds,
            )?;
        } else if arguments.mlp_smoke {
            run_mlp_chain(
                &runtime,
                &application,
                active_tiles,
                arguments.timeout_seconds,
            )?;
        } else {
            run_gemm_benchmark(
                &runtime,
                &application,
                active_tiles,
                arguments.benchmark_rows_per_tile,
                arguments.benchmark_inner,
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
    let left_bytes = packed_binding(&left, |tile, linear, elements| {
        let (row, inner) =
            amp_matrix_coordinates(AmpOrder::Left, Precision::F16, 1, elements, linear)?;
        debug_assert_eq!(row, 0);
        Ok(if u32::from(tile) == inner { 0x3c00 } else { 0 })
    })?;
    let right_bytes = packed_binding(&right, |tile, linear, elements| {
        let (inner, column) =
            amp_matrix_coordinates(AmpOrder::Right, Precision::F16, 64, elements / 64, linear)?;
        Ok(gemm_right_value(inner, u32::from(tile) * 64 + column))
    })?;
    if left.slices.len() != usize::from(active_tiles)
        || right.slices.len() != usize::from(active_tiles)
    {
        bail!("GEMM bindings do not cover every active tile");
    }
    let mut session = runtime.host_session(application)?;
    session.start()?;
    let initialized = session.invoke_streaming_deferred("initialize", &right_bytes)?;
    session.collect(&initialized)?;
    let executed = session.invoke_streaming_deferred("run", &left_bytes)?;
    runtime
        .device()
        .write_sync_mark(ipu_driver::pci::HSP_GS2_CONTROL, 1)?;
    diagnose_completion(runtime, application, Duration::from_secs(timeout_seconds))?;
    let output = session.collect(&executed)?;
    verify_gemm_output(application, active_tiles, &output)
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
    let left_bytes = packed_binding(&left, |tile, linear, elements| {
        let (_, inner) =
            amp_matrix_coordinates(AmpOrder::Left, Precision::F16, 1, elements, linear)?;
        Ok(if u32::from(tile) == inner { 0x3c00 } else { 0 })
    })?;
    let right0_bytes = packed_binding(&right0, |tile, linear, elements| {
        let (inner, column) =
            amp_matrix_coordinates(AmpOrder::Right, Precision::F16, 64, elements / 64, linear)?;
        let column = u32::from(tile) * 64 + column;
        Ok(if column < 64 {
            gemm_right_value(inner, column)
        } else {
            0
        })
    })?;
    let right1_bytes = packed_binding(&right1, |tile, linear, elements| {
        let (inner, column) =
            amp_matrix_coordinates(AmpOrder::Right, Precision::F16, 64, elements / 64, linear)?;
        let column = u32::from(tile) * 64 + column;
        Ok(if inner == column { 0x3c00 } else { 0 })
    })?;
    let mut weights = Vec::with_capacity(right0_bytes.len() + right1_bytes.len());
    weights.extend_from_slice(&right0_bytes);
    weights.extend_from_slice(&right1_bytes);

    let mut session = runtime.host_session(application)?;
    session.start()?;
    let initialized = session.invoke_streaming_deferred("initialize", &weights)?;
    session.collect(&initialized)?;
    let executed = session.invoke_streaming_deferred("run", &left_bytes)?;
    runtime
        .device()
        .write_sync_mark(ipu_driver::pci::HSP_GS2_CONTROL, 1)?;
    diagnose_completion(runtime, application, Duration::from_secs(timeout_seconds))?;
    let output = session.collect(&executed)?;
    verify_mlp_output(application, active_tiles, &output)
}

#[allow(clippy::too_many_arguments)]
fn run_gemm_benchmark(
    runtime: &Runtime,
    application: &Application,
    active_tiles: u16,
    rows_per_tile: u32,
    inner: u32,
    clock_hz: u64,
    timeout_seconds: u64,
) -> Result<()> {
    if rows_per_tile == 0 || inner == 0 || !inner.is_multiple_of(64) || clock_hz == 0 {
        bail!(
            "benchmark rows must be nonzero, inner must be a nonzero multiple of 64, and clock must be nonzero"
        );
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
    let mut session = runtime.host_session(application)?;
    session.start()?;
    let initialized = session.invoke_streaming_deferred("initialize", &right_bytes)?;
    session.collect(&initialized)?;
    let executed = session.invoke_streaming_deferred("run", &left_bytes)?;
    runtime
        .device()
        .write_sync_mark(ipu_driver::pci::HSP_GS2_CONTROL, 1)?;
    diagnose_completion(runtime, application, Duration::from_secs(timeout_seconds))?;
    let output = session.collect(&executed)?;
    let maximum_absolute_error = verify_benchmark_output(application, &output, inner)?;
    let starts = binding_u32_values(application, &output, "profile.start-cycle")?;
    let ends = binding_u32_values(application, &output, "profile.end-cycle")?;
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
    let cycles = durations.iter().copied().max().unwrap_or(0);
    if cycles == 0 {
        bail!("benchmark cycle interval is zero");
    }
    let rows = u64::from(active_tiles) * u64::from(rows_per_tile);
    let flops = 2.0 * rows as f64 * f64::from(inner) * 64.0;
    let seconds = f64::from(cycles) / clock_hz as f64;
    let tflops = flops / seconds / 1.0e12;
    let peak_tflops = clock_hz as f64 * f64::from(active_tiles) * 128.0 / 1.0e12;
    let minimum_cycles = durations.iter().copied().min().unwrap_or(cycles);
    println!(
        "benchmark=gemm-f16 rows={rows} inner={inner} columns=64 tiles={active_tiles} cycles={cycles} minimumTileCycles={minimum_cycles} deviceMicroseconds={:.3} tflops={tflops:.3} peakTflops={peak_tflops:.3} efficiencyPercent={:.2} maximumAbsoluteError={maximum_absolute_error:.6}",
        seconds * 1.0e6,
        tflops / peak_tflops * 100.0,
    );
    Ok(())
}

fn verify_benchmark_output(application: &Application, bytes: &[u8], inner: u32) -> Result<f32> {
    let binding = application
        .outputs
        .first()
        .filter(|binding| binding.name == "output.0")
        .context("benchmark package has no graph output")?;
    let size = binding
        .slices
        .iter()
        .map(|slice| slice.file_offset + slice.size)
        .max()
        .context("benchmark output has no slices")?;
    let output = bytes
        .get(..usize::try_from(size)?)
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

fn binding_u32_values(application: &Application, bytes: &[u8], name: &str) -> Result<Vec<u32>> {
    let mut binding_base = 0u64;
    for binding in &application.outputs {
        let binding_size = binding
            .slices
            .iter()
            .map(|slice| slice.file_offset + slice.size)
            .max()
            .unwrap_or(0);
        if binding.name == name {
            let mut values = Vec::with_capacity(binding.slices.len());
            for slice in &binding.slices {
                if slice.size != 4 {
                    bail!("profile binding {name} contains a non-u32 slice");
                }
                let start = usize::try_from(binding_base + slice.file_offset)?;
                let end = start.checked_add(4).context("profile offset overflow")?;
                let raw = bytes
                    .get(start..end)
                    .with_context(|| format!("profile binding {name} exceeds host output"))?;
                values.push(u32::from_le_bytes(raw.try_into().unwrap()));
            }
            return Ok(values);
        }
        binding_base = binding_base
            .checked_add(binding_size)
            .context("host output binding offset overflow")?;
    }
    bail!("package has no {name} output binding")
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
    for row in 0..active_tiles {
        let slice = output
            .slices
            .iter()
            .find(|slice| slice.tile == u32::from(row))
            .with_context(|| format!("MLP output has no slice for tile {row}"))?;
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
            let input = half_to_f32(gemm_right_value(u32::from(row), column));
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
    for slice in &binding.slices {
        if slice.size == 0 {
            continue;
        }
        if slice.size & 1 != 0 {
            bail!("binding {} has a non-F16 slice", binding.name);
        }
        let elements = u32::try_from(slice.size / 2)?;
        let tile = u16::try_from(slice.tile)?;
        for linear in 0..elements {
            let bits = value(tile, linear, elements)?;
            let offset = usize::try_from(slice.file_offset + u64::from(linear) * 2)?;
            bytes[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    Ok(bytes)
}

fn verify_gemm_output(application: &Application, active_tiles: u16, bytes: &[u8]) -> Result<()> {
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
    for row in 0..active_tiles {
        let slice = output
            .slices
            .iter()
            .find(|slice| slice.tile == u32::from(row))
            .with_context(|| format!("GEMM output has no slice for tile {row}"))?;
        let elements = u32::try_from(slice.size / 2)?;
        for linear in 0..elements {
            let (_, column) = amp_matrix_coordinates(
                AmpOrder::Output,
                Precision::F16,
                1,
                u32::from(active_tiles) * 64,
                linear,
            )?;
            let offset = usize::try_from(slice.file_offset + u64::from(linear) * 2)?;
            let actual = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            let expected = gemm_right_value(u32::from(row), column);
            checked += 1;
            if actual != expected && mismatches.len() < 16 {
                mismatches.push((row, column, expected, actual));
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
