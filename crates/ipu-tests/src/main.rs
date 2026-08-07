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
    #[arg(long)]
    gemm_smoke: bool,
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
    if arguments.gemm_smoke {
        runtime.load(
            &application,
            &bootloader_bytes,
            application.host_exchange.startup_mark,
        )?;
        run_gemm(
            &runtime,
            &application,
            active_tiles,
            arguments.timeout_seconds,
        )?;
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
        if slice.size == 0 || slice.size & 1 != 0 {
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
