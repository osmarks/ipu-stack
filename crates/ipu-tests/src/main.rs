use anyhow::{Context, Result, bail};
use clap::Parser;
use ipu_codegen::{ComputeGraph, PackageConfig, build_package};
use ipu_elf::Toolchain;
use ipu_package::Application;
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
    let application = build_package(
        &ComputeGraph::default(),
        &PackageConfig {
            toolchain: Toolchain::from_sdk(&arguments.sdk),
            runtime_source,
            build_directory: std::env::temp_dir().join("ipu-stack-package"),
            tile_count: arguments.tiles,
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
    runtime.load(&application, &bootloader_bytes, 0)?;
    diagnose_completion(
        &runtime,
        &application,
        Duration::from_secs(arguments.timeout_seconds),
    )?;
    println!(
        "package={} tiles={} hardwareTest=PASS",
        arguments.package.display(),
        application.tiles.len()
    );
    Ok(())
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
