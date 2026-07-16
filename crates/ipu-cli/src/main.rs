use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ipu_compiler::{
    CompilerOptions, EncoderConfig, EncoderWeights, compile, encoder_graph, encoder_reference,
};
use ipu_driver::{Device, Loader, block_device_interrupt_signals};
use ipu_elf::{LinkOptions, Toolchain, inspect_object, link};
use ipu_package::{Application, EntryPoint, SEGMENT_EXECUTE, SEGMENT_READ, Segment, TileImage};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use tracing::{info, info_span};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(version, about = "Independent Graphcore IPU compiler and runtime")]
struct Arguments {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    KernelCompile {
        source: PathBuf,
        output_directory: PathBuf,
        #[arg(long)]
        name: Option<String>,
        #[arg(long, env = "POPLAR_SDK_ENABLED")]
        sdk: PathBuf,
        #[arg(last = true)]
        flags: Vec<String>,
    },
    ObjectInspect {
        object: PathBuf,
    },
    ObjectLink {
        #[arg(required = true)]
        objects: Vec<PathBuf>,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value = "0x4c000", value_parser = parse_u32)]
        base: u32,
        #[arg(long)]
        entry: String,
        #[arg(long, value_parser = parse_symbol)]
        symbol: Vec<(String, u32)>,
    },
    EncoderPlan {
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 64)]
        tiles: u16,
    },
    EncoderReference,
    PackageInspect {
        package: PathBuf,
    },
    PackageRuntimeFixture {
        object: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        tiles: u32,
    },
    DeviceProbe {
        #[arg(long, default_value = "/dev/ipu0")]
        device: String,
    },
    Load {
        package: PathBuf,
        bootloader: PathBuf,
        configuration: PathBuf,
        #[arg(long, default_value_t = 0)]
        final_mark: u32,
        #[arg(long, default_value = "/dev/ipu0")]
        device: String,
    },
}

fn main() -> Result<()> {
    let filter = || EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    if std::env::var("IPU_LOG_FORMAT").as_deref() == Ok("json") {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter())
            .with_target(false)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(filter())
            .with_target(false)
            .init();
    }
    let command = Arguments::parse().command;
    let command_name = command.name();
    let span = info_span!("command", command = command_name);
    let _entered = span.enter();
    info!("starting command");
    match command {
        Command::KernelCompile {
            source,
            output_directory,
            name,
            sdk,
            flags,
        } => {
            let name = name.unwrap_or_else(|| {
                source
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            });
            let artifact =
                Toolchain::from_sdk(sdk).compile(source, output_directory, &name, &flags)?;
            let summary = artifact.inspect()?;
            println!("object={}", artifact.object.display());
            println!("metadata={}", artifact.metadata.display());
            println!("sha256={}", hex::encode(artifact.digest()?));
            println!("defined={}", summary.defined_symbols.join(","));
            println!("relocations={:?}", summary.relocation_types);
        }
        Command::ObjectInspect { object } => {
            let summary = inspect_object(
                &fs::read(&object).with_context(|| format!("read {}", object.display()))?,
            )?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::ObjectLink {
            objects,
            output,
            base,
            entry,
            symbol,
        } => {
            let objects = objects
                .iter()
                .map(|path| fs::read(path).with_context(|| format!("read {}", path.display())))
                .collect::<Result<Vec<_>>>()?;
            let image = link(
                &objects,
                &LinkOptions {
                    image_base: base,
                    entry_symbol: entry,
                    externals: symbol.into_iter().collect::<HashMap<_, _>>(),
                },
            )?;
            fs::write(&output, &image.bytes)?;
            println!(
                "base=0x{:x} entry=0x{:x} bytes={} symbols={}",
                image.base,
                image.entry,
                image.bytes.len(),
                image.symbols.len()
            );
        }
        Command::EncoderPlan { output, tiles } => {
            let graph = encoder_graph(EncoderConfig::default())?;
            let schedule = compile(
                &graph,
                &CompilerOptions {
                    tile_count: tiles,
                    ..CompilerOptions::default()
                },
            )?;
            fs::write(&output, serde_json::to_vec_pretty(&schedule)?)?;
            println!(
                "ops={} tensors={} phases={} allocations={} tiles={} output={}",
                graph.ops.len(),
                graph.tensors.len(),
                schedule.phases.len(),
                schedule.allocations.len(),
                schedule.tile_count,
                output.display()
            );
        }
        Command::EncoderReference => {
            let config = EncoderConfig::default();
            let weights = EncoderWeights::deterministic(config);
            let input: Vec<_> = (0..config.sequence * config.hidden)
                .map(|index| ((index * 13 % 257) as f32 - 128.0) / 128.0)
                .collect();
            let output = encoder_reference(config, &input, &weights)?;
            println!(
                "elements={} checksum={:.9} min={:.9} max={:.9}",
                output.len(),
                output.iter().sum::<f32>(),
                output.iter().copied().fold(f32::INFINITY, f32::min),
                output.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            );
        }
        Command::PackageInspect { package } => {
            let app = Application::read(fs::File::open(&package)?)?;
            let stored: usize = app.blobs.iter().map(|blob| blob.bytes.len()).sum();
            println!(
                "compiler={} tiles={} blobs={} uncompressedBytes={} inputs={} outputs={} weights={} entries={}",
                app.compiler_version,
                app.tiles.len(),
                app.blobs.len(),
                stored,
                app.inputs.len(),
                app.outputs.len(),
                app.weights.len(),
                app.entry_points.len()
            );
        }
        Command::PackageRuntimeFixture {
            object,
            output,
            tiles,
        } => {
            let object = fs::read(object)?;
            let image = link(
                &[object],
                &LinkOptions {
                    image_base: ipu_driver::APPLICATION_LOAD_BASE,
                    entry_symbol: "ipu_stack_start".into(),
                    externals: HashMap::new(),
                },
            )?;
            let mut app = Application::default();
            let blob = app.add_blob(image.bytes);
            for physical_tile in 0..tiles {
                app.tiles.push(TileImage {
                    physical_tile,
                    entry_point: image.entry,
                    command_address: 0,
                    diagnostic_address: 0x52020,
                    segments: vec![Segment {
                        address: image.base,
                        memory_size: app.blobs[blob].bytes.len() as u32,
                        blob,
                        blob_offset: 0,
                        file_size: app.blobs[blob].bytes.len() as u32,
                        flags: SEGMENT_READ | SEGMENT_EXECUTE,
                    }],
                });
            }
            app.entry_points.push(EntryPoint {
                name: "worker-gang-test".into(),
                command: 0,
            });
            app.write(fs::File::create(&output)?)?;
            println!(
                "tiles={} linkedBytes={} blobs={} output={}",
                tiles,
                app.blobs[blob].bytes.len(),
                app.blobs.len(),
                output.display()
            );
        }
        Command::DeviceProbe { device } => {
            let device = Device::open(&device)?;
            println!(
                "ccsr=0x{:08x} autoloader=0x{:08x} gs1={} gs2={} attach=PASS",
                device.read_config(ipu_driver::pci::CCSR)?,
                device.read_config(ipu_driver::pci::AUTOLD_CSR)?,
                device.read_config(ipu_driver::pci::HSP_GS1_CONTROL)? & ipu_driver::HSP_MARK_MASK,
                device.read_config(ipu_driver::pci::HSP_GS2_CONTROL)? & ipu_driver::HSP_MARK_MASK,
            );
        }
        Command::Load {
            package,
            bootloader,
            configuration,
            final_mark,
            device,
        } => {
            let app = Application::read(fs::File::open(package)?)?;
            let bootloader = fs::read(bootloader)?;
            let configuration = fs::read(configuration)?;
            block_device_interrupt_signals()?;
            let device = Device::open(&device)?;
            device.initialize()?;
            device.replay_configuration(&configuration)?;
            Loader::new(&device, &bootloader)?.load(&app, final_mark)?;
            device.write_config(
                ipu_driver::pci::EXCHANGE_WINDOW_BASE,
                ipu_driver::pci::EXCHANGE_WINDOW_HEXOPT,
            )?;
            device.set_mark(1)?;
            std::thread::sleep(std::time::Duration::from_millis(100));
            println!("tiles={} directLoad=PASS", app.tiles.len());
        }
    }
    info!("command completed");
    Ok(())
}

impl Command {
    fn name(&self) -> &'static str {
        match self {
            Self::KernelCompile { .. } => "kernel-compile",
            Self::ObjectInspect { .. } => "object-inspect",
            Self::ObjectLink { .. } => "object-link",
            Self::EncoderPlan { .. } => "encoder-plan",
            Self::EncoderReference => "encoder-reference",
            Self::PackageInspect { .. } => "package-inspect",
            Self::PackageRuntimeFixture { .. } => "package-runtime-fixture",
            Self::DeviceProbe { .. } => "device-probe",
            Self::Load { .. } => "load",
        }
    }
}

fn parse_u32(value: &str) -> Result<u32, String> {
    let parsed = value.strip_prefix("0x").map_or_else(
        || value.parse::<u32>(),
        |digits| u32::from_str_radix(digits, 16),
    );
    parsed.map_err(|error| error.to_string())
}

fn parse_symbol(value: &str) -> Result<(String, u32), String> {
    let Some((name, value)) = value.split_once('=') else {
        return Err("expected NAME=VALUE".into());
    };
    if name.is_empty() {
        return Err("symbol name is empty".into());
    }
    Ok((name.into(), parse_u32(value)?))
}
