use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use ipu_compiler::{
    CompilerOptions, EncoderConfig, EncoderWeights, compile, encoder_graph, encoder_reference,
};
use ipu_driver::{Device, HostBuffer, HostSession, Loader, block_device_interrupt_signals};
use ipu_elf::{LinkOptions, Toolchain, inspect_object, link};
use ipu_exchange::Topology;
use ipu_package::{
    Application, EntryPoint, HostCall, HostExchange, HostPage, HostSlice, SEGMENT_EXECUTE,
    SEGMENT_READ, SEGMENT_WRITE, Segment, TileImage,
};
use object::{Object, ObjectSegment};
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
    ExchangePlan {
        sender: u16,
        receiver: u16,
        count: u32,
        #[arg(short, long)]
        output: Option<PathBuf>,
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
    PackageExtractTile {
        package: PathBuf,
        physical_tile: u32,
        #[arg(short, long)]
        output: PathBuf,
    },
    PackageImportIpuimg {
        image: PathBuf,
        manifest: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    PackageElfDirectory {
        directory: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 1)]
        external_syncs: u32,
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
        #[arg(long)]
        entry: Option<String>,
        #[arg(long, hide = true)]
        break_on_sync_tile: Vec<u16>,
        #[arg(long, default_value = "/dev/ipu0")]
        device: String,
    },
    RunDiagnostic {
        package: PathBuf,
        bootloader: PathBuf,
        configuration: PathBuf,
        binding: String,
        #[arg(long)]
        slice: Vec<usize>,
        #[arg(long, hide = true)]
        break_on_sync_tile: Vec<u16>,
        #[arg(long, default_value = "/dev/ipu0")]
        device: String,
    },
    RunOutput {
        package: PathBuf,
        bootloader: PathBuf,
        configuration: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 4096)]
        page_size: usize,
        #[arg(long, default_value = "0x40", value_parser = parse_u32)]
        page_offset: u32,
        #[arg(long, default_value_t = 4)]
        size: usize,
        #[arg(long)]
        entry: Option<String>,
        #[arg(long, default_value = "/dev/ipu0")]
        device: String,
    },
    HostRun {
        package: PathBuf,
        bootloader: PathBuf,
        configuration: PathBuf,
        #[arg(long, default_value = "/dev/ipu0")]
        device: String,
        #[arg(long, value_parser = parse_named_path)]
        input: Vec<(String, PathBuf)>,
        #[arg(long, value_parser = parse_named_path)]
        output: Vec<(String, PathBuf)>,
        #[arg(required = true)]
        calls: Vec<String>,
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
                    retained_symbols: Vec::new(),
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
        Command::ExchangePlan {
            sender,
            receiver,
            count,
            output,
        } => {
            let plan = Topology::c600().point_to_point(sender, receiver, count)?;
            let encoded = format!(
                "sender {sender}:{}\nreceiver {receiver}:{}\n",
                format_plan_row(&plan.sender),
                format_plan_row(&plan.receiver)
            );
            if let Some(output) = output {
                fs::write(&output, encoded)?;
                println!(
                    "sender={sender} receiver={receiver} count={count} output={}",
                    output.display()
                );
            } else {
                print!("{encoded}");
            }
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
            let exchanges = schedule.lower_exchanges(&Topology::c600())?;
            fs::write(
                &output,
                serde_json::to_vec_pretty(&serde_json::json!({
                    "schedule": &schedule,
                    "exchange": &exchanges,
                }))?,
            )?;
            let launches: u32 = exchanges.iter().map(|phase| phase.cost.launches).sum();
            let exchange_cycles: u64 = exchanges
                .iter()
                .map(|phase| phase.cost.estimated_cycles)
                .sum();
            println!(
                "ops={} tensors={} phases={} allocations={} tiles={} exchangeLaunches={} exchangeCycles={} output={}",
                graph.ops.len(),
                graph.tensors.len(),
                schedule.phases.len(),
                schedule.allocations.len(),
                schedule.tile_count,
                launches,
                exchange_cycles,
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
                "compiler={} tiles={} blobs={} uncompressedBytes={} inputs={} outputs={} weights={} entries={} configWrites={}",
                app.compiler_version,
                app.tiles.len(),
                app.blobs.len(),
                stored,
                app.inputs.len(),
                app.outputs.len(),
                app.weights.len(),
                app.entry_points.len(),
                app.device_config_writes.len()
            );
        }
        Command::PackageExtractTile {
            package,
            physical_tile,
            output,
        } => {
            let app = Application::read(fs::File::open(package)?)?;
            let bytes = app.tile_image(physical_tile)?;
            fs::write(&output, &bytes)?;
            println!(
                "physicalTile={} bytes={} output={}",
                physical_tile,
                bytes.len(),
                output.display()
            );
        }
        Command::PackageImportIpuimg {
            image,
            manifest,
            output,
        } => {
            let mut app = Application::import_ipuimg(&fs::read(&image)?)?;
            app.host_exchange = parse_host_manifest(&fs::read_to_string(&manifest)?)?;
            app.entry_points = app
                .host_exchange
                .calls
                .iter()
                .map(|call| EntryPoint {
                    name: call.name.clone(),
                    command: call.command,
                    external_syncs: call.phases,
                })
                .collect();
            app.write(fs::File::create(&output)?)?;
            println!(
                "tiles={} blobs={} calls={} output={}",
                app.tiles.len(),
                app.blobs.len(),
                app.host_exchange.calls.len(),
                output.display()
            );
        }
        Command::PackageElfDirectory {
            directory,
            output,
            external_syncs,
        } => {
            let topology = Topology::c600();
            let mut app = Application::default();
            let mut blobs = HashMap::<Vec<u8>, usize>::new();
            for physical in 0..topology.tile_count() as u16 {
                let bytes = fs::read(directory.join(format!("t_{physical}.elf")))?;
                let elf = object::File::parse(bytes.as_slice())?;
                let mut segments = Vec::new();
                for segment in elf.segments() {
                    let data = segment.data()?;
                    if data.is_empty()
                        || segment.address() < u64::from(ipu_driver::APPLICATION_LOAD_BASE)
                    {
                        continue;
                    }
                    let data = data.to_vec();
                    let blob = match blobs.get(&data) {
                        Some(&blob) => blob,
                        None => {
                            let blob = app.add_blob(data.clone());
                            blobs.insert(data, blob);
                            blob
                        }
                    };
                    segments.push(Segment {
                        address: segment.address() as u32,
                        memory_size: segment.size() as u32,
                        blob,
                        blob_offset: 0,
                        file_size: app.blobs[blob].bytes.len() as u32,
                        flags: SEGMENT_READ | SEGMENT_WRITE | SEGMENT_EXECUTE,
                    });
                }
                app.tiles.push(TileImage {
                    physical_tile: u32::from(physical),
                    entry_point: elf.entry() as u32,
                    command_address: 0,
                    diagnostic_address: 0,
                    segments,
                });
            }
            app.entry_points.push(EntryPoint {
                name: "elf-directory".into(),
                command: 0,
                external_syncs,
            });
            app.write(fs::File::create(&output)?)?;
            println!(
                "tiles={} uniqueBlobs={} externalSyncs={} output={}",
                app.tiles.len(),
                app.blobs.len(),
                external_syncs,
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
            entry,
            break_on_sync_tile,
            device,
        } => {
            let app = Application::read(fs::File::open(package)?)?;
            let bootloader = fs::read(bootloader)?;
            let configuration = fs::read(configuration)?;
            block_device_interrupt_signals()?;
            let device = Device::open(&device)?;
            device.initialize()?;
            device.replay_configuration(&configuration)?;
            apply_device_config_writes(&device, &app)?;
            Loader::new(&device, &bootloader)?.load(&app, final_mark)?;
            let entry = match entry {
                Some(name) => app
                    .entry_points
                    .iter()
                    .find(|entry| entry.name == name)
                    .ok_or_else(|| anyhow::anyhow!("unknown entry point {name}"))?,
                None => app
                    .entry_points
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("package has no entry point"))?,
            };
            device.write_config(
                ipu_driver::pci::EXCHANGE_WINDOW_BASE,
                ipu_driver::pci::EXCHANGE_WINDOW_HEXOPT,
            )?;
            for &physical_tile in &break_on_sync_tile {
                device.set_break_on_sync(physical_tile, true)?;
            }
            device.set_mark(1)?;
            info!(entry = %entry.name, "application startup released");
            if entry.external_syncs != 0 {
                device.write_config(ipu_driver::pci::HSP_GS2_CONTROL, entry.external_syncs)?;
                info!(
                    syncs = entry.external_syncs,
                    entry = %entry.name,
                    "application device sync credits preloaded"
                );
            }
            println!(
                "tiles={} entry={} releases={} directLoad=PASS",
                app.tiles.len(),
                entry.name,
                entry.external_syncs + 1
            );
        }
        Command::RunDiagnostic {
            package,
            bootloader,
            configuration,
            binding,
            slice,
            break_on_sync_tile,
            device,
        } => {
            let app = Application::read(fs::File::open(package)?)?;
            let entry = app
                .entry_points
                .first()
                .ok_or_else(|| anyhow::anyhow!("package has no entry point"))?;
            let binding = app
                .outputs
                .iter()
                .find(|candidate| candidate.name == binding)
                .ok_or_else(|| anyhow::anyhow!("unknown output binding {binding}"))?;
            let selected = if slice.is_empty() {
                (0..binding.slices.len()).collect::<Vec<_>>()
            } else {
                slice
            };
            let bootloader = fs::read(bootloader)?;
            let configuration = fs::read(configuration)?;
            block_device_interrupt_signals()?;
            let device = Device::open(&device)?;
            device.initialize()?;
            device.replay_configuration(&configuration)?;
            apply_device_config_writes(&device, &app)?;
            Loader::new(&device, &bootloader)?.load(&app, 0)?;
            for physical_tile in break_on_sync_tile {
                device.set_break_on_sync(physical_tile, true)?;
            }
            device.write_config(
                ipu_driver::pci::EXCHANGE_WINDOW_BASE,
                ipu_driver::pci::EXCHANGE_WINDOW_HEXOPT,
            )?;
            device.write_config(ipu_driver::pci::HSP_GS2_CONTROL, entry.external_syncs)?;
            device.set_mark(1)?;
            let completion_tile = app
                .outputs
                .iter()
                .find(|candidate| candidate.name == "completion")
                .and_then(|binding| binding.slices.first())
                .ok_or_else(|| anyhow::anyhow!("package has no completion binding"))?;
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while device.tile_context_state(completion_tile.tile as u16, 0)? != 3 {
                if std::time::Instant::now() >= deadline {
                    let error = anyhow::anyhow!("diagnostic completion timed out");
                    let mut states = [[0usize; 4]; 7];
                    for tile in &app.tiles {
                        for (context, context_states) in states.iter_mut().enumerate() {
                            let state = device
                                .tile_context_state(tile.physical_tile as u16, context as u32)?;
                            context_states[state as usize] += 1;
                        }
                    }
                    let mut context_diagnostics = Vec::new();
                    for physical_tile in [0, 2, 32, 53, 63] {
                        if matches!(device.tile_context_state(physical_tile, 0)?, 2 | 3) {
                            context_diagnostics.push((
                                physical_tile,
                                0,
                                0,
                                device.read_tile_program_counter(physical_tile, 0)?,
                                0,
                                0,
                            ));
                        }
                        if matches!(device.tile_context_state(physical_tile, 1)?, 2 | 3) {
                            let status = device.read_tile_worker_status(physical_tile, 1)?;
                            context_diagnostics.push((
                                physical_tile,
                                1,
                                u32::from(ipu_driver::TileException::from_status(status) as u8),
                                device.read_tile_program_counter(physical_tile, 1)?,
                                device.read_tile_m_register(physical_tile, 1, 1)?,
                                device.read_tile_m_register(physical_tile, 1, 15)?,
                            ));
                        }
                    }
                    tracing::error!(
                        ?states,
                        ?context_diagnostics,
                        %error,
                        "diagnostic application did not complete"
                    );
                    return Err(error);
                }
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            if device.read_tile_word(completion_tile.tile as u16, completion_tile.tile_address)?
                != 1
            {
                bail!("coordinator trapped before storing completion");
            }
            for index in selected {
                let region = binding
                    .slices
                    .get(index)
                    .ok_or_else(|| anyhow::anyhow!("binding slice {index} is out of range"))?;
                if region.size != 4 {
                    bail!("diagnostic binding slices must contain one u32");
                }
                let value = device.read_tile_word(region.tile as u16, region.tile_address)?;
                println!(
                    "binding={} slice={} physicalTile={} value={}",
                    binding.name, index, region.tile, value
                );
            }
        }
        Command::RunOutput {
            package,
            bootloader,
            configuration,
            output,
            page_size,
            page_offset,
            size,
            entry,
            device,
        } => {
            let app = Application::read(fs::File::open(package)?)?;
            let entry = match entry {
                Some(name) => app
                    .entry_points
                    .iter()
                    .find(|candidate| candidate.name == name)
                    .ok_or_else(|| anyhow::anyhow!("unknown entry point {name}"))?,
                None => app
                    .entry_points
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("package has no entry point"))?,
            };
            let page_offset = page_offset as usize;
            if size == 0 || page_offset > page_size || size > page_size - page_offset {
                bail!("output slice is outside the attached host page");
            }
            let bootloader = fs::read(bootloader)?;
            let configuration = fs::read(configuration)?;
            block_device_interrupt_signals()?;
            let device = Device::open(&device)?;
            device.initialize()?;
            device.replay_configuration(&configuration)?;
            apply_device_config_writes(&device, &app)?;
            Loader::new(&device, &bootloader)?.load(&app, 0)?;
            let page = HostBuffer::new(page_size)?;
            let command_page = HostBuffer::new(4096)?;
            device.set_mark(1)?;
            device.wait_mark(
                ipu_driver::pci::HSP_GS2_CONTROL,
                0,
                std::time::Duration::from_secs(10),
            )?;
            device.write_config(
                ipu_driver::pci::EXCHANGE_WINDOW_BASE,
                ipu_driver::pci::EXCHANGE_WINDOW_HEXOPT,
            )?;
            command_page.attach(&device, 1)?;
            device.write_config(
                ipu_driver::pci::EXCHANGE_WINDOW_BASE,
                ipu_driver::pci::EXCHANGE_WINDOW_HEXOPT,
            )?;
            page.attach(&device, 0)?;
            device.write_config(ipu_driver::pci::HSP_GS2_CONTROL, 1)?;
            device.wait_mark(
                ipu_driver::pci::HSP_GS2_CONTROL,
                0,
                std::time::Duration::from_secs(10),
            )?;
            const HOST_OUTPUT_PHASES: u32 = 6;
            let bulk_syncs = entry
                .external_syncs
                .checked_sub(1 + HOST_OUTPUT_PHASES)
                .ok_or_else(|| anyhow::anyhow!("entry point has too few output phases"))?;
            device.write_config(ipu_driver::pci::HSP_GS2_CONTROL, bulk_syncs)?;
            device.wait_mark(
                ipu_driver::pci::HSP_GS2_CONTROL,
                0,
                std::time::Duration::from_secs(10),
            )?;
            for phase in 0..HOST_OUTPUT_PHASES {
                device.write_config(ipu_driver::pci::HSP_GS2_CONTROL, 1)?;
                device
                    .wait_mark(
                        ipu_driver::pci::HSP_GS2_CONTROL,
                        0,
                        std::time::Duration::from_secs(10),
                    )
                    .with_context(|| format!("host output phase {phase}"))?;
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            fs::write(&output, &page.bytes()[page_offset..page_offset + size])?;
            device.detach_buffer(0);
            device.detach_buffer(1);
            println!(
                "entry={} outputBytes={} output={} directOutput=EXPERIMENTAL",
                entry.name,
                size,
                output.display()
            );
        }
        Command::HostRun {
            package,
            bootloader,
            configuration,
            device,
            input,
            output,
            calls,
        } => {
            let app = Application::read(fs::File::open(package)?)?;
            let protocol = app.host_exchange.clone();
            let input: HashMap<_, _> = input.into_iter().collect();
            let output: HashMap<_, _> = output.into_iter().collect();
            let bootloader = fs::read(bootloader)?;
            let configuration = fs::read(configuration)?;
            block_device_interrupt_signals()?;
            let device = Device::open(&device)?;
            device.initialize()?;
            device.replay_configuration(&configuration)?;
            apply_device_config_writes(&device, &app)?;
            Loader::new(&device, &bootloader)?.load(&app, protocol.startup_mark)?;
            let mut session = HostSession::new(&device, protocol)?;
            session.start()?;
            for call in calls {
                let bytes = input
                    .get(&call)
                    .map(fs::read)
                    .transpose()?
                    .unwrap_or_default();
                let result = session.invoke(&call, &bytes)?;
                if let Some(path) = output.get(&call) {
                    fs::write(path, &result)?;
                } else if !result.is_empty() {
                    bail!(
                        "call {call} produced {} bytes without --output",
                        result.len()
                    );
                }
            }
            println!("hostCalls={} directHostRun=PASS", app.entry_points.len());
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
            Self::ExchangePlan { .. } => "exchange-plan",
            Self::EncoderPlan { .. } => "encoder-plan",
            Self::EncoderReference => "encoder-reference",
            Self::PackageInspect { .. } => "package-inspect",
            Self::PackageExtractTile { .. } => "package-extract-tile",
            Self::PackageImportIpuimg { .. } => "package-import-ipuimg",
            Self::PackageElfDirectory { .. } => "package-elf-directory",
            Self::DeviceProbe { .. } => "device-probe",
            Self::Load { .. } => "load",
            Self::RunDiagnostic { .. } => "run-diagnostic",
            Self::RunOutput { .. } => "run-output",
            Self::HostRun { .. } => "host-run",
        }
    }
}

fn format_plan_row(row: &[u32]) -> String {
    row.iter().map(|word| format!(" 0x{word:08x}")).collect()
}

fn apply_device_config_writes(device: &Device, app: &Application) -> Result<()> {
    for write in &app.device_config_writes {
        device.write_config(write.offset, write.value)?;
        tracing::debug!(
            offset = format_args!("0x{:x}", write.offset),
            value = format_args!("0x{:08x}", write.value),
            "applied application device configuration"
        );
    }
    Ok(())
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

fn parse_named_path(value: &str) -> Result<(String, PathBuf), String> {
    let Some((name, path)) = value.split_once('=') else {
        return Err("expected NAME=PATH".into());
    };
    if name.is_empty() || path.is_empty() {
        return Err("expected non-empty NAME=PATH".into());
    }
    Ok((name.into(), path.into()))
}

fn parse_host_manifest(text: &str) -> Result<HostExchange> {
    let mut host = HostExchange::default();
    let mut header = false;
    for (line_index, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        let invalid = || anyhow::anyhow!("host manifest line {}: {line}", line_index + 1);
        if !header {
            if fields.as_slice() != ["IPU-HOST-EXCHANGE", "1"] {
                return Err(invalid());
            }
            header = true;
            continue;
        }
        match fields.as_slice() {
            ["startup-mark", mark] => host.startup_mark = mark.parse()?,
            ["page", index, size] => host.pages.push(HostPage {
                index: index.parse()?,
                size: size.parse()?,
            }),
            ["attach", index] => host.attach_order.push(index.parse()?),
            ["command", page, offset] => {
                host.command_page = page.parse()?;
                host.command_offset = offset.parse()?;
            }
            ["call", name, command, phases] => host.calls.push(HostCall {
                name: (*name).into(),
                command: command.parse()?,
                phases: phases.parse()?,
                inputs: Vec::new(),
                outputs: Vec::new(),
            }),
            [
                direction @ ("input" | "output"),
                name,
                page,
                page_offset,
                file_offset,
                size,
            ] => {
                let call = host
                    .calls
                    .iter_mut()
                    .find(|call| call.name == *name)
                    .ok_or_else(invalid)?;
                let slice = HostSlice {
                    page: page.parse()?,
                    page_offset: page_offset.parse()?,
                    file_offset: file_offset.parse()?,
                    size: size.parse()?,
                };
                if *direction == "input" {
                    call.inputs.push(slice);
                } else {
                    call.outputs.push(slice);
                }
            }
            _ => return Err(invalid()),
        }
    }
    if !header {
        bail!("empty host manifest");
    }
    Ok(host)
}
