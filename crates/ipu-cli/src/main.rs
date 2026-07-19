use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use ipu_compiler::{
    CompilerOptions, EncoderConfig, EncoderWeights, compile, encoder_graph, encoder_reference,
};
use ipu_driver::{Device, HostBuffer, HostSession, Loader, block_device_interrupt_signals};
use ipu_elf::{LinkOptions, Toolchain, inspect_object, link};
use ipu_exchange::Topology;
use ipu_package::{
    Application, EntryPoint, HostCall, HostExchange, HostPage, HostSlice, ProfileReport,
    ProfileStepKind, SEGMENT_EXECUTE, SEGMENT_READ, SEGMENT_WRITE, Segment, TileImage,
};
use object::{Object, ObjectSegment};
use std::collections::{BTreeMap, HashMap};
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
    ProfileInspect {
        profile: PathBuf,
    },
    ProfileRender {
        profile: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
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
        Command::ProfileInspect { profile } => {
            let report = ProfileReport::read(fs::File::open(profile)?)?;
            let mut phases = BTreeMap::<(u32, u32, String), u32>::new();
            for tile in &report.tiles {
                for sample in &tile.samples {
                    let cycles = sample.end_cycle.wrapping_sub(sample.start_cycle);
                    phases
                        .entry((
                            sample.step.phase,
                            sample.step.epoch,
                            sample.step.operation.clone(),
                        ))
                        .and_modify(|maximum| *maximum = (*maximum).max(cycles))
                        .or_insert(cycles);
                }
            }
            let mut operations = BTreeMap::<String, (usize, u64, u32)>::new();
            for ((_, _, operation), cycles) in phases {
                let entry = operations.entry(operation).or_default();
                entry.0 += 1;
                entry.1 += u64::from(cycles);
                entry.2 = entry.2.max(cycles);
            }
            for (operation, (phases, cycles, maximum)) in operations {
                println!(
                    "operation={operation} phases={phases} cycles={cycles} maxPhaseCycles={maximum}"
                );
            }
        }
        Command::ProfileRender { profile, output } => {
            let report = ProfileReport::read(fs::File::open(&profile)?)?;
            fs::write(&output, render_profile_html(&report)?)?;
            println!(
                "profile={} tiles={} output={}",
                profile.display(),
                report.tiles.len(),
                output.display()
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
            Self::ProfileInspect { .. } => "profile-inspect",
            Self::ProfileRender { .. } => "profile-render",
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

fn render_profile_html(report: &ProfileReport) -> Result<String> {
    let tiles = report
        .tiles
        .iter()
        .map(|tile| {
            let samples = tile
                .samples
                .iter()
                .map(|sample| {
                    serde_json::json!({
                        "phase": sample.step.phase,
                        "epoch": sample.step.epoch,
                        "localIndex": sample.step.local_index,
                        "operation": sample.step.operation,
                        "kind": match sample.step.kind {
                            ProfileStepKind::Exchange => "exchange",
                            ProfileStepKind::Compute => "compute",
                        },
                        "start": sample.start_cycle,
                        "end": sample.end_cycle,
                        "duration": sample.end_cycle.wrapping_sub(sample.start_cycle),
                    })
                })
                .collect::<Vec<_>>();
            serde_json::json!({
                "physicalTile": tile.physical_tile,
                "samples": samples,
            })
        })
        .collect::<Vec<_>>();
    let total_samples: usize = report.tiles.iter().map(|tile| tile.samples.len()).sum();
    let payload = serde_json::json!({
        "clockHz": report.clock_hz,
        "tileCount": report.tiles.len(),
        "sampleCount": total_samples,
        "tiles": tiles,
    });
    let payload = serde_json::to_string(&payload)?
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026");
    Ok(PROFILE_REPORT_HTML.replace("__PROFILE_JSON__", &payload))
}

const PROFILE_REPORT_HTML: &str = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IPU Profile</title>
<style>
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --ink: #18202a;
  --muted: #637083;
  --line: #d8dee8;
  --exchange: #1f8a70;
  --compute: #c15a2e;
  --selected: #335cff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font: 13px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--bg);
}
header, main { padding: 16px 20px; }
header {
  display: grid;
  grid-template-columns: minmax(180px, 1fr) auto;
  gap: 12px;
  align-items: end;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
  position: sticky;
  top: 0;
  z-index: 5;
}
h1 { margin: 0; font-size: 20px; font-weight: 700; letter-spacing: 0; }
.stats { color: var(--muted); display: flex; gap: 12px; flex-wrap: wrap; margin-top: 2px; }
.controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }
label { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); }
input, select, button {
  height: 30px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
  padding: 0 9px;
  font: inherit;
}
input[type="number"] { width: 78px; }
button { cursor: pointer; }
button.active { border-color: var(--selected); color: var(--selected); }
main { display: grid; gap: 18px; }
section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
}
.section-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
}
.section-title { font-weight: 700; }
.hint { color: var(--muted); }
.svg-wrap { overflow: auto; }
svg { display: block; min-width: 100%; background: #fff; }
.bar { stroke: rgba(0,0,0,.16); stroke-width: 1; cursor: pointer; }
.bar.dim { opacity: .18; }
.bar.selected { stroke: var(--selected); stroke-width: 2; }
.label { pointer-events: none; fill: #fff; font-size: 11px; dominant-baseline: middle; }
.axis, .tile-label { fill: var(--muted); font-size: 11px; }
.grid { stroke: #edf0f5; stroke-width: 1; }
#details {
  display: grid;
  grid-template-columns: repeat(5, minmax(120px, 1fr));
  gap: 10px;
  padding: 12px;
  border-top: 1px solid var(--line);
}
.detail b { display: block; font-size: 11px; color: var(--muted); font-weight: 600; }
.detail span { font-variant-numeric: tabular-nums; }
#tooltip {
  position: fixed;
  max-width: 360px;
  pointer-events: none;
  background: #111827;
  color: #fff;
  border-radius: 6px;
  padding: 8px 9px;
  font-size: 12px;
  box-shadow: 0 8px 24px rgba(0,0,0,.18);
  display: none;
  z-index: 20;
}
@media (max-width: 760px) {
  header { grid-template-columns: 1fr; }
  .controls { justify-content: flex-start; }
  #details { grid-template-columns: 1fr 1fr; }
}
</style>
</head>
<body>
<header>
  <div>
    <h1>IPU Profile</h1>
    <div class="stats" id="stats"></div>
  </div>
  <div class="controls">
    <label>Search <input id="search" type="search" placeholder="operation"></label>
    <label>Kind <select id="kind"><option value="all">all</option><option value="compute">compute</option><option value="exchange">exchange</option></select></label>
    <label>Tiles <input id="tileStart" type="number" min="0" value="0"> to <input id="tileEnd" type="number" min="0" value="63"></label>
    <button id="zoomOut" type="button">-</button>
    <button id="zoomIn" type="button">+</button>
    <button id="reset" type="button">reset</button>
  </div>
</header>
<main>
  <section>
    <div class="section-head">
      <div class="section-title">Flamegraph</div>
      <div class="hint">Click a bar to focus matching timeline samples.</div>
    </div>
    <div class="svg-wrap"><svg id="flame"></svg></div>
  </section>
  <section>
    <div class="section-head">
      <div class="section-title">Tilewise Timeline</div>
      <div class="hint">Rendered by physical tile; horizontal scale is cycle count.</div>
    </div>
    <div class="svg-wrap"><svg id="timeline"></svg></div>
    <div id="details"></div>
  </section>
</main>
<div id="tooltip"></div>
<script>
const DATA = __PROFILE_JSON__;
const state = { query: "", kind: "all", tileStart: 0, tileEnd: 63, zoom: 1, selected: null };
const ns = "http://www.w3.org/2000/svg";
const flame = document.getElementById("flame");
const timeline = document.getElementById("timeline");
const tooltip = document.getElementById("tooltip");
const details = document.getElementById("details");
const colors = { exchange: "#1f8a70", compute: "#c15a2e" };
const textColor = { exchange: "#ffffff", compute: "#ffffff" };
const tiles = DATA.tiles.slice().sort((a, b) => a.physicalTile - b.physicalTile);
const maxTile = tiles.reduce((m, t) => Math.max(m, t.physicalTile), 0);
document.getElementById("tileEnd").value = Math.min(maxTile, 63);
state.tileEnd = Math.min(maxTile, 63);
document.getElementById("stats").textContent =
  `${DATA.tileCount} tiles  ${DATA.sampleCount} samples  ${formatCycles(totalCycles())} aggregate cycles  ${formatHz(DATA.clockHz)}`;

function totalCycles() {
  return tiles.flatMap(t => t.samples).reduce((sum, s) => sum + s.duration, 0);
}
function formatHz(hz) {
  if (!hz) return "clock unknown";
  if (hz >= 1e9) return `${(hz / 1e9).toFixed(2)} GHz`;
  if (hz >= 1e6) return `${(hz / 1e6).toFixed(1)} MHz`;
  return `${hz} Hz`;
}
function formatCycles(value) {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}G`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return `${value}`;
}
function clear(svg) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
}
function el(name, attrs = {}, text = "") {
  const node = document.createElementNS(ns, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text) node.textContent = text;
  return node;
}
function matches(sample) {
  if (state.kind !== "all" && sample.kind !== state.kind) return false;
  if (state.query && !sample.operation.toLowerCase().includes(state.query)) return false;
  if (state.selected) {
    if (state.selected.kind && sample.kind !== state.selected.kind) return false;
    if (state.selected.operation && sample.operation !== state.selected.operation) return false;
    if (state.selected.phase !== undefined && sample.phase !== state.selected.phase) return false;
    if (state.selected.epoch !== undefined && sample.epoch !== state.selected.epoch) return false;
  }
  return true;
}
function showTip(event, lines) {
  tooltip.innerHTML = lines.map(line => `<div>${escapeHtml(line)}</div>`).join("");
  tooltip.style.display = "block";
  tooltip.style.left = `${Math.min(window.innerWidth - 380, event.clientX + 12)}px`;
  tooltip.style.top = `${event.clientY + 12}px`;
}
function hideTip() { tooltip.style.display = "none"; }
function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function drawFlame() {
  clear(flame);
  const width = Math.max(960, flame.parentElement.clientWidth - 2);
  const rowH = 24, top = 18;
  const root = { name: "profile", value: 0, depth: 0, children: new Map() };
  for (const tile of tiles) {
    for (const sample of tile.samples) {
      if (state.kind !== "all" && sample.kind !== state.kind) continue;
      if (state.query && !sample.operation.toLowerCase().includes(state.query)) continue;
      let kind = root.children.get(sample.kind);
      if (!kind) root.children.set(sample.kind, kind = { name: sample.kind, kind: sample.kind, value: 0, depth: 1, children: new Map() });
      let op = kind.children.get(sample.operation);
      if (!op) kind.children.set(sample.operation, op = { name: sample.operation, kind: sample.kind, operation: sample.operation, value: 0, depth: 2, children: new Map() });
      const phaseKey = `phase ${sample.phase} epoch ${sample.epoch}`;
      let phase = op.children.get(phaseKey);
      if (!phase) op.children.set(phaseKey, phase = { name: phaseKey, kind: sample.kind, operation: sample.operation, phase: sample.phase, epoch: sample.epoch, value: 0, depth: 3, children: new Map() });
      phase.value += sample.duration;
      op.value += sample.duration;
      kind.value += sample.duration;
      root.value += sample.duration;
    }
  }
  const rows = [root, ...[...root.children.values()].sort(byValue)];
  const nodes = [];
  layout([...root.children.values()].sort(byValue), 0, width, 1, nodes);
  const height = top + 4 * rowH + 8;
  flame.setAttribute("viewBox", `0 0 ${width} ${height}`);
  flame.setAttribute("width", width);
  flame.setAttribute("height", height);
  flame.appendChild(el("text", { x: 8, y: 12, class: "axis" }, root.value ? `${formatCycles(root.value)} cycles shown` : "No samples match"));
  for (const node of nodes) {
    if (node.w < 1) continue;
    const rect = el("rect", {
      x: node.x, y: top + node.depth * rowH, width: Math.max(1, node.w - 1), height: rowH - 3,
      fill: node.kind ? colors[node.kind] : "#73808f", class: "bar"
    });
    rect.addEventListener("mousemove", event => showTip(event, tooltipLines(node)));
    rect.addEventListener("mouseleave", hideTip);
    rect.addEventListener("click", () => {
      state.selected = { kind: node.kind, operation: node.operation, phase: node.phase, epoch: node.epoch };
      render();
    });
    flame.appendChild(rect);
    if (node.w > 54) {
      flame.appendChild(el("text", {
        x: node.x + 5, y: top + node.depth * rowH + rowH / 2 - 1, class: "label", fill: textColor[node.kind] || "#fff"
      }, node.name.length > node.w / 7 ? node.name.slice(0, Math.max(3, Math.floor(node.w / 7) - 1)) + "..." : node.name));
    }
  }
}
function layout(children, x, w, depth, out) {
  const total = children.reduce((sum, child) => sum + child.value, 0);
  let cursor = x;
  for (const child of children) {
    const childW = total ? w * child.value / total : 0;
    out.push({ ...child, x: cursor, w: childW, depth });
    layout([...child.children.values()].sort(byValue), cursor, childW, depth + 1, out);
    cursor += childW;
  }
}
function byValue(a, b) { return b.value - a.value || a.name.localeCompare(b.name); }
function tooltipLines(node) {
  const pct = totalCycles() ? `${(100 * node.value / totalCycles()).toFixed(2)}% aggregate` : "";
  return [node.name, `${formatCycles(node.value)} cycles ${pct}`, node.kind || ""].filter(Boolean);
}
function drawTimeline() {
  clear(timeline);
  const visibleTiles = tiles.filter(t => t.physicalTile >= state.tileStart && t.physicalTile <= state.tileEnd);
  const left = 64, top = 22, right = 16, rowH = 17;
  let maxEnd = 1;
  for (const tile of visibleTiles) {
    for (const sample of tile.samples) {
      maxEnd = Math.max(maxEnd, sample.start + sample.duration);
    }
  }
  const width = Math.max(1024, Math.round((timeline.parentElement.clientWidth - 2) * state.zoom));
  const plotW = width - left - right;
  const height = top + Math.max(1, visibleTiles.length) * rowH + 28;
  timeline.setAttribute("viewBox", `0 0 ${width} ${height}`);
  timeline.setAttribute("width", width);
  timeline.setAttribute("height", height);
  for (let i = 0; i <= 10; i++) {
    const x = left + plotW * i / 10;
    timeline.appendChild(el("line", { x1: x, y1: top - 12, x2: x, y2: height - 20, class: "grid" }));
    timeline.appendChild(el("text", { x: x + 2, y: 12, class: "axis" }, formatCycles(Math.round(maxEnd * i / 10))));
  }
  visibleTiles.forEach((tile, index) => {
    const y = top + index * rowH;
    timeline.appendChild(el("text", { x: 8, y: y + 11, class: "tile-label" }, `tile ${tile.physicalTile}`));
    timeline.appendChild(el("line", { x1: left, y1: y + rowH - 2, x2: width - right, y2: y + rowH - 2, class: "grid" }));
    for (const sample of tile.samples) {
      const x = left + plotW * sample.start / maxEnd;
      const w = Math.max(1, plotW * sample.duration / maxEnd);
      const selected = matches(sample);
      const rect = el("rect", {
        x, y: y + 2, width: w, height: rowH - 5, fill: colors[sample.kind],
        class: `bar ${selected ? "" : "dim"} ${sampleSelected(sample) ? "selected" : ""}`,
      });
      rect.addEventListener("mousemove", event => showTip(event, [
        `tile ${tile.physicalTile}`,
        sample.operation,
        `${sample.kind} phase ${sample.phase} epoch ${sample.epoch}`,
        `start ${sample.start}  duration ${sample.duration} cycles`,
      ]));
      rect.addEventListener("mouseleave", hideTip);
      rect.addEventListener("click", () => {
        state.selected = { kind: sample.kind, operation: sample.operation, phase: sample.phase, epoch: sample.epoch };
        render();
      });
      timeline.appendChild(rect);
    }
  });
}
function sampleSelected(sample) {
  return state.selected &&
    (!state.selected.kind || sample.kind === state.selected.kind) &&
    (!state.selected.operation || sample.operation === state.selected.operation) &&
    (state.selected.phase === undefined || sample.phase === state.selected.phase) &&
    (state.selected.epoch === undefined || sample.epoch === state.selected.epoch);
}
function drawDetails() {
  const samples = tiles.flatMap(tile => tile.samples.map(sample => ({ tile: tile.physicalTile, ...sample }))).filter(matches);
  const cycles = samples.reduce((sum, sample) => sum + sample.duration, 0);
  const longest = samples.reduce((best, sample) => sample.duration > (best?.duration || 0) ? sample : best, null);
  const ops = new Set(samples.map(sample => sample.operation)).size;
  details.innerHTML = [
    ["Selected samples", samples.length],
    ["Selected cycles", formatCycles(cycles)],
    ["Operations", ops],
    ["Longest sample", longest ? `${formatCycles(longest.duration)} cycles` : "none"],
    ["Longest owner", longest ? `tile ${longest.tile} ${longest.operation}` : "none"],
  ].map(([key, value]) => `<div class="detail"><b>${escapeHtml(key)}</b><span>${escapeHtml(value)}</span></div>`).join("");
}
function render() {
  state.query = document.getElementById("search").value.trim().toLowerCase();
  state.kind = document.getElementById("kind").value;
  state.tileStart = Number(document.getElementById("tileStart").value || 0);
  state.tileEnd = Number(document.getElementById("tileEnd").value || maxTile);
  if (state.tileStart > state.tileEnd) [state.tileStart, state.tileEnd] = [state.tileEnd, state.tileStart];
  drawFlame();
  drawTimeline();
  drawDetails();
}
document.getElementById("search").addEventListener("input", () => { state.selected = null; render(); });
document.getElementById("kind").addEventListener("change", () => { state.selected = null; render(); });
document.getElementById("tileStart").addEventListener("change", render);
document.getElementById("tileEnd").addEventListener("change", render);
document.getElementById("zoomIn").addEventListener("click", () => { state.zoom = Math.min(8, state.zoom * 1.35); render(); });
document.getElementById("zoomOut").addEventListener("click", () => { state.zoom = Math.max(1, state.zoom / 1.35); render(); });
document.getElementById("reset").addEventListener("click", () => {
  state.selected = null; state.zoom = 1;
  document.getElementById("search").value = "";
  document.getElementById("kind").value = "all";
  document.getElementById("tileStart").value = 0;
  document.getElementById("tileEnd").value = Math.min(maxTile, 63);
  render();
});
window.addEventListener("resize", render);
render();
</script>
</body>
</html>
"##;

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
