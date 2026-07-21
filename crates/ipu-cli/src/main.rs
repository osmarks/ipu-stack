use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use ipu_compiler::{
    CompilerOptions, EncoderConfig, EncoderWeights, compile, encoder_graph, encoder_reference,
};
use ipu_driver::{Device, HostBuffer, HostSession, Loader, block_device_interrupt_signals};
use ipu_elf::{LinkOptions, Toolchain, inspect_object, link};
use ipu_exchange::Topology;
use ipu_package::{
    Application, EntryPoint, HostCall, HostExchange, HostPage, HostSlice, MemoryProfile,
    ProfileReport, ProfileStepKind, SEGMENT_EXECUTE, SEGMENT_READ, SEGMENT_WRITE, Segment,
    TileImage,
};
use ipu_profile::{GroupBy, MetadataFilter, Query, SortBy, StepKind, query as query_profile};
use object::{Object, ObjectSegment};
use std::collections::{BTreeMap, BTreeSet, HashMap};
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

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ProfileGroupArgument {
    Kind,
    Kernel,
    Operation,
    Phase,
    Tile,
    Metadata,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ProfileSortArgument {
    PhaseCycles,
    WorkCycles,
    MaximumCycles,
    Samples,
    Name,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ProfileKindArgument {
    Compute,
    Exchange,
    Synchronization,
    Idle,
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
        #[arg(long)]
        bindings: bool,
        #[arg(long)]
        tile: Option<u32>,
        #[arg(long, requires = "tile")]
        address: Option<u32>,
    },
    ProfileInspect {
        profile: PathBuf,
    },
    ProfileRender {
        profile: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    ProfileQuery {
        /// Cap'n Proto cycle profile to query.
        profile: PathBuf,
        /// Dimension used to aggregate matching samples.
        #[arg(long, value_enum, default_value = "kernel")]
        group_by: ProfileGroupArgument,
        /// Ordering for aggregate rows.
        #[arg(long, value_enum, default_value = "phase-cycles")]
        sort_by: ProfileSortArgument,
        /// Select only compute or exchange samples.
        #[arg(long, value_enum)]
        kind: Option<ProfileKindArgument>,
        /// Select an exact kernel symbol.
        #[arg(long)]
        kernel: Option<String>,
        /// Select operation labels containing this case-sensitive text.
        #[arg(long)]
        operation_contains: Option<String>,
        /// Select a physical tile; repeat for multiple tiles.
        #[arg(long)]
        tile: Vec<u32>,
        /// Select a scheduled phase; repeat for multiple phases.
        #[arg(long)]
        phase: Vec<u32>,
        /// Require metadata NAME or NAME=VALUE; repeat to combine filters.
        #[arg(long, value_parser = parse_metadata_filter)]
        metadata: Vec<MetadataFilter>,
        /// Metadata field used by --group-by metadata.
        #[arg(long)]
        metadata_key: Option<String>,
        /// Select samples active at this per-tile normalized cycle offset.
        #[arg(long)]
        at: Option<u64>,
        /// Maximum aggregate rows; zero means unlimited.
        #[arg(long, default_value_t = 20)]
        limit: usize,
        /// Include this many longest matching samples with full metadata.
        #[arg(long, default_value_t = 0)]
        samples: usize,
        /// Emit a stable machine-readable JSON result.
        #[arg(long)]
        json: bool,
    },
    MemoryInspect {
        profile: PathBuf,
        #[arg(long)]
        tile: Vec<u16>,
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
                    regions: Vec::new(),
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
        Command::PackageInspect {
            package,
            bindings,
            tile,
            address,
        } => {
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
            if bindings {
                for (kind, group) in [
                    ("input", &app.inputs),
                    ("output", &app.outputs),
                    ("weight", &app.weights),
                ] {
                    for binding in group {
                        println!(
                            "binding kind={kind} name={} dtype={} shape={:?} slices={}",
                            binding.name,
                            binding.dtype,
                            binding.shape,
                            binding.slices.len()
                        );
                        for (index, slice) in binding.slices.iter().enumerate() {
                            println!(
                                "slice binding={} index={index} tile={} address={:#x} offset={} size={}",
                                binding.name,
                                slice.tile,
                                slice.tile_address,
                                slice.file_offset,
                                slice.size
                            );
                        }
                    }
                }
            }
            if let Some(physical_tile) = tile {
                let tile = app
                    .tiles
                    .iter()
                    .find(|tile| tile.physical_tile == physical_tile)
                    .ok_or_else(|| {
                        anyhow::anyhow!("package has no physical tile {physical_tile}")
                    })?;
                println!(
                    "tile={} entry={:#x} command={:#x} diagnostic={:#x} segments={}",
                    tile.physical_tile,
                    tile.entry_point,
                    tile.command_address,
                    tile.diagnostic_address,
                    tile.segments.len()
                );
                for segment in &tile.segments {
                    println!(
                        "segment address={:#x} memorySize={} fileSize={} flags={:#x} blob={} blobOffset={}",
                        segment.address,
                        segment.memory_size,
                        segment.file_size,
                        segment.flags,
                        segment.blob,
                        segment.blob_offset
                    );
                }
                if let Some(address) = address {
                    let segment = tile
                        .segments
                        .iter()
                        .find(|segment| {
                            address >= segment.address
                                && address < segment.address + segment.file_size
                        })
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "address 0x{address:x} is not stored for tile {physical_tile}"
                            )
                        })?;
                    let offset = usize::try_from(
                        segment.blob_offset + u64::from(address - segment.address),
                    )?;
                    let bytes = &app.blobs[segment.blob].bytes;
                    for (index, word) in bytes[offset..].chunks_exact(4).take(16).enumerate() {
                        println!(
                            "word address={:#x} value={:#010x}",
                            address + u32::try_from(index * 4)?,
                            u32::from_le_bytes(word.try_into().unwrap())
                        );
                    }
                }
            }
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
        Command::ProfileQuery {
            profile,
            group_by,
            sort_by,
            kind,
            kernel,
            operation_contains,
            tile,
            phase,
            metadata,
            metadata_key,
            at,
            limit,
            samples,
            json,
        } => {
            if matches!(group_by, ProfileGroupArgument::Metadata) && metadata_key.is_none() {
                bail!("--group-by metadata requires --metadata-key NAME");
            }
            let profile = ProfileReport::read(fs::File::open(profile)?)?;
            let result = query_profile(
                &profile,
                &Query {
                    group_by: match group_by {
                        ProfileGroupArgument::Kind => GroupBy::Kind,
                        ProfileGroupArgument::Kernel => GroupBy::Kernel,
                        ProfileGroupArgument::Operation => GroupBy::Operation,
                        ProfileGroupArgument::Phase => GroupBy::Phase,
                        ProfileGroupArgument::Tile => GroupBy::Tile,
                        ProfileGroupArgument::Metadata => GroupBy::Metadata,
                    },
                    sort_by: match sort_by {
                        ProfileSortArgument::PhaseCycles => SortBy::PhaseCycles,
                        ProfileSortArgument::WorkCycles => SortBy::WorkCycles,
                        ProfileSortArgument::MaximumCycles => SortBy::MaximumCycles,
                        ProfileSortArgument::Samples => SortBy::Samples,
                        ProfileSortArgument::Name => SortBy::Name,
                    },
                    kind: kind.map(|kind| match kind {
                        ProfileKindArgument::Compute => StepKind::Compute,
                        ProfileKindArgument::Exchange => StepKind::Exchange,
                        ProfileKindArgument::Synchronization => StepKind::Synchronization,
                        ProfileKindArgument::Idle => StepKind::Idle,
                    }),
                    kernel,
                    operation_contains,
                    tiles: tile.into_iter().collect::<BTreeSet<_>>(),
                    phases: phase.into_iter().collect::<BTreeSet<_>>(),
                    metadata,
                    metadata_key,
                    at_offset: at,
                    limit: (limit != 0).then_some(limit),
                    sample_limit: samples,
                },
            );
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!(
                    "tiles={} samples={}/{} spanCycles={} spanMs={:.6} clockHz={} at={}",
                    result.tile_count,
                    result.matched_sample_count,
                    result.sample_count,
                    result.profile_span_cycles,
                    result.profile_span_ms,
                    result.clock_hz,
                    result
                        .at_offset
                        .map_or_else(|| "-".into(), |offset| offset.to_string())
                );
                println!(
                    "name\tphases\ttiles\tsamples\tphaseCycles\tphaseMs\tworkCycles\tavgActiveTiles\tmean\tp50\tp95\tmax"
                );
                for group in &result.groups {
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{:.2}\t{:.2}\t{}\t{}\t{}",
                        group.name,
                        group.phase_count,
                        group.tile_count,
                        group.sample_count,
                        group.phase_cycles,
                        group.phase_ms,
                        group.work_cycles,
                        group.average_active_tiles,
                        group.mean_cycles,
                        group.p50_cycles,
                        group.p95_cycles,
                        group.maximum_cycles,
                    );
                }
                for sample in &result.samples {
                    println!(
                        "sample tile={} offset={} duration={} phase={}/{} kind={} kernel={:?} operation={:?} metadata={}",
                        sample.physical_tile,
                        sample.offset,
                        sample.duration,
                        sample.phase,
                        sample.epoch,
                        sample.kind,
                        sample.kernel,
                        sample.operation,
                        serde_json::to_string(&sample.metadata)?,
                    );
                }
            }
        }
        Command::MemoryInspect { profile, tile } => {
            let report = MemoryProfile::read(fs::File::open(profile)?)?;
            for memory in report.tiles.iter().filter(|memory| {
                tile.is_empty()
                    || tile.contains(&memory.logical_tile)
                    || tile.contains(&memory.physical_tile)
            }) {
                println!(
                    "logicalTile={} physicalTile={} regions={}",
                    memory.logical_tile,
                    memory.physical_tile,
                    memory.regions.len()
                );
                for region in &memory.regions {
                    let live_until = if region.live_until == usize::MAX {
                        "end".into()
                    } else {
                        region.live_until.to_string()
                    };
                    println!(
                        "  address=0x{:x} size={} end=0x{:x} category={} tensor={} live={}..{} name={:?}",
                        region.address,
                        region.size,
                        region.address + region.size,
                        region.category,
                        region
                            .tensor
                            .map_or_else(|| "-".into(), |tensor| tensor.to_string()),
                        region.live_from,
                        live_until,
                        region.name
                    );
                }
            }
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
            let generated_call =
                calls.len() == 1 && protocol.calls.len() == 1 && calls[0] == protocol.calls[0].name;
            if generated_call {
                let call = &calls[0];
                let bytes = input
                    .get(call)
                    .map(fs::read)
                    .transpose()?
                    .unwrap_or_default();
                let result = ipu_runtime::run_host_with_options(
                    &app,
                    &bootloader,
                    &configuration,
                    &device,
                    &bytes,
                    ipu_runtime::HostRunOptions::default(),
                )
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                if let Some(path) = output.get(call) {
                    fs::write(path, &result)?;
                } else if !result.is_empty() {
                    bail!(
                        "call {call} produced {} bytes without --output",
                        result.len()
                    );
                }
            } else {
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
                    let result = session.invoke(&call, &bytes).map_err(|error| {
                        anyhow::anyhow!(
                            "{error}; device state: {}",
                            host_run_device_summary(&device, &app)
                        )
                    })?;
                    if let Some(path) = output.get(&call) {
                        fs::write(path, &result)?;
                    } else if !result.is_empty() {
                        bail!(
                            "call {call} produced {} bytes without --output",
                            result.len()
                        );
                    }
                }
            }
            println!("hostCalls={} directHostRun=PASS", app.entry_points.len());
        }
    }
    info!("command completed");
    Ok(())
}

fn host_run_device_summary(device: &Device, app: &Application) -> String {
    let mut counts = [0usize; 4];
    let mut exceptions = Vec::new();
    let mut read_errors = 0usize;
    for tile in &app.tiles {
        let physical = tile.physical_tile as u16;
        match device.tile_context_state(physical, 0) {
            Ok(state @ 0..=3) => {
                counts[state as usize] += 1;
                if state == 3 && exceptions.len() < 8 {
                    let status = device
                        .read_tile_context_status(physical, 0)
                        .map(ipu_driver::TileException::from_status);
                    let pc = device.read_tile_program_counter(physical, 0);
                    exceptions.push(format!("{}:{status:?}@{pc:?}", tile.physical_tile));
                }
            }
            Ok(_) | Err(_) => read_errors += 1,
        }
    }
    format!("context0={counts:?}, read_errors={read_errors}, supervisor_exceptions={exceptions:?}")
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
            Self::ProfileQuery { .. } => "profile-query",
            Self::MemoryInspect { .. } => "memory-inspect",
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
    #[derive(Clone, Copy, Hash, PartialEq, Eq)]
    struct StepKey {
        phase: u32,
        epoch: u32,
        operation: u32,
        kernel: u32,
        metadata: u32,
        kind: u8,
    }

    fn intern_string(
        values: &mut Vec<String>,
        indices: &mut HashMap<String, u32>,
        value: &str,
    ) -> u32 {
        if let Some(index) = indices.get(value) {
            return *index;
        }
        let index = values.len() as u32;
        values.push(value.into());
        indices.insert(value.into(), index);
        index
    }

    let mut strings = Vec::new();
    let mut string_indices = HashMap::new();
    let mut metadata_sets = Vec::<Vec<[u32; 2]>>::new();
    let mut metadata_indices = HashMap::<Vec<[u32; 2]>, u32>::new();
    let mut steps = Vec::<StepKey>::new();
    let mut step_indices = HashMap::<StepKey, u32>::new();
    let tiles = report
        .tiles
        .iter()
        .map(|tile| {
            let base_cycle = tile
                .samples
                .first()
                .map(|sample| sample.start_cycle)
                .unwrap_or(0);
            let samples = tile
                .samples
                .iter()
                .map(|sample| {
                    let metadata = sample
                        .step
                        .metadata
                        .iter()
                        .map(|entry| {
                            [
                                intern_string(&mut strings, &mut string_indices, &entry.name),
                                intern_string(&mut strings, &mut string_indices, &entry.value),
                            ]
                        })
                        .collect::<Vec<_>>();
                    let metadata = *metadata_indices.entry(metadata.clone()).or_insert_with(|| {
                        let index = metadata_sets.len() as u32;
                        metadata_sets.push(metadata);
                        index
                    });
                    let step = StepKey {
                        phase: sample.step.phase,
                        epoch: sample.step.epoch,
                        operation: intern_string(
                            &mut strings,
                            &mut string_indices,
                            &sample.step.operation,
                        ),
                        kernel: intern_string(
                            &mut strings,
                            &mut string_indices,
                            &sample.step.kernel,
                        ),
                        metadata,
                        kind: match sample.step.kind {
                            ProfileStepKind::Exchange => 0,
                            ProfileStepKind::Compute => 1,
                            ProfileStepKind::Synchronization => 2,
                            ProfileStepKind::Idle => 3,
                        },
                    };
                    let step = *step_indices.entry(step).or_insert_with(|| {
                        let index = steps.len() as u32;
                        steps.push(step);
                        index
                    });
                    serde_json::json!([
                        step,
                        sample.start_cycle.wrapping_sub(base_cycle),
                        sample.end_cycle.wrapping_sub(sample.start_cycle),
                    ])
                })
                .collect::<Vec<_>>();
            serde_json::json!({
                "physicalTile": tile.physical_tile,
                "samples": samples,
            })
        })
        .collect::<Vec<_>>();
    let total_samples: usize = report.tiles.iter().map(|tile| tile.samples.len()).sum();
    let mut metadata = Vec::new();
    let metadata_sets = metadata_sets
        .into_iter()
        .map(|entries| {
            let start = metadata.len() as u32;
            let count = entries.len() as u32;
            for [name, value] in entries {
                metadata.extend([name, value]);
            }
            [start, count]
        })
        .collect::<Vec<_>>();
    let steps = steps
        .into_iter()
        .map(|step| {
            serde_json::json!([
                step.phase,
                step.epoch,
                step.operation,
                step.kernel,
                step.metadata,
                step.kind,
            ])
        })
        .collect::<Vec<_>>();
    let payload = serde_json::json!({
        "clockHz": report.clock_hz,
        "tileCount": report.tiles.len(),
        "sampleCount": total_samples,
        "strings": strings,
        "metadata": metadata,
        "metadataSets": metadata_sets,
        "steps": steps,
        "tiles": tiles,
    });
    let payload = serde_json::to_string(&payload)?
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026");
    Ok(PROFILE_REPORT_HTML.replace("__PROFILE_JSON__", &payload))
}

const PROFILE_REPORT_HTML: &str = include_str!("profile_report.html");

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

fn parse_metadata_filter(value: &str) -> Result<MetadataFilter, String> {
    let (name, expected) = value
        .split_once('=')
        .map_or((value, None), |(name, expected)| (name, Some(expected)));
    if name.is_empty() {
        return Err("metadata filter name is empty".into());
    }
    Ok(MetadataFilter {
        name: name.into(),
        value: expected.map(Into::into),
    })
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
