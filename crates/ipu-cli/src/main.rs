mod profile_report;
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use ipu_driver::{Device, block_device_interrupt_signals};
use ipu_elf::{LinkOptions, Toolchain, inspect_object, link, source_tree_digest};
use ipu_package::{Application, ProfileReport, ProfileStepKind};
use ipu_profile::{
    GroupBy, Query, SortBy, calibrate_profiles, exchange_activity_summary, exchange_boundaries,
    phase_work, query,
};
use ipu_runtime::Runtime;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::PathBuf;
use tracing::info;

#[derive(Parser)]
#[command(version, about = "Low-level Graphcore IPU tools")]
struct Arguments {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Hash kernel sources to identify compatible cycle measurements.
    KernelBuildId {
        source_directory: PathBuf,
    },
    KernelCompile {
        source: PathBuf,
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
    PackageInspect {
        package: PathBuf,
        #[arg(long)]
        bindings: bool,
        #[arg(long)]
        tile: Option<u32>,
    },
    /// Separate exchange arrival imbalance from scheduled transfer duration.
    ProfileBarriers {
        profile: PathBuf,
        #[arg(long)]
        json: bool,
    },
    ProfileInspect {
        profile: PathBuf,
    },
    ProfileExtract {
        package: PathBuf,
        device_output: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 1_500_000_000)]
        clock_hz: u64,
    },
    ProfileRender {
        profile: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        /// Embed all transfer details for a portable (potentially large) HTML file.
        #[arg(long)]
        single_file: bool,
    },
    /// Render an exact tile placement dump with lazily loaded tile data.
    MemoryProfileRender {
        profile: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Collate profiled local work into a machine-readable estimator database.
    ProfileCalibrate {
        #[arg(required = true)]
        profiles: Vec<PathBuf>,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value = "ipu21")]
        target: String,
        /// Kernel/runtime build identity used to reject stale measurements.
        #[arg(long)]
        build_id: String,
    },
    /// Find short or sparsely occupied compute rounds and their next global barrier.
    ProfilePhases {
        profile: PathBuf,
        #[arg(long)]
        epoch: Option<u32>,
        /// Maximum measured compute-span cycles (excluding the following exchange).
        #[arg(long)]
        max_cycles: Option<u64>,
        /// Maximum average fraction of device tiles computing, from 0 to 1.
        #[arg(long)]
        max_occupancy: Option<f64>,
        /// Include rounds whose preparation or following exchange overlaps this offset.
        #[arg(long)]
        at_offset: Option<u64>,
        #[arg(long)]
        shared_clock: bool,
        #[arg(long, value_enum, default_value = "occupancy")]
        sort_by: PhaseSort,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    ProfileQuery {
        profile: PathBuf,
        #[arg(long, value_enum, default_value = "kernel")]
        group_by: ProfileGroup,
        #[arg(long, value_enum, default_value = "phase-cycles")]
        sort_by: ProfileSort,
        #[arg(long, value_enum)]
        kind: Option<ProfileKind>,
        #[arg(long)]
        kernel: Option<String>,
        #[arg(long)]
        operation_contains: Option<String>,
        #[arg(long)]
        tile: Vec<u32>,
        #[arg(long)]
        phase: Vec<u32>,
        /// Include the longest matching individual samples.
        #[arg(long, default_value_t = 0)]
        samples: usize,
        /// Restrict matches to samples active at this normalized cycle offset.
        #[arg(long)]
        at_offset: Option<u64>,
        /// Preserve the leading shared-clock interval before every tile has entered profiling.
        #[arg(long)]
        shared_clock: bool,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long)]
        json: bool,
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

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PhaseSort {
    Occupancy,
    Duration,
    Offset,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ProfileGroup {
    Kind,
    Kernel,
    Operation,
    Phase,
    Tile,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ProfileSort {
    PhaseCycles,
    WorkCycles,
    MaximumCycles,
    Samples,
    Name,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ProfileKind {
    Compute,
    Exchange,
    Synchronization,
    Idle,
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    ipu_runtime::init_tracing();
    match Arguments::parse().command {
        Command::KernelBuildId { source_directory } => {
            println!("{}", source_tree_digest(source_directory)?);
        }
        Command::KernelCompile {
            source,
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
            let artifact = Toolchain::from_sdk(sdk).compile(&source, &name, &flags)?;
            let summary = artifact.inspect()?;
            println!("object={}", artifact.object.display());
            println!("metadata={}", artifact.metadata.display());
            println!("sha256={}", hex::encode(artifact.digest()?));
            println!("defined={}", summary.defined_symbols.join(","));
            println!("relocations={:?}", summary.relocation_types);
        }
        Command::ObjectInspect { object } => {
            let bytes = fs::read(&object).with_context(|| format!("read {}", object.display()))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&inspect_object(&bytes)?)?
            );
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
                    image_base: ipu_target::ipu21::memory::TILE_MEMORY_BASE,
                    regions: vec![(
                        base,
                        ipu_target::ipu21::memory::IPU21_EXECUTABLE_MEMORY_LIMIT,
                    )],
                    entry_symbol: entry,
                    retained_symbols: Vec::new(),
                    externals: symbol.into_iter().collect(),
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
        Command::PackageInspect {
            package,
            bindings,
            tile,
        } => inspect_package(&package, bindings, tile)?,
        Command::ProfileInspect { profile } => {
            let report = ProfileReport::read(fs::File::open(&profile)?)?;
            let samples: usize = report.tiles.iter().map(|tile| tile.samples.len()).sum();
            let exchange = exchange_activity_summary(&report);
            println!(
                "clockHz={} tiles={} samples={} exchangeSamples={} describedExchangeSamples={} sendIntervals={} receiveIntervals={} partnerBusyIntervals={} estimatedSendWorkCycles={} estimatedReceiveWorkCycles={} estimatedSimultaneousWorkCycles={} estimatedPartnerBusyWorkCycles={} estimatedExchangeIdleWorkCycles={} measuredExchangePhaseCycles={} scheduledExchangeEventCycles={} exchangeArrivalWaitCycles={} exchangePhaseBoundaryCycles={}",
                report.clock_hz,
                report.tiles.len(),
                samples,
                exchange.exchange_samples,
                exchange.described_samples,
                exchange.send_intervals,
                exchange.receive_intervals,
                exchange.partner_busy_intervals,
                exchange.estimated_send_work_cycles,
                exchange.estimated_receive_work_cycles,
                exchange.estimated_simultaneous_work_cycles,
                exchange.estimated_partner_busy_work_cycles,
                exchange.estimated_idle_work_cycles,
                exchange.measured_phase_cycles,
                exchange.scheduled_event_cycles,
                exchange.arrival_wait_cycles,
                exchange.phase_boundary_cycles,
            );
        }
        Command::ProfileExtract {
            package,
            device_output,
            output,
            clock_hz,
        } => {
            let application = Application::read(fs::File::open(&package)?)?;
            let device_output = fs::read(&device_output)?;
            let report = application.profile_report(&device_output, clock_hz)?;
            report.write(fs::File::create(&output)?)?;
            println!(
                "package={} deviceOutputBytes={} tiles={} profile={}",
                package.display(),
                device_output.len(),
                report.tiles.len(),
                output.display(),
            );
        }
        Command::ProfileRender {
            profile,
            output,
            single_file,
        } => {
            let report = ProfileReport::read(fs::File::open(&profile)?)?;
            profile_report::write(&report, &output, single_file)?;
            println!(
                "profile={} tiles={} output={}",
                profile.display(),
                report.tiles.len(),
                output.display()
            );
        }
        Command::MemoryProfileRender { profile, output } => {
            ipu_codegen::render_memory_profile(fs::File::open(&profile)?, &output)?;
            println!("profile={} output={}", profile.display(), output.display());
        }
        Command::ProfileCalibrate {
            profiles,
            output,
            target,
            build_id,
        } => {
            let reports = profiles
                .iter()
                .map(|profile| {
                    ProfileReport::read(fs::File::open(profile)?)
                        .with_context(|| format!("reading {}", profile.display()))
                })
                .collect::<Result<Vec<_>>>()?;
            let database =
                calibrate_profiles(&reports, target, build_id).map_err(anyhow::Error::msg)?;
            fs::write(&output, serde_json::to_vec_pretty(&database)?)?;
            println!(
                "profiles={} measurements={} output={}",
                profiles.len(),
                database.measurements.len(),
                output.display()
            );
        }
        Command::ProfileBarriers { profile, json } => {
            let report = ProfileReport::read(fs::File::open(profile)?)?;
            let boundaries = exchange_boundaries(&report);
            if json {
                println!("{}", serde_json::to_string_pretty(&boundaries)?);
            } else {
                println!(
                    "Shared-clock cycle offsets; arrival spreads can overlap and must not be summed."
                );
                for phase in boundaries {
                    println!(
                        "epoch={} phase={} entry={}..{} exit={} lateTile={} arrivalSpread={} afterLastArrival={} scheduled={}",
                        phase.epoch,
                        phase.phase,
                        phase.first_entry,
                        phase.last_entry,
                        phase.last_exit,
                        phase.last_arriving_tile,
                        phase.arrival_spread_cycles,
                        phase.after_last_arrival_cycles,
                        phase
                            .scheduled_event_cycles
                            .map_or_else(|| "unknown".into(), |n| n.to_string()),
                    );
                }
            }
        }
        Command::ProfilePhases {
            profile,
            epoch,
            max_cycles,
            max_occupancy,
            at_offset,
            shared_clock,
            sort_by,
            limit,
            json,
        } => {
            if max_occupancy.is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v)) {
                bail!("--max-occupancy must be between 0 and 1");
            }
            let report = ProfileReport::read(fs::File::open(profile)?)?;
            let mut groups = phase_work(&report, shared_clock);
            groups.retain(|g| {
                epoch.is_none_or(|e| e == g.epoch)
                    && max_cycles.is_none_or(|c| g.span_cycles <= c)
                    && max_occupancy.is_none_or(|o| g.occupancy <= o)
                    && at_offset.is_none_or(|at| {
                        g.exchange
                            .as_ref()
                            .map_or(g.start, |b| g.start.min(b.first_entry))
                            <= at
                            && at
                                < g.exchange
                                    .as_ref()
                                    .map_or(g.end, |b| b.last_exit.max(g.end))
                    })
            });
            groups.sort_by(|a, b| {
                match sort_by {
                    PhaseSort::Occupancy => a
                        .occupancy
                        .total_cmp(&b.occupancy)
                        .then_with(|| b.span_cycles.cmp(&a.span_cycles)),
                    PhaseSort::Duration => b.span_cycles.cmp(&a.span_cycles),
                    PhaseSort::Offset => a.start.cmp(&b.start),
                }
                .then_with(|| a.start.cmp(&b.start))
            });
            if limit != 0 {
                groups.truncate(limit);
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&groups)?);
            } else {
                println!(
                    "{} offsets; rounds can overlap; occupancy is measured compute coverage.",
                    if shared_clock {
                        "Shared-clock"
                    } else {
                        "Renderer-cropped"
                    }
                );
                for g in groups {
                    let phase = g
                        .next_exchange_phase
                        .map_or("end".into(), |p| (p & 0x7fffffff).to_string());
                    println!(
                        "epoch={} next={} compute={}..{} span={} occupancy={:.2}% tiles={} work={} exposed={}",
                        g.epoch,
                        phase,
                        g.start,
                        g.end,
                        g.span_cycles,
                        g.occupancy * 100.0,
                        g.active_tiles,
                        g.work_cycles,
                        g.exposed_preparation_cycles
                            .map_or("N/A".into(), |n| n.to_string())
                    );
                    for k in &g.kernels {
                        println!(
                            "  {} {} tiles={} calls={} max={} cycles",
                            k.operation, k.kernel, k.tiles, k.samples, k.maximum_cycles
                        );
                    }
                    if let Some(b) = &g.exchange {
                        println!(
                            "  barrier={}..{} exchangeEnd={} scheduled={} lateTile={} unattributed={} exchangeOccupancy={}",
                            b.first_entry,
                            b.last_entry,
                            b.last_exit,
                            b.scheduled_event_cycles.unwrap_or(0),
                            b.last_arriving_tile,
                            g.unattributed_before_barrier_cycles,
                            g.exchange_occupancy
                                .map_or("N/A".into(), |v| format!("{:.2}%", v * 100.0))
                        );
                        for k in &g.late_tile_kernels {
                            println!(
                                "  late tile: {} {} {}..{}",
                                k.operation, k.kernel, k.start, k.end
                            );
                        }
                    }
                }
            }
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
            samples,
            at_offset,
            shared_clock,
            limit,
            json,
        } => {
            let report = ProfileReport::read(fs::File::open(profile)?)?;
            let result = query(
                &report,
                &Query {
                    group_by: match group_by {
                        ProfileGroup::Kind => GroupBy::Kind,
                        ProfileGroup::Kernel => GroupBy::Kernel,
                        ProfileGroup::Operation => GroupBy::Operation,
                        ProfileGroup::Phase => GroupBy::Phase,
                        ProfileGroup::Tile => GroupBy::Tile,
                    },
                    sort_by: match sort_by {
                        ProfileSort::PhaseCycles => SortBy::PhaseCycles,
                        ProfileSort::WorkCycles => SortBy::WorkCycles,
                        ProfileSort::MaximumCycles => SortBy::MaximumCycles,
                        ProfileSort::Samples => SortBy::Samples,
                        ProfileSort::Name => SortBy::Name,
                    },
                    kind: kind.map(|kind| match kind {
                        ProfileKind::Compute => ProfileStepKind::Compute,
                        ProfileKind::Exchange => ProfileStepKind::Exchange,
                        ProfileKind::Synchronization => ProfileStepKind::Synchronization,
                        ProfileKind::Idle => ProfileStepKind::Idle,
                    }),
                    kernel,
                    operation_contains,
                    tiles: tile.into_iter().collect::<BTreeSet<_>>(),
                    phases: phase.into_iter().collect::<BTreeSet<_>>(),
                    at_offset,
                    shared_clock,
                    limit: (limit != 0).then_some(limit),
                    sample_limit: samples,
                    ..Query::default()
                },
            );
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!(
                    "tiles={} matched={}/{} spanCycles={} spanMs={:.6}",
                    result.tile_count,
                    result.matched_sample_count,
                    result.sample_count,
                    result.profile_span_cycles,
                    result.profile_span_ms
                );
                for group in result.groups {
                    println!(
                        "name={:?} phases={} tiles={} samples={} timelineCycles={} workCycles={} range={}..{} p95={} max={} useful={} coverage={:.1}%",
                        group.name,
                        group.phase_count,
                        group.tile_count,
                        group.sample_count,
                        group.phase_cycles,
                        group.work_cycles,
                        group.first_offset,
                        group.last_offset,
                        group.p95_cycles,
                        group.maximum_cycles,
                        group
                            .useful_utilization
                            .map_or_else(|| "n/a".into(), |v| format!("{:.1}%", v * 100.0)),
                        100.0 * group.estimated_work_cycles as f64
                            / group.work_cycles.max(1) as f64
                    );
                }
                for sample in result.samples {
                    println!(
                        "sample tile={} range={}..{} cycles={} phase={}/{} operation={:?} kernel={:?}",
                        sample.physical_tile,
                        sample.offset,
                        sample.offset + u64::from(sample.duration),
                        sample.duration,
                        sample.phase,
                        sample.epoch,
                        sample.operation,
                        sample.kernel,
                    );
                }
            }
        }
        Command::DeviceProbe { device } => {
            block_device_interrupt_signals()?;
            let device = Device::open(&device)?;
            device.initialize()?;
            println!("deviceProbe=PASS");
        }
        Command::Load {
            package,
            bootloader,
            configuration,
            final_mark,
            device,
        } => {
            let application = Application::read(fs::File::open(package)?)?;
            let runtime = Runtime::open(&device, &fs::read(configuration)?)?;
            runtime.load(&application, &fs::read(bootloader)?, final_mark)?;
            println!("tiles={} load=PASS", application.tiles.len());
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
            let application = Application::read(fs::File::open(package)?)?;
            let runtime = Runtime::open(&device, &fs::read(configuration)?)?;
            runtime.load(
                &application,
                &fs::read(bootloader)?,
                application.host_exchange.startup_mark,
            )?;
            let inputs = input.into_iter().collect::<HashMap<_, _>>();
            let outputs = output.into_iter().collect::<HashMap<_, _>>();
            let mut session = runtime.host_session(&application)?;
            session.start()?;
            for call in &calls {
                let input = inputs
                    .get(call)
                    .map(fs::read)
                    .transpose()?
                    .unwrap_or_default();
                let result = session.invoke(call, &input)?;
                match (result.is_empty(), outputs.get(call)) {
                    (_, Some(path)) => fs::write(path, result)?,
                    (true, None) => {}
                    (false, None) => {
                        bail!("call {call:?} produced output without --output {call}=PATH")
                    }
                }
            }
            println!("calls={} hostRun=PASS", calls.len());
        }
    }
    info!("command completed");
    Ok(())
}

fn inspect_package(path: &PathBuf, bindings: bool, physical_tile: Option<u32>) -> Result<()> {
    let application = Application::read(fs::File::open(path)?)?;
    let stored: usize = application
        .tiles
        .iter()
        .flat_map(|tile| &tile.segments)
        .map(|segment| segment.data.len())
        .sum();
    println!(
        "producer={} tiles={} storedBytes={} inputs={} outputs={} weights={} entries={} calls={}",
        application.compiler_version,
        application.tiles.len(),
        stored,
        application.inputs.len(),
        application.outputs.len(),
        application.weights.len(),
        application.entry_points.len(),
        application.host_exchange.calls.len()
    );
    if bindings {
        for (kind, group) in [
            ("input", &application.inputs),
            ("output", &application.outputs),
            ("weight", &application.weights),
        ] {
            for binding in group {
                println!(
                    "binding kind={kind} name={:?} dtype={} shape={:?} slices={} allocatedBytes={}",
                    binding.name,
                    binding.dtype,
                    binding.shape,
                    binding.slices.len(),
                    binding.slices.iter().map(|slice| slice.size).sum::<u64>()
                );
            }
        }
    }
    if let Some(physical_tile) = physical_tile {
        let tile = application
            .tiles
            .iter()
            .find(|tile| tile.physical_tile == physical_tile)
            .with_context(|| format!("package has no physical tile {physical_tile}"))?;
        println!(
            "tile={} entry=0x{:x} command=0x{:x} diagnostic=0x{:x} segments={}",
            tile.physical_tile,
            tile.entry_point,
            tile.command_address,
            tile.diagnostic_address,
            tile.segments.len()
        );
        for segment in &tile.segments {
            println!(
                "segment address=0x{:x} memorySize={} dataBytes={} flags=0x{:x}",
                segment.address,
                segment.memory_size,
                segment.data.len(),
                segment.flags
            );
        }
    }
    Ok(())
}

fn parse_u32(value: &str) -> Result<u32, String> {
    let (digits, radix) = value
        .strip_prefix("0x")
        .map_or((value, 10), |digits| (digits, 16));
    u32::from_str_radix(digits, radix).map_err(|error| error.to_string())
}

fn parse_symbol(value: &str) -> Result<(String, u32), String> {
    let (name, value) = value
        .split_once('=')
        .ok_or_else(|| "expected NAME=ADDRESS".to_string())?;
    Ok((name.into(), parse_u32(value)?))
}

fn parse_named_path(value: &str) -> Result<(String, PathBuf), String> {
    let (name, path) = value
        .split_once('=')
        .ok_or_else(|| "expected NAME=PATH".to_string())?;
    if name.is_empty() || path.is_empty() {
        return Err("expected non-empty NAME=PATH".into());
    }
    Ok((name.into(), path.into()))
}
