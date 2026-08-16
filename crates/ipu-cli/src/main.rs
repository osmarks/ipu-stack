use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use ipu_driver::{Device, block_device_interrupt_signals};
use ipu_elf::{LinkOptions, Toolchain, inspect_object, link, source_tree_digest};
use ipu_package::{Application, ProfileExchangeActivityKind, ProfileReport, ProfileStepKind};
use ipu_profile::{
    GroupBy, Query, SortBy, calibrate_profiles, cycle_origin, exchange_activity_summary, query,
};
use ipu_runtime::Runtime;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::PathBuf;
use tracing::info;
use tracing_subscriber::EnvFilter;

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
    ProfileQuery {
        profile: PathBuf,
        #[arg(long, default_value_t = GroupBy::Kernel)]
        group_by: GroupBy,
        #[arg(long, default_value_t = SortBy::PhaseCycles)]
        sort_by: SortBy,
        #[arg(long)]
        kind: Option<ProfileStepKind>,
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

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    init_tracing();
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
                    image_base: ipu_package::TILE_MEMORY_BASE,
                    regions: vec![(base, ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT)],
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
                    group_by,
                    sort_by,
                    kind,
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
                        "name={:?} phases={} tiles={} samples={} timelineCycles={} workCycles={} range={}..{} p95={} max={}",
                        group.name,
                        group.phase_count,
                        group.tile_count,
                        group.sample_count,
                        group.phase_cycles,
                        group.work_cycles,
                        group.first_offset,
                        group.last_offset,
                        group.p95_cycles,
                        group.maximum_cycles
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
                    "binding kind={kind} name={:?} dtype={} shape={:?} slices={}",
                    binding.name,
                    binding.dtype,
                    binding.shape,
                    binding.slices.len()
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

fn render_profile_html(report: &ProfileReport) -> Result<String> {
    #[derive(Clone, Copy, Hash, PartialEq, Eq)]
    struct StepKey {
        phase: u32,
        epoch: u32,
        operation: u32,
        kernel: u32,
        metadata: u32,
        kind: u8,
        exchange_event_cycles: u32,
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
    let mut activity_sets = Vec::<Vec<[u32; 3]>>::new();
    let mut activity_indices = HashMap::<Vec<[u32; 3]>, u32>::new();
    let mut steps = Vec::<StepKey>::new();
    let mut step_indices = HashMap::<StepKey, u32>::new();
    let base_cycle = cycle_origin(report);
    let tiles = report
        .tiles
        .iter()
        .map(|tile| {
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
                        exchange_event_cycles: sample.step.exchange_event_cycles,
                    };
                    let step = *step_indices.entry(step).or_insert_with(|| {
                        let index = steps.len() as u32;
                        steps.push(step);
                        index
                    });
                    let activities = sample
                        .step
                        .exchange_activities
                        .iter()
                        .map(|activity| {
                            [
                                match activity.kind {
                                    ProfileExchangeActivityKind::Send => 0,
                                    ProfileExchangeActivityKind::Receive => 1,
                                    ProfileExchangeActivityKind::PartnerBusy => 2,
                                },
                                activity.start_cycle,
                                activity.end_cycle,
                            ]
                        })
                        .collect::<Vec<_>>();
                    let activities =
                        *activity_indices
                            .entry(activities.clone())
                            .or_insert_with(|| {
                                let index = activity_sets.len() as u32;
                                activity_sets.push(activities);
                                index
                            });
                    serde_json::json!([
                        step,
                        sample.start_cycle.wrapping_sub(base_cycle),
                        sample.end_cycle.wrapping_sub(sample.start_cycle),
                        activities,
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
                step.exchange_event_cycles,
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
        "activitySets": activity_sets,
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

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    if std::env::var("IPU_LOG_FORMAT").as_deref() == Ok("json") {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .with_target(false)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .init();
    }
}
