use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use ipu_compiler::{
    Allocation, AllocationKind, CompilerOptions, EncoderConfig, EncoderWeights, Phase, Schedule,
    TensorId, Transfer, compile, encoder_graph, encoder_reference,
};
use ipu_driver::{Device, HostBuffer, HostSession, Loader, block_device_interrupt_signals};
use ipu_elf::{LinkOptions, Toolchain, inspect_object, link};
use ipu_exchange::Topology;
use ipu_package::{
    Application, Binding, EntryPoint, HostCall, HostExchange, HostPage, HostSlice, RegionSlice,
    SEGMENT_EXECUTE, SEGMENT_READ, SEGMENT_WRITE, Segment, TileImage,
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
    PackageRuntimeFixture {
        object: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        tiles: u32,
    },
    PackageElfDirectory {
        directory: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 1)]
        external_syncs: u32,
    },
    PackageExchangeFixture {
        object: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 1)]
        pairs: u16,
        #[arg(long, default_value_t = 64)]
        count: u32,
        #[arg(long, default_value_t = 0)]
        sender_base: u16,
        #[arg(long, default_value_t = 274)]
        receiver_base: u16,
        #[arg(long)]
        second_receiver_base: Option<u16>,
        #[arg(long, value_parser = parse_u32)]
        global_sync_route: u32,
        #[arg(long, value_parser = parse_u32)]
        global_sync_packet_address: u32,
        #[arg(long, value_parser = parse_u32)]
        global_sync_release_address: u32,
    },
    PackageParallelSumFixture {
        object: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, value_parser = parse_u32)]
        global_sync_route: u32,
        #[arg(long, value_parser = parse_u32)]
        global_sync_packet_address: u32,
        #[arg(long, value_parser = parse_u32)]
        global_sync_release_address: u32,
        #[arg(long, default_value = "0x50160", value_parser = parse_u32)]
        host_packet_address: u32,
        #[arg(long, default_value = "0x50180", value_parser = parse_u32)]
        host_dummy_address: u32,
        #[arg(long, default_value = "0x40", value_parser = parse_u32)]
        host_output_offset: u32,
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
                external_syncs: 1,
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
        Command::PackageExchangeFixture {
            object,
            output,
            pairs,
            count,
            sender_base,
            receiver_base,
            second_receiver_base,
            global_sync_route,
            global_sync_packet_address,
            global_sync_release_address,
        } => {
            let topology = Topology::c600();
            let endpoint_extent = pairs
                .checked_mul(2)
                .and_then(|extent| extent.checked_sub(1))
                .ok_or_else(|| anyhow::anyhow!("exchange pair count overflow"))?;
            if pairs == 0
                || usize::from(endpoint_extent) > topology.tile_count()
                || usize::from(sender_base) + usize::from(endpoint_extent) > topology.tile_count()
                || usize::from(receiver_base) + usize::from(endpoint_extent) > topology.tile_count()
                || sender_base.abs_diff(receiver_base) < endpoint_extent
                || second_receiver_base.is_some_and(|base| {
                    usize::from(base) + usize::from(endpoint_extent) > topology.tile_count()
                        || base.abs_diff(sender_base) < endpoint_extent
                        || base.abs_diff(receiver_base) < endpoint_extent
                })
            {
                bail!("exchange pair count is out of range");
            }
            if global_sync_route > 0x00ff_ffff {
                bail!("global sync route is out of range");
            }
            let tile_memory = ipu_package::TILE_MEMORY_BASE
                ..ipu_package::TILE_MEMORY_BASE + ipu_driver::TILE_MEMORY_SIZE as u32;
            if global_sync_packet_address & 7 != 4
                || global_sync_release_address & 3 != 0
                || !tile_memory.contains(&global_sync_packet_address)
                || !tile_memory.contains(&global_sync_release_address)
                || global_sync_packet_address + 16 > tile_memory.end
                || global_sync_release_address + 4 > tile_memory.end
                || (global_sync_packet_address..global_sync_packet_address + 16)
                    .contains(&global_sync_release_address)
            {
                bail!("global sync buffer allocation is invalid");
            }
            let bytes = count
                .checked_mul(4)
                .filter(|bytes| *bytes != 0)
                .ok_or_else(|| anyhow::anyhow!("exchange byte count overflow"))?;
            let guard_words = 16u32;
            // Keep worker sync storage, receive staging, executable plans, and
            // outgoing data in separate SRAM regions. Exchange instructions
            // can raise TEXCPT_CONFLICT when plan and payload share an element.
            let destination_segment_address = ipu_exchange::EXCHANGE_WINDOW_BASE + 0x1000;
            let destination_address = destination_segment_address + guard_words * 4;
            let plan_address = ipu_exchange::EXCHANGE_WINDOW_BASE + 0x8000;
            let source_address = plan_address + 0x8000;
            let mut transfers = Vec::new();
            let mut allocations = Vec::new();
            let mut sources = BTreeMap::new();
            let mut destinations = BTreeMap::new();
            for pair in 0..pairs {
                let tensor = TensorId(usize::from(pair));
                let source = sender_base + pair * 2;
                let destination = receiver_base + pair * 2;
                if topology.physical(source)? == 0 || topology.physical(destination)? == 0 {
                    bail!("physical tile 0 is reserved for the global sync coordinator");
                }
                let mut requested_destinations = vec![destination];
                sources.insert(source, pair);
                destinations.insert(destination, pair);
                allocations.push(Allocation {
                    tensor,
                    tile: source,
                    address: source_address,
                    size: bytes,
                    live_from: 0,
                    live_until: 1,
                    kind: AllocationKind::Home,
                });
                if let Some(base) = second_receiver_base {
                    let second_destination = base + pair * 2;
                    if topology.physical(second_destination)? == 0 {
                        bail!("physical tile 0 is reserved for the global sync coordinator");
                    }
                    requested_destinations.push(second_destination);
                    destinations.insert(second_destination, pair);
                }
                for &requested in &requested_destinations {
                    transfers.push(Transfer {
                        source_tile: source,
                        destination_tile: requested,
                        tensor,
                        bytes,
                    });
                    allocations.push(Allocation {
                        tensor,
                        tile: requested,
                        address: destination_address,
                        size: bytes,
                        live_from: 0,
                        live_until: 1,
                        kind: AllocationKind::ExchangeStaging { phase: 0 },
                    });
                }
            }
            let schedule = Schedule {
                layouts: Vec::new(),
                phases: vec![Phase::Exchange { transfers }],
                allocations,
                tile_count: topology.tile_count() as u16,
                peak_sram: BTreeMap::new(),
            };
            let lowered = schedule.lower_exchanges(&topology)?;
            if lowered.len() != 1 || lowered[0].epochs.len() != 1 {
                bail!("matching fixture did not lower to one exchange epoch");
            }
            let image = link(
                &[fs::read(object)?],
                &LinkOptions {
                    image_base: ipu_driver::APPLICATION_LOAD_BASE,
                    entry_symbol: "ipu_stack_exchange_start".into(),
                    externals: HashMap::new(),
                },
            )?;
            let mut app = Application::default();
            let sync_base_offset = image
                .symbols
                .get("ipu_stack_worker_sync_base")
                .copied()
                .ok_or_else(|| anyhow::anyhow!("exchange runtime has no worker sync symbol"))?
                .checked_sub(image.base)
                .ok_or_else(|| anyhow::anyhow!("worker sync symbol precedes image"))?
                as usize;
            let symbol_offset = |name: &str| -> Result<usize> {
                Ok(image
                    .symbols
                    .get(name)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("exchange runtime has no {name} symbol"))?
                    .checked_sub(image.base)
                    .ok_or_else(|| anyhow::anyhow!("{name} symbol precedes image"))?
                    as usize)
            };
            let execute_offset = symbol_offset("ipu_stack_execute")?;
            let active_base_offset = symbol_offset("ipu_stack_active_base")?;
            let plan_address_offset = symbol_offset("ipu_stack_plan_address")?;
            let nonparticipant_redirect_offset =
                symbol_offset("ipu_stack_nonparticipant_redirect")?;
            let coordinator_nonparticipant_redirect_offset =
                symbol_offset("ipu_stack_coordinator_nonparticipant_redirect")?;
            let pre_sync_offset = symbol_offset("ipu_stack_pre_exchange_sync")?;
            let nonmaster_redirect_offset = symbol_offset("ipu_stack_nonmaster_redirect")?;
            let global_sync_endpoint_offset = symbol_offset("ipu_stack_global_sync_endpoint")?;
            let global_sync_send0_offset = symbol_offset("ipu_stack_global_sync_send0")?;
            let global_sync_send1_offset = symbol_offset("ipu_stack_global_sync_send1")?;
            let global_sync_release_offset = symbol_offset("ipu_stack_global_sync_release")?;
            let mut code_blobs = BTreeMap::new();
            for logical in 0..topology.tile_count() as u16 {
                let mut code = image.bytes.clone();
                let physical = u32::from(topology.physical(logical)?);
                let participant =
                    sources.contains_key(&logical) || destinations.contains_key(&logical);
                code[sync_base_offset..sync_base_offset + 4]
                    .copy_from_slice(&(0x1900_0000 | (u32::from(logical) * 8)).to_le_bytes());
                code[plan_address_offset..plan_address_offset + 4]
                    .copy_from_slice(&(0x1900_0000 | plan_address).to_le_bytes());
                if participant {
                    let address = if sources.contains_key(&logical) {
                        source_address
                    } else {
                        destination_address
                    };
                    code[active_base_offset..active_base_offset + 4]
                        .copy_from_slice(&(0x1990_0000 | address).to_le_bytes());
                }
                if !participant {
                    let redirect_offset = if physical == 0 {
                        coordinator_nonparticipant_redirect_offset
                    } else {
                        nonparticipant_redirect_offset
                    };
                    let redirect = u32::from_le_bytes(
                        code[redirect_offset..redirect_offset + 4]
                            .try_into()
                            .expect("instruction word"),
                    );
                    code[execute_offset..execute_offset + 4]
                        .copy_from_slice(&redirect.to_le_bytes());
                }
                if logical == 0 {
                    for (offset, address) in [
                        (global_sync_send0_offset, global_sync_packet_address),
                        (global_sync_send1_offset, global_sync_packet_address + 8),
                    ] {
                        let instruction = ipu_exchange::encode_send(1, 3, address >> 2)?;
                        code[offset..offset + 4].copy_from_slice(&instruction.to_le_bytes());
                    }
                    let release =
                        ipu_exchange::encode_send(0, 3, global_sync_release_address >> 2)?;
                    code[global_sync_release_offset..global_sync_release_offset + 4]
                        .copy_from_slice(&release.to_le_bytes());
                    let mut instruction = u32::from_le_bytes(
                        code[global_sync_endpoint_offset..global_sync_endpoint_offset + 4]
                            .try_into()
                            .expect("instruction word"),
                    );
                    instruction = (instruction & !0x1f_ffff) | (0x600 + physical);
                    code[global_sync_endpoint_offset..global_sync_endpoint_offset + 4]
                        .copy_from_slice(&instruction.to_le_bytes());
                } else {
                    let redirect = u32::from_le_bytes(
                        code[nonmaster_redirect_offset..nonmaster_redirect_offset + 4]
                            .try_into()
                            .expect("instruction word"),
                    );
                    code[pre_sync_offset..pre_sync_offset + 4]
                        .copy_from_slice(&redirect.to_le_bytes());
                }
                code_blobs.insert(logical, app.add_blob(code));
            }
            let destination_words = count + guard_words * 2;
            let destination_initial = words_to_bytes(
                &(0..destination_words)
                    .map(|_| 0x6bad_f00d)
                    .collect::<Vec<_>>(),
            );
            let destination_blob = app.add_blob(destination_initial);
            for logical in 0..topology.tile_count() as u16 {
                let physical = u32::from(topology.physical(logical)?);
                let mut segments = vec![Segment {
                    address: image.base,
                    memory_size: image.bytes.len() as u32,
                    blob: code_blobs[&logical],
                    blob_offset: 0,
                    file_size: image.bytes.len() as u32,
                    flags: SEGMENT_READ | SEGMENT_EXECUTE,
                }];
                if sources.contains_key(&logical) || destinations.contains_key(&logical) {
                    let row = lowered[0].epochs[0].row_for(logical);
                    let plan_blob = app.add_blob(words_to_bytes(&row));
                    segments.push(Segment {
                        address: plan_address,
                        memory_size: (ipu_exchange::PLAN_WORDS * 4) as u32,
                        blob: plan_blob,
                        blob_offset: 0,
                        file_size: (ipu_exchange::PLAN_WORDS * 4) as u32,
                        flags: SEGMENT_READ | SEGMENT_WRITE | SEGMENT_EXECUTE,
                    });
                }
                if logical == 0 {
                    let packet_blob = app.add_blob(words_to_bytes(&[
                        1,
                        0,
                        // The low route identifier belongs to the exchange
                        // allocation represented by the device configuration.
                        0xcc00_0000 | global_sync_route,
                        0x0000_4001,
                    ]));
                    segments.push(Segment {
                        address: global_sync_packet_address,
                        memory_size: 16,
                        blob: packet_blob,
                        blob_offset: 0,
                        file_size: 16,
                        flags: SEGMENT_READ,
                    });
                    let release_blob = app.add_blob(words_to_bytes(&[0]));
                    segments.push(Segment {
                        address: global_sync_release_address,
                        memory_size: 4,
                        blob: release_blob,
                        blob_offset: 0,
                        file_size: 4,
                        flags: SEGMENT_READ,
                    });
                }
                if let Some(&pair) = sources.get(&logical) {
                    let source = (0..count)
                        .map(|index| exchange_fixture_value(pair, index))
                        .collect::<Vec<_>>();
                    let blob = app.add_blob(words_to_bytes(&source));
                    segments.push(Segment {
                        address: source_address,
                        memory_size: bytes,
                        blob,
                        blob_offset: 0,
                        file_size: bytes,
                        flags: SEGMENT_READ,
                    });
                } else if destinations.contains_key(&logical) {
                    segments.push(Segment {
                        address: destination_segment_address,
                        memory_size: destination_words * 4,
                        blob: destination_blob,
                        blob_offset: 0,
                        file_size: destination_words * 4,
                        flags: SEGMENT_READ | SEGMENT_WRITE,
                    });
                }
                app.tiles.push(TileImage {
                    physical_tile: physical,
                    entry_point: image.base,
                    command_address: plan_address,
                    diagnostic_address: if destinations.contains_key(&logical) {
                        destination_address
                    } else {
                        0
                    },
                    segments,
                });
            }
            app.tiles.sort_by_key(|tile| tile.physical_tile);
            app.entry_points.push(EntryPoint {
                name: "exchange-matching".into(),
                command: 0,
                external_syncs: 3,
            });
            app.write(fs::File::create(&output)?)?;
            println!(
                "pairs={} count={} launches={} sender0Logical={} sender0Physical={} receiver0Logical={} receiver0Physical={} expectedFirst=0x{:08x} expectedLast=0x{:08x} output={}",
                pairs,
                count,
                lowered[0].cost.launches,
                sender_base,
                topology.physical(sender_base)?,
                receiver_base,
                topology.physical(receiver_base)?,
                exchange_fixture_value(0, 0),
                exchange_fixture_value(0, count - 1),
                output.display()
            );
        }
        Command::PackageParallelSumFixture {
            object,
            output,
            global_sync_route,
            global_sync_packet_address,
            global_sync_release_address,
            host_packet_address,
            host_dummy_address,
            host_output_offset,
        } => {
            let topology = Topology::c600();
            if global_sync_route > 0x00ff_ffff {
                bail!("global sync route is out of range");
            }
            let plan_base = ipu_exchange::EXCHANGE_WINDOW_BASE + 0x8000;
            let plan_stride = 40u32;
            let command_address = plan_base + 0x2000;
            let staging_address = ipu_exchange::EXCHANGE_WINDOW_BASE + 0x1040;
            let accumulator_address = plan_base + 0x8000;

            let mut allocations = Vec::new();
            for tile in 1..topology.tile_count() as u16 {
                allocations.push(Allocation {
                    tensor: TensorId(usize::from(tile)),
                    tile,
                    address: accumulator_address,
                    size: 4,
                    live_from: 0,
                    live_until: usize::MAX,
                    kind: AllocationKind::Home,
                });
            }
            let mut phases = Vec::new();
            let mut active: Vec<u16> = (1..topology.tile_count() as u16).collect();
            while active.len() > 1 {
                let phase_index = phases.len();
                let mut transfers = Vec::new();
                let mut next = Vec::with_capacity(active.len().div_ceil(2));
                for pair in active.chunks(2) {
                    let receiver = pair[0];
                    next.push(receiver);
                    if let Some(&sender) = pair.get(1) {
                        let tensor = TensorId(usize::from(sender));
                        transfers.push(Transfer {
                            source_tile: sender,
                            destination_tile: receiver,
                            tensor,
                            bytes: 4,
                        });
                        allocations.push(Allocation {
                            tensor,
                            tile: receiver,
                            address: staging_address,
                            size: 4,
                            live_from: phase_index,
                            live_until: phase_index + 1,
                            kind: AllocationKind::ExchangeStaging { phase: phase_index },
                        });
                    }
                }
                phases.push(Phase::Exchange { transfers });
                active = next;
            }
            let root = active[0];
            let schedule = Schedule {
                layouts: Vec::new(),
                phases,
                allocations,
                tile_count: topology.tile_count() as u16,
                peak_sram: BTreeMap::new(),
            };
            let lowered = schedule.lower_exchanges(&topology)?;
            let launches: Vec<_> = lowered.iter().flat_map(|phase| &phase.epochs).collect();
            let host_plan_address = plan_base + launches.len() as u32 * plan_stride;
            let host_program = ipu_exchange::assemble_tile_to_host_program(
                topology.physical(root)?,
                accumulator_address,
                host_output_offset,
                64,
                host_packet_address,
                host_dummy_address,
            )?;
            let host_xreq_packet_address = host_packet_address
                .checked_add(16)
                .ok_or_else(|| anyhow::anyhow!("host packet address overflow"))?;
            let host_xreq_program =
                ipu_exchange::assemble_tile_to_host_xreq_program(host_xreq_packet_address)?;
            let host_command_program = ipu_exchange::assemble_host_command_read_program(
                host_packet_address,
                host_dummy_address,
                0x1000,
            )?;
            let host_command_plan_address =
                host_plan_address + ((host_program.instructions.len() as u32 * 4 + 7) & !7);
            let host_xreq_plan_address = host_command_plan_address
                + ((host_command_program.instructions.len() as u32 * 4 + 7) & !7);
            let image = link(
                &[fs::read(object)?],
                &LinkOptions {
                    image_base: ipu_driver::APPLICATION_LOAD_BASE,
                    entry_symbol: "ipu_stack_exchange_loop_start".into(),
                    externals: HashMap::new(),
                },
            )?;
            let symbol_offset = |name: &str| -> Result<usize> {
                Ok(image
                    .symbols
                    .get(name)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("exchange loop runtime has no {name} symbol"))?
                    .checked_sub(image.base)
                    .ok_or_else(|| anyhow::anyhow!("{name} symbol precedes image"))?
                    as usize)
            };
            let worker_sync_offset = symbol_offset("ipu_stack_loop_worker_sync_base")?;
            let command_offset = symbol_offset("ipu_stack_command_table_address")?;
            let pre_sync_offset = symbol_offset("ipu_stack_pre_sync_dispatch")?;
            let nonmaster_redirect_offset = symbol_offset("ipu_stack_nonmaster_pre_sync_redirect")?;
            let endpoint_offset = symbol_offset("ipu_stack_loop_global_sync_endpoint")?;
            let send0_offset = symbol_offset("ipu_stack_loop_global_sync_send0")?;
            let send1_offset = symbol_offset("ipu_stack_loop_global_sync_send1")?;
            let release_offset = symbol_offset("ipu_stack_loop_global_sync_release")?;

            let mut app = Application::default();
            let mut code_blobs = BTreeMap::new();
            for logical in 0..topology.tile_count() as u16 {
                let physical = u32::from(topology.physical(logical)?);
                let mut code = image.bytes.clone();
                code[worker_sync_offset..worker_sync_offset + 4]
                    .copy_from_slice(&(0x1900_0000 | (u32::from(logical) * 8)).to_le_bytes());
                let mut command_instruction = u32::from_le_bytes(
                    code[command_offset..command_offset + 4]
                        .try_into()
                        .expect("instruction word"),
                );
                command_instruction = (command_instruction & !0x1f_ffff) | command_address;
                code[command_offset..command_offset + 4]
                    .copy_from_slice(&command_instruction.to_le_bytes());
                if physical == 0 {
                    for (offset, address) in [
                        (send0_offset, global_sync_packet_address),
                        (send1_offset, global_sync_packet_address + 8),
                    ] {
                        code[offset..offset + 4].copy_from_slice(
                            &ipu_exchange::encode_send(1, 3, address >> 2)?.to_le_bytes(),
                        );
                    }
                    code[release_offset..release_offset + 4].copy_from_slice(
                        &ipu_exchange::encode_send(0, 3, global_sync_release_address >> 2)?
                            .to_le_bytes(),
                    );
                    let mut endpoint = u32::from_le_bytes(
                        code[endpoint_offset..endpoint_offset + 4]
                            .try_into()
                            .expect("instruction word"),
                    );
                    endpoint = (endpoint & !0x1f_ffff) | (0x600 + physical);
                    code[endpoint_offset..endpoint_offset + 4]
                        .copy_from_slice(&endpoint.to_le_bytes());
                } else {
                    let redirect =
                        code[nonmaster_redirect_offset..nonmaster_redirect_offset + 4].to_vec();
                    code[pre_sync_offset..pre_sync_offset + 4].copy_from_slice(&redirect);
                }
                code_blobs.insert(logical, app.add_blob(code));
            }

            let mut receiver_tiles = std::collections::BTreeSet::new();
            for phase in &lowered {
                for epoch in &phase.epochs {
                    for group in &epoch.groups {
                        receiver_tiles.extend(group.destination_tiles.iter().copied());
                    }
                }
            }
            for logical in 0..topology.tile_count() as u16 {
                let physical = u32::from(topology.physical(logical)?);
                let mut segments = vec![Segment {
                    address: image.base,
                    memory_size: image.bytes.len() as u32,
                    blob: code_blobs[&logical],
                    blob_offset: 0,
                    file_size: image.bytes.len() as u32,
                    flags: SEGMENT_READ | SEGMENT_EXECUTE,
                }];
                let mut commands = Vec::with_capacity((launches.len() + 4) * 16);
                for (launch_index, epoch) in launches.iter().enumerate() {
                    let plan_address = plan_base + launch_index as u32 * plan_stride;
                    let mut role = if physical == 0 { 1 } else { 2 };
                    let mut accumulator = 0;
                    let mut staging = 0;
                    if let Some(row) = epoch.tile_rows.get(&logical) {
                        role = 3;
                        accumulator = accumulator_address;
                        if epoch
                            .groups
                            .iter()
                            .any(|group| group.destination_tiles.contains(&logical))
                        {
                            role = 4;
                            accumulator = accumulator_address;
                            staging = staging_address;
                        }
                        let blob = app.add_blob(words_to_bytes(row));
                        segments.push(Segment {
                            address: plan_address,
                            memory_size: (ipu_exchange::PLAN_WORDS * 4) as u32,
                            blob,
                            blob_offset: 0,
                            file_size: (ipu_exchange::PLAN_WORDS * 4) as u32,
                            flags: SEGMENT_READ | SEGMENT_WRITE | SEGMENT_EXECUTE,
                        });
                    }
                    commands.extend_from_slice(&words_to_bytes(&[
                        role,
                        plan_address,
                        accumulator,
                        staging,
                    ]));
                }
                commands.extend_from_slice(&words_to_bytes(&[
                    if physical == 0 { 8 } else { 2 },
                    host_command_plan_address,
                    0,
                    0,
                ]));
                let host_role = if physical == 0 {
                    7
                } else if logical == root {
                    5
                } else {
                    2
                };
                commands.extend_from_slice(&words_to_bytes(&[
                    host_role,
                    if physical == 0 {
                        host_xreq_plan_address
                    } else {
                        host_plan_address
                    },
                    accumulator_address,
                    0,
                ]));
                commands.extend_from_slice(&words_to_bytes(&[1, 0, 0, 0]));
                commands.extend_from_slice(&words_to_bytes(&[0, 0, 0, 0]));
                let command_blob = app.add_blob(commands.clone());
                segments.push(Segment {
                    address: command_address,
                    memory_size: commands.len() as u32,
                    blob: command_blob,
                    blob_offset: 0,
                    file_size: commands.len() as u32,
                    flags: SEGMENT_READ,
                });
                if logical != 0 {
                    let initial = if logical == root {
                        u32::from(logical) + 2
                    } else {
                        u32::from(logical) + 1
                    };
                    let blob = app.add_blob(words_to_bytes(&[initial]));
                    segments.push(Segment {
                        address: accumulator_address,
                        memory_size: 64,
                        blob,
                        blob_offset: 0,
                        file_size: 4,
                        flags: SEGMENT_READ | SEGMENT_WRITE,
                    });
                }
                if logical == root {
                    let plan = app.add_blob(words_to_bytes(&host_program.instructions));
                    segments.push(Segment {
                        address: host_plan_address,
                        memory_size: (host_program.instructions.len() * 4) as u32,
                        blob: plan,
                        blob_offset: 0,
                        file_size: (host_program.instructions.len() * 4) as u32,
                        flags: SEGMENT_READ | SEGMENT_EXECUTE,
                    });
                    let packets = app.add_blob(words_to_bytes(&host_program.packet_words));
                    segments.push(Segment {
                        address: host_packet_address,
                        memory_size: 16,
                        blob: packets,
                        blob_offset: 0,
                        file_size: 16,
                        flags: SEGMENT_READ,
                    });
                }
                if physical == 0 {
                    let command_plan =
                        app.add_blob(words_to_bytes(&host_command_program.instructions));
                    segments.push(Segment {
                        address: host_command_plan_address,
                        memory_size: (host_command_program.instructions.len() * 4) as u32,
                        blob: command_plan,
                        blob_offset: 0,
                        file_size: (host_command_program.instructions.len() * 4) as u32,
                        flags: SEGMENT_READ | SEGMENT_EXECUTE,
                    });
                    let command_packets =
                        app.add_blob(words_to_bytes(&host_command_program.packet_words));
                    segments.push(Segment {
                        address: host_packet_address,
                        memory_size: 16,
                        blob: command_packets,
                        blob_offset: 0,
                        file_size: 16,
                        flags: SEGMENT_READ,
                    });
                    let plan = app.add_blob(words_to_bytes(&host_xreq_program.instructions));
                    segments.push(Segment {
                        address: host_xreq_plan_address,
                        memory_size: (host_xreq_program.instructions.len() * 4) as u32,
                        blob: plan,
                        blob_offset: 0,
                        file_size: (host_xreq_program.instructions.len() * 4) as u32,
                        flags: SEGMENT_READ | SEGMENT_EXECUTE,
                    });
                    let packet = app.add_blob(words_to_bytes(&host_xreq_program.packet_words[..2]));
                    segments.push(Segment {
                        address: host_xreq_packet_address,
                        memory_size: 8,
                        blob: packet,
                        blob_offset: 0,
                        file_size: 8,
                        flags: SEGMENT_READ,
                    });
                }
                if receiver_tiles.contains(&logical) {
                    let blob = app.add_blob(words_to_bytes(&[0]));
                    segments.push(Segment {
                        address: staging_address,
                        memory_size: 4,
                        blob,
                        blob_offset: 0,
                        file_size: 4,
                        flags: SEGMENT_READ | SEGMENT_WRITE,
                    });
                }
                if physical == 0 {
                    let packet_blob = app.add_blob(words_to_bytes(&[
                        1,
                        0,
                        0xcc00_0000 | global_sync_route,
                        0x0000_4001,
                    ]));
                    segments.push(Segment {
                        address: global_sync_packet_address,
                        memory_size: 16,
                        blob: packet_blob,
                        blob_offset: 0,
                        file_size: 16,
                        flags: SEGMENT_READ,
                    });
                    let release_blob = app.add_blob(words_to_bytes(&[0]));
                    segments.push(Segment {
                        address: global_sync_release_address,
                        memory_size: 4,
                        blob: release_blob,
                        blob_offset: 0,
                        file_size: 4,
                        flags: SEGMENT_READ,
                    });
                }
                app.tiles.push(TileImage {
                    physical_tile: physical,
                    entry_point: image.base,
                    command_address,
                    diagnostic_address: if logical == root {
                        accumulator_address
                    } else {
                        0
                    },
                    segments,
                });
            }
            app.tiles.sort_by_key(|tile| tile.physical_tile);
            // D2H uses global entry and coordinator XREQ phases. A following
            // all-tile launch contributes global entry and external completion
            // only after the source's sync 0 has retired.
            let external_syncs = launches.len() as u32 * 2 + 6;
            app.entry_points.push(EntryPoint {
                name: "parallel-sum".into(),
                command: 0,
                external_syncs,
            });
            app.outputs.push(Binding {
                name: "sum".into(),
                dtype: "u32".into(),
                shape: vec![1],
                slices: vec![RegionSlice {
                    tile: u32::from(topology.physical(root)?),
                    tile_address: accumulator_address,
                    file_offset: 0,
                    size: 4,
                }],
            });
            app.write(fs::File::create(&output)?)?;
            let expected = (topology.tile_count() as u32)
                .checked_mul(topology.tile_count() as u32 + 1)
                .expect("C600 sum fits u32")
                / 2;
            println!(
                "tiles={} rootLogical={} rootPhysical={} rounds={} launches={} expected={} resultAddress=0x{:x} output={}",
                topology.tile_count(),
                root,
                topology.physical(root)?,
                lowered.len(),
                launches.len(),
                expected,
                accumulator_address,
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
            Self::PackageRuntimeFixture { .. } => "package-runtime-fixture",
            Self::PackageElfDirectory { .. } => "package-elf-directory",
            Self::PackageExchangeFixture { .. } => "package-exchange-fixture",
            Self::PackageParallelSumFixture { .. } => "package-parallel-sum-fixture",
            Self::DeviceProbe { .. } => "device-probe",
            Self::Load { .. } => "load",
            Self::RunOutput { .. } => "run-output",
            Self::HostRun { .. } => "host-run",
        }
    }
}

fn format_plan_row(row: &[u32]) -> String {
    row.iter().map(|word| format!(" 0x{word:08x}")).collect()
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn exchange_fixture_value(pair: u16, index: u32) -> u32 {
    0x1357_9bdf ^ u32::from(pair).wrapping_mul(0x7f4a_7c15) ^ index.wrapping_mul(0x9e37_79b9)
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
