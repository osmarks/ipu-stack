use ipu_compiler::{MemoryConstraint, MemoryPlacement, Schedule, find_free_region};
use ipu_driver::{Device, DriverError, HostSession, Loader, block_device_interrupt_signals};
use ipu_elf::{LinkOptions, link};
use ipu_exchange::Topology;
use ipu_package::{
    Application, Binding, EntryPoint, HostCall, HostExchange, HostPage, RegionSlice,
    SEGMENT_EXECUTE, SEGMENT_READ, SEGMENT_WRITE, Segment, TileImage,
};
use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;

mod static_codegen;

const PLAN_BASE: u32 = ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES;
const GLOBAL_SYNC_PACKET_ORIGIN: u16 = 0;
const GLOBAL_SYNC_HOST_PAGE: u32 = 1;
const GLOBAL_SYNC_HOST_OFFSET: u32 = GLOBAL_SYNC_HOST_PAGE * ipu_exchange::HOST_PAGE_BYTES;

pub fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false);
    if std::env::var("IPU_LOG_FORMAT").as_deref() == Ok("json") {
        builder.json().try_init().ok();
    } else {
        builder.try_init().ok();
    }
}

#[derive(Clone, Debug)]
pub struct InitialBuffer {
    pub tile: u16,
    pub address: u32,
    pub words: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct ExecutableGraph {
    pub schedule: Schedule,
    pub initial_buffers: Vec<InitialBuffer>,
    pub outputs: Vec<Binding>,
    pub host_inputs: Vec<Binding>,
    pub host_outputs: Vec<Binding>,
}

#[derive(Clone, Debug)]
pub struct DiagnosticResults {
    pub bindings: BTreeMap<String, Vec<u32>>,
}

fn packed_addresses(base: u32, sizes: &[usize], alignment: u32) -> Result<(Vec<u32>, u32)> {
    let mut addresses = Vec::with_capacity(sizes.len());
    let mut cursor = base;
    for &size in sizes {
        cursor = align_up(cursor, alignment);
        addresses.push(cursor);
        cursor = cursor
            .checked_add(u32::try_from(size)?)
            .ok_or("packed runtime region overflow")?;
    }
    Ok((addresses, cursor))
}

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub fn package_graph(graph: &ExecutableGraph, objects: &[Vec<u8>]) -> Result<Application> {
    if !graph.host_inputs.is_empty() || !graph.host_outputs.is_empty() {
        return Err("static host exchange is not implemented".into());
    }

    let topology = Topology::c600();
    if usize::from(graph.schedule.tile_count) != topology.tile_count() {
        return Err("the direct C600 runtime requires a schedule for every discovered tile".into());
    }
    let programs = graph.schedule.lower_tile_programs(&topology)?;
    let exchange_count = programs
        .first()
        .map(|program| {
            program
                .steps
                .iter()
                .filter(|step| matches!(step, ipu_compiler::LoweredTileStep::Exchange { .. }))
                .count()
        })
        .unwrap_or(0);
    if programs.iter().any(|program| {
        program
            .steps
            .iter()
            .filter(|step| matches!(step, ipu_compiler::LoweredTileStep::Exchange { .. }))
            .count()
            != exchange_count
    }) {
        return Err("per-tile programs disagree on exchange launch count".into());
    }
    let startup_exchange = matches!(
        graph.schedule.phases.first(),
        Some(ipu_compiler::Phase::Exchange { .. })
    );
    let barrier_count = exchange_count - usize::from(startup_exchange && exchange_count != 0);

    let plan_sizes = (0..exchange_count)
        .map(|index| {
            programs
                .iter()
                .filter_map(|program| {
                    program
                        .steps
                        .iter()
                        .filter_map(|step| match step {
                            ipu_compiler::LoweredTileStep::Exchange { row, .. } => Some(row.len()),
                            _ => None,
                        })
                        .nth(index)
                })
                .max()
                .unwrap_or(0)
                .checked_mul(4)
                .ok_or("exchange plan size overflow")
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let (plan_addresses, plan_end) = packed_addresses(PLAN_BASE, &plan_sizes, 8)?;
    let global_sync_origin_logical = (0..graph.schedule.tile_count)
        .find(|logical| topology.physical(*logical) == Ok(GLOBAL_SYNC_PACKET_ORIGIN))
        .ok_or("schedule does not contain the global sync packet origin")?;
    let global_sync_packet_address = find_free_region(
        &graph.schedule.allocations,
        global_sync_origin_logical,
        64,
        0,
        usize::MAX,
        MemoryConstraint {
            base: ipu_exchange::EXCHANGE_WINDOW_BASE,
            limit: ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::HOST_TO_TILE_WINDOW_BYTES,
            alignment: 32,
            placement: MemoryPlacement::High,
        },
    )?;
    let global_sync_release_address = global_sync_packet_address + 32;
    let completion_address = align_up(plan_end, 64);
    let runtime_end = completion_address
        .checked_add(4)
        .ok_or("static runtime address overflow")?;

    let mut retained_symbols = vec![
        static_codegen::WORKER_BARRIER.into(),
        static_codegen::GLOBAL_BARRIER_MASTER.into(),
        static_codegen::GLOBAL_BARRIER_FOLLOWER.into(),
        static_codegen::COMPLETE.into(),
    ];
    retained_symbols.extend(programs.iter().flat_map(|program| {
        program.steps.iter().filter_map(|step| match step {
            ipu_compiler::LoweredTileStep::Compute(command) => {
                Some(format!("ipu_stack_{}", command.specialization.operation))
            }
            _ => None,
        })
    }));
    retained_symbols.sort();
    retained_symbols.dedup();
    let image = link(
        objects,
        &LinkOptions {
            image_base: ipu_driver::APPLICATION_LOAD_BASE,
            entry_symbol: "ipu_stack_static_start".into(),
            retained_symbols,
            externals: HashMap::new(),
        },
    )?;
    let program_base = align_up(
        image
            .base
            .checked_add(u32::try_from(image.bytes.len())?)
            .ok_or("linked image address overflow")?,
        8,
    );
    let generated = programs
        .iter()
        .map(|program| {
            let barrier = if topology.physical(program.tile)? == GLOBAL_SYNC_PACKET_ORIGIN {
                static_codegen::GLOBAL_BARRIER_MASTER
            } else {
                static_codegen::GLOBAL_BARRIER_FOLLOWER
            };
            static_codegen::emit(
                program,
                program_base,
                &image.symbols,
                &plan_addresses,
                image.symbols[barrier],
                startup_exchange,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let program_end = generated.iter().try_fold(program_base, |end, code| {
        let code_end = program_base
            .checked_add(u32::try_from(code.len())?)
            .ok_or("generated tile program address overflow")?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(end.max(code_end))
    })?;
    if program_end > ipu_exchange::EXCHANGE_WINDOW_BASE {
        return Err("static tile program exceeds the application code region".into());
    }
    for allocation in &graph.schedule.allocations {
        let end = allocation
            .address
            .checked_add(allocation.size)
            .ok_or("allocation address overflow")?;
        if ranges_overlap(image.base, program_end, allocation.address, end)
            || ranges_overlap(PLAN_BASE, runtime_end, allocation.address, end)
        {
            return Err(format!(
                "static runtime region overlaps tensor {} on tile {}",
                allocation.tensor.0, allocation.tile
            )
            .into());
        }
    }

    let symbol_offset = |name: &str| -> Result<usize> {
        Ok(image
            .symbols
            .get(name)
            .copied()
            .ok_or_else(|| format!("static runtime has no {name} symbol"))?
            .checked_sub(image.base)
            .ok_or_else(|| format!("{name} precedes the linked image"))? as usize)
    };
    let program_offset = symbol_offset("ipu_stack_static_program_address")?;
    let worker_sync_offset = symbol_offset("ipu_stack_static_worker_sync_base")?;
    let completion_offset = symbol_offset("ipu_stack_static_completion_address")?;
    let global_sync_send0_offset = symbol_offset("ipu_stack_static_global_barrier_send0")?;
    let global_sync_send1_offset = symbol_offset("ipu_stack_static_global_barrier_send1")?;
    let global_sync_master_release_offset =
        symbol_offset("ipu_stack_static_global_barrier_master_release_plan")?;
    let global_sync_follower_release_offset =
        symbol_offset("ipu_stack_static_global_barrier_follower_release_plan")?;
    let global_sync_header = ipu_exchange::host_to_tile_packet(
        GLOBAL_SYNC_PACKET_ORIGIN,
        global_sync_release_address,
        GLOBAL_SYNC_HOST_OFFSET,
        4,
    )?;
    let global_sync_receivers = (0..graph.schedule.tile_count)
        .filter(|logical| *logical != global_sync_origin_logical)
        .collect::<Vec<_>>();
    let mut global_sync_release =
        topology.multicast(global_sync_origin_logical, &global_sync_receivers, 1, 0)?;
    ipu_exchange::patch_sender_address(
        &mut global_sync_release.sender,
        global_sync_release_address,
    )?;
    let mut global_sync_receiver_rows = BTreeMap::new();
    for (logical, row) in global_sync_receivers
        .iter()
        .copied()
        .zip(&mut global_sync_release.receivers)
    {
        ipu_exchange::patch_multicast_receiver_address(row, global_sync_release_address)?;
        global_sync_receiver_rows.insert(logical, *row);
    }
    let initial: HashMap<_, _> = graph
        .initial_buffers
        .iter()
        .map(|buffer| ((buffer.tile, buffer.address), words_to_bytes(&buffer.words)))
        .collect();

    let completion_physical_tile = graph
        .outputs
        .iter()
        .flat_map(|binding| &binding.slices)
        .map(|slice| slice.tile)
        .next()
        .or_else(|| {
            programs
                .first()
                .and_then(|program| topology.physical(program.tile).ok())
                .map(u32::from)
        })
        .ok_or("static graph has no tile for diagnostic completion")?;
    let mut app = Application::default();
    for (program, generated_code) in programs.iter().zip(generated) {
        let logical = program.tile;
        let physical = u32::from(topology.physical(logical)?);
        let mut support_code = image.bytes.clone();
        patch_setzi_immediate(&mut support_code, program_offset, program_base)?;
        patch_setzi_immediate(
            &mut support_code,
            worker_sync_offset,
            u32::from(logical) * 8,
        )?;
        patch_setzi_immediate(&mut support_code, completion_offset, completion_address)?;
        if physical == u32::from(GLOBAL_SYNC_PACKET_ORIGIN) {
            patch_instruction(
                &mut support_code,
                global_sync_send0_offset,
                ipu_exchange::encode_send(1, 3, global_sync_packet_address >> 2)?,
            )?;
            patch_instruction(
                &mut support_code,
                global_sync_send1_offset,
                ipu_exchange::encode_send(1, 3, (global_sync_packet_address + 8) >> 2)?,
            )?;
            patch_plan_body(
                &mut support_code,
                global_sync_master_release_offset,
                &global_sync_release.sender,
                3,
            )?;
        } else {
            patch_plan_body(
                &mut support_code,
                global_sync_follower_release_offset,
                &global_sync_receiver_rows[&logical],
                4,
            )?;
        }

        let mut segments = Vec::new();
        let support_blob = app.add_blob(support_code);
        segments.push(Segment {
            address: image.base,
            memory_size: image.bytes.len() as u32,
            blob: support_blob,
            blob_offset: 0,
            file_size: image.bytes.len() as u32,
            flags: SEGMENT_READ | SEGMENT_EXECUTE,
        });
        let generated_size = u32::try_from(generated_code.len())?;
        let generated_blob = app.add_blob(generated_code);
        segments.push(Segment {
            address: program_base,
            memory_size: generated_size,
            blob: generated_blob,
            blob_offset: 0,
            file_size: generated_size,
            flags: SEGMENT_READ | SEGMENT_EXECUTE,
        });

        let mut exchange_index = 0usize;
        for step in &program.steps {
            if let ipu_compiler::LoweredTileStep::Exchange { row, .. } = step {
                let bytes = words_to_bytes(row);
                let size = u32::try_from(bytes.len())?;
                let blob = app.add_blob(bytes);
                segments.push(Segment {
                    address: plan_addresses[exchange_index],
                    memory_size: size,
                    blob,
                    blob_offset: 0,
                    file_size: size,
                    flags: SEGMENT_READ | SEGMENT_EXECUTE,
                });
                exchange_index += 1;
            }
        }
        if physical == u32::from(GLOBAL_SYNC_PACKET_ORIGIN) {
            let packet = app.add_blob(words_to_bytes(&[
                1,
                0,
                global_sync_header.word0,
                global_sync_header.word1,
            ]));
            segments.push(Segment {
                address: global_sync_packet_address,
                memory_size: 16,
                blob: packet,
                blob_offset: 0,
                file_size: 16,
                flags: SEGMENT_READ,
            });
            let release = app.add_blob(vec![0; 4]);
            segments.push(Segment {
                address: global_sync_release_address,
                memory_size: 4,
                blob: release,
                blob_offset: 0,
                file_size: 4,
                flags: SEGMENT_READ | SEGMENT_WRITE,
            });
        }
        append_initial_segments(&mut app, &mut segments, graph, &initial, logical)?;
        app.tiles.push(TileImage {
            physical_tile: physical,
            entry_point: image.base,
            command_address: 0,
            diagnostic_address: completion_address,
            segments,
        });
    }
    app.tiles.sort_by_key(|tile| tile.physical_tile);
    app.outputs = graph.outputs.clone();
    app.outputs.push(Binding {
        name: "runtime-completion".into(),
        dtype: "u32".into(),
        shape: vec![1],
        slices: vec![RegionSlice {
            tile: completion_physical_tile,
            tile_address: completion_address,
            file_offset: 0,
            size: 4,
        }],
    });
    app.entry_points.push(EntryPoint {
        name: "graph".into(),
        command: 0,
        external_syncs: u32::try_from(barrier_count)?
            .checked_mul(2)
            .and_then(|syncs| syncs.checked_add(1))
            .ok_or("external sync count overflow")?,
    });
    if barrier_count != 0 {
        app.host_exchange = HostExchange {
            pages: vec![HostPage {
                index: GLOBAL_SYNC_HOST_PAGE,
                size: u64::from(ipu_exchange::HOST_PAGE_BYTES),
            }],
            attach_order: vec![GLOBAL_SYNC_HOST_PAGE],
            ..HostExchange::default()
        };
    }
    app.validate()?;
    info!(
        tiles = app.tiles.len(),
        exchange_launches = exchange_count,
        max_program_bytes = program_end - program_base,
        "packaged static executable graph"
    );
    Ok(app)
}

fn append_initial_segments(
    app: &mut Application,
    segments: &mut Vec<Segment>,
    graph: &ExecutableGraph,
    initial: &HashMap<(u16, u32), Vec<u8>>,
    logical: u16,
) -> Result<()> {
    let mut allocations = BTreeMap::<(u32, u32), Vec<u8>>::new();
    for allocation in graph
        .schedule
        .allocations
        .iter()
        .filter(|allocation| allocation.tile == logical)
    {
        let key = (allocation.address, allocation.size);
        let bytes = initial
            .get(&(logical, allocation.address))
            .cloned()
            .unwrap_or_else(|| vec![0; allocation.size as usize]);
        if bytes.len() > allocation.size as usize {
            return Err(format!("initializer exceeds allocation on tile {logical}").into());
        }
        let mut padded = vec![0; allocation.size as usize];
        padded[..bytes.len()].copy_from_slice(&bytes);
        if let Some(previous) = allocations.insert(key, padded.clone())
            && previous != padded
        {
            return Err(format!("conflicting initializers on tile {logical}").into());
        }
    }
    for ((address, size), bytes) in allocations {
        let blob = app.add_blob(bytes);
        segments.push(Segment {
            address,
            memory_size: size,
            blob,
            blob_offset: 0,
            file_size: size,
            flags: SEGMENT_READ | SEGMENT_WRITE,
        });
    }
    Ok(())
}

pub fn run_host(
    app: &Application,
    bootloader: &[u8],
    configuration: &[u8],
    device_path: &str,
    input: &[u8],
) -> Result<Vec<u8>> {
    if app.host_exchange.calls.is_empty() {
        return Err("application has no generated host graph call".into());
    }
    block_device_interrupt_signals()?;
    let device = Device::open(device_path)?;
    device.initialize()?;
    device.replay_configuration(configuration)?;
    Loader::new(&device, bootloader)?.load(app, app.host_exchange.startup_mark)?;
    for write in &app.device_config_writes {
        device.write_config(write.offset, write.value)?;
    }
    let mut session = HostSession::new(&device, app.host_exchange.clone())?;
    let calls = app.host_exchange.calls.clone();
    if let Err(error) = session.start() {
        return Err(format!(
            "generated host startup rendezvous: {error}; supervisor states: {}; device outputs: {}",
            supervisor_state_summary(&device, app),
            host_source_summary(&device, app)
        )
        .into());
    }
    let output_size = calls
        .iter()
        .flat_map(|call| &call.outputs)
        .map(|slice| slice.file_offset + slice.size)
        .max()
        .unwrap_or(0);
    let mut output = vec![0; usize::try_from(output_size)?];
    for call in &calls {
        let call_output = match session.invoke(&call.name, call_input(call, input)?) {
            Ok(output) => output,
            Err(error) => {
                let states = supervisor_state_summary(&device, app);
                let sources = host_source_summary(&device, app);
                return Err(format!(
                    "generated host call {}: {error}; supervisor states: {states}; device outputs: {sources}",
                    call.name
                )
                .into());
            }
        };
        for slice in &call.outputs {
            let start = usize::try_from(slice.file_offset)?;
            let end = usize::try_from(slice.file_offset + slice.size)?;
            output[start..end].copy_from_slice(&call_output[start..end]);
        }
    }
    device.write_config(ipu_driver::pci::HSP_GS2_CONTROL, 1)?;
    verify_runtime_completion(&device, app)?;
    debug!(states = %supervisor_state_summary(&device, app), "host exchange supervisor states");
    debug!(sources = %host_source_summary(&device, app), "host exchange device sources");
    Ok(output)
}

fn verify_runtime_completion(device: &Device, app: &Application) -> Result<()> {
    let completion = app
        .outputs
        .iter()
        .find(|binding| binding.name == "runtime-completion")
        .and_then(|binding| binding.slices.first())
        .ok_or("application has no runtime completion binding")?;
    let deadline = Instant::now() + Duration::from_secs(1);
    while !completion_reached(device, completion.tile as u16)? {
        if Instant::now() >= deadline {
            return Err(format!(
                "host graph did not complete; supervisor states: {}; device outputs: {}",
                supervisor_state_summary(device, app),
                host_source_summary(device, app)
            )
            .into());
        }
        std::thread::sleep(Duration::from_micros(100));
    }
    if device.read_tile_word(completion.tile as u16, completion.tile_address)? != 1 {
        return Err("host graph reached its completion trap before storing completion".into());
    }
    Ok(())
}

fn call_input<'a>(call: &HostCall, input: &'a [u8]) -> Result<&'a [u8]> {
    let size = call
        .inputs
        .iter()
        .map(|slice| slice.file_offset + slice.size)
        .max()
        .unwrap_or(0);
    Ok(input
        .get(..usize::try_from(size)?)
        .ok_or("host input is shorter than generated call range")?)
}

fn host_source_summary(device: &Device, app: &Application) -> String {
    app.inputs
        .iter()
        .chain(&app.outputs)
        .filter(|binding| binding.name != "runtime-completion")
        .flat_map(|binding| {
            binding.slices.iter().map(|slice| {
                let physical_tile = slice.tile as u16;
                let value = device
                    .tile_context_state(physical_tile, 0)
                    .and_then(|state| {
                        if state == 0 {
                            device.read_tile_word_from_inactive_context(
                                physical_tile,
                                1,
                                slice.tile_address,
                            )
                        } else {
                            device.read_tile_word(physical_tile, slice.tile_address)
                        }
                    })
                    .map(|word| format!("0x{word:08x}"))
                    .unwrap_or_else(|error| format!("error({error})"));
                format!(
                    "{}@{}:{:#x}={value}",
                    binding.name, slice.tile, slice.tile_address
                )
            })
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn supervisor_state_summary(device: &Device, app: &Application) -> String {
    let mut counts = [0usize; 4];
    let mut samples = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut read_errors = 0usize;
    for tile in &app.tiles {
        match device.tile_context_state(tile.physical_tile as u16, 0) {
            Ok(state @ 0..=3) => {
                counts[state as usize] += 1;
                if samples[state as usize].len() < 8 {
                    samples[state as usize].push(tile.physical_tile);
                }
            }
            Ok(_) | Err(_) => read_errors += 1,
        }
    }
    let program_counters = app
        .tiles
        .iter()
        .take(8)
        .map(|tile| {
            let pc = device
                .read_tile_program_counter(tile.physical_tile as u16, 0)
                .map(|pc| format!("0x{pc:x}"))
                .unwrap_or_else(|error| format!("error({error})"));
            format!("{}:{pc}", tile.physical_tile)
        })
        .collect::<Vec<_>>();
    let milestones = app
        .tiles
        .iter()
        .take(8)
        .map(|tile| {
            let physical_tile = tile.physical_tile as u16;
            let address = tile.diagnostic_address + 4;
            let value = device
                .tile_context_state(physical_tile, 0)
                .and_then(|state| {
                    if state == 0 {
                        device.read_tile_word_from_inactive_context(physical_tile, 1, address)
                    } else {
                        device.read_tile_word(physical_tile, address)
                    }
                })
                .map(|value| format!("0x{value:x}"))
                .unwrap_or_else(|error| format!("error({error})"));
            format!("{}:{value}", tile.physical_tile)
        })
        .collect::<Vec<_>>();
    format!(
        "0={} {:?}, 1={} {:?}, 2={} {:?}, 3={} {:?}, errors={read_errors}, pc={program_counters:?}, milestones={milestones:?}",
        counts[0], samples[0], counts[1], samples[1], counts[2], samples[2], counts[3], samples[3]
    )
}

pub fn run_diagnostic(
    app: &Application,
    bootloader: &[u8],
    configuration: &[u8],
    device_path: &str,
) -> Result<DiagnosticResults> {
    let entry = app
        .entry_points
        .first()
        .ok_or("application has no entry point")?;
    let completion = app
        .outputs
        .iter()
        .find(|binding| binding.name == "runtime-completion")
        .and_then(|binding| binding.slices.first())
        .ok_or("application has no runtime completion binding")?;
    block_device_interrupt_signals()?;
    let device = Device::open(device_path)?;
    device.initialize()?;
    device.replay_configuration(configuration)?;
    Loader::new(&device, bootloader)?.load(app, 0)?;
    for write in &app.device_config_writes {
        device.write_config(write.offset, write.value)?;
    }
    let mut host_pages = if app.host_exchange.pages.is_empty() {
        None
    } else {
        let mut session = HostSession::new(&device, app.host_exchange.clone())?;
        session.attach()?;
        Some(session)
    };
    device.write_config(
        ipu_driver::pci::EXCHANGE_WINDOW_BASE,
        ipu_driver::pci::EXCHANGE_WINDOW_HEXOPT,
    )?;
    device.write_config(ipu_driver::pci::HSP_GS2_CONTROL, 1)?;
    device.set_mark(1)?;
    for phase in 1..entry.external_syncs {
        if let Err(error) =
            device.wait_mark(ipu_driver::pci::HSP_GS2_CONTROL, 0, Duration::from_secs(10))
        {
            return Err(format!(
                "diagnostic external sync phase {phase}: {error}; supervisor states: {}; device outputs: {}",
                supervisor_state_summary(&device, app),
                host_source_summary(&device, app)
            )
            .into());
        }
        device.write_config(ipu_driver::pci::HSP_GS2_CONTROL, 1)?;
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    let output_tiles = app
        .outputs
        .iter()
        .flat_map(|binding| &binding.slices)
        .map(|slice| slice.tile as u16)
        .collect::<std::collections::BTreeSet<_>>();
    loop {
        let mut all_complete = true;
        for &physical_tile in &output_tiles {
            if !completion_reached(&device, physical_tile)? {
                all_complete = false;
                break;
            }
        }
        if all_complete {
            break;
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "diagnostic graph did not reach its completion trap; supervisor states: {}; device outputs: {}",
                supervisor_state_summary(&device, app),
                host_source_summary(&device, app)
            )
            .into());
        }
        std::thread::sleep(Duration::from_micros(100));
    }
    if device.read_tile_word(completion.tile as u16, completion.tile_address)? != 1 {
        return Err("diagnostic completion rendezvous preceded the completion store".into());
    }

    let mut bindings = BTreeMap::new();
    for binding in &app.outputs {
        let total_words = binding
            .slices
            .iter()
            .map(|slice| slice.file_offset + slice.size)
            .max()
            .unwrap_or(0)
            .div_ceil(4) as usize;
        let mut words = vec![0; total_words];
        for slice in &binding.slices {
            if slice.file_offset & 3 != 0 || slice.size & 3 != 0 || slice.tile_address & 3 != 0 {
                return Err(format!("binding {} is not word aligned", binding.name).into());
            }
            for byte_offset in (0..slice.size).step_by(4) {
                words[(slice.file_offset + byte_offset) as usize / 4] = device
                    .read_tile_word(slice.tile as u16, slice.tile_address + byte_offset as u32)?;
            }
        }
        debug!(binding = %binding.name, words = words.len(), "read diagnostic binding");
        bindings.insert(binding.name.clone(), words);
    }
    drop(host_pages.take());
    Ok(DiagnosticResults { bindings })
}

fn patch_setzi_immediate(code: &mut [u8], offset: usize, value: u32) -> Result<()> {
    const IMMEDIATE_MASK: u32 = (1 << 20) - 1;
    if value > IMMEDIATE_MASK || offset + 4 > code.len() {
        return Err("setzi patch is outside its encoding range".into());
    }
    let mut instruction = u32::from_le_bytes(code[offset..offset + 4].try_into()?);
    instruction = (instruction & !IMMEDIATE_MASK) | value;
    code[offset..offset + 4].copy_from_slice(&instruction.to_le_bytes());
    Ok(())
}

fn completion_reached(device: &Device, physical_tile: u16) -> Result<bool> {
    match device.tile_context_state(physical_tile, 0) {
        Ok(3) => Ok(true),
        Ok(_) | Err(DriverError::Timeout(_)) => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn patch_instruction(code: &mut [u8], offset: usize, instruction: u32) -> Result<()> {
    let word = code
        .get_mut(offset..offset + 4)
        .ok_or("instruction patch is outside the linked image")?;
    word.copy_from_slice(&instruction.to_le_bytes());
    Ok(())
}

fn patch_plan_body(
    code: &mut [u8],
    offset: usize,
    row: &ipu_exchange::PlanRow,
    expected_words: usize,
) -> Result<()> {
    let end = row
        .iter()
        .position(|word| *word == ipu_exchange::RETURN_M10_INSTRUCTION)
        .ok_or("global synchronization plan has no return")?;
    if end != expected_words {
        return Err(format!(
            "global synchronization plan has {end} body words, runtime reserves {expected_words}"
        )
        .into());
    }
    for (index, instruction) in row[..end].iter().copied().enumerate() {
        patch_instruction(code, offset + index * 4, instruction)?;
    }
    Ok(())
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

const fn align_up(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

const fn ranges_overlap(left_start: u32, left_end: u32, right_start: u32, right_end: u32) -> bool {
    left_start < right_end && right_start < left_end
}
