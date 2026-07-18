use ipu_compiler::Schedule;
use ipu_driver::{Device, DriverError, HostSession, Loader, block_device_interrupt_signals};
use ipu_elf::{LinkOptions, link};
use ipu_exchange::Topology;
use ipu_package::{
    Application, Binding, EntryPoint, HostCall, HostExchange, HostPage, HostSlice, RegionSlice,
    SEGMENT_EXECUTE, SEGMENT_READ, SEGMENT_WRITE, Segment, TileImage,
};
use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;

mod static_codegen;

const PLAN_BASE: u32 = ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES;
const HOST_DATA_START: u32 = 64;
const HOST_CLOSE_ADDRESS: u32 = ipu_exchange::EXCHANGE_WINDOW_BASE + 0x160;
const HOST_PACKET_SEARCH_BASE: u32 = ipu_exchange::EXCHANGE_WINDOW_BASE + 0x180;

#[derive(Clone, Copy, Debug)]
enum HostDirection {
    ToTile,
    ToHost,
}

#[derive(Clone, Copy, Debug)]
struct StaticHostTransfer {
    direction: HostDirection,
    physical_tile: u16,
    tile_address: u32,
    host_offset: u32,
    bytes: u32,
}

struct StaticHostLayout {
    inputs: Vec<StaticHostTransfer>,
    outputs: Vec<StaticHostTransfer>,
    protocol: HostExchange,
}
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

fn allocate_host_packet_addresses(schedule: &Schedule, sizes: &[usize]) -> Result<Vec<u32>> {
    let limit = ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::HOST_TO_TILE_WINDOW_BYTES;
    let mut candidate = HOST_PACKET_SEARCH_BASE;
    while candidate < limit {
        let (addresses, end) = packed_addresses(candidate, sizes, 8)?;
        if end <= limit
            && schedule.allocations.iter().all(|allocation| {
                !ranges_overlap(
                    candidate,
                    end,
                    allocation.address,
                    allocation.address.saturating_add(allocation.size),
                )
            })
        {
            return Ok(addresses);
        }
        candidate = candidate
            .checked_add(32)
            .ok_or("host packet allocation overflow")?;
    }
    Err("no exchange-window range is available for host packet tables".into())
}

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub fn package_graph(graph: &ExecutableGraph, objects: &[Vec<u8>]) -> Result<Application> {
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
    let host = build_static_host_layout(graph)?;
    let host_transfers = host
        .inputs
        .iter()
        .chain(&host.outputs)
        .copied()
        .collect::<Vec<_>>();
    let host_plan_sizes = host_transfers
        .iter()
        .map(|transfer| host_phase_size(*transfer))
        .collect::<Result<Vec<_>>>()?;
    let (host_plan_addresses, host_plan_end) = packed_addresses(plan_end, &host_plan_sizes, 8)?;
    let host_packet_sizes = host_transfers
        .iter()
        .map(|transfer| host_packet_size(*transfer))
        .collect::<Result<Vec<_>>>()?;
    let host_packet_addresses =
        allocate_host_packet_addresses(&graph.schedule, &host_packet_sizes)?;
    let completion_address = align_up(host_plan_end, 64);
    let runtime_end = completion_address
        .checked_add(4)
        .ok_or("static runtime address overflow")?;

    let mut retained_symbols = vec![
        static_codegen::WORKER_BARRIER.into(),
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
            static_codegen::emit(
                program,
                program_base,
                &image.symbols,
                &plan_addresses,
                &host_plan_addresses[..host.inputs.len()],
                &host_plan_addresses[host.inputs.len()..],
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
        append_static_host_segments(
            &mut app,
            &mut segments,
            physical as u16,
            &host_transfers,
            &host_plan_addresses,
            &host_packet_addresses,
        )?;
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
    app.inputs = graph.host_inputs.clone();
    app.outputs = graph.outputs.clone();
    app.outputs.extend(graph.host_outputs.clone());
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
        external_syncs: 1,
    });
    app.host_exchange = host.protocol;
    app.validate()?;
    info!(
        tiles = app.tiles.len(),
        exchange_launches = exchange_count,
        host_inputs = graph.host_inputs.len(),
        host_outputs = graph.host_outputs.len(),
        max_program_bytes = program_end - program_base,
        "packaged static executable graph"
    );
    Ok(app)
}

fn build_static_host_layout(graph: &ExecutableGraph) -> Result<StaticHostLayout> {
    if graph.host_inputs.is_empty() && graph.host_outputs.is_empty() {
        return Ok(StaticHostLayout {
            inputs: Vec::new(),
            outputs: Vec::new(),
            protocol: HostExchange::default(),
        });
    }

    let mut host_cursor = HOST_DATA_START;
    let mut input_file_cursor = 0u64;
    let mut output_file_cursor = 0u64;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut calls = Vec::new();
    append_host_bindings(
        &graph.host_inputs,
        HostDirection::ToTile,
        &mut host_cursor,
        &mut input_file_cursor,
        &mut inputs,
        &mut calls,
    )?;
    append_host_bindings(
        &graph.host_outputs,
        HostDirection::ToHost,
        &mut host_cursor,
        &mut output_file_cursor,
        &mut outputs,
        &mut calls,
    )?;
    let call_inputs = calls
        .iter()
        .flat_map(|call| call.inputs.iter().cloned())
        .collect();
    let call_outputs = calls
        .iter()
        .flat_map(|call| call.outputs.iter().cloned())
        .collect();
    let transfer_count = inputs.len() + outputs.len();
    calls = vec![HostCall {
        name: "graph".into(),
        command: 0,
        phases: u32::try_from(transfer_count + 1)?,
        inputs: call_inputs,
        outputs: call_outputs,
    }];

    let data_page_count = align_up(
        host_cursor.max(ipu_exchange::HOST_PAGE_BYTES),
        ipu_exchange::HOST_PAGE_BYTES,
    ) / ipu_exchange::HOST_PAGE_BYTES;
    let command_page = data_page_count;
    Ok(StaticHostLayout {
        inputs,
        outputs,
        protocol: HostExchange {
            startup_mark: ipu_driver::HOST_EXCHANGE_HANDOFF_MARK,
            command_page,
            command_offset: 0,
            pages: (0..=command_page)
                .map(|index| HostPage {
                    index,
                    size: u64::from(ipu_exchange::HOST_PAGE_BYTES),
                })
                .collect(),
            attach_order: std::iter::once(command_page)
                .chain(0..data_page_count)
                .collect(),
            calls,
        },
    })
}

fn append_host_bindings(
    bindings: &[Binding],
    direction: HostDirection,
    host_cursor: &mut u32,
    file_cursor: &mut u64,
    transfers: &mut Vec<StaticHostTransfer>,
    calls: &mut Vec<HostCall>,
) -> Result<()> {
    for binding in bindings {
        let binding_base = align_up(*host_cursor, 64);
        let binding_file_base = *file_cursor;
        for slice in &binding.slices {
            let host_offset = binding_base
                .checked_add(u32::try_from(slice.file_offset)?)
                .ok_or("host binding offset overflow")?;
            let transfer = StaticHostTransfer {
                direction,
                physical_tile: u16::try_from(slice.tile)?,
                tile_address: slice.tile_address,
                host_offset,
                bytes: u32::try_from(slice.size)?,
            };
            match direction {
                HostDirection::ToTile => {
                    ipu_exchange::plan_host_to_tile(
                        transfer.physical_tile,
                        transfer.tile_address,
                        transfer.host_offset,
                        transfer.bytes,
                    )?;
                }
                HostDirection::ToHost => {
                    ipu_exchange::plan_tile_to_host(
                        transfer.physical_tile,
                        transfer.tile_address,
                        transfer.host_offset,
                        transfer.bytes,
                    )?;
                }
            }
            let mut host_slices = Vec::new();
            append_host_slices(
                &mut host_slices,
                host_offset,
                binding_file_base + slice.file_offset,
                slice.size,
            )?;
            let index = transfers.len();
            transfers.push(transfer);
            calls.push(HostCall {
                name: match direction {
                    HostDirection::ToTile => format!("host-input-{index}"),
                    HostDirection::ToHost => format!("host-output-{index}"),
                },
                command: 0,
                phases: 0,
                inputs: matches!(direction, HostDirection::ToTile)
                    .then_some(host_slices.clone())
                    .unwrap_or_default(),
                outputs: matches!(direction, HostDirection::ToHost)
                    .then_some(host_slices)
                    .unwrap_or_default(),
            });
        }
        let size = binding_size(binding)?;
        *host_cursor = u32::try_from(u64::from(binding_base) + size)?;
        *file_cursor = file_cursor
            .checked_add(size)
            .ok_or("host binding file range overflow")?;
    }
    Ok(())
}

fn append_host_slices(
    slices: &mut Vec<HostSlice>,
    mut host_offset: u32,
    mut file_offset: u64,
    mut size: u64,
) -> Result<()> {
    while size != 0 {
        let page = host_offset / ipu_exchange::HOST_PAGE_BYTES;
        let page_offset = host_offset % ipu_exchange::HOST_PAGE_BYTES;
        let count = size.min(u64::from(ipu_exchange::HOST_PAGE_BYTES - page_offset));
        slices.push(HostSlice {
            page,
            page_offset: u64::from(page_offset),
            file_offset,
            size: count,
        });
        host_offset = host_offset
            .checked_add(u32::try_from(count)?)
            .ok_or("host page offset overflow")?;
        file_offset = file_offset
            .checked_add(count)
            .ok_or("host file offset overflow")?;
        size -= count;
    }
    Ok(())
}

fn binding_size(binding: &Binding) -> Result<u64> {
    binding.slices.iter().try_fold(0u64, |size, slice| {
        slice
            .file_offset
            .checked_add(slice.size)
            .map(|end| size.max(end))
            .ok_or_else(|| "host binding size overflow".into())
    })
}

fn host_target_program(
    transfer: StaticHostTransfer,
    packet_address: u32,
) -> Result<ipu_exchange::TileToHostProgram> {
    Ok(match transfer.direction {
        HostDirection::ToTile => ipu_exchange::assemble_host_to_tile_target_program(
            transfer.physical_tile,
            transfer.tile_address,
            transfer.host_offset,
            transfer.bytes,
            packet_address,
        )?,
        HostDirection::ToHost => ipu_exchange::assemble_tile_to_host_target_program(
            transfer.physical_tile,
            transfer.tile_address,
            transfer.host_offset,
            transfer.bytes,
            packet_address,
            HOST_CLOSE_ADDRESS,
        )?,
    })
}

fn host_phase_size(transfer: StaticHostTransfer) -> Result<usize> {
    let target = host_target_program(transfer, PLAN_BASE + 8)?;
    let hierarchy = ipu_exchange::host_hierarchy(transfer.physical_tile)?;
    let target_words = if hierarchy.xreq_physical_tile == transfer.physical_tile {
        ipu_exchange::wrap_local_host_operation(&target.instructions, PLAN_BASE)?
    } else {
        ipu_exchange::wrap_host_target_operation(transfer.physical_tile, &target.instructions)?
    };
    let xreq = ipu_exchange::assemble_host_xreq_program(transfer.physical_tile, PLAN_BASE)?;
    let xreq_words =
        ipu_exchange::wrap_host_xreq_operation(hierarchy.xreq_physical_tile, &xreq.instructions)?;
    Ok(target_words.len().max(xreq_words.len()).max(3) * 4)
}

fn host_packet_size(transfer: StaticHostTransfer) -> Result<usize> {
    let target = host_target_program(transfer, PLAN_BASE + 8)?;
    Ok(8 + target.packet_words.len() * 4)
}

fn append_static_host_segments(
    app: &mut Application,
    segments: &mut Vec<Segment>,
    physical_tile: u16,
    transfers: &[StaticHostTransfer],
    plan_addresses: &[u32],
    packet_addresses: &[u32],
) -> Result<()> {
    for ((transfer, &plan_address), &packet_address) in
        transfers.iter().zip(plan_addresses).zip(packet_addresses)
    {
        let hierarchy = ipu_exchange::host_hierarchy(transfer.physical_tile)?;
        let target = host_target_program(*transfer, packet_address + 8)?;
        let (instructions, packet_words) = if hierarchy.xreq_physical_tile == transfer.physical_tile
            && physical_tile == transfer.physical_tile
        {
            let mut packets = vec![1, 0];
            packets.extend_from_slice(&target.packet_words);
            (
                ipu_exchange::wrap_local_host_operation(&target.instructions, packet_address)?,
                Some(packets),
            )
        } else if physical_tile == hierarchy.xreq_physical_tile {
            let xreq =
                ipu_exchange::assemble_host_xreq_program(transfer.physical_tile, packet_address)?;
            (
                ipu_exchange::wrap_host_xreq_operation(physical_tile, &xreq.instructions)?,
                Some(xreq.packet_words),
            )
        } else if physical_tile == transfer.physical_tile {
            (
                ipu_exchange::wrap_host_target_operation(physical_tile, &target.instructions)?,
                Some(target.packet_words),
            )
        } else {
            (
                vec![
                    ipu_exchange::sans(255),
                    ipu_exchange::SYNC_ANS_INSTRUCTION,
                    ipu_exchange::RETURN_M10_INSTRUCTION,
                ],
                None,
            )
        };
        let instruction_bytes = words_to_bytes(&instructions);
        let size = u32::try_from(instruction_bytes.len())?;
        let blob = app.add_blob(instruction_bytes);
        segments.push(Segment {
            address: plan_address,
            memory_size: size,
            blob,
            blob_offset: 0,
            file_size: size,
            flags: SEGMENT_READ | SEGMENT_EXECUTE,
        });
        if let Some(packet_words) = packet_words {
            let address = if physical_tile == transfer.physical_tile
                && hierarchy.xreq_physical_tile != transfer.physical_tile
            {
                packet_address + 8
            } else {
                packet_address
            };
            let packet_bytes = words_to_bytes(&packet_words);
            let size = u32::try_from(packet_bytes.len())?;
            let blob = app.add_blob(packet_bytes);
            segments.push(Segment {
                address,
                memory_size: size,
                blob,
                blob_offset: 0,
                file_size: size,
                flags: SEGMENT_READ,
            });
        }
    }
    Ok(())
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

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

const fn align_up(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

const fn ranges_overlap(left_start: u32, left_end: u32, right_start: u32, right_end: u32) -> bool {
    left_start < right_end && right_start < left_end
}
