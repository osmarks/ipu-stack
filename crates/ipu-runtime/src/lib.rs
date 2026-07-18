use ipu_compiler::{AllocationKind, Phase, Schedule};
use ipu_driver::{Device, HostSession, Loader, block_device_interrupt_signals};
use ipu_elf::{LinkOptions, link};
use ipu_exchange::{
    GlobalSyncProgram, TileToHostProgram, Topology, patch_multicast_receiver_address,
};
use ipu_package::{
    Application, Binding, DeviceConfigWrite, EntryPoint, HostCall, HostExchange, HostPage,
    HostSlice, RegionSlice, SEGMENT_EXECUTE, SEGMENT_READ, SEGMENT_WRITE, Segment, TileImage,
};
use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;

const COMMAND_WORDS: usize = 5;
const COMMAND_BYTES: u32 = (COMMAND_WORDS * 4) as u32;
const PLAN_BASE: u32 = ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES;
const TILE_MUX_HOST_BASE: u32 = 0x600;
const WORKER_CONTEXTS: u32 = 7;
const WORKER_CONTEXT_BYTES: u32 = 0x30;
const HOST_PACKET_ALIGNMENT: u32 = 32;
const INVALID_HOST_COMMAND: u32 = u32::MAX;
const HOST_COMMAND_RUN: u32 = 0;
const HOST_COMMAND_TILE_TO_HOST: u32 = 1;
const HOST_COMMAND_HOST_TO_TILE: u32 = 2;
const HOST_PACKET_BASE: u32 = align_up(
    ipu_exchange::EXCHANGE_WINDOW_BASE + WORKER_CONTEXTS * WORKER_CONTEXT_BYTES,
    HOST_PACKET_ALIGNMENT,
);

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

#[derive(Clone, Copy)]
#[repr(u32)]
enum RuntimeRole {
    Inactive = 2,
    ExchangeAbsolute = 10,
    Compute = 11,
    ComputeNoop = 12,
    HostProgram = 13,
    HostIdle = 14,
    HostProgramUnbarriered = 15,
    HostIdleUnbarriered = 16,
    CommandGuard = 17,
    CommandFollower = 18,
    CommandCoordinator = 19,
    CommandHandlerFollower = 20,
}

impl RuntimeRole {
    const fn word(self) -> u32 {
        self as u32
    }
}

#[derive(Clone, Copy)]
struct RuntimeLayout {
    plan_stride: u32,
    host_packet_stride: u32,
    command_address: u32,
    completion_address: u32,
    milestone_address: u32,
    sync_packet_address: u32,
    sync_release_address: u32,
    host_command_address: u32,
}

impl RuntimeLayout {
    fn new(
        schedule: &Schedule,
        plan_bytes: &[usize],
        packet_bytes: &[usize],
        command_count: usize,
    ) -> Result<Self> {
        let plan_stride = align_up(
            u32::try_from(plan_bytes.iter().copied().max().unwrap_or(1))?,
            8,
        );
        let plan_end = PLAN_BASE
            .checked_add(
                u32::try_from(command_count)?
                    .checked_mul(plan_stride)
                    .ok_or("plan size overflow")?,
            )
            .ok_or("plan address overflow")?;
        let command_address = align_up(plan_end, 64);
        let command_end = command_address
            .checked_add(
                u32::try_from(command_count)?
                    .checked_mul(COMMAND_BYTES)
                    .ok_or("command size overflow")?,
            )
            .ok_or("command address overflow")?;
        let completion_address = align_up(command_end, 64);
        let milestone_address = completion_address + 4;
        let sync_packet_address = align_up(completion_address + 4, 8);
        let sync_release_address = sync_packet_address + 16;
        let runtime_end = sync_release_address + 4;
        let host_packet_stride = align_up(
            u32::try_from(packet_bytes.iter().copied().max().unwrap_or(1))?,
            HOST_PACKET_ALIGNMENT,
        );
        let host_packet_end = HOST_PACKET_BASE
            .checked_add(
                u32::try_from(command_count)?
                    .checked_mul(host_packet_stride)
                    .ok_or("host packet size overflow")?,
            )
            .ok_or("host packet address overflow")?;
        let host_control_end = host_packet_end
            .checked_add(HOST_PACKET_ALIGNMENT)
            .ok_or("host control address overflow")?;
        if host_control_end
            > ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::HOST_TO_TILE_WINDOW_BYTES
        {
            return Err("host packet table exceeds the exchange window".into());
        }

        for allocation in &schedule.allocations {
            let allocation_end = allocation
                .address
                .checked_add(allocation.size)
                .ok_or("allocation address overflow")?;
            if ranges_overlap(PLAN_BASE, runtime_end, allocation.address, allocation_end)
                || ranges_overlap(
                    HOST_PACKET_BASE,
                    host_packet_end,
                    allocation.address,
                    allocation_end,
                )
                || ranges_overlap(
                    host_packet_end,
                    host_control_end,
                    allocation.address,
                    allocation_end,
                )
            {
                return Err(format!(
                    "runtime region overlaps tensor {} on tile {}",
                    allocation.tensor.0, allocation.tile
                )
                .into());
            }
        }
        Ok(Self {
            plan_stride,
            host_packet_stride,
            command_address,
            completion_address,
            milestone_address,
            sync_packet_address,
            sync_release_address,
            host_command_address: host_packet_end,
        })
    }

    fn plan_address(self, command: usize) -> Result<u32> {
        Ok(PLAN_BASE
            + u32::try_from(command)?
                .checked_mul(self.plan_stride)
                .ok_or("plan address overflow")?)
    }

    fn host_packet_address(self, command: usize) -> Result<u32> {
        Ok(HOST_PACKET_BASE
            + u32::try_from(command)?
                .checked_mul(self.host_packet_stride)
                .ok_or("host packet address overflow")?)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HostDirection {
    ControlRead,
    ToTile,
    ToHost,
}

#[derive(Clone, Copy)]
struct HostTransfer {
    direction: HostDirection,
    physical_tile: u16,
    tile_address: u32,
    host_offset: u32,
    bytes: u32,
}

struct HostLayout {
    inputs: Vec<HostTransfer>,
    run: Option<HostTransfer>,
    outputs: Vec<HostTransfer>,
    protocol: HostExchange,
}

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub fn package_graph(graph: &ExecutableGraph, objects: &[Vec<u8>]) -> Result<Application> {
    let topology = Topology::c600();
    if usize::from(graph.schedule.tile_count) != topology.tile_count() {
        return Err("the direct C600 runtime requires a schedule for every discovered tile".into());
    }
    let lowered = graph.schedule.lower_exchanges(&topology)?;
    let exchange_by_phase: HashMap<_, _> =
        lowered.iter().map(|phase| (phase.phase, phase)).collect();
    if lowered.iter().any(|phase| phase.epochs.len() != 1) {
        return Err("runtime expects one statically scheduled launch per exchange phase".into());
    }
    let host = build_host_layout(graph)?;
    for (index, transfer) in host.inputs.iter().enumerate() {
        debug!(
            command = index,
            direction = ?transfer.direction,
            physical_tile = transfer.physical_tile,
            tile_address = format_args!("{:#x}", transfer.tile_address),
            host_offset = format_args!("{:#x}", transfer.host_offset),
            bytes = transfer.bytes,
            "lowered host input command"
        );
    }
    for (index, transfer) in host.outputs.iter().enumerate() {
        debug!(
            command = host.inputs.len() + graph.schedule.phases.len() + index,
            direction = ?transfer.direction,
            physical_tile = transfer.physical_tile,
            tile_address = format_args!("{:#x}", transfer.tile_address),
            host_offset = format_args!("{:#x}", transfer.host_offset),
            bytes = transfer.bytes,
            "lowered host output command"
        );
    }
    let run_commands = usize::from(host.run.is_some()) * 2;
    let command_count = host.inputs.len()
        + host.inputs.len() / 2
        + run_commands
        + graph.schedule.phases.len()
        + host.outputs.len()
        + host.outputs.len() / 2;
    let mut plan_bytes = Vec::new();
    let mut packet_bytes = Vec::new();
    for commands in host.inputs.chunks_exact(2) {
        plan_bytes.extend([
            host_plan_size(&commands[0])?.max(ipu_exchange::PLAN_WORDS * 4),
            0,
            host_plan_size(&commands[1])?,
        ]);
        packet_bytes.extend([
            host_packet_size(&commands[0])?,
            0,
            host_packet_size(&commands[1])?,
        ]);
    }
    if let Some(run) = &host.run {
        plan_bytes.extend([host_plan_size(run)?.max(ipu_exchange::PLAN_WORDS * 4), 0]);
        packet_bytes.extend([host_packet_size(run)?, 0]);
    }
    for (phase_index, phase) in graph.schedule.phases.iter().enumerate() {
        plan_bytes.push(match phase {
            Phase::Exchange { .. } => exchange_by_phase[&phase_index].epochs[0]
                .tile_rows
                .values()
                .map(|row| row.len() * 4)
                .max()
                .unwrap_or(4),
            Phase::Compute { .. } => 0,
        });
        packet_bytes.push(0);
    }
    for commands in host.outputs.chunks_exact(2) {
        plan_bytes.extend([
            host_plan_size(&commands[0])?.max(ipu_exchange::PLAN_WORDS * 4),
            0,
            host_plan_size(&commands[1])?,
        ]);
        packet_bytes.extend([
            host_packet_size(&commands[0])?,
            0,
            host_packet_size(&commands[1])?,
        ]);
    }
    plan_bytes.resize(command_count, 0);
    packet_bytes.resize(command_count, 0);
    let layout = RuntimeLayout::new(
        &graph.schedule,
        &plan_bytes,
        &packet_bytes,
        command_count + 1,
    )?;
    let command_broadcast = command_broadcast_rows(&topology, layout.host_command_address)?;
    let exchange_commands = graph
        .schedule
        .phases
        .iter()
        .filter(|phase| matches!(phase, Phase::Exchange { .. }))
        .count();
    let global_sync = ipu_exchange::c600_global_sync();
    let image = link(
        objects,
        &LinkOptions {
            image_base: ipu_driver::APPLICATION_LOAD_BASE,
            entry_symbol: "ipu_stack_exchange_loop_start".into(),
            retained_symbols: graph
                .schedule
                .phases
                .iter()
                .flat_map(|phase| match phase {
                    Phase::Compute { commands, .. } => commands.as_slice(),
                    Phase::Exchange { .. } => &[],
                })
                .map(|command| format!("ipu_stack_{}", command.specialization.operation))
                .collect(),
            externals: HashMap::new(),
        },
    )?;
    let symbol_offset = |name: &str| -> Result<usize> {
        Ok(image
            .symbols
            .get(name)
            .copied()
            .ok_or_else(|| format!("exchange runtime has no {name} symbol"))?
            .checked_sub(image.base)
            .ok_or_else(|| format!("{name} precedes the linked image"))? as usize)
    };
    let worker_sync_offset = symbol_offset("ipu_stack_loop_worker_sync_base")?;
    let command_offset = symbol_offset("ipu_stack_command_table_address")?;
    let completion_offset = symbol_offset("ipu_stack_completion_address")?;
    let milestone_offset = image
        .symbols
        .get("ipu_stack_milestone_address")
        .copied()
        .map(|address| {
            address
                .checked_sub(image.base)
                .ok_or("ipu_stack_milestone_address precedes the linked image")
                .map(|offset| offset as usize)
        })
        .transpose()?;
    let completion_master_offset = symbol_offset("ipu_stack_completion_master_flag")?;
    let pre_sync_master_offset = symbol_offset("ipu_stack_pre_sync_master_flag")?;
    let device_sync_master_offset = symbol_offset("ipu_stack_device_sync_master_flag")?;
    let endpoint_offset = symbol_offset("ipu_stack_loop_global_sync_endpoint")?;
    let send0_offset = symbol_offset("ipu_stack_loop_global_sync_send0")?;
    let send1_offset = symbol_offset("ipu_stack_loop_global_sync_send1")?;
    let release_offset = symbol_offset("ipu_stack_loop_global_sync_release")?;

    let initial: HashMap<_, _> = graph
        .initial_buffers
        .iter()
        .map(|buffer| ((buffer.tile, buffer.address), words_to_bytes(&buffer.words)))
        .collect();
    let mut app = Application::default();
    for logical in 0..graph.schedule.tile_count {
        let physical = u32::from(topology.physical(logical)?);
        let mut code = image.bytes.clone();
        patch_setzi_immediate(&mut code, worker_sync_offset, u32::from(logical) * 8)?;
        patch_setzi_immediate(&mut code, command_offset, layout.command_address)?;
        patch_setzi_immediate(&mut code, completion_offset, layout.completion_address)?;
        if let Some(offset) = milestone_offset {
            patch_setzi_immediate(&mut code, offset, layout.milestone_address)?;
        }
        let is_sync_master = physical == u32::from(global_sync.packet_origin_physical_tile);
        for offset in [
            completion_master_offset,
            pre_sync_master_offset,
            device_sync_master_offset,
        ] {
            patch_setzi_immediate(&mut code, offset, u32::from(is_sync_master))?;
        }
        if is_sync_master {
            for (offset, address) in [
                (send0_offset, layout.sync_packet_address),
                (send1_offset, layout.sync_packet_address + 8),
            ] {
                code[offset..offset + 4]
                    .copy_from_slice(&ipu_exchange::encode_send(1, 3, address >> 2)?.to_le_bytes());
            }
            code[release_offset..release_offset + 4].copy_from_slice(
                &ipu_exchange::encode_send(0, 3, layout.sync_release_address >> 2)?.to_le_bytes(),
            );
            patch_setzi_immediate(&mut code, endpoint_offset, TILE_MUX_HOST_BASE + physical)?;
        }

        let mut segments = Vec::new();
        let code_blob = app.add_blob(code);
        segments.push(Segment {
            address: image.base,
            memory_size: image.bytes.len() as u32,
            blob: code_blob,
            blob_offset: 0,
            file_size: image.bytes.len() as u32,
            flags: SEGMENT_READ | SEGMENT_EXECUTE,
        });

        let mut commands = Vec::new();
        let mut command_index = 0usize;
        for pair in host.inputs.chunks_exact(2) {
            let retry_index = command_index;
            append_host_command(
                &mut app,
                &mut segments,
                &mut commands,
                pair[0],
                physical as u16,
                HostCommandContext {
                    layout: &layout,
                    command_index,
                    barrier_after: true,
                    command_dispatch: true,
                    command_receiver: command_broadcast.get(&logical).map(Vec::as_slice),
                    handler_follower: false,
                },
            )?;
            command_index += 1;
            append_command_guard(
                &mut commands,
                HOST_COMMAND_HOST_TO_TILE,
                layout.command_address + u32::try_from(retry_index)? * COMMAND_BYTES,
                layout.host_command_address,
            );
            command_index += 1;
            append_host_command(
                &mut app,
                &mut segments,
                &mut commands,
                pair[1],
                physical as u16,
                HostCommandContext {
                    layout: &layout,
                    command_index,
                    barrier_after: true,
                    command_dispatch: false,
                    command_receiver: None,
                    handler_follower: true,
                },
            )?;
            command_index += 1;
        }
        if let Some(run) = host.run {
            let retry_index = command_index;
            append_host_command(
                &mut app,
                &mut segments,
                &mut commands,
                run,
                physical as u16,
                HostCommandContext {
                    layout: &layout,
                    command_index,
                    barrier_after: true,
                    command_dispatch: true,
                    command_receiver: command_broadcast.get(&logical).map(Vec::as_slice),
                    handler_follower: false,
                },
            )?;
            command_index += 1;
            append_command_guard(
                &mut commands,
                HOST_COMMAND_RUN,
                layout.command_address + u32::try_from(retry_index)? * COMMAND_BYTES,
                layout.host_command_address,
            );
            command_index += 1;
        }
        for (phase_index, phase) in graph.schedule.phases.iter().enumerate() {
            match phase {
                Phase::Exchange { .. } => {
                    let lowered = exchange_by_phase
                        .get(&phase_index)
                        .ok_or("missing lowered exchange phase")?;
                    let epoch = &lowered.epochs[0];
                    let (role, plan_address) = if let Some(row) = epoch.tile_rows.get(&logical) {
                        let address = layout.plan_address(command_index)?;
                        let bytes = words_to_bytes(row);
                        let blob = app.add_blob(bytes.clone());
                        segments.push(Segment {
                            address,
                            memory_size: bytes.len() as u32,
                            blob,
                            blob_offset: 0,
                            file_size: bytes.len() as u32,
                            flags: SEGMENT_READ | SEGMENT_WRITE | SEGMENT_EXECUTE,
                        });
                        (RuntimeRole::ExchangeAbsolute, address)
                    } else {
                        (RuntimeRole::Inactive, 0)
                    };
                    commands.extend_from_slice(&words_to_bytes(&[
                        role.word(),
                        plan_address,
                        0,
                        0,
                        0,
                    ]));
                }
                Phase::Compute {
                    commands: phase_commands,
                    ..
                } => {
                    let matching = phase_commands
                        .iter()
                        .filter(|command| command.tile == logical)
                        .collect::<Vec<_>>();
                    if matching.len() > 1 {
                        return Err(format!(
                            "multiple compute commands for tile {logical} in one phase"
                        )
                        .into());
                    }
                    if let Some(command) = matching.first() {
                        if command.inputs.len() != 2 {
                            return Err(format!(
                                "runtime kernel {} requires two inputs",
                                command.specialization.operation
                            )
                            .into());
                        }
                        let kernel_name = format!("ipu_stack_{}", command.specialization.operation);
                        let kernel_entry =
                            image.symbols.get(&kernel_name).copied().ok_or_else(|| {
                                format!("linked image has no kernel {kernel_name}")
                            })?;
                        let output = home_address(&graph.schedule, command.output, logical)?;
                        let left = compute_input_address(
                            &graph.schedule,
                            command.inputs[0],
                            logical,
                            phase_index,
                        )?;
                        let right = compute_input_address(
                            &graph.schedule,
                            command.inputs[1],
                            logical,
                            phase_index,
                        )?;
                        commands.extend_from_slice(&words_to_bytes(&[
                            RuntimeRole::Compute.word(),
                            kernel_entry,
                            output,
                            left,
                            right,
                        ]));
                    } else {
                        commands.extend_from_slice(&words_to_bytes(&[
                            RuntimeRole::ComputeNoop.word(),
                            0,
                            0,
                            0,
                            0,
                        ]));
                    }
                }
            }
            command_index += 1;
        }
        for pair in host.outputs.chunks_exact(2) {
            let retry_index = command_index;
            append_host_command(
                &mut app,
                &mut segments,
                &mut commands,
                pair[0],
                physical as u16,
                HostCommandContext {
                    layout: &layout,
                    command_index,
                    barrier_after: true,
                    command_dispatch: true,
                    command_receiver: command_broadcast.get(&logical).map(Vec::as_slice),
                    handler_follower: false,
                },
            )?;
            command_index += 1;
            append_command_guard(
                &mut commands,
                HOST_COMMAND_TILE_TO_HOST,
                layout.command_address + u32::try_from(retry_index)? * COMMAND_BYTES,
                layout.host_command_address,
            );
            command_index += 1;
            append_host_command(
                &mut app,
                &mut segments,
                &mut commands,
                pair[1],
                physical as u16,
                HostCommandContext {
                    layout: &layout,
                    command_index,
                    barrier_after: false,
                    command_dispatch: false,
                    command_receiver: None,
                    handler_follower: true,
                },
            )?;
            command_index += 1;
        }
        debug_assert_eq!(command_index, command_count);
        commands.extend_from_slice(&words_to_bytes(&[0, 0, 0, 0, 0]));
        let command_blob = app.add_blob(commands.clone());
        segments.push(Segment {
            address: layout.command_address,
            memory_size: commands.len() as u32,
            blob: command_blob,
            blob_offset: 0,
            file_size: commands.len() as u32,
            flags: SEGMENT_READ,
        });
        let milestone_blob = app.add_blob(vec![0; 4]);
        segments.push(Segment {
            address: layout.milestone_address,
            memory_size: 4,
            blob: milestone_blob,
            blob_offset: 0,
            file_size: 4,
            flags: SEGMENT_READ | SEGMENT_WRITE,
        });
        let host_command_blob = app.add_blob(INVALID_HOST_COMMAND.to_le_bytes().to_vec());
        segments.push(Segment {
            address: layout.host_command_address,
            memory_size: 4,
            blob: host_command_blob,
            blob_offset: 0,
            file_size: 4,
            flags: SEGMENT_READ | SEGMENT_WRITE,
        });

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
        if physical == u32::from(global_sync.packet_origin_physical_tile) {
            add_global_sync_segments(&mut app, &mut segments, global_sync, layout);
        }
        app.tiles.push(TileImage {
            physical_tile: physical,
            entry_point: image.base,
            command_address: layout.command_address,
            diagnostic_address: layout.completion_address,
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
            tile: u32::from(global_sync.packet_origin_physical_tile),
            tile_address: layout.completion_address,
            file_offset: 0,
            size: 4,
        }],
    });
    app.device_config_writes
        .extend(global_sync.config_writes.map(|write| DeviceConfigWrite {
            offset: write.offset,
            value: write.value,
        }));
    app.host_exchange = host.protocol;
    app.entry_points.push(EntryPoint {
        name: "graph".into(),
        command: 0,
        external_syncs: 1 + u32::try_from(command_count + exchange_commands)?,
    });
    app.validate()?;
    info!(
        tiles = app.tiles.len(),
        commands = command_count,
        host_inputs = graph.host_inputs.len(),
        host_outputs = graph.host_outputs.len(),
        "packaged executable graph"
    );
    Ok(app)
}

fn build_host_layout(graph: &ExecutableGraph) -> Result<HostLayout> {
    if graph.host_inputs.is_empty() && graph.host_outputs.is_empty() {
        return Ok(HostLayout {
            inputs: Vec::new(),
            run: None,
            outputs: Vec::new(),
            protocol: HostExchange::default(),
        });
    }

    const HOST_PAGE_BYTES: u32 = ipu_exchange::HOST_PAGE_BYTES;
    const DATA_START: u32 = 64;
    let input_page_end = host_bindings_end(&graph.host_inputs, DATA_START)?;
    let output_start = align_up(input_page_end, 64);
    let output_page_end = host_bindings_end(&graph.host_outputs, output_start)?;
    let data_page_count =
        align_up(output_page_end.max(HOST_PAGE_BYTES), HOST_PAGE_BYTES) / HOST_PAGE_BYTES;
    let command_page = data_page_count;
    let command_host_offset = command_page
        .checked_mul(HOST_PAGE_BYTES)
        .ok_or("command page offset overflow")?;
    let mut page_cursor = DATA_START;
    let mut input_cursor = 0u64;
    let mut output_cursor = 0u64;
    let mut input_slices = Vec::new();
    let mut output_slices = Vec::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    let control_read = HostTransfer {
        direction: HostDirection::ControlRead,
        physical_tile: 0,
        tile_address: ipu_exchange::EXCHANGE_WINDOW_BASE,
        host_offset: command_host_offset,
        bytes: 4,
    };
    let run = (!graph.schedule.phases.is_empty()).then_some(control_read);
    for binding in &graph.host_inputs {
        let size = binding_size(binding)?;
        let page_base = u64::from(align_up(page_cursor, 64));
        for slice in &binding.slices {
            let host_offset = u32::try_from(page_base + slice.file_offset)?;
            append_host_slices(
                &mut input_slices,
                host_offset,
                input_cursor + slice.file_offset,
                slice.size,
                HOST_PAGE_BYTES,
            )?;
            ipu_exchange::plan_host_to_tile(
                u16::try_from(slice.tile)?,
                slice.tile_address,
                host_offset,
                u32::try_from(slice.size)?,
            )?;
            inputs.push(control_read);
            inputs.push(HostTransfer {
                direction: HostDirection::ToTile,
                physical_tile: u16::try_from(slice.tile)?,
                tile_address: slice.tile_address,
                host_offset,
                bytes: u32::try_from(slice.size)?,
            });
        }
        page_cursor = u32::try_from(page_base + size)?;
        input_cursor = input_cursor
            .checked_add(size)
            .ok_or("host input size overflow")?;
    }
    page_cursor = output_start;
    for binding in &graph.host_outputs {
        let size = binding_size(binding)?;
        let page_base = u64::from(align_up(page_cursor, 64));
        for slice in &binding.slices {
            let host_offset = u32::try_from(page_base + slice.file_offset)?;
            append_host_slices(
                &mut output_slices,
                host_offset,
                output_cursor + slice.file_offset,
                slice.size,
                HOST_PAGE_BYTES,
            )?;
            ipu_exchange::plan_tile_to_host(
                u16::try_from(slice.tile)?,
                slice.tile_address,
                host_offset,
                u32::try_from(slice.size)?,
            )?;
            outputs.push(control_read);
            outputs.push(HostTransfer {
                direction: HostDirection::ToHost,
                physical_tile: u16::try_from(slice.tile)?,
                tile_address: slice.tile_address,
                host_offset,
                bytes: u32::try_from(slice.size)?,
            });
        }
        page_cursor = u32::try_from(page_base + size)?;
        output_cursor = output_cursor
            .checked_add(size)
            .ok_or("host output size overflow")?;
    }
    let operations = inputs.len() / 2 + usize::from(run.is_some()) + outputs.len() / 2;
    let mut calls = Vec::with_capacity(operations);
    for (index, commands) in inputs.chunks_exact(2).enumerate() {
        let transfer = commands[1];
        calls.push(HostCall {
            name: format!("host-input-{index}"),
            command: HOST_COMMAND_HOST_TO_TILE,
            phases: 3 + u32::from(index + 1 != operations),
            inputs: transfer_host_slices(&input_slices, transfer)?,
            outputs: Vec::new(),
        });
    }
    if run.is_some() {
        calls.push(HostCall {
            name: "device-run".into(),
            command: HOST_COMMAND_RUN,
            phases: 2,
            inputs: Vec::new(),
            outputs: Vec::new(),
        });
    }
    let input_operations = inputs.len() / 2 + usize::from(run.is_some());
    for (output_index, commands) in outputs.chunks_exact(2).enumerate() {
        let operation = input_operations + output_index;
        let transfer = commands[1];
        calls.push(HostCall {
            name: format!("host-output-{output_index}"),
            command: HOST_COMMAND_TILE_TO_HOST,
            phases: 3 + u32::from(operation + 1 != operations),
            inputs: Vec::new(),
            outputs: transfer_host_slices(&output_slices, transfer)?,
        });
    }
    Ok(HostLayout {
        inputs,
        run,
        outputs,
        protocol: HostExchange {
            startup_mark: ipu_driver::HOST_EXCHANGE_HANDOFF_MARK,
            command_page,
            command_offset: 0,
            pages: (0..=command_page)
                .map(|index| HostPage {
                    index,
                    size: u64::from(HOST_PAGE_BYTES),
                })
                .collect(),
            attach_order: std::iter::once(command_page)
                .chain(0..data_page_count)
                .collect(),
            calls,
        },
    })
}

fn transfer_host_slices(slices: &[HostSlice], transfer: HostTransfer) -> Result<Vec<HostSlice>> {
    let start = u64::from(transfer.host_offset);
    let end = start
        .checked_add(u64::from(transfer.bytes))
        .ok_or("host transfer range overflow")?;
    let selected = slices
        .iter()
        .filter(|slice| {
            let slice_start = u64::from(slice.page) * u64::from(ipu_exchange::HOST_PAGE_BYTES)
                + slice.page_offset;
            slice_start >= start && slice_start + slice.size <= end
        })
        .cloned()
        .collect::<Vec<_>>();
    if selected.iter().map(|slice| slice.size).sum::<u64>() != u64::from(transfer.bytes) {
        return Err("host slices do not cover generated transfer".into());
    }
    Ok(selected)
}

fn host_bindings_end(bindings: &[Binding], start: u32) -> Result<u32> {
    let mut cursor = start;
    for binding in bindings {
        let base = align_up(cursor, 64);
        cursor = u32::try_from(u64::from(base) + binding_size(binding)?)?;
    }
    Ok(cursor)
}

fn append_host_slices(
    slices: &mut Vec<HostSlice>,
    mut host_offset: u32,
    mut file_offset: u64,
    mut size: u64,
    page_bytes: u32,
) -> Result<()> {
    while size != 0 {
        let page = host_offset / page_bytes;
        let page_offset = host_offset % page_bytes;
        let count = size.min(u64::from(page_bytes - page_offset));
        slices.push(HostSlice {
            page,
            page_offset: u64::from(page_offset),
            file_offset,
            size: count,
        });
        host_offset = host_offset
            .checked_add(u32::try_from(count)?)
            .ok_or("host slice offset overflow")?;
        file_offset = file_offset
            .checked_add(count)
            .ok_or("host file offset overflow")?;
        size -= count;
    }
    Ok(())
}

fn binding_size(binding: &Binding) -> Result<u64> {
    binding
        .slices
        .iter()
        .try_fold(0u64, |size, slice| {
            slice
                .file_offset
                .checked_add(slice.size)
                .map(|end| size.max(end))
                .ok_or("binding size overflow")
        })
        .map_err(Into::into)
}

fn host_plan_size(transfer: &HostTransfer) -> Result<usize> {
    Ok(assemble_host_program(
        *transfer,
        sizing_host_role(transfer.direction),
        HOST_PACKET_BASE,
        ipu_exchange::EXCHANGE_WINDOW_BASE,
    )?
    .instructions
    .len()
        * 4)
}

fn host_packet_size(transfer: &HostTransfer) -> Result<usize> {
    Ok(assemble_host_program(
        *transfer,
        sizing_host_role(transfer.direction),
        HOST_PACKET_BASE,
        ipu_exchange::EXCHANGE_WINDOW_BASE,
    )?
    .packet_words
    .len()
        * 4)
}

fn sizing_host_role(direction: HostDirection) -> HostProgramRole {
    if matches!(direction, HostDirection::ToHost) {
        HostProgramRole::Source
    } else {
        HostProgramRole::Complete
    }
}

fn command_broadcast_rows(
    topology: &Topology,
    command_address: u32,
) -> Result<HashMap<u16, Vec<u32>>> {
    let receivers = (1..u16::try_from(topology.tile_count())?).collect::<Vec<_>>();
    let multicast = topology.multicast(0, &receivers, 1, 0)?;
    let mut rows = HashMap::with_capacity(receivers.len());
    for (logical, mut row) in receivers.into_iter().zip(multicast.receivers) {
        patch_multicast_receiver_address(&mut row, command_address)?;
        rows.insert(logical, row);
    }
    Ok(rows
        .into_iter()
        .map(|(tile, row)| (tile, row.into_iter().collect()))
        .collect())
}

fn append_command_guard(
    commands: &mut Vec<u8>,
    expected: u32,
    retry_address: u32,
    command_address: u32,
) {
    commands.extend_from_slice(&words_to_bytes(&[
        RuntimeRole::CommandGuard.word(),
        expected,
        retry_address,
        command_address,
        0,
    ]));
}

fn assemble_host_program(
    transfer: HostTransfer,
    role: HostProgramRole,
    packet_address: u32,
    command_address: u32,
) -> Result<TileToHostProgram> {
    match transfer.direction {
        HostDirection::ControlRead if role == HostProgramRole::Complete => {
            Ok(ipu_exchange::assemble_host_command_read_program(
                packet_address,
                command_address,
                transfer.host_offset,
            )?)
        }
        HostDirection::ToTile if role == HostProgramRole::Complete => {
            Ok(ipu_exchange::assemble_host_to_tile_program(
                transfer.physical_tile,
                transfer.tile_address,
                transfer.host_offset,
                transfer.bytes,
                packet_address,
            )?)
        }
        HostDirection::ToTile if role == HostProgramRole::Coordinator => {
            Ok(ipu_exchange::assemble_host_to_tile_coordinator_program(
                packet_address,
                ipu_exchange::plan_host_to_tile(
                    transfer.physical_tile,
                    transfer.tile_address,
                    transfer.host_offset,
                    transfer.bytes,
                )?
                .len(),
            )?)
        }
        HostDirection::ToTile if role == HostProgramRole::Source => {
            Ok(ipu_exchange::assemble_host_to_tile_receiver_program(
                transfer.physical_tile,
                transfer.tile_address,
                transfer.host_offset,
                transfer.bytes,
                packet_address,
            )?)
        }
        HostDirection::ToHost if role == HostProgramRole::Source => {
            Ok(ipu_exchange::assemble_tile_to_host_source_program(
                transfer.physical_tile,
                transfer.tile_address,
                transfer.host_offset,
                transfer.bytes,
                packet_address,
                command_address,
            )?)
        }
        _ => Err("host program role does not match transfer direction".into()),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HostProgramRole {
    Complete,
    Coordinator,
    Source,
}

struct HostCommandContext<'a> {
    layout: &'a RuntimeLayout,
    command_index: usize,
    barrier_after: bool,
    command_dispatch: bool,
    command_receiver: Option<&'a [u32]>,
    handler_follower: bool,
}

fn append_host_command(
    app: &mut Application,
    segments: &mut Vec<Segment>,
    commands: &mut Vec<u8>,
    transfer: HostTransfer,
    physical_tile: u16,
    context: HostCommandContext<'_>,
) -> Result<()> {
    let HostCommandContext {
        layout,
        command_index,
        barrier_after,
        command_dispatch,
        command_receiver,
        handler_follower,
    } = context;
    if command_dispatch && transfer.direction != HostDirection::ControlRead {
        return Err("command dispatch requires a control-read transfer".into());
    }
    const HOST_COORDINATOR_TILE: u16 = 0;
    let program_role = match transfer.direction {
        HostDirection::ControlRead => {
            (physical_tile == HOST_COORDINATOR_TILE).then_some(HostProgramRole::Complete)
        }
        HostDirection::ToTile if transfer.physical_tile == HOST_COORDINATOR_TILE => {
            (physical_tile == HOST_COORDINATOR_TILE).then_some(HostProgramRole::Complete)
        }
        HostDirection::ToTile if physical_tile == HOST_COORDINATOR_TILE => {
            Some(HostProgramRole::Coordinator)
        }
        HostDirection::ToTile if physical_tile == transfer.physical_tile => {
            Some(HostProgramRole::Source)
        }
        HostDirection::ToTile => None,
        HostDirection::ToHost if physical_tile == transfer.physical_tile => {
            Some(HostProgramRole::Source)
        }
        HostDirection::ToHost => None,
    };
    let (role, address) = if let Some(program_role) = program_role {
        let plan_address = layout.plan_address(command_index)?;
        let packet_address = layout.host_packet_address(command_index)?;
        let program = assemble_host_program(
            transfer,
            program_role,
            packet_address,
            layout.host_command_address,
        )?;
        let instructions = words_to_bytes(&program.instructions);
        let size = u32::try_from(instructions.len())?;
        let blob = app.add_blob(instructions);
        segments.push(Segment {
            address: plan_address,
            memory_size: size,
            blob,
            blob_offset: 0,
            file_size: size,
            flags: SEGMENT_READ | SEGMENT_EXECUTE,
        });
        let packets = words_to_bytes(&program.packet_words);
        let packet_size = u32::try_from(packets.len())?;
        let packet_blob = app.add_blob(packets);
        segments.push(Segment {
            address: packet_address,
            memory_size: packet_size,
            blob: packet_blob,
            blob_offset: 0,
            file_size: packet_size,
            flags: SEGMENT_READ,
        });
        (
            if command_dispatch {
                RuntimeRole::CommandCoordinator
            } else if barrier_after {
                RuntimeRole::HostProgram
            } else {
                RuntimeRole::HostProgramUnbarriered
            },
            plan_address,
        )
    } else if let Some(receiver) = command_receiver {
        let plan_address = layout.plan_address(command_index)?;
        let instructions = words_to_bytes(receiver);
        let size = u32::try_from(instructions.len())?;
        let blob = app.add_blob(instructions);
        segments.push(Segment {
            address: plan_address,
            memory_size: size,
            blob,
            blob_offset: 0,
            file_size: size,
            flags: SEGMENT_READ | SEGMENT_EXECUTE,
        });
        (RuntimeRole::CommandFollower, plan_address)
    } else {
        (
            if command_dispatch {
                return Err("noncoordinator command dispatch has no receiver row".into());
            } else if handler_follower {
                RuntimeRole::CommandHandlerFollower
            } else if barrier_after {
                RuntimeRole::HostIdle
            } else {
                RuntimeRole::HostIdleUnbarriered
            },
            0,
        )
    };
    commands.extend_from_slice(&words_to_bytes(&[role.word(), address, 0, 0, 0]));
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
    let first = &calls[0];
    session.prepare(&first.name, call_input(first, input)?)?;
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
    for (index, call) in calls.iter().enumerate() {
        if index != 0 {
            session.prepare(&call.name, call_input(call, input)?)?;
        }
        let call_output = match session.invoke_prepared(&call.name) {
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
    while device.tile_context_state(completion.tile as u16, 0)? != 3 {
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
    device.write_config(
        ipu_driver::pci::EXCHANGE_WINDOW_BASE,
        ipu_driver::pci::EXCHANGE_WINDOW_HEXOPT,
    )?;
    device.write_config(ipu_driver::pci::HSP_GS2_CONTROL, entry.external_syncs)?;
    device.set_mark(1)?;
    let deadline = Instant::now() + Duration::from_secs(10);
    while device.tile_context_state(completion.tile as u16, 0)? != 3 {
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
    Ok(DiagnosticResults { bindings })
}

fn add_global_sync_segments(
    app: &mut Application,
    segments: &mut Vec<Segment>,
    global_sync: GlobalSyncProgram,
    layout: RuntimeLayout,
) {
    for (address, bytes) in [
        (
            layout.sync_packet_address,
            words_to_bytes(&global_sync.packet_words),
        ),
        (
            layout.sync_release_address,
            words_to_bytes(&[global_sync.release_word]),
        ),
    ] {
        let blob = app.add_blob(bytes.clone());
        segments.push(Segment {
            address,
            memory_size: bytes.len() as u32,
            blob,
            blob_offset: 0,
            file_size: bytes.len() as u32,
            flags: SEGMENT_READ,
        });
    }
}

fn home_address(schedule: &Schedule, tensor: ipu_compiler::TensorId, tile: u16) -> Result<u32> {
    schedule
        .allocations
        .iter()
        .find(|allocation| {
            allocation.tensor == tensor
                && allocation.tile == tile
                && allocation.kind == AllocationKind::Home
        })
        .map(|allocation| allocation.address)
        .ok_or_else(|| format!("missing home tensor {} on tile {tile}", tensor.0).into())
}

fn compute_input_address(
    schedule: &Schedule,
    tensor: ipu_compiler::TensorId,
    tile: u16,
    phase: usize,
) -> Result<u32> {
    schedule
        .allocations
        .iter()
        .find(|allocation| {
            allocation.tensor == tensor
                && allocation.tile == tile
                && allocation.live_until == phase
                && matches!(allocation.kind, AllocationKind::ExchangeStaging { .. })
        })
        .map(|allocation| allocation.address)
        .map_or_else(|| home_address(schedule, tensor, tile), Ok)
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

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

const fn align_up(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

const fn ranges_overlap(left_start: u32, left_end: u32, right_start: u32, right_end: u32) -> bool {
    left_start < right_end && right_start < left_end
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_graph() -> ExecutableGraph {
        ExecutableGraph {
            schedule: Schedule {
                layouts: Vec::new(),
                phases: Vec::new(),
                allocations: Vec::new(),
                tile_count: 1472,
                peak_sram: BTreeMap::new(),
            },
            initial_buffers: Vec::new(),
            outputs: Vec::new(),
            host_inputs: Vec::new(),
            host_outputs: Vec::new(),
        }
    }

    fn test_binding(name: &str, slices: Vec<RegionSlice>) -> Binding {
        Binding {
            name: name.into(),
            dtype: "u32".into(),
            shape: vec![slices.iter().map(|slice| slice.size).sum::<u64>() as u32 / 4],
            slices,
        }
    }

    #[test]
    fn host_layout_separates_input_and_output_storage() {
        let mut graph = empty_graph();
        graph.host_inputs.push(test_binding(
            "input",
            vec![RegionSlice {
                tile: 63,
                tile_address: 0x53000,
                file_offset: 0,
                size: 64,
            }],
        ));
        graph.host_outputs.push(test_binding(
            "output",
            vec![
                RegionSlice {
                    tile: 2,
                    tile_address: 0x54000,
                    file_offset: 0,
                    size: 64,
                },
                RegionSlice {
                    tile: 63,
                    tile_address: 0x54000,
                    file_offset: 64,
                    size: 64,
                },
            ],
        ));
        let layout = build_host_layout(&graph).unwrap();
        assert!(layout.inputs.iter().any(|transfer| {
            matches!(transfer.direction, HostDirection::ToTile) && transfer.physical_tile == 63
        }));
        assert_eq!(
            layout
                .outputs
                .iter()
                .filter(|transfer| matches!(transfer.direction, HostDirection::ToHost))
                .count(),
            2
        );
        let input = layout
            .inputs
            .iter()
            .find(|transfer| matches!(transfer.direction, HostDirection::ToTile))
            .unwrap();
        assert!(layout.outputs.iter().all(|output| {
            !matches!(output.direction, HostDirection::ToHost)
                || !ranges_overlap(
                    input.host_offset,
                    input.host_offset + input.bytes,
                    output.host_offset,
                    output.host_offset + output.bytes,
                )
        }));
    }

    #[test]
    fn host_layout_pages_large_bindings_without_crossing_packets() {
        let binding = Binding {
            name: "payload".into(),
            dtype: "u8".into(),
            shape: vec![4096],
            slices: vec![RegionSlice {
                tile: 0,
                tile_address: 0x53000,
                file_offset: 0,
                size: 4096,
            }],
        };
        let graph = ExecutableGraph {
            schedule: Schedule {
                layouts: Vec::new(),
                phases: Vec::new(),
                allocations: Vec::new(),
                tile_count: 1472,
                peak_sram: BTreeMap::new(),
            },
            initial_buffers: Vec::new(),
            outputs: Vec::new(),
            host_inputs: vec![binding.clone()],
            host_outputs: vec![binding],
        };

        let layout = build_host_layout(&graph).unwrap();
        let calls = &layout.protocol.calls;
        assert_eq!(layout.inputs.len() % 2, 0);
        assert_eq!(layout.outputs.len() % 2, 0);
        assert!(layout.inputs.chunks_exact(2).all(|commands| {
            matches!(commands[0].direction, HostDirection::ControlRead)
                && matches!(commands[1].direction, HostDirection::ToTile)
        }));
        assert!(layout.outputs.chunks_exact(2).all(|commands| {
            matches!(commands[0].direction, HostDirection::ControlRead)
                && matches!(commands[1].direction, HostDirection::ToHost)
        }));
        assert!(
            layout
                .protocol
                .pages
                .iter()
                .all(|page| page.size == u64::from(ipu_exchange::HOST_PAGE_BYTES))
        );
        assert_eq!(
            calls
                .iter()
                .flat_map(|call| &call.inputs)
                .map(|slice| slice.size)
                .sum::<u64>(),
            4096
        );
        assert_eq!(
            calls
                .iter()
                .flat_map(|call| &call.outputs)
                .map(|slice| slice.size)
                .sum::<u64>(),
            4096
        );
        assert!(calls.iter().all(|call| {
            call.inputs.iter().chain(&call.outputs).all(|slice| {
                slice.page != layout.protocol.command_page
                    && slice.page_offset + slice.size <= u64::from(ipu_exchange::HOST_PAGE_BYTES)
            })
        }));
        assert!(layout.inputs.iter().chain(&layout.outputs).all(|transfer| {
            let chunks = match transfer.direction {
                HostDirection::ControlRead | HostDirection::ToTile => {
                    ipu_exchange::plan_host_to_tile(
                        transfer.physical_tile,
                        transfer.tile_address,
                        transfer.host_offset,
                        transfer.bytes,
                    )
                }
                HostDirection::ToHost => ipu_exchange::plan_tile_to_host(
                    transfer.physical_tile,
                    transfer.tile_address,
                    transfer.host_offset,
                    transfer.bytes,
                ),
            }
            .unwrap();
            chunks.iter().all(|chunk| {
                chunk.host_offset / ipu_exchange::HOST_PAGE_BYTES
                    == (chunk.host_offset + chunk.bytes - 1) / ipu_exchange::HOST_PAGE_BYTES
            })
        }));
        assert_eq!(calls[0].command, HOST_COMMAND_HOST_TO_TILE);
        assert_eq!(calls[0].phases, 4);
        assert_eq!(calls[1].command, HOST_COMMAND_TILE_TO_HOST);
        assert_eq!(calls[1].phases, 3);
        let output = layout
            .outputs
            .iter()
            .find(|transfer| matches!(transfer.direction, HostDirection::ToHost))
            .unwrap();
        assert!(host_packet_size(output).unwrap() > HOST_PACKET_ALIGNMENT as usize);
    }
}
