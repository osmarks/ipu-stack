use ipu_compiler::{AllocationKind, Phase, Schedule};
use ipu_driver::{Device, Loader, block_device_interrupt_signals};
use ipu_elf::{LinkOptions, link};
use ipu_exchange::{GlobalSyncProgram, Topology};
use ipu_package::{
    Application, Binding, DeviceConfigWrite, EntryPoint, RegionSlice, SEGMENT_EXECUTE,
    SEGMENT_READ, SEGMENT_WRITE, Segment, TileImage,
};
use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};
use tracing::{debug, info};

const COMMAND_WORDS: usize = 5;
const COMMAND_BYTES: u32 = (COMMAND_WORDS * 4) as u32;
const PLAN_BASE: u32 = ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES;
const TILE_MUX_HOST_BASE: u32 = 0x600;

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
}

impl RuntimeRole {
    const fn word(self) -> u32 {
        self as u32
    }
}

#[derive(Clone, Copy)]
struct RuntimeLayout {
    plan_stride: u32,
    command_address: u32,
    cycle_address: u32,
    completion_address: u32,
    sync_packet_address: u32,
    sync_release_address: u32,
}

impl RuntimeLayout {
    fn new(schedule: &Schedule, launch_rows: &[usize], command_count: usize) -> Result<Self> {
        let plan_stride = align_up(
            u32::try_from(launch_rows.iter().copied().max().unwrap_or(1))? * 4,
            8,
        );
        let plan_end = PLAN_BASE
            .checked_add(
                u32::try_from(launch_rows.len())?
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
        let cycle_address = align_up(command_end, 64);
        let completion_address = cycle_address + 4;
        let sync_packet_address = align_up(completion_address + 4, 8);
        let sync_release_address = sync_packet_address + 16;
        let runtime_end = sync_release_address + 4;

        for allocation in &schedule.allocations {
            let allocation_end = allocation
                .address
                .checked_add(allocation.size)
                .ok_or("allocation address overflow")?;
            if ranges_overlap(PLAN_BASE, runtime_end, allocation.address, allocation_end) {
                return Err(format!(
                    "runtime region overlaps tensor {} on tile {}",
                    allocation.tensor.0, allocation.tile
                )
                .into());
            }
        }
        Ok(Self {
            plan_stride,
            command_address,
            cycle_address,
            completion_address,
            sync_packet_address,
            sync_release_address,
        })
    }

    fn plan_address(self, launch: usize) -> Result<u32> {
        Ok(PLAN_BASE
            + u32::try_from(launch)?
                .checked_mul(self.plan_stride)
                .ok_or("plan address overflow")?)
    }
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
    let launch_rows = lowered
        .iter()
        .flat_map(|phase| phase.epochs.iter())
        .map(|epoch| epoch.tile_rows.values().map(Vec::len).max().unwrap_or(1))
        .collect::<Vec<_>>();
    let command_count = graph.schedule.phases.len() + 1;
    let layout = RuntimeLayout::new(&graph.schedule, &launch_rows, command_count)?;
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
    let cycle_start_offset = symbol_offset("ipu_stack_cycle_start_address")?;
    let cycle_end_offset = symbol_offset("ipu_stack_cycle_end_address")?;
    let completion_offset = symbol_offset("ipu_stack_completion_address")?;
    let completion_dispatch_offset = symbol_offset("ipu_stack_completion_dispatch")?;
    let nonmaster_completion_offset = symbol_offset("ipu_stack_nonmaster_completion_redirect")?;
    let pre_sync_offset = symbol_offset("ipu_stack_pre_sync_dispatch")?;
    let nonmaster_redirect_offset = symbol_offset("ipu_stack_nonmaster_pre_sync_redirect")?;
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
        patch_setzi_immediate(&mut code, cycle_start_offset, layout.cycle_address)?;
        patch_setzi_immediate(&mut code, cycle_end_offset, layout.cycle_address)?;
        patch_setzi_immediate(&mut code, completion_offset, layout.completion_address)?;
        if physical == u32::from(global_sync.packet_origin_physical_tile) {
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
        } else {
            copy_instruction(&mut code, nonmaster_redirect_offset, pre_sync_offset);
            copy_instruction(
                &mut code,
                nonmaster_completion_offset,
                completion_dispatch_offset,
            );
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
        let mut launch = 0usize;
        for (phase_index, phase) in graph.schedule.phases.iter().enumerate() {
            match phase {
                Phase::Exchange { .. } => {
                    let lowered = exchange_by_phase
                        .get(&phase_index)
                        .ok_or("missing lowered exchange phase")?;
                    if lowered.epochs.len() != 1 {
                        return Err(
                            "runtime expects one statically scheduled launch per exchange phase"
                                .into(),
                        );
                    }
                    let epoch = &lowered.epochs[0];
                    let (role, plan_address) = if let Some(row) = epoch.tile_rows.get(&logical) {
                        let address = layout.plan_address(launch)?;
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
                    launch += 1;
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
        }
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
    app.outputs = graph.outputs.clone();
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
    let launches = lowered
        .iter()
        .map(|phase| phase.epochs.len() as u32)
        .sum::<u32>();
    app.entry_points.push(EntryPoint {
        name: "graph".into(),
        command: 0,
        external_syncs: 1 + launches * 2,
    });
    app.validate()?;
    info!(
        tiles = app.tiles.len(),
        launches, "packaged executable graph"
    );
    Ok(app)
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
    for write in &app.device_config_writes {
        device.write_config(write.offset, write.value)?;
    }
    Loader::new(&device, bootloader)?.load(app, 0)?;
    device.write_config(
        ipu_driver::pci::EXCHANGE_WINDOW_BASE,
        ipu_driver::pci::EXCHANGE_WINDOW_HEXOPT,
    )?;
    device.write_config(ipu_driver::pci::HSP_GS2_CONTROL, entry.external_syncs)?;
    device.set_mark(1)?;
    let deadline = Instant::now() + Duration::from_secs(10);
    while device.tile_context_state(completion.tile as u16, 0)? != 3 {
        if Instant::now() >= deadline {
            return Err("diagnostic graph execution timed out".into());
        }
        std::thread::sleep(Duration::from_micros(100));
    }
    if device.read_tile_word(completion.tile as u16, completion.tile_address)? != 1 {
        return Err("coordinator trapped before recording completion".into());
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

fn copy_instruction(code: &mut [u8], source: usize, destination: usize) {
    let instruction = code[source..source + 4].to_vec();
    code[destination..destination + 4].copy_from_slice(&instruction);
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
