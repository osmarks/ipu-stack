use ipu_compiler::Schedule;
use ipu_driver::{Device, DriverError, HostSession, Loader, block_device_interrupt_signals};
use ipu_elf::{LinkOptions, link};
use ipu_exchange::Topology;
use ipu_package::{
    Application, Binding, EntryPoint, HostCall, HostExchange, HostPage, HostSlice, MemoryProfile,
    MemoryRegion, RegionSlice, SEGMENT_EXECUTE, SEGMENT_READ, SEGMENT_WRITE, Segment, TileImage,
    TileMemory,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;
use std::time::{Duration, Instant};
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;

mod siglip;
mod static_codegen;
pub use siglip::{
    HostTensorSet, SiglipEncoderLayer, SiglipMapHead, SiglipWeightStorage, append_host_a16_matrix,
    append_siglip_encoder_layer, append_siglip_map_head, append_siglip_post_layer_norm,
};

mod blocked_data;
pub use blocked_data::{
    BlockLayout, block_binding, block_binding_typed, block_coordinates, blocked_matrix,
    blocked_matrix_f8_f143, blocked_matrix_f16, f143_from_f32, f143_scale, f143_to_f32, normal_f16,
};

const PLAN_BASE: u32 = ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES;
const HOST_DATA_START: u32 = 64;
const HOST_CLOSE_ADDRESS: u32 = ipu_exchange::EXCHANGE_WINDOW_BASE + 0x160;
const HOST_PACKET_ADDRESS: u32 = ipu_exchange::EXCHANGE_WINDOW_BASE;
const HOST_STAGING_SEARCH_BASE: u32 = ipu_exchange::EXCHANGE_WINDOW_BASE + 0x180;
const HOST_RUN_DESCRIPTOR_WORDS: u32 = 5;
const WORKER_STACK_HEADROOM: u32 = 0xe0;
const WORKER_SYNC_STRIDE: u32 = 0x100;
const WORKER_SYNC_REGISTERS: u32 = 7;

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
    copy_destination: Option<u32>,
}

struct StaticHostLayout {
    inputs: Vec<StaticHostTransfer>,
    outputs: Vec<StaticHostTransfer>,
    protocol: HostExchange,
}

struct TileHostPlans {
    start: u32,
    ordinary_objects: Vec<Range<u32>>,
    data_objects: Vec<Range<u32>>,
    addresses: Vec<u32>,
    packet_copies: Vec<Option<HostPacketCopy>>,
    run_tables: Vec<Option<u32>>,
    run_state: u32,
    end: u32,
}

struct TileExchangePlans {
    addresses: Vec<u32>,
    rows: Vec<(u32, Vec<u32>)>,
    compute_runs: Vec<static_codegen::ExchangeComputeRun>,
    templates: Vec<static_codegen::StaticTemplatePlan>,
    end: u32,
}

#[derive(Clone, Copy)]
struct HostPacketCopy {
    source: u32,
    destination: u32,
    words: u32,
}

const ENTRY_TRAMPOLINE_BYTES: u32 = 8;

#[cfg(test)]
fn executable_region_base(
    graph: &ExecutableGraph,
    runtime_end: u32,
    required_size: u32,
) -> Result<u32> {
    let allocations = graph
        .schedule
        .allocations
        .iter()
        .map(allocation_range)
        .collect::<Result<Vec<_>>>()?;
    executable_region_base_for_tile(&allocations, None, runtime_end, required_size, &[])
}

fn executable_region_base_for_tile(
    allocation_ranges: &[(u32, u32)],
    tile: Option<u16>,
    runtime_end: u32,
    required_size: u32,
    additional_reserved: &[(u32, u32)],
) -> Result<u32> {
    let element_size = ipu_package::TILE_MEMORY_ELEMENT_SIZE;
    let memory_end = ipu_package::TILE_MEMORY_BASE
        .checked_add(ipu_package::TILE_MEMORY_SIZE)
        .ok_or("tile memory range overflow")?;
    let mut reserved = vec![
        (
            ipu_package::TILE_MEMORY_BASE,
            ipu_driver::APPLICATION_LOAD_BASE + ENTRY_TRAMPOLINE_BYTES,
        ),
        (
            ipu_exchange::EXCHANGE_WINDOW_BASE,
            ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES,
        ),
        (
            ipu_package::IPU21_INTERLEAVED_MEMORY_BASE,
            ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT,
        ),
        (PLAN_BASE, runtime_end),
    ];
    reserved.extend_from_slice(allocation_ranges);
    reserved.extend_from_slice(additional_reserved);
    let mut reserved = reserved
        .into_iter()
        .map(|(start, end)| (align_down(start, element_size), align_up(end, element_size)))
        .collect::<Vec<_>>();
    reserved.sort_unstable();

    let mut cursor = align_up(
        ipu_driver::APPLICATION_LOAD_BASE + ENTRY_TRAMPOLINE_BYTES,
        element_size,
    );
    let required_size = align_up(required_size, element_size);
    for (start, end) in reserved {
        if end <= cursor || start >= memory_end {
            continue;
        }
        let candidate_end = cursor
            .checked_add(required_size)
            .ok_or("executable region address overflow")?;
        if candidate_end <= start {
            return Ok(cursor);
        }
        cursor = align_up(cursor.max(end), element_size);
    }
    if cursor
        .checked_add(required_size)
        .is_some_and(|end| end <= memory_end)
    {
        Ok(cursor)
    } else {
        Err(format!(
            "no {}tile-memory interval can hold {required_size} bytes of executable code",
            tile.map_or("common ".into(), |tile| format!("tile {tile} "))
        )
        .into())
    }
}

fn executable_region_bases_for_tile(
    allocation_ranges: &[(u32, u32)],
    tile: u16,
    runtime_end: u32,
    image_size: u32,
    program_size: u32,
    additional_reserved: &[(u32, u32)],
) -> Result<(u32, u32)> {
    let place = |size, reserved: &[(u32, u32)]| {
        let mut all_reserved = additional_reserved.to_vec();
        all_reserved.extend_from_slice(reserved);
        executable_region_base_for_tile(
            allocation_ranges,
            Some(tile),
            runtime_end,
            size,
            &all_reserved,
        )
    };
    if program_size >= image_size {
        let program_base = place(program_size, &[])?;
        let program_end = program_base
            .checked_add(program_size)
            .ok_or("generated program address overflow")?;
        let image_base = place(image_size, &[(program_base, program_end)])?;
        Ok((image_base, program_base))
    } else {
        let image_base = place(image_size, &[])?;
        let image_end = image_base
            .checked_add(image_size)
            .ok_or("linked image address overflow")?;
        let program_base = place(program_size, &[(image_base, image_end)])?;
        Ok((image_base, program_base))
    }
}

fn data_region_base_for_tile(
    allocation_ranges: &[(u32, u32)],
    tile: u16,
    runtime_end: u32,
    required_size: u32,
    alignment: u32,
    additional_reserved: &[(u32, u32)],
) -> Result<u32> {
    if required_size == 0 || !alignment.is_power_of_two() {
        return Err("data region size must be nonzero and alignment must be a power of two".into());
    }
    let memory_end = ipu_package::TILE_MEMORY_BASE
        .checked_add(ipu_package::TILE_MEMORY_SIZE)
        .ok_or("tile memory range overflow")?;
    let mut reserved = vec![
        (
            ipu_package::TILE_MEMORY_BASE,
            ipu_driver::APPLICATION_LOAD_BASE + ENTRY_TRAMPOLINE_BYTES,
        ),
        (
            ipu_exchange::EXCHANGE_WINDOW_BASE,
            ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES,
        ),
        (PLAN_BASE, runtime_end),
    ];
    reserved.extend_from_slice(allocation_ranges);
    reserved.extend_from_slice(additional_reserved);
    reserved.sort_unstable();

    let mut cursor = align_up(ipu_package::TILE_MEMORY_BASE, alignment);
    let mut free_bytes = 0u32;
    let mut largest_gap = 0u32;
    for (start, end) in reserved {
        if end <= cursor || start >= memory_end {
            continue;
        }
        let gap = start.saturating_sub(cursor);
        free_bytes = free_bytes.saturating_add(gap);
        largest_gap = largest_gap.max(gap);
        let candidate_end = cursor
            .checked_add(required_size)
            .ok_or("data region address overflow")?;
        if candidate_end <= start {
            return Ok(cursor);
        }
        cursor = align_up(cursor.max(end), alignment);
    }
    let final_gap = memory_end.saturating_sub(cursor);
    free_bytes = free_bytes.saturating_add(final_gap);
    largest_gap = largest_gap.max(final_gap);
    if cursor
        .checked_add(required_size)
        .is_some_and(|end| end <= memory_end)
    {
        Ok(cursor)
    } else {
        Err(format!(
            "no tile-memory interval can hold {required_size} bytes of static data on tile {tile}: {free_bytes} free bytes, {largest_gap}-byte largest gap"
        )
        .into())
    }
}

fn pack_data_objects_for_tile(
    allocation_ranges: &[(u32, u32)],
    tile: u16,
    runtime_end: u32,
    objects: &[Range<u32>],
    allow_interleaved: bool,
    additional_reserved: &[(u32, u32)],
) -> Result<(BTreeMap<u32, u32>, Vec<(u32, u32)>)> {
    let memory_end = ipu_package::TILE_MEMORY_BASE
        .checked_add(ipu_package::TILE_MEMORY_SIZE)
        .ok_or("tile memory range overflow")?;
    let mut reserved = vec![
        (
            ipu_package::TILE_MEMORY_BASE,
            ipu_driver::APPLICATION_LOAD_BASE + ENTRY_TRAMPOLINE_BYTES,
        ),
        (
            ipu_exchange::EXCHANGE_WINDOW_BASE,
            ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES,
        ),
        (PLAN_BASE, runtime_end),
    ];
    if !allow_interleaved {
        reserved.push((
            ipu_package::IPU21_INTERLEAVED_MEMORY_BASE,
            ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT,
        ));
    }
    reserved.extend_from_slice(allocation_ranges);
    reserved.extend_from_slice(additional_reserved);
    reserved.sort_unstable();
    let mut gaps = Vec::new();
    let mut cursor = ipu_package::TILE_MEMORY_BASE;
    for (start, end) in reserved {
        if end <= cursor || start >= memory_end {
            continue;
        }
        let gap_start = align_up(cursor, 8);
        if gap_start < start {
            gaps.push((gap_start, start));
        }
        cursor = cursor.max(end);
    }
    let gap_start = align_up(cursor, 8);
    if gap_start < memory_end {
        gaps.push((gap_start, memory_end));
    }

    let mut order = (0..objects.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|&index| std::cmp::Reverse(objects[index].len()));
    let mut relocations = BTreeMap::new();
    let mut placed = Vec::with_capacity(objects.len());
    for index in order {
        let object = &objects[index];
        let size = u32::try_from(object.len())?;
        let (gap_index, address) = gaps
            .iter()
            .enumerate()
            .filter_map(|(gap_index, &(start, end))| {
                let address = align_up(start, 8);
                let remaining = end.checked_sub(address.checked_add(size)?)?;
                Some((gap_index, address, remaining))
            })
            .min_by_key(|&(_, _, remaining)| remaining)
            .map(|(gap, address, _)| (gap, address))
            .ok_or_else(|| {
                let free = gaps.iter().map(|(start, end)| end - start).sum::<u32>();
                let largest = gaps
                    .iter()
                    .map(|(start, end)| end - start)
                    .max()
                    .unwrap_or(0);
                format!(
                    "no tile-memory gap can hold a {size}-byte static data object on tile {tile}: {free} free bytes, {largest}-byte largest gap"
                )
            })?;
        gaps[gap_index].0 = address + size;
        relocations.insert(object.start, address);
        placed.push((address, address + size));
    }
    placed.sort_unstable();
    let mut merged = Vec::<(u32, u32)>::new();
    for (start, end) in placed {
        if let Some(previous) = merged.last_mut()
            && start <= previous.1
        {
            previous.1 = previous.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    Ok((relocations, merged))
}

fn allocation_range(allocation: &ipu_compiler::Allocation) -> Result<(u32, u32)> {
    Ok((
        allocation.address,
        allocation
            .address
            .checked_add(allocation.size)
            .ok_or("allocation address overflow")?,
    ))
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
pub struct StaticTemplateRegion {
    pub name: String,
    pub phase_instances: Vec<Range<usize>>,
}

#[derive(Clone, Debug)]
pub struct ProfileTileLayout {
    pub physical_tile: u32,
    pub file_offset: usize,
    pub steps: Vec<ipu_package::ProfileStep>,
    pub boundary_samples: bool,
}

#[derive(Clone, Debug)]
pub struct ProfileLayout {
    pub output_offset: usize,
    pub tiles: Vec<ProfileTileLayout>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileGranularity {
    Graph,
    Phase,
    Step,
}

impl ProfileGranularity {
    pub fn from_environment() -> Result<Self> {
        if std::env::var_os("IPU_PROFILE_AGGREGATE").is_some() {
            return Ok(Self::Graph);
        }
        match std::env::var("IPU_PROFILE_GRANULARITY")
            .unwrap_or_else(|_| "phase".into())
            .as_str()
        {
            "graph" => Ok(Self::Graph),
            "phase" => Ok(Self::Phase),
            "step" => Ok(Self::Step),
            value => Err(format!(
                "IPU_PROFILE_GRANULARITY must be graph, phase, or step; got {value:?}"
            )
            .into()),
        }
    }
}

impl ProfileLayout {
    pub fn decode(&self, output: &[u8], clock_hz: u64) -> Result<ipu_package::ProfileReport> {
        let profile = output
            .get(self.output_offset..)
            .ok_or("profile output offset exceeds host result")?;
        let mut tiles = Vec::with_capacity(self.tiles.len());
        for tile in &self.tiles {
            let size = if tile.boundary_samples {
                tile.steps
                    .len()
                    .checked_add(1)
                    .and_then(|count| count.checked_mul(4))
            } else {
                tile.steps.len().checked_mul(8)
            }
            .ok_or("profile sample size overflow")?;
            let bytes = profile
                .get(tile.file_offset..tile.file_offset + size)
                .ok_or("profile tile range exceeds host result")?;
            let samples = tile
                .steps
                .iter()
                .enumerate()
                .map(|(index, step)| {
                    let (start, end) = if tile.boundary_samples {
                        (index * 4, index * 4 + 4)
                    } else {
                        (index * 8, index * 8 + 4)
                    };
                    ipu_package::CycleSample {
                        step: step.clone(),
                        start_cycle: u32::from_le_bytes(
                            bytes[start..start + 4].try_into().unwrap(),
                        ),
                        end_cycle: u32::from_le_bytes(bytes[end..end + 4].try_into().unwrap()),
                    }
                })
                .collect();
            tiles.push(ipu_package::TileProfile {
                physical_tile: tile.physical_tile,
                samples,
            });
        }
        Ok(ipu_package::ProfileReport { clock_hz, tiles })
    }
}

#[derive(Clone, Debug)]
pub struct DiagnosticResults {
    pub bindings: BTreeMap<String, Vec<u32>>,
}

fn allocate_low_runtime_range(
    schedule: &Schedule,
    size: u32,
    physical_tiles: &BTreeSet<u16>,
    reserved: &[(u32, u32)],
) -> Result<u32> {
    let topology = Topology::c600();
    let limit = ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::HOST_TO_TILE_WINDOW_BYTES;
    let mut candidate = HOST_STAGING_SEARCH_BASE;
    while candidate < limit {
        let end = candidate
            .checked_add(size)
            .ok_or("low runtime allocation overflow")?;
        if end <= limit
            && schedule.allocations.iter().all(|allocation| {
                topology
                    .physical(allocation.tile)
                    .map(|physical_tile| {
                        !physical_tiles.contains(&physical_tile)
                            || allocation.kind != ipu_compiler::AllocationKind::Home
                            || !ranges_overlap(
                                candidate,
                                end,
                                allocation.address,
                                allocation.address.saturating_add(allocation.size),
                            )
                    })
                    .unwrap_or(false)
            })
            && reserved
                .iter()
                .all(|&(start, stop)| !ranges_overlap(candidate, end, start, stop))
        {
            return Ok(candidate);
        }
        candidate = candidate
            .checked_add(32)
            .ok_or("low runtime allocation overflow")?;
    }
    Err("no exchange-window range is available for host runtime data".into())
}

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Clone, Copy, Debug, Default)]
pub struct HostRunOptions {
    pub write_jitter: Option<HostWriteJitter>,
}

#[derive(Clone, Copy, Debug)]
pub struct HostWriteJitter {
    pub seed: u64,
    pub max_delay: Duration,
}

type HostInspector<'a> = dyn FnMut(&Device, &[u8]) -> Result<()> + 'a;

impl HostRunOptions {
    pub fn from_environment() -> Result<Self> {
        let Some(max_delay) = optional_environment_number("IPU_HOST_WRITE_JITTER_MAX_US")? else {
            return Ok(Self::default());
        };
        let seed = optional_environment_number("IPU_HOST_WRITE_JITTER_SEED")?
            .unwrap_or_else(|| fastrand::u64(..));
        Ok(Self {
            write_jitter: Some(HostWriteJitter {
                seed,
                max_delay: Duration::from_micros(max_delay),
            }),
        })
    }
}

fn optional_environment_number(name: &str) -> Result<Option<u64>> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(None);
    };
    let value = value
        .to_str()
        .ok_or_else(|| format!("{name} is not valid UTF-8"))?;
    let (digits, radix) = value
        .strip_prefix("0x")
        .map_or((value, 10), |digits| (digits, 16));
    Ok(Some(u64::from_str_radix(digits, radix).map_err(
        |error| format!("invalid {name} value {value:?}: {error}"),
    )?))
}

pub fn package_graph(graph: &ExecutableGraph, objects: &[Vec<u8>]) -> Result<Application> {
    package_graph_impl(graph, objects, &[], None, &[])
}

pub fn package_graph_with_templates(
    graph: &ExecutableGraph,
    objects: &[Vec<u8>],
    templates: &[StaticTemplateRegion],
) -> Result<Application> {
    package_graph_impl(graph, objects, &[], None, templates)
}

pub fn allocator_memory_profile(graph: &ExecutableGraph) -> Result<MemoryProfile> {
    let topology = Topology::c600();
    let mut binding_intervals = vec![Vec::<(u32, u32, &str)>::new(); topology.tile_count()];
    for binding in graph
        .host_inputs
        .iter()
        .chain(&graph.host_outputs)
        .chain(&graph.outputs)
    {
        for slice in &binding.slices {
            let tile = usize::try_from(slice.tile)?;
            let end = slice
                .tile_address
                .saturating_add(u32::try_from(slice.size).unwrap_or(u32::MAX));
            binding_intervals[tile].push((slice.tile_address, end, &binding.name));
        }
    }
    let binding_prefix_ends = binding_intervals
        .iter_mut()
        .map(|intervals| {
            intervals.sort_unstable_by_key(|&(start, end, _)| (start, end));
            let mut maximum = 0;
            intervals
                .iter()
                .map(|&(_, end, _)| {
                    maximum = maximum.max(end);
                    maximum
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut output_names = HashMap::<(u16, ipu_compiler::TensorId), String>::new();
    for phase in &graph.schedule.phases {
        let ipu_compiler::Phase::Compute { commands, .. } = phase else {
            continue;
        };
        for command in commands {
            output_names
                .entry((command.tile, command.output))
                .or_insert_with(|| allocation_command_name(command));
        }
    }
    let mut tiles = Vec::with_capacity(usize::from(graph.schedule.tile_count));
    for logical_tile in 0..graph.schedule.tile_count {
        let physical_tile = topology.physical(logical_tile)?;
        let mut regions = graph
            .schedule
            .allocations
            .iter()
            .filter(|allocation| allocation.tile == logical_tile)
            .map(|allocation| {
                let allocation_end = allocation.address.saturating_add(allocation.size);
                let intervals = &binding_intervals[usize::from(physical_tile)];
                let prefix_ends = &binding_prefix_ends[usize::from(physical_tile)];
                let mut interval =
                    intervals.partition_point(|&(start, _, _)| start < allocation_end);
                let mut names = Vec::new();
                while interval != 0 && prefix_ends[interval - 1] > allocation.address {
                    interval -= 1;
                    let (_, end, name) = intervals[interval];
                    if end > allocation.address && !names.contains(&name) {
                        names.push(name);
                    }
                }
                names.sort_unstable();
                let category = match allocation.kind {
                    ipu_compiler::AllocationKind::Home => "home",
                    ipu_compiler::AllocationKind::ExchangeStaging { .. } => "exchange_staging",
                };
                let name = allocation_profile_name(graph, allocation, &names, &output_names);
                MemoryRegion {
                    address: allocation.address,
                    size: allocation.size,
                    category: category.into(),
                    name,
                    tensor: Some(allocation.tensor.0),
                    live_from: allocation.live_from,
                    live_until: allocation.live_until,
                }
            })
            .collect::<Vec<_>>();
        regions.sort_by_key(|region| (region.address, region.live_from, region.live_until));
        tiles.push(TileMemory {
            logical_tile,
            physical_tile,
            regions,
        });
    }
    Ok(MemoryProfile {
        memory_base: ipu_package::TILE_MEMORY_BASE,
        memory_size: ipu_package::TILE_MEMORY_SIZE,
        tiles,
    })
}

fn allocation_command_name(command: &ipu_compiler::KernelCommand) -> String {
    let label = command
        .metadata
        .get("label")
        .cloned()
        .unwrap_or_else(|| command.specialization.operation.clone());
    if command.specialization.operation.starts_with("gemm_")
        && let (Some(row), Some(column)) = (
            command.metadata.get("output_block_row"),
            command.metadata.get("output_block_column"),
        )
    {
        format!("GEMM accumulator block ({row}, {column})")
    } else {
        format!("output of {label}")
    }
}

fn allocation_profile_name(
    graph: &ExecutableGraph,
    allocation: &ipu_compiler::Allocation,
    binding_names: &[&str],
    output_names: &HashMap<(u16, ipu_compiler::TensorId), String>,
) -> String {
    if !binding_names.is_empty() {
        return binding_names.join(", ");
    }
    if let Some(name) = output_names.get(&(allocation.tile, allocation.tensor)) {
        return name.clone();
    }
    if let ipu_compiler::AllocationKind::ExchangeStaging { phase } = allocation.kind
        && let Some(ipu_compiler::Phase::Compute { commands, .. }) =
            graph.schedule.phases.get(phase + 1)
        && let Some(command) = commands.iter().find(|command| {
            command.tile == allocation.tile && command.inputs.contains(&allocation.tensor)
        })
    {
        let label = command
            .metadata
            .get("label")
            .cloned()
            .unwrap_or_else(|| command.specialization.operation.clone());
        return format!("input to {label}");
    }
    match allocation.kind {
        ipu_compiler::AllocationKind::Home => format!("tensor {} home", allocation.tensor.0),
        ipu_compiler::AllocationKind::ExchangeStaging { phase } => {
            format!("tensor {} staging for phase {phase}", allocation.tensor.0)
        }
    }
}

pub fn package_graph_profiled(
    graph: &ExecutableGraph,
    objects: &[Vec<u8>],
) -> Result<(Application, ProfileLayout)> {
    package_graph_with_profile(graph, objects, ProfileGranularity::Step)
}

pub fn package_graph_timed(
    graph: &ExecutableGraph,
    objects: &[Vec<u8>],
) -> Result<(Application, ProfileLayout)> {
    package_graph_with_profile(graph, objects, ProfileGranularity::Graph)
}

pub fn package_graph_profiled_with(
    graph: &ExecutableGraph,
    objects: &[Vec<u8>],
    granularity: ProfileGranularity,
) -> Result<(Application, ProfileLayout)> {
    package_graph_with_profile(graph, objects, granularity)
}

fn profile_metadata(name: impl Into<String>, value: impl ToString) -> ipu_package::ProfileMetadata {
    ipu_package::ProfileMetadata {
        name: name.into(),
        value: value.to_string(),
    }
}

fn compute_profile_step(
    local_index: usize,
    command: &ipu_compiler::LoweredComputeCommand,
) -> ipu_package::ProfileStep {
    let kernel = command.specialization.operation.clone();
    let operation = command.metadata.get("label").cloned().unwrap_or_else(|| {
        if command.specialization.role.is_empty() {
            kernel.clone()
        } else {
            format!("{} ({})", kernel, command.specialization.role)
        }
    });
    let mut metadata = vec![
        profile_metadata("role", &command.specialization.role),
        profile_metadata(
            "shape",
            command
                .specialization
                .shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("x"),
        ),
        profile_metadata("worker_count", command.specialization.worker_count),
        profile_metadata("alignment", command.specialization.alignment),
        profile_metadata("output_tensor", command.output.0),
        profile_metadata("output_address", format!("0x{:x}", command.output_address)),
    ];
    for (index, (&tensor, &address)) in command
        .inputs
        .iter()
        .zip(&command.input_addresses)
        .enumerate()
    {
        metadata.push(profile_metadata(format!("input_{index}_tensor"), tensor.0));
        metadata.push(profile_metadata(
            format!("input_{index}_address"),
            format!("0x{address:x}"),
        ));
    }
    for (index, argument) in command.arguments.iter().enumerate() {
        metadata.push(profile_metadata(format!("argument_{index}"), argument));
    }
    metadata.extend(
        command
            .metadata
            .iter()
            .map(|(name, value)| profile_metadata(name, value)),
    );
    ipu_package::ProfileStep {
        local_index: local_index as u32,
        phase: command.phase as u32,
        epoch: 0,
        operation,
        kind: ipu_package::ProfileStepKind::Compute,
        kernel,
        metadata,
    }
}

fn idle_compute_profile_step(
    schedule: &ipu_compiler::Schedule,
    local_index: usize,
    op: ipu_compiler::OpId,
    phase: usize,
) -> ipu_package::ProfileStep {
    let commands = match &schedule.phases[phase] {
        ipu_compiler::Phase::Compute { commands, .. } => commands.as_slice(),
        ipu_compiler::Phase::Exchange { .. } => &[],
    };
    let first = commands.first();
    let (operation, metadata) = if let Some(command) = first {
        let mut common = command.metadata.clone();
        common.retain(|name, value| {
            name != "label"
                && commands
                    .iter()
                    .all(|candidate| candidate.metadata.get(name) == Some(value))
        });
        let mut metadata = vec![
            profile_metadata("active", "false"),
            profile_metadata("scheduled_kernel", &command.specialization.operation),
            profile_metadata("scheduled_role", &command.specialization.role),
            profile_metadata(
                "scheduled_shape",
                command
                    .specialization
                    .shape
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join("x"),
            ),
        ];
        metadata.extend(
            common
                .into_iter()
                .map(|(name, value)| profile_metadata(name, value)),
        );
        (
            format!(
                "idle during {} ({})",
                command.specialization.operation, command.specialization.role
            ),
            metadata,
        )
    } else {
        (
            format!("idle compute op {}", op.0),
            vec![profile_metadata("active", "false")],
        )
    };
    ipu_package::ProfileStep {
        local_index: local_index as u32,
        phase: phase as u32,
        epoch: 0,
        operation,
        kind: ipu_package::ProfileStepKind::Idle,
        kernel: String::new(),
        metadata,
    }
}

fn exchange_profile_step(
    schedule: &ipu_compiler::Schedule,
    tile: u16,
    local_index: usize,
    phase: usize,
    epoch: usize,
    active: bool,
) -> ipu_package::ProfileStep {
    let transfers = match &schedule.phases[phase] {
        ipu_compiler::Phase::Exchange { transfers } => transfers.as_slice(),
        ipu_compiler::Phase::Compute { .. } => &[],
    };
    let sends = transfers
        .iter()
        .filter(|transfer| transfer.source_tile == tile)
        .collect::<Vec<_>>();
    let receives = transfers
        .iter()
        .filter(|transfer| transfer.destination_tile == tile)
        .collect::<Vec<_>>();
    let next_command = schedule.phases.get(phase + 1).and_then(|next| match next {
        ipu_compiler::Phase::Compute { commands, .. } => commands
            .iter()
            .find(|command| command.tile == tile)
            .or_else(|| commands.first()),
        ipu_compiler::Phase::Exchange { .. } => None,
    });
    let operation = next_command.map_or_else(
        || format!("exchange phase {phase}"),
        |command| {
            format!(
                "exchange for {} ({})",
                command.specialization.operation, command.specialization.role
            )
        },
    );
    let mut metadata = vec![
        profile_metadata("active", active),
        profile_metadata("send_count", sends.len()),
        profile_metadata("receive_count", receives.len()),
        profile_metadata(
            "send_bytes",
            sends
                .iter()
                .map(|transfer| u64::from(transfer.bytes))
                .sum::<u64>(),
        ),
        profile_metadata(
            "receive_bytes",
            receives
                .iter()
                .map(|transfer| u64::from(transfer.bytes))
                .sum::<u64>(),
        ),
    ];
    for (index, transfer) in sends.iter().enumerate() {
        metadata.push(profile_metadata(
            format!("send_{index}"),
            format!(
                "tensor={},destination_tile={},bytes={}",
                transfer.tensor.0, transfer.destination_tile, transfer.bytes
            ),
        ));
    }
    for (index, transfer) in receives.iter().enumerate() {
        metadata.push(profile_metadata(
            format!("receive_{index}"),
            format!(
                "tensor={},source_tile={},bytes={}",
                transfer.tensor.0, transfer.source_tile, transfer.bytes
            ),
        ));
    }
    if let Some(command) = next_command {
        metadata.push(profile_metadata(
            "next_kernel",
            &command.specialization.operation,
        ));
        metadata.push(profile_metadata("next_role", &command.specialization.role));
        metadata.extend(
            command
                .metadata
                .iter()
                .filter(|(name, _)| name.as_str() != "label")
                .map(|(name, value)| profile_metadata(format!("next_{name}"), value)),
        );
    }
    ipu_package::ProfileStep {
        local_index: local_index as u32,
        phase: phase as u32,
        epoch: epoch as u32,
        operation,
        kind: ipu_package::ProfileStepKind::Exchange,
        kernel: String::new(),
        metadata,
    }
}

fn synchronization_profile_step(
    schedule: &ipu_compiler::Schedule,
    tile: u16,
    local_index: usize,
    phase: usize,
    epoch: usize,
    active: bool,
) -> ipu_package::ProfileStep {
    let exchange = exchange_profile_step(schedule, tile, local_index, phase, epoch, active);
    ipu_package::ProfileStep {
        operation: format!("sync before {}", exchange.operation),
        kind: ipu_package::ProfileStepKind::Synchronization,
        metadata: exchange
            .metadata
            .into_iter()
            .filter(|entry| {
                entry.name == "active"
                    || entry.name == "next_kernel"
                    || entry.name == "next_role"
                    || entry.name.starts_with("next_layer")
            })
            .collect(),
        ..exchange
    }
}

fn lowered_step_phase(step: &ipu_compiler::LoweredTileStep) -> usize {
    match step {
        ipu_compiler::LoweredTileStep::Exchange { phase, .. }
        | ipu_compiler::LoweredTileStep::IdleCompute { phase, .. } => *phase,
        ipu_compiler::LoweredTileStep::Compute(command) => command.phase,
    }
}

fn phase_compute_profile_step(
    schedule: &ipu_compiler::Schedule,
    tile: u16,
    local_index: usize,
    phase: usize,
) -> ipu_package::ProfileStep {
    match &schedule.phases[phase] {
        ipu_compiler::Phase::Exchange { .. } => unreachable!("expected a compute phase"),
        ipu_compiler::Phase::Compute { op, commands } => {
            let active_commands = commands
                .iter()
                .filter(|command| command.tile == tile)
                .collect::<Vec<_>>();
            let Some(command) = active_commands.first().copied() else {
                return idle_compute_profile_step(schedule, local_index, *op, phase);
            };
            let kernel = command.specialization.operation.clone();
            let operation = command
                .metadata
                .get("label")
                .cloned()
                .unwrap_or_else(|| kernel.clone());
            let mut metadata = vec![
                profile_metadata("role", &command.specialization.role),
                profile_metadata(
                    "shape",
                    command
                        .specialization
                        .shape
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join("x"),
                ),
                profile_metadata("worker_count", command.specialization.worker_count),
                profile_metadata("active_commands", active_commands.len()),
            ];
            metadata.extend(
                command
                    .metadata
                    .iter()
                    .filter(|(name, _)| name.as_str() != "label")
                    .map(|(name, value)| profile_metadata(name, value)),
            );
            ipu_package::ProfileStep {
                local_index: local_index as u32,
                phase: phase as u32,
                epoch: 0,
                operation,
                kind: ipu_package::ProfileStepKind::Compute,
                kernel,
                metadata,
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ProfileBoundary {
    after_sync: bool,
    after_step: bool,
}

fn compact_exchange_profile(step: &mut ipu_package::ProfileStep) {
    step.metadata.retain(|entry| {
        matches!(
            entry.name.as_str(),
            "active" | "send_count" | "receive_count" | "send_bytes" | "receive_bytes"
        ) || entry.name.starts_with("next_")
    });
}

fn append_exchange_profile(
    schedule: &ipu_compiler::Schedule,
    program: &ipu_compiler::LoweredTileProgram,
    steps: &mut Vec<ipu_package::ProfileStep>,
    boundaries: &mut [ProfileBoundary],
    step_index: usize,
    compact: bool,
) {
    let ipu_compiler::LoweredTileStep::Exchange { phase, epoch, row } = &program.steps[step_index]
    else {
        unreachable!("expected a lowered exchange step")
    };
    let active = row.first() != Some(&ipu_exchange::SANS_INACTIVE_INSTRUCTION);
    let local_index = steps.len();
    steps.push(synchronization_profile_step(
        schedule,
        program.tile,
        local_index,
        *phase,
        *epoch,
        active,
    ));
    boundaries[step_index].after_sync = true;
    let mut exchange = exchange_profile_step(
        schedule,
        program.tile,
        local_index + 1,
        *phase,
        *epoch,
        active,
    );
    if compact {
        compact_exchange_profile(&mut exchange);
    }
    steps.push(exchange);
    boundaries[step_index].after_step = true;
}

fn profile_steps(
    schedule: &ipu_compiler::Schedule,
    program: &ipu_compiler::LoweredTileProgram,
    granularity: ProfileGranularity,
) -> (Vec<ipu_package::ProfileStep>, Vec<ProfileBoundary>) {
    let mut steps = Vec::new();
    let mut boundaries = vec![ProfileBoundary::default(); program.steps.len()];

    match granularity {
        ProfileGranularity::Graph => {}
        ProfileGranularity::Step => {
            for (step_index, step) in program.steps.iter().enumerate() {
                match step {
                    ipu_compiler::LoweredTileStep::Exchange { .. } => {
                        append_exchange_profile(
                            schedule,
                            program,
                            &mut steps,
                            &mut boundaries,
                            step_index,
                            false,
                        );
                    }
                    ipu_compiler::LoweredTileStep::Compute(command) => {
                        steps.push(compute_profile_step(steps.len(), command));
                        boundaries[step_index].after_step = true;
                    }
                    ipu_compiler::LoweredTileStep::IdleCompute { op, phase } => {
                        steps.push(idle_compute_profile_step(
                            schedule,
                            steps.len(),
                            *op,
                            *phase,
                        ));
                        boundaries[step_index].after_step = true;
                    }
                }
            }
        }
        ProfileGranularity::Phase => {
            let mut start = 0usize;
            while start < program.steps.len() {
                let phase = lowered_step_phase(&program.steps[start]);
                let end = program.steps[start + 1..]
                    .iter()
                    .position(|step| lowered_step_phase(step) != phase)
                    .map_or(program.steps.len(), |offset| start + 1 + offset);
                match &program.steps[start] {
                    ipu_compiler::LoweredTileStep::Exchange { .. } => {
                        for (step_index, step) in program.steps[start..end].iter().enumerate() {
                            if !matches!(step, ipu_compiler::LoweredTileStep::Exchange { .. }) {
                                unreachable!("one phase lowered to mixed step kinds")
                            }
                            append_exchange_profile(
                                schedule,
                                program,
                                &mut steps,
                                &mut boundaries,
                                start + step_index,
                                true,
                            );
                        }
                    }
                    ipu_compiler::LoweredTileStep::Compute(_)
                    | ipu_compiler::LoweredTileStep::IdleCompute { .. } => {
                        steps.push(phase_compute_profile_step(
                            schedule,
                            program.tile,
                            steps.len(),
                            phase,
                        ));
                        boundaries[end - 1].after_step = true;
                    }
                }
                start = end;
            }
        }
    }
    (steps, boundaries)
}

fn package_graph_with_profile(
    graph: &ExecutableGraph,
    objects: &[Vec<u8>],
    granularity: ProfileGranularity,
) -> Result<(Application, ProfileLayout)> {
    let topology = Topology::c600();
    let programs = graph.schedule.lower_tile_programs(&topology)?;
    let output_offset = graph
        .host_outputs
        .iter()
        .try_fold(0u64, |offset, binding| {
            binding_size(binding).and_then(|size| {
                offset
                    .checked_add(size)
                    .ok_or_else(|| "profile output offset overflow".into())
            })
        })?;
    let mut profile_graph = graph.clone();
    let mut profile_code = Vec::with_capacity(programs.len());
    let mut profile_tiles = Vec::with_capacity(programs.len());
    let mut slices = Vec::with_capacity(programs.len());
    let mut file_offset = 0usize;
    let profile_tensor_base = profile_graph
        .schedule
        .allocations
        .iter()
        .map(|allocation| allocation.tensor.0)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or("profile tensor id overflow")?;
    for program in &programs {
        let (mut steps, boundaries) = profile_steps(&graph.schedule, program, granularity);
        if granularity == ProfileGranularity::Graph {
            steps.push(ipu_package::ProfileStep {
                local_index: 0,
                phase: 0,
                epoch: 0,
                operation: "graph".into(),
                kind: ipu_package::ProfileStepKind::Compute,
                kernel: String::new(),
                metadata: Vec::new(),
            });
        }
        let sample_count = if granularity == ProfileGranularity::Graph {
            1
        } else {
            steps
                .len()
                .checked_add(1)
                .ok_or("profile sample count overflow")?
        };
        let sample_bytes = if granularity == ProfileGranularity::Graph {
            8
        } else {
            4
        };
        let size = u32::try_from(
            sample_count
                .checked_mul(sample_bytes)
                .ok_or("profile size overflow")?,
        )?;
        if size == 0 {
            return Err("profile contains no sample storage".into());
        }
        let address = align_up(
            profile_graph
                .schedule
                .allocations
                .iter()
                .filter(|allocation| allocation.tile == program.tile)
                .map(|allocation| allocation.address.saturating_add(allocation.size))
                .max()
                .unwrap_or(PLAN_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES)
                .max(PLAN_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES),
            8,
        );
        if address
            .checked_add(size)
            .is_none_or(|end| end > ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE)
        {
            return Err(format!("profile samples exhaust SRAM on tile {}", program.tile).into());
        }
        let physical = u32::from(topology.physical(program.tile)?);
        profile_graph
            .schedule
            .allocations
            .push(ipu_compiler::Allocation {
                tensor: ipu_compiler::TensorId(profile_tensor_base + usize::from(program.tile)),
                tile: program.tile,
                address,
                size,
                live_from: 0,
                live_until: usize::MAX,
                kind: ipu_compiler::AllocationKind::Home,
            });
        let after_sync = boundaries
            .iter()
            .map(|boundary| boundary.after_sync)
            .collect::<Vec<_>>();
        let after_step = boundaries
            .iter()
            .map(|boundary| boundary.after_step)
            .collect::<Vec<_>>();
        if granularity != ProfileGranularity::Graph {
            let boundary_count = boundaries
                .iter()
                .map(|boundary| usize::from(boundary.after_sync) + usize::from(boundary.after_step))
                .sum::<usize>();
            debug_assert_eq!(boundary_count, steps.len());
        }
        profile_code.push(static_codegen::ProfileCode {
            initial: address,
            after_sync,
            after_step,
            aggregate_end: (granularity == ProfileGranularity::Graph).then_some(address + 4),
        });
        slices.push(RegionSlice {
            tile: physical,
            tile_address: address,
            file_offset: file_offset as u64,
            size: u64::from(size),
        });
        profile_tiles.push(ProfileTileLayout {
            physical_tile: physical,
            file_offset,
            steps,
            boundary_samples: granularity != ProfileGranularity::Graph,
        });
        file_offset += size as usize;
    }
    profile_graph.host_outputs.push(Binding {
        name: "runtime-profile".into(),
        dtype: "u32".into(),
        shape: vec![(file_offset / 4) as u32],
        slices,
    });
    let app = package_graph_impl(&profile_graph, objects, &profile_code, Some(programs), &[])?;
    Ok((
        app,
        ProfileLayout {
            output_offset: usize::try_from(output_offset)?,
            tiles: profile_tiles,
        },
    ))
}

fn package_graph_impl(
    graph: &ExecutableGraph,
    objects: &[Vec<u8>],
    profile_code: &[static_codegen::ProfileCode],
    lowered_programs: Option<Vec<ipu_compiler::LoweredTileProgram>>,
    template_regions: &[StaticTemplateRegion],
) -> Result<Application> {
    let topology = Topology::c600();
    if usize::from(graph.schedule.tile_count) != topology.tile_count() {
        return Err("the direct C600 runtime requires a schedule for every discovered tile".into());
    }
    let programs = match lowered_programs {
        Some(programs) => programs,
        None => graph.schedule.lower_tile_programs(&topology)?,
    };
    if !profile_code.is_empty() && profile_code.len() != programs.len() {
        return Err("profile layout tile count differs from schedule".into());
    }
    let mut allocation_ranges_by_tile = vec![Vec::new(); usize::from(graph.schedule.tile_count)];
    for allocation in &graph.schedule.allocations {
        let ranges = allocation_ranges_by_tile
            .get_mut(usize::from(allocation.tile))
            .ok_or("allocation tile exceeds schedule tile count")?;
        ranges.push(allocation_range(allocation)?);
    }
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
    let mut tile_exchange_plans = programs
        .iter()
        .map(|program| -> Result<TileExchangePlans> {
            let mut cursor = PLAN_BASE;
            let mut unique = HashMap::<Vec<u32>, u32>::new();
            let mut addresses = Vec::with_capacity(exchange_count);
            let mut rows = Vec::new();
            let mut patches = Vec::with_capacity(exchange_count);
            for (row, templated) in program.steps.iter().filter_map(|step| match step {
                ipu_compiler::LoweredTileStep::Exchange { phase, row, .. } => Some((
                    row,
                    template_regions.iter().any(|region| {
                        region
                            .phase_instances
                            .iter()
                            .any(|phases| phases.contains(phase))
                    }),
                )),
                _ => None,
            }) {
                let mut stored_row = row.to_vec();
                let sender = templated
                    .then(|| ipu_exchange::normalize_sender_instruction(&mut stored_row))
                    .flatten();
                if let Some(return_word) = stored_row
                    .iter()
                    .position(|&instruction| instruction == ipu_exchange::RETURN_M10_INSTRUCTION)
                {
                    stored_row.truncate(return_word + 1);
                }
                let address = if let Some(&address) = unique.get(&stored_row) {
                    address
                } else {
                    cursor = align_up(cursor, 8);
                    let address = cursor;
                    cursor = cursor
                        .checked_add(u32::try_from(stored_row.len() * 4)?)
                        .ok_or("exchange plan address overflow")?;
                    unique.insert(stored_row.clone(), address);
                    rows.push((address, stored_row));
                    address
                };
                addresses.push(address);
                patches.push(
                    sender
                        .map(|(word, instruction)| -> Result<_> {
                            Ok(static_codegen::StaticPlanPatch {
                                word_address: address
                                    .checked_add(u32::try_from(word * 4)?)
                                    .ok_or("exchange plan patch address overflow")?,
                                word_offset: u16::try_from(word)?,
                                instruction,
                            })
                        })
                        .transpose()?,
                );
            }
            let (compute_runs, end) = static_codegen::plan_exchange_compute_runs(
                program,
                &addresses,
                cursor,
                profile_code.is_empty() && template_regions.is_empty(),
            )?;
            let (templates, _) = static_codegen::plan_static_templates(
                program,
                &addresses,
                &patches,
                template_regions,
                0,
            )?;
            Ok(TileExchangePlans {
                addresses,
                rows,
                compute_runs,
                templates,
                end,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    debug!(
        minimum_plan_end = format_args!(
            "0x{:x}",
            tile_exchange_plans
                .iter()
                .map(|plans| plans.end)
                .min()
                .unwrap_or(PLAN_BASE)
        ),
        maximum_plan_end = format_args!(
            "0x{:x}",
            tile_exchange_plans
                .iter()
                .map(|plans| plans.end)
                .max()
                .unwrap_or(PLAN_BASE)
        ),
        "deduplicated device exchange plans"
    );
    let host = build_static_host_layout(graph)?;
    let host_transfers = host
        .inputs
        .iter()
        .chain(&host.outputs)
        .copied()
        .collect::<Vec<_>>();
    for (index, transfer) in host_transfers.iter().enumerate() {
        let hierarchy = ipu_exchange::host_hierarchy(transfer.physical_tile)?;
        debug!(
            index,
            direction = ?transfer.direction,
            physical_tile = transfer.physical_tile,
            xreq_physical_tile = hierarchy.xreq_physical_tile,
            tile_address = format_args!("0x{:x}", transfer.tile_address),
            host_offset = format_args!("0x{:x}", transfer.host_offset),
            bytes = transfer.bytes,
            copy_destination = transfer.copy_destination.map(|address| format!("0x{address:x}")),
            "planned static host transfer"
        );
    }
    let mut tile_host_plans = programs
        .iter()
        .zip(&tile_exchange_plans)
        .map(|(program, exchange_plans)| -> Result<TileHostPlans> {
            let plan_end = exchange_plans.end;
            let physical = topology.physical(program.tile)?;
            let follower_address = align_up(plan_end, 64);
            let mut ordinary_objects = vec![follower_address..follower_address + 3 * 4];
            let mut data_objects = Vec::new();
            let mut cursor = if host_transfers.is_empty() {
                plan_end
            } else {
                follower_address + 3 * 4
            };
            let mut addresses = Vec::with_capacity(host_transfers.len());
            let mut packet_copies = Vec::with_capacity(host_transfers.len());
            let mut instruction_addresses = HashMap::<Vec<u32>, u32>::new();
            let mut packet_addresses = HashMap::<Vec<u32>, u32>::new();
            for &transfer in &host_transfers {
                if host_phase_is_active(physical, &transfer) {
                    let (instructions, packet_words) =
                        host_phase_instructions(physical, transfer, HOST_PACKET_ADDRESS)?;
                    let packet_words = packet_words.ok_or("active host phase has no packet")?;
                    let address = if let Some(&address) = instruction_addresses.get(&instructions) {
                        address
                    } else {
                        cursor = align_up(cursor, 8);
                        let address = cursor;
                        cursor = cursor
                            .checked_add(u32::try_from(instructions.len() * 4)?)
                            .ok_or("static host plan address overflow")?;
                        instruction_addresses.insert(instructions, address);
                        ordinary_objects.push(address..cursor);
                        address
                    };
                    addresses.push(address);
                    let source = if let Some(&source) = packet_addresses.get(&packet_words) {
                        source
                    } else {
                        cursor = align_up(cursor, 4);
                        let source = cursor;
                        cursor = cursor
                            .checked_add(
                                u32::try_from(packet_words.len())?
                                    .checked_mul(4)
                                    .ok_or("host packet size overflow")?,
                            )
                            .ok_or("static host packet address overflow")?;
                        packet_addresses.insert(packet_words.clone(), source);
                        data_objects.push(source..cursor);
                        source
                    };
                    let words = u32::try_from(packet_words.len())?;
                    let hierarchy = ipu_exchange::host_hierarchy(transfer.physical_tile)?;
                    let destination = if physical == transfer.physical_tile
                        && hierarchy.xreq_physical_tile != transfer.physical_tile
                    {
                        HOST_PACKET_ADDRESS + 8
                    } else {
                        HOST_PACKET_ADDRESS
                    };
                    packet_copies.push(Some(HostPacketCopy {
                        source,
                        destination,
                        words,
                    }));
                } else {
                    addresses.push(follower_address);
                    packet_copies.push(None);
                }
            }
            let mut run_tables = vec![None; host_transfers.len()];
            for range in [
                0..host.inputs.len(),
                host.inputs.len()..host_transfers.len(),
            ] {
                let mut index = range.start;
                while index < range.end {
                    if !host_phase_is_active(physical, &host_transfers[index]) {
                        index += 1;
                        continue;
                    }
                    let start = index;
                    while index < range.end
                        && host_phase_is_active(physical, &host_transfers[index])
                    {
                        index += 1;
                    }
                    cursor = align_up(cursor, 4);
                    run_tables[start] = Some(cursor);
                    cursor = cursor
                        .checked_add(
                            u32::try_from(index - start)?
                                .checked_mul(HOST_RUN_DESCRIPTOR_WORDS * 4)
                                .ok_or("static host run descriptor size overflow")?,
                        )
                        .ok_or("static host run descriptor address overflow")?;
                    data_objects.push(run_tables[start].unwrap()..cursor);
                }
            }
            let run_state = align_up(cursor, 4);
            let end = run_state
                .checked_add(8)
                .ok_or("static host run state address overflow")?;
            Ok(TileHostPlans {
                start: follower_address,
                ordinary_objects,
                data_objects: {
                    data_objects.push(run_state..end);
                    data_objects
                },
                addresses,
                packet_copies,
                run_tables,
                run_state,
                end,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut host_runtime_ranges = Vec::with_capacity(tile_host_plans.len());
    let mut worker_sync_addresses = Vec::with_capacity(tile_host_plans.len());
    let mut completion_addresses = Vec::with_capacity(tile_host_plans.len());
    for (tile_index, plans) in tile_host_plans.iter_mut().enumerate() {
        let old_worker_sync = align_up(plans.end, 8);
        let old_completion = align_up(
            old_worker_sync + WORKER_STACK_HEADROOM + WORKER_SYNC_REGISTERS * WORKER_SYNC_STRIDE,
            64,
        );
        let old_end = old_completion
            .checked_add(4)
            .ok_or("static host runtime address overflow")?;
        plans.ordinary_objects.push(old_worker_sync..old_end);
        let tile = programs[tile_index].tile;
        let (mut relocations, mut ranges) = pack_data_objects_for_tile(
            &allocation_ranges_by_tile[usize::from(tile)],
            tile,
            tile_exchange_plans[tile_index].end,
            &plans.ordinary_objects,
            false,
            &[],
        )?;
        let (data_relocations, data_ranges) = pack_data_objects_for_tile(
            &allocation_ranges_by_tile[usize::from(tile)],
            tile,
            tile_exchange_plans[tile_index].end,
            &plans.data_objects,
            true,
            &ranges,
        )?;
        relocations.extend(data_relocations);
        ranges.extend(data_ranges);
        ranges.sort_unstable();
        let relocate = |address: u32| -> Result<u32> {
            relocations
                .get(&address)
                .copied()
                .ok_or_else(|| format!("missing relocation for static object 0x{address:x}").into())
        };
        for address in &mut plans.addresses {
            *address = relocate(*address)?;
        }
        for copy in plans.packet_copies.iter_mut().flatten() {
            copy.source = relocate(copy.source)?;
        }
        for address in plans.run_tables.iter_mut().flatten() {
            *address = relocate(*address)?;
        }
        plans.run_state = relocate(plans.run_state)?;
        let worker_sync = relocate(old_worker_sync)?;
        let completion = worker_sync + (old_completion - old_worker_sync);
        plans.start = ranges.iter().map(|&(start, _)| start).min().unwrap_or(0);
        plans.end = ranges.iter().map(|&(_, end)| end).max().unwrap_or(0);
        host_runtime_ranges.push(ranges);
        worker_sync_addresses.push(worker_sync);
        completion_addresses.push(completion);
        if tile == 0 {
            info!(
                ranges = ?host_runtime_ranges[tile_index],
                first_plan = plans.addresses.first().map(|address| format!("0x{address:x}")),
                first_packet = plans
                    .packet_copies
                    .first()
                    .and_then(|copy| *copy)
                    .map(|copy| format!("0x{:x}", copy.source)),
                first_run_table = plans
                    .run_tables
                    .first()
                    .and_then(|address| *address)
                    .map(|address| format!("0x{address:x}")),
                run_state = format_args!("0x{:x}", plans.run_state),
                worker_sync = format_args!("0x{worker_sync:x}"),
                completion = format_args!("0x{completion:x}"),
                "packed segmented host runtime"
            );
        }
    }
    debug!(
        minimum_end = format_args!(
            "0x{:x}",
            tile_host_plans
                .iter()
                .map(|plans| plans.end)
                .min()
                .unwrap_or(PLAN_BASE)
        ),
        maximum_end = format_args!(
            "0x{:x}",
            tile_host_plans
                .iter()
                .map(|plans| plans.end)
                .max()
                .unwrap_or(PLAN_BASE)
        ),
        "packed host exchange plans"
    );
    let mut template_record_ranges: Vec<Vec<(u32, u32)>> = vec![Vec::new(); programs.len()];
    if let Some((tile_index, template, words, changed_words)) = tile_exchange_plans
        .iter()
        .enumerate()
        .flat_map(|(tile_index, plans)| {
            plans.templates.iter().map(move |template| {
                let words = template.records.iter().map(Vec::len).sum::<usize>();
                let changed_words = template
                    .records
                    .windows(2)
                    .map(|records| {
                        records[0]
                            .iter()
                            .zip(&records[1])
                            .filter(|(left, right)| left != right)
                            .count()
                    })
                    .sum::<usize>();
                (tile_index, template, words, changed_words)
            })
        })
        .max_by_key(|(_, _, words, _)| *words)
    {
        info!(
            logical_tile = programs[tile_index].tile,
            template = template.name,
            instances = template.records.len(),
            record_words = words,
            adjacent_changed_words = changed_words,
            "largest static template record set"
        );
    }
    for (tile_index, plans) in tile_exchange_plans.iter_mut().enumerate() {
        let tile = programs[tile_index].tile;
        let runtime_end = plans.end;
        for template in &mut plans.templates {
            for (record_address, record) in
                template.record_addresses.iter_mut().zip(&template.records)
            {
                let size = u32::try_from(record.len())?
                    .checked_mul(4)
                    .ok_or("static template record size overflow")?;
                if size == 0 {
                    *record_address = ipu_package::TILE_MEMORY_BASE;
                    continue;
                }
                let address = data_region_base_for_tile(
                    &allocation_ranges_by_tile[usize::from(tile)],
                    tile,
                    runtime_end,
                    size,
                    4,
                    &host_runtime_ranges[tile_index]
                        .iter()
                        .copied()
                        .chain(template_record_ranges[tile_index].iter().copied())
                        .collect::<Vec<_>>(),
                )?;
                let end = address
                    .checked_add(size)
                    .ok_or("static template record address overflow")?;
                *record_address = address;
                template_record_ranges[tile_index].push((address, end));
            }
        }
    }

    let mut runtime_symbols = vec![
        static_codegen::WORKER_BARRIER.into(),
        static_codegen::COMPLETE.into(),
        static_codegen::HOST_RUN.into(),
        static_codegen::REPEAT_CALL.into(),
    ];
    if !profile_code.is_empty() {
        runtime_symbols.push(static_codegen::SAMPLE_CYCLE.into());
        runtime_symbols.push(static_codegen::SAMPLE_CYCLE_NEXT.into());
    }
    let tile_retained_symbols = programs
        .iter()
        .zip(&tile_exchange_plans)
        .map(|(program, exchange_plans)| {
            let mut symbols = runtime_symbols.clone();
            if !exchange_plans.compute_runs.is_empty() {
                symbols.push(static_codegen::EXCHANGE_COMPUTE_RUN.into());
            }
            symbols.extend(program.steps.iter().filter_map(|step| match step {
                ipu_compiler::LoweredTileStep::Compute(command) => {
                    Some(format!("ipu_stack_{}", command.specialization.operation))
                }
                ipu_compiler::LoweredTileStep::IdleCompute { .. } => None,
                _ => None,
            }));
            symbols.sort();
            symbols.dedup();
            symbols
        })
        .collect::<Vec<_>>();
    let mut preliminary_cache = HashMap::<Vec<String>, ipu_elf::LinkedImage>::new();
    for symbols in &tile_retained_symbols {
        if preliminary_cache.contains_key(symbols) {
            continue;
        }
        preliminary_cache.insert(
            symbols.clone(),
            link(
                objects,
                &LinkOptions {
                    image_base: ipu_driver::APPLICATION_LOAD_BASE,
                    entry_symbol: "ipu_stack_static_start".into(),
                    retained_symbols: symbols.clone(),
                    externals: HashMap::new(),
                },
            )?,
        );
    }
    let preliminary_images = tile_retained_symbols
        .iter()
        .map(|symbols| -> Result<ipu_elf::LinkedImage> {
            Ok(preliminary_cache
                .get(symbols)
                .ok_or("missing preliminary tile image")?
                .clone())
        })
        .collect::<Result<Vec<_>>>()?;
    let emit_program =
        |program_index: usize, symbols: &BTreeMap<String, u32>, generated_base: u32| {
            let program = &programs[program_index];
            let host_plans = &tile_host_plans[program_index];
            let physical = topology.physical(program.tile)?;
            let host_inputs = host_plans.addresses[..host.inputs.len()]
                .iter()
                .copied()
                .zip(&host_plans.run_tables[..host.inputs.len()])
                .zip(&host.inputs)
                .map(
                    |((address, &run_table), transfer)| static_codegen::HostPhaseCall {
                        address,
                        active: host_phase_is_active(physical, transfer),
                        run_table,
                    },
                )
                .collect::<Vec<_>>();
            let host_outputs = host_plans.addresses[host.inputs.len()..]
                .iter()
                .copied()
                .zip(&host_plans.run_tables[host.inputs.len()..])
                .zip(&host.outputs)
                .map(
                    |((address, &run_table), transfer)| static_codegen::HostPhaseCall {
                        address,
                        active: host_phase_is_active(physical, transfer),
                        run_table,
                    },
                )
                .collect::<Vec<_>>();
            static_codegen::emit(
                program,
                symbols,
                &tile_exchange_plans[program_index].addresses,
                &tile_exchange_plans[program_index].compute_runs,
                &tile_exchange_plans[program_index].templates,
                static_codegen::HostCode {
                    inputs: &host_inputs,
                    outputs: &host_outputs,
                    run_state: host_plans.run_state,
                },
                profile_code.get(program_index),
                generated_base,
            )
        };
    let preliminary_generated = preliminary_images
        .iter()
        .enumerate()
        .map(|(index, image)| emit_program(index, &image.symbols, 0))
        .collect::<Result<Vec<_>>>()?;
    let executable_bases = programs
        .iter()
        .zip(&preliminary_images)
        .zip(&preliminary_generated)
        .zip(&template_record_ranges)
        .map(|(((program, image), generated), template_records)| {
            executable_region_bases_for_tile(
                &allocation_ranges_by_tile[usize::from(program.tile)],
                program.tile,
                tile_exchange_plans[usize::from(program.tile)].end,
                u32::try_from(image.bytes.len())?,
                u32::try_from(generated.len())?,
                &host_runtime_ranges[usize::from(program.tile)]
                    .iter()
                    .copied()
                    .chain(template_records.iter().copied())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let image_bases = executable_bases
        .iter()
        .map(|&(image, _)| image)
        .collect::<Vec<_>>();
    let program_bases = executable_bases
        .iter()
        .map(|&(_, program)| program)
        .collect::<Vec<_>>();
    let mut image_cache = HashMap::<(u32, Vec<String>), ipu_elf::LinkedImage>::new();
    for ((&image_base, symbols), preliminary) in image_bases
        .iter()
        .zip(&tile_retained_symbols)
        .zip(&preliminary_images)
    {
        let key = (image_base, symbols.clone());
        if image_cache.contains_key(&key) {
            continue;
        }
        let image = link(
            objects,
            &LinkOptions {
                image_base,
                entry_symbol: "ipu_stack_static_start".into(),
                retained_symbols: symbols.clone(),
                externals: HashMap::new(),
            },
        )?;
        if image.bytes.len() != preliminary.bytes.len() {
            return Err("linked image size changed after executable placement".into());
        }
        image_cache.insert(key, image);
    }
    let images = image_bases
        .iter()
        .zip(&tile_retained_symbols)
        .map(|(&base, symbols)| -> Result<ipu_elf::LinkedImage> {
            Ok(image_cache
                .get(&(base, symbols.clone()))
                .ok_or("missing linked tile image")?
                .clone())
        })
        .collect::<Result<Vec<_>>>()?;
    if let Some((index, generated_code)) = preliminary_generated
        .iter()
        .enumerate()
        .max_by_key(|(_, code)| code.len())
    {
        let plan_bytes = tile_exchange_plans[index].end - PLAN_BASE;
        let host_runtime_bytes = host_runtime_ranges[index]
            .iter()
            .map(|(start, end)| end - start)
            .sum::<u32>();
        let templates = &tile_exchange_plans[index].templates;
        if templates.is_empty() {
            let steps = static_codegen::step_code_size(
                &programs[index],
                &tile_exchange_plans[index].compute_runs,
            );
            let step_bytes = steps.exchange + steps.compute + steps.fused_run;
            info!(
                logical_tile = programs[index].tile,
                generated_bytes = generated_code.len(),
                generated_exchange_bytes = steps.exchange,
                generated_compute_bytes = steps.compute,
                generated_unrolled_compute_calls = steps.compute_calls,
                generated_fused_compute_calls = steps.fused_compute_calls,
                generated_compute_argument_words = steps.compute_argument_words,
                generated_fused_run_bytes = steps.fused_run,
                generated_fused_runs = tile_exchange_plans[index].compute_runs.len(),
                generated_host_and_control_bytes = generated_code.len().saturating_sub(step_bytes),
                support_image_bytes = images[index].bytes.len(),
                exchange_plan_bytes = plan_bytes,
                host_plan_and_state_bytes = host_runtime_bytes,
                "largest generated tile program breakdown"
            );
        } else {
            let template_record_bytes = templates
                .iter()
                .flat_map(|template| &template.records)
                .map(|record| record.len() * 4)
                .sum::<usize>();
            let template_instances = templates
                .iter()
                .map(|template| template.records.len())
                .sum::<usize>();
            let template_names = templates
                .iter()
                .map(|template| template.name.as_str())
                .collect::<Vec<_>>();
            info!(
                logical_tile = programs[index].tile,
                generated_bytes = generated_code.len(),
                template_names = ?template_names,
                template_instances,
                template_record_bytes,
                support_image_bytes = images[index].bytes.len(),
                exchange_plan_bytes = plan_bytes,
                host_plan_and_state_bytes = host_runtime_bytes,
                "largest generated tile program uses static templates"
            );
        }
    }
    let generated = images
        .iter()
        .zip(&program_bases)
        .enumerate()
        .map(|(index, (image, &base))| emit_program(index, &image.symbols, base))
        .collect::<Result<Vec<_>>>()?;
    if generated
        .iter()
        .zip(&preliminary_generated)
        .any(|(final_code, preliminary)| final_code.len() != preliminary.len())
    {
        return Err("generated program size changed after executable placement".into());
    }
    for allocation in &graph.schedule.allocations {
        let end = allocation
            .address
            .checked_add(allocation.size)
            .ok_or("allocation address overflow")?;
        let runtime_end = tile_exchange_plans[usize::from(allocation.tile)].end;
        let program_base = program_bases[usize::from(allocation.tile)];
        let program_end = program_base
            .checked_add(u32::try_from(
                generated[usize::from(allocation.tile)].len(),
            )?)
            .ok_or("generated tile program address overflow")?;
        let image = &images[usize::from(allocation.tile)];
        let image_end = image
            .base
            .checked_add(u32::try_from(image.bytes.len())?)
            .ok_or("linked image address overflow")?;
        if ranges_overlap(image.base, image_end, allocation.address, end)
            || ranges_overlap(program_base, program_end, allocation.address, end)
            || ranges_overlap(PLAN_BASE, runtime_end, allocation.address, end)
            || host_runtime_ranges[usize::from(allocation.tile)]
                .iter()
                .any(|&(start, stop)| ranges_overlap(start, stop, allocation.address, end))
            || template_record_ranges[usize::from(allocation.tile)]
                .iter()
                .any(|&(start, stop)| ranges_overlap(start, stop, allocation.address, end))
        {
            return Err(format!(
                "static runtime 0x{PLAN_BASE:x}..0x{runtime_end:x} overlaps tensor {} on tile {} at 0x{:x}..0x{end:x}",
                allocation.tensor.0, allocation.tile, allocation.address
            )
            .into());
        }
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
    let completion_logical_tile = programs
        .iter()
        .find(|program| {
            topology.physical(program.tile).map(u32::from).ok() == Some(completion_physical_tile)
        })
        .map(|program| program.tile)
        .ok_or("diagnostic completion tile is outside the schedule")?;
    let max_program_bytes = generated.iter().map(Vec::len).max().unwrap_or(0);
    let mut app = Application::default();
    for (
        tile_index,
        (((((program, generated_code), &program_base), image), host_plans), &completion_address),
    ) in programs
        .iter()
        .zip(generated)
        .zip(&program_bases)
        .zip(&images)
        .zip(&tile_host_plans)
        .zip(&completion_addresses)
        .enumerate()
    {
        let logical = program.tile;
        let physical = u32::from(topology.physical(logical)?);
        let mut entry_code = Vec::with_capacity(ENTRY_TRAMPOLINE_BYTES as usize);
        entry_code.extend_from_slice(&ipu_exchange::encode_setzi_m(0, image.entry)?.to_le_bytes());
        entry_code.extend_from_slice(&ipu_exchange::encode_br_m(0)?.to_le_bytes());
        let entry_blob = app.add_blob(entry_code);
        let mut support_code = image.bytes.clone();
        let symbol_offset = |name: &str| -> Result<usize> {
            Ok(image
                .symbols
                .get(name)
                .copied()
                .ok_or_else(|| format!("static runtime has no {name} symbol"))?
                .checked_sub(image.base)
                .ok_or_else(|| format!("{name} precedes the linked image"))?
                as usize)
        };
        let program_offset = symbol_offset("ipu_stack_static_program_address")?;
        let worker_context_offset = symbol_offset("ipu_stack_static_worker_sync_context_base")?;
        let worker_base_offset = symbol_offset("ipu_stack_static_worker_base")?;
        let prng_seed_base_offset = symbol_offset("ipu_stack_static_prng_seed_base")?;
        let sample_worker_base_offset = (!profile_code.is_empty())
            .then(|| symbol_offset("ipu_stack_static_sample_worker_base"))
            .transpose()?;
        let sample_next_worker_base_offset = (!profile_code.is_empty())
            .then(|| symbol_offset("ipu_stack_static_sample_next_worker_base"))
            .transpose()?;
        let completion_offset = symbol_offset("ipu_stack_static_completion_address")?;
        patch_setzi_immediate(&mut support_code, program_offset, program_base)?;
        patch_setzi_immediate(
            &mut support_code,
            worker_context_offset,
            u32::from(logical) * 8,
        )?;
        let worker_base = worker_sync_addresses[tile_index] + WORKER_STACK_HEADROOM;
        patch_setzi_immediate(&mut support_code, worker_base_offset, worker_base)?;
        let prng_seed_base = (physical + 1) << 3;
        patch_setzi_immediate(&mut support_code, prng_seed_base_offset, prng_seed_base)?;
        if let Some(offset) = sample_worker_base_offset {
            patch_setzi_immediate(&mut support_code, offset, worker_base)?;
        }
        if let Some(offset) = sample_next_worker_base_offset {
            patch_setzi_immediate(&mut support_code, offset, worker_base)?;
        }
        patch_setzi_immediate(&mut support_code, completion_offset, completion_address)?;
        let mut segments = vec![Segment {
            address: ipu_driver::APPLICATION_LOAD_BASE,
            memory_size: ENTRY_TRAMPOLINE_BYTES,
            blob: entry_blob,
            blob_offset: 0,
            file_size: ENTRY_TRAMPOLINE_BYTES,
            flags: SEGMENT_READ | SEGMENT_EXECUTE,
        }];
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

        let plan_region_size = usize::try_from(tile_exchange_plans[tile_index].end - PLAN_BASE)?;
        let mut plan_region = vec![0; plan_region_size];
        for (plan_address, row) in &tile_exchange_plans[tile_index].rows {
            write_plan_bytes(&mut plan_region, *plan_address, &words_to_bytes(row))?;
        }
        for run in &tile_exchange_plans[tile_index].compute_runs {
            write_plan_bytes(
                &mut plan_region,
                run.table_address,
                &words_to_bytes(&run.table_entries),
            )?;
        }
        let mut host_regions = host_runtime_ranges[tile_index]
            .iter()
            .map(|&(start, end)| Ok((start, vec![0; usize::try_from(end - start)?])))
            .collect::<Result<Vec<_>>>()?;
        write_static_host_plans(
            physical as u16,
            StaticHostPacketLayout {
                transfers: &host_transfers,
                plan_addresses: &host_plans.addresses,
                packet_copies: &host_plans.packet_copies,
                run_tables: &host_plans.run_tables,
            },
            &mut host_regions,
        )?;
        if !plan_region.is_empty() {
            let plan_size = u32::try_from(plan_region.len())?;
            let plan_blob = app.add_blob(plan_region);
            segments.push(Segment {
                address: PLAN_BASE,
                memory_size: plan_size,
                blob: plan_blob,
                blob_offset: 0,
                file_size: plan_size,
                flags: SEGMENT_READ | SEGMENT_WRITE | SEGMENT_EXECUTE,
            });
        }
        for (address, bytes) in host_regions {
            let size = u32::try_from(bytes.len())?;
            let blob = app.add_blob(bytes);
            segments.push(Segment {
                address,
                memory_size: size,
                blob,
                blob_offset: 0,
                file_size: size,
                flags: SEGMENT_READ | SEGMENT_WRITE | SEGMENT_EXECUTE,
            });
        }
        let mut template_segments = Vec::<(u32, Vec<u8>)>::new();
        for template in &tile_exchange_plans[tile_index].templates {
            for (&address, record) in template.record_addresses.iter().zip(&template.records) {
                if record.is_empty() {
                    continue;
                }
                let words = record
                    .iter()
                    .map(|word| match word {
                        static_codegen::StaticTemplateRecordWord::Value(value) => Ok(*value),
                        static_codegen::StaticTemplateRecordWord::Symbol(name) => image
                            .symbols
                            .get(name)
                            .copied()
                            .ok_or_else(|| format!("static template references missing {name}")),
                    })
                    .collect::<std::result::Result<Vec<_>, String>>()?;
                let bytes = words_to_bytes(&words);
                if let Some((start, contents)) = template_segments.last_mut() {
                    if start.checked_add(u32::try_from(contents.len())?) == Some(address) {
                        contents.extend_from_slice(&bytes);
                        continue;
                    }
                }
                template_segments.push((address, bytes));
            }
        }
        for (address, bytes) in template_segments {
            let size = u32::try_from(bytes.len())?;
            let blob = app.add_blob(bytes);
            segments.push(Segment {
                address,
                memory_size: size,
                blob,
                blob_offset: 0,
                file_size: size,
                flags: SEGMENT_READ,
            });
        }
        append_initial_segments(&mut app, &mut segments, &initial, logical)?;
        app.tiles.push(TileImage {
            physical_tile: physical,
            entry_point: ipu_driver::APPLICATION_LOAD_BASE,
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
            tile_address: completion_addresses[usize::from(completion_logical_tile)],
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
        max_program_bytes,
        distinct_image_bases = image_cache.len(),
        minimum_image_base = format_args!("0x{:x}", image_bases.iter().min().copied().unwrap_or(0)),
        maximum_image_base = format_args!("0x{:x}", image_bases.iter().max().copied().unwrap_or(0)),
        minimum_program_base =
            format_args!("0x{:x}", program_bases.iter().min().copied().unwrap_or(0)),
        maximum_program_base =
            format_args!("0x{:x}", program_bases.iter().max().copied().unwrap_or(0)),
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
    let host_to_tile_limit =
        ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::HOST_TO_TILE_WINDOW_BYTES;
    let staging_bytes = inputs
        .iter()
        .filter(|transfer| {
            transfer
                .tile_address
                .checked_add(transfer.bytes)
                .is_none_or(|end| end > host_to_tile_limit)
        })
        .map(|transfer| transfer.bytes)
        .max()
        .unwrap_or(0);
    let staging_tiles = inputs
        .iter()
        .filter(|transfer| {
            transfer
                .tile_address
                .checked_add(transfer.bytes)
                .is_none_or(|end| end > host_to_tile_limit)
        })
        .map(|transfer| transfer.physical_tile)
        .collect::<BTreeSet<_>>();
    let staging_range = if staging_bytes == 0 {
        None
    } else {
        let address =
            allocate_low_runtime_range(&graph.schedule, staging_bytes, &staging_tiles, &[])?;
        Some((address, address + staging_bytes))
    };
    for transfer in &mut inputs {
        if transfer
            .tile_address
            .checked_add(transfer.bytes)
            .is_none_or(|end| end > host_to_tile_limit)
        {
            let destination = transfer.tile_address;
            transfer.tile_address = staging_range.ok_or("missing host staging allocation")?.0;
            transfer.copy_destination = Some(destination);
        }
        ipu_exchange::plan_host_to_tile(
            transfer.physical_tile,
            transfer.tile_address,
            transfer.host_offset,
            transfer.bytes,
        )?;
    }
    let call_inputs = calls
        .iter()
        .flat_map(|call| call.inputs.iter().cloned())
        .collect();
    let call_outputs = calls
        .iter()
        .flat_map(|call| call.outputs.iter().cloned())
        .collect();
    let phases = host_transfer_phase_count(u32::try_from(inputs.len() + outputs.len())?)?;
    calls = vec![HostCall {
        name: "graph".into(),
        command: 0,
        phases,
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

fn host_transfer_phase_count(transfers: u32) -> Result<u32> {
    if transfers == 0 {
        return Ok(0);
    }
    // Leave the final transfer and graph-close rendezvous to state-driven
    // finalization, before D2H pages are copied out of the host arena.
    transfers
        .checked_mul(2)
        .and_then(|phases| phases.checked_sub(1))
        .ok_or_else(|| "host phase count overflow".into())
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
            let mut host_offset = binding_base
                .checked_add(u32::try_from(slice.file_offset)?)
                .ok_or("host binding offset overflow")?;
            let mut tile_address = slice.tile_address;
            let mut file_offset = binding_file_base + slice.file_offset;
            let mut remaining = u32::try_from(slice.size)?;
            while remaining != 0 {
                let page_bytes =
                    ipu_exchange::HOST_PAGE_BYTES - host_offset % ipu_exchange::HOST_PAGE_BYTES;
                let bytes = remaining.min(page_bytes);
                let transfer = StaticHostTransfer {
                    direction,
                    physical_tile: u16::try_from(slice.tile)?,
                    tile_address,
                    host_offset,
                    bytes,
                    copy_destination: None,
                };
                if matches!(direction, HostDirection::ToHost) {
                    ipu_exchange::plan_tile_to_host(
                        transfer.physical_tile,
                        transfer.tile_address,
                        transfer.host_offset,
                        transfer.bytes,
                    )?;
                }
                let mut host_slices = Vec::new();
                append_host_slices(&mut host_slices, host_offset, file_offset, u64::from(bytes))?;
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
                host_offset = host_offset
                    .checked_add(bytes)
                    .ok_or("host transfer offset overflow")?;
                tile_address = tile_address
                    .checked_add(bytes)
                    .ok_or("host transfer tile range overflow")?;
                file_offset = file_offset
                    .checked_add(u64::from(bytes))
                    .ok_or("host transfer file range overflow")?;
                remaining -= bytes;
            }
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

fn host_phase_is_active(physical_tile: u16, transfer: &StaticHostTransfer) -> bool {
    ipu_exchange::host_hierarchy(transfer.physical_tile).is_ok_and(|hierarchy| {
        physical_tile == transfer.physical_tile || physical_tile == hierarchy.xreq_physical_tile
    })
}

fn host_phase_instructions(
    physical_tile: u16,
    transfer: StaticHostTransfer,
    packet_address: u32,
) -> Result<(Vec<u32>, Option<Vec<u32>>)> {
    let target = host_target_program(transfer, packet_address + 8)?;
    let hierarchy = ipu_exchange::host_hierarchy(transfer.physical_tile)?;
    if hierarchy.xreq_physical_tile == transfer.physical_tile
        && physical_tile == transfer.physical_tile
    {
        let mut packets = vec![1, 0];
        packets.extend_from_slice(&target.packet_words);
        Ok((
            ipu_exchange::wrap_combined_host_operation(
                transfer.physical_tile,
                &target.instructions,
                packet_address,
            )?,
            Some(packets),
        ))
    } else if physical_tile == hierarchy.xreq_physical_tile {
        let xreq =
            ipu_exchange::assemble_host_xreq_program(transfer.physical_tile, packet_address)?;
        Ok((
            ipu_exchange::wrap_host_xreq_operation(physical_tile, &xreq.instructions)?,
            Some(xreq.packet_words),
        ))
    } else if physical_tile == transfer.physical_tile {
        Ok((
            ipu_exchange::wrap_host_target_operation(physical_tile, &target.instructions)?,
            Some(target.packet_words),
        ))
    } else {
        Ok((
            vec![
                ipu_exchange::sans(1),
                ipu_exchange::SYNC_ANS_INSTRUCTION,
                ipu_exchange::RETURN_M10_INSTRUCTION,
            ],
            None,
        ))
    }
}

struct StaticHostPacketLayout<'a> {
    transfers: &'a [StaticHostTransfer],
    plan_addresses: &'a [u32],
    packet_copies: &'a [Option<HostPacketCopy>],
    run_tables: &'a [Option<u32>],
}

fn write_static_host_plans(
    physical_tile: u16,
    layout: StaticHostPacketLayout<'_>,
    regions: &mut [(u32, Vec<u8>)],
) -> Result<()> {
    let StaticHostPacketLayout {
        transfers,
        plan_addresses,
        packet_copies,
        run_tables,
    } = layout;
    let mut follower_written = false;
    for ((transfer, &plan_address), &packet_copy) in
        transfers.iter().zip(plan_addresses).zip(packet_copies)
    {
        let active = host_phase_is_active(physical_tile, transfer);
        if !active && follower_written {
            continue;
        }
        let (instructions, packet_words) =
            host_phase_instructions(physical_tile, *transfer, HOST_PACKET_ADDRESS)?;
        let instruction_bytes = words_to_bytes(&instructions);
        write_sparse_region_bytes(regions, plan_address, &instruction_bytes)?;
        if let Some(packet_words) = packet_words {
            let packet_copy = packet_copy.ok_or("active host phase has no packet copy")?;
            if packet_copy.words != u32::try_from(packet_words.len())? {
                return Err("host packet copy size changed after layout".into());
            }
            write_sparse_region_bytes(regions, packet_copy.source, &words_to_bytes(&packet_words))?;
        }
        follower_written |= !active;
    }
    for (start, &table_address) in run_tables.iter().enumerate() {
        let Some(table_address) = table_address else {
            continue;
        };
        let mut descriptors = Vec::new();
        let mut index = start;
        while index < transfers.len()
            && (index == start || run_tables[index].is_none())
            && host_phase_is_active(physical_tile, &transfers[index])
        {
            let transfer = transfers[index];
            let copy = (transfer.physical_tile == physical_tile)
                .then_some(transfer.copy_destination)
                .flatten();
            let packet = packet_copies[index].ok_or("active host run has no packet copy")?;
            let copy_words = copy.map_or(0, |_| transfer.bytes / 4);
            if copy_words >= 1 << 23 || packet.words >= 1 << 8 {
                return Err("host descriptor copy count exceeds packed field".into());
            }
            let packet_destination = match packet.destination {
                HOST_PACKET_ADDRESS => 0,
                address if address == HOST_PACKET_ADDRESS + 8 => 1 << 23,
                _ => return Err("host packet destination is not encodable".into()),
            };
            descriptors.extend_from_slice(&[
                plan_addresses[index],
                copy.unwrap_or(0),
                copy.map_or(0, |_| transfer.tile_address),
                copy_words | packet_destination | (packet.words << 24),
                packet.source,
            ]);
            index += 1;
        }
        write_sparse_region_bytes(regions, table_address, &words_to_bytes(&descriptors))?;
    }
    Ok(())
}

fn write_plan_bytes(plan_region: &mut [u8], address: u32, bytes: &[u8]) -> Result<()> {
    write_region_bytes(plan_region, PLAN_BASE, address, bytes)
}

fn write_region_bytes(region: &mut [u8], base: u32, address: u32, bytes: &[u8]) -> Result<()> {
    let start = usize::try_from(
        address
            .checked_sub(base)
            .ok_or("static data address precedes its region")?,
    )?;
    let end = start
        .checked_add(bytes.len())
        .ok_or("static plan range overflow")?;
    let destination = region
        .get_mut(start..end)
        .ok_or("static plan exceeds reserved plan region")?;
    destination.copy_from_slice(bytes);
    Ok(())
}

fn write_sparse_region_bytes(
    regions: &mut [(u32, Vec<u8>)],
    address: u32,
    bytes: &[u8],
) -> Result<()> {
    let (base, region) = regions
        .iter_mut()
        .find(|(base, region)| {
            address >= *base
                && address
                    .checked_add(bytes.len() as u32)
                    .is_some_and(|end| end <= *base + region.len() as u32)
        })
        .ok_or("static data object is outside its packed regions")?;
    write_region_bytes(region, *base, address, bytes)
}

fn append_initial_segments(
    app: &mut Application,
    segments: &mut Vec<Segment>,
    initial: &HashMap<(u16, u32), Vec<u8>>,
    logical: u16,
) -> Result<()> {
    let mut ranges = initial
        .iter()
        .filter(|((tile, _), contents)| *tile == logical && !contents.is_empty())
        .map(|((_, address), contents)| {
            Ok((
                *address,
                address
                    .checked_add(u32::try_from(contents.len())?)
                    .ok_or("initializer range overflow")?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    ranges.sort_unstable();
    let mut merged = Vec::<(u32, u32)>::new();
    for (start, end) in ranges {
        if let Some(previous) = merged.last_mut()
            && start <= previous.1
        {
            previous.1 = previous.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    for (address, end) in merged {
        let size = usize::try_from(end - address)?;
        let mut bytes = vec![0; size];
        let mut initialized = vec![false; size];
        for (&(initial_tile, initial_address), contents) in initial {
            if initial_tile != logical || contents.is_empty() {
                continue;
            }
            let initial_end = initial_address
                .checked_add(u32::try_from(contents.len())?)
                .ok_or("initializer range overflow")?;
            if initial_address < address || initial_end > end {
                continue;
            }
            let offset = usize::try_from(initial_address - address)?;
            for (index, &value) in contents.iter().enumerate() {
                if initialized[offset + index] && bytes[offset + index] != value {
                    return Err(format!("conflicting initializers on tile {logical}").into());
                }
                bytes[offset + index] = value;
                initialized[offset + index] = true;
            }
        }
        let blob = app.add_blob(bytes);
        segments.push(Segment {
            address,
            memory_size: u32::try_from(size)?,
            blob,
            blob_offset: 0,
            file_size: u32::try_from(size)?,
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
    run_host_with_options(
        app,
        bootloader,
        configuration,
        device_path,
        input,
        HostRunOptions::default(),
    )
}

pub fn run_host_with_options(
    app: &Application,
    bootloader: &[u8],
    configuration: &[u8],
    device_path: &str,
    input: &[u8],
    options: HostRunOptions,
) -> Result<Vec<u8>> {
    run_host_impl(
        app,
        bootloader,
        configuration,
        device_path,
        input,
        options,
        None,
    )
}

pub fn run_host_with_inspector(
    app: &Application,
    bootloader: &[u8],
    configuration: &[u8],
    device_path: &str,
    input: &[u8],
    options: HostRunOptions,
    mut inspector: impl FnMut(&Device, &[u8]) -> Result<()>,
) -> Result<Vec<u8>> {
    run_host_impl(
        app,
        bootloader,
        configuration,
        device_path,
        input,
        options,
        Some(&mut inspector),
    )
}

fn run_host_impl(
    app: &Application,
    bootloader: &[u8],
    configuration: &[u8],
    device_path: &str,
    input: &[u8],
    options: HostRunOptions,
    inspector: Option<&mut HostInspector<'_>>,
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
    if let Some(jitter) = options.write_jitter {
        info!(
            seed = jitter.seed,
            max_delay_us = jitter.max_delay.as_micros(),
            "enabling host acknowledgement jitter"
        );
        session.set_write_jitter(jitter.seed, jitter.max_delay);
    }
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
    if calls.len() != 1 {
        return Err("the static host runtime requires exactly one generated call".into());
    }
    let call = &calls[0];
    let deferred = session
        .invoke_deferred(&call.name, call_input(call, input)?)
        .map_err(|error| {
            format!(
                "generated host call {}: {error}; supervisor states: {}; device outputs: {}",
                call.name,
                supervisor_state_summary(&device, app),
                host_source_summary(&device, app)
            )
        })?;
    finish_host_graph(&device, app)?;
    verify_runtime_completion(&device, app)?;
    let call_output = session.collect(&deferred)?;
    let mut output = vec![0; usize::try_from(output_size)?];
    for slice in &call.outputs {
        let start = usize::try_from(slice.file_offset)?;
        let end = usize::try_from(slice.file_offset + slice.size)?;
        output[start..end].copy_from_slice(&call_output[start..end]);
    }
    debug!(states = %supervisor_state_summary(&device, app), "host exchange supervisor states");
    debug!(sources = %host_source_summary(&device, app), "host exchange device sources");
    drop(session);
    if let Some(inspector) = inspector {
        inspector(&device, &output)?;
    }
    Ok(output)
}

fn finish_host_graph(device: &Device, app: &Application) -> Result<()> {
    let completion_tile = app
        .outputs
        .iter()
        .find(|binding| binding.name == "runtime-completion")
        .and_then(|binding| binding.slices.first())
        .map(|slice| slice.tile as u16)
        .ok_or("application has no runtime completion binding")?;
    for _ in 0..2 {
        if completion_reached(device, completion_tile)? {
            return Ok(());
        }
        device.write_sync_mark(ipu_driver::pci::HSP_GS2_CONTROL, 1)?;
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if completion_reached(device, completion_tile)? {
                return Ok(());
            }
            if device.read_config(ipu_driver::pci::HSP_GS2_CONTROL)? == 0 {
                break;
            }
            if Instant::now() >= deadline {
                return Err("host graph did not consume its final acknowledgement".into());
            }
            std::thread::sleep(Duration::from_micros(100));
        }
    }
    Ok(())
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
            let gs1 = device.read_config(ipu_driver::pci::HSP_GS1_CONTROL);
            let gs2 = device.read_config(ipu_driver::pci::HSP_GS2_CONTROL);
            return Err(format!(
                "host graph did not complete; HSP GS1={gs1:?} GS2={gs2:?}; supervisor states: {}; device outputs: {}",
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
    let active_contexts = app
        .tiles
        .iter()
        .filter(|tile| device.tile_context_state(tile.physical_tile as u16, 0).ok() == Some(1))
        .map(|tile| {
            let states = (0..7)
                .map(|context| device.tile_context_state(tile.physical_tile as u16, context))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map(|states| {
                    let exceptions = (1..7)
                        .filter(|&context| states[context as usize] == 3)
                        .map(|context| {
                            let status =
                                device.read_tile_worker_status(tile.physical_tile as u16, context);
                            let pc = device
                                .read_tile_program_counter(tile.physical_tile as u16, context);
                            match (status, pc) {
                                (Ok(status), Ok(pc)) => format!(
                                    "c{context}:{}@0x{pc:x}",
                                    ipu_driver::TileException::from_status(status)
                                ),
                                (status, pc) => format!("c{context}:status={status:?},pc={pc:?}"),
                            }
                        })
                        .collect::<Vec<_>>();
                    format!("{states:?}/{exceptions:?}")
                })
                .unwrap_or_else(|error| format!("error({error})"));
            format!("{}:{states}", tile.physical_tile)
        })
        .collect::<Vec<_>>();
    format!(
        "0={} {:?}, 1={} {:?}, 2={} {:?}, 3={} {:?}, errors={read_errors}, active_contexts={active_contexts:?}, pc={program_counters:?}, milestones={milestones:?}",
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
    let mut host_session = HostSession::new(&device, app.host_exchange.clone())?;
    host_session.start()?;
    for phase in 0..entry.external_syncs {
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
        device.write_sync_mark(ipu_driver::pci::HSP_GS2_CONTROL, 1)?;
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
    drop(host_session);
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

const fn align_down(value: u32, alignment: u32) -> u32 {
    value & !(alignment - 1)
}

const fn ranges_overlap(left_start: u32, left_end: u32, right_start: u32, right_end: u32) -> bool {
    left_start < right_end && right_start < left_end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executable_placement_reserves_complete_memory_elements() {
        let element = ipu_package::TILE_MEMORY_ELEMENT_SIZE;
        let runtime_end = PLAN_BASE + element / 2;
        let allocation_address = align_up(runtime_end, element) + element;
        let graph = ExecutableGraph {
            schedule: Schedule {
                layouts: Vec::new(),
                phases: Vec::new(),
                allocations: vec![ipu_compiler::Allocation {
                    tensor: ipu_compiler::TensorId(1),
                    tile: 0,
                    address: allocation_address,
                    size: 4,
                    live_from: 0,
                    live_until: 1,
                    kind: ipu_compiler::AllocationKind::Home,
                }],
                tile_count: 1,
                peak_sram: BTreeMap::new(),
            },
            initial_buffers: Vec::new(),
            outputs: Vec::new(),
            host_inputs: Vec::new(),
            host_outputs: Vec::new(),
        };
        let required = element + 1;
        let base = executable_region_base(&graph, runtime_end, required).unwrap();
        let end = base + align_up(required, element);

        assert_eq!(base % element, 0);
        assert!(base >= align_up(runtime_end, element));
        assert!(!ranges_overlap(
            base,
            end,
            align_down(allocation_address, element),
            align_up(allocation_address + 4, element),
        ));
    }

    #[test]
    fn static_data_placement_uses_multiple_available_intervals() {
        let memory_base = ipu_package::TILE_MEMORY_BASE;
        let first_gap_start = PLAN_BASE + 0x1000;
        let first_gap_end = first_gap_start + 20;
        let allocations = vec![
            (
                memory_base + ENTRY_TRAMPOLINE_BYTES,
                ipu_exchange::EXCHANGE_WINDOW_BASE,
            ),
            (first_gap_end, first_gap_end + 0x1000),
        ];
        let first =
            data_region_base_for_tile(&allocations, 0, first_gap_start, 16, 4, &[]).unwrap();
        let second = data_region_base_for_tile(
            &allocations,
            0,
            first_gap_start,
            16,
            4,
            &[(first, first + 16)],
        )
        .unwrap();

        for address in [first, second] {
            assert_eq!(address % 4, 0);
            assert!(address >= memory_base);
            assert!(address + 16 <= memory_base + ipu_package::TILE_MEMORY_SIZE);
            assert!(allocations.iter().all(|&(start, end)| !ranges_overlap(
                address,
                address + 16,
                start,
                end
            )));
        }
        assert!(!ranges_overlap(first, first + 16, second, second + 16));
        assert!(second >= first_gap_end + 0x1000);
    }

    #[test]
    fn boundary_profile_reuses_adjacent_timestamps() {
        let step = |local_index| ipu_package::ProfileStep {
            local_index,
            phase: local_index,
            epoch: 0,
            operation: format!("step-{local_index}"),
            kind: ipu_package::ProfileStepKind::Compute,
            kernel: "test_kernel".into(),
            metadata: Vec::new(),
        };
        let layout = ProfileLayout {
            output_offset: 0,
            tiles: vec![ProfileTileLayout {
                physical_tile: 7,
                file_offset: 0,
                steps: vec![step(0), step(1)],
                boundary_samples: true,
            }],
        };
        let output = [11u32, 23, 47]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let report = layout.decode(&output, 1_500_000_000).unwrap();

        assert_eq!(report.tiles[0].samples[0].start_cycle, 11);
        assert_eq!(report.tiles[0].samples[0].end_cycle, 23);
        assert_eq!(report.tiles[0].samples[1].start_cycle, 23);
        assert_eq!(report.tiles[0].samples[1].end_cycle, 47);
    }

    #[test]
    fn compute_profile_preserves_kernel_semantics() {
        let command = ipu_compiler::LoweredComputeCommand {
            op: ipu_compiler::OpId(4),
            phase: 7,
            output: ipu_compiler::TensorId(11),
            inputs: vec![ipu_compiler::TensorId(9), ipu_compiler::TensorId(10)],
            output_address: 0x80000,
            input_addresses: vec![0x50000, 0x54000],
            arguments: vec![64],
            specialization: ipu_compiler::SpecializationKey {
                operation: "gemm_f32_accumulate".into(),
                shape: vec![64, 64, 64],
                worker_count: 6,
                role: "inner-block-3".into(),
                alignment: 32,
            },
            metadata: BTreeMap::from([
                ("label".into(), "GEMM block (2, 5) inner block 3".into()),
                ("output_block_row".into(), "2".into()),
            ]),
        };
        let step = compute_profile_step(8, &command);

        assert_eq!(step.kernel, "gemm_f32_accumulate");
        assert_eq!(step.operation, "GEMM block (2, 5) inner block 3");
        assert!(
            step.metadata
                .iter()
                .any(|entry| entry.name == "shape" && entry.value == "64x64x64")
        );
        assert!(
            step.metadata
                .iter()
                .any(|entry| entry.name == "output_block_row" && entry.value == "2")
        );
    }

    #[test]
    fn phase_profile_groups_commands_and_keeps_compact_semantics() {
        let command = |tile, label: &str| ipu_compiler::KernelCommand {
            tile,
            output: ipu_compiler::TensorId(3),
            inputs: vec![ipu_compiler::TensorId(1), ipu_compiler::TensorId(2)],
            arguments: vec![64],
            specialization: ipu_compiler::SpecializationKey {
                operation: "gelu_c16_to_a8".into(),
                shape: vec![64, 64],
                worker_count: 6,
                role: "activation".into(),
                alignment: 32,
            },
            metadata: BTreeMap::from([
                ("label".into(), label.into()),
                ("layer".into(), "3".into()),
            ]),
        };
        let schedule = ipu_compiler::Schedule {
            layouts: Vec::new(),
            phases: vec![ipu_compiler::Phase::Compute {
                op: ipu_compiler::OpId(9),
                commands: vec![command(7, "layer 3 GeLU"), command(7, "second command")],
            }],
            allocations: Vec::new(),
            tile_count: 8,
            peak_sram: BTreeMap::new(),
        };

        let step = phase_compute_profile_step(&schedule, 7, 0, 0);

        assert_eq!(step.operation, "layer 3 GeLU");
        assert_eq!(step.kernel, "gelu_c16_to_a8");
        assert!(
            step.metadata
                .iter()
                .any(|entry| entry.name == "active_commands" && entry.value == "2")
        );
        assert!(
            step.metadata
                .iter()
                .any(|entry| entry.name == "layer" && entry.value == "3")
        );
        assert!(!step.metadata.iter().any(|entry| entry.name == "label"));
    }

    #[test]
    fn exchange_profile_separates_sync_exchange_and_idle_compute() {
        let scheduled_command = ipu_compiler::KernelCommand {
            tile: 1,
            output: ipu_compiler::TensorId(3),
            inputs: vec![ipu_compiler::TensorId(1), ipu_compiler::TensorId(2)],
            arguments: Vec::new(),
            specialization: ipu_compiler::SpecializationKey {
                operation: "add_u32".into(),
                shape: vec![64],
                worker_count: 6,
                role: "elementwise".into(),
                alignment: 8,
            },
            metadata: BTreeMap::new(),
        };
        let schedule = ipu_compiler::Schedule {
            layouts: Vec::new(),
            phases: vec![
                ipu_compiler::Phase::Exchange {
                    transfers: Vec::new(),
                },
                ipu_compiler::Phase::Compute {
                    op: ipu_compiler::OpId(2),
                    commands: vec![scheduled_command],
                },
            ],
            allocations: Vec::new(),
            tile_count: 2,
            peak_sram: BTreeMap::new(),
        };
        let program = ipu_compiler::LoweredTileProgram {
            tile: 0,
            steps: vec![
                ipu_compiler::LoweredTileStep::Exchange {
                    phase: 0,
                    epoch: 0,
                    row: vec![ipu_exchange::SANS_INACTIVE_INSTRUCTION].into(),
                },
                ipu_compiler::LoweredTileStep::IdleCompute {
                    op: ipu_compiler::OpId(2),
                    phase: 1,
                },
            ],
        };

        let (steps, boundaries) = profile_steps(&schedule, &program, ProfileGranularity::Phase);

        assert_eq!(
            steps.iter().map(|step| step.kind).collect::<Vec<_>>(),
            vec![
                ipu_package::ProfileStepKind::Synchronization,
                ipu_package::ProfileStepKind::Exchange,
                ipu_package::ProfileStepKind::Idle,
            ]
        );
        assert!(boundaries[0].after_sync);
        assert!(boundaries[0].after_step);
        assert!(boundaries[1].after_step);
        assert!(
            steps[1]
                .metadata
                .iter()
                .any(|entry| entry.name == "active" && entry.value == "false")
        );
    }

    #[test]
    fn allocator_profile_covers_every_tile_and_labels_bindings() {
        let graph = ExecutableGraph {
            schedule: ipu_compiler::Schedule {
                layouts: Vec::new(),
                phases: Vec::new(),
                allocations: vec![ipu_compiler::Allocation {
                    tensor: ipu_compiler::TensorId(3),
                    tile: 0,
                    address: 0xa0000,
                    size: 4096,
                    live_from: 0,
                    live_until: usize::MAX,
                    kind: ipu_compiler::AllocationKind::Home,
                }],
                tile_count: 1472,
                peak_sram: BTreeMap::new(),
            },
            initial_buffers: Vec::new(),
            outputs: Vec::new(),
            host_inputs: vec![Binding {
                name: "left".into(),
                dtype: "f32".into(),
                shape: vec![1024],
                slices: vec![RegionSlice {
                    tile: u32::from(Topology::c600().physical(0).unwrap()),
                    tile_address: 0xa0000,
                    file_offset: 0,
                    size: 4096,
                }],
            }],
            host_outputs: Vec::new(),
        };
        let profile = allocator_memory_profile(&graph).unwrap();

        assert_eq!(profile.tiles.len(), 1472);
        assert_eq!(profile.tiles[0].regions.len(), 1);
        assert_eq!(profile.tiles[0].regions[0].name, "left");
        assert!(
            profile
                .tiles
                .iter()
                .skip(1)
                .all(|tile| tile.regions.is_empty())
        );
    }

    #[test]
    fn host_phase_count_defers_the_last_rendezvous() {
        assert_eq!(host_transfer_phase_count(0).unwrap(), 0);
        for transfers in 1..1024 {
            let phases = host_transfer_phase_count(transfers).unwrap();
            assert_eq!(phases.div_ceil(2), transfers);
            assert_eq!(phases % 2, 1);
        }
    }

    #[test]
    fn static_objects_use_multiple_gaps_when_no_single_gap_is_large_enough() {
        let memory_end = ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE;
        let allocations = [
            (ipu_package::TILE_MEMORY_BASE, 0x90000),
            (0x900a0, 0x90200),
            (0x902a0, memory_end),
        ];
        let objects = [0x1000..0x1080, 0x2000..0x2080];

        let (relocations, ranges) =
            pack_data_objects_for_tile(&allocations, 0, PLAN_BASE, &objects, false, &[]).unwrap();

        assert_eq!(ranges.len(), 2);
        assert!(objects.iter().all(|object| {
            relocations.get(&object.start).is_some_and(|&address| {
                ranges
                    .iter()
                    .any(|&(start, end)| address >= start && address + 0x80 <= end)
            })
        }));
    }
}
