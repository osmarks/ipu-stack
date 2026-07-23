use ipu_compiler::Schedule;
use ipu_driver::{Device, DriverError, HostSession, Loader, block_device_interrupt_signals};
use ipu_elf::{LinkOptions, link};
use ipu_exchange::Topology;
use ipu_package::{
    Application, Binding, EntryPoint, HostCall, HostExchange, HostPage, HostSlice, MemoryProfile,
    MemoryRegion, RegionSlice, SEGMENT_EXECUTE, SEGMENT_READ, SEGMENT_WRITE, Segment, TileImage,
    TileMemory,
};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ops::Range;
use std::time::{Duration, Instant};
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;

mod placement;
mod siglip;
mod static_codegen;
pub use siglip::{
    AttentionKernelVariant, DeferredResidualAdd, HostTensorSet, SiglipEncoderLayer,
    SiglipEncoderPrecision, SiglipEncoderTuning, SiglipLinearPrecision, SiglipMapHead,
    SiglipProfileStage, SiglipWeightStorage, append_host_a16_matrix,
    append_host_a16_matrix_in_arenas, append_siglip_encoder_layer,
    append_siglip_encoder_layer_batched_with_precision, append_siglip_encoder_layer_with_precision,
    append_siglip_map_head, append_siglip_map_head_with_memory_policy,
    append_siglip_post_layer_norm, append_siglip_post_layer_norm_with_memory_policy,
    attention_kernel_variant, consolidate_attention_kernel_variants, defer_terminal_residual_add,
    fuse_deferred_residual_into_layer_norm, materialize_deferred_residual_add,
};

mod blocked_data;
pub use blocked_data::{
    BlockLayout, block_binding, block_binding_typed, block_coordinates, blocked_matrix,
    blocked_matrix_f8_f143, blocked_matrix_f8_f143_by_block, blocked_matrix_f16, f143_block_scales,
    f143_from_f32, f143_scale, f143_to_f32, normal_f16,
};

const PLAN_BASE: u32 = ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES;
const HOST_DATA_START: u32 = ipu_exchange::HOST_PAGE_BYTES;
const HOST_CLOSE_ADDRESS: u32 = ipu_exchange::EXCHANGE_WINDOW_BASE + 0x160;
const HOST_PACKET_ADDRESS: u32 = ipu_exchange::EXCHANGE_WINDOW_BASE;
const HOST_STAGING_SEARCH_BASE: u32 = ipu_exchange::EXCHANGE_WINDOW_BASE + 0x180;
const HOST_RUN_DESCRIPTOR_WORDS: u32 = 3;
const WORKER_STACK_HEADROOM: u32 = 0xe0;
const WORKER_SYNC_STRIDE: u32 = 0x100;
const WORKER_CONTEXTS: u32 = 6;
const TILE_CONTEXT_STACKS: u32 = WORKER_CONTEXTS + 1;

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
    weights: Vec<StaticHostTransfer>,
    inputs: Vec<StaticHostTransfer>,
    outputs: Vec<StaticHostTransfer>,
    staging_address: u32,
    protocol: HostExchange,
}

struct TileHostPlans {
    start: u32,
    executable_objects: Vec<Range<u32>>,
    ordinary_data_objects: Vec<Range<u32>>,
    data_objects: Vec<Range<u32>>,
    addresses: Vec<u32>,
    packet_copies: Vec<Option<HostPacketCopy>>,
    run_tables: Vec<Option<u32>>,
    end: u32,
}

struct TileExchangePlans {
    addresses: Vec<u32>,
    rows: Vec<(u32, Vec<u32>)>,
    compute_runs: Vec<static_codegen::ExchangeComputeRun>,
    templates: Vec<static_codegen::StaticTemplatePlan>,
    kernel_symbols: Vec<String>,
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

#[cfg(test)]
fn executable_region_base_for_tile(
    allocation_ranges: &[(u32, u32)],
    tile: Option<u16>,
    runtime_end: u32,
    required_size: u32,
    additional_reserved: &[(u32, u32)],
) -> Result<u32> {
    let regions = executable_regions_for_tile(allocation_ranges, runtime_end, additional_reserved)?;
    let element_size = ipu_package::TILE_MEMORY_ELEMENT_SIZE;
    let required_size = align_up(required_size, element_size);
    if let Some(base) = regions
        .iter()
        .find_map(|&(start, end)| (end - start >= required_size).then_some(start))
    {
        return Ok(base);
    }
    let free_bytes = regions.iter().map(|&(start, end)| end - start).sum::<u32>();
    let largest_gap = regions
        .iter()
        .map(|&(start, end)| end - start)
        .max()
        .unwrap_or(0);
    Err(format!(
        "no {}tile-memory interval can hold {required_size} bytes of executable code: {free_bytes} executable bytes free, {largest_gap}-byte largest gap",
        tile.map_or("common ".into(), |tile| format!("tile {tile} "))
    )
    .into())
}

fn executable_regions_for_tile(
    allocation_ranges: &[(u32, u32)],
    runtime_end: u32,
    additional_reserved: &[(u32, u32)],
) -> Result<Vec<(u32, u32)>> {
    let element_size = ipu_package::TILE_MEMORY_ELEMENT_SIZE;
    let memory_end = ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT;
    let reserved = vec![
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
    let mut space = placement::AddressSpace::new(
        ipu_driver::APPLICATION_LOAD_BASE + ENTRY_TRAMPOLINE_BYTES..memory_end,
    );
    space.reserve_all(
        reserved
            .into_iter()
            .chain(allocation_ranges.iter().copied())
            .chain(additional_reserved.iter().copied()),
    );
    Ok(space.free_regions(element_size))
}

#[cfg(test)]
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
    let gaps = data_regions_for_tile(allocation_ranges, runtime_end, true, additional_reserved)?;
    if let Some(address) = gaps
        .iter()
        .filter_map(|&(start, end)| {
            let address = align_up(start, alignment);
            let remaining = end.checked_sub(address.checked_add(required_size)?)?;
            Some((address, remaining))
        })
        .min_by_key(|&(_, remaining)| remaining)
        .map(|(address, _)| address)
    {
        return Ok(address);
    }
    let free_bytes = gaps.iter().map(|(start, end)| end - start).sum::<u32>();
    let largest_gap = gaps
        .iter()
        .map(|(start, end)| end - start)
        .max()
        .unwrap_or(0);
    Err(format!(
        "no tile-memory interval can hold {required_size} bytes of static data on tile {tile}: {free_bytes} free bytes, {largest_gap}-byte largest gap"
    )
    .into())
}

fn data_regions_for_tile(
    allocation_ranges: &[(u32, u32)],
    runtime_end: u32,
    allow_interleaved: bool,
    additional_reserved: &[(u32, u32)],
) -> Result<Vec<(u32, u32)>> {
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
    let mut space = placement::AddressSpace::new(ipu_package::TILE_MEMORY_BASE..memory_end);
    space.reserve_all(reserved);
    Ok(space.free_regions(1))
}

#[cfg(test)]
fn pack_data_objects_for_tile(
    allocation_ranges: &[(u32, u32)],
    tile: u16,
    runtime_end: u32,
    objects: &[Range<u32>],
    allow_interleaved: bool,
    additional_reserved: &[(u32, u32)],
) -> Result<(BTreeMap<u32, u32>, Vec<(u32, u32)>)> {
    let gaps = data_regions_for_tile(
        allocation_ranges,
        runtime_end,
        allow_interleaved,
        additional_reserved,
    )?;

    pack_objects_in_gaps(tile, objects, gaps, "static data")
}

fn pack_executable_objects_for_tile(
    allocation_ranges: &[(u32, u32)],
    tile: u16,
    runtime_end: u32,
    objects: &[Range<u32>],
    additional_reserved: &[(u32, u32)],
    additional_available: &[(u32, u32)],
) -> Result<(BTreeMap<u32, u32>, Vec<(u32, u32)>, Vec<(u32, u32)>)> {
    let mut gaps =
        executable_regions_for_tile(allocation_ranges, runtime_end, additional_reserved)?;
    gaps.extend_from_slice(additional_available);
    let (relocations, placed) = pack_objects_in_gaps(tile, objects, gaps, "static executable")?;
    let element_size = ipu_package::TILE_MEMORY_ELEMENT_SIZE;
    let mut elements = placed
        .iter()
        .copied()
        .map(|(start, end)| (align_down(start, element_size), align_up(end, element_size)))
        .collect::<Vec<_>>();
    elements.sort_unstable();
    let mut merged = Vec::<(u32, u32)>::new();
    for (start, end) in elements {
        if let Some(previous) = merged.last_mut()
            && start <= previous.1
        {
            previous.1 = previous.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    Ok((relocations, placed, merged))
}

fn pack_objects_in_gaps(
    tile: u16,
    objects: &[Range<u32>],
    mut gaps: Vec<(u32, u32)>,
    description: &str,
) -> Result<(BTreeMap<u32, u32>, Vec<(u32, u32)>)> {
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
                    "no tile-memory gap can hold a {size}-byte {description} object on tile {tile}: {free} free bytes, {largest}-byte largest gap"
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

fn pack_sized_objects_in_gaps(
    tile: u16,
    sizes: &[u32],
    mut gaps: Vec<(u32, u32)>,
    alignment: u32,
    description: &str,
) -> Result<Vec<(u32, u32)>> {
    if !alignment.is_power_of_two() || sizes.contains(&0) {
        return Err(format!("invalid {description} placement request").into());
    }
    let mut order = (0..sizes.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|&index| std::cmp::Reverse(sizes[index]));
    let mut placed = vec![(0, 0); sizes.len()];
    for index in order {
        let size = align_up(sizes[index], alignment);
        let (gap_index, address) = gaps
            .iter()
            .enumerate()
            .filter_map(|(gap_index, &(start, end))| {
                let address = align_up(start, alignment);
                let remaining = end.checked_sub(address.checked_add(size)?)?;
                Some((gap_index, address, remaining))
            })
            .min_by_key(|&(_, _, remaining)| remaining)
            .map(|(gap_index, address, _)| (gap_index, address))
            .ok_or_else(|| {
                let free = gaps.iter().map(|(start, end)| end - start).sum::<u32>();
                let largest = gaps
                    .iter()
                    .map(|(start, end)| end - start)
                    .max()
                    .unwrap_or(0);
                format!(
                    "no tile-memory gap can hold a {size}-byte {description} object on tile {tile}: {free} free bytes, {largest}-byte largest gap"
                )
            })?;
        let end = address
            .checked_add(size)
            .ok_or_else(|| format!("{description} placement overflow"))?;
        gaps[gap_index].0 = end;
        placed[index] = (address, end);
    }
    Ok(placed)
}

fn pack_generated_and_support_images(
    tile: u16,
    generated_size: u32,
    support_size: u32,
    occupied: Vec<(u32, u32)>,
) -> Result<[(u32, u32); 2]> {
    let placed = pack_sized_objects_in_gaps(
        tile,
        &[generated_size, support_size],
        occupied,
        ipu_package::TILE_MEMORY_ELEMENT_SIZE,
        "executable images",
    )?;
    Ok([placed[0], placed[1]])
}

fn validate_pinned_executable_images(
    tile: u16,
    generated_size: u32,
    support_size: u32,
    pinned: [(u32, u32); 2],
    free_regions: &[(u32, u32)],
) -> Result<[(u32, u32); 2]> {
    for ((name, required), (start, end)) in [
        ("generated program", generated_size),
        ("support image", support_size),
    ]
    .into_iter()
    .zip(pinned)
    {
        if end.checked_sub(start).is_none_or(|bytes| bytes < required) {
            return Err(format!(
                "pinned {name} interval 0x{start:x}..0x{end:x} on tile {tile} is shorter than {required} bytes"
            )
            .into());
        }
        if !free_regions
            .iter()
            .any(|&(free_start, free_end)| start >= free_start && end <= free_end)
        {
            return Err(format!(
                "tensor relocation did not preserve pinned {name} interval 0x{start:x}..0x{end:x} on tile {tile}"
            )
            .into());
        }
    }
    if ranges_overlap(pinned[0].0, pinned[0].1, pinned[1].0, pinned[1].1) {
        return Err(format!("pinned executable images overlap on tile {tile}").into());
    }
    Ok(pinned)
}

#[cfg(test)]
fn allocation_range(allocation: &ipu_compiler::Allocation) -> Result<(u32, u32)> {
    Ok((
        allocation.address,
        allocation
            .address
            .checked_add(allocation.size)
            .ok_or("allocation address overflow")?,
    ))
}

fn allocation_footprints_by_tile(
    graph: &ExecutableGraph,
    tile_count: usize,
    memory_constraints: Option<&RelocationMemoryConstraints>,
) -> Result<Vec<Vec<(u32, u32)>>> {
    let mut footprints = vec![Vec::new(); tile_count];
    for (index, allocation) in graph.schedule.allocations.iter().enumerate() {
        if matches!(
            allocation.kind,
            ipu_compiler::AllocationKind::HomeAlias { .. }
        ) {
            continue;
        }
        let bytes = memory_constraints
            .map(|constraints| constraints.access_extent(index, allocation.size))
            .unwrap_or(allocation.size);
        footprints[usize::from(allocation.tile)].push((
            allocation.address,
            allocation
                .address
                .checked_add(bytes)
                .ok_or("allocation address overflow")?,
        ));
    }
    for tile in &mut footprints {
        tile.sort_unstable();
        let mut merged = Vec::<(u32, u32)>::with_capacity(tile.len());
        for &(start, end) in tile.iter() {
            if let Some(previous) = merged.last_mut()
                && start <= previous.1
            {
                previous.1 = previous.1.max(end);
            } else {
                merged.push((start, end));
            }
        }
        *tile = merged;
    }
    Ok(footprints)
}

#[derive(Clone, Debug)]
struct AllocationRelocation {
    tensor: ipu_compiler::TensorId,
    tile: u16,
    old: Range<u32>,
    new_start: u32,
    live_from: usize,
    live_until: usize,
    home: bool,
}

fn is_movable_transient_storage(
    allocation: &ipu_compiler::Allocation,
    arenas: &[ipu_compiler::MemoryArena],
) -> bool {
    if matches!(
        allocation.kind,
        ipu_compiler::AllocationKind::HomeAlias { .. }
    ) || allocation.live_until == usize::MAX
    {
        return false;
    }
    let end = allocation.address.saturating_add(allocation.size);
    arenas
        .iter()
        .any(|arena| allocation.address >= arena.base && end <= arena.limit)
}

fn is_movable_resident_home(
    allocation: &ipu_compiler::Allocation,
    arenas: &[ipu_compiler::MemoryArena],
) -> bool {
    if !matches!(allocation.kind, ipu_compiler::AllocationKind::Home)
        || allocation.live_until != usize::MAX
    {
        return false;
    }
    let end = allocation.address.saturating_add(allocation.size);
    arenas
        .iter()
        .any(|arena| allocation.address >= arena.base && end <= arena.limit)
}

fn default_transient_arenas() -> Vec<ipu_compiler::MemoryArena> {
    vec![
        ipu_compiler::MemoryArena::low(PLAN_BASE, ipu_package::IPU21_INTERLEAVED_MEMORY_BASE),
        ipu_compiler::MemoryArena::low(
            ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT,
            ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
        ),
    ]
}

fn merge_address_ranges(mut ranges: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    ranges.sort_unstable();
    let mut merged = Vec::<(u32, u32)>::with_capacity(ranges.len());
    for (start, end) in ranges {
        if start >= end {
            continue;
        }
        if let Some(previous) = merged.last_mut()
            && start <= previous.1
        {
            previous.1 = previous.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}

fn effective_element_reservations(
    ranges: impl IntoIterator<Item = (u32, u32)>,
) -> Result<Vec<(u32, u32)>> {
    let mut elements = Vec::new();
    for (start, end) in ranges {
        if start >= end {
            continue;
        }
        let bytes = end
            .checked_sub(start)
            .ok_or("executable memory range underflow")?;
        elements.extend(
            ipu_package::ipu21_effective_memory_elements(start, bytes)
                .ok_or_else(|| {
                    format!("executable memory range 0x{start:x}..0x{end:x} is outside tile SRAM")
                })?
                .into_iter()
                .map(|(_, element_start, element_end)| (element_start, element_end)),
        );
    }
    Ok(merge_address_ranges(elements))
}

fn validate_data_element_separation(
    tile: u16,
    description: &str,
    data_ranges: &[(u32, u32)],
    executable_elements: &[(u32, u32)],
) -> Result<()> {
    for &(data_start, data_end) in data_ranges {
        if let Some(&(code_start, code_end)) = executable_elements
            .iter()
            .find(|&&(start, end)| ranges_overlap(data_start, data_end, start, end))
        {
            return Err(format!(
                "{description} on tile {tile} at 0x{data_start:x}..0x{data_end:x} shares executable memory element 0x{code_start:x}..0x{code_end:x}"
            )
            .into());
        }
    }
    Ok(())
}

fn static_placement_ranges(
    reserved: &[(u32, u32)],
    graph_allocations: &[(u32, u32)],
) -> Vec<(u32, u32)> {
    let interleaved =
        ipu_package::IPU21_INTERLEAVED_MEMORY_BASE..ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT;
    merge_address_ranges(
        reserved
            .iter()
            .copied()
            .chain(
                graph_allocations
                    .iter()
                    .copied()
                    .filter(|&(start, end)| start < interleaved.end && end > interleaved.start),
            )
            .collect(),
    )
}

fn allocate_from_sorted_ranges(
    permanent: &[(u32, u32)],
    active: &BTreeMap<(u32, usize), u32>,
    forbidden_starts: &[(u32, u32)],
    size: u32,
    arenas: &[ipu_compiler::MemoryArena],
    alignment: u32,
) -> Option<u32> {
    let allowed_start =
        |gap_start: u32, gap_end: u32, placement: ipu_compiler::MemoryPlacement| match placement {
            ipu_compiler::MemoryPlacement::Low => {
                let mut candidate = align_up(gap_start, alignment);
                loop {
                    if candidate.checked_add(size).is_none_or(|end| end > gap_end) {
                        return None;
                    }
                    let Some(&(_, end)) = forbidden_starts
                        .iter()
                        .find(|&&(start, end)| start <= candidate && candidate < end)
                    else {
                        return Some(candidate);
                    };
                    candidate = align_up(end, alignment);
                }
            }
            ipu_compiler::MemoryPlacement::High => {
                let mut candidate = gap_end
                    .checked_sub(size)
                    .map(|candidate| align_down(candidate, alignment))?;
                loop {
                    if candidate < gap_start {
                        return None;
                    }
                    let Some(&(start, _)) = forbidden_starts
                        .iter()
                        .rev()
                        .find(|&&(start, end)| start <= candidate && candidate < end)
                    else {
                        return Some(candidate);
                    };
                    candidate = start
                        .checked_sub(1)
                        .map(|candidate| align_down(candidate, alignment))?;
                }
            }
        };
    for arena in arenas {
        match arena.placement {
            ipu_compiler::MemoryPlacement::Low => {
                let mut cursor = align_up(arena.base, alignment);
                let mut permanent = permanent.iter().copied().peekable();
                let mut active = active
                    .iter()
                    .map(|(&(start, _), &end)| (start, end))
                    .peekable();
                loop {
                    let range = match (permanent.peek(), active.peek()) {
                        (Some(left), Some(right)) if left.0 <= right.0 => permanent.next(),
                        (Some(_), Some(_)) => active.next(),
                        (Some(_), None) => permanent.next(),
                        (None, Some(_)) => active.next(),
                        (None, None) => break,
                    };
                    let (start, end) = range.unwrap();
                    if end <= cursor || start >= arena.limit {
                        continue;
                    }
                    if let Some(candidate) = allowed_start(cursor, start, arena.placement) {
                        return Some(candidate);
                    }
                    cursor = align_up(cursor.max(end), alignment);
                    if cursor >= arena.limit {
                        break;
                    }
                }
                if let Some(candidate) = allowed_start(cursor, arena.limit, arena.placement) {
                    return Some(candidate);
                }
            }
            ipu_compiler::MemoryPlacement::High => {
                let mut cursor = arena.limit;
                let mut permanent = permanent.iter().copied().rev().peekable();
                let mut active = active
                    .iter()
                    .rev()
                    .map(|(&(start, _), &end)| (start, end))
                    .peekable();
                loop {
                    let range = match (permanent.peek(), active.peek()) {
                        (Some(left), Some(right)) if left.0 >= right.0 => permanent.next(),
                        (Some(_), Some(_)) => active.next(),
                        (Some(_), None) => permanent.next(),
                        (None, Some(_)) => active.next(),
                        (None, None) => break,
                    };
                    let (start, end) = range.unwrap();
                    if start >= cursor || end <= arena.base {
                        continue;
                    }
                    if let Some(candidate) = allowed_start(end, cursor, arena.placement) {
                        return Some(candidate);
                    }
                    cursor = cursor.min(start);
                    if cursor <= arena.base {
                        break;
                    }
                }
                if let Some(candidate) = allowed_start(arena.base, cursor, arena.placement) {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

fn relocation_arenas_for_allocation(
    allocation: &ipu_compiler::Allocation,
    arenas: &[ipu_compiler::MemoryArena],
    requires_interleaved: bool,
) -> Result<Vec<ipu_compiler::MemoryArena>> {
    let interleaved =
        ipu_package::IPU21_INTERLEAVED_MEMORY_BASE..ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT;

    let mut compatible = Vec::new();
    for arena in arenas {
        let append = |compatible: &mut Vec<ipu_compiler::MemoryArena>, base, limit| {
            let base = arena.base.max(base);
            let limit = arena.limit.min(limit);
            if base < limit {
                compatible.push(ipu_compiler::MemoryArena {
                    base,
                    limit,
                    placement: arena.placement,
                });
            }
        };
        if requires_interleaved {
            append(&mut compatible, interleaved.start, interleaved.end);
        } else {
            append(
                &mut compatible,
                ipu_package::TILE_MEMORY_BASE,
                interleaved.start,
            );
            append(&mut compatible, interleaved.start, interleaved.end);
            append(
                &mut compatible,
                interleaved.end,
                ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
            );
        }
    }
    if compatible.is_empty() {
        let class = if requires_interleaved {
            "required interleaved"
        } else {
            "compatible"
        };
        return Err(format!(
            "memory policy has no {class} relocation arena for tensor {} on tile {}",
            allocation.tensor.0, allocation.tile,
        )
        .into());
    }
    Ok(compatible)
}

fn allocation_requires_interleaved(allocation: &ipu_compiler::Allocation) -> bool {
    let end = allocation.address.saturating_add(allocation.size);
    allocation.address >= ipu_package::IPU21_INTERLEAVED_MEMORY_BASE
        && end <= ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT
}

fn placement_diagnostics(
    permanent: &[(u32, u32)],
    active: &BTreeMap<(u32, usize), u32>,
    arenas: &[ipu_compiler::MemoryArena],
    alignment: u32,
) -> (u32, u32, u32) {
    let mut capacity = 0u32;
    let mut free = 0u32;
    let mut largest = 0u32;
    for arena in arenas {
        capacity = capacity.saturating_add(arena.limit.saturating_sub(arena.base));
        let occupied = merge_address_ranges(
            permanent
                .iter()
                .copied()
                .chain(active.iter().map(|(&(start, _), &end)| (start, end)))
                .filter_map(|(start, end)| {
                    let start = start.max(arena.base);
                    let end = end.min(arena.limit);
                    (start < end).then_some((start, end))
                })
                .collect(),
        );
        let mut cursor = align_up(arena.base, alignment);
        for (start, end) in occupied {
            if cursor < start {
                let bytes = start - cursor;
                free = free.saturating_add(bytes);
                largest = largest.max(bytes);
            }
            cursor = align_up(cursor.max(end), alignment);
        }
        if cursor < arena.limit {
            let bytes = arena.limit - cursor;
            free = free.saturating_add(bytes);
            largest = largest.max(bytes);
        }
    }
    (capacity, free, largest)
}

fn fixed_allocation_ranges_by_tile(
    graph: &ExecutableGraph,
    tile_count: usize,
    memory_constraints: &RelocationMemoryConstraints,
) -> Result<Vec<Vec<(u32, u32)>>> {
    let default_arenas = default_transient_arenas();
    let transient_arenas = graph
        .memory_policy
        .as_ref()
        .map_or(default_arenas.as_slice(), |policy| {
            policy.transient.as_slice()
        });
    let mut ranges = vec![Vec::new(); tile_count];
    for (index, allocation) in graph.schedule.allocations.iter().enumerate() {
        let movable = matches!(
            allocation.kind,
            ipu_compiler::AllocationKind::HomeAlias { .. }
        ) || is_movable_transient_storage(allocation, transient_arenas);
        if movable && !memory_constraints.pinned.contains(&index) {
            continue;
        }
        let bytes = memory_constraints.access_extent(index, allocation.size);
        ranges[usize::from(allocation.tile)].push((
            allocation.address,
            allocation
                .address
                .checked_add(bytes)
                .ok_or("fixed allocation address overflow")?,
        ));
    }
    Ok(ranges)
}

fn immovable_allocation_ranges_by_tile(
    graph: &ExecutableGraph,
    tile_count: usize,
    memory_constraints: &RelocationMemoryConstraints,
) -> Result<Vec<Vec<(u32, u32)>>> {
    let default_arenas = default_transient_arenas();
    let transient_arenas = graph
        .memory_policy
        .as_ref()
        .map_or(default_arenas.as_slice(), |policy| {
            policy.transient.as_slice()
        });
    let resident_arenas = graph
        .memory_policy
        .as_ref()
        .map_or(&[][..], |policy| policy.resident.as_slice());
    let mut ranges = vec![Vec::new(); tile_count];
    for (index, allocation) in graph.schedule.allocations.iter().enumerate() {
        let movable = matches!(
            allocation.kind,
            ipu_compiler::AllocationKind::HomeAlias { .. }
        ) || is_movable_transient_storage(allocation, transient_arenas)
            || is_movable_resident_home(allocation, resident_arenas);
        if movable && !memory_constraints.pinned.contains(&index) {
            continue;
        }
        let bytes = memory_constraints.access_extent(index, allocation.size);
        ranges[usize::from(allocation.tile)].push((
            allocation.address,
            allocation
                .address
                .checked_add(bytes)
                .ok_or("immovable allocation address overflow")?,
        ));
    }
    Ok(ranges)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum MemoryElementNeighbor {
    Allocation {
        index: usize,
        offset: u32,
        bytes: u32,
    },
    Fixed {
        address: u32,
        bytes: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct MemoryElementRelation {
    own_offset: u32,
    own_bytes: u32,
    neighbor: MemoryElementNeighbor,
}

#[derive(Default)]
struct RelocationMemoryConstraints {
    relations: HashMap<usize, Vec<MemoryElementRelation>>,
    pinned: HashSet<usize>,
    required_interleaved: HashSet<usize>,
    /// Maximum base-relative address touched by a kernel. This reserves
    /// pipelined read-ahead against code/static objects without changing data
    /// allocation ownership or lifetime coloring.
    access_extents: HashMap<usize, u32>,
}

impl RelocationMemoryConstraints {
    fn access_extent(&self, index: usize, logical_size: u32) -> u32 {
        self.access_extents
            .get(&index)
            .copied()
            .unwrap_or(logical_size)
            .max(logical_size)
    }
}

fn memory_spans_share_effective_element(
    first_address: u32,
    first_bytes: u32,
    second_address: u32,
    second_bytes: u32,
) -> Result<bool> {
    let first = ipu_package::ipu21_effective_memory_elements(first_address, first_bytes)
        .ok_or("first kernel operand span is outside tile SRAM")?;
    let second = ipu_package::ipu21_effective_memory_elements(second_address, second_bytes)
        .ok_or("second kernel operand span is outside tile SRAM")?;
    Ok(first
        .iter()
        .any(|first| second.iter().any(|second| first.0 == second.0)))
}

fn relocation_memory_constraints(
    graph: &ExecutableGraph,
    resolved: &ipu_compiler::ResolvedKernelMemoryConstraints,
) -> Result<RelocationMemoryConstraints> {
    fn allocation_endpoint(
        graph: &ExecutableGraph,
        endpoint: ipu_compiler::ResolvedMemoryOperand,
        access_extents: &mut HashMap<usize, u32>,
    ) -> Result<(Option<usize>, u32)> {
        let Some(index) = endpoint.allocation else {
            return Ok((None, endpoint.address));
        };
        let allocation = graph
            .schedule
            .allocations
            .get(index)
            .ok_or("resolved kernel memory constraint has an invalid allocation index")?;
        if allocation.tile != endpoint.tile {
            return Err("resolved kernel memory constraint changes tiles".into());
        }
        let offset = endpoint
            .address
            .checked_sub(allocation.address)
            .ok_or("resolved kernel operand precedes its owning allocation")?;
        if endpoint.bytes == 0 || offset >= allocation.size {
            return Err("resolved kernel operand base lies outside its owning allocation".into());
        }
        let access_end = offset
            .checked_add(endpoint.bytes)
            .ok_or("resolved kernel operand access span overflows")?;
        access_extents
            .entry(index)
            .and_modify(|extent| *extent = (*extent).max(access_end))
            .or_insert(access_end);
        Ok((Some(index), offset))
    }

    let mut constraints = RelocationMemoryConstraints::default();
    for &class in &resolved.classes {
        let endpoint = allocation_endpoint(graph, class.operand, &mut constraints.access_extents)?;
        match (class.class, endpoint) {
            (ipu_compiler::KernelMemoryClass::Ipu21Interleaved, (Some(index), 0)) => {
                let end = class
                    .operand
                    .address
                    .checked_add(class.operand.bytes)
                    .ok_or("interleaved kernel allocation address overflow")?;
                if class.operand.address < ipu_package::IPU21_INTERLEAVED_MEMORY_BASE
                    || end > ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT
                {
                    return Err("kernel allocation is outside required interleaved memory".into());
                }
                constraints.required_interleaved.insert(index);
            }
            (ipu_compiler::KernelMemoryClass::Ipu21Interleaved, (Some(_), _)) => {
                return Err("interleaved kernel operand is not allocation-aligned".into());
            }
            (ipu_compiler::KernelMemoryClass::Ipu21Interleaved, (None, address)) => {
                let end = address
                    .checked_add(class.operand.bytes)
                    .ok_or("fixed interleaved operand span overflow")?;
                if !(ipu_package::IPU21_INTERLEAVED_MEMORY_BASE
                    ..ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT)
                    .contains(&address)
                    || end > ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT
                {
                    return Err("fixed kernel operand is outside interleaved memory".into());
                }
            }
        }
    }
    for &separation in &resolved.separations {
        if separation.first.tile != separation.second.tile {
            return Err("one kernel memory-element constraint spans multiple tiles".into());
        }
        let first = allocation_endpoint(graph, separation.first, &mut constraints.access_extents)?;
        let second =
            allocation_endpoint(graph, separation.second, &mut constraints.access_extents)?;
        match (first, second) {
            ((None, first), (None, second)) => {
                if memory_spans_share_effective_element(
                    first,
                    separation.first.bytes,
                    second,
                    separation.second.bytes,
                )? {
                    return Err("fixed kernel operands require distinct memory elements".into());
                }
            }
            ((Some(index), own_offset), (None, address))
            | ((None, address), (Some(index), own_offset)) => {
                let (own_bytes, neighbor_bytes) = if separation.first.allocation.is_some() {
                    (separation.first.bytes, separation.second.bytes)
                } else {
                    (separation.second.bytes, separation.first.bytes)
                };
                constraints
                    .relations
                    .entry(index)
                    .or_default()
                    .push(MemoryElementRelation {
                        own_offset,
                        own_bytes,
                        neighbor: MemoryElementNeighbor::Fixed {
                            address,
                            bytes: neighbor_bytes,
                        },
                    });
            }
            ((Some(first), first_offset), (Some(second), second_offset)) if first == second => {
                let allocation = &graph.schedule.allocations[first];
                let first_address = allocation.address.saturating_add(first_offset);
                let second_address = allocation.address.saturating_add(second_offset);
                if memory_spans_share_effective_element(
                    first_address,
                    separation.first.bytes,
                    second_address,
                    separation.second.bytes,
                )? {
                    return Err(
                        "two operands in one allocation require distinct memory elements".into(),
                    );
                }
                // Their relative offsets are fixed, but crossing an element boundary can
                // depend on the allocation base. Preserve a known-valid base.
                constraints.pinned.insert(first);
            }
            ((Some(first), first_offset), (Some(second), second_offset)) => {
                constraints
                    .relations
                    .entry(first)
                    .or_default()
                    .push(MemoryElementRelation {
                        own_offset: first_offset,
                        own_bytes: separation.first.bytes,
                        neighbor: MemoryElementNeighbor::Allocation {
                            index: second,
                            offset: second_offset,
                            bytes: separation.second.bytes,
                        },
                    });
                constraints
                    .relations
                    .entry(second)
                    .or_default()
                    .push(MemoryElementRelation {
                        own_offset: second_offset,
                        own_bytes: separation.second.bytes,
                        neighbor: MemoryElementNeighbor::Allocation {
                            index: first,
                            offset: first_offset,
                            bytes: separation.first.bytes,
                        },
                    });
            }
        }
    }
    for relations in constraints.relations.values_mut() {
        relations.sort_unstable();
        relations.dedup();
    }
    Ok(constraints)
}

fn compact_transient_allocations_around(
    graph: &mut ExecutableGraph,
    topology: &Topology,
    reservations: &[Vec<(u32, u32)>],
    memory_constraints: Option<&ipu_compiler::ResolvedKernelMemoryConstraints>,
    reason: &str,
) -> Result<usize> {
    compact_allocations_around(
        graph,
        topology,
        reservations,
        memory_constraints,
        reason,
        false,
    )
}

fn compact_all_allocations_around(
    graph: &mut ExecutableGraph,
    topology: &Topology,
    reservations: &[Vec<(u32, u32)>],
    memory_constraints: Option<&ipu_compiler::ResolvedKernelMemoryConstraints>,
    reason: &str,
) -> Result<usize> {
    compact_allocations_around(
        graph,
        topology,
        reservations,
        memory_constraints,
        reason,
        true,
    )
}

fn compact_allocations_around(
    graph: &mut ExecutableGraph,
    topology: &Topology,
    reservations: &[Vec<(u32, u32)>],
    resolved_memory_constraints: Option<&ipu_compiler::ResolvedKernelMemoryConstraints>,
    reason: &str,
    move_resident: bool,
) -> Result<usize> {
    if reservations.len() != topology.tile_count() {
        return Err("invalid transient repacking reservations".into());
    }
    let memory_constraints = resolved_memory_constraints
        .map(|constraints| relocation_memory_constraints(graph, constraints))
        .transpose()?
        .unwrap_or_default();
    let physical_to_logical = (0..u16::try_from(topology.tile_count())?)
        .map(|logical| Ok((u32::from(topology.physical(logical)?), logical)))
        .collect::<Result<HashMap<_, _>>>()?;
    let ordinary_arenas = default_transient_arenas();
    let transient_arenas = graph
        .memory_policy
        .as_ref()
        .map_or(ordinary_arenas.as_slice(), |policy| {
            policy.transient.as_slice()
        })
        .to_vec();
    if transient_arenas.is_empty() {
        return Err("transient allocation policy has no arenas".into());
    }
    let resident_arenas = graph
        .memory_policy
        .as_ref()
        .filter(|_| move_resident)
        .map_or(&[][..], |policy| policy.resident.as_slice())
        .to_vec();
    let mut transient_by_tile = vec![Vec::new(); topology.tile_count()];
    let mut resident_by_tile = vec![Vec::new(); topology.tile_count()];
    let mut fixed_by_tile = vec![Vec::new(); topology.tile_count()];
    for (index, allocation) in graph.schedule.allocations.iter().enumerate() {
        if is_movable_transient_storage(allocation, &transient_arenas) {
            transient_by_tile[usize::from(allocation.tile)].push(index);
        } else if is_movable_resident_home(allocation, &resident_arenas) {
            resident_by_tile[usize::from(allocation.tile)].push(index);
        } else if !matches!(
            allocation.kind,
            ipu_compiler::AllocationKind::HomeAlias { .. }
        ) {
            fixed_by_tile[usize::from(allocation.tile)].push((
                allocation.address,
                allocation.address.saturating_add(allocation.size),
            ));
        }
    }
    let placements = transient_by_tile
        .into_par_iter()
        .zip(resident_by_tile)
        .enumerate()
        .map(|(tile, (mut transients, mut residents))| -> Result<Vec<(usize, u32)>> {
            let permanent = merge_address_ranges(
                reservations[tile]
                    .iter()
                .copied()
                .chain(fixed_by_tile[tile].iter().copied())
                .collect(),
            );
            let mut result = Vec::with_capacity(residents.len() + transients.len());
            // Place resident and transient objects in one offline lifetime
            // coloring problem. Resident objects cover the complete timeline;
            // transient storage can be reused across disjoint phase intervals.
            let phase_count = transients
                    .iter()
                    .map(|&index| graph.schedule.allocations[index].live_until)
                    .filter(|&live_until| live_until != usize::MAX)
                    .max()
                    .unwrap_or(0)
                    .max(graph.schedule.phases.len().saturating_add(1));
            let mut occupied_by_phase = vec![Vec::<(u32, u32)>::new(); phase_count];
            let lifetime = |allocation: &ipu_compiler::Allocation| {
                    let end = if allocation.live_until == usize::MAX {
                        phase_count
                    } else {
                        allocation.live_until
                    };
                    allocation.live_from.min(phase_count)..end.min(phase_count)
            };
            let mut movable = Vec::with_capacity(residents.len() + transients.len());
            movable.append(&mut residents);
            movable.append(&mut transients);
            let mut compact = Vec::with_capacity(movable.len());
            for index in movable {
                    let allocation = &graph.schedule.allocations[index];
                    if memory_constraints.pinned.contains(&index) {
                        let range = (
                            allocation.address,
                            allocation.address.saturating_add(allocation.size),
                        );
                        for occupied in &mut occupied_by_phase[lifetime(allocation)] {
                            occupied.push(range);
                        }
                        result.push((index, allocation.address));
                    } else {
                        compact.push(index);
                    }
            }
            let compact_set = compact.iter().copied().collect::<HashSet<_>>();
            let mut placed_addresses = result.iter().copied().collect::<HashMap<_, _>>();
            compact.sort_unstable_by_key(|&index| {
                    let allocation = &graph.schedule.allocations[index];
                    (
                        !memory_constraints.required_interleaved.contains(&index),
                        allocation.live_until != usize::MAX,
                        std::cmp::Reverse(allocation.size),
                        allocation.live_from,
                        allocation.live_until,
                        allocation.tensor.0,
                    )
            });
            for index in compact {
                    let allocation = &graph.schedule.allocations[index];
                    let arenas = if allocation.live_until == usize::MAX {
                        &resident_arenas
                    } else {
                        &transient_arenas
                    };
                    let requires_interleaved = (resolved_memory_constraints.is_none()
                        && allocation_requires_interleaved(allocation))
                        || memory_constraints.required_interleaved.contains(&index);
                    let compatible_arenas = relocation_arenas_for_allocation(
                        allocation,
                        arenas,
                        requires_interleaved,
                    )?;
                    let allocation_lifetime = lifetime(allocation);
                    let mut occupied = permanent.clone();
                    occupied.extend(
                        occupied_by_phase[allocation_lifetime.clone()]
                            .iter()
                            .flatten()
                            .copied(),
                    );
                    let occupied = merge_address_ranges(occupied);
                    let access_extent = memory_constraints.access_extent(index, allocation.size);
                    let forbidden_starts = memory_constraints
                        .relations
                        .get(&index)
                        .into_iter()
                        .flatten()
                        .flat_map(|relation| {
                            let (neighbor_address, neighbor_bytes) = match relation.neighbor {
                                MemoryElementNeighbor::Fixed { address, bytes } => (address, bytes),
                                MemoryElementNeighbor::Allocation {
                                    index: neighbor,
                                    offset,
                                    bytes,
                                } => {
                                    let base = placed_addresses.get(&neighbor).copied().or_else(|| {
                                        (!compact_set.contains(&neighbor)).then_some(
                                            graph.schedule.allocations[neighbor].address,
                                        )
                                    });
                                    let Some(address) =
                                        base.and_then(|base| base.checked_add(offset))
                                    else {
                                        return Vec::new();
                                    };
                                    (address, bytes)
                                }
                            };
                            ipu_package::ipu21_effective_memory_elements(
                                neighbor_address,
                                neighbor_bytes,
                            )
                            .into_iter()
                            .flatten()
                            .filter_map(|(_, start, end)| {
                                let start = i64::from(start)
                                    - i64::from(relation.own_offset)
                                    - i64::from(relation.own_bytes)
                                    + 1;
                                let end = i64::from(end) - i64::from(relation.own_offset);
                                let start = u32::try_from(start.max(0)).ok()?;
                                let end = u32::try_from(end.max(0)).ok()?;
                                (start < end).then_some((start, end))
                            })
                            .collect::<Vec<_>>()
                        })
                        .chain(reservations[tile].iter().filter_map(|&(start, end)| {
                            let forbidden_start = start.saturating_sub(access_extent.saturating_sub(1));
                            (forbidden_start < end).then_some((forbidden_start, end))
                        }))
                        .collect::<Vec<_>>();
                    let forbidden_starts = merge_address_ranges(forbidden_starts);
                    let address = allocate_from_sorted_ranges(
                        &occupied,
                        &BTreeMap::new(),
                        &forbidden_starts,
                        allocation.size,
                        &compatible_arenas,
                        32,
                    )
                    .ok_or_else(|| {
                        let (capacity, free, largest) = placement_diagnostics(
                            &occupied,
                            &BTreeMap::new(),
                            &compatible_arenas,
                            32,
                        );
                        format!(
                            "cannot compact tensor {} on tile {} for {reason}: no arena can hold a {}-byte SRAM allocation with a {}-byte static access span at lifetime {}..{} ({} bytes capacity, {} bytes free, {}-byte largest aligned gap)",
                            allocation.tensor.0, allocation.tile, allocation.size, access_extent,
                            allocation.live_from, allocation.live_until, capacity, free, largest,
                        )
                    })?;
                    let range = (address, address.saturating_add(allocation.size));
                    for phase in &mut occupied_by_phase[allocation_lifetime] {
                        phase.push(range);
                    }
                placed_addresses.insert(index, address);
                result.push((index, address));
            }
            Ok(result)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut relocations = Vec::new();
    for (index, new_address) in placements.into_iter().flatten() {
        let allocation = &graph.schedule.allocations[index];
        if allocation.address == new_address {
            continue;
        }
        let old_end = allocation
            .address
            .checked_add(allocation.size)
            .ok_or("allocation relocation range overflow")?;
        relocations.push(AllocationRelocation {
            tensor: allocation.tensor,
            tile: allocation.tile,
            old: allocation.address..old_end,
            new_start: new_address,
            live_from: allocation.live_from,
            live_until: allocation.live_until,
            home: matches!(allocation.kind, ipu_compiler::AllocationKind::Home),
        });
        graph.schedule.allocations[index].address = new_address;
    }
    let moved_storage = relocations.len();
    let mut relocation_indices_by_location =
        HashMap::<(ipu_compiler::TensorId, u16), Vec<usize>>::default();
    for (index, relocation) in relocations.iter().enumerate() {
        relocation_indices_by_location
            .entry((relocation.tensor, relocation.tile))
            .or_default()
            .push(index);
    }
    let mut pending_aliases = graph
        .schedule
        .allocations
        .iter()
        .enumerate()
        .filter_map(|(index, allocation)| {
            matches!(
                allocation.kind,
                ipu_compiler::AllocationKind::HomeAlias { .. }
            )
            .then_some(index)
        })
        .collect::<Vec<_>>();
    loop {
        let mut propagated = 0usize;
        pending_aliases.retain(|&allocation_index| {
            let (tensor, tile, old_start, size, live_from, live_until, source) = {
                let allocation = &graph.schedule.allocations[allocation_index];
                let ipu_compiler::AllocationKind::HomeAlias { source } = allocation.kind else {
                    unreachable!();
                };
                (
                    allocation.tensor,
                    allocation.tile,
                    allocation.address,
                    allocation.size,
                    allocation.live_from,
                    allocation.live_until,
                    source,
                )
            };
            let old_end = old_start.saturating_add(size);
            let relocated = relocation_indices_by_location
                .get(&(source, tile))
                .into_iter()
                .flatten()
                .map(|&index| &relocations[index])
                .filter(|relocation| {
                    old_start >= relocation.old.start
                        && old_end <= relocation.old.end
                        && relocation.live_from <= live_from
                        && relocation.live_until >= live_until
                })
                .map(|relocation| relocation.new_start + (old_start - relocation.old.start))
                .collect::<BTreeSet<_>>();
            if relocated.len() > 1 {
                return true;
            }
            let Some(&new_start) = relocated.first() else {
                return true;
            };
            graph.schedule.allocations[allocation_index].address = new_start;
            let relocation_index = relocations.len();
            relocations.push(AllocationRelocation {
                tensor,
                tile,
                old: old_start..old_end,
                new_start,
                live_from,
                live_until,
                home: false,
            });
            relocation_indices_by_location
                .entry((tensor, tile))
                .or_default()
                .push(relocation_index);
            propagated += 1;
            false
        });
        if propagated == 0 {
            break;
        }
    }
    for &allocation_index in &pending_aliases {
        let allocation = &graph.schedule.allocations[allocation_index];
        let ipu_compiler::AllocationKind::HomeAlias { source } = allocation.kind else {
            unreachable!();
        };
        let old_end = allocation.address.saturating_add(allocation.size);
        let candidates = relocation_indices_by_location
            .get(&(source, allocation.tile))
            .into_iter()
            .flatten()
            .map(|&index| &relocations[index])
            .filter(|relocation| {
                allocation.address >= relocation.old.start
                    && old_end <= relocation.old.end
                    && relocation.live_from <= allocation.live_from
                    && relocation.live_until >= allocation.live_until
            })
            .count();
        if candidates > 1 {
            return Err(format!(
                "alias tensor {} on tile {} at 0x{:x} has ambiguous relocation ownership from source tensor {}",
                allocation.tensor.0, allocation.tile, allocation.address, source.0,
            )
            .into());
        }
    }
    let mut relocations_by_location =
        HashMap::<(ipu_compiler::TensorId, u16), Vec<&AllocationRelocation>>::default();
    let mut home_relocations_by_tile = vec![Vec::new(); topology.tile_count()];
    for relocation in &relocations {
        relocations_by_location
            .entry((relocation.tensor, relocation.tile))
            .or_default()
            .push(relocation);
        if relocation.home {
            home_relocations_by_tile[usize::from(relocation.tile)].push(relocation);
        }
    }
    for (phase, scheduled) in graph.schedule.phases.iter_mut().enumerate() {
        let ipu_compiler::Phase::Exchange { transfers } = scheduled else {
            continue;
        };
        for transfer in transfers {
            let Some(address) = &mut transfer.staging_address else {
                continue;
            };
            let candidates = relocations_by_location
                .get(&(transfer.tensor, transfer.destination_tile))
                .into_iter()
                .flatten()
                .filter(|relocation| {
                    relocation.live_from <= phase
                        && phase < relocation.live_until
                        && relocation.old.contains(address)
                })
                .collect::<Vec<_>>();
            if candidates.len() > 1 {
                return Err(format!(
                    "transfer of tensor {} to tile {} at phase {phase} has ambiguous relocation ownership for 0x{address:x}",
                    transfer.tensor.0, transfer.destination_tile,
                )
                .into());
            }
            if let Some(relocation) = candidates.first() {
                *address = relocation.new_start + (*address - relocation.old.start);
            }
        }
    }
    for buffer in &mut graph.initial_buffers {
        relocate_boundary_address(
            &mut buffer.address,
            &home_relocations_by_tile[usize::from(buffer.tile)],
            GraphBoundary::Input,
        )?;
    }
    for (bindings, boundary) in [
        (&mut graph.host_weights, GraphBoundary::Input),
        (&mut graph.host_inputs, GraphBoundary::Input),
        (&mut graph.outputs, GraphBoundary::Output),
        (&mut graph.host_outputs, GraphBoundary::Output),
    ] {
        for binding in bindings {
            for slice in &mut binding.slices {
                let logical = *physical_to_logical
                    .get(&slice.tile)
                    .ok_or("host binding references a physical tile outside the topology")?;
                relocate_boundary_address(
                    &mut slice.tile_address,
                    &home_relocations_by_tile[usize::from(logical)],
                    boundary,
                )?;
            }
        }
    }
    graph.schedule.validate_allocations()?;
    Ok(moved_storage)
}

#[cfg(test)]
fn relocate_transient_allocations_for_executables(
    graph: &mut ExecutableGraph,
    topology: &Topology,
    reservations: &[[(u32, u32); 2]],
) -> Result<usize> {
    let element = ipu_package::TILE_MEMORY_ELEMENT_SIZE;
    let reservations = reservations
        .iter()
        .map(|ranges| {
            ranges
                .iter()
                .map(|&(start, end)| (align_down(start, element), align_up(end, element)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    compact_transient_allocations_around(
        graph,
        topology,
        &reservations,
        None,
        "measured executable placement",
    )
}

#[derive(Clone, Copy)]
enum GraphBoundary {
    Input,
    Output,
}

fn relocate_boundary_address(
    address: &mut u32,
    relocations: &[&AllocationRelocation],
    boundary: GraphBoundary,
) -> Result<()> {
    let candidates = relocations
        .iter()
        .filter(|relocation| relocation.old.contains(address))
        .collect::<Vec<_>>();
    let selected_lifetime = match boundary {
        GraphBoundary::Input => candidates.iter().map(|entry| entry.live_from).min(),
        GraphBoundary::Output => candidates.iter().map(|entry| entry.live_until).max(),
    };
    let Some(selected_lifetime) = selected_lifetime else {
        return Ok(());
    };
    let relocated = candidates
        .into_iter()
        .filter(|relocation| match boundary {
            GraphBoundary::Input => relocation.live_from == selected_lifetime,
            GraphBoundary::Output => relocation.live_until == selected_lifetime,
        })
        .map(|relocation| relocation.new_start + (*address - relocation.old.start))
        .collect::<BTreeSet<_>>();
    if relocated.len() != 1 {
        let boundary = match boundary {
            GraphBoundary::Input => "input",
            GraphBoundary::Output => "output",
        };
        return Err(format!(
            "address 0x{address:x} has ambiguous allocation ownership at the graph {boundary} boundary"
        )
        .into());
    }
    *address = *relocated.first().unwrap();
    Ok(())
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
    pub memory_policy: Option<ipu_compiler::MemoryPolicy>,
    pub initial_buffers: Vec<InitialBuffer>,
    pub outputs: Vec<Binding>,
    pub host_weights: Vec<Binding>,
    pub host_inputs: Vec<Binding>,
    pub host_outputs: Vec<Binding>,
}

#[derive(Clone, Debug)]
pub struct StaticTemplateRegion {
    pub name: String,
    pub phase_instances: Vec<Range<usize>>,
}

impl From<ipu_compiler::RepeatedRegion> for StaticTemplateRegion {
    fn from(region: ipu_compiler::RepeatedRegion) -> Self {
        Self {
            name: region.name,
            phase_instances: region.phase_instances,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StaticProfileRegion {
    pub name: String,
    pub phases: Range<usize>,
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
    package_graph_impl(graph, objects, &[], None, &[], 1)
}

pub fn package_graph_owned(graph: ExecutableGraph, objects: &[Vec<u8>]) -> Result<Application> {
    package_graph_impl_owned(graph, objects, &[], None, &[], 1)
}

pub fn package_graph_repeated(
    graph: &ExecutableGraph,
    objects: &[Vec<u8>],
    invocations: u32,
) -> Result<Application> {
    if invocations == 0 {
        return Err("graph invocation count must be nonzero".into());
    }
    package_graph_impl(graph, objects, &[], None, &[], invocations)
}

pub fn package_graph_repeated_owned(
    graph: ExecutableGraph,
    objects: &[Vec<u8>],
    invocations: u32,
) -> Result<Application> {
    if invocations == 0 {
        return Err("graph invocation count must be nonzero".into());
    }
    package_graph_impl_owned(graph, objects, &[], None, &[], invocations)
}

pub fn package_graph_with_templates(
    graph: &ExecutableGraph,
    objects: &[Vec<u8>],
    templates: &[StaticTemplateRegion],
) -> Result<Application> {
    package_graph_impl(graph, objects, &[], None, templates, 1)
}

pub fn package_graph_with_templates_owned(
    graph: ExecutableGraph,
    objects: &[Vec<u8>],
    templates: &[StaticTemplateRegion],
) -> Result<Application> {
    package_graph_impl_owned(graph, objects, &[], None, templates, 1)
}

pub fn package_graph_repeated_with_templates(
    graph: &ExecutableGraph,
    objects: &[Vec<u8>],
    templates: &[StaticTemplateRegion],
    invocations: u32,
) -> Result<Application> {
    if invocations == 0 {
        return Err("graph invocation count must be nonzero".into());
    }
    package_graph_impl(graph, objects, &[], None, templates, invocations)
}

pub fn package_graph_repeated_with_templates_owned(
    graph: ExecutableGraph,
    objects: &[Vec<u8>],
    templates: &[StaticTemplateRegion],
    invocations: u32,
) -> Result<Application> {
    if invocations == 0 {
        return Err("graph invocation count must be nonzero".into());
    }
    package_graph_impl_owned(graph, objects, &[], None, templates, invocations)
}

pub fn allocator_memory_profile(graph: &ExecutableGraph) -> Result<MemoryProfile> {
    let topology = Topology::c600();
    let mut binding_intervals = vec![Vec::<(u32, u32, &str)>::new(); topology.tile_count()];
    for binding in graph
        .host_weights
        .iter()
        .chain(&graph.host_inputs)
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
                    ipu_compiler::AllocationKind::HomeAlias { .. } => "home_alias",
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
        regions.extend(
            graph
                .schedule
                .phases
                .iter()
                .enumerate()
                .flat_map(|(phase, entry)| match entry {
                    ipu_compiler::Phase::Exchange { transfers } => transfers
                        .iter()
                        .filter(move |transfer| transfer.destination_tile == logical_tile)
                        .filter_map(move |transfer| {
                            transfer.staging_address.map(|address| MemoryRegion {
                                address,
                                size: transfer.bytes,
                                category: "exchange_staging".into(),
                                name: format!("exchange staging tensor {}", transfer.tensor.0),
                                tensor: Some(transfer.tensor.0),
                                live_from: phase,
                                live_until: phase + 1,
                            })
                        })
                        .collect::<Vec<_>>(),
                    ipu_compiler::Phase::Compute { .. } => Vec::new(),
                }),
        );
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
        .unwrap_or_else(|| command.specialization.operation.to_string());
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
            .unwrap_or_else(|| command.specialization.operation.to_string());
        return format!("input to {label}");
    }
    match allocation.kind {
        ipu_compiler::AllocationKind::Home => format!("tensor {} home", allocation.tensor.0),
        ipu_compiler::AllocationKind::HomeAlias { source } => {
            format!("tensor {} alias of {}", allocation.tensor.0, source.0)
        }
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

pub fn package_graph_repeated_with_templates_profiled_regions(
    graph: &ExecutableGraph,
    objects: &[Vec<u8>],
    templates: &[StaticTemplateRegion],
    regions: &[StaticProfileRegion],
    invocations: u32,
) -> Result<(Application, ProfileLayout)> {
    if invocations == 0 {
        return Err("profiled graph invocation count must be nonzero".into());
    }
    package_graph_with_profile_options(
        graph,
        objects,
        ProfileSelection::Regions(regions),
        templates,
        invocations,
    )
}

pub fn package_graph_repeated_with_templates_profiled_with(
    graph: &ExecutableGraph,
    objects: &[Vec<u8>],
    templates: &[StaticTemplateRegion],
    granularity: ProfileGranularity,
    invocations: u32,
) -> Result<(Application, ProfileLayout)> {
    if invocations == 0 {
        return Err("profiled graph invocation count must be nonzero".into());
    }
    package_graph_with_profile_options(
        graph,
        objects,
        ProfileSelection::Granularity(granularity, None),
        templates,
        invocations,
    )
}

pub fn package_graph_repeated_with_templates_profiled_with_regions(
    graph: &ExecutableGraph,
    objects: &[Vec<u8>],
    templates: &[StaticTemplateRegion],
    regions: &[StaticProfileRegion],
    granularity: ProfileGranularity,
    invocations: u32,
) -> Result<(Application, ProfileLayout)> {
    if invocations == 0 {
        return Err("profiled graph invocation count must be nonzero".into());
    }
    package_graph_with_profile_options(
        graph,
        objects,
        ProfileSelection::Granularity(granularity, Some(regions)),
        templates,
        invocations,
    )
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
            kernel.to_string()
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
        kernel: kernel.to_string(),
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
                .unwrap_or_else(|| kernel.to_string());
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
                kernel: kernel.to_string(),
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

fn region_profile_steps(
    schedule: &ipu_compiler::Schedule,
    program: &ipu_compiler::LoweredTileProgram,
    regions: &[StaticProfileRegion],
) -> Result<(Vec<ipu_package::ProfileStep>, Vec<ProfileBoundary>)> {
    if regions.is_empty()
        || regions[0].phases.start != 0
        || regions.last().unwrap().phases.end != schedule.phases.len()
        || regions.iter().any(|region| region.phases.is_empty())
        || regions
            .windows(2)
            .any(|pair| pair[0].phases.end != pair[1].phases.start)
    {
        return Err(
            "profile regions must be nonempty and exactly partition the graph phases".into(),
        );
    }
    let mut boundaries = vec![ProfileBoundary::default(); program.steps.len()];
    let mut steps = Vec::with_capacity(regions.len());
    for (local_index, region) in regions.iter().enumerate() {
        let boundary = program
            .steps
            .iter()
            .rposition(|step| lowered_step_phase(step) < region.phases.end)
            .ok_or("profile region has no lowered tile step")?;
        if lowered_step_phase(&program.steps[boundary]) < region.phases.start {
            return Err(format!("profile region {} has no lowered tile step", region.name).into());
        }
        if boundaries[boundary].after_step {
            return Err("multiple profile regions end at the same tile step".into());
        }
        boundaries[boundary].after_step = true;
        steps.push(ipu_package::ProfileStep {
            local_index: u32::try_from(local_index)?,
            phase: u32::try_from(region.phases.start)?,
            epoch: 0,
            operation: region.name.clone(),
            kind: ipu_package::ProfileStepKind::Compute,
            kernel: region.name.clone(),
            metadata: vec![
                profile_metadata("phase_start", region.phases.start),
                profile_metadata("phase_end", region.phases.end),
            ],
        });
    }
    Ok((steps, boundaries))
}

#[derive(Clone, Copy)]
enum ProfileSelection<'a> {
    Granularity(ProfileGranularity, Option<&'a [StaticProfileRegion]>),
    Regions(&'a [StaticProfileRegion]),
}

struct ProfileRelayout {
    graph: ExecutableGraph,
}

impl std::fmt::Debug for ProfileRelayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("profile graph requires final-layout lowering")
    }
}

impl std::fmt::Display for ProfileRelayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("profile graph requires final-layout lowering")
    }
}

impl std::error::Error for ProfileRelayout {}

struct PackageRelayout {
    graph: ExecutableGraph,
}

impl std::fmt::Debug for PackageRelayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("package requires final-layout lowering")
    }
}

impl std::fmt::Display for PackageRelayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("package requires final-layout lowering")
    }
}

impl std::error::Error for PackageRelayout {}

fn annotate_semantic_regions(
    steps: &mut [ipu_package::ProfileStep],
    regions: &[StaticProfileRegion],
) -> Result<()> {
    for step in steps {
        let phase = usize::try_from(step.phase)?;
        let index = regions.partition_point(|region| region.phases.end <= phase);
        let region = regions
            .get(index)
            .filter(|region| region.phases.contains(&phase))
            .ok_or_else(|| format!("profile phase {phase} is outside the semantic regions"))?;
        step.metadata
            .push(profile_metadata("semantic_region", &region.name));
    }
    Ok(())
}

fn package_graph_with_profile(
    graph: &ExecutableGraph,
    objects: &[Vec<u8>],
    granularity: ProfileGranularity,
) -> Result<(Application, ProfileLayout)> {
    package_graph_with_profile_options(
        graph,
        objects,
        ProfileSelection::Granularity(granularity, None),
        &[],
        1,
    )
}

fn package_graph_with_profile_options(
    graph: &ExecutableGraph,
    objects: &[Vec<u8>],
    selection: ProfileSelection<'_>,
    templates: &[StaticTemplateRegion],
    invocations: u32,
) -> Result<(Application, ProfileLayout)> {
    let topology = Topology::c600();
    let profile_started = Instant::now();
    let programs = graph.schedule.lower_tile_programs(&topology)?;
    let lowering_elapsed = profile_started.elapsed();
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
    let aggregate = matches!(
        selection,
        ProfileSelection::Granularity(ProfileGranularity::Graph, _)
    );
    let profile_tensor_base = graph
        .schedule
        .allocations
        .maximum_tensor_id()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or("profile tensor id overflow")?;
    struct PreparedProfileTile {
        steps: Vec<ipu_package::ProfileStep>,
        boundaries: Vec<ProfileBoundary>,
        size: u32,
        address: u32,
        physical_tile: u32,
    }
    let profile_arena = ipu_compiler::MemoryArena::high(
        PLAN_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES,
        ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
    );
    let profile_occupied = graph.schedule.allocations.all_occupied_intervals_by_tile(
        graph.schedule.tile_count,
        profile_arena.base,
        profile_arena.limit,
    );
    let prepared = programs
        .par_iter()
        .map(|program| -> Result<PreparedProfileTile> {
            let (mut steps, boundaries) = match selection {
                ProfileSelection::Granularity(granularity, regions) => {
                    let (mut steps, boundaries) =
                        profile_steps(&graph.schedule, program, granularity);
                    if let Some(regions) = regions {
                        annotate_semantic_regions(&mut steps, regions)?;
                    }
                    (steps, boundaries)
                }
                ProfileSelection::Regions(regions) => {
                    region_profile_steps(&graph.schedule, program, regions)?
                }
            };
            if aggregate {
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
            let sample_count = if aggregate {
                1
            } else {
                steps
                    .len()
                    .checked_add(1)
                    .ok_or("profile sample count overflow")?
            };
            let sample_bytes = if aggregate { 8 } else { 4 };
            let size = u32::try_from(
                sample_count
                    .checked_mul(sample_bytes)
                    .ok_or("profile size overflow")?,
            )?;
            if size == 0 {
                return Err("profile contains no sample storage".into());
            }
            let mut occupied = profile_occupied[usize::from(program.tile)].clone();
            let address = ipu_compiler::allocate_from_occupied_arenas(
                &mut occupied,
                size,
                std::slice::from_ref(&profile_arena),
                8,
            )?;
            Ok(PreparedProfileTile {
                steps,
                boundaries,
                size,
                address,
                physical_tile: u32::from(topology.physical(program.tile)?),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    info!(
        tile_lowering_ms = lowering_elapsed.as_millis(),
        profile_layout_ms = (profile_started.elapsed() - lowering_elapsed).as_millis(),
        tiles = prepared.len(),
        "prepared all-tile profile layout"
    );
    let mut profile_graph = graph.clone();
    let mut profile_code = Vec::with_capacity(programs.len());
    let mut profile_tiles = Vec::with_capacity(programs.len());
    let mut slices = Vec::with_capacity(programs.len());
    let mut file_offset = 0usize;
    for (program, prepared) in programs.iter().zip(prepared) {
        let profile_tensor =
            ipu_compiler::TensorId(profile_tensor_base + usize::from(program.tile));
        profile_graph
            .schedule
            .allocations
            .push(ipu_compiler::Allocation {
                tensor: profile_tensor,
                tile: program.tile,
                address: prepared.address,
                size: prepared.size,
                live_from: 0,
                live_until: usize::MAX,
                kind: ipu_compiler::AllocationKind::Home,
            });
        let after_sync = prepared
            .boundaries
            .iter()
            .map(|boundary| boundary.after_sync)
            .collect::<Vec<_>>();
        let after_step = prepared
            .boundaries
            .iter()
            .map(|boundary| boundary.after_step)
            .collect::<Vec<_>>();
        if !aggregate {
            let boundary_count = prepared
                .boundaries
                .iter()
                .map(|boundary| usize::from(boundary.after_sync) + usize::from(boundary.after_step))
                .sum::<usize>();
            debug_assert_eq!(boundary_count, prepared.steps.len());
        }
        profile_code.push(static_codegen::ProfileCode {
            allocation: Some(profile_tensor),
            initial: prepared.address,
            after_sync,
            after_step,
            aggregate_end: aggregate.then_some(prepared.address + 4),
        });
        slices.push(RegionSlice {
            tile: prepared.physical_tile,
            tile_address: prepared.address,
            file_offset: file_offset as u64,
            size: u64::from(prepared.size),
        });
        profile_tiles.push(ProfileTileLayout {
            physical_tile: prepared.physical_tile,
            file_offset,
            steps: prepared.steps,
            boundary_samples: !aggregate,
        });
        file_offset += prepared.size as usize;
    }
    profile_graph.host_outputs.push(Binding {
        name: "runtime-profile".into(),
        dtype: "u32".into(),
        shape: vec![(file_offset / 4) as u32],
        slices,
    });
    let app = match package_graph_impl_owned(
        profile_graph,
        objects,
        &profile_code,
        Some(programs),
        templates,
        invocations,
    ) {
        Ok(app) => app,
        Err(error) => match error.downcast::<ProfileRelayout>() {
            Ok(relayout) => {
                let mut relocated = relayout.graph;
                relocated
                    .schedule
                    .allocations
                    .retain(|allocation| allocation.tensor.0 < profile_tensor_base);
                let binding = relocated
                    .host_outputs
                    .pop()
                    .ok_or("relocated profile graph lost its profile binding")?;
                if binding.name != "runtime-profile" {
                    return Err("relocated profile graph has an unexpected final binding".into());
                }
                info!("rebuilding profile boundaries after final memory placement");
                return package_graph_with_profile_options(
                    &relocated,
                    objects,
                    selection,
                    templates,
                    invocations,
                );
            }
            Err(error) => return Err(error),
        },
    };
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
    invocations: u32,
) -> Result<Application> {
    package_graph_impl_owned(
        graph.clone(),
        objects,
        profile_code,
        lowered_programs,
        template_regions,
        invocations,
    )
}

fn package_graph_impl_owned(
    mut graph: ExecutableGraph,
    objects: &[Vec<u8>],
    profile_code: &[static_codegen::ProfileCode],
    mut lowered_programs: Option<Vec<ipu_compiler::LoweredTileProgram>>,
    template_regions: &[StaticTemplateRegion],
    invocations: u32,
) -> Result<Application> {
    let mut executable_placement_history = Vec::new();
    loop {
        match package_graph_impl_attempt(
            graph,
            objects,
            profile_code,
            lowered_programs.take(),
            template_regions,
            invocations,
            &mut executable_placement_history,
        ) {
            Ok(app) => return Ok(app),
            Err(error) => match error.downcast::<PackageRelayout>() {
                Ok(relayout) => graph = relayout.graph,
                Err(error) => return Err(error),
            },
        }
    }
}

fn plan_tile_exchange(
    program: &ipu_compiler::LoweredTileProgram,
    template_regions: &[StaticTemplateRegion],
    enable_compute_runs: bool,
    cyclic_templates: bool,
) -> Result<TileExchangePlans> {
    #[derive(Clone)]
    struct PreparedExchange {
        row: Vec<u32>,
        sender: Option<(usize, u32)>,
        template_key: Option<(usize, usize, usize)>,
    }

    let exchange_count = program
        .steps
        .iter()
        .filter(|step| matches!(step, ipu_compiler::LoweredTileStep::Exchange { .. }))
        .count();
    let mut phase_templates = HashMap::new();
    for (region_index, region) in template_regions.iter().enumerate() {
        for (instance, phases) in region.phase_instances.iter().enumerate() {
            for (relative_phase, phase) in phases.clone().enumerate() {
                phase_templates.insert(phase, (region_index, instance, relative_phase));
            }
        }
    }
    let mut epochs_by_phase = HashMap::<usize, usize>::new();
    let mut prepared = Vec::with_capacity(exchange_count);
    let mut sequences = HashMap::<(usize, usize, usize), Vec<Option<Vec<u32>>>>::new();
    for step in &program.steps {
        let ipu_compiler::LoweredTileStep::Exchange { phase, row, .. } = step else {
            continue;
        };
        let epoch = epochs_by_phase.entry(*phase).or_default();
        let template = phase_templates
            .get(phase)
            .map(|&(region, instance, relative_phase)| (region, instance, relative_phase, *epoch));
        *epoch += 1;
        let mut stored_row = row.to_vec();
        let sender = template
            .is_some()
            .then(|| ipu_exchange::normalize_sender_instruction(&mut stored_row))
            .flatten();
        if let Some(return_word) = stored_row
            .iter()
            .position(|&instruction| instruction == ipu_exchange::RETURN_M10_INSTRUCTION)
        {
            stored_row.truncate(return_word + 1);
        }
        let template_key = template.map(|(region, instance, relative_phase, epoch)| {
            let key = (region, relative_phase, epoch);
            let rows = sequences
                .entry(key)
                .or_insert_with(|| vec![None; template_regions[region].phase_instances.len()]);
            rows[instance] = Some(stored_row.clone());
            key
        });
        prepared.push(PreparedExchange {
            row: stored_row,
            sender,
            template_key,
        });
    }
    let mut dynamic_sequences = HashMap::new();
    for (key, rows) in sequences {
        let rows = rows
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or("template exchange sequence is incomplete")?;
        if rows[1..].iter().any(|row| row != &rows[0]) {
            dynamic_sequences.insert(key, rows.iter().map(Vec::len).max().unwrap_or_default());
        }
    }

    let mut cursor = PLAN_BASE;
    let mut unique = HashMap::<Vec<u32>, u32>::new();
    let mut template_addresses = HashMap::<(usize, usize, usize), u32>::new();
    let mut addresses = Vec::with_capacity(exchange_count);
    let mut rows = Vec::new();
    let mut plan_rows = Vec::with_capacity(exchange_count);
    let mut patches = Vec::with_capacity(exchange_count);
    for exchange in prepared {
        let dynamic_key = exchange
            .template_key
            .filter(|key| dynamic_sequences.contains_key(key));
        let address = if let Some(key) = dynamic_key {
            if let Some(&address) = template_addresses.get(&key) {
                address
            } else {
                cursor = align_up(cursor, 8);
                let address = cursor;
                let mut canonical_row = exchange.row.clone();
                canonical_row.resize(dynamic_sequences[&key], 0);
                cursor = cursor
                    .checked_add(u32::try_from(canonical_row.len() * 4)?)
                    .ok_or("exchange plan address overflow")?;
                template_addresses.insert(key, address);
                rows.push((address, canonical_row));
                address
            }
        } else if let Some(&address) = unique.get(&exchange.row) {
            address
        } else {
            cursor = align_up(cursor, 8);
            let address = cursor;
            cursor = cursor
                .checked_add(u32::try_from(exchange.row.len() * 4)?)
                .ok_or("exchange plan address overflow")?;
            unique.insert(exchange.row.clone(), address);
            rows.push((address, exchange.row.clone()));
            address
        };
        addresses.push(address);
        plan_rows.push(exchange.row);
        patches.push(
            exchange
                .sender
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
        enable_compute_runs,
    )?;
    let (templates, _) = static_codegen::plan_static_templates(
        program,
        &addresses,
        &plan_rows,
        &patches,
        template_regions,
        0,
        cyclic_templates,
    )?;
    let mut kernel_symbols = program
        .steps
        .iter()
        .filter_map(|step| match step {
            ipu_compiler::LoweredTileStep::Compute(command) => {
                Some(format!("ipu_stack_{}", command.specialization.operation))
            }
            _ => None,
        })
        .chain(
            templates
                .iter()
                .flat_map(static_codegen::template_retained_symbols),
        )
        .collect::<Vec<_>>();
    kernel_symbols.sort_unstable();
    kernel_symbols.dedup();
    Ok(TileExchangePlans {
        addresses,
        rows,
        compute_runs,
        templates,
        kernel_symbols,
        end,
    })
}

fn merge_kernel_memory_constraints(
    constraints: impl IntoIterator<Item = ipu_compiler::ResolvedKernelMemoryConstraints>,
) -> ipu_compiler::ResolvedKernelMemoryConstraints {
    let mut merged = ipu_compiler::ResolvedKernelMemoryConstraints::default();
    for mut constraints in constraints {
        merged.classes.append(&mut constraints.classes);
        merged.separations.append(&mut constraints.separations);
    }
    merged.classes.sort_unstable();
    merged.classes.dedup();
    merged.separations.sort_unstable();
    merged.separations.dedup();
    merged
}

fn validate_kernel_memory_constraints(
    constraints: &ipu_compiler::ResolvedKernelMemoryConstraints,
    executable_elements: &[Vec<(u32, u32)>],
) -> Result<()> {
    let validate_address = |operand: ipu_compiler::ResolvedMemoryOperand| -> Result<()> {
        let ranges = executable_elements
            .get(usize::from(operand.tile))
            .ok_or("kernel memory constraint references a tile outside the executable")?;
        let elements = ipu_package::ipu21_effective_memory_elements(operand.address, operand.bytes)
            .ok_or("kernel operand span is outside tile SRAM")?;
        for (_, element_start, element_end) in elements {
            if let Some(&(start, end)) = ranges
                .iter()
                .find(|&&(start, end)| ranges_overlap(element_start, element_end, start, end))
            {
                return Err(format!(
                    "kernel operand on tile {} at 0x{:x}..0x{:x} touches executable element 0x{start:x}..0x{end:x}",
                    operand.tile,
                    operand.address,
                    operand.address.saturating_add(operand.bytes),
                )
                .into());
            }
        }
        Ok(())
    };
    for class in &constraints.classes {
        validate_address(class.operand)?;
        match class.class {
            ipu_compiler::KernelMemoryClass::Ipu21Interleaved => {
                let end = class
                    .operand
                    .address
                    .checked_add(class.operand.bytes)
                    .ok_or("interleaved kernel operand span overflow")?;
                if class.operand.address < ipu_package::IPU21_INTERLEAVED_MEMORY_BASE
                    || end > ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT
                {
                    return Err(format!(
                        "kernel operand on tile {} at 0x{:x}..0x{end:x} is outside required interleaved memory",
                        class.operand.tile, class.operand.address,
                    )
                    .into());
                }
            }
        }
    }
    for separation in &constraints.separations {
        validate_address(separation.first)?;
        validate_address(separation.second)?;
        if separation.first.tile == separation.second.tile
            && memory_spans_share_effective_element(
                separation.first.address,
                separation.first.bytes,
                separation.second.address,
                separation.second.bytes,
            )?
        {
            return Err(format!(
                "kernel operand spans on tile {} at 0x{:x}..0x{:x} and 0x{:x}..0x{:x} share an effective memory element",
                separation.first.tile,
                separation.first.address,
                separation.first.address.saturating_add(separation.first.bytes),
                separation.second.address,
                separation.second.address.saturating_add(separation.second.bytes),
            )
            .into());
        }
    }
    Ok(())
}

fn package_graph_impl_attempt(
    mut graph: ExecutableGraph,
    objects: &[Vec<u8>],
    profile_code: &[static_codegen::ProfileCode],
    lowered_programs: Option<Vec<ipu_compiler::LoweredTileProgram>>,
    template_regions: &[StaticTemplateRegion],
    invocations: u32,
    executable_placement_history: &mut Vec<Vec<[(u32, u32); 2]>>,
) -> Result<Application> {
    let topology = Topology::c600();
    if usize::from(graph.schedule.tile_count) != topology.tile_count() {
        return Err("the direct C600 runtime requires a schedule for every discovered tile".into());
    }
    validate_resident_host_bindings(&graph, &topology)?;
    graph.schedule.validate_allocations()?;
    let stream_templates =
        lowered_programs.is_none() && profile_code.is_empty() && !template_regions.is_empty();
    let (mut programs, mut tile_exchange_plans, resolved_memory_constraints) = if stream_templates {
        let prepare_started = Instant::now();
        let lowering = graph.schedule.prepare_tile_program_lowering(&topology)?;
        let prepare_elapsed = prepare_started.elapsed();
        let lowered = (0..graph.schedule.tile_count)
            .into_par_iter()
            .map(|tile| -> Result<_> {
                let mut program = lowering.lower(tile, false)?;
                let memory_constraints = graph
                    .schedule
                    .resolve_memory_constraints(std::slice::from_ref(&program))?;
                let mut plans =
                    plan_tile_exchange(&program, template_regions, false, invocations > 1)?;
                static_codegen::compact_template_instances(&mut program, &mut plans.templates)?;
                Ok((program, plans, memory_constraints))
            })
            .collect::<Result<Vec<_>>>()?;
        info!(
            prepare_ms = prepare_elapsed.as_millis(),
            lower_and_compact_ms = (prepare_started.elapsed() - prepare_elapsed).as_millis(),
            total_ms = prepare_started.elapsed().as_millis(),
            "lowered and compacted tile programs"
        );
        let mut programs = Vec::with_capacity(lowered.len());
        let mut plans = Vec::with_capacity(lowered.len());
        let mut constraints = Vec::with_capacity(lowered.len());
        for (program, plan, memory_constraints) in lowered {
            programs.push(program);
            plans.push(plan);
            constraints.push(memory_constraints);
        }
        (
            programs,
            plans,
            merge_kernel_memory_constraints(constraints),
        )
    } else {
        let programs = match lowered_programs {
            Some(programs) => programs,
            None => graph.schedule.lower_tile_programs_for_codegen(&topology)?,
        };
        let plans = programs
            .par_iter()
            .map(|program| {
                plan_tile_exchange(
                    program,
                    template_regions,
                    profile_code.is_empty() && template_regions.is_empty(),
                    invocations > 1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let memory_constraints = graph.schedule.resolve_memory_constraints(&programs)?;
        (programs, plans, memory_constraints)
    };
    info!(
        memory_classes = resolved_memory_constraints.classes.len(),
        memory_separations = resolved_memory_constraints.separations.len(),
        "resolved kernel memory constraints"
    );
    let relocation_constraints =
        relocation_memory_constraints(&graph, &resolved_memory_constraints)?;
    if !profile_code.is_empty() && profile_code.len() != programs.len() {
        return Err("profile layout tile count differs from schedule".into());
    }
    let profile_code = profile_code
        .iter()
        .zip(&programs)
        .map(|(code, program)| -> Result<_> {
            let mut resolved = code.clone();
            let Some(tensor) = code.allocation else {
                return Ok(resolved);
            };
            let allocation = graph
                .schedule
                .allocations
                .iter()
                .find(|allocation| {
                    allocation.tensor == tensor
                        && allocation.tile == program.tile
                        && matches!(allocation.kind, ipu_compiler::AllocationKind::Home)
                })
                .ok_or_else(|| {
                    format!(
                        "profile storage tensor {} is not allocated on tile {}",
                        tensor.0, program.tile
                    )
                })?;
            let aggregate_offset = code.aggregate_end.map(|end| end - code.initial);
            resolved.initial = allocation.address;
            resolved.aggregate_end = match aggregate_offset {
                Some(offset) => Some(
                    allocation
                        .address
                        .checked_add(offset)
                        .ok_or("relocated profile counter address overflow")?,
                ),
                None => None,
            };
            Ok(resolved)
        })
        .collect::<Result<Vec<_>>>()?;
    let allocation_ranges_by_tile = allocation_footprints_by_tile(
        &graph,
        usize::from(graph.schedule.tile_count),
        Some(&relocation_constraints),
    )?;
    let fixed_allocation_ranges =
        fixed_allocation_ranges_by_tile(&graph, topology.tile_count(), &relocation_constraints)?;
    let immovable_allocation_ranges = immovable_allocation_ranges_by_tile(
        &graph,
        topology.tile_count(),
        &relocation_constraints,
    )?;
    let exchange_count = tile_exchange_plans
        .first()
        .map(|plans| plans.addresses.len())
        .unwrap_or(0);
    if tile_exchange_plans
        .iter()
        .any(|plans| plans.addresses.len() != exchange_count)
    {
        return Err("per-tile programs disagree on exchange launch count".into());
    }
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
    if !stream_templates && profile_code.is_empty() && !template_regions.is_empty() {
        for (program, plans) in programs.iter_mut().zip(&mut tile_exchange_plans) {
            static_codegen::compact_template_instances(program, &mut plans.templates)?;
        }
    }
    let host = build_static_host_layout(&graph, invocations)?;
    let host_transfers = host
        .weights
        .iter()
        .chain(&host.inputs)
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
        .par_iter()
        .zip(&tile_exchange_plans)
        .map(|(program, exchange_plans)| -> Result<TileHostPlans> {
            let plan_end = exchange_plans.end;
            let physical = topology.physical(program.tile)?;
            let follower_address = align_up(plan_end, 64);
            let mut executable_objects = vec![follower_address..follower_address + 3 * 4];
            let ordinary_data_objects = Vec::new();
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
                        executable_objects.push(address..cursor);
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
            let mut run_table_addresses = HashMap::<Vec<u32>, u32>::new();
            let weight_end = host.weights.len();
            let input_end = weight_end + host.inputs.len();
            for range in [
                0..weight_end,
                weight_end..input_end,
                input_end..host_transfers.len(),
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
                        && addresses[index] == addresses[start]
                    {
                        index += 1;
                    }
                    let descriptors = host_run_descriptor_words(
                        physical,
                        &host_transfers[start..index],
                        &packet_copies[start..index],
                    )?;
                    let address = if let Some(&address) = run_table_addresses.get(&descriptors) {
                        address
                    } else {
                        cursor = align_up(cursor, 4);
                        let address = cursor;
                        cursor = cursor
                            .checked_add(
                                u32::try_from(descriptors.len())?
                                    .checked_mul(4)
                                    .ok_or("static host run descriptor size overflow")?,
                            )
                            .ok_or("static host run descriptor address overflow")?;
                        data_objects.push(address..cursor);
                        run_table_addresses.insert(descriptors, address);
                        address
                    };
                    run_tables[start] = Some(address);
                }
            }
            Ok(TileHostPlans {
                start: follower_address,
                executable_objects,
                ordinary_data_objects,
                data_objects,
                addresses,
                packet_copies,
                run_tables,
                end: cursor,
            })
        })
        .collect::<Result<Vec<_>>>()?;
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
    if !template_regions.is_empty() {
        runtime_symbols.push(static_codegen::TEMPLATE_PATCH.into());
    }
    let tile_retained_symbols = programs
        .par_iter()
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
            symbols.extend(
                exchange_plans
                    .templates
                    .iter()
                    .flat_map(static_codegen::template_retained_symbols),
            );
            symbols.extend(exchange_plans.kernel_symbols.iter().cloned());
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
                    regions: Vec::new(),
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
    info!(
        tiles = preliminary_images.len(),
        "linked preliminary tile support images"
    );
    // Instruction fetch and data access cannot share a tile-memory element. Measure the
    // address-invariant generated code before relocating host state so data packing can
    // preserve enough complete elements for it.
    let preliminary_program_sizes = programs
        .par_iter()
        .enumerate()
        .map(|(program_index, program)| -> Result<u32> {
            let mut symbols = BTreeMap::new();
            for name in [
                static_codegen::WORKER_BARRIER,
                static_codegen::COMPLETE,
                static_codegen::HOST_RUN,
                static_codegen::REPEAT_CALL,
                static_codegen::EXCHANGE_COMPUTE_RUN,
                static_codegen::TEMPLATE_PATCH,
                static_codegen::SAMPLE_CYCLE,
                static_codegen::SAMPLE_CYCLE_NEXT,
            ] {
                symbols.insert(name.into(), ipu_driver::APPLICATION_LOAD_BASE);
            }
            for step in &program.steps {
                if let ipu_compiler::LoweredTileStep::Compute(command) = step {
                    symbols.insert(
                        format!("ipu_stack_{}", command.specialization.operation),
                        ipu_driver::APPLICATION_LOAD_BASE,
                    );
                }
            }
            for name in &tile_exchange_plans[program_index].kernel_symbols {
                symbols.insert(name.clone(), ipu_driver::APPLICATION_LOAD_BASE);
            }
            let plans = &tile_host_plans[program_index];
            let physical = topology.physical(program.tile)?;
            let weight_end = host.weights.len();
            let input_end = weight_end + host.inputs.len();
            let host_weights = host_phase_calls(plans, physical, 0, &host.weights);
            let host_inputs = host_phase_calls(plans, physical, weight_end, &host.inputs);
            let host_outputs = host_phase_calls(plans, physical, input_end, &host.outputs);
            Ok(u32::try_from(
                static_codegen::emit(
                    program,
                    &symbols,
                    &tile_exchange_plans[program_index].addresses,
                    &tile_exchange_plans[program_index].compute_runs,
                    &tile_exchange_plans[program_index].templates,
                    static_codegen::HostCode {
                        weights: &host_weights,
                        inputs: &host_inputs,
                        outputs: &host_outputs,
                    },
                    profile_code.get(program_index),
                    0,
                    invocations,
                )?
                .len(),
            )?)
        })
        .collect::<Result<Vec<_>>>()?;
    info!(
        tiles = preliminary_program_sizes.len(),
        "measured generated tile programs"
    );
    let program_reservation_sizes = preliminary_program_sizes
        .iter()
        .zip(&tile_host_plans)
        .map(|(&program_size, plans)| -> Result<u32> {
            plans
                .executable_objects
                .iter()
                .try_fold(program_size, |size, object| {
                    let object_size = u32::try_from(object.len())?;
                    align_up(size, 8)
                        .checked_add(object_size)
                        .ok_or_else(|| "generated program and host plan size overflow".into())
                })
        })
        .collect::<Result<Vec<_>>>()?;
    // Generated and support code are independently relocatable. Reserving them
    // separately avoids requiring one artificial contiguous executable extent.
    let executable_reservations = if let Some(pinned) = executable_placement_history.last() {
        programs
            .par_iter()
            .zip(&tile_exchange_plans)
            .zip(&program_reservation_sizes)
            .zip(&preliminary_images)
            .zip(pinned)
            .map(
                |((((program, plans), &program_size), image), &pinned)| -> Result<_> {
                    let regions = executable_regions_for_tile(
                        &allocation_ranges_by_tile[usize::from(program.tile)],
                        plans.end,
                        &[],
                    )?;
                    validate_pinned_executable_images(
                        program.tile,
                        program_size,
                        u32::try_from(image.bytes.len())?,
                        pinned,
                        &regions,
                    )
                },
            )
            .collect::<Result<Vec<_>>>()?
    } else {
        let attempted = programs
            .par_iter()
            .zip(&tile_exchange_plans)
            .zip(&program_reservation_sizes)
            .zip(&preliminary_images)
            .map(
                |(((program, plans), &program_size), image)| -> Result<[(u32, u32); 2]> {
                    let regions = executable_regions_for_tile(
                        &allocation_ranges_by_tile[usize::from(program.tile)],
                        plans.end,
                        &[],
                    )?;
                    pack_generated_and_support_images(
                        program.tile,
                        program_size,
                        u32::try_from(image.bytes.len())?,
                        regions,
                    )
                },
            )
            .collect::<Result<Vec<_>>>();
        match attempted {
            Ok(reservations) => reservations,
            Err(error) => {
                let place = |allocation_ranges: &[Vec<(u32, u32)>]| {
                    programs
                    .par_iter()
                    .zip(&tile_exchange_plans)
                    .zip(&program_reservation_sizes)
                    .zip(&preliminary_images)
                    .map(
                        |(((program, plans), &program_size), image)| -> Result<[(u32, u32); 2]> {
                            let regions = executable_regions_for_tile(
                                &allocation_ranges[usize::from(program.tile)],
                                plans.end,
                                &[],
                            )?;
                            let placed = pack_generated_and_support_images(
                                program.tile,
                                program_size,
                                u32::try_from(image.bytes.len())?,
                                regions,
                            )?;
                            Ok(placed)
                        },
                    )
                    .collect::<Result<Vec<_>>>()
                };
                let (desired, move_resident) = match place(&fixed_allocation_ranges) {
                    Ok(desired) => (desired, false),
                    Err(_) => (place(&immovable_allocation_ranges)?, true),
                };
                executable_placement_history.push(desired.clone());
                let element = ipu_package::TILE_MEMORY_ELEMENT_SIZE;
                let reservations = desired
                    .iter()
                    .enumerate()
                    .map(|(tile, ranges)| {
                        ranges
                            .iter()
                            .map(|&(start, end)| {
                                (align_down(start, element), align_up(end, element))
                            })
                            .chain(std::iter::once((PLAN_BASE, tile_exchange_plans[tile].end)))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let moved = if move_resident {
                    compact_all_allocations_around(
                        &mut graph,
                        &topology,
                        &reservations,
                        Some(&resolved_memory_constraints),
                        "measured executable placement",
                    )?
                } else {
                    compact_transient_allocations_around(
                        &mut graph,
                        &topology,
                        &reservations,
                        Some(&resolved_memory_constraints),
                        "measured executable placement",
                    )?
                };
                if moved == 0 {
                    return Err(error);
                }
                info!(
                    moved,
                    move_resident, "relocated transient tensors for measured executable images"
                );
                if !profile_code.is_empty() {
                    return Err(Box::new(ProfileRelayout { graph }));
                }
                return Err(Box::new(PackageRelayout { graph }));
            }
        }
    };
    info!(
        tiles = executable_reservations.len(),
        "placed measured tile executables"
    );
    let program_reservations = executable_reservations
        .iter()
        .map(|reservations| reservations[0])
        .collect::<Vec<_>>();
    let support_reservations = executable_reservations
        .iter()
        .map(|reservations| reservations[1])
        .collect::<Vec<_>>();
    let image_executable_elements = executable_reservations
        .iter()
        .map(|reservations| effective_element_reservations(reservations.iter().copied()))
        .collect::<Result<Vec<_>>>()?;
    let mut host_executable_placements = tile_host_plans
        .iter()
        .enumerate()
        .map(|(tile_index, plans)| {
            let tile = programs[tile_index].tile;
            let program_tail_start = program_reservations[tile_index]
                .0
                .checked_add(preliminary_program_sizes[tile_index])
                .ok_or("generated program tail overflow")?;
            pack_executable_objects_for_tile(
                &allocation_ranges_by_tile[usize::from(tile)],
                tile,
                tile_exchange_plans[tile_index].end,
                &plans.executable_objects,
                &[
                    program_reservations[tile_index],
                    support_reservations[tile_index],
                ],
                &[(program_tail_start, program_reservations[tile_index].1)],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    #[derive(Clone, Copy)]
    enum TemplateDataObject {
        RecordPrimary { template: usize },
        RecordSecondary { template: usize },
        PatchPair { template: usize, instance: usize },
        PatchTable { template: usize },
        Shared { template: usize },
    }
    let mut template_record_ranges: Vec<Vec<(u32, u32)>> = vec![Vec::new(); programs.len()];
    let mut host_runtime_ranges = Vec::with_capacity(tile_host_plans.len());
    let mut worker_sync_addresses = Vec::with_capacity(tile_host_plans.len());
    let mut completion_addresses = Vec::with_capacity(tile_host_plans.len());
    let mut static_relocation_reservations = Vec::with_capacity(programs.len());
    let mut executable_element_reservations = Vec::with_capacity(programs.len());
    for (tile_index, plans) in tile_exchange_plans.iter_mut().enumerate() {
        let tile = programs[tile_index].tile;
        let runtime_end = plans.end;
        let host_plans = &mut tile_host_plans[tile_index];
        let old_worker_sync = align_up(host_plans.end, 8);
        let old_worker_end = old_worker_sync
            .checked_add(WORKER_STACK_HEADROOM + (TILE_CONTEXT_STACKS - 1) * WORKER_SYNC_STRIDE)
            .ok_or("static host runtime address overflow")?;
        host_plans
            .ordinary_data_objects
            .push(old_worker_sync..old_worker_end);
        let (mut relocations, executable_ranges, executable_elements) =
            std::mem::take(&mut host_executable_placements[tile_index]);
        if tile == 0
            && let Some((template, instance, patch_words)) = plans
                .templates
                .iter()
                .enumerate()
                .flat_map(|(template, plan)| {
                    let record_words = plan.records.first().map_or(0, Vec::len);
                    let split = usize::from(plan.record_split);
                    plan.patches
                        .iter()
                        .enumerate()
                        .skip(1)
                        .map(move |(instance, patch)| {
                            let words = static_codegen::template_patch_ranges(record_words, split)
                                .into_iter()
                                .map(|slots| {
                                    static_codegen::template_patch_storage_words_range(slots, patch)
                                        .max(1)
                                })
                                .sum::<usize>();
                            (template, instance, words)
                        })
                })
                .max_by_key(|&(_, _, words)| words)
        {
            let patch = &plans.templates[template].patches[instance];
            let mut delta_counts = BTreeMap::<i16, usize>::new();
            let mut wide_values = 0usize;
            for (_, value) in patch {
                match value {
                    static_codegen::StaticTemplatePatchValue::Delta(delta) => {
                        *delta_counts.entry(*delta).or_default() += 1;
                    }
                    static_codegen::StaticTemplatePatchValue::Delta32(_)
                    | static_codegen::StaticTemplatePatchValue::Difference { .. } => {
                        wide_values += 1
                    }
                }
            }
            let common_delta = delta_counts
                .iter()
                .max_by_key(|(_, count)| **count)
                .map(|(&delta, &count)| (delta, count));
            info!(
                template = plans.templates[template].name,
                instance,
                patch_words,
                changed_words = patch.len(),
                narrow_deltas = patch.len() - wide_values,
                wide_values,
                distinct_deltas = delta_counts.len(),
                ?common_delta,
                "largest static template transition patch"
            );
        }
        let mut objects = Vec::<(TemplateDataObject, usize)>::new();
        let mut canonical_patches = HashMap::<
            (
                usize,
                usize,
                Vec<(u16, static_codegen::StaticTemplatePatchValue)>,
            ),
            (usize, usize),
        >::new();
        let mut patch_aliases = Vec::<(usize, usize, usize, usize)>::new();
        for (template, plan) in plans.templates.iter().enumerate() {
            let record_words = plan.records.first().map_or(0, Vec::len);
            let split = usize::from(plan.record_split);
            objects.extend([
                (TemplateDataObject::RecordPrimary { template }, split),
                (
                    TemplateDataObject::RecordSecondary { template },
                    record_words - split,
                ),
                (
                    TemplateDataObject::PatchTable { template },
                    plan.patches.len().saturating_sub(1) * usize::from(record_words != 0),
                ),
                (TemplateDataObject::Shared { template }, plan.shared.len()),
            ]);
            for (instance, patch) in plan.patches.iter().enumerate().skip(1) {
                if record_words == 0 {
                    continue;
                }
                let key = (record_words, split, plan.patches[instance].clone());
                if let Some(&(canonical_template, canonical_instance)) = canonical_patches.get(&key)
                {
                    patch_aliases.push((
                        template,
                        instance,
                        canonical_template,
                        canonical_instance,
                    ));
                    continue;
                }
                canonical_patches.insert(key, (template, instance));
                let words = static_codegen::template_patch_ranges(record_words, split)
                    .into_iter()
                    .map(|slots| {
                        static_codegen::template_patch_storage_words_range(slots, patch).max(1)
                    })
                    .sum();
                objects.push((TemplateDataObject::PatchPair { template, instance }, words));
            }
        }
        if tile == 0 {
            let patch_words = objects
                .iter()
                .filter_map(|(object, words)| {
                    matches!(object, TemplateDataObject::PatchPair { .. }).then_some(*words)
                })
                .sum::<usize>();
            info!(
                transition_patches = canonical_patches.len() + patch_aliases.len(),
                unique_transition_patches = canonical_patches.len(),
                interned_transition_patches = patch_aliases.len(),
                patch_bytes = patch_words * 4,
                "compacted static template transition patches"
            );
        }

        let mut placed_objects = Vec::new();
        let mut sizes = Vec::new();
        for (object, words) in objects {
            if words == 0 {
                match object {
                    TemplateDataObject::RecordPrimary { template } => plans.templates[template]
                        .record_addresses
                        .fill(ipu_package::TILE_MEMORY_BASE),
                    TemplateDataObject::RecordSecondary { template } => plans.templates[template]
                        .record_secondary_addresses
                        .fill(ipu_package::TILE_MEMORY_BASE),
                    TemplateDataObject::PatchPair { .. } => unreachable!(),
                    TemplateDataObject::PatchTable { template } => {
                        plans.templates[template].patch_table_address =
                            ipu_package::TILE_MEMORY_BASE
                    }
                    TemplateDataObject::Shared { template } => {
                        plans.templates[template].shared_address = ipu_package::TILE_MEMORY_BASE
                    }
                }
                continue;
            }
            sizes.push(
                u32::try_from(words)?
                    .checked_mul(4)
                    .ok_or("static template record size overflow")?,
            );
            placed_objects.push(object);
        }
        let host_data_count = host_plans.data_objects.len();
        let mut data_sizes = host_plans
            .data_objects
            .iter()
            .map(|object| -> Result<u32> { Ok(u32::try_from(object.len())?) })
            .collect::<Result<Vec<_>>>()?;
        data_sizes.extend(sizes);
        let plan_executable_elements =
            effective_element_reservations(std::iter::once((PLAN_BASE, runtime_end)))?;
        let executable_reserved = merge_address_ranges(
            executable_elements
                .iter()
                .copied()
                .chain(image_executable_elements[tile_index].iter().copied())
                .chain(plan_executable_elements)
                .collect(),
        );
        executable_element_reservations.push(executable_reserved.clone());
        let place = |allocation_ranges: &[(u32, u32)]| -> Result<_> {
            // Linker/runtime data can displace ordinary graph allocations, which
            // are subsequently relocated. Interleaved footprints are hardware
            // placement requirements for AMP/PACE kernels and remain reserved.
            let allocation_ranges = static_placement_ranges(
                allocation_ranges,
                &allocation_ranges_by_tile[usize::from(tile)],
            );
            let ordinary_gaps = data_regions_for_tile(
                &allocation_ranges,
                runtime_end,
                false,
                &executable_reserved,
            )?;
            let (ordinary_relocations, ordinary_ranges) = pack_objects_in_gaps(
                tile,
                &host_plans.ordinary_data_objects,
                ordinary_gaps,
                "non-interleaved static data",
            )?;
            let data_reserved = executable_reserved
                .iter()
                .copied()
                .chain(ordinary_ranges.iter().copied())
                .collect::<Vec<_>>();
            let data_gaps =
                data_regions_for_tile(&allocation_ranges, runtime_end, true, &data_reserved)?;
            let data_placements =
                pack_sized_objects_in_gaps(tile, &data_sizes, data_gaps, 4, "static data")?;
            Ok((ordinary_relocations, ordinary_ranges, data_placements))
        };
        let (ordinary_relocations, ordinary_ranges, data_placements) =
            place(&allocation_ranges_by_tile[usize::from(tile)])
                .or_else(|_| place(&fixed_allocation_ranges[usize::from(tile)]))
                .or_else(|_| place(&immovable_allocation_ranges[usize::from(tile)]))?;
        validate_data_element_separation(
            tile,
            "static runtime data",
            &ordinary_ranges
                .iter()
                .copied()
                .chain(data_placements.iter().copied())
                .collect::<Vec<_>>(),
            &executable_reserved,
        )?;
        relocations.extend(ordinary_relocations);
        let (host_data_placements, placements) = data_placements.split_at(host_data_count);
        for (object, &(address, _)) in host_plans.data_objects.iter().zip(host_data_placements) {
            relocations.insert(object.start, address);
        }
        for (object, &(address, end)) in placed_objects.iter().zip(placements) {
            match *object {
                TemplateDataObject::RecordPrimary { template } => {
                    plans.templates[template].record_addresses.fill(address)
                }
                TemplateDataObject::RecordSecondary { template } => plans.templates[template]
                    .record_secondary_addresses
                    .fill(address),
                TemplateDataObject::PatchPair { template, instance } => {
                    let plan = &mut plans.templates[template];
                    let record_words = plan.records.first().map_or(0, Vec::len);
                    let mut cursor = address;
                    let ranges = static_codegen::template_patch_ranges(
                        record_words,
                        usize::from(plan.record_split),
                    );
                    let mut addresses = Vec::with_capacity(ranges.len());
                    for slots in ranges {
                        addresses.push(cursor);
                        let words = static_codegen::template_patch_storage_words_range(
                            slots,
                            &plan.patches[instance],
                        )
                        .max(1);
                        cursor = cursor
                            .checked_add(
                                u32::try_from(words)?
                                    .checked_mul(4)
                                    .ok_or("static patch pair size overflow")?,
                            )
                            .ok_or("static patch pair address overflow")?;
                    }
                    debug_assert_eq!(cursor, end);
                    plan.patch_addresses[instance] = addresses;
                }
                TemplateDataObject::PatchTable { template } => {
                    plans.templates[template].patch_table_address = address
                }
                TemplateDataObject::Shared { template } => {
                    plans.templates[template].shared_address = address
                }
            }
        }
        for (template, instance, canonical_template, canonical_instance) in patch_aliases {
            plans.templates[template].patch_addresses[instance] =
                plans.templates[canonical_template].patch_addresses[canonical_instance].clone();
        }
        template_record_ranges[tile_index].extend_from_slice(placements);

        let relocate = |address: u32| -> Result<u32> {
            relocations
                .get(&address)
                .copied()
                .ok_or_else(|| format!("missing relocation for static object 0x{address:x}").into())
        };
        for address in &mut host_plans.addresses {
            *address = relocate(*address)?;
        }
        for copy in host_plans.packet_copies.iter_mut().flatten() {
            copy.source = relocate(copy.source)?;
        }
        for address in host_plans.run_tables.iter_mut().flatten() {
            *address = relocate(*address)?;
        }
        let worker_sync = relocate(old_worker_sync)?;
        let completion = worker_sync;
        let mut ranges = ordinary_ranges;
        ranges.extend_from_slice(host_data_placements);
        ranges.extend(executable_ranges);
        ranges.sort_unstable();
        host_plans.start = ranges.iter().map(|&(start, _)| start).min().unwrap_or(0);
        host_plans.end = ranges.iter().map(|&(_, end)| end).max().unwrap_or(0);
        host_runtime_ranges.push(ranges);
        worker_sync_addresses.push(worker_sync);
        completion_addresses.push(completion);
        static_relocation_reservations.push(
            data_placements
                .iter()
                .copied()
                .chain(executable_reserved)
                .chain(host_runtime_ranges[tile_index].iter().copied())
                .chain(std::iter::once((PLAN_BASE, runtime_end)))
                .collect::<Vec<_>>(),
        );
        if tile == 0 {
            info!(
                ranges = ?host_runtime_ranges[tile_index],
                first_plan = host_plans.addresses.first().map(|address| format!("0x{address:x}")),
                first_packet = host_plans
                    .packet_copies
                    .first()
                    .and_then(|copy| *copy)
                    .map(|copy| format!("0x{:x}", copy.source)),
                first_run_table = host_plans
                    .run_tables
                    .first()
                    .and_then(|address| *address)
                    .map(|address| format!("0x{address:x}")),
                worker_sync = format_args!("0x{worker_sync:x}"),
                completion = format_args!("0x{completion:x}"),
                "packed unified static runtime"
            );
        }
    }
    let overlaps_static_reservation = |index: usize, allocation: &ipu_compiler::Allocation| {
        let access_extent = relocation_constraints.access_extent(index, allocation.size);
        matches!(allocation.kind, ipu_compiler::AllocationKind::Home)
            && static_relocation_reservations[usize::from(allocation.tile)]
                .iter()
                .any(|&(start, end)| {
                    ranges_overlap(
                        allocation.address,
                        allocation.address.saturating_add(access_extent),
                        start,
                        end,
                    )
                })
    };
    let needs_resident_relocation =
        graph
            .schedule
            .allocations
            .iter()
            .enumerate()
            .any(|(index, allocation)| {
                allocation.live_until == usize::MAX
                    && overlaps_static_reservation(index, allocation)
            });
    let needs_transient_relocation =
        graph
            .schedule
            .allocations
            .iter()
            .enumerate()
            .any(|(index, allocation)| {
                allocation.live_until != usize::MAX
                    && overlaps_static_reservation(index, allocation)
            });
    if needs_resident_relocation || needs_transient_relocation {
        let moved = if needs_resident_relocation {
            compact_all_allocations_around(
                &mut graph,
                &topology,
                &static_relocation_reservations,
                Some(&resolved_memory_constraints),
                "static runtime placement",
            )?
        } else {
            match compact_transient_allocations_around(
                &mut graph,
                &topology,
                &static_relocation_reservations,
                Some(&resolved_memory_constraints),
                "static runtime placement",
            ) {
                Ok(moved) => moved,
                Err(_) if graph.memory_policy.is_some() => compact_all_allocations_around(
                    &mut graph,
                    &topology,
                    &static_relocation_reservations,
                    Some(&resolved_memory_constraints),
                    "static runtime placement",
                )?,
                Err(error) => return Err(error),
            }
        };
        if moved == 0 {
            return Err("static template data overlaps transients that cannot be relocated".into());
        }
        info!(
            moved,
            needs_resident_relocation, "relocated graph allocations for static runtime"
        );
        if !profile_code.is_empty() {
            return Err(Box::new(ProfileRelayout { graph }));
        }
        return Err(Box::new(PackageRelayout { graph }));
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
    if let Some((tile_index, template, words, changed_words, narrow_deltas)) = tile_exchange_plans
        .iter()
        .enumerate()
        .flat_map(|(tile_index, plans)| {
            plans.templates.iter().map(move |template| {
                let words =
                    template.shared.len() + template.records.iter().map(Vec::len).sum::<usize>();
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
                let narrow_deltas = template
                    .records
                    .windows(2)
                    .flat_map(|records| records[0].iter().zip(&records[1]))
                    .filter(|(left, right)| left != right)
                    .filter(|(left, right)| match (left, right) {
                        (
                            static_codegen::StaticTemplateRecordWord::Value(left),
                            static_codegen::StaticTemplateRecordWord::Value(right),
                        ) => i16::try_from(i64::from(*right) - i64::from(*left)).is_ok(),
                        _ => false,
                    })
                    .count();
                (tile_index, template, words, changed_words, narrow_deltas)
            })
        })
        .max_by_key(|(_, _, words, _, _)| *words)
    {
        let tile = programs[tile_index].tile;
        let allocations = graph
            .schedule
            .allocations
            .iter()
            .filter(|allocation| allocation.tile == tile)
            .collect::<Vec<_>>();
        let address_kind = |word: &static_codegen::StaticTemplateRecordWord| {
            let static_codegen::StaticTemplateRecordWord::Value(address) = word else {
                return 0;
            };
            allocations
                .iter()
                .find(|allocation| {
                    *address >= allocation.address
                        && *address < allocation.address.saturating_add(allocation.size)
                })
                .map_or(0, |allocation| {
                    if allocation.live_until == usize::MAX {
                        1
                    } else {
                        2
                    }
                })
        };
        let changed_address_kinds = template
            .records
            .windows(2)
            .flat_map(|records| records[0].iter().zip(&records[1]))
            .filter(|(left, right)| left != right)
            .fold([0usize; 4], |mut counts, (left, right)| {
                let left = address_kind(left);
                let right = address_kind(right);
                let kind = if left == right { left } else { 3 };
                counts[kind] += 1;
                counts
            });
        info!(
            logical_tile = tile,
            template = template.name,
            instances = template.records.len(),
            record_words = words,
            adjacent_changed_words = changed_words,
            narrow_delta_words = narrow_deltas,
            changed_non_allocation_words = changed_address_kinds[0],
            changed_resident_addresses = changed_address_kinds[1],
            changed_transient_addresses = changed_address_kinds[2],
            changed_mixed_addresses = changed_address_kinds[3],
            "largest static template record set"
        );
    }
    let emit_program =
        |program_index: usize, symbols: &BTreeMap<String, u32>, generated_base: u32| {
            let program = &programs[program_index];
            let host_plans = &tile_host_plans[program_index];
            let physical = topology.physical(program.tile)?;
            let weight_end = host.weights.len();
            let input_end = weight_end + host.inputs.len();
            let host_weights = host_phase_calls(host_plans, physical, 0, &host.weights);
            let host_inputs = host_phase_calls(host_plans, physical, weight_end, &host.inputs);
            let host_outputs = host_phase_calls(host_plans, physical, input_end, &host.outputs);
            static_codegen::emit(
                program,
                symbols,
                &tile_exchange_plans[program_index].addresses,
                &tile_exchange_plans[program_index].compute_runs,
                &tile_exchange_plans[program_index].templates,
                static_codegen::HostCode {
                    weights: &host_weights,
                    inputs: &host_inputs,
                    outputs: &host_outputs,
                },
                profile_code.get(program_index),
                generated_base,
                invocations,
            )
        };
    let preliminary_generated = preliminary_images
        .par_iter()
        .enumerate()
        .map(|(index, image)| emit_program(index, &image.symbols, 0))
        .collect::<Result<Vec<_>>>()?;
    if preliminary_generated
        .iter()
        .zip(&preliminary_program_sizes)
        .any(|(generated, &size)| generated.len() != size as usize)
    {
        return Err("generated program size changed after static-data relocation".into());
    }
    let program_bases = program_reservations
        .iter()
        .map(|&(base, _)| base)
        .collect::<Vec<_>>();
    let image_regions = programs
        .par_iter()
        .zip(&program_bases)
        .zip(&preliminary_generated)
        .zip(&template_record_ranges)
        .map(
            |(((program, &program_base), generated), template_records)| {
                let program_end = program_base
                    .checked_add(u32::try_from(generated.len())?)
                    .ok_or("generated program address overflow")?;
                let mut regions = executable_regions_for_tile(
                    &allocation_ranges_by_tile[usize::from(program.tile)],
                    tile_exchange_plans[usize::from(program.tile)].end,
                    &host_runtime_ranges[usize::from(program.tile)]
                        .iter()
                        .copied()
                        .chain(template_records.iter().copied())
                        .chain(std::iter::once((program_base, program_end)))
                        .collect::<Vec<_>>(),
                )?;
                // Generated and support code deliberately share executable memory
                // elements. Re-add the measured support interval after excluding the
                // generated bytes from generally available executable regions.
                regions.push(support_reservations[usize::from(program.tile)]);
                regions.sort_unstable();
                let mut merged = Vec::<(u32, u32)>::new();
                for (start, end) in regions {
                    if let Some(previous) = merged.last_mut()
                        && start <= previous.1
                    {
                        previous.1 = previous.1.max(end);
                    } else {
                        merged.push((start, end));
                    }
                }
                Ok(merged)
            },
        )
        .collect::<Result<Vec<_>>>()?;
    let mut image_cache = HashMap::<(Vec<(u32, u32)>, Vec<String>), ipu_elf::LinkedImage>::new();
    for (tile_index, (regions, symbols)) in
        image_regions.iter().zip(&tile_retained_symbols).enumerate()
    {
        let key = (regions.clone(), symbols.clone());
        if image_cache.contains_key(&key) {
            continue;
        }
        let image = link(
            objects,
            &LinkOptions {
                image_base: regions.first().ok_or("tile has no executable region")?.0,
                regions: regions.clone(),
                entry_symbol: "ipu_stack_static_start".into(),
                retained_symbols: symbols.clone(),
                externals: HashMap::new(),
            },
        )
        .map_err(|error| {
            format!(
                "failed to link support image for logical tile {} ({} bytes preliminary, \
                 reservation {:?}, executable regions {:?}): {error}",
                programs[tile_index].tile,
                preliminary_images[tile_index].bytes.len(),
                support_reservations[tile_index],
                regions,
            )
        })?;
        image_cache.insert(key, image);
    }
    let images = image_regions
        .par_iter()
        .zip(&tile_retained_symbols)
        .map(|(regions, symbols)| -> Result<ipu_elf::LinkedImage> {
            Ok(image_cache
                .get(&(regions.clone(), symbols.clone()))
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
                .map(|template| {
                    let record = template.records.first().map_or(0, Vec::len) * 4;
                    let record_words = template.records.first().map_or(0, Vec::len);
                    let split = usize::from(template.record_split);
                    let patches = template
                        .patches
                        .iter()
                        .skip(1)
                        .filter(|patch| !patch.is_empty())
                        .map(|patch| {
                            static_codegen::template_patch_ranges(record_words, split)
                                .into_iter()
                                .map(|slots| {
                                    static_codegen::template_patch_storage_words_range(slots, patch)
                                })
                                .sum::<usize>()
                        })
                        .sum::<usize>()
                        * 4;
                    let patch_table = template.patches.len().saturating_sub(1) * 8;
                    template.shared.len() * 4 + record + patches + patch_table
                })
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
        .par_iter()
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
    validate_kernel_memory_constraints(
        &resolved_memory_constraints,
        &executable_element_reservations,
    )?;
    programs
        .par_iter()
        .try_for_each(|program| -> Result<()> {
            let tile = usize::from(program.tile);
            let executable_elements = &executable_element_reservations[tile];
            for step in &program.steps {
                let ipu_compiler::LoweredTileStep::Compute(command) = step else {
                    continue;
                };
                for (tensor, address, operand) in std::iter::once((
                    command.output,
                    command.output_address,
                    "output",
                ))
                .chain(
                    command
                        .inputs
                        .iter()
                        .copied()
                        .zip(command.input_addresses.iter().copied())
                        .map(|(tensor, address)| (tensor, address, "input")),
                ) {
                    if let Some(&(start, end)) = executable_elements
                        .iter()
                        .find(|&&(start, end)| start <= address && address < end)
                    {
                        return Err(format!(
                            "kernel {} phase {} tile {} resolves {operand} tensor {} to 0x{address:x} in executable element 0x{start:x}..0x{end:x}",
                            command.specialization.operation,
                            command.phase,
                            command.tile,
                            tensor.0,
                        )
                        .into());
                    }
                }
                let resolve_operand = |operand: ipu_compiler::KernelOperand| -> Result<_> {
                    let (name, tensor, address) = match operand {
                        ipu_compiler::KernelOperand::Output => {
                            ("output", command.output, command.output_address)
                        }
                        ipu_compiler::KernelOperand::Input(index) => (
                            "input",
                            *command.inputs.get(index).ok_or_else(|| {
                                format!(
                                    "kernel {} memory ABI refers to missing input {index}",
                                    command.specialization.operation
                                )
                            })?,
                            *command.input_addresses.get(index).ok_or_else(|| {
                                format!(
                                    "lowered kernel {} has no address for input {index}",
                                    command.specialization.operation
                                )
                            })?,
                        ),
                    };
                    Ok((
                        name,
                        tensor,
                        address,
                        command.specialization.operand_access_bytes(operand)?,
                    ))
                };
                for constraint in command.specialization.memory_constraints() {
                    let ipu_compiler::KernelMemoryConstraint::DistinctEffectiveElements(operands) =
                        constraint
                    else {
                        let ipu_compiler::KernelMemoryConstraint::InClass(operand, class) = constraint
                        else {
                            unreachable!()
                        };
                        let (name, tensor, address, bytes) = resolve_operand(*operand)?;
                        match class {
                            ipu_compiler::KernelMemoryClass::Ipu21Interleaved
                                if address < ipu_package::IPU21_INTERLEAVED_MEMORY_BASE
                                    || address.checked_add(bytes).is_none_or(|end| {
                                        end > ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT
                                    }) =>
                            {
                                return Err(format!(
                                    "kernel {} phase {} tile {} resolves {name} tensor {} outside required interleaved memory at 0x{address:x} for {bytes} bytes",
                                    command.specialization.operation,
                                    command.phase,
                                    command.tile,
                                    tensor.0,
                                )
                                .into());
                            }
                            _ => {}
                        }
                        continue;
                    };
                    let mut elements = BTreeMap::<u8, (&str, u32, u32, u32)>::new();
                    for operand in *operands {
                        let (name, tensor, address, bytes) = resolve_operand(*operand)?;
                        let touched = ipu_package::ipu21_effective_memory_elements(address, bytes)
                            .ok_or_else(|| {
                                format!(
                                    "kernel {} phase {} tile {} resolves {name} tensor {} outside tile SRAM at 0x{address:x} for {bytes} bytes",
                                    command.specialization.operation,
                                    command.phase,
                                    command.tile,
                                    tensor.0,
                                )
                            })?;
                        for (element, start, end) in touched {
                            if let Some((other_name, other_tensor, other_address, _)) =
                                elements.insert(element, (name, tensor.0 as u32, address, end))
                            {
                                return Err(format!(
                                    "kernel {} phase {} tile {} maps {other_name} tensor {other_tensor} at 0x{other_address:x} and {name} tensor {} at 0x{address:x} to effective memory element {element} (0x{start:x}..0x{end:x})",
                                    command.specialization.operation,
                                    command.phase,
                                    command.tile,
                                    tensor.0,
                                )
                                .into());
                            }
                        }
                    }
                }
            }
            Ok(())
        })?;
    graph
        .schedule
        .allocations
        .par_iter()
        .try_for_each(|allocation| -> Result<()> {
            let end = allocation
                .address
                .checked_add(allocation.size)
                .ok_or("allocation address overflow")?;
            let tile = usize::from(allocation.tile);
            let runtime_end = tile_exchange_plans[tile].end;
            let program_base = program_bases[tile];
            let program_end = program_base
                .checked_add(u32::try_from(generated[tile].len())?)
                .ok_or("generated tile program address overflow")?;
            let image = &images[tile];
            if executable_element_reservations[tile]
                .iter()
                .any(|&(start, stop)| {
                    ranges_overlap(start, stop, allocation.address, end)
                })
                || image.segments.iter().any(|segment| {
                ranges_overlap(
                    segment.address,
                    segment.address + segment.size as u32,
                    allocation.address,
                    end,
                )
            }) || ranges_overlap(program_base, program_end, allocation.address, end)
                || ranges_overlap(PLAN_BASE, runtime_end, allocation.address, end)
                || host_runtime_ranges[tile]
                    .iter()
                    .any(|&(start, stop)| ranges_overlap(start, stop, allocation.address, end))
                || template_record_ranges[tile]
                    .iter()
                    .any(|&(start, stop)| ranges_overlap(start, stop, allocation.address, end))
            {
                return Err(format!(
                    "static runtime or executable memory element overlaps tensor {} ({:?}) on tile {} at 0x{:x}..0x{end:x}",
                    allocation.tensor.0, allocation.kind, allocation.tile, allocation.address
                )
                .into());
            }
            Ok(())
        })?;

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
            let address = image
                .symbols
                .get(name)
                .copied()
                .ok_or_else(|| format!("static runtime has no {name} symbol"))?;
            let segment = image
                .segments
                .iter()
                .find(|segment| {
                    address >= segment.address && address < segment.address + segment.size as u32
                })
                .ok_or_else(|| format!("{name} is outside linked image segments"))?;
            Ok(segment.offset + usize::try_from(address - segment.address)?)
        };
        let program_offset = symbol_offset("ipu_stack_static_program_address")?;
        let worker_context_offset = symbol_offset("ipu_stack_static_worker_sync_context_base")?;
        let worker_base_offset = symbol_offset("ipu_stack_static_worker_base")?;
        let prng_seed_base_offset = symbol_offset("ipu_stack_static_prng_seed_base")?;
        let host_staging_offset = symbol_offset("ipu_stack_static_host_staging_address")?;
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
        patch_setzi_immediate(&mut support_code, host_staging_offset, host.staging_address)?;
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
        segments.extend(image.segments.iter().map(|segment| Segment {
            address: segment.address,
            memory_size: segment.size as u32,
            blob: support_blob,
            blob_offset: segment.offset as u64,
            file_size: segment.size as u32,
            flags: SEGMENT_READ | SEGMENT_EXECUTE,
        }));
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
        let mut written_patch_addresses = HashSet::new();
        for template in &tile_exchange_plans[tile_index].templates {
            let mut resolve_word =
                |word: &static_codegen::StaticTemplateRecordWord| -> Result<u32> {
                    match word {
                        static_codegen::StaticTemplateRecordWord::Value(value) => Ok(*value),
                        static_codegen::StaticTemplateRecordWord::Symbol(name) => {
                            image.symbols.get(name).copied().ok_or_else(|| {
                                format!("static template references missing {name}").into()
                            })
                        }
                    }
                };
            static_codegen::validate_template_transitions(template, &mut resolve_word)?;
            static_codegen::validate_template_kernel_operands(template)?;
            let first_record = template.records.first().map(Vec::as_slice).unwrap_or(&[]);
            let split = usize::from(template.record_split);
            for (address, record) in [
                (template.shared_address, template.shared.as_slice()),
                (template.record_addresses[0], &first_record[..split]),
                (
                    template.record_secondary_addresses[0],
                    &first_record[split..],
                ),
            ] {
                if record.is_empty() {
                    continue;
                }
                let words = record
                    .iter()
                    .map(&mut resolve_word)
                    .collect::<Result<Vec<_>>>()?;
                let bytes = words_to_bytes(&words);
                if let Some((start, contents)) = template_segments.last_mut() {
                    if start.checked_add(u32::try_from(contents.len())?) == Some(address) {
                        contents.extend_from_slice(&bytes);
                        continue;
                    }
                }
                template_segments.push((address, bytes));
            }
            let patch_table = template
                .patch_addresses
                .iter()
                .skip(1)
                .filter_map(|addresses| addresses.first().copied())
                .collect::<Vec<_>>();
            if !patch_table.is_empty() {
                template_segments
                    .push((template.patch_table_address, words_to_bytes(&patch_table)));
            }
            for (instance, patch) in template.patches.iter().enumerate().skip(1) {
                for (part, slots) in
                    static_codegen::template_patch_ranges(first_record.len(), split)
                        .into_iter()
                        .enumerate()
                {
                    let address = template.patch_addresses[instance][part];
                    if !written_patch_addresses.insert(address) {
                        continue;
                    }
                    if static_codegen::template_patch_storage_words_range(slots.clone(), patch) == 0
                    {
                        template_segments.push((address, 0u32.to_le_bytes().to_vec()));
                        continue;
                    }
                    let words = static_codegen::serialize_template_patch_range(
                        slots.clone(),
                        patch,
                        &mut resolve_word,
                    )?;
                    template_segments.push((address, words_to_bytes(&words)));
                }
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
    app.weights = graph.host_weights.clone();
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
        minimum_image_base = format_args!(
            "0x{:x}",
            images
                .iter()
                .flat_map(|image| &image.segments)
                .map(|segment| segment.address)
                .min()
                .unwrap_or(0)
        ),
        maximum_image_base = format_args!(
            "0x{:x}",
            images
                .iter()
                .flat_map(|image| &image.segments)
                .map(|segment| segment.address)
                .max()
                .unwrap_or(0)
        ),
        minimum_program_base =
            format_args!("0x{:x}", program_bases.iter().min().copied().unwrap_or(0)),
        maximum_program_base =
            format_args!("0x{:x}", program_bases.iter().max().copied().unwrap_or(0)),
        "packaged static executable graph"
    );
    Ok(app)
}

fn validate_resident_host_bindings(graph: &ExecutableGraph, topology: &Topology) -> Result<()> {
    let mut resident_ends_by_tile = HashMap::<u16, Vec<(u32, u32)>>::new();
    for allocation in &graph.schedule.allocations {
        if allocation.kind != ipu_compiler::AllocationKind::Home
            || allocation.live_from != 0
            || allocation.live_until != usize::MAX
        {
            continue;
        }
        let physical = topology.physical(allocation.tile)?;
        let end = allocation
            .address
            .checked_add(allocation.size)
            .ok_or("resident allocation address overflow")?;
        resident_ends_by_tile
            .entry(physical)
            .or_default()
            .push((allocation.address, end));
    }
    for intervals in resident_ends_by_tile.values_mut() {
        intervals.sort_unstable_by_key(|&(start, _)| start);
        let mut maximum_end = 0;
        for (_, end) in intervals {
            maximum_end = maximum_end.max(*end);
            *end = maximum_end;
        }
    }

    for binding in &graph.host_weights {
        for slice in &binding.slices {
            let end = slice
                .tile_address
                .checked_add(u32::try_from(slice.size)?)
                .ok_or("resident host binding address overflow")?;
            let resident = resident_ends_by_tile
                .get(&(slice.tile as u16))
                .and_then(|intervals| {
                    let upper =
                        intervals.partition_point(|&(start, _)| start <= slice.tile_address);
                    upper.checked_sub(1).map(|index| intervals[index].1)
                })
                .is_some_and(|resident_end| resident_end >= end);
            if !resident {
                return Err(format!(
                    "resident host tensor {} slice on physical tile {} at 0x{:x}..0x{end:x} has no permanent allocation",
                    binding.name, slice.tile, slice.tile_address
                )
                .into());
            }
        }
    }
    Ok(())
}

fn build_static_host_layout(graph: &ExecutableGraph, invocations: u32) -> Result<StaticHostLayout> {
    if graph.host_weights.is_empty()
        && graph.host_inputs.is_empty()
        && graph.host_outputs.is_empty()
    {
        return Ok(StaticHostLayout {
            weights: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            staging_address: 0,
            protocol: HostExchange::default(),
        });
    }

    let mut weight_file_cursor = 0u64;
    let mut input_file_cursor = 0u64;
    let mut output_file_cursor = 0u64;
    let mut weights = Vec::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut calls = Vec::new();
    append_host_bindings(
        &graph.host_weights,
        HostDirection::ToTile,
        &mut weight_file_cursor,
        &mut weights,
        &mut calls,
    )?;
    append_host_bindings(
        &graph.host_inputs,
        HostDirection::ToTile,
        &mut input_file_cursor,
        &mut inputs,
        &mut calls,
    )?;
    append_host_bindings(
        &graph.host_outputs,
        HostDirection::ToHost,
        &mut output_file_cursor,
        &mut outputs,
        &mut calls,
    )?;
    let host_to_tile_limit =
        ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::HOST_TO_TILE_WINDOW_BYTES;
    let staging_bytes = weights
        .iter()
        .chain(&inputs)
        .filter(|transfer| {
            transfer
                .tile_address
                .checked_add(transfer.bytes)
                .is_none_or(|end| end > host_to_tile_limit)
        })
        .map(|transfer| transfer.bytes)
        .max()
        .unwrap_or(0);
    let staging_tiles = weights
        .iter()
        .chain(&inputs)
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
    for transfer in weights.iter_mut().chain(&mut inputs) {
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
    let weight_transfer_count = weights.len();
    let input_transfer_count = inputs.len();
    let weight_inputs = calls[..weight_transfer_count]
        .iter()
        .flat_map(|call| call.inputs.iter().cloned())
        .collect();
    let call_inputs = calls[weight_transfer_count..weight_transfer_count + input_transfer_count]
        .iter()
        .flat_map(|call| call.inputs.iter().cloned())
        .collect();
    let call_outputs = calls[weight_transfer_count + input_transfer_count..]
        .iter()
        .flat_map(|call| call.outputs.iter().cloned())
        .collect();
    let graph_transfers = u32::try_from(inputs.len() + outputs.len())?;
    info!(
        weight_transfers = weights.len(),
        input_transfers = inputs.len(),
        output_transfers = outputs.len(),
        staging_bytes,
        "planned static host-transfer layout"
    );
    let graph_phases = if invocations == 1 {
        host_transfer_phase_count(graph_transfers)?
    } else {
        graph_transfers
            .checked_mul(2)
            .ok_or("host graph phase count overflow")?
    };
    calls = Vec::new();
    if !weights.is_empty() {
        calls.push(HostCall {
            name: "initialize".into(),
            command: 0,
            invocations: 1,
            phases: u32::try_from(weights.len())?
                .checked_mul(2)
                .ok_or("host initialization phase count overflow")?,
            inputs: weight_inputs,
            outputs: Vec::new(),
        });
    }
    calls.push(HostCall {
        name: "graph".into(),
        command: 0,
        invocations,
        phases: graph_phases,
        inputs: call_inputs,
        outputs: call_outputs,
    });

    let command_page = 0;
    let data_page = 1;
    Ok(StaticHostLayout {
        weights,
        inputs,
        outputs,
        staging_address: staging_range.map_or(0, |range| range.0),
        protocol: HostExchange {
            startup_mark: ipu_driver::HOST_EXCHANGE_HANDOFF_MARK,
            command_page,
            command_offset: 0,
            pages: [command_page, data_page]
                .into_iter()
                .map(|index| HostPage {
                    index,
                    size: u64::from(ipu_exchange::HOST_PAGE_BYTES),
                })
                .collect(),
            attach_order: vec![command_page, data_page],
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
    file_cursor: &mut u64,
    transfers: &mut Vec<StaticHostTransfer>,
    calls: &mut Vec<HostCall>,
) -> Result<()> {
    for binding in bindings {
        let binding_file_base = *file_cursor;
        for slice in &binding.slices {
            let mut tile_address = slice.tile_address;
            let mut file_offset = binding_file_base + slice.file_offset;
            let mut remaining = u32::try_from(slice.size)?;
            while remaining != 0 {
                let host_offset = HOST_DATA_START;
                let bytes = remaining.min(ipu_exchange::HOST_PAGE_BYTES);
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
                    invocations: 1,
                    inputs: matches!(direction, HostDirection::ToTile)
                        .then_some(host_slices.clone())
                        .unwrap_or_default(),
                    outputs: matches!(direction, HostDirection::ToHost)
                        .then_some(host_slices)
                        .unwrap_or_default(),
                });
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

fn host_phase_calls(
    plans: &TileHostPlans,
    physical_tile: u16,
    offset: usize,
    transfers: &[StaticHostTransfer],
) -> Vec<static_codegen::HostPhaseCall> {
    plans.addresses[offset..offset + transfers.len()]
        .iter()
        .copied()
        .zip(&plans.run_tables[offset..offset + transfers.len()])
        .zip(transfers)
        .map(
            |((address, &run_table), transfer)| static_codegen::HostPhaseCall {
                address,
                active: host_phase_is_active(physical_tile, transfer),
                run_table,
            },
        )
        .collect()
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

fn host_run_descriptor_words(
    physical_tile: u16,
    transfers: &[StaticHostTransfer],
    packet_copies: &[Option<HostPacketCopy>],
) -> Result<Vec<u32>> {
    if transfers.len() != packet_copies.len() {
        return Err("host run transfer and packet counts differ".into());
    }
    let mut descriptors = Vec::with_capacity(transfers.len() * HOST_RUN_DESCRIPTOR_WORDS as usize);
    for (&transfer, &packet) in transfers.iter().zip(packet_copies) {
        if !host_phase_is_active(physical_tile, &transfer) {
            return Err("inactive transfer included in a static host run".into());
        }
        let copy = (transfer.physical_tile == physical_tile)
            .then_some(transfer.copy_destination)
            .flatten();
        let packet = packet.ok_or("active host run has no packet copy")?;
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
            copy.unwrap_or(0),
            copy_words | packet_destination | (packet.words << 24),
            packet.source,
        ]);
    }
    Ok(descriptors)
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
    let mut written_tables = HashMap::<u32, Vec<u32>>::new();
    for (start, &table_address) in run_tables.iter().enumerate() {
        let Some(table_address) = table_address else {
            continue;
        };
        let mut index = start;
        while index < transfers.len()
            && (index == start || run_tables[index].is_none())
            && host_phase_is_active(physical_tile, &transfers[index])
        {
            index += 1;
        }
        let descriptors = host_run_descriptor_words(
            physical_tile,
            &transfers[start..index],
            &packet_copies[start..index],
        )?;
        if let Some(previous) = written_tables.insert(table_address, descriptors.clone()) {
            if previous != descriptors {
                return Err("aliased static host run tables have different contents".into());
            }
            continue;
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
    let graph_call = calls
        .iter()
        .find(|call| call.name == "graph")
        .ok_or("application has no generated graph call")?;
    let output_size = graph_call
        .outputs
        .iter()
        .map(|slice| slice.file_offset + slice.size)
        .max()
        .unwrap_or(0);
    let invocation_input_bytes = bindings_size(&app.inputs)?;
    let resident_input_bytes = bindings_size(&app.weights)?;
    let expected_input_bytes = invocation_input_bytes
        .checked_mul(u64::from(graph_call.invocations))
        .ok_or("batched host input size overflow")?
        .checked_add(resident_input_bytes)
        .ok_or("host input size overflow")?;
    if input.len() != usize::try_from(expected_input_bytes)? {
        return Err(format!(
            "application expects {expected_input_bytes} input bytes ({} invocations of {} bytes, {} resident), got {}",
            graph_call.invocations,
            invocation_input_bytes,
            resident_input_bytes,
            input.len()
        )
        .into());
    }
    let invocation_region_bytes = invocation_input_bytes * u64::from(graph_call.invocations);
    let (invocation_inputs, resident_input) =
        input.split_at(usize::try_from(invocation_region_bytes)?);
    if let Some(call) = calls.iter().find(|call| call.name == "initialize") {
        let completed = session
            .invoke_streaming_deferred(&call.name, call_input(call, resident_input)?)
            .map_err(|error| generated_call_error(&device, app, call, error))?;
        session.collect(&completed)?;
    }
    let mut output = Vec::with_capacity(
        usize::try_from(output_size)? * usize::try_from(graph_call.invocations)?,
    );
    for invocation in 0..graph_call.invocations {
        let start = usize::try_from(u64::from(invocation) * invocation_input_bytes)?;
        let end = start + usize::try_from(invocation_input_bytes)?;
        let deferred = session
            .invoke_streaming_deferred(
                &graph_call.name,
                call_input(graph_call, &invocation_inputs[start..end])?,
            )
            .map_err(|error| generated_call_error(&device, app, graph_call, error))?;
        if invocation + 1 == graph_call.invocations {
            finish_host_graph(&device, app)?;
            verify_runtime_completion(&device, app)?;
        }
        let call_output = session.collect(&deferred)?;
        output.extend_from_slice(&call_output[..usize::try_from(output_size)?]);
    }
    debug!(states = %supervisor_state_summary(&device, app), "host exchange supervisor states");
    debug!(sources = %host_source_summary(&device, app), "host exchange device sources");
    drop(session);
    if let Some(inspector) = inspector {
        inspector(&device, &output)?;
    }
    Ok(output)
}

fn generated_call_error(
    device: &Device,
    app: &Application,
    call: &HostCall,
    error: DriverError,
) -> Box<dyn std::error::Error + Send + Sync> {
    format!(
        "generated host call {}: {error}; supervisor states: {}; device outputs: {}",
        call.name,
        supervisor_state_summary(device, app),
        host_source_summary(device, app)
    )
    .into()
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

fn bindings_size(bindings: &[Binding]) -> Result<u64> {
    bindings.iter().try_fold(0u64, |total, binding| {
        total
            .checked_add(binding_size(binding)?)
            .ok_or_else(|| "binding set size overflow".into())
    })
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
        .take(16)
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
    let supervisor_exceptions = app
        .tiles
        .iter()
        .filter(|tile| device.tile_context_state(tile.physical_tile as u16, 0).ok() == Some(3))
        .take(8)
        .map(|tile| {
            let status = device.read_tile_context_status(tile.physical_tile as u16, 0);
            let pc = device
                .read_tile_program_counter(tile.physical_tile as u16, 0)
                .map(|pc| format!("0x{pc:x}"))
                .unwrap_or_else(|error| format!("error({error})"));
            match status {
                Ok(status) => {
                    let exception = ipu_driver::TileException::from_status(status);
                    let registers = (exception == ipu_driver::TileException::InvalidMemoryAddress)
                        .then(|| {
                            [2, 3, 4, 5, 10, 11, 15]
                                .into_iter()
                                .map(|register| {
                                    device
                                        .read_tile_m_register(
                                            tile.physical_tile as u16,
                                            0,
                                            register,
                                        )
                                        .map(|value| format!("m{register}=0x{value:x}"))
                                        .unwrap_or_else(|error| {
                                            format!("m{register}=error({error})")
                                        })
                                })
                                .collect::<Vec<_>>()
                                .join(",")
                        });
                    if let Some(registers) = registers {
                        format!("{}:{exception}@{pc}[{registers}]", tile.physical_tile)
                    } else {
                        format!("{}:{exception}@{pc}", tile.physical_tile)
                    }
                }
                Err(error) => format!("{}:status=error({error}),pc={pc}", tile.physical_tile),
            }
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
        .take(16)
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
                                (Ok(status), Ok(pc)) => {
                                    let exception = ipu_driver::TileException::from_status(status);
                                    let registers = (context == 1
                                        && matches!(
                                            exception,
                                            ipu_driver::TileException::InvalidMemoryAddress
                                                | ipu_driver::TileException::MemoryConflict
                                        ))
                                    .then(|| {
                                        [0, 1, 2, 3, 9, 12]
                                            .into_iter()
                                            .map(|register| {
                                                device
                                                    .read_tile_m_register(
                                                        tile.physical_tile as u16,
                                                        context,
                                                        register,
                                                    )
                                                    .map(|value| format!("m{register}=0x{value:x}"))
                                                    .unwrap_or_else(|error| {
                                                        format!("m{register}=error({error})")
                                                    })
                                            })
                                            .collect::<Vec<_>>()
                                            .join(",")
                                    });
                                    match registers {
                                        Some(registers) => {
                                            format!("c{context}:{exception}@0x{pc:x}[{registers}]")
                                        }
                                        None => format!("c{context}:{exception}@0x{pc:x}"),
                                    }
                                }
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
        "0={} {:?}, 1={} {:?}, 2={} {:?}, 3={} {:?}, errors={read_errors}, supervisor_exceptions={supervisor_exceptions:?}, active_contexts={active_contexts:?}, milestones={milestones:?}",
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
    use std::sync::Arc;

    #[test]
    fn allocation_footprints_merge_owned_storage_and_ignore_aliases() {
        let graph = ExecutableGraph {
            memory_policy: None,
            schedule: Schedule {
                layouts: Vec::new(),
                phases: Vec::new(),
                allocations: vec![
                    ipu_compiler::Allocation {
                        tensor: ipu_compiler::TensorId(1),
                        tile: 0,
                        address: 0x60000,
                        size: 64,
                        live_from: 0,
                        live_until: 1,
                        kind: ipu_compiler::AllocationKind::Home,
                    },
                    ipu_compiler::Allocation {
                        tensor: ipu_compiler::TensorId(2),
                        tile: 0,
                        address: 0x60040,
                        size: 32,
                        live_from: 1,
                        live_until: 2,
                        kind: ipu_compiler::AllocationKind::Home,
                    },
                    ipu_compiler::Allocation {
                        tensor: ipu_compiler::TensorId(3),
                        tile: 0,
                        address: 0x60010,
                        size: 16,
                        live_from: 0,
                        live_until: 1,
                        kind: ipu_compiler::AllocationKind::HomeAlias {
                            source: ipu_compiler::TensorId(1),
                        },
                    },
                    ipu_compiler::Allocation {
                        tensor: ipu_compiler::TensorId(4),
                        tile: 0,
                        address: ipu_exchange::EXCHANGE_WINDOW_BASE,
                        size: 16,
                        live_from: 0,
                        live_until: 1,
                        kind: ipu_compiler::AllocationKind::ExchangeStaging { phase: 0 },
                    },
                ]
                .into(),
                tile_count: 1,
                peak_sram: BTreeMap::new(),
            },
            initial_buffers: Vec::new(),
            outputs: Vec::new(),
            host_weights: Vec::new(),
            host_inputs: Vec::new(),
            host_outputs: Vec::new(),
        };

        assert_eq!(
            allocation_footprints_by_tile(&graph, 1, None).unwrap(),
            vec![vec![
                (
                    ipu_exchange::EXCHANGE_WINDOW_BASE,
                    ipu_exchange::EXCHANGE_WINDOW_BASE + 16,
                ),
                (0x60000, 0x60060),
            ]]
        );
    }

    #[test]
    fn measured_executable_relocation_updates_aliases_and_literal_addresses() {
        let topology = Topology::c600();
        let old_address = 0x60000;
        let mut reservations = vec![[(0, 0); 2]; topology.tile_count()];
        reservations[0] = [(0x5c000, 0x64000), (0x64000, 0x68000)];
        let mut graph = ExecutableGraph {
            memory_policy: None,
            schedule: Schedule {
                layouts: Vec::new(),
                phases: Vec::new(),
                allocations: vec![
                    ipu_compiler::Allocation {
                        tensor: ipu_compiler::TensorId(1),
                        tile: 0,
                        address: old_address,
                        size: 64,
                        live_from: 1,
                        live_until: 2,
                        kind: ipu_compiler::AllocationKind::Home,
                    },
                    ipu_compiler::Allocation {
                        tensor: ipu_compiler::TensorId(2),
                        tile: 0,
                        address: old_address + 16,
                        size: 16,
                        live_from: 1,
                        live_until: 2,
                        kind: ipu_compiler::AllocationKind::HomeAlias {
                            source: ipu_compiler::TensorId(1),
                        },
                    },
                ]
                .into(),
                tile_count: u16::try_from(topology.tile_count()).unwrap(),
                peak_sram: BTreeMap::new(),
            },
            initial_buffers: vec![InitialBuffer {
                tile: 0,
                address: old_address + 4,
                words: vec![1],
            }],
            outputs: Vec::new(),
            host_weights: Vec::new(),
            host_inputs: vec![Binding {
                name: "input".into(),
                dtype: "u32".into(),
                shape: vec![1],
                slices: vec![RegionSlice {
                    tile: u32::from(topology.physical(0).unwrap()),
                    tile_address: old_address + 8,
                    file_offset: 0,
                    size: 4,
                }],
            }],
            host_outputs: Vec::new(),
        };

        assert_eq!(
            relocate_transient_allocations_for_executables(&mut graph, &topology, &reservations)
                .unwrap(),
            1
        );
        let relocated = graph.schedule.allocations[0].address;
        assert!(
            (PLAN_BASE..ipu_package::IPU21_INTERLEAVED_MEMORY_BASE).contains(&relocated)
                || (ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT
                    ..ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE)
                    .contains(&relocated)
        );
        assert!(reservations[0].iter().all(|&(start, end)| !ranges_overlap(
            align_down(relocated, ipu_package::TILE_MEMORY_ELEMENT_SIZE),
            align_up(
                relocated + graph.schedule.allocations[0].size,
                ipu_package::TILE_MEMORY_ELEMENT_SIZE,
            ),
            start,
            end,
        )));
        assert_eq!(graph.schedule.allocations[1].address, relocated + 16);
        assert_eq!(graph.initial_buffers[0].address, relocated + 4);
        assert_eq!(graph.host_inputs[0].slices[0].tile_address, relocated + 8);
    }

    #[test]
    fn measured_executable_relocation_moves_owned_exchange_staging() {
        let topology = Topology::c600();
        let old_address = 0x60000;
        let mut reservations = vec![[(0, 0); 2]; topology.tile_count()];
        reservations[0] = [(0x5c000, 0x64000), (0x64000, 0x68000)];
        let tensor = ipu_compiler::TensorId(1);
        let mut graph = ExecutableGraph {
            memory_policy: None,
            schedule: Schedule {
                layouts: Vec::new(),
                phases: vec![ipu_compiler::Phase::Exchange {
                    transfers: vec![ipu_compiler::Transfer {
                        source_tile: 1,
                        destination_tile: 0,
                        tensor,
                        bytes: 64,
                        staging_address: Some(old_address),
                    }],
                }],
                allocations: vec![
                    ipu_compiler::Allocation {
                        tensor,
                        tile: 1,
                        address: 0x90000,
                        size: 64,
                        live_from: 0,
                        live_until: usize::MAX,
                        kind: ipu_compiler::AllocationKind::Home,
                    },
                    ipu_compiler::Allocation {
                        tensor,
                        tile: 0,
                        address: old_address,
                        size: 64,
                        live_from: 0,
                        live_until: 2,
                        kind: ipu_compiler::AllocationKind::ExchangeStaging { phase: 0 },
                    },
                ]
                .into(),
                tile_count: u16::try_from(topology.tile_count()).unwrap(),
                peak_sram: BTreeMap::new(),
            },
            initial_buffers: Vec::new(),
            outputs: Vec::new(),
            host_weights: Vec::new(),
            host_inputs: Vec::new(),
            host_outputs: Vec::new(),
        };

        assert_eq!(
            relocate_transient_allocations_for_executables(&mut graph, &topology, &reservations)
                .unwrap(),
            1
        );
        let relocated = graph.schedule.allocations[1].address;
        assert_ne!(relocated, old_address);
        assert!(reservations[0].iter().all(|&(start, end)| !ranges_overlap(
            relocated,
            relocated + 64,
            start,
            end,
        )));
        let ipu_compiler::Phase::Exchange { transfers } = &graph.schedule.phases[0] else {
            unreachable!()
        };
        assert_eq!(transfers[0].staging_address, Some(relocated));
    }

    #[test]
    fn literal_relocation_resolves_reused_addresses_by_lifetime() {
        let first = AllocationRelocation {
            tensor: ipu_compiler::TensorId(1),
            tile: 0,
            old: 0x60000..0x61000,
            new_start: 0x70000,
            live_from: 0,
            live_until: 5,
            home: true,
        };
        let second = AllocationRelocation {
            tensor: ipu_compiler::TensorId(2),
            tile: 0,
            old: first.old.clone(),
            new_start: 0x80000,
            live_from: 5,
            live_until: usize::MAX,
            home: true,
        };
        let relocations = [&first, &second];
        let mut early = 0x60040;
        let mut late = early;

        relocate_boundary_address(&mut early, &relocations, GraphBoundary::Input).unwrap();
        relocate_boundary_address(&mut late, &relocations, GraphBoundary::Output).unwrap();

        assert_eq!(early, 0x70040);
        assert_eq!(late, 0x80040);
    }

    #[test]
    fn relocation_preserves_required_interleaving_and_allows_ordinary_spill() {
        let interleaved = ipu_compiler::Allocation {
            tensor: ipu_compiler::TensorId(1),
            tile: 0,
            address: ipu_package::IPU21_INTERLEAVED_MEMORY_BASE,
            size: 1024,
            live_from: 0,
            live_until: 1,
            kind: ipu_compiler::AllocationKind::Home,
        };
        let ordinary = ipu_compiler::Allocation {
            address: ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT,
            tensor: ipu_compiler::TensorId(2),
            ..interleaved.clone()
        };
        let policy = ipu_compiler::MemoryPolicy::ipu21(
            PLAN_BASE,
            ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
            &[
                ipu_compiler::Ipu21MemoryRegion::OrdinaryHigh,
                ipu_compiler::Ipu21MemoryRegion::Interleaved,
                ipu_compiler::Ipu21MemoryRegion::OrdinaryLow,
            ],
            &[
                ipu_compiler::Ipu21MemoryRegion::OrdinaryLow,
                ipu_compiler::Ipu21MemoryRegion::Interleaved,
                ipu_compiler::Ipu21MemoryRegion::OrdinaryHigh,
            ],
        )
        .unwrap();

        let interleaved_arenas =
            relocation_arenas_for_allocation(&interleaved, &policy.transient, true).unwrap();
        let ordinary_arenas =
            relocation_arenas_for_allocation(&ordinary, &policy.transient, false).unwrap();

        assert!(interleaved_arenas.iter().all(|arena| {
            arena.base >= ipu_package::IPU21_INTERLEAVED_MEMORY_BASE
                && arena.limit <= ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT
        }));
        assert!(ordinary_arenas.iter().any(|arena| {
            arena.base <= ipu_package::IPU21_INTERLEAVED_MEMORY_BASE
                && arena.limit >= ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT
        }));

        let ordinary_reservation = (PLAN_BASE, PLAN_BASE + 0x100);
        let ordinary_movable = (PLAN_BASE + 0x200, PLAN_BASE + 0x300);
        let interleaved_footprint = (
            ipu_package::IPU21_INTERLEAVED_MEMORY_BASE,
            ipu_package::IPU21_INTERLEAVED_MEMORY_BASE + 0x400,
        );
        let static_ranges = static_placement_ranges(
            &[ordinary_reservation],
            &[ordinary_movable, interleaved_footprint],
        );
        assert!(static_ranges.contains(&ordinary_reservation));
        assert!(static_ranges.contains(&interleaved_footprint));
        assert!(!static_ranges.contains(&ordinary_movable));
    }

    #[test]
    fn kernel_access_spans_do_not_expand_storage_ownership() {
        let topology = Topology::c600();
        let first = ipu_compiler::TensorId(1);
        let second = ipu_compiler::TensorId(2);
        let logical_bytes = 32;
        let access_bytes = 48;
        let graph = ExecutableGraph {
            memory_policy: Some(ipu_compiler::MemoryPolicy::contiguous(0x80000, 0x90000)),
            schedule: Schedule {
                layouts: Vec::new(),
                phases: vec![ipu_compiler::Phase::Compute {
                    op: ipu_compiler::OpId(0),
                    commands: Vec::new(),
                }],
                allocations: vec![
                    ipu_compiler::Allocation {
                        tensor: first,
                        tile: 0,
                        address: 0x80000,
                        size: logical_bytes,
                        live_from: 0,
                        live_until: 1,
                        kind: ipu_compiler::AllocationKind::Home,
                    },
                    ipu_compiler::Allocation {
                        tensor: second,
                        tile: 0,
                        address: 0x80020,
                        size: logical_bytes,
                        live_from: 0,
                        live_until: 1,
                        kind: ipu_compiler::AllocationKind::Home,
                    },
                ]
                .into(),
                tile_count: u16::try_from(topology.tile_count()).unwrap(),
                peak_sram: BTreeMap::new(),
            },
            initial_buffers: Vec::new(),
            outputs: Vec::new(),
            host_weights: Vec::new(),
            host_inputs: Vec::new(),
            host_outputs: Vec::new(),
        };
        let resolved = ipu_compiler::ResolvedKernelMemoryConstraints {
            classes: vec![ipu_compiler::ResolvedMemoryClassConstraint {
                operand: ipu_compiler::ResolvedMemoryOperand {
                    allocation: Some(0),
                    tile: 0,
                    address: 0x80000,
                    bytes: access_bytes,
                },
                class: ipu_compiler::KernelMemoryClass::Ipu21Interleaved,
            }],
            separations: Vec::new(),
        };
        let constraints = relocation_memory_constraints(&graph, &resolved).unwrap();
        assert!(constraints.required_interleaved.contains(&0));
        assert_eq!(constraints.access_extent(0, logical_bytes), access_bytes);
        assert_eq!(graph.schedule.allocations[0].size, logical_bytes);
        assert_eq!(graph.schedule.allocations[1].address, 0x80020);
        graph.schedule.validate_allocations().unwrap();
    }

    #[test]
    fn compaction_preserves_kernel_memory_constraints_and_static_access_spans() {
        let topology = Topology::c600();
        let output = ipu_compiler::TensorId(1);
        let input = ipu_compiler::TensorId(2);
        let unrelated = ipu_compiler::TensorId(3);
        let command = Arc::new(ipu_compiler::KernelCommand {
            tile: 0,
            output,
            inputs: vec![input],
            arguments: Vec::new(),
            specialization: Arc::new(ipu_compiler::SpecializationKey {
                operation: "gemm_f16_accumulate_small_rows".into(),
                shape: vec![8, 16, 16],
                worker_count: 6,
                role: "test".into(),
                alignment: 8,
                abi: ipu_compiler::KernelAbi::pace_with_left_access_tail(
                    8 * 16 * 2,
                    8 * 16 * 2,
                    1,
                    ipu_compiler::PACE_F16_LEFT_ACCESS_TAIL_BYTES,
                    false,
                ),
            }),
            metadata: BTreeMap::new(),
        });
        let mut graph = ExecutableGraph {
            memory_policy: Some(ipu_compiler::MemoryPolicy::contiguous(0x80000, 0x90000)),
            schedule: Schedule {
                layouts: Vec::new(),
                phases: vec![ipu_compiler::Phase::Compute {
                    op: ipu_compiler::OpId(0),
                    commands: vec![command.clone()],
                }],
                allocations: vec![
                    ipu_compiler::Allocation {
                        tensor: output,
                        tile: 0,
                        address: 0x80000,
                        size: 0x1000,
                        live_from: 0,
                        live_until: 2,
                        kind: ipu_compiler::AllocationKind::Home,
                    },
                    ipu_compiler::Allocation {
                        tensor: input,
                        tile: 0,
                        address: 0x88000,
                        size: 8 * 16 * 2,
                        live_from: 0,
                        live_until: 2,
                        kind: ipu_compiler::AllocationKind::Home,
                    },
                    ipu_compiler::Allocation {
                        tensor: unrelated,
                        tile: 0,
                        address: 0x8c000,
                        size: 0x6000,
                        live_from: 0,
                        live_until: 2,
                        kind: ipu_compiler::AllocationKind::Home,
                    },
                ]
                .into(),
                tile_count: u16::try_from(topology.tile_count()).unwrap(),
                peak_sram: BTreeMap::new(),
            },
            initial_buffers: Vec::new(),
            outputs: Vec::new(),
            host_weights: Vec::new(),
            host_inputs: Vec::new(),
            host_outputs: Vec::new(),
        };
        let programs = vec![ipu_compiler::LoweredTileProgram {
            tile: 0,
            steps: vec![ipu_compiler::LoweredTileStep::Compute(
                ipu_compiler::LoweredComputeCommand {
                    op: ipu_compiler::OpId(0),
                    phase: 0,
                    phase_tile_command_index: 0,
                    command,
                    output_address: 0x80000,
                    input_addresses: smallvec::smallvec![0x88000],
                },
            )],
        }];
        let memory_constraints = graph
            .schedule
            .resolve_memory_constraints(&programs)
            .unwrap();
        let static_reservation = (0x88100, 0x88200);
        let mut reservations = vec![Vec::new(); topology.tile_count()];
        reservations[0].push(static_reservation);

        compact_transient_allocations_around(
            &mut graph,
            &topology,
            &reservations,
            Some(&memory_constraints),
            "unit test",
        )
        .unwrap();

        assert!(
            !memory_spans_share_effective_element(
                graph.schedule.allocations[0].address,
                8 * 16 * 2,
                graph.schedule.allocations[1].address,
                8 * 16 * 2 + ipu_compiler::PACE_F16_LEFT_ACCESS_TAIL_BYTES,
            )
            .unwrap()
        );
        let input_address = graph.schedule.allocations[1].address;
        assert!(!ranges_overlap(
            input_address,
            input_address + 8 * 16 * 2 + ipu_compiler::PACE_F16_LEFT_ACCESS_TAIL_BYTES,
            static_reservation.0,
            static_reservation.1,
        ));
        graph.schedule.validate_allocations().unwrap();
    }

    #[test]
    fn compact_repacking_reuses_addresses_across_disjoint_lifetimes() {
        let topology = Topology::c600();
        let arena = 0x88000..0xe8000;
        let mut graph = ExecutableGraph {
            memory_policy: Some(ipu_compiler::MemoryPolicy::contiguous(
                arena.start,
                arena.end,
            )),
            schedule: Schedule {
                layouts: Vec::new(),
                phases: Vec::new(),
                allocations: [(0usize, 1usize, 0x90000), (1, 2, 0xd0000)]
                    .into_iter()
                    .enumerate()
                    .map(
                        |(tensor, (live_from, live_until, address))| ipu_compiler::Allocation {
                            tensor: ipu_compiler::TensorId(tensor),
                            tile: 0,
                            address,
                            size: 0x1000,
                            live_from,
                            live_until,
                            kind: ipu_compiler::AllocationKind::Home,
                        },
                    )
                    .collect(),
                tile_count: u16::try_from(topology.tile_count()).unwrap(),
                peak_sram: BTreeMap::new(),
            },
            initial_buffers: Vec::new(),
            outputs: Vec::new(),
            host_weights: Vec::new(),
            host_inputs: Vec::new(),
            host_outputs: Vec::new(),
        };
        let reservations = vec![Vec::new(); topology.tile_count()];

        compact_transient_allocations_around(
            &mut graph,
            &topology,
            &reservations,
            None,
            "unit test",
        )
        .unwrap();

        assert_eq!(
            graph.schedule.allocations[0].address, graph.schedule.allocations[1].address,
            "equal-shaped values with disjoint lifetimes should share storage"
        );
    }

    #[test]
    fn compaction_propagates_relocation_through_nested_aliases() {
        let topology = Topology::c600();
        let owner = ipu_compiler::TensorId(0);
        let first_alias = ipu_compiler::TensorId(1);
        let second_alias = ipu_compiler::TensorId(2);
        let old_owner = 0x90000;
        let mut graph = ExecutableGraph {
            memory_policy: Some(ipu_compiler::MemoryPolicy::contiguous(0x88000, 0xe8000)),
            schedule: Schedule {
                layouts: Vec::new(),
                phases: Vec::new(),
                allocations: vec![
                    ipu_compiler::Allocation {
                        tensor: owner,
                        tile: 0,
                        address: old_owner,
                        size: 0x1000,
                        live_from: 0,
                        live_until: 2,
                        kind: ipu_compiler::AllocationKind::Home,
                    },
                    ipu_compiler::Allocation {
                        tensor: first_alias,
                        tile: 0,
                        address: old_owner + 0x100,
                        size: 0x400,
                        live_from: 0,
                        live_until: 2,
                        kind: ipu_compiler::AllocationKind::HomeAlias { source: owner },
                    },
                    ipu_compiler::Allocation {
                        tensor: second_alias,
                        tile: 0,
                        address: old_owner + 0x200,
                        size: 0x100,
                        live_from: 0,
                        live_until: 1,
                        kind: ipu_compiler::AllocationKind::HomeAlias {
                            source: first_alias,
                        },
                    },
                ]
                .into(),
                tile_count: u16::try_from(topology.tile_count()).unwrap(),
                peak_sram: BTreeMap::new(),
            },
            initial_buffers: Vec::new(),
            outputs: Vec::new(),
            host_weights: Vec::new(),
            host_inputs: Vec::new(),
            host_outputs: Vec::new(),
        };
        let mut reservations = vec![Vec::new(); topology.tile_count()];
        reservations[0].push((old_owner, old_owner + 0x1000));

        compact_transient_allocations_around(
            &mut graph,
            &topology,
            &reservations,
            None,
            "unit test",
        )
        .unwrap();

        let new_owner = graph.schedule.allocations[0].address;
        assert_ne!(new_owner, old_owner);
        assert_eq!(graph.schedule.allocations[1].address, new_owner + 0x100);
        assert_eq!(graph.schedule.allocations[2].address, new_owner + 0x200);
        graph.schedule.validate_allocations().unwrap();
    }

    #[test]
    fn global_repacking_places_residents_before_lifetime_overlapped_transients() {
        let topology = Topology::c600();
        let arena = 0x88000..0xe8000;
        let resident_address = 0x90000;
        let transient_address = 0x92000;
        let mut reservations = vec![Vec::new(); topology.tile_count()];
        reservations[0].push((0x90000, 0xa0000));
        let mut graph = ExecutableGraph {
            memory_policy: Some(ipu_compiler::MemoryPolicy::contiguous(
                arena.start,
                arena.end,
            )),
            schedule: Schedule {
                layouts: Vec::new(),
                phases: Vec::new(),
                allocations: vec![
                    ipu_compiler::Allocation {
                        tensor: ipu_compiler::TensorId(1),
                        tile: 0,
                        address: resident_address,
                        size: 4096,
                        live_from: 0,
                        live_until: usize::MAX,
                        kind: ipu_compiler::AllocationKind::Home,
                    },
                    ipu_compiler::Allocation {
                        tensor: ipu_compiler::TensorId(2),
                        tile: 0,
                        address: transient_address,
                        size: 4096,
                        live_from: 0,
                        live_until: 1,
                        kind: ipu_compiler::AllocationKind::Home,
                    },
                ]
                .into(),
                tile_count: u16::try_from(topology.tile_count()).unwrap(),
                peak_sram: BTreeMap::new(),
            },
            initial_buffers: Vec::new(),
            outputs: Vec::new(),
            host_weights: vec![Binding {
                name: "weight".into(),
                dtype: "u32".into(),
                shape: vec![1],
                slices: vec![RegionSlice {
                    tile: u32::from(topology.physical(0).unwrap()),
                    tile_address: resident_address + 8,
                    file_offset: 0,
                    size: 4,
                }],
            }],
            host_inputs: Vec::new(),
            host_outputs: Vec::new(),
        };

        assert_eq!(
            compact_all_allocations_around(
                &mut graph,
                &topology,
                &reservations,
                None,
                "unit test",
            )
                .unwrap(),
            2
        );
        let resident = &graph.schedule.allocations[0];
        let transient = &graph.schedule.allocations[1];
        for allocation in [resident, transient] {
            assert!(allocation.address >= arena.start);
            assert!(allocation.address + allocation.size <= arena.end);
            assert!(!ranges_overlap(
                allocation.address,
                allocation.address + allocation.size,
                reservations[0][0].0,
                reservations[0][0].1,
            ));
        }
        assert!(!ranges_overlap(
            resident.address,
            resident.address + resident.size,
            transient.address,
            transient.address + transient.size,
        ));
        assert_eq!(
            graph.host_weights[0].slices[0].tile_address,
            resident.address + 8
        );
    }

    #[test]
    fn executable_objects_are_packed_without_order_induced_fragmentation() {
        let element = ipu_package::TILE_MEMORY_ELEMENT_SIZE;
        let gaps = vec![
            (0x10000, 0x10000 + 2 * element),
            (0x20000, 0x20000 + element),
        ];
        let placed = pack_sized_objects_in_gaps(
            0,
            &[element, 2 * element],
            gaps,
            element,
            "test executable",
        )
        .unwrap();

        assert_eq!(placed[0].1 - placed[0].0, element);
        assert_eq!(placed[1].1 - placed[1].0, 2 * element);
        assert!(!ranges_overlap(
            placed[0].0,
            placed[0].1,
            placed[1].0,
            placed[1].1,
        ));
    }

    #[test]
    fn generated_and_support_images_use_independent_executable_gaps() {
        let element = ipu_package::TILE_MEMORY_ELEMENT_SIZE;
        let gaps = vec![(0x10000, 0x10000 + element), (0x20000, 0x20000 + element)];

        let [generated, support] =
            pack_generated_and_support_images(0, element, element, gaps).unwrap();

        assert_eq!(generated.1 - generated.0, element);
        assert_eq!(support.1 - support.0, element);
        assert!(!ranges_overlap(
            generated.0,
            generated.1,
            support.0,
            support.1,
        ));
    }

    #[test]
    fn pinned_executable_images_survive_tensor_relayout() {
        let element = ipu_package::TILE_MEMORY_ELEMENT_SIZE;
        let pinned = [(0x50000, 0x50000 + element), (0x58000, 0x58000 + element)];
        let free = vec![(0x4c000, 0x54000), (0x58000, 0x58000 + 2 * element)];

        assert_eq!(
            validate_pinned_executable_images(17, element / 2, element, pinned, &free).unwrap(),
            pinned
        );
        assert!(
            validate_pinned_executable_images(
                17,
                element / 2,
                element,
                pinned,
                &[(0x4c000, 0x50000), (0x58000, 0x58000 + 2 * element)],
            )
            .is_err()
        );
    }

    #[test]
    fn executable_placement_reserves_complete_memory_elements() {
        let element = ipu_package::TILE_MEMORY_ELEMENT_SIZE;
        let runtime_end = PLAN_BASE + element / 2;
        let allocation_address = align_up(runtime_end, element) + element;
        let graph = ExecutableGraph {
            memory_policy: None,
            host_weights: Vec::new(),
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
                }]
                .into(),
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
    fn fine_profile_steps_include_their_semantic_region() {
        let mut steps = [0, 2, 5].map(|phase| ipu_package::ProfileStep {
            local_index: phase,
            phase,
            epoch: 0,
            operation: format!("phase-{phase}"),
            kind: ipu_package::ProfileStepKind::Compute,
            kernel: "test".into(),
            metadata: Vec::new(),
        });
        let regions = [
            StaticProfileRegion {
                name: "embedding".into(),
                phases: 0..2,
            },
            StaticProfileRegion {
                name: "encoder".into(),
                phases: 2..6,
            },
        ];

        annotate_semantic_regions(&mut steps, &regions).unwrap();

        let names = steps
            .iter()
            .map(|step| {
                step.metadata
                    .iter()
                    .find(|entry| entry.name == "semantic_region")
                    .unwrap()
                    .value
                    .as_str()
            })
            .collect::<Vec<_>>();
        assert_eq!(names, ["embedding", "encoder", "encoder"]);
    }

    #[test]
    fn compute_profile_preserves_kernel_semantics() {
        let command = ipu_compiler::LoweredComputeCommand {
            op: ipu_compiler::OpId(4),
            phase: 7,
            phase_tile_command_index: 0,
            command: Arc::new(ipu_compiler::KernelCommand {
                tile: 7,
                output: ipu_compiler::TensorId(11),
                inputs: vec![ipu_compiler::TensorId(9), ipu_compiler::TensorId(10)],
                arguments: vec![64],
                specialization: Arc::new(ipu_compiler::SpecializationKey {
                    operation: "gemm_f32_accumulate".into(),
                    shape: vec![64, 64, 64],
                    worker_count: 6,
                    role: "inner-block-3".into(),
                    alignment: 32,
                    abi: ipu_compiler::KernelAbi::pace(
                        64 * 64 * 4,
                        64 * 64 * 4,
                        64 * 64 * 4,
                        false,
                    ),
                }),
                metadata: BTreeMap::from([
                    ("label".into(), "GEMM block (2, 5) inner block 3".into()),
                    ("output_block_row".into(), "2".into()),
                ]),
            }),
            output_address: 0x80000,
            input_addresses: smallvec::smallvec![0x50000, 0x54000],
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
            specialization: Arc::new(ipu_compiler::SpecializationKey {
                operation: "gelu_c16_to_a8".into(),
                shape: vec![64, 64],
                worker_count: 6,
                role: "activation".into(),
                alignment: 32,
                abi: ipu_compiler::KernelAbi::Generic,
            }),
            metadata: BTreeMap::from([
                ("label".into(), label.into()),
                ("layer".into(), "3".into()),
            ]),
        };
        let schedule = ipu_compiler::Schedule {
            layouts: Vec::new(),
            phases: vec![ipu_compiler::Phase::Compute {
                op: ipu_compiler::OpId(9),
                commands: vec![
                    command(7, "layer 3 GeLU").into(),
                    command(7, "second command").into(),
                ],
            }],
            allocations: Vec::new().into(),
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
            specialization: Arc::new(ipu_compiler::SpecializationKey {
                operation: "add_u32".into(),
                shape: vec![64],
                worker_count: 6,
                role: "elementwise".into(),
                alignment: 8,
                abi: ipu_compiler::KernelAbi::Generic,
            }),
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
                    commands: vec![scheduled_command.into()],
                },
            ],
            allocations: Vec::new().into(),
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
            memory_policy: None,
            host_weights: Vec::new(),
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
                }]
                .into(),
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

    #[test]
    fn executable_objects_reserve_complete_memory_elements() {
        let element = ipu_package::TILE_MEMORY_ELEMENT_SIZE;
        let allocation = (0x60080, 0x60100);
        let objects = [0x1000..0x1080, 0x2000..0x2080];

        let (relocations, storage, elements) =
            pack_executable_objects_for_tile(&[allocation], 0, PLAN_BASE, &objects, &[], &[])
                .unwrap();

        assert_eq!(relocations.len(), objects.len());
        assert!(elements.iter().all(|&(start, end)| {
            start % element == 0
                && end % element == 0
                && !ranges_overlap(start, end, allocation.0, allocation.1)
        }));
        assert!(relocations.values().all(|&address| {
            storage
                .iter()
                .any(|&(start, end)| address >= start && address < end)
        }));
    }

    #[test]
    fn executable_tail_cannot_be_reused_for_data() {
        let element = ipu_package::TILE_MEMORY_ELEMENT_SIZE;
        let code_start = ipu_package::TILE_MEMORY_BASE + 3 * element;
        let used_code_end = code_start + element / 2;
        let executable = effective_element_reservations([(code_start, used_code_end)]).unwrap();

        assert!(
            validate_data_element_separation(
                0,
                "test data",
                &[(used_code_end, used_code_end + 4)],
                &executable,
            )
            .is_err()
        );
        assert!(
            validate_data_element_separation(
                0,
                "test data",
                &[(code_start + element, code_start + element + 4)],
                &executable,
            )
            .is_ok()
        );
    }
}
