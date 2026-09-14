//! Deterministic placement of logical shards in IPU21 tile SRAM.

mod dump;
mod exchange;
pub(crate) mod profile;
mod search;
pub(crate) use exchange::ExchangeConflicts;

use crate::MemoryClass;
use crate::low::{LowProgram, TileWorkList, TileWorkRef};
use crate::memory::IPU21_DATA_BASE;
use crate::{BlockValueId, ShardDefinition};
use crate::{StorageError, shard_storage_bytes};
use ipu_package::{
    IPU21_APPLICATION_MEMORY_LIMIT, IPU21_INTERLEAVED_ELEMENT_SIZE, IPU21_INTERLEAVED_MEMORY_BASE,
    TILE_MEMORY_ELEMENT_SIZE,
};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub shard_addresses: BTreeMap<BlockValueId, u32>,
    /// Actual sequence stride keyed by the Repeat body argument.
    /// Empty shards may share an address and have zero stride.
    pub sequence_strides: BTreeMap<BlockValueId, u32>,
    pub tile_auxiliary_ranges: Vec<Vec<(u32, u32)>>,
    pub auxiliary_allocations: Vec<Vec<AuxiliaryAllocation>>,
}

/// Non-tensor storage participating in the same lifetime/geometry search.
#[derive(Clone, Debug)]
pub(crate) struct AuxiliaryRequest {
    pub name: String,
    pub bytes: u32,
    pub alignment: u32,
    pub first: u32,
    pub last: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuxiliaryAllocation {
    pub name: String,
    pub address: u32,
    pub bytes: u32,
    pub first: u32,
    pub last: u32,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum PlacementError {
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error("shard {0} refers to an invalid alias")]
    InvalidAlias(u32),
    #[error("alias group crosses tile or memory-class boundaries")]
    IncompatibleAlias,
    #[error("repeat iterated placement constraints overlap incompatibly")]
    IteratedOverlap,
    #[error("repeat inputs do not form a valid word-aligned sequence")]
    IteratedStride,
    #[error("tile {tile} has insufficient {class:?} SRAM for {bytes} bytes")]
    OutOfMemory {
        tile: u16,
        class: MemoryClass,
        bytes: u32,
    },
    #[error("placement arithmetic overflowed")]
    Overflow,
}

#[derive(Clone, Copy, Debug, Default)]
struct Requirement {
    alignment: u32,
    access_tail: u32,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
struct Lifetime {
    first: u32,
    last: u32,
    seen: bool,
}

impl Lifetime {
    fn touch(&mut self, event: u32) {
        if self.seen {
            self.first = self.first.min(event);
            self.last = self.last.max(event);
        } else {
            self.first = event;
            self.last = event;
            self.seen = true;
        }
    }

    fn include(&mut self, other: Self) {
        if other.seen {
            self.touch(other.first);
            self.touch(other.last);
        }
    }
}

/// The host protocol owns this range before and after device execution.
/// It is never available to persistent support, inputs or outputs.
pub(crate) const HOST_SCRATCH_RANGE: (u32, u32) = (
    ipu_exchange::EXCHANGE_WINDOW_BASE,
    ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES,
);

pub fn place(program: &LowProgram) -> Result<Placement, PlacementError> {
    place_with_ranges(
        program,
        &[
            HOST_SCRATCH_RANGE,
            (IPU21_DATA_BASE, IPU21_APPLICATION_MEMORY_LIMIT),
        ],
    )
}

pub(crate) fn place_with_ranges(
    program: &LowProgram,
    available_ranges: &[(u32, u32)],
) -> Result<Placement, PlacementError> {
    place_with_offset(program, available_ranges, 0)
}

pub(crate) fn place_with_offset(
    program: &LowProgram,
    available_ranges: &[(u32, u32)],
    interleaved_offset: u32,
) -> Result<Placement, PlacementError> {
    place_with_auxiliary(program, available_ranges, interleaved_offset, &[])
}

pub(crate) fn place_with_auxiliary(
    program: &LowProgram,
    available_ranges: &[(u32, u32)],
    interleaved_offset: u32,
    auxiliary: &[Vec<AuxiliaryRequest>],
) -> Result<Placement, PlacementError> {
    let started = std::time::Instant::now();
    if auxiliary
        .iter()
        .flatten()
        .any(|r| r.bytes == 0 || !r.alignment.is_power_of_two() || r.first > r.last)
    {
        return Err(PlacementError::Overflow);
    }
    if interleaved_offset >= IPU21_INTERLEAVED_ELEMENT_SIZE {
        return Err(PlacementError::Overflow);
    }
    if available_ranges.iter().any(|&(start, end)| {
        (start < IPU21_DATA_BASE && (start, end) != HOST_SCRATCH_RANGE)
            || end > IPU21_APPLICATION_MEMORY_LIMIT
            || start >= end
    }) || available_ranges
        .windows(2)
        .any(|pair| pair[0].1 > pair[1].0)
    {
        return Err(PlacementError::OutOfMemory {
            tile: 0,
            class: MemoryClass::Ipu21Standard,
            bytes: 0,
        });
    }
    let mut analysis = analyze_allocations(program)?;
    let tile_count = usize::from(program.tile_count).max(auxiliary.len());
    analysis
        .tiles
        .resize_with(tile_count, TileAllocations::default);
    let tile_placements = (0..tile_count)
        .into_par_iter()
        .map(|tile| {
            place_tile(
                program,
                u16::try_from(tile).map_err(|_| PlacementError::Overflow)?,
                available_ranges,
                interleaved_offset,
                &analysis,
                auxiliary.get(tile).map(Vec::as_slice).unwrap_or(&[]),
            )
        })
        .collect::<Result<Vec<_>, PlacementError>>()?;
    let mut addresses = BTreeMap::new();
    let mut sequence_strides = BTreeMap::new();
    let mut tile_auxiliary_ranges = vec![Vec::new(); tile_count];
    let mut auxiliary_allocations = vec![Vec::new(); tile_count];
    for (tile, placed) in tile_placements.into_iter().enumerate() {
        auxiliary_allocations[tile] = placed.auxiliary;
        addresses.extend(placed.addresses);
        sequence_strides.extend(placed.sequence_strides);
        tile_auxiliary_ranges[tile] = placed.unused;
    }

    tracing::debug!(
        tiles = program.tile_count,
        shards = program.shards.len(),
        elapsed_us = started.elapsed().as_micros(),
        "placed tile storage"
    );
    Ok(Placement {
        shard_addresses: addresses,
        sequence_strides,
        tile_auxiliary_ranges,
        auxiliary_allocations,
    })
}

struct AllocationAnalysis {
    tiles: Vec<TileAllocations>,
    root_of_member: Vec<usize>,
    member_offsets: Vec<u32>,
    root_requirements: BTreeMap<usize, Requirement>,
    root_lifetimes: BTreeMap<usize, Lifetime>,
    conflicts: BTreeMap<usize, BTreeSet<usize>>,
}

#[derive(Default)]
struct TileAllocations {
    iterated: Vec<IteratedGroup>,
    members: BTreeMap<usize, Vec<usize>>,
}

fn analyze_allocations(program: &LowProgram) -> Result<AllocationAnalysis, PlacementError> {
    let mut sets = DisjointSets::new(program.shards.len());
    for shard in &program.shards {
        let (target, offset) = match shard.definition {
            ShardDefinition::Alias(target) | ShardDefinition::WritableAlias(target) => (target, 0),
            ShardDefinition::ShiftedAlias { source, offset } => (source, i64::from(offset)),
            _ => continue,
        };
        if program
            .shards
            .get(target.index() as usize)
            .is_none_or(|other| other.tile != shard.tile)
        {
            return Err(PlacementError::InvalidAlias(target.index()));
        }
        sets.union_offset(shard.id.index() as usize, target.index() as usize, offset)?;
    }
    let mut iterated = Vec::<IteratedGroup>::new();
    for tile in &program.tiles {
        collect_repeat_constraints(program, tile, &mut sets, &mut iterated)?;
    }

    let mut pairs = Vec::new();
    let mut requirements = vec![Requirement::default(); program.shards.len()];
    for tile in &program.tiles {
        collect_requirements(program, tile, &mut requirements, &mut pairs);
    }
    // Loopback reads and receives simultaneously. Both access classes share
    // physical memory, so each local source/destination pair must be separated.
    for transfer in program
        .exchange_phases
        .iter()
        .flat_map(|phase| &phase.transfers)
    {
        let source = &program.shards[transfer.source.shard.index() as usize];
        for destination in &transfer.destinations {
            let destination = &program.shards[destination.shard.index() as usize];
            if source.tile == destination.tile {
                pairs.push((source.id.index() as usize, destination.id.index() as usize));
            }
        }
    }
    let mut root_requirements = BTreeMap::<usize, Requirement>::new();
    for (index, requirement) in requirements.iter().copied().enumerate() {
        let root = sets.find(index);
        let combined = root_requirements.entry(root).or_default();
        combined.alignment = combined.alignment.max(requirement.alignment);
        if matches!(
            program.shards[index].definition,
            ShardDefinition::ShiftedAlias { .. }
        ) {
            combined.alignment = combined.alignment.max(32768);
        }
        combined.access_tail = combined.access_tail.max(requirement.access_tail);
    }

    let mut members = BTreeMap::<usize, Vec<usize>>::new();
    let mut root_of_member = vec![0usize; program.shards.len()];
    for (index, root_slot) in root_of_member.iter_mut().enumerate() {
        let root = sets.find(index);
        *root_slot = root;
        members.entry(root).or_default().push(index);
    }
    let mut member_offsets = vec![0; program.shards.len()];
    for group in members.values() {
        let minimum = group.iter().map(|&i| sets.offsets[i]).min().unwrap_or(0);
        for &i in group {
            member_offsets[i] =
                u32::try_from(sets.offsets[i] - minimum).map_err(|_| PlacementError::Overflow)?;
            if !member_offsets[i].is_multiple_of(requirements[i].alignment.max(1)) {
                return Err(PlacementError::IncompatibleAlias);
            }
        }
    }
    validate_alias_groups(program, &members)?;
    let mut conflicts = BTreeMap::<usize, BTreeSet<usize>>::new();
    for (left, right) in pairs {
        let (left, right) = (root_of_member[left], root_of_member[right]);
        if left == right {
            return Err(PlacementError::IncompatibleAlias);
        }
        conflicts.entry(left).or_default().insert(right);
        conflicts.entry(right).or_default().insert(left);
    }
    let lifetimes = collect_lifetimes(program);
    let mut root_lifetimes = BTreeMap::<usize, Lifetime>::new();
    for (index, lifetime) in lifetimes.into_iter().enumerate() {
        root_lifetimes
            .entry(root_of_member[index])
            .or_default()
            .include(lifetime);
    }

    let mut tiles = (0..program.tile_count)
        .map(|_| TileAllocations::default())
        .collect::<Vec<_>>();
    for (root, group) in members {
        tiles[usize::from(program.shards[group[0]].tile)]
            .members
            .insert(root, group);
    }
    for group in iterated {
        tiles[usize::from(group.tile)].iterated.push(group);
    }

    Ok(AllocationAnalysis {
        tiles,
        root_of_member,
        member_offsets,
        root_requirements,
        root_lifetimes,
        conflicts,
    })
}

struct TilePlacement {
    addresses: BTreeMap<BlockValueId, u32>,
    sequence_strides: BTreeMap<BlockValueId, u32>,
    unused: Vec<(u32, u32)>,
    auxiliary: Vec<AuxiliaryAllocation>,
}

fn place_tile(
    program: &LowProgram,
    tile: u16,
    available_ranges: &[(u32, u32)],
    interleaved_offset: u32,
    analysis: &AllocationAnalysis,
    auxiliary: &[AuxiliaryRequest],
) -> Result<TilePlacement, PlacementError> {
    // Both access classes share region 1. A single lifetime-ordered arena
    // lets ordinary storage reuse dead interleaved buffers and vice versa.
    let mut addresses = BTreeMap::new();
    // Host packet bits do not preserve the finite-F16 padding invariant.
    let ranges = available_ranges
        .iter()
        .copied()
        .filter(|range| !program.requires_finite_scratch || *range != HOST_SCRATCH_RANGE)
        .collect::<Vec<_>>();
    let mut arena = Arena::new(&ranges, interleaved_offset);
    allocate_tile(
        program,
        tile,
        analysis,
        &mut arena,
        &mut addresses,
        auxiliary,
    )?;
    let mut sequence_strides = BTreeMap::new();
    for group in &analysis.tiles[usize::from(tile)].iterated {
        // Bind the logical sequence after region selection and alias offsets
        // are known. A singleton advances by its reservation size; a sequence
        // of shifted aliases may advance differently from its backing roots.
        let initial = addresses[&group.argument];
        let stride = if let Some(second) = group.shards.get(1) {
            addresses[second]
                .checked_sub(initial)
                .ok_or(PlacementError::IteratedStride)?
        } else {
            let root = analysis.root_of_member[group.shards[0].index() as usize];
            let (start, end) = arena.root_spans[&root];
            end - start
        };
        if !stride.is_multiple_of(4) {
            return Err(PlacementError::IteratedStride);
        }
        for (iteration, shard) in group.shards.iter().enumerate() {
            if u64::from(addresses[shard])
                != u64::from(initial) + u64::from(stride) * iteration as u64
            {
                return Err(PlacementError::IteratedStride);
            }
        }
        sequence_strides.insert(group.argument, stride);
    }
    let allocations = auxiliary
        .iter()
        .enumerate()
        .map(|(index, request)| AuxiliaryAllocation {
            name: request.name.clone(),
            address: arena.auxiliary_addresses[&index],
            bytes: request.bytes,
            first: request.first,
            last: request.last,
        })
        .collect();
    Ok(TilePlacement {
        addresses,
        sequence_strides,
        unused: arena.unused_ranges(),
        auxiliary: allocations,
    })
}

fn shards_by_tile(
    program: &LowProgram,
    shards: impl IntoIterator<Item = BlockValueId>,
) -> Vec<Vec<BlockValueId>> {
    let mut tiles = vec![Vec::new(); usize::from(program.tile_count)];
    for shard in shards {
        tiles[usize::from(program.shards[shard.index() as usize].tile)].push(shard);
    }
    for shards in &mut tiles {
        shards.sort_unstable();
        shards.dedup();
    }
    tiles
}

fn collect_lifetimes(program: &LowProgram) -> Vec<Lifetime> {
    // Each global phase appears in every tile's projection. Index its touched
    // blocks once instead of scanning all device transfers once per tile.
    let exchanges = program
        .exchange_phases
        .iter()
        .map(|phase| {
            shards_by_tile(
                program,
                phase.transfers.iter().flat_map(|transfer| {
                    std::iter::once(transfer.source.shard)
                        .chain(transfer.destinations.iter().map(|view| view.shard))
                }),
            )
        })
        .collect::<Vec<_>>();
    let mut lifetimes = vec![Lifetime::default(); program.shards.len()];
    for input in &program.inputs {
        for shard in program.value_shards(input.value) {
            lifetimes[shard.index() as usize].touch(0);
            // initialize uploads parameters once; every subsequent run needs them.
            if input.kind == crate::GraphInputKind::Parameter {
                lifetimes[shard.index() as usize].touch(u32::MAX);
            }
        }
    }
    for tile in &program.tiles {
        let mut event = 1u32;
        for work in program.work(tile) {
            touch_work(
                program,
                work,
                tile.tile,
                &mut event,
                &mut lifetimes,
                &exchanges,
            );
        }
    }
    // The host reads every output only after device work has finished.
    for output in &program.outputs {
        for shard in program.value_shards(*output) {
            lifetimes[shard.index() as usize].touch(u32::MAX);
        }
    }
    for (index, lifetime) in lifetimes.iter_mut().enumerate() {
        if !lifetime.seen
            && !matches!(
                program.shards[index].definition,
                ShardDefinition::Unmaterialized
            )
        {
            lifetime.touch(0);
        }
    }
    lifetimes
}

fn touch_work(
    program: &LowProgram,
    work: TileWorkRef<'_>,
    tile: u16,
    event: &mut u32,
    lifetimes: &mut [Lifetime],
    exchanges: &[Vec<Vec<BlockValueId>>],
) {
    let current = *event;
    let mut touch = |shard: BlockValueId| lifetimes[shard.index() as usize].touch(current);
    match work {
        TileWorkRef::Kernel(run) => {
            for view in run.inputs.iter().chain(&run.outputs) {
                touch(view.shard);
            }
        }
        TileWorkRef::LocalCopy(copy) => {
            touch(copy.source);
            touch(copy.destination);
        }
        TileWorkRef::Exchange(id) => {
            for &shard in &exchanges[id.index() as usize][usize::from(tile)] {
                touch(shard);
            }
        }
        TileWorkRef::Repeat(repeat) => {
            for shard in repeat.binding.bound_shards() {
                touch(shard);
            }
            *event = event.saturating_add(1);
            for nested in program.work(&repeat.body) {
                touch_work(program, nested, tile, event, lifetimes, exchanges);
            }
            let end = *event;
            for shard in repeat.binding.bound_shards() {
                lifetimes[shard.index() as usize].touch(end);
            }
            *event = event.saturating_add(1);
            return;
        }
        TileWorkRef::Checkpoint(..) => {}
    }
    *event = event.saturating_add(1);
}

fn collect_repeat_constraints(
    program: &LowProgram,
    tile: &TileWorkList,
    sets: &mut DisjointSets,
    iterated: &mut Vec<IteratedGroup>,
) -> Result<(), PlacementError> {
    for work in program.work(tile) {
        let TileWorkRef::Repeat(repeat) = work else {
            continue;
        };
        for carried in &repeat.binding.carried {
            checked_union(program, sets, carried.initial, carried.argument)?;
            checked_union(program, sets, carried.initial, carried.yielded)?;
            checked_union(program, sets, carried.initial, carried.result)?;
        }
        for invariant in &repeat.binding.invariants {
            checked_union(program, sets, invariant.input, invariant.argument)?;
        }
        for input in &repeat.binding.iterated {
            let first = *input.inputs.first().ok_or(PlacementError::IteratedStride)?;
            checked_union(program, sets, first, input.argument)?;
            iterated.push(IteratedGroup {
                tile: tile.tile,
                shards: input.inputs.clone(),
                argument: input.argument,
            });
        }
        collect_repeat_constraints(program, &repeat.body, sets, iterated)?;
    }
    Ok(())
}

fn collect_requirements(
    program: &LowProgram,
    tile: &TileWorkList,
    requirements: &mut [Requirement],
    pairs: &mut Vec<(usize, usize)>,
) {
    for work in program.work(tile) {
        match work {
            TileWorkRef::Kernel(run) => {
                for operands in &run.requirements.distinct_elements {
                    let shards = operands
                        .iter()
                        .map(|&operand| {
                            run.operand_view(operand)
                                .expect("kernel constraints refer to bound operands")
                                .shard
                                .index() as usize
                        })
                        .collect::<Vec<_>>();
                    for (index, &left) in shards.iter().enumerate() {
                        for &right in &shards[..index] {
                            pairs.push((left, right));
                        }
                    }
                }
                for (view, requirement) in run
                    .inputs
                    .iter()
                    .zip(&run.requirements.inputs)
                    .chain(run.outputs.iter().zip(&run.requirements.outputs))
                {
                    apply_requirement(&mut requirements[view.shard.index() as usize], requirement);
                }
            }
            TileWorkRef::LocalCopy(copy) => {
                requirements[copy.source.index() as usize].alignment =
                    requirements[copy.source.index() as usize].alignment.max(8);
                requirements[copy.destination.index() as usize].alignment = requirements
                    [copy.destination.index() as usize]
                    .alignment
                    .max(8);
            }
            TileWorkRef::Repeat(repeat) => {
                collect_requirements(program, &repeat.body, requirements, pairs)
            }
            TileWorkRef::Exchange(_) | TileWorkRef::Checkpoint(..) => {}
        }
    }
}

fn apply_requirement(target: &mut Requirement, requirement: &crate::KernelAccess) {
    target.alignment = target.alignment.max(requirement.alignment);
    target.access_tail = target.access_tail.max(requirement.access_tail_bytes);
}

fn checked_union(
    program: &LowProgram,
    sets: &mut DisjointSets,
    left: BlockValueId,
    right: BlockValueId,
) -> Result<(), PlacementError> {
    let Some(left_shard) = program.shards.get(left.index() as usize) else {
        return Err(PlacementError::InvalidAlias(left.index()));
    };
    let Some(right_shard) = program.shards.get(right.index() as usize) else {
        return Err(PlacementError::InvalidAlias(right.index()));
    };
    if left_shard.tile != right_shard.tile {
        return Err(PlacementError::IncompatibleAlias);
    }
    sets.union_offset(left.index() as usize, right.index() as usize, 0)
}

fn validate_alias_groups(
    program: &LowProgram,
    members: &BTreeMap<usize, Vec<usize>>,
) -> Result<(), PlacementError> {
    for group in members.values() {
        let first = &program.shards[group[0]];
        if group.iter().any(|&index| {
            let shard = &program.shards[index];
            shard.tile != first.tile
                || shard.tensor_type.format.layout.memory_class
                    != first.tensor_type.format.layout.memory_class
        }) {
            return Err(PlacementError::IncompatibleAlias);
        }
    }
    Ok(())
}

fn allocation_bytes(
    program: &LowProgram,
    members: &[usize],
    member_offsets: &[u32],
    requirement: Requirement,
) -> Result<u32, PlacementError> {
    members.iter().try_fold(0, |maximum, &index| {
        member_offsets[index]
            .checked_add(shard_storage_bytes(&program.shards[index])?)
            .and_then(|bytes| bytes.checked_add(requirement.access_tail))
            .map(|bytes| maximum.max(bytes))
            .ok_or(PlacementError::Overflow)
    })
}

fn memory_element_size(program: &LowProgram, members: &[usize]) -> u32 {
    match program.shards[members[0]]
        .tensor_type
        .format
        .layout
        .memory_class
    {
        MemoryClass::Ipu21Standard => TILE_MEMORY_ELEMENT_SIZE,
        MemoryClass::Ipu21Interleaved => IPU21_INTERLEAVED_ELEMENT_SIZE,
    }
}

fn assign_members(
    addresses: &mut BTreeMap<BlockValueId, u32>,
    members: &[usize],
    member_offsets: &[u32],
    address: u32,
) -> Result<(), PlacementError> {
    for &member in members {
        addresses.insert(
            BlockValueId::from_index(u32::try_from(member).map_err(|_| PlacementError::Overflow)?),
            address
                .checked_add(member_offsets[member])
                .ok_or(PlacementError::Overflow)?,
        );
    }
    Ok(())
}

fn allocation_requests(
    program: &LowProgram,
    analysis: &AllocationAnalysis,
    tile: u16,
) -> Result<Vec<AllocationRequest>, PlacementError> {
    let TileAllocations { iterated, members } = &analysis.tiles[usize::from(tile)];
    let AllocationAnalysis {
        root_of_member,
        member_offsets,
        root_requirements,
        root_lifetimes,
        conflicts,
        ..
    } = analysis;
    let mut grouped = BTreeSet::new();
    for group in iterated {
        for shard in &group.shards {
            if !grouped.insert(root_of_member[shard.index() as usize]) {
                return Err(PlacementError::IteratedOverlap);
            }
        }
    }
    let mut requests = Vec::<AllocationRequest>::new();
    for group in iterated {
        let roots = group
            .shards
            .iter()
            .map(|shard| root_of_member[shard.index() as usize])
            .collect::<Vec<_>>();
        let group_class = program.shards[group.shards[0].index() as usize]
            .tensor_type
            .format
            .layout
            .memory_class;
        // Only sequence members that are actual simultaneous operands require
        // separate elements. External conflicts protect the whole sequence, so
        // the current argument's constraint holds on every Repeat iteration.
        let separate_members = roots.iter().any(|root| {
            conflicts
                .get(root)
                .is_some_and(|others| roots.iter().any(|other| others.contains(other)))
        });
        let alignment = 4.max(
            roots
                .iter()
                .map(|root| {
                    if separate_members {
                        root_requirements[root]
                            .alignment
                            .max(memory_element_size(program, &members[root]))
                    } else {
                        root_requirements[root].alignment
                    }
                })
                .max()
                .unwrap_or(1),
        );
        // Kernel access and loopback bank constraints are only complete after
        // expansion. Derive the physical repeat stride from those requirements.
        let mut stride = 0;
        for root in &roots {
            stride = stride.max(allocation_bytes(
                program,
                &members[root],
                member_offsets,
                root_requirements[root],
            )?);
        }
        let stride = align_up(stride, alignment)?;
        let bytes = stride
            .checked_mul(u32::try_from(roots.len()).map_err(|_| PlacementError::Overflow)?)
            .ok_or(PlacementError::Overflow)?;
        let mut lifetime = Lifetime::default();
        let mut assignments = Vec::with_capacity(roots.len());
        for (index, root) in roots.iter().copied().enumerate() {
            lifetime.include(root_lifetimes[&root]);
            assignments.push((
                root,
                stride
                    .checked_mul(u32::try_from(index).map_err(|_| PlacementError::Overflow)?)
                    .ok_or(PlacementError::Overflow)?,
            ));
        }
        requests.push(AllocationRequest {
            auxiliary: None,
            class: group_class,
            region1_stride: separate_members
                .then(|| align_up(stride, IPU21_INTERLEAVED_ELEMENT_SIZE))
                .transpose()?,
            conflicts: roots
                .iter()
                .filter_map(|root| conflicts.get(root))
                .flatten()
                .filter(|other| !roots.contains(other))
                .copied()
                .collect(),
            lifetime,
            bytes,
            alignment,
            assignments,
        });
    }
    for (&root, root_members) in members {
        let representative = &program.shards[root_members[0]];
        let lifetime = root_lifetimes.get(&root).copied().unwrap_or_default();
        if grouped.contains(&root) || !lifetime.seen {
            continue;
        }
        let requirement = root_requirements.get(&root).copied().unwrap_or_default();
        let bytes = allocation_bytes(program, root_members, member_offsets, requirement)?;
        requests.push(AllocationRequest {
            auxiliary: None,
            class: representative.tensor_type.format.layout.memory_class,
            region1_stride: None,
            conflicts: conflicts
                .get(&root)
                .into_iter()
                .flatten()
                .copied()
                .collect(),
            lifetime,
            bytes,
            alignment: requirement.alignment.max(4),
            assignments: vec![(root, 0)],
        });
    }
    Ok(requests)
}

fn allocate_tile(
    program: &LowProgram,
    tile: u16,
    analysis: &AllocationAnalysis,
    arena: &mut Arena,
    addresses: &mut BTreeMap<BlockValueId, u32>,
    auxiliary: &[AuxiliaryRequest],
) -> Result<(), PlacementError> {
    let members = &analysis.tiles[usize::from(tile)].members;
    let member_offsets = &analysis.member_offsets;
    let mut requests = allocation_requests(program, analysis, tile)?;
    requests.extend(
        auxiliary
            .iter()
            .enumerate()
            .map(|(index, request)| AllocationRequest {
                auxiliary: Some(index),
                class: MemoryClass::Ipu21Standard,
                region1_stride: None,
                lifetime: Lifetime {
                    first: request.first,
                    last: request.last,
                    seen: true,
                },
                bytes: request.bytes,
                alignment: request.alignment,
                assignments: Vec::new(),
                conflicts: Vec::new(),
            }),
    );
    requests.sort_by_key(|request| {
        (
            request.lifetime.first,
            request.class != MemoryClass::Ipu21Interleaved,
            std::cmp::Reverse(request.alignment),
            std::cmp::Reverse(request.bytes),
            request.lifetime.last,
        )
    });
    let initial = arena.clone();
    match allocate_requests(
        program,
        tile,
        &requests,
        members,
        member_offsets,
        arena,
        addresses,
    ) {
        Ok(()) => {
            dump::capture(tile, &requests, arena, true);
            return Ok(());
        }
        Err(PlacementError::OutOfMemory { .. }) => {}
        Err(error) => return Err(error),
    }
    *arena = initial.clone();
    arena.offline = true;
    addresses.clear();
    requests.sort_by_key(|request| {
        (
            std::cmp::Reverse(request.alignment),
            std::cmp::Reverse(request.bytes),
            std::cmp::Reverse(request.lifetime.last - request.lifetime.first),
            request.lifetime.first,
        )
    });
    let result = allocate_requests(
        program,
        tile,
        &requests,
        members,
        member_offsets,
        arena,
        addresses,
    );
    if matches!(result, Err(PlacementError::OutOfMemory { .. })) {
        let searched = search::place(&requests, &initial);
        tracing::debug!(
            tile,
            nodes = searched.nodes,
            excess_live_bytes = searched.excess_live_bytes,
            recovered = searched.placement.is_some(),
            "bounded tile placement search"
        );
        if let Some(placement) = searched.placement {
            *arena = initial;
            arena.offline = true;
            addresses.clear();
            for (request, (start, end)) in requests.iter().zip(placement) {
                arena.record(request, start, end);
                assign_request(request, start, members, member_offsets, addresses)?;
            }
            dump::capture(tile, &requests, arena, true);
            return Ok(());
        }
    }
    dump::capture(tile, &requests, arena, result.is_ok());
    result?;
    tracing::debug!(
        tile,
        "recovered fragmented tile with size-ordered placement"
    );
    Ok(())
}

fn allocate_requests(
    program: &LowProgram,
    tile: u16,
    requests: &[AllocationRequest],
    members: &BTreeMap<usize, Vec<usize>>,
    member_offsets: &[u32],
    arena: &mut Arena,
    addresses: &mut BTreeMap<BlockValueId, u32>,
) -> Result<(), PlacementError> {
    for request in requests {
        let class = request.class;
        let Some(base) = arena.allocate(request) else {
            let representative = request
                .assignments
                .first()
                .map(|(root, _)| &program.shards[members[root][0]]);
            tracing::debug!(
                tile,
                ?class,
                bytes = request.bytes,
                alignment = request.alignment,
                first = request.lifetime.first,
                last = request.lifetime.last,
                free = ?arena.free,
                active = ?arena.active,
                tensor_type = ?representative.map(|shard| &shard.tensor_type),
                "tile allocation does not fit"
            );
            return Err(PlacementError::OutOfMemory {
                tile,
                class,
                bytes: request.bytes,
            });
        };
        assign_request(request, base, members, member_offsets, addresses)?;
    }
    Ok(())
}

fn assign_request(
    request: &AllocationRequest,
    base: u32,
    members: &BTreeMap<usize, Vec<usize>>,
    member_offsets: &[u32],
    addresses: &mut BTreeMap<BlockValueId, u32>,
) -> Result<(), PlacementError> {
    for (index, (root, offset)) in request.assignments.iter().copied().enumerate() {
        let offset = if base >= IPU21_INTERLEAVED_MEMORY_BASE {
            request.region1_stride.map_or(Ok(offset), |stride| {
                stride
                    .checked_mul(u32::try_from(index).map_err(|_| PlacementError::Overflow)?)
                    .ok_or(PlacementError::Overflow)
            })?
        } else {
            offset
        };
        let address = base.checked_add(offset).ok_or(PlacementError::Overflow)?;
        assign_members(addresses, &members[&root], member_offsets, address)?;
    }
    Ok(())
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct AllocationRequest {
    #[serde(default)]
    auxiliary: Option<usize>,
    class: MemoryClass,
    /// Iterated values need a wider physical stride if placed in region 1.
    region1_stride: Option<u32>,
    lifetime: Lifetime,
    bytes: u32,
    alignment: u32,
    assignments: Vec<(usize, u32)>,
    conflicts: Vec<usize>,
}

#[derive(Clone, Debug)]
struct IteratedGroup {
    tile: u16,
    shards: Vec<BlockValueId>,
    argument: BlockValueId,
}

#[derive(Clone)]
struct Arena {
    offline: bool,
    history: Vec<(Lifetime, u32, u32)>,
    root_spans: BTreeMap<usize, (u32, u32)>,
    auxiliary_addresses: BTreeMap<usize, u32>,
    ranges: Vec<(u32, u32)>,
    free: Vec<(u32, u32)>,
    active: Vec<(u32, u32, u32)>,
    occupied: Vec<(u32, u32)>,
    interleaved_offset: u32,
}

impl Arena {
    fn new(ranges: &[(u32, u32)], interleaved_offset: u32) -> Self {
        Self {
            offline: false,
            history: Vec::new(),
            root_spans: BTreeMap::new(),
            auxiliary_addresses: BTreeMap::new(),
            ranges: ranges.to_vec(),
            free: ranges.to_vec(),
            active: Vec::new(),
            occupied: Vec::new(),
            interleaved_offset,
        }
    }

    fn allocate(&mut self, request: &AllocationRequest) -> Option<u32> {
        let first = request.lifetime.first;
        let last = request.lifetime.last;
        if self.offline {
            self.free.clone_from(&self.ranges);
            for &(lifetime, base, end) in &self.history {
                if lifetime.last < first || last < lifetime.first {
                    continue;
                }
                self.free = self
                    .free
                    .iter()
                    .flat_map(|&(start, limit)| {
                        crate::memory::uncovered_ranges(start, limit, &[(base, end)])
                    })
                    .collect();
            }
            self.active.clear();
        }
        let mut retained = Vec::with_capacity(self.active.len());
        let active = std::mem::take(&mut self.active);
        for (active_last, address, active_bytes) in active {
            if active_last < first {
                self.release(address, address.checked_add(active_bytes)?);
            } else {
                retained.push((active_last, address, active_bytes));
            }
        }
        self.active = retained;
        // Expand only conflicting allocations to their effective memory-element
        // boundaries. Region 1 interleaves each pair even for standard accesses.
        let forbidden = crate::memory::merge_ranges(
            request
                .conflicts
                .iter()
                .filter_map(|root| self.root_spans.get(root))
                .flat_map(|&(start, end)| element_spans(start, end))
                .collect(),
        );
        let candidate = request
            .domains(&self.free, self.interleaved_offset)
            .filter_map(|(index, domain)| {
                let mut start = domain.first;
                for &(a, b) in &forbidden {
                    if start < b && a < start.checked_add(domain.bytes)? {
                        start = align_up(b, domain.alignment).ok()?;
                    }
                }
                (start <= domain.last).then_some((
                    (start >= IPU21_INTERLEAVED_MEMORY_BASE, start),
                    index,
                    start,
                    start + domain.bytes,
                ))
            })
            .min_by_key(|candidate| (candidate.0, candidate.1));
        if let Some((_, index, start, end)) = candidate {
            let (base, limit) = self.free[index];
            self.free.remove(index);
            if base < start {
                self.free.push((base, start));
            }
            if end < limit {
                self.free.push((end, limit));
            }
            self.free.sort_unstable();
            self.record(request, start, end);
            return Some(start);
        }
        None
    }

    fn record(&mut self, request: &AllocationRequest, start: u32, end: u32) {
        if let Some(index) = request.auxiliary {
            self.auxiliary_addresses.insert(index, start);
        }
        for &(root, _) in &request.assignments {
            self.root_spans.insert(root, (start, end));
        }
        self.active
            .push((request.lifetime.last, start, end - start));
        self.occupied.push((start, end));
        if self.offline {
            self.history.push((request.lifetime, start, end));
        }
    }

    fn release(&mut self, base: u32, limit: u32) {
        self.free.push((base, limit));
        self.free = crate::memory::merge_ranges(std::mem::take(&mut self.free));
    }

    fn unused_ranges(&self) -> Vec<(u32, u32)> {
        let merged = crate::memory::merge_ranges(self.occupied.clone());
        self.ranges
            .iter()
            // Even unused aperture bytes belong to the next host exchange.
            .filter(|&&range| range != HOST_SCRATCH_RANGE)
            .flat_map(|&(base, limit)| crate::memory::uncovered_ranges(base, limit, &merged))
            .collect()
    }
}

/// Logical intervals covering all physical memory elements touched by a span.
fn element_spans(start: u32, end: u32) -> impl Iterator<Item = (u32, u32)> {
    [false, true].into_iter().filter_map(move |region1| {
        let (start, end, size) = if region1 {
            (
                start.max(IPU21_INTERLEAVED_MEMORY_BASE),
                end,
                IPU21_INTERLEAVED_ELEMENT_SIZE,
            )
        } else {
            (
                start,
                end.min(IPU21_INTERLEAVED_MEMORY_BASE),
                TILE_MEMORY_ELEMENT_SIZE,
            )
        };
        (start < end).then(|| (start / size * size, end.div_ceil(size) * size))
    })
}

fn align_up(value: u32, alignment: u32) -> Result<u32, PlacementError> {
    let alignment = alignment.max(1);
    if !alignment.is_power_of_two() {
        return Err(PlacementError::Overflow);
    }
    value
        .checked_next_multiple_of(alignment)
        .ok_or(PlacementError::Overflow)
}

struct DisjointSets {
    parents: Vec<usize>,
    offsets: Vec<i64>,
}

impl DisjointSets {
    fn new(length: usize) -> Self {
        Self {
            parents: (0..length).collect(),
            offsets: vec![0; length],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        let parent = self.parents[value];
        if parent != value {
            self.parents[value] = self.find(parent);
            self.offsets[value] += self.offsets[parent];
        }
        self.parents[value]
    }

    /// Require address(left) = address(right) + displacement.
    fn union_offset(
        &mut self,
        left: usize,
        right: usize,
        displacement: i64,
    ) -> Result<(), PlacementError> {
        let a = self.find(left);
        let b = self.find(right);
        let delta = self.offsets[right] + displacement - self.offsets[left];
        if a == b {
            if delta != 0 {
                return Err(PlacementError::IncompatibleAlias);
            }
        } else {
            self.parents[b] = a;
            self.offsets[b] = -delta;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ComputeGraph, Ipu21CostModel, KernelBuildPlan, Layout, PipelineConfig, Precision,
        TensorFormat, lower, lower_to_tiles, materialize_kernel_run,
    };

    #[test]
    fn displaced_alias_constraints_preserve_repeat_equalities_and_reject_cycles() {
        let mut sets = DisjointSets::new(5);
        sets.union_offset(1, 0, -32768).unwrap();
        // Repeat invariant/result IDs bind to the shifted value's address.
        sets.union_offset(2, 1, 0).unwrap();
        sets.union_offset(3, 2, 0).unwrap();
        sets.union_offset(4, 0, 0).unwrap();
        for i in 0..5 {
            sets.find(i);
        }
        assert_eq!(sets.offsets[0] - sets.offsets[1], 32768);
        assert_eq!(sets.offsets[1], sets.offsets[2]);
        assert_eq!(sets.offsets[1], sets.offsets[3]);
        assert_eq!(sets.offsets[0], sets.offsets[4]);
        assert_eq!(
            sets.union_offset(0, 3, 0),
            Err(PlacementError::IncompatibleAlias)
        );
    }

    fn request(
        class: MemoryClass,
        bytes: u32,
        alignment: u32,
        first: u32,
        last: u32,
    ) -> AllocationRequest {
        AllocationRequest {
            auxiliary: None,
            class,
            region1_stride: None,
            bytes,
            alignment,
            conflicts: Vec::new(),
            lifetime: Lifetime {
                first,
                last,
                seen: true,
            },
            assignments: Vec::new(),
        }
    }

    #[test]
    fn auxiliary_storage_joins_tensor_placement_and_survives_readback() {
        let mut graph = ComputeGraph::new();
        let parameter = graph.parameter("p", [1, 128]).unwrap();
        let output = graph.gelu(parameter).unwrap();
        graph.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(1).with_input(
            parameter,
            TensorFormat {
                precision: Precision::F16,
                layout: Layout::row_sharded(1),
            },
        );
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(&crate::expand_tiles(&mid).unwrap(), false);
        let requests = [64, 128].map(|bytes| {
            vec![AuxiliaryRequest {
                name: "samples".into(),
                bytes,
                alignment: 4,
                first: 0,
                last: u32::MAX,
            }]
        });
        let base = IPU21_INTERLEAVED_MEMORY_BASE;
        let placement = place_with_auxiliary(&low, &[(base, base + 576)], 0, &requests).unwrap();
        assert_ne!(
            placement.auxiliary_allocations[0][0].address,
            placement.auxiliary_allocations[1][0].address
        );
        for (tile, allocations) in placement.auxiliary_allocations.iter().enumerate() {
            let allocation = &allocations[0];
            assert_eq!(allocation.bytes, requests[tile][0].bytes);
            for shard in low
                .shards
                .iter()
                .filter(|shard| usize::from(shard.tile) == tile)
            {
                if let Some(&address) = placement.shard_addresses.get(&shard.id) {
                    assert!(
                        address + shard_storage_bytes(shard).unwrap() <= allocation.address
                            || allocation.address + allocation.bytes <= address
                    );
                }
            }
            assert!(
                placement.tile_auxiliary_ranges[tile]
                    .iter()
                    .all(|&(start, end)| end <= allocation.address
                        || start >= allocation.address + allocation.bytes)
            );
        }
        // A sample output must not borrow the host aperture, even when that
        // would make an otherwise undersized data arena fit.
        assert!(
            place_with_auxiliary(
                &low,
                &[HOST_SCRATCH_RANGE, (base, base + 512)],
                0,
                &requests
            )
            .is_err()
        );
    }

    #[test]
    fn pointwise_parameter_input_is_never_overwritten() {
        for case in 0..3 {
            let mut graph = ComputeGraph::new();
            let parameter = graph.parameter("p", [8, 64]).unwrap();
            let output = match case {
                0 => graph.gelu(parameter).unwrap(),
                1 => {
                    let view = graph.slice(parameter, 0, 0, 4).unwrap();
                    graph.gelu(view).unwrap()
                }
                _ => graph
                    .repeat(2, [parameter], [], [], |body, args| {
                        Ok(vec![body.gelu(args.carried[0])?])
                    })
                    .unwrap()[0],
            };
            graph.set_outputs([output]).unwrap();
            let config = PipelineConfig::new(4).with_input(
                parameter,
                TensorFormat {
                    precision: Precision::F16,
                    layout: Layout::row_sharded(4),
                },
            );
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(&crate::expand_tiles(&mid).unwrap(), false);
            let analysis = analyze_allocations(&low).unwrap();
            let roots = low
                .value_shards(low.inputs[0].value)
                .iter()
                .map(|id| analysis.root_of_member[id.index() as usize])
                .collect::<BTreeSet<_>>();
            for run in &low.kernel_runs {
                assert!(run.outputs.iter().all(|out| {
                    !roots.contains(&analysis.root_of_member[out.shard.index() as usize])
                }));
            }
        }
    }

    #[test]
    fn host_aperture_is_borrowed_only_between_host_phases() {
        let (base, end) = HOST_SCRATCH_RANGE;
        for offline in [false, true] {
            let mut arena = Arena::new(&[HOST_SCRATCH_RANGE], 0);
            arena.offline = offline;
            let standard = MemoryClass::Ipu21Standard;
            for (first, last) in [(0, 3), (0, u32::MAX), (2, u32::MAX)] {
                assert!(
                    arena
                        .allocate(&request(standard, 8, 8, first, last))
                        .is_none()
                );
            }
            assert!(
                arena
                    .allocate(&request(MemoryClass::Ipu21Interleaved, 8, 8, 1, 2))
                    .is_none()
            );
            assert_eq!(arena.allocate(&request(standard, 8, 8, 1, 2)), Some(base));
            // Filtering also applies to fragments of the aperture.
            assert!(
                arena
                    .allocate(&request(standard, 8, 8, 2, u32::MAX))
                    .is_none()
            );
            assert_eq!(
                arena.allocate(&request(standard, end - base, 8, 3, 4)),
                Some(base)
            );
            assert!(arena.unused_ranges().is_empty());
        }
        assert!(
            Arena::new(&[HOST_SCRATCH_RANGE], 0)
                .unused_ranges()
                .is_empty()
        );
    }

    #[test]
    fn package_reservations_in_region_one_exclude_both_access_classes() {
        let start = IPU21_INTERLEAVED_MEMORY_BASE + 4096;
        let available = [(start, start + 4096)];
        for class in [MemoryClass::Ipu21Standard, MemoryClass::Ipu21Interleaved] {
            let mut graph = ComputeGraph::new();
            let input = graph.host_input("x", [1, 128]).unwrap();
            graph.set_outputs([input]).unwrap();
            let mut layout = Layout::row_sharded(1);
            layout.memory_class = class;
            let config = PipelineConfig::new(1).with_input(
                input,
                TensorFormat {
                    precision: Precision::F16,
                    layout,
                },
            );
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(&crate::expand_tiles(&mid).unwrap(), false);
            let placed = place_with_ranges(&low, &available).unwrap();
            assert!(!placed.shard_addresses.is_empty());
            for &address in placed.shard_addresses.values() {
                assert!(start <= address && address + 256 <= start + 4096);
            }
            assert!(
                placed
                    .tile_auxiliary_ranges
                    .iter()
                    .flatten()
                    .all(|&(a, b)| start <= a && b <= start + 4096)
            );
            assert!(place_with_ranges(&low, &[(start, start + 4)]).is_err());
        }
    }

    #[test]
    fn offline_arena_recovers_fragmentation_and_preserves_lifetimes() {
        let base = IPU21_INTERLEAVED_MEMORY_BASE;
        let ranges = [(base, base + 128)];
        let early = request(MemoryClass::Ipu21Standard, 32, 8, 0, 1);
        let lasting = request(MemoryClass::Ipu21Standard, 32, 8, 0, 3);
        let late = request(MemoryClass::Ipu21Interleaved, 96, 8, 2, 3);
        let mut greedy = Arena::new(&ranges, 0);
        assert_eq!(greedy.allocate(&early), Some(base));
        assert_eq!(greedy.allocate(&lasting), Some(base + 32));
        assert_eq!(greedy.allocate(&late), None);

        let mut offline = Arena::new(&ranges, 0);
        offline.offline = true;
        assert_eq!(offline.allocate(&late), Some(base));
        assert_eq!(offline.allocate(&lasting), Some(base + 96));
        assert_eq!(offline.allocate(&early), Some(base));
        // Endpoints are inclusive: the early buffer is still live at event 1.
        assert_eq!(
            offline.allocate(&request(MemoryClass::Ipu21Standard, 72, 8, 1, 2)),
            None
        );
        assert!(offline.unused_ranges().is_empty());
    }

    #[test]
    fn joint_arena_reuses_region_one_across_classes() {
        let base = IPU21_INTERLEAVED_MEMORY_BASE;
        let mut arena = Arena::new(&[(base, base + 320 * 1024)], 0);
        let a = arena
            .allocate(&request(MemoryClass::Ipu21Interleaved, 256 * 1024, 8, 0, 0))
            .unwrap();
        arena
            .allocate(&request(MemoryClass::Ipu21Standard, 64 * 1024, 8, 0, 0))
            .unwrap();
        let b = arena
            .allocate(&request(MemoryClass::Ipu21Standard, 256 * 1024, 8, 1, 1))
            .unwrap();
        arena
            .allocate(&request(MemoryClass::Ipu21Interleaved, 64 * 1024, 8, 1, 1))
            .unwrap();
        assert_eq!(a, b);
        assert!(arena.unused_ranges().is_empty());
    }

    #[test]
    fn pairwise_conflicts_allow_unaligned_element_starts_and_unrelated_neighbors() {
        let base = IPU21_INTERLEAVED_MEMORY_BASE;
        for offline in [false, true] {
            let mut arena = Arena::new(
                &[(base + 120, base + 3 * IPU21_INTERLEAVED_ELEMENT_SIZE - 40)],
                0,
            );
            arena.offline = offline;
            let mut a = request(MemoryClass::Ipu21Standard, 6912, 8, 0, u32::MAX);
            a.assignments = vec![(0, 0)];
            a.conflicts = vec![2];
            let mut b = request(MemoryClass::Ipu21Interleaved, 6912, 8, 0, u32::MAX);
            b.assignments = vec![(1, 0)];
            b.conflicts = vec![2];
            let mut c = request(MemoryClass::Ipu21Standard, 32768 + 8, 8, 0, u32::MAX);
            c.assignments = vec![(2, 0)];
            c.conflicts = vec![0, 1];
            assert_eq!(arena.allocate(&a), Some(base + 120));
            assert_eq!(arena.allocate(&b), Some(base + 120 + 6912));
            assert_eq!(
                arena.allocate(&c),
                Some(base + IPU21_INTERLEAVED_ELEMENT_SIZE)
            );
        }
    }

    #[test]
    fn joint_arena_respects_region_and_element_constraints() {
        let base = IPU21_INTERLEAVED_MEMORY_BASE;
        let mut arena = Arena::new(&[(base - 64, base + 2 * IPU21_INTERLEAVED_ELEMENT_SIZE)], 0);
        assert_eq!(
            arena.allocate(&request(MemoryClass::Ipu21Standard, 64, 8, 0, 2)),
            Some(base - 64)
        );
        let mut constrained = request(MemoryClass::Ipu21Standard, 8, 8, 0, 2);
        constrained.assignments = vec![(0, 0)];
        assert_eq!(arena.allocate(&constrained), Some(base));
        constrained.class = MemoryClass::Ipu21Interleaved;
        constrained.conflicts = vec![0];
        constrained.assignments = vec![(1, 0)];
        assert_eq!(
            arena.allocate(&constrained),
            Some(base + IPU21_INTERLEAVED_ELEMENT_SIZE)
        );
        // Unrelated live data can fill the tail of either constrained operand.
        assert_eq!(
            arena.allocate(&request(MemoryClass::Ipu21Standard, 8, 8, 2, 2)),
            Some(base + 8)
        );
        constrained.conflicts.push(1);
        assert!(arena.allocate(&constrained).is_none());
        assert!(
            arena
                .allocate(&request(MemoryClass::Ipu21Interleaved, 64, 8, 3, 3))
                .is_some()
        );
    }

    #[test]
    fn partial_elements_accept_payload_without_truncating_it() {
        let limit = IPU21_APPLICATION_MEMORY_LIMIT;
        let base = limit / IPU21_INTERLEAVED_ELEMENT_SIZE * IPU21_INTERLEAVED_ELEMENT_SIZE;
        let mut arena = Arena::new(&[(base, limit)], 0);
        let mut payload = request(MemoryClass::Ipu21Interleaved, limit - base, 8, 0, 1);

        assert_eq!(arena.allocate(&payload), Some(base));
        assert!(
            arena
                .allocate(&request(MemoryClass::Ipu21Standard, 4, 4, 0, 1))
                .is_none()
        );
        payload.lifetime.first = 2;
        payload.lifetime.last = 3;
        assert_eq!(arena.allocate(&payload), Some(base));
        payload.bytes += 1;
        payload.lifetime.first = 4;
        payload.lifetime.last = 5;
        assert!(arena.allocate(&payload).is_none());

        // A payload also fits before an unrelated live allocation in the same
        // element. Unrelated constrained buffers may also share an element.
        let base = IPU21_INTERLEAVED_MEMORY_BASE;
        let mut ordinary_gap = Arena::new(&[(base, base + 1024)], 0);
        payload.bytes = 512;
        assert_eq!(ordinary_gap.allocate(&payload), Some(base));
        assert_eq!(ordinary_gap.allocate(&payload), Some(base + 512));
    }

    #[test]
    fn randomized_joint_allocations_do_not_overlap_live_storage() {
        let mut random = fastrand::Rng::with_seed(0x6a6f_696e_745f_7372);
        let boundary = IPU21_INTERLEAVED_MEMORY_BASE;
        let ranges = [
            (boundary - 65536, boundary - 32768),
            (boundary - 16384, boundary),
            (boundary, IPU21_APPLICATION_MEMORY_LIMIT),
        ];
        for trial in 0..128 {
            let mut arena = Arena::new(&ranges, 256);
            arena.offline = trial % 2 != 0;
            let mut placed = Vec::<(u32, u32, u32, u32, bool)>::new();
            for step in 0..32 {
                let first = if arena.offline {
                    random.u32(0..32)
                } else {
                    step
                };
                for _ in 0..random.u32(1..=6) {
                    let class = if random.bool() {
                        MemoryClass::Ipu21Standard
                    } else {
                        MemoryClass::Ipu21Interleaved
                    };
                    let mut request = request(
                        class,
                        random.u32(1..=65536),
                        1 << random.u32(2..=12),
                        first,
                        first + random.u32(0..=8),
                    );
                    let constrained = random.bool();
                    request.assignments = vec![(placed.len(), 0)];
                    if constrained {
                        request.conflicts = placed
                            .iter()
                            .enumerate()
                            .filter(|(_, (_, _, a, b, c))| {
                                *c && first <= *b && *a <= request.lifetime.last
                            })
                            .map(|(id, _)| id)
                            .collect();
                    }
                    let Some(address) = arena.allocate(&request) else {
                        continue;
                    };
                    let (_, _, bytes) = *arena.active.last().unwrap();
                    let end = address + bytes;
                    assert!(address.is_multiple_of(request.alignment));
                    if class == MemoryClass::Ipu21Interleaved {
                        assert!(address >= boundary + 256);
                    }
                    assert!(
                        ranges
                            .iter()
                            .any(|&(base, limit)| base <= address && end <= limit)
                    );
                    for &(other, other_end, other_first, other_last, other_constrained) in &placed {
                        if first <= other_last && other_first <= request.lifetime.last {
                            assert!(end <= other || other_end <= address);
                            if constrained
                                && other_constrained
                                && (address >= boundary) == (other >= boundary)
                            {
                                let element = if address >= boundary {
                                    IPU21_INTERLEAVED_ELEMENT_SIZE
                                } else {
                                    TILE_MEMORY_ELEMENT_SIZE
                                };
                                assert!(
                                    end.div_ceil(element) <= other / element
                                        || other_end.div_ceil(element) <= address / element
                                );
                            }
                        }
                    }
                    placed.push((address, end, first, request.lifetime.last, constrained));
                }
            }
            for (base, end) in arena.unused_ranges() {
                assert!(
                    placed
                        .iter()
                        .all(|&(other, other_end, _, _, _)| end <= other || other_end <= base)
                );
            }
        }
    }

    #[test]
    fn repeat_bank_separation_reserves_the_sequence_without_padding_each_member() {
        let mut graph = ComputeGraph::new();
        let carried = graph.host_input("carried", [8, 16]).unwrap();
        let parameters = (0..3)
            .map(|index| {
                graph
                    .parameter(format!("parameter.{index}"), [8, 16])
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let sequence = graph
            .value_sequence("parameters", parameters.clone())
            .unwrap();
        let output = graph
            .repeat(3, [carried], [], [sequence], |body, arguments| {
                Ok(vec![body.add(arguments.carried[0], arguments.iterated[0])?])
            })
            .unwrap()[0];
        graph.set_outputs([output]).unwrap();
        let format = TensorFormat {
            precision: Precision::F16,
            layout: Layout::row_sharded(4),
        };
        let mut config = PipelineConfig::new(4).with_input(carried, format.clone());
        for parameter in parameters {
            config.inputs.insert(parameter, format.clone());
        }
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let mut expanded = (*crate::expand_tiles(&mid).unwrap()).clone();
        // Bank constraints may be introduced during expansion, after mid has
        // selected the iterated values' shapes.
        for run in &mut expanded.kernel_runs {
            if run.inputs.len() == 2 {
                std::sync::Arc::make_mut(&mut run.metadata)
                    .requirements
                    .distinct_elements
                    .push(vec![
                        crate::MemoryOperand::Output(0),
                        crate::MemoryOperand::Input(1),
                    ]);
            }
        }
        let low = lower_to_tiles(&std::sync::Arc::new(expanded), false);
        let mut checked = 0;
        for placement in [
            place(&low).unwrap(),
            place_with_ranges(
                &low,
                &[
                    (IPU21_DATA_BASE, IPU21_DATA_BASE + 4),
                    (
                        IPU21_INTERLEAVED_MEMORY_BASE,
                        IPU21_APPLICATION_MEMORY_LIMIT,
                    ),
                ],
            )
            .unwrap(),
            place_with_ranges(
                &low,
                &[(
                    IPU21_INTERLEAVED_MEMORY_BASE,
                    IPU21_APPLICATION_MEMORY_LIMIT,
                )],
            )
            .unwrap(),
        ] {
            for tile in &low.tiles {
                for work in low.work(tile) {
                    let crate::TileWorkRef::Repeat(repeat) = work else {
                        continue;
                    };
                    for input in &repeat.binding.iterated {
                        assert!(
                            placement.sequence_strides[&input.argument] < TILE_MEMORY_ELEMENT_SIZE
                        );
                        let output = placement.shard_addresses[&repeat.binding.carried[0].initial];
                        let element = |address| {
                            if address >= IPU21_INTERLEAVED_MEMORY_BASE {
                                IPU21_INTERLEAVED_ELEMENT_SIZE
                            } else {
                                TILE_MEMORY_ELEMENT_SIZE
                            }
                        };
                        for member in &input.inputs {
                            let address = placement.shard_addresses[member];
                            let bytes =
                                shard_storage_bytes(&low.shards[member.index() as usize]).unwrap();
                            assert!(
                                (address + bytes).div_ceil(element(address)) * element(address)
                                    <= output
                                    || output.div_euclid(element(output)) * element(output)
                                        + element(output)
                                        <= address,
                                "a sequence member shares the output's element"
                            );
                        }
                        for pair in input.inputs.windows(2) {
                            assert_eq!(
                                placement.shard_addresses[&pair[1]]
                                    - placement.shard_addresses[&pair[0]],
                                placement.sequence_strides[&input.argument]
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 0);

        // Independently constrained members still require separate elements.
        let mut analysis = analyze_allocations(&low).unwrap();
        let group = analysis.tiles[0].iterated[0].clone();
        let second_root = analysis.root_of_member[group.shards[1].index() as usize];
        let first_root = analysis.root_of_member[group.shards[0].index() as usize];
        analysis
            .conflicts
            .entry(first_root)
            .or_default()
            .insert(second_root);
        analysis
            .conflicts
            .entry(second_root)
            .or_default()
            .insert(first_root);
        let placed = place_tile(
            &low,
            0,
            &[(IPU21_DATA_BASE, IPU21_APPLICATION_MEMORY_LIMIT)],
            0,
            &analysis,
            &[],
        )
        .unwrap();
        assert_eq!(
            placed.addresses[&group.shards[1]] - placed.addresses[&group.shards[0]],
            TILE_MEMORY_ELEMENT_SIZE
        );
    }

    #[test]
    fn output_lifetimes_follow_ownership_not_output_list_order() {
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [8, 16]).unwrap();
        let output = graph.gelu(input).unwrap();
        graph.set_outputs([output]).unwrap();
        let format = TensorFormat {
            precision: Precision::F16,
            layout: Layout::row_sharded(4),
        };
        let mut config = PipelineConfig::new(4).with_input(input, format.clone());
        config.operator_candidates = vec![crate::ConcreteOperatorCandidate::new(
            crate::MidOperator::Gelu,
            [crate::OperandRequirement::new(format.clone())],
            crate::OperandRequirement::new(format),
        )]
        .into_iter()
        .map(crate::OperatorCandidate::Concrete)
        .collect();
        let candidate = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let mut program = (*crate::expand_tiles(&candidate).unwrap()).clone();
        let work = program
            .body
            .operations
            .iter()
            .find(|work| matches!(work, crate::BlockOperation::Compute { tile: 0, .. }))
            .unwrap()
            .clone();
        program.body.operations.extend([work.clone(), work]);
        program.value_shards[program.outputs[0].index() as usize].reverse();
        let low = lower_to_tiles(&std::sync::Arc::new(program), false);
        let lifetimes = collect_lifetimes(&low);
        for &id in low.value_shards(low.outputs[0]) {
            let tile = low.shards[id.index() as usize].tile;
            assert_eq!(
                lifetimes[id.index() as usize].last,
                u32::MAX,
                "output on tile {tile}"
            );
        }
    }

    #[test]
    fn randomized_gemm_placement_respects_classes_and_kernel_views() {
        let mut random = fastrand::Rng::with_seed(0x706c_6163);
        for _ in 0..48 {
            let tiles = 1_u16 << random.u32(0..=3);
            let rows = u32::from(tiles) * random.u32(1..=8);
            let columns = random.u32(1..=2) * 64;
            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", [rows, 64]).unwrap();
            let right = graph.parameter("right", [64, columns]).unwrap();
            let output = graph.gemm(left, right).unwrap();
            graph.set_outputs([output]).unwrap();
            let config = PipelineConfig::new(tiles)
                .with_input(
                    left,
                    TensorFormat {
                        precision: Precision::F16,
                        layout: Layout::amp_left(64, tiles),
                    },
                )
                .with_input(
                    right,
                    TensorFormat {
                        precision: Precision::F16,
                        layout: Layout::block_major_matrix(64, tiles),
                    },
                );
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(
                &crate::expand_tiles(&mid).unwrap(),
                config.diagnostic_checkpoints,
            );
            let placement = place_with_offset(
                &low,
                &[(IPU21_DATA_BASE, IPU21_APPLICATION_MEMORY_LIMIT)],
                random.u32(0..8) * 4096,
            )
            .unwrap();
            let kernels = KernelBuildPlan::from_program(&low).unwrap();
            let resident = low
                .shards
                .iter()
                .filter(|shard| !matches!(shard.definition, crate::ShardDefinition::Unmaterialized))
                .collect::<Vec<_>>();
            assert_eq!(placement.shard_addresses.len(), resident.len());
            for shard in resident {
                let address = placement.shard_addresses[&shard.id];
                match shard.tensor_type.format.layout.memory_class {
                    MemoryClass::Ipu21Interleaved => {
                        assert!(
                            (IPU21_INTERLEAVED_MEMORY_BASE..IPU21_APPLICATION_MEMORY_LIMIT)
                                .contains(&address)
                        )
                    }
                    MemoryClass::Ipu21Standard => assert!(address >= IPU21_DATA_BASE),
                }
            }
            for tile in &low.tiles {
                for work in low.work(tile) {
                    if let TileWorkRef::Kernel(run) = work {
                        materialize_kernel_run(
                            run,
                            &low.shards,
                            &placement.shard_addresses,
                            &kernels,
                            &BTreeMap::new(),
                        )
                        .unwrap();
                        {
                            let requirements = &run.requirements;
                            for operands in &requirements.distinct_elements {
                                let mut ranges = Vec::new();
                                for operand in operands {
                                    let shards = match operand {
                                        crate::MemoryOperand::Output(index) => {
                                            vec![run.outputs[usize::from(*index)].shard]
                                        }
                                        crate::MemoryOperand::Input(index) => {
                                            vec![run.inputs[usize::from(*index)].shard]
                                        }
                                    };
                                    for shard in shards {
                                        let definition = &low.shards[shard.index() as usize];
                                        let address = placement.shard_addresses[&shard];
                                        let bytes = shard_storage_bytes(definition).unwrap();
                                        ranges.push(crate::exchange::effective_memory_elements(
                                            address,
                                            bytes.div_ceil(4),
                                        ));
                                    }
                                }
                                for (index, left) in ranges.iter().enumerate() {
                                    for right in &ranges[..index] {
                                        assert!(
                                            left.iter().all(|element| !right.contains(element))
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn randomized_sequential_pointwise_values_reuse_dead_input_storage() {
        let mut random = fastrand::Rng::with_seed(0x6c69_7665);
        for _ in 0..48 {
            let tiles = 1_u16 << random.u32(0..=3);
            let rows = u32::from(tiles) * random.u32(1..=8);
            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", [rows, 64]).unwrap();
            let right = graph.host_input("right", [rows, 64]).unwrap();
            let sum = graph.add(left, right).unwrap();
            let output = graph.gelu(sum).unwrap();
            graph.set_outputs([output]).unwrap();
            let format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::row_sharded(tiles),
            };
            let mut config = PipelineConfig::new(tiles)
                .with_input(left, format.clone())
                .with_input(right, format);
            // This is an in-place allocation fixture, independent of whether
            // the cost model prefers another precision or inserts conversions.
            config.operator_candidates.retain(|candidate| {
                candidate.concrete().is_some_and(|candidate| {
                    let format = &candidate.plan.requirements.output.format;
                    format.precision == Precision::F16
                        && format.layout.order == crate::ElementOrder::RowMajor
                })
            });
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            assert_eq!(mid.operations.len(), 2);
            let sum = mid.operations[0].results[0];
            let output = mid.operations[1].results[0];
            let low = lower_to_tiles(
                &crate::expand_tiles(&mid).unwrap(),
                config.diagnostic_checkpoints,
            );
            let placement = place(&low).unwrap();
            for tile in 0..tiles {
                let shard = |value| {
                    low.value_shards(value)
                        .iter()
                        .copied()
                        .find(|id| low.shards[id.index() as usize].tile == tile)
                        .unwrap()
                };
                let sum_address = placement.shard_addresses[&shard(sum)];
                let output_address = placement.shard_addresses[&shard(output)];
                assert_eq!(output_address, sum_address);
            }
        }
    }

    #[test]
    fn randomized_compact_arenas_keep_sequential_phases_below_their_peak_span() {
        let mut random = fastrand::Rng::with_seed(0x636f_6d70_6163_7421);
        for _ in 0..128 {
            let limit = 1 << 20;
            let persistent = random.u32(1..=4096);
            let mut arena = Arena::new(&[(0, limit.min(IPU21_INTERLEAVED_MEMORY_BASE))], 0);
            arena
                .allocate(&request(
                    MemoryClass::Ipu21Standard,
                    persistent,
                    4,
                    0,
                    u32::MAX,
                ))
                .unwrap();
            let mut bound = persistent;
            for phase in 1..=random.u32(2..=16) {
                let mut cursor = persistent;
                for _ in 0..random.u32(1..=8) {
                    let alignment = 1 << random.u32(2..=10);
                    let bytes = random.u32(1..=16 * 1024);
                    cursor = align_up(cursor, alignment).unwrap() + bytes;
                    arena
                        .allocate(&request(
                            MemoryClass::Ipu21Standard,
                            bytes,
                            alignment,
                            phase,
                            phase,
                        ))
                        .unwrap();
                }
                bound = bound.max(cursor);
            }
            assert!(arena.occupied.iter().map(|range| range.1).max().unwrap() <= bound);
        }
    }
}
