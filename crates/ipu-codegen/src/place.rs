//! Deterministic placement of logical shards in IPU21 tile SRAM.

use crate::layout::MemoryClass;
use crate::low::{
    KernelRequirements, LowProgram, LowShardId, ShardDefinition, TileWorkList, TileWorkRef,
};
use crate::memory::IPU21_DATA_BASE;
use crate::mid::{
    AllocationRequirements, MemoryElementRequirement, MemoryOperand, OperandRequirement,
};
use crate::storage::{StorageError, shard_storage_bytes};
use ipu_package::{
    AddressRegion, IPU21_APPLICATION_MEMORY_LIMIT, IPU21_INTERLEAVED_ELEMENT_SIZE,
    IPU21_INTERLEAVED_MEMORY_BASE, TILE_MEMORY_ELEMENT_SIZE,
};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub shard_addresses: BTreeMap<LowShardId, u32>,
    pub tile_auxiliary_ranges: Vec<Vec<AddressRegion>>,
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
    #[error("repeat iterated block is smaller than its required allocation")]
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

pub fn place(program: &LowProgram) -> Result<Placement, PlacementError> {
    place_with_standard_ranges(
        program,
        &[AddressRegion::new(
            IPU21_DATA_BASE,
            IPU21_INTERLEAVED_MEMORY_BASE,
        )],
    )
}

pub(crate) fn place_with_standard_ranges(
    program: &LowProgram,
    standard_ranges: &[AddressRegion],
) -> Result<Placement, PlacementError> {
    if standard_ranges.is_empty()
        || standard_ranges.iter().any(|range| {
            range.start < IPU21_DATA_BASE
                || range.end > IPU21_INTERLEAVED_MEMORY_BASE
                || range.is_empty()
        })
        || standard_ranges
            .windows(2)
            .any(|pair| pair[0].end > pair[1].start)
    {
        return Err(PlacementError::OutOfMemory {
            tile: 0,
            class: MemoryClass::Standard,
            bytes: 0,
        });
    }
    let mut sets = DisjointSets::new(program.shards.len());
    for shard in &program.shards {
        if let ShardDefinition::Alias(target) | ShardDefinition::WritableAlias(target) =
            shard.definition
        {
            checked_union(program, &mut sets, shard.id, target)?;
        }
    }
    let mut iterated = Vec::<IteratedGroup>::new();
    for tile in &program.tiles {
        collect_repeat_constraints(program, tile, &mut sets, &mut iterated)?;
    }

    let mut requirements = vec![AllocationRequirements::default(); program.shards.len()];
    for tile in &program.tiles {
        collect_requirements(program, tile, &mut requirements);
    }
    let mut root_requirements = BTreeMap::<usize, AllocationRequirements>::new();
    for (index, requirement) in requirements.into_iter().enumerate() {
        let root = sets.find(index);
        let combined = root_requirements.entry(root).or_default();
        combined.merge(requirement);
    }

    let mut members = BTreeMap::<usize, Vec<usize>>::new();
    let mut root_of_member = vec![0usize; program.shards.len()];
    for (index, root_slot) in root_of_member.iter_mut().enumerate() {
        let root = sets.find(index);
        *root_slot = root;
        members.entry(root).or_default().push(index);
    }
    validate_alias_groups(program, &members)?;
    let lifetimes = collect_lifetimes(program);
    let mut root_lifetimes = BTreeMap::<usize, Lifetime>::new();
    for (index, lifetime) in lifetimes.into_iter().enumerate() {
        root_lifetimes
            .entry(root_of_member[index])
            .or_default()
            .include(lifetime);
    }

    let tile_placements = (0..usize::from(program.tile_count))
        .into_par_iter()
        .map(|tile| {
            place_tile(
                program,
                u16::try_from(tile).map_err(|_| PlacementError::Overflow)?,
                standard_ranges,
                &iterated,
                &members,
                &root_of_member,
                &root_requirements,
                &root_lifetimes,
            )
        })
        .collect::<Result<Vec<_>, PlacementError>>()?;
    let mut addresses = BTreeMap::new();
    let mut tile_auxiliary_ranges = vec![Vec::new(); usize::from(program.tile_count)];
    for (tile, tile_addresses, unused) in tile_placements {
        addresses.extend(tile_addresses);
        tile_auxiliary_ranges[usize::from(tile)] = unused;
    }

    Ok(Placement {
        shard_addresses: addresses,
        tile_auxiliary_ranges,
    })
}

#[allow(clippy::too_many_arguments)]
fn place_tile(
    program: &LowProgram,
    tile: u16,
    standard_ranges: &[AddressRegion],
    iterated: &[IteratedGroup],
    members: &BTreeMap<usize, Vec<usize>>,
    root_of_member: &[usize],
    root_requirements: &BTreeMap<usize, AllocationRequirements>,
    root_lifetimes: &BTreeMap<usize, Lifetime>,
) -> Result<(u16, BTreeMap<LowShardId, u32>, Vec<AddressRegion>), PlacementError> {
    let mut grouped = BTreeSet::<usize>::new();
    for group in iterated.iter().filter(|group| group.tile == tile) {
        let roots = group
            .shards
            .iter()
            .map(|shard| root_of_member[shard.index() as usize])
            .collect::<Vec<_>>();
        if roots.iter().any(|root| !grouped.insert(*root)) {
            return Err(PlacementError::IteratedOverlap);
        }
    }

    // Region 1 is shared by ordinary and interleaved loads. Place the
    // interleaved working set first, round its boundary to a paired memory
    // element, then return every remaining byte to standard allocations.
    let mut addresses = BTreeMap::new();
    let mut interleaved = Arena::new(
        &[(
            IPU21_INTERLEAVED_MEMORY_BASE,
            IPU21_APPLICATION_MEMORY_LIMIT,
        )],
        true,
    );
    allocate_tile_class(
        program,
        tile,
        MemoryClass::Interleaved,
        iterated,
        &grouped,
        members,
        root_of_member,
        root_requirements,
        root_lifetimes,
        &mut interleaved,
        &mut addresses,
    )?;
    let interleaved_boundary =
        align_up(interleaved.maximum_cursor(), IPU21_INTERLEAVED_ELEMENT_SIZE)?;
    if interleaved_boundary > IPU21_APPLICATION_MEMORY_LIMIT {
        return Err(PlacementError::OutOfMemory {
            tile,
            class: MemoryClass::Interleaved,
            bytes: interleaved_boundary - IPU21_INTERLEAVED_MEMORY_BASE,
        });
    }
    let mut ranges = standard_ranges
        .iter()
        .map(|range| (range.start, range.end))
        .collect::<Vec<_>>();
    ranges.push((interleaved_boundary, IPU21_APPLICATION_MEMORY_LIMIT));
    let mut standard = Arena::new(&ranges, false);
    allocate_tile_class(
        program,
        tile,
        MemoryClass::Standard,
        iterated,
        &grouped,
        members,
        root_of_member,
        root_requirements,
        root_lifetimes,
        &mut standard,
        &mut addresses,
    )?;
    Ok((
        tile,
        addresses,
        standard
            .unused_ranges()
            .into_iter()
            .map(|(start, end)| AddressRegion::new(start, end))
            .collect(),
    ))
}

fn collect_lifetimes(program: &LowProgram) -> Vec<Lifetime> {
    let mut lifetimes = vec![Lifetime::default(); program.shards.len()];
    for input in &program.inputs {
        for shard in &input.shards {
            lifetimes[shard.index() as usize].touch(0);
        }
    }
    for tile in &program.tiles {
        let mut event = 1u32;
        for work in program.work(tile) {
            touch_work(program, work, tile.tile, &mut event, &mut lifetimes);
        }
        for output in &program.outputs {
            if let Some(shard) = output.shards.get(usize::from(tile.tile)) {
                lifetimes[shard.index() as usize].touch(event);
            }
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
) {
    let current = *event;
    let mut touch = |shard: LowShardId| lifetimes[shard.index() as usize].touch(current);
    match work {
        TileWorkRef::Kernel(run) => {
            for view in run.inputs.iter().flat_map(|operand| &operand.views) {
                touch(view.shard);
            }
            touch(run.output.shard);
        }
        TileWorkRef::LocalCopy(copy) => {
            touch(copy.source);
            touch(copy.destination);
        }
        TileWorkRef::Exchange(id) => {
            for transfer in &program.exchange_phases[id.index() as usize].transfers {
                if program.shards[transfer.source.shard.index() as usize].tile == tile {
                    touch(transfer.source.shard);
                }
                for destination in &transfer.destinations {
                    if program.shards[destination.shard.index() as usize].tile == tile {
                        touch(destination.shard);
                    }
                }
            }
        }
        TileWorkRef::Repeat(repeat) => {
            for carried in &repeat.carried {
                touch(carried.initial);
                touch(carried.argument);
                touch(carried.yielded);
                touch(carried.result);
            }
            for invariant in &repeat.invariants {
                touch(invariant.input);
                touch(invariant.argument);
            }
            for iterated in &repeat.iterated {
                for input in &iterated.inputs {
                    touch(*input);
                }
                touch(iterated.argument);
            }
            *event = event.saturating_add(1);
            for nested in program.work(&repeat.body) {
                touch_work(program, nested, tile, event, lifetimes);
            }
            let end = *event;
            for carried in &repeat.carried {
                lifetimes[carried.initial.index() as usize].touch(end);
                lifetimes[carried.argument.index() as usize].touch(end);
                lifetimes[carried.yielded.index() as usize].touch(end);
                lifetimes[carried.result.index() as usize].touch(end);
            }
            for invariant in &repeat.invariants {
                lifetimes[invariant.input.index() as usize].touch(end);
                lifetimes[invariant.argument.index() as usize].touch(end);
            }
            for iterated in &repeat.iterated {
                for input in &iterated.inputs {
                    lifetimes[input.index() as usize].touch(end);
                }
                lifetimes[iterated.argument.index() as usize].touch(end);
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
        for carried in &repeat.carried {
            checked_union(program, sets, carried.initial, carried.argument)?;
            checked_union(program, sets, carried.initial, carried.yielded)?;
            checked_union(program, sets, carried.initial, carried.result)?;
        }
        for invariant in &repeat.invariants {
            checked_union(program, sets, invariant.input, invariant.argument)?;
        }
        for input in &repeat.iterated {
            let first = *input.inputs.first().ok_or(PlacementError::IteratedStride)?;
            checked_union(program, sets, first, input.argument)?;
            iterated.push(IteratedGroup {
                tile: tile.tile,
                shards: input.inputs.clone(),
                stride: input.stride_bytes,
                alignment: input.alignment,
            });
        }
        collect_repeat_constraints(program, &repeat.body, sets, iterated)?;
    }
    Ok(())
}

fn collect_requirements(
    program: &LowProgram,
    tile: &TileWorkList,
    requirements: &mut [AllocationRequirements],
) {
    for work in program.work(tile) {
        match work {
            TileWorkRef::Kernel(run) => {
                let (inputs, output, memory_space) = match &run.requirements {
                    KernelRequirements::Operator(operator_requirements) => (
                        &operator_requirements.inputs[..],
                        &operator_requirements.output,
                        &operator_requirements.memory_space,
                    ),
                    KernelRequirements::Conversion {
                        input,
                        output,
                        memory_space,
                    } => (std::slice::from_ref(input), output, memory_space),
                };
                for operands in &memory_space.distinct_element_groups {
                    for operand in operands {
                        match operand {
                            MemoryOperand::Output => {
                                requirements[run.output.shard.index() as usize]
                                    .require_distinct_element();
                            }
                            MemoryOperand::Input(index) => {
                                if let Some(input) = run.inputs.get(usize::from(*index)) {
                                    for view in &input.views {
                                        requirements[view.shard.index() as usize]
                                            .require_distinct_element();
                                    }
                                }
                            }
                        }
                    }
                }
                for (operand, requirement) in run.inputs.iter().zip(inputs) {
                    for view in &operand.views {
                        apply_requirement(
                            &mut requirements[view.shard.index() as usize],
                            requirement,
                        );
                    }
                }
                apply_requirement(&mut requirements[run.output.shard.index() as usize], output);
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
                collect_requirements(program, &repeat.body, requirements)
            }
            TileWorkRef::Exchange(_) | TileWorkRef::Checkpoint(..) => {}
        }
    }
}

fn apply_requirement(target: &mut AllocationRequirements, requirement: &OperandRequirement) {
    target.merge(requirement.allocation);
}

fn checked_union(
    program: &LowProgram,
    sets: &mut DisjointSets,
    left: LowShardId,
    right: LowShardId,
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
    sets.union(left.index() as usize, right.index() as usize);
    Ok(())
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
    requirement: AllocationRequirements,
) -> Result<u32, PlacementError> {
    let bytes = members
        .iter()
        .map(|&index| {
            shard_storage_bytes(&program.shards[index])?
                .checked_add(requirement.access_tail_bytes)
                .ok_or(PlacementError::Overflow)
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or(PlacementError::Overflow)?;
    if requirement.memory_element == MemoryElementRequirement::Distinct {
        align_up(bytes, memory_element_size(program, members))
    } else {
        Ok(bytes)
    }
}

fn memory_element_size(program: &LowProgram, members: &[usize]) -> u32 {
    match program.shards[members[0]]
        .tensor_type
        .format
        .layout
        .memory_class
    {
        MemoryClass::Standard => TILE_MEMORY_ELEMENT_SIZE,
        MemoryClass::Interleaved => IPU21_INTERLEAVED_ELEMENT_SIZE,
    }
}

fn allocation_alignment(
    program: &LowProgram,
    members: &[usize],
    requirement: AllocationRequirements,
) -> u32 {
    if requirement.memory_element == MemoryElementRequirement::Distinct {
        requirement
            .alignment
            .max(memory_element_size(program, members))
    } else {
        requirement.alignment
    }
}

fn assign_members(
    addresses: &mut BTreeMap<LowShardId, u32>,
    members: &[usize],
    address: u32,
) -> Result<(), PlacementError> {
    for &member in members {
        addresses.insert(
            LowShardId::from_index(u32::try_from(member).map_err(|_| PlacementError::Overflow)?),
            address,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn allocate_tile_class(
    program: &LowProgram,
    tile: u16,
    class: MemoryClass,
    iterated: &[IteratedGroup],
    grouped: &BTreeSet<usize>,
    members: &BTreeMap<usize, Vec<usize>>,
    root_of_member: &[usize],
    root_requirements: &BTreeMap<usize, AllocationRequirements>,
    root_lifetimes: &BTreeMap<usize, Lifetime>,
    arena: &mut Arena,
    addresses: &mut BTreeMap<LowShardId, u32>,
) -> Result<(), PlacementError> {
    let mut requests = Vec::<AllocationRequest>::new();
    for group in iterated.iter().filter(|group| group.tile == tile) {
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
        if group_class != class {
            continue;
        }
        let alignment = group.alignment.max(
            roots
                .iter()
                .map(|root| allocation_alignment(program, &members[root], root_requirements[root]))
                .max()
                .unwrap_or(1),
        );
        for root in &roots {
            let required = allocation_bytes(program, &members[root], root_requirements[root])?;
            if required > group.stride {
                return Err(PlacementError::IteratedStride);
            }
        }
        let bytes = group
            .stride
            .checked_mul(u32::try_from(roots.len()).map_err(|_| PlacementError::Overflow)?)
            .ok_or(PlacementError::Overflow)?;
        let mut lifetime = Lifetime::default();
        let mut assignments = Vec::with_capacity(roots.len());
        for (index, root) in roots.into_iter().enumerate() {
            lifetime.include(root_lifetimes[&root]);
            assignments.push((
                root,
                group
                    .stride
                    .checked_mul(u32::try_from(index).map_err(|_| PlacementError::Overflow)?)
                    .ok_or(PlacementError::Overflow)?,
            ));
        }
        requests.push(AllocationRequest {
            lifetime,
            bytes,
            alignment,
            assignments,
        });
    }
    for (&root, root_members) in members {
        let representative = &program.shards[root_members[0]];
        let lifetime = root_lifetimes.get(&root).copied().unwrap_or_default();
        if representative.tile != tile
            || representative.tensor_type.format.layout.memory_class != class
            || grouped.contains(&root)
            || !lifetime.seen
        {
            continue;
        }
        let requirement = root_requirements.get(&root).copied().unwrap_or_default();
        let bytes = allocation_bytes(program, root_members, requirement)?;
        requests.push(AllocationRequest {
            lifetime,
            bytes,
            alignment: allocation_alignment(program, root_members, requirement).max(4),
            assignments: vec![(root, 0)],
        });
    }
    requests.sort_by_key(|request| (request.lifetime.first, request.lifetime.last));
    for request in requests {
        let Some(base) = arena.allocate(
            request.bytes,
            request.alignment,
            request.lifetime.first,
            request.lifetime.last,
        ) else {
            let representative = &program.shards[members[&request.assignments[0].0][0]];
            tracing::error!(
                tile,
                ?class,
                bytes = request.bytes,
                alignment = request.alignment,
                first = request.lifetime.first,
                last = request.lifetime.last,
                free = ?arena.free,
                active = ?arena.active,
                tensor_type = ?representative.tensor_type,
                "tile allocation does not fit"
            );
            return Err(PlacementError::OutOfMemory {
                tile,
                class,
                bytes: request.bytes,
            });
        };
        for (root, offset) in request.assignments {
            let address = base.checked_add(offset).ok_or(PlacementError::Overflow)?;
            assign_members(addresses, &members[&root], address)?;
        }
    }
    Ok(())
}

struct AllocationRequest {
    lifetime: Lifetime,
    bytes: u32,
    alignment: u32,
    assignments: Vec<(usize, u32)>,
}

#[derive(Clone, Debug)]
struct IteratedGroup {
    tile: u16,
    shards: Vec<LowShardId>,
    stride: u32,
    alignment: u32,
}

struct Arena {
    ranges: Vec<(u32, u32)>,
    free: Vec<(u32, u32)>,
    active: Vec<(u32, u32, u32)>,
    occupied: Vec<(u32, u32)>,
    maximum: u32,
    compact_low: bool,
}

impl Arena {
    fn new(ranges: &[(u32, u32)], compact_low: bool) -> Self {
        Self {
            ranges: ranges.to_vec(),
            free: ranges.to_vec(),
            active: Vec::new(),
            occupied: Vec::new(),
            maximum: ranges[0].0,
            compact_low,
        }
    }

    fn allocate(&mut self, bytes: u32, alignment: u32, first: u32, last: u32) -> Option<u32> {
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
        let candidate = self
            .free
            .iter()
            .enumerate()
            .filter_map(|(index, &(base, limit))| {
                let start = align_up(base, alignment).ok()?;
                let end = start.checked_add(bytes)?;
                (end <= limit).then(|| {
                    let key = if self.compact_low {
                        (start, limit - end)
                    } else {
                        (limit - end, start)
                    };
                    (key, index, start, end)
                })
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
            self.active.push((last, start, bytes));
            self.occupied.push((start, end));
            self.maximum = self.maximum.max(end);
            return Some(start);
        }
        None
    }

    fn release(&mut self, base: u32, limit: u32) {
        self.free.push((base, limit));
        self.free.sort_unstable();
        let mut merged = Vec::<(u32, u32)>::with_capacity(self.free.len());
        for range in self.free.drain(..) {
            match merged.last_mut() {
                Some(previous) if previous.1 == range.0 => previous.1 = range.1,
                _ => merged.push(range),
            }
        }
        self.free = merged;
    }

    fn maximum_cursor(&self) -> u32 {
        self.maximum
    }

    fn unused_ranges(&self) -> Vec<(u32, u32)> {
        let mut occupied = self.occupied.clone();
        occupied.sort_unstable();
        let mut merged = Vec::<(u32, u32)>::new();
        for range in occupied {
            match merged.last_mut() {
                Some(previous) if range.0 <= previous.1 => previous.1 = previous.1.max(range.1),
                _ => merged.push(range),
            }
        }
        let mut unused = Vec::new();
        for &(base, limit) in &self.ranges {
            let mut cursor = base;
            for &(occupied_base, occupied_limit) in &merged {
                if occupied_limit <= cursor || occupied_base >= limit {
                    continue;
                }
                if cursor < occupied_base {
                    unused.push((cursor, occupied_base.min(limit)));
                }
                cursor = cursor.max(occupied_limit);
                if cursor >= limit {
                    break;
                }
            }
            if cursor < limit {
                unused.push((cursor, limit));
            }
        }
        unused
    }
}

fn align_up(value: u32, alignment: u32) -> Result<u32, PlacementError> {
    let alignment = alignment.max(1);
    if !alignment.is_power_of_two() {
        return Err(PlacementError::Overflow);
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(PlacementError::Overflow)
}

struct DisjointSets {
    parents: Vec<usize>,
}

impl DisjointSets {
    fn new(length: usize) -> Self {
        Self {
            parents: (0..length).collect(),
        }
    }

    fn find(&mut self, value: usize) -> usize {
        let parent = self.parents[value];
        if parent != value {
            self.parents[value] = self.find(parent);
        }
        self.parents[value]
    }

    fn union(&mut self, left: usize, right: usize) {
        let left = self.find(left);
        let right = self.find(right);
        if left != right {
            self.parents[right] = left;
        }
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
            let low = lower_to_tiles(&mid, &config).unwrap();
            let placement = place(&low).unwrap();
            let kernels = KernelBuildPlan::from_program(&low).unwrap();
            assert_eq!(placement.shard_addresses.len(), low.shards.len());
            for shard in &low.shards {
                let address = placement.shard_addresses[&shard.id];
                match shard.tensor_type.format.layout.memory_class {
                    MemoryClass::Interleaved => {
                        assert!(
                            (IPU21_INTERLEAVED_MEMORY_BASE..IPU21_APPLICATION_MEMORY_LIMIT)
                                .contains(&address)
                        )
                    }
                    MemoryClass::Standard => assert!(address >= IPU21_DATA_BASE),
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
                        if let KernelRequirements::Operator(requirements) = &run.requirements {
                            for operands in &requirements.memory_space.distinct_element_groups {
                                let mut ranges = Vec::new();
                                for operand in operands {
                                    let shards = match operand {
                                        MemoryOperand::Output => vec![run.output.shard],
                                        MemoryOperand::Input(index) => run.inputs
                                            [usize::from(*index)]
                                        .views
                                        .iter()
                                        .map(|view| view.shard)
                                        .collect(),
                                    };
                                    for shard in shards {
                                        let definition = &low.shards[shard.index() as usize];
                                        let element =
                                            memory_element_size(&low, &[shard.index() as usize]);
                                        let address = placement.shard_addresses[&shard];
                                        assert_eq!(address % element, 0);
                                        let bytes = shard_storage_bytes(definition).unwrap();
                                        ranges.push((
                                            definition.tensor_type.format.layout.memory_class,
                                            address / element,
                                            address.saturating_add(bytes).div_ceil(element),
                                        ));
                                    }
                                }
                                for (index, left) in ranges.iter().enumerate() {
                                    for right in &ranges[..index] {
                                        if left.0 == right.0 {
                                            assert!(left.2 <= right.1 || right.2 <= left.1);
                                        }
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
            let config = PipelineConfig::new(tiles)
                .with_input(left, format.clone())
                .with_input(right, format);
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let sum = mid.operations[0].results[0];
            let output = mid.operations[1].results[0];
            let low = lower_to_tiles(&mid, &config).unwrap();
            let placement = place(&low).unwrap();
            for tile in 0..tiles {
                let shard = |value| {
                    low.shards
                        .iter()
                        .find(|shard| {
                            shard.tile == tile && shard.definition == ShardDefinition::Value(value)
                        })
                        .unwrap()
                        .id
                };
                let left_address = placement.shard_addresses[&shard(mid.inputs[0].value)];
                let right_address = placement.shard_addresses[&shard(mid.inputs[1].value)];
                let sum_address = placement.shard_addresses[&shard(sum)];
                let output_address = placement.shard_addresses[&shard(output)];
                assert_ne!(sum_address, left_address);
                assert_ne!(sum_address, right_address);
                assert_eq!(output_address, left_address.min(right_address));
            }
        }
    }

    #[test]
    fn randomized_compact_arenas_keep_sequential_phases_below_their_peak_span() {
        let mut random = fastrand::Rng::with_seed(0x636f_6d70_6163_7421);
        for _ in 0..128 {
            let limit = 1 << 20;
            let persistent = random.u32(1..=4096);
            let mut arena = Arena::new(&[(0, limit)], true);
            arena.allocate(persistent, 4, 0, u32::MAX).unwrap();
            let mut bound = persistent;
            for phase in 1..=random.u32(2..=16) {
                let mut cursor = persistent;
                for _ in 0..random.u32(1..=8) {
                    let alignment = 1 << random.u32(2..=10);
                    let bytes = random.u32(1..=16 * 1024);
                    cursor = align_up(cursor, alignment).unwrap() + bytes;
                    arena.allocate(bytes, alignment, phase, phase).unwrap();
                }
                bound = bound.max(cursor);
            }
            assert!(arena.maximum_cursor() <= bound);
        }
    }
}
