//! Deterministic placement of logical shards in IPU21 tile SRAM.

use crate::low::{
    KernelRequirements, LowProgram, LowShardId, ShardDefinition, TileWork, TileWorkList,
};
use crate::memory::IPU21_DATA_BASE;
use crate::mid::{MemoryClass, MemoryOperand, MemoryRelation, OperandRequirement};
use crate::storage::{StorageError, shard_storage_bytes};
use ipu_exchange::{EXCHANGE_WINDOW_BASE, EXCHANGE_WINDOW_BYTES};
use ipu_package::{
    IPU21_INTERLEAVED_ELEMENT_SIZE, IPU21_INTERLEAVED_MEMORY_BASE, IPU21_INTERLEAVED_REGION_LIMIT,
    TILE_MEMORY_BASE, TILE_MEMORY_ELEMENT_SIZE, TILE_MEMORY_SIZE,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub shard_addresses: BTreeMap<LowShardId, u32>,
    pub tile_data_end: Vec<u32>,
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
    #[error("tile {tile} exchange phase needs more than the receive window")]
    ExchangeWindow { tile: u16 },
    #[error("tile {tile} exchange staging slots overlap")]
    ExchangeStagingConflict { tile: u16 },
    #[error("placement arithmetic overflowed")]
    Overflow,
}

#[derive(Clone, Copy, Debug, Default)]
struct Requirement {
    alignment: u32,
    access_tail: u32,
    distinct_element: bool,
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
    let mut sets = DisjointSets::new(program.shards.len());
    for shard in &program.shards {
        if let ShardDefinition::Alias(target) = shard.definition {
            checked_union(program, &mut sets, shard.id, target)?;
        }
    }
    let mut iterated = Vec::<IteratedGroup>::new();
    for tile in &program.tiles {
        collect_repeat_constraints(program, tile, &mut sets, &mut iterated)?;
    }

    let mut requirements = vec![Requirement::default(); program.shards.len()];
    for tile in &program.tiles {
        collect_requirements(tile, &mut requirements);
    }
    let mut root_requirements = BTreeMap::<usize, Requirement>::new();
    for (index, requirement) in requirements.into_iter().enumerate() {
        let root = sets.find(index);
        let combined = root_requirements.entry(root).or_default();
        combined.alignment = combined.alignment.max(requirement.alignment);
        combined.access_tail = combined.access_tail.max(requirement.access_tail);
        combined.distinct_element |= requirement.distinct_element;
    }

    let mut members = BTreeMap::<usize, Vec<usize>>::new();
    let mut root_of_member = vec![0usize; program.shards.len()];
    for index in 0..program.shards.len() {
        let root = sets.find(index);
        root_of_member[index] = root;
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

    let mut addresses = BTreeMap::new();
    let mut tile_data_end = vec![IPU21_DATA_BASE; usize::from(program.tile_count)];
    for tile in 0..program.tile_count {
        let mut grouped = BTreeSet::<usize>::new();
        for group in iterated.iter().filter(|group| group.tile == tile) {
            let roots = group
                .shards
                .iter()
                .map(|shard| sets.find(shard.index() as usize))
                .collect::<Vec<_>>();
            if roots.iter().any(|root| !grouped.insert(*root)) {
                return Err(PlacementError::IteratedOverlap);
            }
        }

        // Region 1 is shared by ordinary and interleaved loads. Place the
        // interleaved working set first, round its boundary to a paired memory
        // element, then return every remaining byte to standard allocations.
        let mut interleaved = Arena::new(&[(
            IPU21_INTERLEAVED_MEMORY_BASE,
            IPU21_INTERLEAVED_REGION_LIMIT,
        )]);
        allocate_tile_class(
            program,
            tile,
            MemoryClass::Ipu21Interleaved,
            &iterated,
            &grouped,
            &members,
            &root_of_member,
            &root_requirements,
            &root_lifetimes,
            &mut interleaved,
            &mut addresses,
        )?;
        let interleaved_boundary =
            align_up(interleaved.maximum_cursor(), IPU21_INTERLEAVED_ELEMENT_SIZE)?;
        if interleaved_boundary > IPU21_INTERLEAVED_REGION_LIMIT {
            return Err(PlacementError::OutOfMemory {
                tile,
                class: MemoryClass::Ipu21Interleaved,
                bytes: interleaved_boundary - IPU21_INTERLEAVED_MEMORY_BASE,
            });
        }
        let mut standard = Arena::new(&[
            (IPU21_DATA_BASE, IPU21_INTERLEAVED_MEMORY_BASE),
            (interleaved_boundary, TILE_MEMORY_BASE + TILE_MEMORY_SIZE),
        ]);
        allocate_tile_class(
            program,
            tile,
            MemoryClass::Ipu21Standard,
            &iterated,
            &grouped,
            &members,
            &root_of_member,
            &root_requirements,
            &root_lifetimes,
            &mut standard,
            &mut addresses,
        )?;
        tile_data_end[usize::from(tile)] =
            standard.maximum_cursor().max(interleaved.maximum_cursor());
    }

    // Exchange copies live only from their phase through the immediately
    // following consumer, so every phase can reuse the receive window.
    for phase in &program.exchange_phases {
        let mut cursors = vec![EXCHANGE_WINDOW_BASE; usize::from(program.tile_count)];
        let mut fixed = vec![Vec::<(u32, u32)>::new(); usize::from(program.tile_count)];
        for transfer in &phase.transfers {
            for &destination in &transfer.destinations {
                let shard = &program.shards[destination.index() as usize];
                let ShardDefinition::ExchangeStaging {
                    window_offset: Some(window_offset),
                } = shard.definition
                else {
                    continue;
                };
                let requirement = root_requirements
                    .get(&sets.find(destination.index() as usize))
                    .copied()
                    .unwrap_or_default();
                let address = EXCHANGE_WINDOW_BASE
                    .checked_add(window_offset)
                    .ok_or(PlacementError::Overflow)?;
                if address != align_up(address, requirement.alignment.max(4))? {
                    return Err(PlacementError::ExchangeStagingConflict { tile: shard.tile });
                }
                let bytes =
                    allocation_bytes(program, &[destination.index() as usize], requirement)?;
                let end = address.checked_add(bytes).ok_or(PlacementError::Overflow)?;
                if end > EXCHANGE_WINDOW_BASE + EXCHANGE_WINDOW_BYTES {
                    return Err(PlacementError::ExchangeWindow { tile: shard.tile });
                }
                if fixed[usize::from(shard.tile)]
                    .iter()
                    .any(|&(other_start, other_end)| address < other_end && other_start < end)
                {
                    return Err(PlacementError::ExchangeStagingConflict { tile: shard.tile });
                }
                if addresses
                    .insert(destination, address)
                    .is_some_and(|old| old != address)
                {
                    return Err(PlacementError::ExchangeStagingConflict { tile: shard.tile });
                }
                fixed[usize::from(shard.tile)].push((address, end));
                cursors[usize::from(shard.tile)] = cursors[usize::from(shard.tile)].max(end);
            }
        }
        for transfer in &phase.transfers {
            for &destination in &transfer.destinations {
                let shard = &program.shards[destination.index() as usize];
                if !matches!(shard.definition, ShardDefinition::ExchangeStaging { .. }) {
                    if !addresses.contains_key(&destination) {
                        return Err(PlacementError::InvalidAlias(destination.index()));
                    }
                    continue;
                }
                if matches!(
                    shard.definition,
                    ShardDefinition::ExchangeStaging {
                        window_offset: Some(_)
                    }
                ) {
                    continue;
                }
                let requirement = root_requirements
                    .get(&sets.find(destination.index() as usize))
                    .copied()
                    .unwrap_or_default();
                let cursor = &mut cursors[usize::from(shard.tile)];
                *cursor = align_up(*cursor, requirement.alignment.max(4))?;
                let bytes =
                    allocation_bytes(program, &[destination.index() as usize], requirement)?;
                let end = cursor.checked_add(bytes).ok_or(PlacementError::Overflow)?;
                if end > EXCHANGE_WINDOW_BASE + EXCHANGE_WINDOW_BYTES {
                    return Err(PlacementError::ExchangeWindow { tile: shard.tile });
                }
                if addresses
                    .insert(destination, *cursor)
                    .is_some_and(|old| old != *cursor)
                {
                    return Err(PlacementError::ExchangeStagingConflict { tile: shard.tile });
                }
                *cursor = end;
            }
        }
    }

    Ok(Placement {
        shard_addresses: addresses,
        tile_data_end,
    })
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
        for work in &tile.work {
            touch_work(program, work, tile.tile, event, &mut lifetimes);
            event = event.saturating_add(1);
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
                ShardDefinition::ExchangeStaging { .. }
            )
        {
            // Unreferenced canonical values remain conservatively resident.
            lifetime.touch(0);
        }
    }
    lifetimes
}

fn touch_work(
    program: &LowProgram,
    work: &TileWork,
    tile: u16,
    event: u32,
    lifetimes: &mut [Lifetime],
) {
    let mut touch = |shard: LowShardId| lifetimes[shard.index() as usize].touch(event);
    match work {
        TileWork::Kernel(run) => {
            for view in run.inputs.iter().flat_map(|operand| &operand.views) {
                touch(view.shard);
            }
            touch(run.output.shard);
        }
        TileWork::LocalCopy(copy) => {
            touch(copy.source);
            touch(copy.destination);
        }
        TileWork::Exchange(id) => {
            for transfer in &program.exchange_phases[id.index() as usize].transfers {
                if program.shards[transfer.source.shard.index() as usize].tile == tile {
                    touch(transfer.source.shard);
                }
                for destination in &transfer.destinations {
                    if program.shards[destination.index() as usize].tile == tile {
                        touch(*destination);
                    }
                }
            }
        }
        TileWork::Repeat(repeat) => {
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
            for nested in &repeat.body.work {
                touch_work(program, nested, tile, event, lifetimes);
            }
        }
    }
}

fn collect_repeat_constraints(
    program: &LowProgram,
    tile: &TileWorkList,
    sets: &mut DisjointSets,
    iterated: &mut Vec<IteratedGroup>,
) -> Result<(), PlacementError> {
    for work in &tile.work {
        let TileWork::Repeat(repeat) = work else {
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

fn collect_requirements(tile: &TileWorkList, requirements: &mut [Requirement]) {
    for work in &tile.work {
        match work {
            TileWork::Kernel(run) => {
                let (inputs, output) = match &run.requirements {
                    KernelRequirements::Operator(operator_requirements) => {
                        for relation in &operator_requirements.memory_relations {
                            let MemoryRelation::DistinctElements(operands) = relation;
                            for operand in operands {
                                match operand {
                                    MemoryOperand::Output => {
                                        requirements[run.output.shard.index() as usize]
                                            .distinct_element = true;
                                    }
                                    MemoryOperand::Input(index) => {
                                        if let Some(input) = run.inputs.get(usize::from(*index)) {
                                            for view in &input.views {
                                                requirements[view.shard.index() as usize]
                                                    .distinct_element = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        (
                            &operator_requirements.inputs[..],
                            &operator_requirements.output,
                        )
                    }
                    KernelRequirements::Conversion { input, output } => {
                        (std::slice::from_ref(input), output)
                    }
                };
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
            TileWork::LocalCopy(copy) => {
                requirements[copy.source.index() as usize].alignment =
                    requirements[copy.source.index() as usize].alignment.max(8);
                requirements[copy.destination.index() as usize].alignment = requirements
                    [copy.destination.index() as usize]
                    .alignment
                    .max(8);
            }
            TileWork::Repeat(repeat) => collect_requirements(&repeat.body, requirements),
            TileWork::Exchange(_) => {}
        }
    }
}

fn apply_requirement(target: &mut Requirement, requirement: &OperandRequirement) {
    target.alignment = target.alignment.max(requirement.alignment);
    target.access_tail = target.access_tail.max(requirement.access_tail_bytes);
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
    requirement: Requirement,
) -> Result<u32, PlacementError> {
    let bytes = members
        .iter()
        .map(|&index| {
            shard_storage_bytes(&program.shards[index])?
                .checked_add(requirement.access_tail)
                .ok_or(PlacementError::Overflow)
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or(PlacementError::Overflow)?;
    if requirement.distinct_element {
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
        MemoryClass::Ipu21Standard => TILE_MEMORY_ELEMENT_SIZE,
        MemoryClass::Ipu21Interleaved => IPU21_INTERLEAVED_ELEMENT_SIZE,
    }
}

fn allocation_alignment(program: &LowProgram, members: &[usize], requirement: Requirement) -> u32 {
    if requirement.distinct_element {
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
    root_requirements: &BTreeMap<usize, Requirement>,
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
        if representative.tile != tile
            || representative.tensor_type.format.layout.memory_class != class
            || matches!(
                representative.definition,
                ShardDefinition::ExchangeStaging { .. }
            )
            || grouped.contains(&root)
        {
            continue;
        }
        let requirement = root_requirements.get(&root).copied().unwrap_or_default();
        let bytes = allocation_bytes(program, root_members, requirement)?;
        requests.push(AllocationRequest {
            lifetime: root_lifetimes.get(&root).copied().unwrap_or_default(),
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
    free: Vec<(u32, u32)>,
    active: Vec<(u32, u32, u32)>,
    maximum: u32,
}

impl Arena {
    fn new(ranges: &[(u32, u32)]) -> Self {
        Self {
            free: ranges.to_vec(),
            active: Vec::new(),
            maximum: ranges[0].0,
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
                (end <= limit).then(|| (limit - end, index, start, end))
            })
            .min_by_key(|candidate| (candidate.0, candidate.2));
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
    fn randomized_gemm_placement_respects_classes_exchange_window_and_kernel_views() {
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
                        layout: Layout::amp_right(64, tiles),
                    },
                );
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();
            let placement = place(&low).unwrap();
            let kernels = KernelBuildPlan::from_program(&low).unwrap();
            assert_eq!(placement.shard_addresses.len(), low.shards.len());
            for shard in &low.shards {
                let address = placement.shard_addresses[&shard.id];
                match shard.definition {
                    ShardDefinition::ExchangeStaging {
                        window_offset: Some(window_offset),
                    } => assert_eq!(address, EXCHANGE_WINDOW_BASE + window_offset),
                    ShardDefinition::ExchangeStaging {
                        window_offset: None,
                    } => assert!(
                        (EXCHANGE_WINDOW_BASE..EXCHANGE_WINDOW_BASE + EXCHANGE_WINDOW_BYTES)
                            .contains(&address)
                    ),
                    _ if shard.tensor_type.format.layout.memory_class
                        == MemoryClass::Ipu21Interleaved =>
                    {
                        assert!(
                            (IPU21_INTERLEAVED_MEMORY_BASE..IPU21_INTERLEAVED_REGION_LIMIT)
                                .contains(&address)
                        )
                    }
                    _ => assert!(address >= IPU21_DATA_BASE),
                }
            }
            for tile in &low.tiles {
                for work in &tile.work {
                    if let TileWork::Kernel(run) = work {
                        materialize_kernel_run(
                            run,
                            &low.shards,
                            &placement.shard_addresses,
                            &kernels,
                        )
                        .unwrap();
                        if let KernelRequirements::Operator(requirements) = &run.requirements {
                            for relation in &requirements.memory_relations {
                                let MemoryRelation::DistinctElements(operands) = relation;
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
}
