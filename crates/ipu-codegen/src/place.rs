//! Deterministic placement of logical shards in IPU21 tile SRAM.

mod exchange;
pub(crate) use exchange::ExchangeConflicts;

use crate::low::{LowProgram, TileWorkList, TileWorkRef};
use crate::memory::IPU21_DATA_BASE;
use crate::{BlockValueId, ShardDefinition};
use crate::{MemoryClass, MemoryOperand};
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
    pub tile_auxiliary_ranges: Vec<Vec<(u32, u32)>>,
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
    place_with_ranges(
        program,
        &[(IPU21_DATA_BASE, IPU21_APPLICATION_MEMORY_LIMIT)],
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
    let started = std::time::Instant::now();
    if interleaved_offset >= IPU21_INTERLEAVED_ELEMENT_SIZE {
        return Err(PlacementError::Overflow);
    }
    if available_ranges.iter().any(|&(start, end)| {
        start < IPU21_DATA_BASE || end > IPU21_APPLICATION_MEMORY_LIMIT || start >= end
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
    let AllocationAnalysis {
        iterated,
        members,
        root_of_member,
        root_requirements,
        root_lifetimes,
    } = analyze_allocations(program)?;

    // Alias groups cannot cross tiles. Partition once rather than walking all
    // device allocations for each tile and memory class.
    let mut tile_members = vec![BTreeMap::new(); usize::from(program.tile_count)];
    for (root, members) in members {
        tile_members[usize::from(program.shards[members[0]].tile)].insert(root, members);
    }
    let mut tile_iterated = vec![Vec::new(); usize::from(program.tile_count)];
    for group in iterated {
        tile_iterated[usize::from(group.tile)].push(group);
    }
    let tile_placements = (0..usize::from(program.tile_count))
        .into_par_iter()
        .map(|tile| {
            place_tile(
                program,
                u16::try_from(tile).map_err(|_| PlacementError::Overflow)?,
                available_ranges,
                interleaved_offset,
                &tile_iterated[tile],
                &tile_members[tile],
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

    tracing::debug!(
        tiles = program.tile_count,
        shards = program.shards.len(),
        elapsed_us = started.elapsed().as_micros(),
        "placed tile storage"
    );
    Ok(Placement {
        shard_addresses: addresses,
        tile_auxiliary_ranges,
    })
}

struct AllocationAnalysis {
    iterated: Vec<IteratedGroup>,
    members: BTreeMap<usize, Vec<usize>>,
    root_of_member: Vec<usize>,
    root_requirements: BTreeMap<usize, Requirement>,
    root_lifetimes: BTreeMap<usize, Lifetime>,
}

fn analyze_allocations(program: &LowProgram) -> Result<AllocationAnalysis, PlacementError> {
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

    let mut requirements = vec![Requirement::default(); program.shards.len()];
    for tile in &program.tiles {
        collect_requirements(program, tile, &mut requirements);
    }
    // Loopback reads and receives simultaneously. Both access classes share
    // physical memory, so all local endpoints must occupy separate elements.
    for transfer in program
        .exchange_phases
        .iter()
        .flat_map(|phase| &phase.transfers)
    {
        let source = &program.shards[transfer.source.shard.index() as usize];
        for destination in &transfer.destinations {
            let destination = &program.shards[destination.shard.index() as usize];
            if source.tile == destination.tile {
                requirements[source.id.index() as usize].distinct_element = true;
                requirements[destination.id.index() as usize].distinct_element = true;
            }
        }
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

    Ok(AllocationAnalysis {
        iterated,
        members,
        root_of_member,
        root_requirements,
        root_lifetimes,
    })
}

#[allow(clippy::too_many_arguments)]
fn place_tile(
    program: &LowProgram,
    tile: u16,
    available_ranges: &[(u32, u32)],
    interleaved_offset: u32,
    iterated: &[IteratedGroup],
    members: &BTreeMap<usize, Vec<usize>>,
    root_of_member: &[usize],
    root_requirements: &BTreeMap<usize, Requirement>,
    root_lifetimes: &BTreeMap<usize, Lifetime>,
) -> Result<(u16, BTreeMap<BlockValueId, u32>, Vec<(u32, u32)>), PlacementError> {
    let mut grouped = BTreeSet::<usize>::new();
    for group in iterated {
        let roots = group
            .shards
            .iter()
            .map(|shard| root_of_member[shard.index() as usize])
            .collect::<Vec<_>>();
        if roots.iter().any(|root| !grouped.insert(*root)) {
            return Err(PlacementError::IteratedOverlap);
        }
    }

    // Both access classes share region 1. A single lifetime-ordered arena
    // lets ordinary storage reuse dead interleaved buffers and vice versa.
    let mut addresses = BTreeMap::new();
    let mut arena = Arena::new(available_ranges, interleaved_offset);
    allocate_tile(
        program,
        tile,
        iterated,
        &grouped,
        members,
        root_of_member,
        root_requirements,
        root_lifetimes,
        &mut arena,
        &mut addresses,
    )?;
    Ok((tile, addresses, arena.unused_ranges()))
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
    let outputs = shards_by_tile(
        program,
        program
            .outputs
            .iter()
            .flat_map(|output| output.shards.iter().copied()),
    );
    let mut lifetimes = vec![Lifetime::default(); program.shards.len()];
    for input in &program.inputs {
        for shard in &input.shards {
            lifetimes[shard.index() as usize].touch(0);
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
        for shard in &outputs[usize::from(tile.tile)] {
            lifetimes[shard.index() as usize].touch(event);
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
            for view in run.inputs.iter().flat_map(|operand| &operand.views) {
                touch(view.shard);
            }
            for output in run.outputs() {
                touch(output.shard);
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
                touch_work(program, nested, tile, event, lifetimes, exchanges);
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
    requirements: &mut [Requirement],
) {
    for work in program.work(tile) {
        match work {
            TileWorkRef::Kernel(run) => {
                let inputs = &run.requirements.inputs;
                let output = &run.requirements.output;
                let distinct_elements = &run.requirements.distinct_elements;
                for operands in distinct_elements {
                    for operand in operands {
                        match operand {
                            MemoryOperand::Output => {
                                requirements[run.output.shard.index() as usize].distinct_element =
                                    true;
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
                for (operand, requirement) in run.inputs.iter().zip(inputs) {
                    for view in &operand.views {
                        apply_requirement(
                            &mut requirements[view.shard.index() as usize],
                            requirement,
                        );
                    }
                }
                apply_requirement(&mut requirements[run.output.shard.index() as usize], output);
                for (view, requirement) in run
                    .additional_outputs
                    .iter()
                    .zip(&run.requirements.additional_outputs)
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
                collect_requirements(program, &repeat.body, requirements)
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
    members
        .iter()
        .map(|&index| {
            shard_storage_bytes(&program.shards[index])?
                .checked_add(requirement.access_tail)
                .ok_or(PlacementError::Overflow)
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or(PlacementError::Overflow)
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
    addresses: &mut BTreeMap<BlockValueId, u32>,
    members: &[usize],
    address: u32,
) -> Result<(), PlacementError> {
    for &member in members {
        addresses.insert(
            BlockValueId::from_index(u32::try_from(member).map_err(|_| PlacementError::Overflow)?),
            address,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn allocate_tile(
    program: &LowProgram,
    tile: u16,
    iterated: &[IteratedGroup],
    grouped: &BTreeSet<usize>,
    members: &BTreeMap<usize, Vec<usize>>,
    root_of_member: &[usize],
    root_requirements: &BTreeMap<usize, Requirement>,
    root_lifetimes: &BTreeMap<usize, Lifetime>,
    arena: &mut Arena,
    addresses: &mut BTreeMap<BlockValueId, u32>,
) -> Result<(), PlacementError> {
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
        let distinct_element = roots
            .iter()
            .any(|root| root_requirements[root].distinct_element);
        // The current iteration's argument aliases the first member. Isolate
        // the entire sequence from other allocations, rather than padding every
        // member to an element. Multiple constrained members may also be used
        // together outside the loop, so retain individual separation for those.
        let separate_members = roots
            .iter()
            .filter(|root| root_requirements[root].distinct_element)
            .take(2)
            .count()
            > 1;
        let alignment = group.alignment.max(
            roots
                .iter()
                .map(|root| {
                    if separate_members {
                        allocation_alignment(program, &members[root], root_requirements[root])
                    } else {
                        root_requirements[root].alignment
                    }
                })
                .max()
                .unwrap_or(1),
        );
        // Kernel access and loopback bank constraints are only complete after
        // expansion. Derive the physical repeat stride from those requirements.
        let mut stride = group.stride;
        for root in &roots {
            stride = stride.max(allocation_bytes(
                program,
                &members[root],
                root_requirements[root],
            )?);
        }
        let stride = align_up(stride, alignment)?;
        let bytes = stride
            .checked_mul(u32::try_from(roots.len()).map_err(|_| PlacementError::Overflow)?)
            .ok_or(PlacementError::Overflow)?;
        let mut lifetime = Lifetime::default();
        let mut assignments = Vec::with_capacity(roots.len());
        for (index, root) in roots.into_iter().enumerate() {
            lifetime.include(root_lifetimes[&root]);
            assignments.push((
                root,
                stride
                    .checked_mul(u32::try_from(index).map_err(|_| PlacementError::Overflow)?)
                    .ok_or(PlacementError::Overflow)?,
            ));
        }
        requests.push(AllocationRequest {
            class: group_class,
            region1_stride: separate_members
                .then(|| align_up(stride, IPU21_INTERLEAVED_ELEMENT_SIZE))
                .transpose()?,
            distinct_element,
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
        let bytes = allocation_bytes(program, root_members, requirement)?;
        requests.push(AllocationRequest {
            class: representative.tensor_type.format.layout.memory_class,
            region1_stride: None,
            distinct_element: requirement.distinct_element,
            lifetime,
            bytes,
            alignment: allocation_alignment(program, root_members, requirement).max(4),
            assignments: vec![(root, 0)],
        });
    }
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
    match allocate_requests(program, tile, &requests, members, arena, addresses) {
        Ok(()) => return Ok(()),
        Err(PlacementError::OutOfMemory { .. }) => {}
        Err(error) => return Err(error),
    }
    *arena = initial;
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
    allocate_requests(program, tile, &requests, members, arena, addresses)?;
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
    arena: &mut Arena,
    addresses: &mut BTreeMap<BlockValueId, u32>,
) -> Result<(), PlacementError> {
    for request in requests {
        let class = request.class;
        let Some(base) = arena.allocate(request) else {
            let representative = &program.shards[members[&request.assignments[0].0][0]];
            tracing::debug!(
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
            assign_members(addresses, &members[&root], address)?;
        }
    }
    Ok(())
}

struct AllocationRequest {
    class: MemoryClass,
    /// Iterated values need a wider physical stride if placed in region 1.
    region1_stride: Option<u32>,
    distinct_element: bool,
    lifetime: Lifetime,
    bytes: u32,
    alignment: u32,
    assignments: Vec<(usize, u32)>,
}

#[derive(Clone, Debug)]
struct IteratedGroup {
    tile: u16,
    shards: Vec<BlockValueId>,
    stride: u32,
    alignment: u32,
}

#[derive(Clone)]
struct Arena {
    offline: bool,
    history: Vec<(Lifetime, u32, u32)>,
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
                        if end <= start || limit <= base {
                            return vec![(start, limit)];
                        }
                        let mut pieces = Vec::with_capacity(2);
                        if start < base {
                            pieces.push((start, base));
                        }
                        if end < limit {
                            pieces.push((end, limit));
                        }
                        pieces
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
        let interleaved_offset = self.interleaved_offset;
        let candidate = self
            .free
            .iter()
            .enumerate()
            .flat_map(|(index, &(base, limit))| {
                // Keep candidate spans within one address region even after free
                // ranges coalesce across its boundary.
                [false, true].into_iter().filter_map(move |region1| {
                    let (base, limit) = if region1 {
                        (
                            base.max(
                                IPU21_INTERLEAVED_MEMORY_BASE
                                    + if request.class == MemoryClass::Ipu21Interleaved {
                                        interleaved_offset
                                    } else {
                                        0
                                    },
                            ),
                            limit,
                        )
                    } else {
                        if request.class == MemoryClass::Ipu21Interleaved {
                            return None;
                        }
                        (base, limit.min(IPU21_INTERLEAVED_MEMORY_BASE))
                    };
                    let element = if region1 {
                        IPU21_INTERLEAVED_ELEMENT_SIZE
                    } else {
                        TILE_MEMORY_ELEMENT_SIZE
                    };
                    let alignment =
                        request
                            .alignment
                            .max(if request.distinct_element { element } else { 1 });
                    let bytes = if region1 && let Some(stride) = request.region1_stride {
                        stride.checked_mul(u32::try_from(request.assignments.len()).ok()?)?
                    } else if request.distinct_element {
                        align_up(request.bytes, element).ok()?
                    } else {
                        request.bytes
                    };
                    let start = align_up(base, alignment).ok()?;
                    let mut end = start.checked_add(bytes)?;
                    // The loader stops partway through the last SRAM element.
                    // Reserve all its available bytes, but require only the
                    // actual payload/access tail to fit below the loading limit.
                    // Never truncate at an ordinary free gap: the remainder of
                    // that element may belong to another live allocation.
                    if limit == IPU21_APPLICATION_MEMORY_LIMIT
                        && request.distinct_element
                        && request.region1_stride.is_none()
                        && start.checked_add(request.bytes)? <= limit
                    {
                        end = end.min(limit);
                    }
                    // Ordinary buffers prefer region 0; compact addresses within
                    // each region leave long contiguous spans for later requests.
                    (end <= limit).then_some(((region1, start), index, start, end))
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
            self.active.push((last, start, end - start));
            self.occupied.push((start, end));
            if self.offline {
                self.history.push((request.lifetime, start, end));
            }
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

    fn request(
        class: MemoryClass,
        bytes: u32,
        alignment: u32,
        first: u32,
        last: u32,
    ) -> AllocationRequest {
        AllocationRequest {
            class,
            region1_stride: None,
            bytes,
            alignment,
            distinct_element: false,
            lifetime: Lifetime {
                first,
                last,
                seen: true,
            },
            assignments: Vec::new(),
        }
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
    fn joint_arena_respects_region_and_element_constraints() {
        let base = IPU21_INTERLEAVED_MEMORY_BASE;
        let mut arena = Arena::new(&[(base - 64, base + 2 * IPU21_INTERLEAVED_ELEMENT_SIZE)], 0);
        assert_eq!(
            arena.allocate(&request(MemoryClass::Ipu21Standard, 64, 8, 0, 2)),
            Some(base - 64)
        );
        let mut constrained = request(MemoryClass::Ipu21Standard, 8, 8, 0, 2);
        constrained.distinct_element = true;
        assert_eq!(arena.allocate(&constrained), Some(base));
        constrained.class = MemoryClass::Ipu21Interleaved;
        assert_eq!(
            arena.allocate(&constrained),
            Some(base + IPU21_INTERLEAVED_ELEMENT_SIZE)
        );
        assert!(
            arena
                .allocate(&request(MemoryClass::Ipu21Standard, 8, 8, 2, 2))
                .is_none()
        );
        assert!(
            arena
                .allocate(&request(MemoryClass::Ipu21Interleaved, 64, 8, 3, 3))
                .is_some()
        );
    }

    #[test]
    fn final_partial_element_accepts_payload_but_preserves_exclusivity() {
        let limit = IPU21_APPLICATION_MEMORY_LIMIT;
        let base = limit / IPU21_INTERLEAVED_ELEMENT_SIZE * IPU21_INTERLEAVED_ELEMENT_SIZE;
        let mut arena = Arena::new(&[(base, limit)], 0);
        let mut payload = request(MemoryClass::Ipu21Interleaved, limit - base, 8, 0, 1);
        payload.distinct_element = true;
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

        // The same partial free span elsewhere may border a live allocation.
        let base = IPU21_INTERLEAVED_MEMORY_BASE;
        let mut ordinary_gap = Arena::new(&[(base, base + 1024)], 0);
        payload.bytes = 512;
        assert!(ordinary_gap.allocate(&payload).is_none());
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
            let mut placed = Vec::<(u32, u32, u32, u32)>::new();
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
                    request.distinct_element = random.bool();
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
                    for &(other, other_end, other_first, other_last) in &placed {
                        if first <= other_last && other_first <= request.lifetime.last {
                            assert!(end <= other || other_end <= address);
                        }
                    }
                    placed.push((address, end, first, request.lifetime.last));
                }
            }
            for (base, end) in arena.unused_ranges() {
                assert!(
                    placed
                        .iter()
                        .all(|&(other, other_end, _, _)| end <= other || other_end <= base)
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
        // selected the iterated values' shapes and minimum strides.
        for run in &mut expanded.kernel_runs {
            if run.inputs.len() == 2 {
                std::sync::Arc::make_mut(&mut run.metadata)
                    .requirements
                    .distinct_elements
                    .push(vec![
                        crate::MemoryOperand::Output,
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
                    for input in &repeat.iterated {
                        assert!(input.stride_bytes < TILE_MEMORY_ELEMENT_SIZE);
                        let output = placement.shard_addresses[&repeat.carried[0].initial];
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
                                input.stride_bytes
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
        let group = analysis
            .iterated
            .iter()
            .find(|group| group.tile == 0)
            .unwrap()
            .clone();
        let second_root = analysis.root_of_member[group.shards[1].index() as usize];
        analysis
            .root_requirements
            .get_mut(&second_root)
            .unwrap()
            .distinct_element = true;
        let members = analysis
            .members
            .into_iter()
            .filter(|(_, members)| low.shards[members[0]].tile == 0)
            .collect();
        let (_, addresses, _) = place_tile(
            &low,
            0,
            &[(IPU21_DATA_BASE, IPU21_APPLICATION_MEMORY_LIMIT)],
            0,
            std::slice::from_ref(&group),
            &members,
            &analysis.root_of_member,
            &analysis.root_requirements,
            &analysis.root_lifetimes,
        )
        .unwrap();
        assert_eq!(
            addresses[&group.shards[1]] - addresses[&group.shards[0]],
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
            [crate::OperandRequirement::new(format.clone(), 8)],
            crate::OperandRequirement::new(format, 8),
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
        program.outputs[0].shards.reverse();
        let low = lower_to_tiles(&std::sync::Arc::new(program), false);
        let lifetimes = collect_lifetimes(&low);
        for &id in &low.outputs[0].shards {
            let tile = low.shards[id.index() as usize].tile;
            let end = 1 + low.work(&low.tiles[usize::from(tile)]).count() as u32;
            assert_eq!(
                lifetimes[id.index() as usize].last,
                end,
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
                    low.values
                        .iter()
                        .find(|v| v.value == value)
                        .unwrap()
                        .shards
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
