//! Deterministic placement of logical shards in IPU21 tile SRAM.

use crate::low::{
    KernelRequirements, LowProgram, LowShardId, ShardDefinition, TileWork, TileWorkList,
};
use crate::memory::IPU21_DATA_BASE;
use crate::mid::{MemoryClass, OperandRequirement};
use crate::storage::{StorageError, shard_storage_bytes};
use ipu_exchange::{EXCHANGE_WINDOW_BASE, EXCHANGE_WINDOW_BYTES};
use ipu_package::{
    IPU21_INTERLEAVED_ELEMENT_SIZE, IPU21_INTERLEAVED_MEMORY_BASE, IPU21_INTERLEAVED_REGION_LIMIT,
    TILE_MEMORY_BASE, TILE_MEMORY_SIZE,
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
    #[error("placement arithmetic overflowed")]
    Overflow,
}

#[derive(Clone, Copy, Debug, Default)]
struct Requirement {
    alignment: u32,
    access_tail: u32,
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
    }

    let mut members = BTreeMap::<usize, Vec<usize>>::new();
    for index in 0..program.shards.len() {
        members.entry(sets.find(index)).or_default().push(index);
    }
    validate_alias_groups(program, &members)?;

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
            &root_requirements,
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
            &root_requirements,
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
        for transfer in &phase.transfers {
            for &destination in &transfer.destinations {
                let shard = &program.shards[destination.index() as usize];
                if !matches!(shard.definition, ShardDefinition::ExchangeCopy(_)) {
                    if !addresses.contains_key(&destination) {
                        return Err(PlacementError::InvalidAlias(destination.index()));
                    }
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
                addresses.insert(destination, *cursor);
                *cursor = end;
            }
        }
    }

    Ok(Placement {
        shard_addresses: addresses,
        tile_data_end,
    })
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
                    KernelRequirements::Operator(requirements) => {
                        (&requirements.inputs[..], &requirements.output)
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
    root_requirements: &BTreeMap<usize, Requirement>,
    arena: &mut Arena,
    addresses: &mut BTreeMap<LowShardId, u32>,
) -> Result<(), PlacementError> {
    for group in iterated.iter().filter(|group| group.tile == tile) {
        let roots = group
            .shards
            .iter()
            .map(|shard| {
                members.iter().find_map(|(&root, values)| {
                    values.contains(&(shard.index() as usize)).then_some(root)
                })
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(PlacementError::InvalidAlias(group.shards[0].index()))?;
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
                .filter_map(|root| root_requirements.get(root))
                .map(|requirement| requirement.alignment)
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
        let base = arena
            .allocate(bytes, alignment)
            .ok_or(PlacementError::OutOfMemory { tile, class, bytes })?;
        for (index, root) in roots.into_iter().enumerate() {
            let address = base
                .checked_add(
                    group
                        .stride
                        .checked_mul(u32::try_from(index).map_err(|_| PlacementError::Overflow)?)
                        .ok_or(PlacementError::Overflow)?,
                )
                .ok_or(PlacementError::Overflow)?;
            assign_members(addresses, &members[&root], address)?;
        }
    }
    for (&root, root_members) in members {
        let representative = &program.shards[root_members[0]];
        if representative.tile != tile
            || representative.tensor_type.format.layout.memory_class != class
            || matches!(representative.definition, ShardDefinition::ExchangeCopy(_))
            || grouped.contains(&root)
        {
            continue;
        }
        let requirement = root_requirements.get(&root).copied().unwrap_or_default();
        let bytes = allocation_bytes(program, root_members, requirement)?;
        let address = arena
            .allocate(bytes, requirement.alignment.max(4))
            .ok_or(PlacementError::OutOfMemory { tile, class, bytes })?;
        assign_members(addresses, root_members, address)?;
    }
    Ok(())
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
    range: usize,
    cursor: u32,
}

impl Arena {
    fn new(ranges: &[(u32, u32)]) -> Self {
        Self {
            ranges: ranges.to_vec(),
            range: 0,
            cursor: ranges[0].0,
        }
    }

    fn allocate(&mut self, bytes: u32, alignment: u32) -> Option<u32> {
        while self.range < self.ranges.len() {
            let (base, limit) = self.ranges[self.range];
            let start = align_up(self.cursor.max(base), alignment).ok()?;
            if start.checked_add(bytes)? <= limit {
                self.cursor = start + bytes;
                return Some(start);
            }
            self.range += 1;
            if let Some((base, _)) = self.ranges.get(self.range) {
                self.cursor = *base;
            }
        }
        None
    }

    fn maximum_cursor(&self) -> u32 {
        self.cursor
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
                    ShardDefinition::ExchangeCopy(_) => assert!(
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
                    }
                }
            }
        }
    }
}
