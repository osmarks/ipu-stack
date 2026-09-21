use crate::{HostPhase, HostProgram};
use ipu_package::{
    Binding, HostCall, HostExchange, HostPage, HostSlice, RegionSlice, SEGMENT_EXECUTE,
    SEGMENT_READ, Segment,
};
use ipu_target::Target;
use ipu_target::ipu21::runtime_layout::{
    HOST_CLOSE_ADDRESS, HOST_PACKET_ADDRESS, HOST_STAGING_ADDRESS,
};
use std::collections::{BTreeMap, HashMap, VecDeque};

use super::package::{PackageBuildResult, invalid};

const HOST_DATA_START: u32 = crate::exchange::HOST_PAGE_BYTES;

#[derive(Clone, Copy)]
enum Direction {
    ToTile,
    ToHost,
}

#[derive(Clone, Copy)]
struct Transfer {
    direction: Direction,
    physical_tile: u16,
    tile_address: u32,
    host_offset: u32,
    bytes: u32,
    copy_destination: Option<u32>,
}

struct PendingTransfer {
    transfer: Transfer,
    file_offset: u64,
}

pub(crate) struct HostPackagePlan {
    /// Maximum tile requirement without address-dependent packet deduplication.
    pub descriptor_bytes: u32,
    pub programs: Vec<HostProgram>,
    pub segments: Vec<Vec<Segment>>,
    pub protocol: HostExchange,
    pub end: u32,
    pub staging_address: u32,
}

pub(crate) fn plan(
    target: Target,
    weights: &[Binding],
    inputs: &[Binding],
    outputs: &[Binding],
    execution_tiles: u16,
    base: u32,
    data_ranges: &[Vec<(u32, u32)>],
) -> PackageBuildResult<HostPackagePlan> {
    let Target::Ipu21 = target;

    if data_ranges.len() != usize::from(execution_tiles) {
        return Err(invalid("host plan has no data ranges for every tile"));
    }
    if weights.is_empty() && inputs.is_empty() && outputs.is_empty() {
        return Ok(HostPackagePlan {
            descriptor_bytes: 0,
            programs: vec![HostProgram::default(); usize::from(execution_tiles)],
            segments: vec![Vec::new(); usize::from(execution_tiles)],
            protocol: HostExchange::default(),
            end: base,
            staging_address: 0,
        });
    }

    let pending_weights = collect(weights, Direction::ToTile)?;
    let pending_inputs = collect(inputs, Direction::ToTile)?;
    let pending_outputs = collect(outputs, Direction::ToHost)?;
    let mut slots = pending_weights
        .iter()
        .chain(&pending_inputs)
        .chain(&pending_outputs)
        .map(|pending| (pending.transfer.physical_tile, 0))
        .collect::<BTreeMap<_, _>>();
    for (slot, value) in slots.values_mut().enumerate() {
        *value = u32::try_from(slot)?;
    }
    let (mut weight_phases, weight_slices, weight_ends) = batch(pending_weights, &slots)?;
    let (mut input_phases, input_slices, input_ends) = batch(pending_inputs, &slots)?;
    let (output_phases, output_slices, output_ends) = batch(pending_outputs, &slots)?;
    for transfer in weight_phases.iter_mut().chain(&mut input_phases).flatten() {
        transfer.copy_destination = Some(transfer.tile_address);
        transfer.tile_address = HOST_STAGING_ADDRESS;
        crate::exchange::plan_host_to_tile(
            ipu_target::ipu21::runtime_layout::EXCHANGE_WINDOW_BASE,
            transfer.physical_tile,
            transfer.tile_address,
            transfer.host_offset,
            transfer.bytes,
        )?;
    }
    for transfer in output_phases.iter().flatten() {
        crate::exchange::plan_tile_to_host(
            transfer.physical_tile,
            transfer.tile_address,
            transfer.host_offset,
            transfer.bytes,
        )?;
    }

    let weight_end = weight_phases.len();
    let input_end = weight_end + input_phases.len();
    let graph_batches = input_phases.len() + output_phases.len();
    let phases = weight_phases
        .into_iter()
        .chain(input_phases)
        .chain(output_phases)
        .collect::<Vec<_>>();
    let mut programs = Vec::with_capacity(usize::from(execution_tiles));
    let mut all_segments = Vec::with_capacity(usize::from(execution_tiles));
    let mut maximum_end = base;
    let mut descriptor_bytes = 0;
    for physical_tile in 0..execution_tiles {
        let planned = plan_tile(
            physical_tile,
            &phases,
            base,
            &data_ranges[usize::from(physical_tile)],
        )?;
        maximum_end = maximum_end.max(planned.end);
        descriptor_bytes = descriptor_bytes.max(planned.data_bytes);
        programs.push(HostProgram {
            initialize: planned.calls[..weight_end].to_vec(),
            inputs: planned.calls[weight_end..input_end].to_vec(),
            outputs: planned.calls[input_end..].to_vec(),
        });
        all_segments.push(planned.segments);
    }

    let mut calls = Vec::new();
    if weight_end != 0 {
        calls.push(HostCall {
            name: "initialize".into(),
            command: 0,
            phases: u32::try_from(weight_end * 2)?,
            inputs: weight_slices,
            outputs: Vec::new(),
            invocations: 1,
            input_batch_ends: weight_ends,
            output_batch_ends: Vec::new(),
        });
    }
    calls.push(HostCall {
        name: "run".into(),
        command: 0,
        phases: if graph_batches == 0 {
            0
        } else {
            u32::try_from(graph_batches * 2 - 1)?
        },
        inputs: input_slices,
        outputs: output_slices,
        invocations: 1,
        input_batch_ends: input_ends,
        output_batch_ends: output_ends,
    });
    let data_bytes = u64::from(crate::exchange::HOST_PAGE_BYTES)
        .checked_mul(u64::try_from(slots.len().max(1))?)
        .ok_or_else(|| invalid("host page arena overflow"))?;
    Ok(HostPackagePlan {
        descriptor_bytes,
        programs,
        segments: all_segments,
        protocol: HostExchange {
            startup_mark: ipu_target::ipu21::loader_abi::HOST_EXCHANGE_HANDOFF_MARK,
            command_page: 0,
            command_offset: 0,
            pages: vec![
                HostPage {
                    index: 0,
                    size: u64::from(crate::exchange::HOST_PAGE_BYTES),
                },
                HostPage {
                    index: 1,
                    size: data_bytes,
                },
            ],
            attach_order: vec![0, 1],
            calls,
        },
        end: maximum_end,
        staging_address: HOST_STAGING_ADDRESS,
    })
}

fn collect(bindings: &[Binding], direction: Direction) -> PackageBuildResult<Vec<PendingTransfer>> {
    let mut cursor = 0u64;
    let mut result = Vec::new();
    for binding in bindings {
        for slice in &binding.slices {
            append_slice(&mut result, direction, slice, cursor)?;
        }
        cursor = cursor
            .checked_add(binding.byte_len()?)
            .ok_or_else(|| invalid("host binding offset overflow"))?;
    }
    Ok(result)
}

fn append_slice(
    result: &mut Vec<PendingTransfer>,
    direction: Direction,
    slice: &RegionSlice,
    file_base: u64,
) -> PackageBuildResult<()> {
    let mut tile_address = slice.tile_address;
    let mut file_offset = file_base
        .checked_add(slice.file_offset)
        .ok_or_else(|| invalid("host file offset overflow"))?;
    let mut remaining = u32::try_from(slice.size)?;
    while remaining != 0 {
        let bytes = remaining.min(crate::exchange::HOST_PAGE_BYTES);
        result.push(PendingTransfer {
            transfer: Transfer {
                direction,
                physical_tile: u16::try_from(slice.tile)?,
                tile_address,
                host_offset: 0,
                bytes,
                copy_destination: None,
            },
            file_offset,
        });
        tile_address = tile_address
            .checked_add(bytes)
            .ok_or_else(|| invalid("host tile address overflow"))?;
        file_offset = file_offset
            .checked_add(u64::from(bytes))
            .ok_or_else(|| invalid("host file offset overflow"))?;
        remaining -= bytes;
    }
    Ok(())
}

fn batch(
    pending: Vec<PendingTransfer>,
    slots: &BTreeMap<u16, u32>,
) -> PackageBuildResult<(Vec<Vec<Transfer>>, Vec<HostSlice>, Vec<u32>)> {
    let mut queues = BTreeMap::<u16, VecDeque<_>>::new();
    for transfer in pending {
        queues
            .entry(transfer.transfer.physical_tile)
            .or_default()
            .push_back(transfer);
    }
    let mut phases = Vec::new();
    let mut slices = Vec::new();
    let mut ends = Vec::new();
    while queues.values().any(|queue| !queue.is_empty()) {
        let mut transfers = Vec::new();
        for (&tile, queue) in &mut queues {
            let Some(mut pending) = queue.pop_front() else {
                continue;
            };
            let page_offset = slots[&tile]
                .checked_mul(crate::exchange::HOST_PAGE_BYTES)
                .ok_or_else(|| invalid("host page offset overflow"))?;
            pending.transfer.host_offset = HOST_DATA_START
                .checked_add(page_offset)
                .ok_or_else(|| invalid("host exchange offset overflow"))?;
            slices.push(HostSlice {
                page: 1,
                page_offset: u64::from(page_offset),
                file_offset: pending.file_offset,
                size: u64::from(pending.transfer.bytes),
            });
            transfers.push(pending.transfer);
        }
        phases.push(transfers);
        ends.push(u32::try_from(slices.len())?);
    }
    Ok((phases, slices, ends))
}

struct PlannedTile {
    data_bytes: u32,
    calls: Vec<HostPhase>,
    segments: Vec<Segment>,
    end: u32,
}

fn plan_tile(
    physical_tile: u16,
    phases: &[Vec<Transfer>],
    base: u32,
    data_ranges: &[(u32, u32)],
) -> PackageBuildResult<PlannedTile> {
    let follower = align_up(base, 8)?;
    let mut cursor = follower + 12;
    let mut data_arena = DataArena::new(data_ranges);
    let mut data_bytes = 0u32;
    let mut segments = vec![segment(
        follower,
        words(&inactive_instructions()),
        SEGMENT_READ | SEGMENT_EXECUTE,
    )];
    let mut calls = Vec::with_capacity(phases.len());
    let mut packet_cache = HashMap::<Vec<u32>, u32>::new();
    for phase in phases {
        let target = target(physical_tile, phase);
        let targets = xreq_targets(physical_tile, phase)?;
        if target.is_none() && targets.is_empty() {
            calls.push(HostPhase {
                address: follower,
                active: false,
                run_table: None,
            });
            continue;
        }
        let (instructions, packet_words) = phase_instructions(physical_tile, target, &targets)?;
        cursor = align_up(cursor, 8)?;
        let address = cursor;
        let data = words(&instructions);
        cursor += u32::try_from(data.len())?;
        segments.push(segment(address, data, SEGMENT_READ | SEGMENT_EXECUTE));
        let packet_count = u32::try_from(packet_words.len())?;
        let packet_bytes = packet_count
            .checked_mul(4)
            .ok_or_else(|| invalid("host data size overflow"))?;
        let packet_source = if let Some(&source) = packet_cache.get(&packet_words) {
            source
        } else {
            let packet_data = words(&packet_words);
            let source = data_arena.allocate(packet_bytes, 4)?;
            segments.push(segment(source, packet_data, SEGMENT_READ));
            packet_cache.insert(packet_words, source);
            source
        };
        let descriptors =
            descriptor_words(target, packet_source, packet_count, targets.is_empty())?;
        let descriptor_data = words(&descriptors);
        data_bytes = data_bytes
            .checked_add(packet_bytes)
            .and_then(|bytes| bytes.checked_add(u32::try_from(descriptor_data.len()).ok()?))
            .ok_or_else(|| invalid("host data size overflow"))?;
        let table = data_arena.allocate(u32::try_from(descriptor_data.len())?, 4)?;
        segments.push(segment(table, descriptor_data, SEGMENT_READ));
        calls.push(HostPhase {
            address,
            active: true,
            run_table: Some(table),
        });
    }
    Ok(PlannedTile {
        data_bytes,
        calls,
        segments,
        end: cursor,
    })
}

struct DataArena {
    ranges: Vec<(u32, u32)>,
}

impl DataArena {
    fn new(ranges: &[(u32, u32)]) -> Self {
        Self {
            ranges: ranges.to_vec(),
        }
    }

    fn allocate(&mut self, bytes: u32, alignment: u32) -> PackageBuildResult<u32> {
        let candidate = self
            .ranges
            .iter()
            .enumerate()
            .filter_map(|(index, &(base, limit))| {
                let start = align_up(base, alignment).ok()?;
                let end = start.checked_add(bytes)?;
                (end <= limit).then(|| (limit - end, index, start, end))
            })
            .min_by_key(|candidate| (candidate.0, candidate.2))
            .ok_or_else(|| {
                invalid(format!(
                    "insufficient tile SRAM for {bytes} host-data bytes"
                ))
            })?;
        let (_, index, start, end) = candidate;
        let (base, limit) = self.ranges.swap_remove(index);
        if base < start {
            self.ranges.push((base, start));
        }
        if end < limit {
            self.ranges.push((end, limit));
        }
        Ok(start)
    }
}

fn phase_instructions(
    physical_tile: u16,
    target: Option<Transfer>,
    targets: &[u16],
) -> PackageBuildResult<(Vec<u32>, Vec<u32>)> {
    let target = target.map(target_program).transpose()?;
    let xreq = (!targets.is_empty())
        .then(|| {
            crate::exchange::assemble_host_xreq_program_for_targets(targets, HOST_PACKET_ADDRESS)
        })
        .transpose()?;
    Ok(match (target, xreq) {
        (Some(target), Some(xreq)) => {
            let mut packets = xreq.packet_words;
            packets.extend_from_slice(&target.packet_words);
            (
                crate::exchange::wrap_combined_host_operation(
                    physical_tile,
                    &target.instructions,
                    HOST_PACKET_ADDRESS,
                )?,
                packets,
            )
        }
        (None, Some(xreq)) => (
            crate::exchange::wrap_host_xreq_operation(physical_tile, &xreq.instructions)?,
            xreq.packet_words,
        ),
        (Some(target), None) => (
            crate::exchange::wrap_host_target_operation(physical_tile, &target.instructions)?,
            target.packet_words,
        ),
        (None, None) => return Err(invalid("active host phase has no work")),
    })
}

fn target_program(transfer: Transfer) -> PackageBuildResult<crate::exchange::TileToHostProgram> {
    Ok(match transfer.direction {
        Direction::ToTile => crate::exchange::assemble_host_to_tile_target_program(
            ipu_target::ipu21::runtime_layout::EXCHANGE_WINDOW_BASE,
            transfer.physical_tile,
            transfer.tile_address,
            transfer.host_offset,
            transfer.bytes,
            HOST_PACKET_ADDRESS + 8,
        )?,
        Direction::ToHost => crate::exchange::assemble_tile_to_host_target_program(
            ipu_target::ipu21::runtime_layout::EXCHANGE_WINDOW_BASE,
            transfer.physical_tile,
            transfer.tile_address,
            transfer.host_offset,
            transfer.bytes,
            HOST_PACKET_ADDRESS + 8,
            HOST_CLOSE_ADDRESS,
        )?,
    })
}

fn descriptor_words(
    target: Option<Transfer>,
    packet_source: u32,
    packet_words: u32,
    target_only: bool,
) -> PackageBuildResult<[u32; ipu_target::ipu21::runtime_layout::HOST_RUN_DESCRIPTOR_WORDS]> {
    let (destination, copy_words) = target
        .and_then(|transfer| Some((transfer.copy_destination?, transfer.bytes / 4)))
        .unwrap_or_default();
    if copy_words >= 1 << 23 || packet_words >= 1 << 8 {
        return Err(invalid("host descriptor is not encodable"));
    }
    Ok([
        destination,
        copy_words | (u32::from(target_only) << 23) | (packet_words << 24),
        packet_source,
    ])
}

fn target(physical_tile: u16, phase: &[Transfer]) -> Option<Transfer> {
    phase
        .iter()
        .copied()
        .find(|transfer| transfer.physical_tile == physical_tile)
}

fn xreq_targets(physical_tile: u16, phase: &[Transfer]) -> PackageBuildResult<Vec<u16>> {
    phase
        .iter()
        .filter_map(
            |transfer| match crate::exchange::host_hierarchy(transfer.physical_tile) {
                Ok(hierarchy) if hierarchy.xreq_physical_tile == physical_tile => {
                    Some(Ok(transfer.physical_tile))
                }
                Ok(_) => None,
                Err(error) => Some(Err(error.into())),
            },
        )
        .collect()
}

fn inactive_instructions() -> [u32; 3] {
    [
        ipu_target::ipu21::instruction::sans(1),
        ipu_target::ipu21::instruction::SYNC_ANS_INSTRUCTION,
        ipu_target::ipu21::instruction::RETURN_M10_INSTRUCTION,
    ]
}

fn segment(address: u32, data: Vec<u8>, flags: u32) -> Segment {
    Segment {
        address,
        memory_size: data.len() as u32,
        data,
        flags,
    }
}

fn words(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn align_up(value: u32, alignment: u32) -> PackageBuildResult<u32> {
    value
        .checked_next_multiple_of(alignment)
        .ok_or_else(|| invalid("host plan address overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_arena_skips_small_and_unalignable_holes() {
        let mut arena = DataArena::new(&[(1, 3), (8, 12), (17, 40)]);
        assert_eq!(arena.allocate(8, 8).unwrap(), 24);
        assert_eq!(arena.allocate(8, 8).unwrap(), 32);
        let remaining = arena.ranges.clone();
        assert!(arena.allocate(8, 8).is_err());
        assert_eq!(arena.ranges, remaining);
        assert_eq!(arena.allocate(4, 4).unwrap(), 8);
    }

    #[test]
    fn descriptor_reservation_survives_relocation_and_packet_deduplication() {
        let mut phases = (0..8)
            .map(|_| {
                vec![Transfer {
                    direction: Direction::ToHost,
                    physical_tile: 0,
                    tile_address: 0x80000,
                    host_offset: HOST_DATA_START,
                    bytes: 256,
                    copy_destination: None,
                }]
            })
            .collect::<Vec<_>>();
        let initial = plan_tile(0, &phases, 0x60000, &[(0x70000, 0x80000)]).unwrap();
        let reservation = initial.data_bytes;
        let actual = initial
            .segments
            .iter()
            .filter(|segment| segment.flags & SEGMENT_EXECUTE == 0)
            .map(|segment| segment.memory_size)
            .sum::<u32>();
        assert!(
            actual < reservation,
            "identical packets should share storage"
        );
        for (index, phase) in phases.iter_mut().enumerate() {
            phase[0].tile_address += index as u32 * 1024;
        }
        for base in [0x70000, 0x90000] {
            let relocated = plan_tile(0, &phases, 0x60000, &[(base, base + reservation)]).unwrap();
            assert_eq!(relocated.data_bytes, reservation);
            for segment in relocated
                .segments
                .iter()
                .filter(|segment| segment.flags & SEGMENT_EXECUTE == 0)
            {
                assert!(segment.address >= base);
                assert!(segment.address + segment.memory_size <= base + reservation);
            }
        }
    }
}
