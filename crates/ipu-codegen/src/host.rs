use crate::{HostPhase, HostProgram};
use ipu_package::{
    Binding, HostCall, HostExchange, HostPage, HostSlice, RegionSlice, SEGMENT_EXECUTE,
    SEGMENT_READ, Segment,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use super::package::{PackageBuildResult, invalid};

const HOST_DATA_START: u32 = ipu_exchange::HOST_PAGE_BYTES;
const HOST_PACKET_ADDRESS: u32 = ipu_exchange::EXCHANGE_WINDOW_BASE;
const HOST_CLOSE_ADDRESS: u32 = ipu_exchange::EXCHANGE_WINDOW_BASE + 0x160;
const HOST_STAGING_ADDRESS: u32 = ipu_exchange::EXCHANGE_WINDOW_BASE + 0x180;

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

#[derive(Clone)]
struct Phase {
    transfers: Vec<Transfer>,
}

struct PendingTransfer {
    transfer: Transfer,
    file_offset: u64,
}

#[derive(Clone, Copy)]
struct PacketCopy {
    source: u32,
    destination: u32,
    words: u32,
}

pub(crate) struct HostPackagePlan {
    pub programs: Vec<HostProgram>,
    pub segments: Vec<Vec<Segment>>,
    pub protocol: HostExchange,
    pub end: u32,
    pub staging_address: u32,
}

pub(crate) fn plan(
    weights: &[Binding],
    inputs: &[Binding],
    outputs: &[Binding],
    execution_tiles: u16,
    base: u32,
    data_ranges: &[Vec<(u32, u32)>],
) -> PackageBuildResult<HostPackagePlan> {
    if data_ranges.len() != usize::from(execution_tiles) {
        return Err(invalid("host plan has no data ranges for every tile"));
    }
    if weights.is_empty() && inputs.is_empty() && outputs.is_empty() {
        return Ok(HostPackagePlan {
            programs: vec![HostProgram::default(); usize::from(execution_tiles)],
            segments: vec![Vec::new(); usize::from(execution_tiles)],
            protocol: HostExchange::default(),
            end: base,
            staging_address: 0,
        });
    }

    let mut weight_cursor = 0;
    let mut input_cursor = 0;
    let mut output_cursor = 0;
    let pending_weights = collect(weights, Direction::ToTile, &mut weight_cursor)?;
    let pending_inputs = collect(inputs, Direction::ToTile, &mut input_cursor)?;
    let pending_outputs = collect(outputs, Direction::ToHost, &mut output_cursor)?;
    let participating = pending_weights
        .iter()
        .chain(&pending_inputs)
        .chain(&pending_outputs)
        .map(|pending| pending.transfer.physical_tile)
        .collect::<BTreeSet<_>>();
    let slots = participating
        .into_iter()
        .enumerate()
        .map(|(slot, tile)| Ok((tile, u32::try_from(slot)?)))
        .collect::<PackageBuildResult<BTreeMap<_, _>>>()?;
    let (mut weight_phases, weight_slices, weight_ends) = batch(pending_weights, &slots)?;
    let (mut input_phases, input_slices, input_ends) = batch(pending_inputs, &slots)?;
    let (output_phases, output_slices, output_ends) = batch(pending_outputs, &slots)?;
    for transfer in weight_phases
        .iter_mut()
        .chain(&mut input_phases)
        .flat_map(|phase| &mut phase.transfers)
    {
        transfer.copy_destination = Some(transfer.tile_address);
        transfer.tile_address = HOST_STAGING_ADDRESS;
        ipu_exchange::plan_host_to_tile(
            transfer.physical_tile,
            transfer.tile_address,
            transfer.host_offset,
            transfer.bytes,
        )?;
    }
    for transfer in output_phases.iter().flat_map(|phase| &phase.transfers) {
        ipu_exchange::plan_tile_to_host(
            transfer.physical_tile,
            transfer.tile_address,
            transfer.host_offset,
            transfer.bytes,
        )?;
    }

    let phases = weight_phases
        .iter()
        .chain(&input_phases)
        .chain(&output_phases)
        .cloned()
        .collect::<Vec<_>>();
    let mut programs = Vec::with_capacity(usize::from(execution_tiles));
    let mut all_segments = Vec::with_capacity(usize::from(execution_tiles));
    let mut maximum_end = base;
    for physical_tile in 0..execution_tiles {
        let planned = plan_tile(
            physical_tile,
            &phases,
            base,
            &data_ranges[usize::from(physical_tile)],
        )?;
        maximum_end = maximum_end.max(planned.end);
        let weight_end = weight_phases.len();
        let input_end = weight_end + input_phases.len();
        programs.push(HostProgram {
            initialize: planned.calls[..weight_end].to_vec(),
            inputs: planned.calls[weight_end..input_end].to_vec(),
            outputs: planned.calls[input_end..].to_vec(),
        });
        all_segments.push(planned.segments);
    }

    let mut calls = Vec::new();
    if !weight_phases.is_empty() {
        calls.push(HostCall {
            name: "initialize".into(),
            command: 0,
            phases: u32::try_from(weight_phases.len() * 2)?,
            inputs: weight_slices,
            outputs: Vec::new(),
            invocations: 1,
            input_batch_ends: weight_ends,
            output_batch_ends: Vec::new(),
        });
    }
    let graph_batches = input_phases.len() + output_phases.len();
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
    let data_bytes = u64::from(ipu_exchange::HOST_PAGE_BYTES)
        .checked_mul(u64::try_from(slots.len().max(1))?)
        .ok_or_else(|| invalid("host page arena overflow"))?;
    Ok(HostPackagePlan {
        programs,
        segments: all_segments,
        protocol: HostExchange {
            startup_mark: ipu_driver::HOST_EXCHANGE_HANDOFF_MARK,
            command_page: 0,
            command_offset: 0,
            pages: vec![
                HostPage {
                    index: 0,
                    size: u64::from(ipu_exchange::HOST_PAGE_BYTES),
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

fn collect(
    bindings: &[Binding],
    direction: Direction,
    cursor: &mut u64,
) -> PackageBuildResult<Vec<PendingTransfer>> {
    let mut result = Vec::new();
    for binding in bindings {
        let base = *cursor;
        for slice in &binding.slices {
            append_slice(&mut result, direction, slice, base)?;
        }
        *cursor = cursor
            .checked_add(binding_size(binding)?)
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
        let bytes = remaining.min(ipu_exchange::HOST_PAGE_BYTES);
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

fn binding_size(binding: &Binding) -> PackageBuildResult<u64> {
    binding.slices.iter().try_fold(0, |size, slice| {
        slice
            .file_offset
            .checked_add(slice.size)
            .map(|end| size.max(end))
            .ok_or_else(|| invalid("binding size overflow"))
    })
}

fn batch(
    pending: Vec<PendingTransfer>,
    slots: &BTreeMap<u16, u32>,
) -> PackageBuildResult<(Vec<Phase>, Vec<HostSlice>, Vec<u32>)> {
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
                .checked_mul(ipu_exchange::HOST_PAGE_BYTES)
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
        phases.push(Phase { transfers });
        ends.push(u32::try_from(slices.len())?);
    }
    Ok((phases, slices, ends))
}

struct PlannedTile {
    calls: Vec<HostPhase>,
    segments: Vec<Segment>,
    end: u32,
}

fn plan_tile(
    physical_tile: u16,
    phases: &[Phase],
    base: u32,
    data_ranges: &[(u32, u32)],
) -> PackageBuildResult<PlannedTile> {
    let follower = align_up(base, 8)?;
    let mut cursor = follower + 12;
    let mut data_arena = DataArena::new(data_ranges);
    let mut segments = vec![segment(
        follower,
        words(&inactive_instructions()),
        SEGMENT_READ | SEGMENT_EXECUTE,
    )];
    let mut calls = Vec::with_capacity(phases.len());
    let mut packet_cache = HashMap::<Vec<u32>, u32>::new();
    for phase in phases {
        if !active(physical_tile, phase) {
            calls.push(HostPhase {
                address: follower,
                active: false,
                run_table: None,
            });
            continue;
        }
        let (instructions, packet_words) = phase_instructions(physical_tile, phase)?;
        cursor = align_up(cursor, 8)?;
        let address = cursor;
        let data = words(&instructions);
        cursor += u32::try_from(data.len())?;
        segments.push(segment(address, data, SEGMENT_READ | SEGMENT_EXECUTE));
        let packet_source = if let Some(&source) = packet_cache.get(&packet_words) {
            source
        } else {
            let packet_data = words(&packet_words);
            let source = data_arena.allocate(u32::try_from(packet_data.len())?, 4)?;
            segments.push(segment(source, packet_data, SEGMENT_READ));
            packet_cache.insert(packet_words.clone(), source);
            source
        };
        let packet = PacketCopy {
            source: packet_source,
            destination: if xreq_targets(physical_tile, phase)?.is_empty() {
                HOST_PACKET_ADDRESS + 8
            } else {
                HOST_PACKET_ADDRESS
            },
            words: u32::try_from(packet_words.len())?,
        };
        let descriptors = descriptor_words(physical_tile, phase, packet)?;
        let descriptor_data = words(&descriptors);
        let table = data_arena.allocate(u32::try_from(descriptor_data.len())?, 4)?;
        segments.push(segment(table, descriptor_data, SEGMENT_READ));
        calls.push(HostPhase {
            address,
            active: true,
            run_table: Some(table),
        });
    }
    Ok(PlannedTile {
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
                (end <= limit).then_some((limit - end, index, start, end))
            })
            .min_by_key(|candidate| (candidate.0, candidate.2))
            .ok_or_else(|| {
                invalid(format!(
                    "insufficient tile SRAM for {bytes} host-data bytes"
                ))
            })?;
        let (_, index, start, end) = candidate;
        let (base, limit) = self.ranges.remove(index);
        if base < start {
            self.ranges.push((base, start));
        }
        if end < limit {
            self.ranges.push((end, limit));
        }
        self.ranges.sort_unstable();
        Ok(start)
    }
}

fn phase_instructions(
    physical_tile: u16,
    phase: &Phase,
) -> PackageBuildResult<(Vec<u32>, Vec<u32>)> {
    let target = target(physical_tile, phase)
        .map(|transfer| target_program(transfer, HOST_PACKET_ADDRESS + 8))
        .transpose()?;
    let targets = xreq_targets(physical_tile, phase)?;
    let xreq = (!targets.is_empty())
        .then(|| {
            ipu_exchange::assemble_host_xreq_program_for_targets(&targets, HOST_PACKET_ADDRESS)
        })
        .transpose()?;
    Ok(match (target, xreq) {
        (Some(target), Some(xreq)) => {
            let mut packets = xreq.packet_words;
            packets.extend_from_slice(&target.packet_words);
            (
                ipu_exchange::wrap_combined_host_operation(
                    physical_tile,
                    &target.instructions,
                    HOST_PACKET_ADDRESS,
                )?,
                packets,
            )
        }
        (None, Some(xreq)) => (
            ipu_exchange::wrap_host_xreq_operation(physical_tile, &xreq.instructions)?,
            xreq.packet_words,
        ),
        (Some(target), None) => (
            ipu_exchange::wrap_host_target_operation(physical_tile, &target.instructions)?,
            target.packet_words,
        ),
        (None, None) => return Err(invalid("active host phase has no work")),
    })
}

fn target_program(
    transfer: Transfer,
    packet_address: u32,
) -> PackageBuildResult<ipu_exchange::TileToHostProgram> {
    Ok(match transfer.direction {
        Direction::ToTile => ipu_exchange::assemble_host_to_tile_target_program(
            transfer.physical_tile,
            transfer.tile_address,
            transfer.host_offset,
            transfer.bytes,
            packet_address,
        )?,
        Direction::ToHost => ipu_exchange::assemble_tile_to_host_target_program(
            transfer.physical_tile,
            transfer.tile_address,
            transfer.host_offset,
            transfer.bytes,
            packet_address,
            HOST_CLOSE_ADDRESS,
        )?,
    })
}

fn descriptor_words(
    physical_tile: u16,
    phase: &Phase,
    packet: PacketCopy,
) -> PackageBuildResult<Vec<u32>> {
    let target = target(physical_tile, phase);
    let copy_words = target
        .filter(|transfer| transfer.copy_destination.is_some())
        .map_or(0, |transfer| transfer.bytes / 4);
    if copy_words >= 1 << 23 || packet.words >= 1 << 8 {
        return Err(invalid("host descriptor is not encodable"));
    }
    let packet_destination = match packet.destination {
        HOST_PACKET_ADDRESS => 0,
        address if address == HOST_PACKET_ADDRESS + 8 => 1 << 23,
        _ => return Err(invalid("host packet destination is not encodable")),
    };
    Ok(vec![
        target
            .and_then(|transfer| transfer.copy_destination)
            .unwrap_or(0),
        copy_words | packet_destination | (packet.words << 24),
        packet.source,
    ])
}

fn target(physical_tile: u16, phase: &Phase) -> Option<Transfer> {
    phase
        .transfers
        .iter()
        .copied()
        .find(|transfer| transfer.physical_tile == physical_tile)
}

fn xreq_targets(physical_tile: u16, phase: &Phase) -> PackageBuildResult<Vec<u16>> {
    phase
        .transfers
        .iter()
        .filter_map(
            |transfer| match ipu_exchange::host_hierarchy(transfer.physical_tile) {
                Ok(hierarchy) if hierarchy.xreq_physical_tile == physical_tile => {
                    Some(Ok(transfer.physical_tile))
                }
                Ok(_) => None,
                Err(error) => Some(Err(error.into())),
            },
        )
        .collect()
}

fn active(physical_tile: u16, phase: &Phase) -> bool {
    target(physical_tile, phase).is_some()
        || xreq_targets(physical_tile, phase).is_ok_and(|targets| !targets.is_empty())
}

fn inactive_instructions() -> Vec<u32> {
    vec![
        ipu_exchange::sans(1),
        ipu_exchange::SYNC_ANS_INSTRUCTION,
        ipu_exchange::RETURN_M10_INSTRUCTION,
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
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| invalid("host plan address overflow"))
}
