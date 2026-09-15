use super::hazards::MemoryHistory;
use super::program::{
    PhaseProgramBuilder, PhaseTransferTiming, ScheduledPayloadTiming, patch_receiver_address,
    patch_sender_address, point_to_point,
};
use super::{
    ExchangeActivity, ExchangeActivityKind, ExchangeItemWidth, ExchangeLoweringError,
    PendingTransfer, PhysicalExchangePhase, SchedulingProblem,
};
use crate::ExchangePhaseId;
use ipu_target::ipu21::fabric::Topology;
use ipu_target::ipu21::memory::effective_memory_elements;
use std::cmp::Reverse;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TileAvailability {
    pub(super) send: u32,
    pub(super) receive: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TilePredecessor {
    pub(super) send: Option<usize>,
    pub(super) receive: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct MaterializedTiming {
    pub(super) end: u32,
    pub(super) predecessor: Option<usize>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct TileMemorySchedule {
    sends: MemoryHistory,
    receives: MemoryHistory,
}

pub(super) struct MaterializedSchedule {
    pub(super) builder: PhaseProgramBuilder,
    pub(super) horizon: u32,
    pub(super) tile_availability: Vec<TileAvailability>,
    memory_accesses: Vec<TileMemorySchedule>,
    pub(super) activities: Vec<Vec<ExchangeActivity>>,
    pub(super) order: Vec<usize>,
    pub(super) timings: Vec<Option<MaterializedTiming>>,
    moving_base: Vec<bool>,
}

impl MaterializedSchedule {
    pub(super) fn new(tile_count: u16, transfers: &[PendingTransfer]) -> Self {
        let transfer_count = transfers.len();
        // A fixed history-entry budget becomes an accidental size cutoff: one
        // complete incremental pass itself grows quadratically per endpoint.
        // Allow four conservative full-pass bounds, with headroom for small
        // phases. This bounds retry effort relative to the input's inherent work.
        let mut endpoints = vec![(0u64, 0u64); usize::from(tile_count)];
        for transfer in transfers {
            endpoints[usize::from(transfer.source)].0 += 1;
            for &(tile, _) in &transfer.destinations {
                endpoints[usize::from(tile)].1 += 1;
            }
        }
        let history_work = endpoints.into_iter().fold(0u64, |work, (sends, receives)| {
            // Each receive can introduce mux, pointer and mode controls.
            work.saturating_add((sends + receives).saturating_mul(1 + sends + 3 * receives))
        });
        let validation_budget = (512 * 1024 * 1024).max(history_work.saturating_mul(4));
        Self {
            builder: PhaseProgramBuilder::new(tile_count).with_validation_budget(validation_budget),
            horizon: 0,
            tile_availability: vec![TileAvailability::default(); usize::from(tile_count)],
            memory_accesses: (0..tile_count)
                .map(|_| TileMemorySchedule::default())
                .collect(),
            activities: vec![Vec::new(); usize::from(tile_count)],
            order: Vec::with_capacity(transfer_count),
            timings: vec![None; transfer_count],
            moving_base: {
                let mut moving = vec![false; usize::from(tile_count)];
                for transfer in transfers {
                    moving[usize::from(transfer.source)] |= transfer.moving_source();
                }
                moving
            },
        }
    }

    pub(super) fn append(
        &mut self,
        topology: &Topology,
        pending: &[PendingTransfer],
        incoming_bases: &[u32],
        receive_counts: &[usize],
        index: usize,
        dependency_ready: u32,
        validate_encoding: bool,
        last_transfer: &mut [TilePredecessor],
    ) -> Result<u32, ExchangeLoweringError> {
        let transfer = &pending[index];
        let moving = transfer.moving_source();
        let current = &mut self.moving_base[usize::from(transfer.source)];
        if *current != moving {
            self.builder
                .switch_outgoing_base(transfer.source, if moving { 6 } else { 15 })?;
            *current = moving;
        }

        let (blocking_tile, latest_availability) = std::iter::once((
            transfer.source,
            self.tile_availability[usize::from(transfer.source)].send,
        ))
        .chain(transfer.reserved_source.into_iter().map(|tile| {
            let availability = self.tile_availability[usize::from(tile)];
            (tile, availability.send)
        }))
        .chain(
            transfer
                .destinations
                .iter()
                .map(|&(tile, _)| (tile, self.tile_availability[usize::from(tile)].receive)),
        )
        .max_by_key(|&(tile, availability)| (availability, Reverse(tile)))
        .unwrap_or((transfer.source, 0));
        let blocking_tile = if dependency_ready > latest_availability {
            transfer.source
        } else {
            blocking_tile
        };
        let predecessor = if blocking_tile == transfer.source
            || transfer.reserved_source == Some(blocking_tile)
        {
            last_transfer[usize::from(blocking_tile)].send
        } else {
            last_transfer[usize::from(blocking_tile)].receive
        };
        let timing = append_transfer(
            topology,
            &self.memory_accesses,
            incoming_bases,
            receive_counts,
            transfer,
            u32::try_from(index).map_err(|_| ExchangeLoweringError::Overflow)?,
            dependency_ready,
            &mut self.builder,
            validate_encoding,
        )?;
        let payload_end = timing.payload_end;
        self.memory_accesses[usize::from(transfer.source)]
            .sends
            .record(
                &transfer.source_elements,
                timing.payload_start,
                timing.sender_horizon,
            );
        let activity = ExchangeActivity {
            fanout: transfer.destinations.len() as u16,
            paired: transfer.width == ExchangeItemWidth::Paired64,
            transfer: u32::try_from(index).map_err(|_| ExchangeLoweringError::Overflow)?,
            kind: ExchangeActivityKind::Send,
            start_cycle: timing.payload_start,
            end_cycle: payload_end,
            memory_end_cycle: timing.sender_horizon,
            address: transfer.source_address(),
            words: transfer.words,
        };
        self.activities[usize::from(transfer.source)].push(activity);
        if let Some(tile) = transfer.reserved_source {
            self.activities[usize::from(tile)].push(ExchangeActivity {
                kind: ExchangeActivityKind::PartnerBusy,
                end_cycle: timing.sender_horizon,
                ..activity
            });
        }
        let mut completion = payload_end;
        for (&(tile, address), receiver) in transfer.destinations.iter().zip(&timing.receivers) {
            self.memory_accesses[usize::from(tile)].receives.record(
                &effective_memory_elements(address, transfer.words),
                receiver.payload_start,
                receiver.horizon,
            );
            self.activities[usize::from(tile)].push(ExchangeActivity {
                kind: ExchangeActivityKind::Receive,
                start_cycle: receiver.payload_start,
                end_cycle: receiver.payload_end,
                memory_end_cycle: receiver.horizon,
                address,
                ..activity
            });
            self.tile_availability[usize::from(tile)].receive = receiver.payload_end;
            last_transfer[usize::from(tile)].receive = Some(index);
            completion = completion.max(receiver.payload_end);
        }
        self.tile_availability[usize::from(transfer.source)].send = timing.payload_end;
        if let Some(tile) = transfer.reserved_source {
            self.tile_availability[usize::from(tile)].send = timing.sender_horizon;
            last_transfer[usize::from(tile)].send = Some(index);
        }
        last_transfer[usize::from(transfer.source)].send = Some(index);
        self.order.push(index);
        self.timings[index] = Some(MaterializedTiming {
            end: completion,
            predecessor,
        });
        Ok(completion)
    }

    pub(super) fn into_phase(
        self,
        id: ExchangePhaseId,
        incoming_bases: Vec<u32>,
    ) -> Result<PhysicalExchangePhase, ExchangeLoweringError> {
        let encoded = self.builder.finish()?;
        if encoded.event_cycles != self.horizon {
            return Err(ExchangeLoweringError::Invariant(format!(
                "phase {} row horizon {} differs from scheduled horizon {}",
                id.index(),
                encoded.event_cycles,
                self.horizon
            )));
        }
        let tile_count = encoded.programs.len();
        Ok(PhysicalExchangePhase {
            id,
            active: encoded.programs.iter().map(Option::is_some).collect(),
            programs: encoded
                .programs
                .into_iter()
                .map(|program| program.unwrap_or_else(crate::exchange::EncodedRow::inactive))
                .collect(),
            incoming_bases,
            tile_event_cycles: encoded.tile_event_cycles,
            event_cycles: self.horizon,
            activities: self.activities,
            repeat_patches: vec![Vec::new(); tile_count],
            outgoing_bases: vec![None; tile_count],
        })
    }

    pub(super) fn finish_horizon(&mut self) {
        self.horizon = self.builder.event_cycles();
        // The row builder can fill earlier gaps. Profiles follow execution
        // order; relocation uses the transfer identity retained by encoding.
        for activities in &mut self.activities {
            activities.sort_by_key(|activity| activity.start_cycle);
        }
    }
}

pub(super) fn materialize_valid_schedule_order(
    topology: &Topology,
    problem: &SchedulingProblem<'_>,
    incoming_bases: &[u32],
    receive_counts: &[usize],
    order: &[usize],
) -> Result<MaterializedSchedule, ExchangeLoweringError> {
    match materialize_schedule_order(
        topology,
        problem,
        incoming_bases,
        receive_counts,
        order,
        false,
    ) {
        Ok(schedule) => Ok(schedule),
        Err(ExchangeLoweringError::Exchange(crate::exchange::ExchangeError::Schedule(
            "SENDPICP instruction alignment",
        ))) => materialize_schedule_order(
            topology,
            problem,
            incoming_bases,
            receive_counts,
            order,
            true,
        ),
        Err(error) => Err(error),
    }
}

pub(super) fn materialize_schedule_order(
    topology: &Topology,
    problem: &SchedulingProblem<'_>,
    incoming_bases: &[u32],
    receive_counts: &[usize],
    order: &[usize],
    validate_encoding: bool,
) -> Result<MaterializedSchedule, ExchangeLoweringError> {
    let pending = problem.transfers;
    let tile_count = problem.tile_count;
    if order.len() != pending.len() {
        return Err(ExchangeLoweringError::Overflow);
    }
    let mut schedule = MaterializedSchedule::new(tile_count, pending);
    let mut last_transfer = vec![TilePredecessor::default(); usize::from(tile_count)];
    let predecessors = &problem.predecessors;
    let mut completion = vec![None; pending.len()];
    for &index in order {
        if index >= pending.len()
            || completion[index].is_some()
            || predecessors[index]
                .iter()
                .any(|&before| completion[before].is_none())
        {
            return Err(ExchangeLoweringError::Invariant(
                "exchange order is not a topological permutation".into(),
            ));
        }
        let dependency_ready = predecessors[index]
            .iter()
            .filter_map(|predecessor| completion[*predecessor])
            .max()
            .unwrap_or(0);
        completion[index] = Some(schedule.append(
            topology,
            pending,
            incoming_bases,
            receive_counts,
            index,
            dependency_ready,
            validate_encoding,
            &mut last_transfer,
        )?);
    }
    schedule.finish_horizon();
    if schedule_encoding_is_valid(&schedule)? {
        Ok(schedule)
    } else {
        Err(crate::exchange::ExchangeError::Schedule("SENDPICP instruction alignment").into())
    }
}

pub(super) fn schedule_encoding_is_valid(
    schedule: &MaterializedSchedule,
) -> Result<bool, ExchangeLoweringError> {
    match schedule.builder.finish() {
        Ok(_) => Ok(true),
        Err(crate::exchange::ExchangeError::Schedule("SENDPICP instruction alignment")) => {
            Ok(false)
        }
        Err(error) => Err(error.into()),
    }
}

pub(super) fn encoded_row_storage(
    schedule: &MaterializedSchedule,
) -> Result<(usize, usize), ExchangeLoweringError> {
    let encoded = schedule.builder.finish()?;
    Ok(encoded
        .programs
        .iter()
        .fold((0, 0), |(maximum, total), row| {
            let words = row.as_ref().map_or(0, |row| row.words().len());
            (maximum.max(words), total + words)
        }))
}

pub(super) fn endpoint_work_lower_bound(pending: &[PendingTransfer], tile_count: u16) -> u32 {
    let mut send_words = vec![0u64; usize::from(tile_count)];
    let mut receive_words = vec![0u64; usize::from(tile_count)];
    for transfer in pending {
        let items = u64::from(transfer.item_count().unwrap_or(transfer.words));
        send_words[usize::from(transfer.source)] += items;
        if let Some(tile) = transfer.reserved_source {
            send_words[usize::from(tile)] += items;
        }
        for &(tile, _) in &transfer.destinations {
            receive_words[usize::from(tile)] += items;
        }
    }
    send_words
        .into_iter()
        .zip(receive_words)
        .map(|(send, receive)| send.max(receive))
        .max()
        .unwrap_or(0)
        .min(u64::from(u32::MAX)) as u32
}

fn append_transfer(
    topology: &Topology,
    memory_accesses: &[TileMemorySchedule],
    incoming_bases: &[u32],
    receive_counts: &[usize],
    transfer: &PendingTransfer,
    message: u32,
    requested_offset: u32,
    builder: &mut PhaseProgramBuilder,
    validate_encoding: bool,
) -> Result<PhaseTransferTiming, ExchangeLoweringError> {
    let source = transfer.source;
    let source_address = transfer.source_address();
    let destinations = &transfer.destinations;
    let source_elements = &transfer.source_elements;
    let words = transfer.words;
    let width = transfer.width;
    if words == 0 || source_elements.is_empty() {
        return Err(ExchangeLoweringError::UnalignedPayload);
    }
    if destinations.iter().any(|&(tile, address)| {
        tile == source
            && (destinations.len() < 2
                || effective_memory_elements(address, words)
                    .iter()
                    .any(|element| source_elements.contains(element)))
    }) {
        return Err(ExchangeLoweringError::Invariant(
            "multicast loopback requires separate source and destination memory elements".into(),
        ));
    }
    let tiles = destinations.iter().map(|entry| entry.0).collect::<Vec<_>>();
    let item_count = width.item_count(words)?;
    let reserved_tile = match width {
        ExchangeItemWidth::Word32 => None,
        ExchangeItemWidth::Paired64 => Some(topology.paired_logical(source)?),
    };
    let reserved_tiles = reserved_tile.as_slice();
    let point_receiver = width == ExchangeItemWidth::Word32
        && destinations.first().is_some_and(|&(tile, address)| {
            destinations.len() == 1
                && receive_counts[usize::from(tile)] == 1
                && incoming_bases[usize::from(tile)] == address
        });
    let mut plan = if width == ExchangeItemWidth::Paired64 {
        crate::exchange::paired_multicast(&topology, source, &tiles, item_count)?
    } else if point_receiver {
        point_to_point(&topology, source, tiles[0], words)?
    } else {
        crate::exchange::multicast(&topology, source, &tiles, item_count, 0)?
    };
    patch_sender_address(&mut plan.sender, source_address)?;
    if !point_receiver {
        for (row, (_, address)) in plan.receivers.iter_mut().zip(destinations) {
            patch_receiver_address(row, *address)?;
        }
    }
    plan.sender.message = message;
    let mut schedule_offset = requested_offset;
    loop {
        let previous = schedule_offset;
        schedule_offset = if validate_encoding {
            builder.earliest_transfer_offset(
                source,
                reserved_tiles,
                &tiles,
                &plan,
                item_count,
                schedule_offset,
            )?
        } else {
            builder.earliest_transfer_offset_deferred(
                source,
                reserved_tiles,
                &tiles,
                &plan,
                item_count,
                schedule_offset,
            )?
        };
        let timing =
            builder.transfer_timing_at(source, &tiles, &plan, schedule_offset, item_count)?;
        schedule_offset = schedule_offset.max(memory_safe_transfer_offset(
            memory_accesses,
            source,
            destinations,
            source_elements,
            words,
            timing.payload_start,
            timing.sender_horizon,
            &timing.receivers,
            schedule_offset,
        )?);
        if schedule_offset == previous {
            break;
        }
    }
    Ok(builder.append_transfer_at(
        source,
        reserved_tiles,
        &tiles,
        &plan,
        schedule_offset,
        item_count,
    )?)
}

fn memory_safe_transfer_offset(
    memory_accesses: &[TileMemorySchedule],
    source: u16,
    destinations: &[(u16, u32)],
    source_elements: &[u32],
    words: u32,
    payload_start: u32,
    payload_end: u32,
    receivers: &[ScheduledPayloadTiming],
    schedule_offset: u32,
) -> Result<u32, ExchangeLoweringError> {
    let mut safe_offset = schedule_offset;
    let source_clash = memory_accesses[usize::from(source)].receives.conflict_end(
        source_elements,
        payload_start,
        payload_end,
    );
    let receiver_clash = destinations
        .iter()
        .zip(receivers)
        .filter_map(|(&(tile, address), receiver)| {
            memory_accesses[usize::from(tile)]
                .sends
                .conflict_end(
                    &effective_memory_elements(address, words),
                    receiver.payload_start,
                    receiver.horizon,
                )
                .map(|conflict| conflict.saturating_sub(receiver.payload_start))
        })
        .max();
    let source_delay = source_clash.map(|end| end.saturating_sub(payload_start));
    if let Some(delay) = source_delay.into_iter().chain(receiver_clash).max() {
        safe_offset = safe_offset
            .checked_add(delay)
            .ok_or(ExchangeLoweringError::Overflow)?;
    }
    Ok(safe_offset)
}
