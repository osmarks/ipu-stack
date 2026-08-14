//! Physical exchange programs generated from logical shard transfers.

use crate::{
    ExchangePhaseId, LogicalExchange, LowProgram, LowShardId, Placement, ShardDefinition,
    logical_view_byte_spans, view_byte_spans,
};
use ipu_exchange::{
    MAX_TRANSFER_WORDS, MulticastPlan, PlanProgramBuilder, RETURN_M10_INSTRUCTION, Topology,
    finalize_point_receiver, patch_receiver_address, patch_sender_address,
    patch_sender_instruction, sender_instruction_offsets,
};
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

#[cfg(test)]
use ipu_exchange::plan_event_cycles;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalExchangePhase {
    pub id: ExchangePhaseId,
    /// Whether each logical tile participates in this phase's timed program.
    pub active: Vec<bool>,
    /// Synchronization-free timed supervisor program indexed by logical tile.
    pub programs: Vec<Vec<u32>>,
    /// Per-tile base used by point-to-point receive rows in this phase.
    pub incoming_bases: Vec<u32>,
    /// Final local exchange event indexed by logical tile. Inactive tiles use zero.
    pub tile_event_cycles: Vec<u32>,
    pub event_cycles: u32,
    /// Static per-tile role intervals on the exchange event timeline.
    pub activities: Vec<Vec<ExchangeActivity>>,
    /// Per-tile replacement words which specialize a reusable row for each
    /// structured-repeat iteration.
    pub repeat_patches: Vec<Vec<ExchangeRowPatch>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExchangeRowPatch {
    pub word_offset: u32,
    pub values: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExchangeActivity {
    pub kind: ExchangeActivityKind,
    pub start_cycle: u32,
    pub end_cycle: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExchangeActivityKind {
    Send,
    Receive,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExchangeLoweringOptions {
    pub diagnostics: bool,
}

#[derive(Clone, Debug, Default)]
struct TilePressure {
    send_roles: u32,
    receive_roles: u32,
    send_words: u64,
    receive_words: u64,
    last_transfer: Option<usize>,
}

#[derive(Clone, Debug)]
struct ScheduledTransferDiagnostic {
    source: u16,
    receiver_count: usize,
    words: u32,
    start: u32,
    end: u32,
    blocking_tile: u16,
    predecessor: Option<usize>,
}

struct PhaseDiagnostics {
    tiles: Vec<TilePressure>,
    transfers: Vec<ScheduledTransferDiagnostic>,
    source_words: u64,
    destination_words: u64,
    multicast_chunks: usize,
    maximum_fanout: usize,
    maximum_endpoint_roles: usize,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ExchangeLoweringError {
    #[error(transparent)]
    Exchange(#[from] ipu_exchange::ExchangeError),
    #[error(transparent)]
    Storage(#[from] crate::StorageError),
    #[error("exchange refers to an unplaced shard")]
    UnplacedShard,
    #[error("exchange destination is not writable")]
    InvalidDestination,
    #[error("exchange payload is not a nonempty whole number of words")]
    UnalignedPayload,
    #[error("exchange source and destination byte counts differ")]
    SizeMismatch,
    #[error("exchange address arithmetic overflowed")]
    Overflow,
    #[error("structured-repeat exchange rows have incompatible shapes")]
    IncompatibleRepeatRows,
}

pub fn lower_exchanges(
    program: &LowProgram,
    placement: &Placement,
    topology: &Topology,
    options: ExchangeLoweringOptions,
) -> Result<Vec<PhysicalExchangePhase>, ExchangeLoweringError> {
    lower_static_exchanges(program, placement, topology, options)
}

fn lower_static_exchanges(
    program: &LowProgram,
    placement: &Placement,
    topology: &Topology,
    options: ExchangeLoweringOptions,
) -> Result<Vec<PhysicalExchangePhase>, ExchangeLoweringError> {
    let mut repeat_inputs = BTreeMap::<LowShardId, Vec<LowShardId>>::new();
    for repeat in &program.repeat_runs {
        for iterated in &repeat.iterated {
            match repeat_inputs.entry(iterated.argument) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(iterated.inputs.clone());
                }
                std::collections::btree_map::Entry::Occupied(entry)
                    if entry.get() != &iterated.inputs =>
                {
                    return Err(ExchangeLoweringError::IncompatibleRepeatRows);
                }
                std::collections::btree_map::Entry::Occupied(_) => {}
            }
        }
    }
    program
        .exchange_phases
        .iter()
        .map(|phase| {
            let pending = phase
                .transfers
                .par_iter()
                .map(|transfer| prepare_transfer(program, placement, transfer))
                .collect::<Result<Vec<_>, ExchangeLoweringError>>()?
                .into_iter()
                .flatten()
                .collect();
            let pending = coalesce_pending_transfers(pending);
            let mut receive_counts = vec![0usize; usize::from(program.tile_count)];
            for transfer in &pending {
                for &(tile, _) in &transfer.destinations {
                    receive_counts[usize::from(tile)] += 1;
                }
            }
            let mut incoming_bases = vec![None::<u32>; usize::from(program.tile_count)];
            for transfer in &pending {
                if let [(tile, address)] = transfer.destinations.as_slice() {
                    if receive_counts[usize::from(*tile)] == 1 {
                        incoming_bases[usize::from(*tile)] = Some(*address);
                    }
                }
            }
            let incoming_bases = incoming_bases
                .into_iter()
                .map(|base| base.unwrap_or(0))
                .collect::<Vec<_>>();
            let mut destination_multiplicity = BTreeMap::new();
            for transfer in &pending {
                for &(tile, address) in &transfer.destinations {
                    *destination_multiplicity
                        .entry((tile, address, transfer.words))
                        .or_insert(0usize) += 1;
                }
            }
            let maximum_identical_destinations = destination_multiplicity
                .values()
                .copied()
                .max()
                .unwrap_or(0);
            if pending.len() > 1_000 || maximum_identical_destinations > 1 {
                tracing::info!(
                    phase = phase.id.index(),
                    transfers = pending.len(),
                    maximum_identical_destinations,
                    "prepared large physical exchange phase"
                );
            }
            let (split, atomic) = rayon::join(
                || {
                    materialize_greedy_schedule(
                        topology,
                        &pending,
                        &incoming_bases,
                        &receive_counts,
                        program.tile_count,
                        AvailabilityPolicy::SplitRoles,
                    )
                },
                || {
                    materialize_greedy_schedule(
                        topology,
                        &pending,
                        &incoming_bases,
                        &receive_counts,
                        program.tile_count,
                        AvailabilityPolicy::AtomicEndpoints,
                    )
                },
            );
            let split = split?;
            let atomic = atomic?;
            let split_horizon = schedule_score(&split);
            let atomic_horizon = schedule_score(&atomic);
            let endpoint_lower_bound = endpoint_work_lower_bound(&pending, program.tile_count);
            let (mut schedule, mut selected_kind) = if atomic_horizon < split_horizon {
                (atomic, "atomic")
            } else {
                (split, "split")
            };
            let mut neighborhood_improvements = 0usize;
            loop {
                let repaired_order =
                    critical_neighborhood_order(&pending, program.tile_count, &schedule);
                if repaired_order == schedule.order {
                    break;
                }
                let repaired = materialize_schedule_order(
                    topology,
                    &pending,
                    &incoming_bases,
                    &receive_counts,
                    program.tile_count,
                    &repaired_order,
                )?;
                if schedule_score(&repaired) >= schedule_score(&schedule) {
                    break;
                }
                schedule = repaired;
                selected_kind = "critical-neighborhood";
                neighborhood_improvements += 1;
            }
            if pending.len() > 1_000 {
                tracing::info!(
                    phase = phase.id.index(),
                    split_horizon,
                    atomic_horizon,
                    selected_horizon = schedule.horizon,
                    endpoint_lower_bound,
                    lower_bound_gap = schedule.horizon.saturating_sub(endpoint_lower_bound),
                    selected_kind,
                    neighborhood_improvements,
                    "optimized physical exchange schedule"
                );
            }
            let MaterializedSchedule {
                mut builders,
                horizon,
                tile_availability,
                activities,
                scheduled_sends,
                order,
                timings,
            } = schedule;
            let mut diagnostics = options
                .diagnostics
                .then(|| PhaseDiagnostics::new(program.tile_count));
            if let Some(diagnostics) = &mut diagnostics {
                let mut endpoint_roles = vec![0usize; usize::from(program.tile_count)];
                for tile in pending.iter().flat_map(PendingTransfer::tiles) {
                    endpoint_roles[usize::from(tile)] += 1;
                }
                diagnostics.maximum_endpoint_roles = endpoint_roles.into_iter().max().unwrap_or(0);
                for &index in &order {
                    let transfer = &pending[index];
                    let timing = timings[index].ok_or(ExchangeLoweringError::Overflow)?;
                    diagnostics.record(
                        transfer.source,
                        transfer.destinations.iter().map(|entry| entry.0),
                        transfer.words,
                        timing.start,
                        timing.end,
                        timing.blocking_tile,
                    );
                }
            }
            if let Some(diagnostics) = diagnostics {
                let diagnostic_availability = tile_availability
                    .iter()
                    .map(|availability| availability.send.max(availability.receive))
                    .collect::<Vec<_>>();
                diagnostics.emit(
                    phase.id.index(),
                    &phase.provenance,
                    horizon,
                    &diagnostic_availability,
                    &builders,
                );
            }
            let mut active = Vec::with_capacity(usize::from(program.tile_count));
            let mut tile_event_cycles = Vec::with_capacity(usize::from(program.tile_count));
            let programs = (0..program.tile_count)
                .map(|tile| match builders.remove(&tile) {
                    Some(builder) => {
                        active.push(true);
                        tile_event_cycles.push(builder.event_cycles());
                        Ok(builder.finish()?)
                    }
                    None => {
                        active.push(false);
                        tile_event_cycles.push(0);
                        Ok(inactive_exchange_program())
                    }
                })
                .collect::<Result<Vec<_>, ipu_exchange::ExchangeError>>()?;
            let repeat_patches = programs
                .iter()
                .enumerate()
                .map(|(tile, program)| {
                    let offsets = sender_instruction_offsets(program).collect::<Vec<_>>();
                    if offsets.len() != scheduled_sends[tile].len() {
                        return Err(ExchangeLoweringError::IncompatibleRepeatRows);
                    }
                    offsets
                        .into_iter()
                        .zip(&scheduled_sends[tile])
                        .filter_map(|(word_offset, &(source_shard, source_offset))| {
                            repeat_inputs.get(&source_shard).map(|inputs| {
                                let values = inputs
                                    .iter()
                                    .map(|input| {
                                        let address = placement
                                            .shard_addresses
                                            .get(input)
                                            .copied()
                                            .ok_or(ExchangeLoweringError::UnplacedShard)?
                                            .checked_add(source_offset)
                                            .ok_or(ExchangeLoweringError::Overflow)?;
                                        let mut instruction = program[word_offset];
                                        patch_sender_instruction(&mut instruction, address)?;
                                        Ok(instruction)
                                    })
                                    .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
                                if values.first() != Some(&program[word_offset]) {
                                    return Err(ExchangeLoweringError::IncompatibleRepeatRows);
                                }
                                Ok(ExchangeRowPatch {
                                    word_offset: u32::try_from(word_offset)
                                        .map_err(|_| ExchangeLoweringError::Overflow)?,
                                    values,
                                })
                            })
                        })
                        .collect::<Result<Vec<_>, ExchangeLoweringError>>()
                })
                .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
            if pending.len() > 1_000 {
                let (tile, words) = programs
                    .iter()
                    .enumerate()
                    .map(|(tile, row)| (tile, row.len()))
                    .max_by_key(|entry| entry.1)
                    .unwrap_or((0, 0));
                tracing::info!(
                    phase = phase.id.index(),
                    tile,
                    row_words = words,
                    horizon,
                    "finished large physical exchange phase"
                );
            }
            Ok(PhysicalExchangePhase {
                id: phase.id,
                active,
                programs,
                incoming_bases,
                tile_event_cycles,
                event_cycles: horizon,
                activities,
                repeat_patches,
            })
        })
        .collect()
}

fn prepare_transfer(
    program: &LowProgram,
    placement: &Placement,
    transfer: &LogicalExchange,
) -> Result<Vec<PendingTransfer>, ExchangeLoweringError> {
    let source = &program.shards[transfer.source.shard.index() as usize];
    let logical_order = transfer.order == crate::low::ExchangeOrder::Semantic
        && transfer.destinations.iter().any(|view| {
            program.shards[view.shard.index() as usize]
                .tensor_type
                .format
                .layout
                .order
                != source.tensor_type.format.layout.order
        });
    let source_base = placement
        .shard_addresses
        .get(&source.id)
        .copied()
        .ok_or(ExchangeLoweringError::UnplacedShard)?;
    let destinations = transfer
        .destinations
        .iter()
        .map(|view| {
            let shard = &program.shards[view.shard.index() as usize];
            if matches!(shard.definition, ShardDefinition::Alias(_)) {
                return Err(ExchangeLoweringError::InvalidDestination);
            }
            Ok((
                shard.tile,
                placement
                    .shard_addresses
                    .get(&view.shard)
                    .copied()
                    .ok_or(ExchangeLoweringError::UnplacedShard)?,
                if logical_order {
                    logical_view_byte_spans(shard, view)?
                } else {
                    view_byte_spans(shard, view)?
                },
            ))
        })
        .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
    if destinations.is_empty() {
        return Err(ExchangeLoweringError::SizeMismatch);
    }
    let source_spans = if logical_order {
        logical_view_byte_spans(source, &transfer.source)?
    } else {
        view_byte_spans(source, &transfer.source)?
    };
    let source_bytes = source_spans.iter().try_fold(0u32, |total, span| {
        total
            .checked_add(span.bytes)
            .ok_or(ExchangeLoweringError::Overflow)
    })?;
    for (_, _, spans) in &destinations {
        let destination_bytes = spans.iter().try_fold(0u32, |total, span| {
            total
                .checked_add(span.bytes)
                .ok_or(ExchangeLoweringError::Overflow)
        })?;
        if destination_bytes != source_bytes {
            return Err(ExchangeLoweringError::SizeMismatch);
        }
    }
    let mut pending = Vec::new();
    let mut source_index = 0usize;
    let mut source_offset = 0u32;
    let mut destination_positions = vec![(0usize, 0u32); destinations.len()];
    while source_index < source_spans.len() {
        let source_span = source_spans[source_index];
        if source_span.bytes == 0 || source_span.offset & 0b11 != 0 {
            return Err(ExchangeLoweringError::UnalignedPayload);
        }
        let mut chunk_bytes = (source_span.bytes - source_offset).min(
            MAX_TRANSFER_WORDS
                .checked_mul(4)
                .ok_or(ExchangeLoweringError::Overflow)?,
        );
        for ((index, offset), (_, _, spans)) in destination_positions.iter().zip(&destinations) {
            let span = spans
                .get(*index)
                .ok_or(ExchangeLoweringError::SizeMismatch)?;
            if span.offset & 0b11 != 0 {
                return Err(ExchangeLoweringError::UnalignedPayload);
            }
            chunk_bytes = chunk_bytes.min(span.bytes - *offset);
        }
        if chunk_bytes == 0 || chunk_bytes & 0b11 != 0 {
            return Err(ExchangeLoweringError::UnalignedPayload);
        }
        let source_address = source_base
            .checked_add(source_span.offset)
            .and_then(|address| address.checked_add(source_offset))
            .ok_or(ExchangeLoweringError::Overflow)?;
        let destination_entries = destinations
            .iter()
            .zip(&destination_positions)
            .map(|((tile, base, spans), (index, offset))| {
                let span = spans
                    .get(*index)
                    .ok_or(ExchangeLoweringError::SizeMismatch)?;
                Ok((
                    *tile,
                    base.checked_add(span.offset)
                        .and_then(|address| address.checked_add(*offset))
                        .ok_or(ExchangeLoweringError::Overflow)?,
                ))
            })
            .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
        pending.push(PendingTransfer {
            source: source.tile,
            source_shard: source.id,
            source_offset: source_span
                .offset
                .checked_add(source_offset)
                .ok_or(ExchangeLoweringError::Overflow)?,
            destinations: destination_entries,
            source_address,
            words: chunk_bytes / 4,
        });
        source_offset += chunk_bytes;
        if source_offset == source_span.bytes {
            source_index += 1;
            source_offset = 0;
        }
        for ((index, offset), (_, _, spans)) in destination_positions.iter_mut().zip(&destinations)
        {
            *offset += chunk_bytes;
            if *offset == spans[*index].bytes {
                *index += 1;
                *offset = 0;
            }
        }
    }
    if destination_positions
        .iter()
        .zip(&destinations)
        .any(|((index, offset), (_, _, spans))| *index != spans.len() || *offset != 0)
    {
        return Err(ExchangeLoweringError::SizeMismatch);
    }
    Ok(pending)
}

impl PhaseDiagnostics {
    fn new(tile_count: u16) -> Self {
        Self {
            tiles: vec![TilePressure::default(); usize::from(tile_count)],
            transfers: Vec::new(),
            source_words: 0,
            destination_words: 0,
            multicast_chunks: 0,
            maximum_fanout: 0,
            maximum_endpoint_roles: 0,
        }
    }

    fn record(
        &mut self,
        source: u16,
        destinations: impl Iterator<Item = u16>,
        words: u32,
        start: u32,
        end: u32,
        blocking_tile: u16,
    ) {
        let id = self.transfers.len();
        let predecessor = self.tiles[usize::from(blocking_tile)].last_transfer;
        let destinations = destinations.collect::<Vec<_>>();
        let source_pressure = &mut self.tiles[usize::from(source)];
        source_pressure.send_roles += 1;
        source_pressure.send_words += u64::from(words);
        source_pressure.last_transfer = Some(id);
        for &tile in &destinations {
            let pressure = &mut self.tiles[usize::from(tile)];
            pressure.receive_roles += 1;
            pressure.receive_words += u64::from(words);
            pressure.last_transfer = Some(id);
        }
        self.source_words += u64::from(words);
        self.destination_words += u64::from(words) * destinations.len() as u64;
        self.multicast_chunks += usize::from(destinations.len() > 1);
        self.maximum_fanout = self.maximum_fanout.max(destinations.len());
        self.transfers.push(ScheduledTransferDiagnostic {
            source,
            receiver_count: destinations.len(),
            words,
            start,
            end,
            blocking_tile,
            predecessor,
        });
    }

    fn emit(
        &self,
        phase: u32,
        provenance: &crate::WorkProvenance,
        horizon: u32,
        tile_availability: &[u32],
        builders: &BTreeMap<u16, PlanProgramBuilder>,
    ) {
        let role_word_lower_bound = self
            .tiles
            .iter()
            .map(|tile| tile.send_words + tile.receive_words)
            .max()
            .unwrap_or(0);
        let mut busiest_tiles = self
            .tiles
            .iter()
            .enumerate()
            .filter(|(_, tile)| tile.send_roles != 0 || tile.receive_roles != 0)
            .map(|(tile, pressure)| {
                let encoded_end = builders
                    .get(&(tile as u16))
                    .map_or(0, PlanProgramBuilder::event_cycles);
                (
                    tile as u16,
                    pressure.send_roles,
                    pressure.receive_roles,
                    pressure.send_words,
                    pressure.receive_words,
                    tile_availability[tile],
                    encoded_end,
                    horizon.saturating_sub(encoded_end),
                )
            })
            .collect::<Vec<_>>();
        busiest_tiles
            .sort_unstable_by_key(|tile| (Reverse(tile.3 + tile.4), Reverse(tile.5), tile.0));
        busiest_tiles.truncate(8);

        let active_builders = builders.len() as u64;
        let total_final_padding = builders
            .values()
            .map(|builder| u64::from(horizon.saturating_sub(builder.event_cycles())))
            .sum::<u64>();
        let maximum_final_padding = builders
            .values()
            .map(|builder| horizon.saturating_sub(builder.event_cycles()))
            .max()
            .unwrap_or(0);
        let maximum_scheduled_wait = self
            .tiles
            .iter()
            .enumerate()
            .map(|(tile, pressure)| {
                let payload = pressure.send_words + pressure.receive_words;
                u64::from(tile_availability[tile]).saturating_sub(payload)
            })
            .max()
            .unwrap_or(0);

        let critical_transfer = self
            .transfers
            .iter()
            .enumerate()
            .max_by_key(|(_, transfer)| transfer.end)
            .map(|(id, _)| id);
        let mut critical_chain = Vec::new();
        let mut cursor = critical_transfer;
        while let Some(id) = cursor {
            critical_chain.push(id);
            cursor = self.transfers[id].predecessor;
        }
        critical_chain.reverse();
        let critical_chain_length = critical_chain.len();
        let critical_chain_tail = critical_chain
            .iter()
            .rev()
            .take(12)
            .rev()
            .map(|&id| {
                let transfer = &self.transfers[id];
                (
                    id,
                    transfer.source,
                    transfer.receiver_count,
                    transfer.words,
                    transfer.start,
                    transfer.end,
                    transfer.blocking_tile,
                )
            })
            .collect::<Vec<_>>();

        tracing::info!(
            phase,
            ?provenance,
            scheduled_chunks = self.transfers.len(),
            multicast_chunks = self.multicast_chunks,
            maximum_fanout = self.maximum_fanout,
            maximum_endpoint_roles = self.maximum_endpoint_roles,
            source_words = self.source_words,
            destination_words = self.destination_words,
            role_word_lower_bound_cycles = role_word_lower_bound,
            scheduled_horizon_cycles = horizon,
            scheduler_excess_cycles = u64::from(horizon).saturating_sub(role_word_lower_bound),
            maximum_scheduled_wait_cycles = maximum_scheduled_wait,
            mean_final_padding_cycles = if active_builders == 0 {
                0
            } else {
                total_final_padding / active_builders
            },
            maximum_final_padding_cycles = maximum_final_padding,
            critical_chain_length,
            ?critical_chain_tail,
            ?busiest_tiles,
            "exchange scheduler diagnostics"
        );
    }
}

struct PendingTransfer {
    source: u16,
    source_shard: LowShardId,
    source_offset: u32,
    destinations: Vec<(u16, u32)>,
    source_address: u32,
    words: u32,
}

impl PendingTransfer {
    fn tiles(&self) -> impl Iterator<Item = u16> + '_ {
        std::iter::once(self.source).chain(self.destinations.iter().map(|entry| entry.0))
    }
}

fn coalesce_pending_transfers(transfers: Vec<PendingTransfer>) -> Vec<PendingTransfer> {
    let mut merged = Vec::<PendingTransfer>::with_capacity(transfers.len());
    for transfer in transfers {
        let Some(previous) = merged.last_mut() else {
            merged.push(transfer);
            continue;
        };
        let previous_bytes = previous.words * 4;
        let combined_words = previous.words.checked_add(transfer.words);
        let contiguous = previous.source == transfer.source
            && previous.source_shard == transfer.source_shard
            && previous
                .source_offset
                .checked_add(previous_bytes)
                .is_some_and(|end| end == transfer.source_offset)
            && previous.destinations.len() == transfer.destinations.len()
            && previous
                .source_address
                .checked_add(previous_bytes)
                .is_some_and(|end| end == transfer.source_address)
            && previous
                .destinations
                .iter()
                .zip(&transfer.destinations)
                .all(
                    |(&(left_tile, left_address), &(right_tile, right_address))| {
                        left_tile == right_tile
                            && left_address
                                .checked_add(previous_bytes)
                                .is_some_and(|end| end == right_address)
                    },
                );
        if contiguous && combined_words.is_some_and(|words| words <= MAX_TRANSFER_WORDS) {
            previous.words = combined_words.expect("checked above");
        } else {
            merged.push(transfer);
        }
    }
    merged
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ReadyTransfer {
    earliest_start: Reverse<u32>,
    endpoint_pressure: u64,
    fanout: usize,
    words: u32,
    source: Reverse<u16>,
    index: Reverse<usize>,
}

/// Incrementally list-schedules dependency-ready multicast hyperedges. Heap
/// keys are lower bounds on their start time and are refreshed lazily as
/// shared endpoints become busy.
struct TransferScheduler<'a> {
    transfers: &'a [PendingTransfer],
    word_pressure: Vec<u64>,
    dependents: Vec<Vec<usize>>,
    indegrees: Vec<usize>,
    ready: BinaryHeap<ReadyTransfer>,
    completed: usize,
}

impl<'a> TransferScheduler<'a> {
    fn new(transfers: &'a [PendingTransfer], tile_count: u16) -> Self {
        let mut word_pressure = vec![0u64; usize::from(tile_count)];
        for transfer in transfers {
            for tile in transfer.tiles() {
                word_pressure[usize::from(tile)] += u64::from(transfer.words);
            }
        }

        let mut dependents = vec![Vec::new(); transfers.len()];
        let mut indegrees = vec![0usize; transfers.len()];
        for (before, after) in memory_dependencies(transfers, tile_count) {
            dependents[before].push(after);
            indegrees[after] += 1;
        }
        let mut scheduler = Self {
            transfers,
            word_pressure,
            dependents,
            indegrees,
            ready: BinaryHeap::new(),
            completed: 0,
        };
        for index in 0..transfers.len() {
            if scheduler.indegrees[index] == 0 {
                scheduler.push_ready(index, 0);
            }
        }
        scheduler
    }

    fn push_ready(&mut self, index: usize, earliest_start: u32) {
        let transfer = &self.transfers[index];
        let endpoint_pressure = transfer
            .tiles()
            // Bytes, rather than role count, approximate how long selecting
            // this hyperedge frees work on the phase's congested endpoints.
            .map(|tile| self.word_pressure[usize::from(tile)])
            .sum::<u64>();
        self.ready.push(ReadyTransfer {
            earliest_start: Reverse(earliest_start),
            endpoint_pressure,
            fanout: transfer.destinations.len(),
            words: transfer.words,
            source: Reverse(transfer.source),
            index: Reverse(index),
        });
    }

    fn next(&mut self, tile_availability: &[TileAvailability]) -> Option<usize> {
        loop {
            let candidate = self.ready.pop()?;
            let index = candidate.index.0;
            let transfer = &self.transfers[index];
            let earliest_start =
                std::iter::once(tile_availability[usize::from(transfer.source)].send)
                    .chain(
                        transfer
                            .destinations
                            .iter()
                            .map(|&(tile, _)| tile_availability[usize::from(tile)].receive),
                    )
                    .max()
                    .unwrap_or(0);
            if candidate.earliest_start.0 == earliest_start {
                return Some(index);
            }
            self.push_ready(index, earliest_start);
        }
    }

    fn complete(&mut self, index: usize) {
        self.completed += 1;
        let dependents = std::mem::take(&mut self.dependents[index]);
        for dependent in dependents {
            self.indegrees[dependent] -= 1;
            if self.indegrees[dependent] == 0 {
                self.push_ready(dependent, 0);
            }
        }
    }

    fn is_complete(&self) -> bool {
        self.completed == self.transfers.len()
    }
}

#[derive(Clone, Copy)]
struct TransferAccess {
    transfer: usize,
    start: u64,
    end: u64,
    write: bool,
}

#[derive(Clone, Default)]
struct AccessFrontier {
    last_write: Option<usize>,
    reads: Vec<usize>,
}

/// Preserves the original order of overlapping accesses while allowing the
/// scheduler to reorder transfers whose tile-memory effects are independent.
fn memory_dependencies(transfers: &[PendingTransfer], tile_count: u16) -> BTreeSet<(usize, usize)> {
    let mut accesses = vec![Vec::new(); usize::from(tile_count)];
    for (index, transfer) in transfers.iter().enumerate() {
        let bytes = u64::from(transfer.words) * 4;
        accesses[usize::from(transfer.source)].push(TransferAccess {
            transfer: index,
            start: u64::from(transfer.source_address),
            end: u64::from(transfer.source_address) + bytes,
            write: false,
        });
        for &(tile, address) in &transfer.destinations {
            accesses[usize::from(tile)].push(TransferAccess {
                transfer: index,
                start: u64::from(address),
                end: u64::from(address) + bytes,
                write: true,
            });
        }
    }

    let mut dependencies = BTreeSet::new();
    for tile_accesses in &accesses {
        let mut boundaries = tile_accesses
            .iter()
            .flat_map(|access| [access.start, access.end])
            .collect::<Vec<_>>();
        boundaries.sort_unstable();
        boundaries.dedup();
        let mut frontier = vec![AccessFrontier::default(); boundaries.len().saturating_sub(1)];
        for access in tile_accesses {
            let start = boundaries
                .binary_search(&access.start)
                .expect("access start is a boundary");
            let end = boundaries
                .binary_search(&access.end)
                .expect("access end is a boundary");
            for state in &mut frontier[start..end] {
                if let Some(previous) = state.last_write
                    && previous != access.transfer
                {
                    dependencies.insert((previous, access.transfer));
                }
                if access.write {
                    for &previous in &state.reads {
                        if previous != access.transfer {
                            dependencies.insert((previous, access.transfer));
                        }
                    }
                    state.reads.clear();
                    state.last_write = Some(access.transfer);
                } else if state.reads.last() != Some(&access.transfer) {
                    state.reads.push(access.transfer);
                }
            }
        }
    }
    dependencies
}

struct ScheduledTransfer<'a> {
    source: u16,
    destinations: &'a [(u16, u32)],
    source_address: u32,
    words: u32,
    schedule_offset: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct TileAvailability {
    send: u32,
    receive: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AvailabilityPolicy {
    SplitRoles,
    AtomicEndpoints,
}

#[derive(Clone, Copy, Debug)]
struct MaterializedTiming {
    start: u32,
    end: u32,
    blocking_tile: u16,
    predecessor: Option<usize>,
}

struct MaterializedSchedule {
    builders: BTreeMap<u16, PlanProgramBuilder>,
    horizon: u32,
    tile_availability: Vec<TileAvailability>,
    activities: Vec<Vec<ExchangeActivity>>,
    scheduled_sends: Vec<Vec<(LowShardId, u32)>>,
    order: Vec<usize>,
    timings: Vec<Option<MaterializedTiming>>,
}

impl MaterializedSchedule {
    fn new(tile_count: u16, transfer_count: usize) -> Self {
        Self {
            builders: BTreeMap::new(),
            horizon: 0,
            tile_availability: vec![TileAvailability::default(); usize::from(tile_count)],
            activities: vec![Vec::new(); usize::from(tile_count)],
            scheduled_sends: vec![Vec::new(); usize::from(tile_count)],
            order: Vec::with_capacity(transfer_count),
            timings: vec![None; transfer_count],
        }
    }

    fn append(
        &mut self,
        topology: &Topology,
        pending: &[PendingTransfer],
        incoming_bases: &[u32],
        receive_counts: &[usize],
        index: usize,
        policy: AvailabilityPolicy,
        last_transfer: &mut [Option<usize>],
    ) -> Result<(), ExchangeLoweringError> {
        let transfer = &pending[index];
        let (blocking_tile, latest_availability) = std::iter::once((
            transfer.source,
            self.tile_availability[usize::from(transfer.source)].send,
        ))
        .chain(
            transfer
                .destinations
                .iter()
                .map(|&(tile, _)| (tile, self.tile_availability[usize::from(tile)].receive)),
        )
        .max_by_key(|&(tile, availability)| (availability, Reverse(tile)))
        .unwrap_or((transfer.source, 0));
        let predecessor = last_transfer[usize::from(blocking_tile)];
        let timing = append_transfer(
            topology,
            incoming_bases,
            receive_counts,
            ScheduledTransfer {
                source: transfer.source,
                destinations: &transfer.destinations,
                source_address: transfer.source_address,
                words: transfer.words,
                schedule_offset: latest_availability,
            },
            &mut self.horizon,
            &mut self.builders,
        )?;
        let payload_end = timing
            .start
            .checked_add(transfer.words)
            .ok_or(ExchangeLoweringError::Overflow)?;
        self.scheduled_sends[usize::from(transfer.source)]
            .push((transfer.source_shard, transfer.source_offset));
        self.activities[usize::from(transfer.source)].push(ExchangeActivity {
            kind: ExchangeActivityKind::Send,
            start_cycle: timing.start,
            end_cycle: payload_end,
        });
        for &(tile, _) in &transfer.destinations {
            self.activities[usize::from(tile)].push(ExchangeActivity {
                kind: ExchangeActivityKind::Receive,
                start_cycle: timing.start,
                end_cycle: payload_end,
            });
        }
        match policy {
            AvailabilityPolicy::SplitRoles => {
                self.tile_availability[usize::from(transfer.source)] = TileAvailability {
                    // The sender remains reserved until every old receiver's
                    // mux selection has propagated through the fabric.
                    send: timing.end,
                    receive: timing.sender_end,
                };
                for (&(tile, _), &receiver_end) in
                    transfer.destinations.iter().zip(&timing.receiver_ends)
                {
                    self.tile_availability[usize::from(tile)] = TileAvailability {
                        send: receiver_end,
                        receive: payload_end,
                    };
                }
            }
            AvailabilityPolicy::AtomicEndpoints => {
                let next = timing
                    .end
                    .checked_add(1)
                    .ok_or(ExchangeLoweringError::Overflow)?;
                for tile in transfer.tiles() {
                    self.tile_availability[usize::from(tile)] = TileAvailability {
                        send: next,
                        receive: next,
                    };
                }
            }
        }
        for tile in transfer.tiles() {
            last_transfer[usize::from(tile)] = Some(index);
        }
        self.order.push(index);
        self.timings[index] = Some(MaterializedTiming {
            start: timing.start,
            end: timing.end,
            blocking_tile,
            predecessor,
        });
        Ok(())
    }

    fn finish_horizon(&mut self) {
        self.horizon = self
            .builders
            .values()
            .map(PlanProgramBuilder::event_cycles)
            .max()
            .unwrap_or(0);
    }
}

fn materialize_greedy_schedule(
    topology: &Topology,
    pending: &[PendingTransfer],
    incoming_bases: &[u32],
    receive_counts: &[usize],
    tile_count: u16,
    policy: AvailabilityPolicy,
) -> Result<MaterializedSchedule, ExchangeLoweringError> {
    let mut schedule = MaterializedSchedule::new(tile_count, pending.len());
    let mut scheduler = TransferScheduler::new(pending, tile_count);
    let mut last_transfer = vec![None; usize::from(tile_count)];
    while let Some(index) = scheduler.next(&schedule.tile_availability) {
        schedule.append(
            topology,
            pending,
            incoming_bases,
            receive_counts,
            index,
            policy,
            &mut last_transfer,
        )?;
        scheduler.complete(index);
    }
    debug_assert!(scheduler.is_complete());
    schedule.finish_horizon();
    Ok(schedule)
}

fn materialize_schedule_order(
    topology: &Topology,
    pending: &[PendingTransfer],
    incoming_bases: &[u32],
    receive_counts: &[usize],
    tile_count: u16,
    order: &[usize],
) -> Result<MaterializedSchedule, ExchangeLoweringError> {
    if order.len() != pending.len() {
        return Err(ExchangeLoweringError::Overflow);
    }
    let mut schedule = MaterializedSchedule::new(tile_count, pending.len());
    let mut last_transfer = vec![None; usize::from(tile_count)];
    for &index in order {
        schedule.append(
            topology,
            pending,
            incoming_bases,
            receive_counts,
            index,
            AvailabilityPolicy::SplitRoles,
            &mut last_transfer,
        )?;
    }
    schedule.finish_horizon();
    Ok(schedule)
}

fn schedule_score(schedule: &MaterializedSchedule) -> u32 {
    schedule.horizon
}

fn endpoint_work_lower_bound(pending: &[PendingTransfer], tile_count: u16) -> u32 {
    let mut words = vec![0u64; usize::from(tile_count)];
    for transfer in pending {
        words[usize::from(transfer.source)] += u64::from(transfer.words);
        for &(tile, _) in &transfer.destinations {
            words[usize::from(tile)] += u64::from(transfer.words);
        }
    }
    words
        .into_iter()
        .max()
        .unwrap_or(0)
        .min(u64::from(u32::MAX)) as u32
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RepairReady {
    earliest_start: Reverse<u32>,
    epoch: Reverse<usize>,
    contiguous_receivers: usize,
    in_neighborhood: bool,
    endpoint_pressure: u64,
    incumbent_rank: Reverse<usize>,
    index: Reverse<usize>,
}

fn repair_ready(
    index: usize,
    pending: &[PendingTransfer],
    availability: &[u32],
    next_receive_address: &[Option<u32>],
    word_pressure: &[u64],
    rank: &[usize],
    epoch_width: usize,
    neighborhood: &[bool],
) -> RepairReady {
    let transfer = &pending[index];
    let earliest_start = transfer
        .tiles()
        .map(|tile| availability[usize::from(tile)])
        .max()
        .unwrap_or(0);
    let contiguous_receivers = transfer
        .destinations
        .iter()
        .filter(|(tile, address)| next_receive_address[usize::from(*tile)] == Some(*address))
        .count();
    let endpoint_pressure = transfer
        .tiles()
        .map(|tile| word_pressure[usize::from(tile)])
        .sum();
    RepairReady {
        earliest_start: Reverse(earliest_start),
        epoch: Reverse(rank[index] / epoch_width),
        contiguous_receivers,
        in_neighborhood: neighborhood[index],
        endpoint_pressure,
        incumbent_rank: Reverse(rank[index]),
        index: Reverse(index),
    }
}

fn critical_neighborhood_order(
    pending: &[PendingTransfer],
    tile_count: u16,
    incumbent: &MaterializedSchedule,
) -> Vec<usize> {
    if pending.len() < 2 {
        return incumbent.order.clone();
    }
    let mut critical = vec![false; pending.len()];
    let mut cursor = incumbent
        .timings
        .iter()
        .enumerate()
        .filter_map(|(index, timing)| timing.map(|timing| (index, timing.end)))
        .max_by_key(|entry| entry.1)
        .map(|entry| entry.0);
    while let Some(index) = cursor {
        if std::mem::replace(&mut critical[index], true) {
            break;
        }
        cursor = incumbent.timings[index].and_then(|timing| timing.predecessor);
    }

    let mut tile_orders = vec![Vec::new(); usize::from(tile_count)];
    for &index in &incumbent.order {
        for tile in pending[index].tiles() {
            tile_orders[usize::from(tile)].push(index);
        }
    }
    let mut neighborhood = critical.clone();
    for order in &tile_orders {
        if order.is_empty() {
            continue;
        }
        let radius = order.len().isqrt().max(1);
        for position in 0..order.len() {
            if !critical[order[position]] {
                continue;
            }
            let start = position.saturating_sub(radius);
            let end = (position + radius + 1).min(order.len());
            for &index in &order[start..end] {
                neighborhood[index] = true;
            }
        }
    }
    if neighborhood.iter().filter(|selected| **selected).count() < 2 {
        return incumbent.order.clone();
    }

    let mut dependents = vec![Vec::new(); pending.len()];
    let mut indegrees = vec![0usize; pending.len()];
    for (before, after) in memory_dependencies(pending, tile_count) {
        dependents[before].push(after);
        indegrees[after] += 1;
    }
    let mut rank = vec![0usize; pending.len()];
    for (position, &index) in incumbent.order.iter().enumerate() {
        rank[index] = position;
    }
    let epoch_width = pending.len().isqrt().max(1);
    let mut word_pressure = vec![0u64; usize::from(tile_count)];
    for transfer in pending {
        for tile in transfer.tiles() {
            word_pressure[usize::from(tile)] += u64::from(transfer.words);
        }
    }
    let mut availability = vec![0u32; usize::from(tile_count)];
    let mut next_receive_address = vec![None; usize::from(tile_count)];
    let mut ready = BinaryHeap::new();
    for index in 0..pending.len() {
        if indegrees[index] == 0 {
            ready.push(repair_ready(
                index,
                pending,
                &availability,
                &next_receive_address,
                &word_pressure,
                &rank,
                epoch_width,
                &neighborhood,
            ));
        }
    }
    let mut order = Vec::with_capacity(pending.len());
    while let Some(candidate) = ready.pop() {
        let index = candidate.index.0;
        let refreshed = repair_ready(
            index,
            pending,
            &availability,
            &next_receive_address,
            &word_pressure,
            &rank,
            epoch_width,
            &neighborhood,
        );
        if refreshed != candidate {
            ready.push(refreshed);
            continue;
        }
        let transfer = &pending[index];
        let start = candidate.earliest_start.0;
        let end = match start.checked_add(transfer.words) {
            Some(end) => end,
            None => return incumbent.order.clone(),
        };
        availability[usize::from(transfer.source)] = end;
        next_receive_address[usize::from(transfer.source)] = None;
        let bytes = match transfer.words.checked_mul(4) {
            Some(bytes) => bytes,
            None => return incumbent.order.clone(),
        };
        for &(tile, address) in &transfer.destinations {
            availability[usize::from(tile)] = end;
            next_receive_address[usize::from(tile)] = address.checked_add(bytes);
        }
        for tile in transfer.tiles() {
            word_pressure[usize::from(tile)] =
                word_pressure[usize::from(tile)].saturating_sub(u64::from(transfer.words));
        }
        order.push(index);
        for dependent in std::mem::take(&mut dependents[index]) {
            indegrees[dependent] -= 1;
            if indegrees[dependent] == 0 {
                ready.push(repair_ready(
                    dependent,
                    pending,
                    &availability,
                    &next_receive_address,
                    &word_pressure,
                    &rank,
                    epoch_width,
                    &neighborhood,
                ));
            }
        }
    }
    if order.len() == pending.len() {
        order
    } else {
        incumbent.order.clone()
    }
}

fn append_transfer(
    topology: &Topology,
    incoming_bases: &[u32],
    receive_counts: &[usize],
    transfer: ScheduledTransfer<'_>,
    horizon: &mut u32,
    builders: &mut BTreeMap<u16, PlanProgramBuilder>,
) -> Result<ScheduledTransferTiming, ExchangeLoweringError> {
    let ScheduledTransfer {
        source,
        destinations,
        source_address,
        words,
        schedule_offset: requested_offset,
    } = transfer;
    if words == 0 {
        return Err(ExchangeLoweringError::UnalignedPayload);
    }
    let tiles = destinations.iter().map(|entry| entry.0).collect::<Vec<_>>();
    let point_receiver = destinations.first().is_some_and(|&(tile, address)| {
        destinations.len() == 1
            && receive_counts[usize::from(tile)] == 1
            && incoming_bases[usize::from(tile)] == address
    });
    let mut plan = if point_receiver {
        let point = topology.point_to_point(source, tiles[0], words)?;
        MulticastPlan {
            sender: point.sender,
            receivers: vec![finalize_point_receiver(
                &point.receiver,
                topology.physical(source)?,
            )?],
        }
    } else {
        topology.multicast(source, &tiles, words, 0)?
    };
    patch_sender_address(&mut plan.sender, source_address)?;
    if !point_receiver {
        for (row, (_, address)) in plan.receivers.iter_mut().zip(destinations) {
            patch_receiver_address(row, *address)?;
        }
    }
    let mut schedule_offset = requested_offset;
    loop {
        let previous_offset = schedule_offset;
        if let Some(builder) = builders.get(&source) {
            schedule_offset =
                builder.earliest_scheduled_row_offset(&plan.sender, schedule_offset)?;
        }
        for (&tile, row) in tiles.iter().zip(&plan.receivers) {
            if let Some(builder) = builders.get(&tile) {
                schedule_offset =
                    builder.earliest_scheduled_receiver_offset(row, schedule_offset)?;
            }
        }
        if schedule_offset == previous_offset {
            break;
        }
    }
    builders
        .entry(source)
        .or_default()
        .append_scheduled_row_at(&plan.sender, schedule_offset)?;
    for (&tile, row) in tiles.iter().zip(&plan.receivers) {
        builders
            .entry(tile)
            .or_default()
            .append_scheduled_receiver_row_at(row, schedule_offset, words)?;
    }
    let sender_end = builders
        .get(&source)
        .map(PlanProgramBuilder::event_cycles)
        .ok_or(ExchangeLoweringError::Overflow)?;
    let receiver_ends = tiles
        .iter()
        .map(|tile| {
            builders
                .get(tile)
                .map(PlanProgramBuilder::event_cycles)
                .ok_or(ExchangeLoweringError::Overflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let transfer_end = receiver_ends
        .iter()
        .copied()
        .chain(std::iter::once(sender_end))
        .max()
        .unwrap_or(schedule_offset);
    *horizon = (*horizon).max(transfer_end);
    Ok(ScheduledTransferTiming {
        start: schedule_offset,
        end: transfer_end,
        sender_end,
        receiver_ends,
    })
}

struct ScheduledTransferTiming {
    start: u32,
    end: u32,
    sender_end: u32,
    receiver_ends: Vec<u32>,
}

pub fn inactive_exchange_program() -> Vec<u32> {
    vec![RETURN_M10_INSTRUCTION]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ComputeGraph, Ipu21CostModel, Layout, PipelineConfig, Precision, TensorFormat, lower,
        lower_to_tiles, place,
    };

    #[test]
    fn randomized_transfer_schedules_preserve_hazards_without_endpoint_overlap() {
        let mut random = fastrand::Rng::with_seed(0x736c_6f74);
        for _ in 0..64 {
            let tile_count = random.u16(2..=32);
            let transfer_count = random.usize(1..=256);
            let transfers = (0..transfer_count)
                .map(|_| {
                    let source = random.u16(0..tile_count);
                    let receiver_count = random.usize(1..=usize::from(tile_count.min(8) - 1));
                    let mut receivers = Vec::with_capacity(receiver_count);
                    while receivers.len() != receiver_count {
                        let tile = random.u16(0..tile_count);
                        if tile != source && !receivers.contains(&tile) {
                            receivers.push(tile);
                        }
                    }
                    PendingTransfer {
                        source,
                        source_shard: LowShardId::from_index(u32::from(source)),
                        source_offset: 0,
                        destinations: receivers.into_iter().map(|tile| (tile, 0)).collect(),
                        source_address: 0,
                        words: random.u32(1..=MAX_TRANSFER_WORDS),
                    }
                })
                .collect::<Vec<_>>();
            let dependencies = memory_dependencies(&transfers, tile_count);
            let mut scheduler = TransferScheduler::new(&transfers, tile_count);
            let mut availability = vec![TileAvailability::default(); usize::from(tile_count)];
            let mut occurrences = vec![0u8; transfers.len()];
            let mut intervals = vec![(0u32, 0u32); transfers.len()];
            while let Some(index) = scheduler.next(&availability) {
                occurrences[index] += 1;
                let transfer = &transfers[index];
                let start = std::iter::once(availability[usize::from(transfer.source)].send)
                    .chain(
                        transfer
                            .destinations
                            .iter()
                            .map(|&(tile, _)| availability[usize::from(tile)].receive),
                    )
                    .max()
                    .unwrap_or(0);
                let end = start.saturating_add(transfers[index].words);
                intervals[index] = (start, end);
                availability[usize::from(transfer.source)] = TileAvailability {
                    send: end,
                    receive: end,
                };
                for &(tile, _) in &transfer.destinations {
                    availability[usize::from(tile)] = TileAvailability {
                        send: end,
                        receive: end,
                    };
                }
                scheduler.complete(index);
            }
            assert!(scheduler.is_complete());
            assert!(occurrences.into_iter().all(|count| count == 1));
            for &(before, after) in &dependencies {
                assert!(intervals[before].1 <= intervals[after].0);
            }
            for tile in 0..tile_count {
                let mut tile_intervals = transfers
                    .iter()
                    .enumerate()
                    .filter(|(_, transfer)| transfer.tiles().any(|endpoint| endpoint == tile))
                    .map(|(index, _)| intervals[index])
                    .collect::<Vec<_>>();
                tile_intervals.sort_unstable();
                assert!(tile_intervals.windows(2).all(|pair| pair[0].1 <= pair[1].0));
            }

            let mut incumbent = MaterializedSchedule::new(tile_count, transfers.len());
            incumbent.order.extend(0..transfers.len());
            let mut last_transfer = vec![None; usize::from(tile_count)];
            for (index, transfer) in transfers.iter().enumerate() {
                let predecessor = transfer
                    .tiles()
                    .filter_map(|tile| last_transfer[usize::from(tile)])
                    .max();
                for tile in transfer.tiles() {
                    last_transfer[usize::from(tile)] = Some(index);
                }
                incumbent.timings[index] = Some(MaterializedTiming {
                    start: index as u32,
                    end: index as u32 + 1,
                    blocking_tile: transfer.source,
                    predecessor,
                });
            }
            let repaired = critical_neighborhood_order(&transfers, tile_count, &incumbent);
            let mut repaired_positions = vec![usize::MAX; transfers.len()];
            for (position, &index) in repaired.iter().enumerate() {
                assert_eq!(repaired_positions[index], usize::MAX);
                repaired_positions[index] = position;
            }
            assert!(
                repaired_positions
                    .iter()
                    .all(|position| *position != usize::MAX)
            );
            for &(before, after) in &dependencies {
                assert!(repaired_positions[before] < repaired_positions[after]);
            }
        }
    }

    #[test]
    fn randomized_gemm_exchanges_produce_one_executable_row_per_tile() {
        let mut random = fastrand::Rng::with_seed(0x6578_6368);
        for _ in 0..32 {
            let tiles = 1_u16 << random.u32(1..=3);
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
            let phases = lower_exchanges(
                &low,
                &placement,
                &Topology::c600(),
                ExchangeLoweringOptions::default(),
            )
            .unwrap();
            assert_eq!(phases.len(), low.exchange_phases.len());
            for phase in phases {
                assert_eq!(phase.programs.len(), usize::from(tiles));
                assert_eq!(phase.active.len(), usize::from(tiles));
                assert_eq!(phase.tile_event_cycles.len(), usize::from(tiles));
                assert_eq!(phase.activities.len(), usize::from(tiles));
                assert!(phase.event_cycles != 0);
                assert!(phase.activities.iter().flatten().next().is_some());
                for activities in &phase.activities {
                    for activity in activities {
                        assert!(activity.start_cycle < activity.end_cycle);
                        assert!(activity.end_cycle <= phase.event_cycles);
                    }
                    for pair in activities.windows(2) {
                        assert!(pair[0].end_cycle <= pair[1].start_cycle);
                    }
                }
                for ((active, program), local_cycles) in phase
                    .active
                    .iter()
                    .zip(&phase.programs)
                    .zip(&phase.tile_event_cycles)
                {
                    assert_eq!(program.last(), Some(&RETURN_M10_INSTRUCTION));
                    assert_eq!(*active, program.len() > 1);
                    assert_eq!(*active, *local_cycles != 0);
                    assert_eq!(plan_event_cycles(program).unwrap(), *local_cycles);
                    assert!(*local_cycles <= phase.event_cycles);
                    assert!(!program.contains(&ipu_exchange::SYNC_SUPERVISOR_INSTRUCTION));
                }
            }
        }
    }
}
