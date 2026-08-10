//! Physical exchange programs generated from logical shard transfers.

use crate::{
    ExchangePhaseId, LowProgram, LowShardId, Placement, ShardDefinition, logical_view_byte_spans,
    view_byte_spans,
};
use ipu_exchange::{
    MAX_TRANSFER_WORDS, MulticastPlan, PlanProgramBuilder, RETURN_M10_INSTRUCTION, Topology,
    finalize_point_receiver, patch_receiver_address, patch_sender_address,
    patch_sender_instruction, plan_event_cycles, sender_instruction_offsets,
};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalExchangePhase {
    pub id: ExchangePhaseId,
    /// Whether each logical tile participates in this phase's timed program.
    pub active: Vec<bool>,
    /// Synchronization-free timed supervisor program indexed by logical tile.
    pub programs: Vec<Vec<u32>>,
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
    conflict_slots: usize,
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
            let mut builders = BTreeMap::<u16, PlanProgramBuilder>::new();
            let mut horizon = 0u32;
            let mut tile_availability = vec![0u32; usize::from(program.tile_count)];
            let mut activities = vec![Vec::new(); usize::from(program.tile_count)];
            let mut scheduled_sends = vec![Vec::new(); usize::from(program.tile_count)];
            let mut diagnostics = options
                .diagnostics
                .then(|| PhaseDiagnostics::new(program.tile_count));
            let mut pending = Vec::new();
            for transfer in &phase.transfers {
                let source = &program.shards[transfer.source.shard.index() as usize];
                let logical_order = transfer.destinations.iter().any(|view| {
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
                let mut source_index = 0usize;
                let mut source_offset = 0u32;
                let mut destination_positions = vec![(0usize, 0u32); destinations.len()];
                while source_index < source_spans.len() {
                    let source_span = source_spans[source_index];
                    if source_span.bytes == 0 || source_span.offset & 3 != 0 {
                        return Err(ExchangeLoweringError::UnalignedPayload);
                    }
                    let mut chunk_bytes = (source_span.bytes - source_offset).min(
                        MAX_TRANSFER_WORDS
                            .checked_mul(4)
                            .ok_or(ExchangeLoweringError::Overflow)?,
                    );
                    for ((index, offset), (_, _, spans)) in
                        destination_positions.iter().zip(&destinations)
                    {
                        let span = spans
                            .get(*index)
                            .ok_or(ExchangeLoweringError::SizeMismatch)?;
                        if span.offset & 3 != 0 {
                            return Err(ExchangeLoweringError::UnalignedPayload);
                        }
                        chunk_bytes = chunk_bytes.min(span.bytes - *offset);
                    }
                    if chunk_bytes == 0 || chunk_bytes & 3 != 0 {
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
                    for ((index, offset), (_, _, spans)) in
                        destination_positions.iter_mut().zip(&destinations)
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
            }
            let pending = coalesce_pending_transfers(pending);
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
            let conflict_slots = schedule_slots(&pending, program.tile_count);
            if let Some(diagnostics) = &mut diagnostics {
                diagnostics.conflict_slots = conflict_slots.len();
            }
            for slot in conflict_slots {
                for index in slot {
                    let transfer = &pending[index];
                    let (blocking_tile, latest_availability) = transfer
                        .tiles()
                        .map(|tile| (tile, tile_availability[usize::from(tile)]))
                        .max_by_key(|&(tile, availability)| (availability, Reverse(tile)))
                        .unwrap_or((transfer.source, 0));
                    let schedule_offset = if latest_availability == 0 {
                        0
                    } else {
                        latest_availability
                            .checked_add(1)
                            .ok_or(ExchangeLoweringError::Overflow)?
                    };
                    let timing = append_transfer(
                        topology,
                        ScheduledTransfer {
                            source: transfer.source,
                            destinations: &transfer.destinations,
                            source_address: transfer.source_address,
                            words: transfer.words,
                            schedule_offset,
                        },
                        &mut horizon,
                        &mut builders,
                    )?;
                    scheduled_sends[usize::from(transfer.source)]
                        .push((transfer.source_shard, transfer.source_offset));
                    activities[usize::from(transfer.source)].push(ExchangeActivity {
                        kind: ExchangeActivityKind::Send,
                        start_cycle: schedule_offset,
                        end_cycle: timing.sender_end,
                    });
                    for (&(tile, _), &end_cycle) in
                        transfer.destinations.iter().zip(&timing.receiver_ends)
                    {
                        activities[usize::from(tile)].push(ExchangeActivity {
                            kind: ExchangeActivityKind::Receive,
                            start_cycle: schedule_offset,
                            end_cycle,
                        });
                    }
                    if let Some(diagnostics) = &mut diagnostics {
                        diagnostics.record(
                            transfer.source,
                            transfer.destinations.iter().map(|entry| entry.0),
                            transfer.words,
                            schedule_offset,
                            timing.end,
                            blocking_tile,
                        );
                    }
                    for tile in transfer.tiles() {
                        tile_availability[usize::from(tile)] = timing.end;
                    }
                }
            }
            if let Some(diagnostics) = diagnostics {
                diagnostics.emit(
                    phase.id.index(),
                    &phase.provenance,
                    horizon,
                    &tile_availability,
                    &builders,
                );
            }
            let mut active = Vec::with_capacity(usize::from(program.tile_count));
            let programs = (0..program.tile_count)
                .map(|tile| match builders.remove(&tile) {
                    Some(builder) => {
                        active.push(true);
                        Ok(builder.finish(horizon)?)
                    }
                    None => {
                        active.push(false);
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
                event_cycles: horizon,
                activities,
                repeat_patches,
            })
        })
        .collect()
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
            conflict_slots: 0,
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
            conflict_slots = self.conflict_slots,
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

/// Places multicast hyperedges into endpoint-disjoint scheduling layers.
/// Independent ready transfers are pressure-prioritized, while overlapping
/// tile-memory accesses impose program-order precedence between layers.
fn schedule_slots(transfers: &[PendingTransfer], tile_count: u16) -> Vec<Vec<usize>> {
    let mut pressure = vec![0usize; usize::from(tile_count)];
    for transfer in transfers {
        for tile in transfer.tiles() {
            pressure[usize::from(tile)] += 1;
        }
    }

    let priority = |index: usize| {
        let transfer = &transfers[index];
        let endpoint_pressure = transfer
            .tiles()
            .map(|tile| pressure[usize::from(tile)])
            .sum::<usize>();
        (
            endpoint_pressure,
            transfer.destinations.len(),
            transfer.words,
            Reverse(transfer.source),
            Reverse(index),
        )
    };

    let dependencies = memory_dependencies(transfers, tile_count);
    let mut dependents = vec![Vec::new(); transfers.len()];
    let mut prerequisites = vec![Vec::new(); transfers.len()];
    let mut indegrees = vec![0usize; transfers.len()];
    for (before, after) in dependencies {
        dependents[before].push(after);
        prerequisites[after].push(before);
        indegrees[after] += 1;
    }
    let mut ready = BinaryHeap::new();
    for (index, &indegree) in indegrees.iter().enumerate() {
        if indegree == 0 {
            ready.push((priority(index), index));
        }
    }
    let mut ordered = Vec::with_capacity(transfers.len());
    while let Some((_, index)) = ready.pop() {
        ordered.push(index);
        for &dependent in &dependents[index] {
            indegrees[dependent] -= 1;
            if indegrees[dependent] == 0 {
                ready.push((priority(dependent), dependent));
            }
        }
    }
    debug_assert_eq!(ordered.len(), transfers.len());

    let mut slots = Vec::<Vec<usize>>::new();
    let mut used_slots = vec![Vec::<u64>::new(); usize::from(tile_count)];
    let mut first_free = vec![0usize; usize::from(tile_count)];
    let mut assigned_slots = vec![usize::MAX; transfers.len()];
    for index in ordered {
        let transfer = &transfers[index];
        let endpoint_slot = transfer
            .tiles()
            .map(|tile| first_free[usize::from(tile)])
            .max()
            .unwrap_or(0);
        let dependency_slot = prerequisites[index]
            .iter()
            .map(|&before| assigned_slots[before].saturating_add(1))
            .max()
            .unwrap_or(0);
        let mut slot = endpoint_slot.max(dependency_slot);
        while transfer.tiles().any(|tile| {
            let words = &used_slots[usize::from(tile)];
            words
                .get(slot / u64::BITS as usize)
                .is_some_and(|word| word & (1 << (slot % u64::BITS as usize)) != 0)
        }) {
            slot += 1;
        }
        for tile in transfer.tiles() {
            let tile = usize::from(tile);
            let word_index = slot / u64::BITS as usize;
            if used_slots[tile].len() <= word_index {
                used_slots[tile].resize(word_index + 1, 0);
            }
            used_slots[tile][word_index] |= 1 << (slot % u64::BITS as usize);
            if first_free[tile] == slot {
                loop {
                    let candidate = first_free[tile];
                    let word = used_slots[tile]
                        .get(candidate / u64::BITS as usize)
                        .copied()
                        .unwrap_or(0);
                    if word & (1 << (candidate % u64::BITS as usize)) == 0 {
                        break;
                    }
                    first_free[tile] += 1;
                }
            }
        }
        if slots.len() <= slot {
            slots.resize_with(slot + 1, Vec::new);
        }
        slots[slot].push(index);
        assigned_slots[index] = slot;
    }
    slots
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

fn append_transfer(
    topology: &Topology,
    transfer: ScheduledTransfer<'_>,
    horizon: &mut u32,
    builders: &mut BTreeMap<u16, PlanProgramBuilder>,
) -> Result<ScheduledTransferTiming, ExchangeLoweringError> {
    let ScheduledTransfer {
        source,
        destinations,
        source_address,
        words,
        schedule_offset,
    } = transfer;
    if words == 0 {
        return Err(ExchangeLoweringError::UnalignedPayload);
    }
    let tiles = destinations.iter().map(|entry| entry.0).collect::<Vec<_>>();
    let mut plan = if tiles.len() == 1 && schedule_offset == 0 {
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
    for (row, (_, address)) in plan.receivers.iter_mut().zip(destinations) {
        patch_receiver_address(row, *address)?;
    }
    builders
        .entry(source)
        .or_default()
        .append_scheduled_row_at(&plan.sender, schedule_offset)?;
    for (&tile, row) in tiles.iter().zip(&plan.receivers) {
        builders
            .entry(tile)
            .or_default()
            .append_scheduled_row_at(row, schedule_offset)?;
    }
    let sender_end = plan_event_cycles(&plan.sender)?
        .checked_add(schedule_offset)
        .ok_or(ExchangeLoweringError::Overflow)?;
    let receiver_ends = plan
        .receivers
        .iter()
        .map(|row| {
            plan_event_cycles(row)?.checked_add(schedule_offset).ok_or(
                ipu_exchange::ExchangeError::Schedule("exchange row offset overflow"),
            )
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
        end: transfer_end,
        sender_end,
        receiver_ends,
    })
}

struct ScheduledTransferTiming {
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
    fn randomized_schedule_slots_preserve_hazards_without_endpoint_overlap() {
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
            let slots = schedule_slots(&transfers, tile_count);
            let mut occurrences = vec![0u8; transfers.len()];
            let mut assigned_slots = vec![usize::MAX; transfers.len()];
            for (slot_index, slot) in slots.into_iter().enumerate() {
                let mut used_tiles = vec![false; usize::from(tile_count)];
                for index in slot {
                    occurrences[index] += 1;
                    assigned_slots[index] = slot_index;
                    for tile in transfers[index].tiles() {
                        assert!(!std::mem::replace(&mut used_tiles[usize::from(tile)], true));
                    }
                }
            }
            assert!(occurrences.into_iter().all(|count| count == 1));
            for (before, after) in memory_dependencies(&transfers, tile_count) {
                assert!(assigned_slots[before] < assigned_slots[after]);
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
                        layout: Layout::amp_right(64, tiles),
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
                assert_eq!(phase.activities.len(), usize::from(tiles));
                assert!(phase.event_cycles != 0);
                assert!(phase.activities.iter().flatten().next().is_some());
                for activities in &phase.activities {
                    for activity in activities {
                        assert!(activity.start_cycle < activity.end_cycle);
                        assert!(activity.end_cycle <= phase.event_cycles);
                    }
                    for pair in activities.windows(2) {
                        assert!(pair[0].end_cycle < pair[1].start_cycle);
                    }
                }
                for (active, program) in phase.active.iter().zip(&phase.programs) {
                    assert_eq!(program.last(), Some(&RETURN_M10_INSTRUCTION));
                    assert_eq!(*active, program.len() > 1);
                    assert!(!program.contains(&ipu_exchange::SYNC_SUPERVISOR_INSTRUCTION));
                }
            }
        }
    }
}
