//! Physical exchange programs generated from logical shard transfers.

use crate::{
    ExchangePhaseId, LowProgram, Placement, ShardDefinition, logical_view_byte_spans,
    view_byte_spans,
};
use ipu_exchange::{
    MAX_TRANSFER_WORDS, MulticastPlan, PLAN_WORDS, PlanProgramBuilder, RETURN_M10_INSTRUCTION,
    SANS_INACTIVE_INSTRUCTION, SYNC_ANS_INSTRUCTION, Topology, finalize_point_receiver,
    offset_plan, patch_receiver_address, patch_sender_address, plan_event_cycles,
};
use std::cmp::Reverse;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalExchangePhase {
    pub id: ExchangePhaseId,
    /// Executable supervisor row indexed by logical tile.
    pub rows: Vec<Vec<u32>>,
    pub event_cycles: u32,
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
}

pub fn lower_exchanges(
    program: &LowProgram,
    placement: &Placement,
    topology: &Topology,
    options: ExchangeLoweringOptions,
) -> Result<Vec<PhysicalExchangePhase>, ExchangeLoweringError> {
    program
        .exchange_phases
        .iter()
        .map(|phase| {
            let mut builders = BTreeMap::<u16, PlanProgramBuilder>::new();
            let mut horizon = 0u32;
            let mut tile_availability = vec![0u32; usize::from(program.tile_count)];
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
            let conflict_slots = conflict_slots(&pending, program.tile_count);
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
                    let transfer_end = append_transfer(
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
                    if let Some(diagnostics) = &mut diagnostics {
                        diagnostics.record(
                            transfer.source,
                            transfer.destinations.iter().map(|entry| entry.0),
                            transfer.words,
                            schedule_offset,
                            transfer_end,
                            blocking_tile,
                        );
                    }
                    for tile in transfer.tiles() {
                        tile_availability[usize::from(tile)] = transfer_end;
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
            let rows = (0..program.tile_count)
                .map(|tile| match builders.remove(&tile) {
                    Some(builder) => Ok(builder.finish(horizon)?),
                    None => Ok(inactive_exchange_row()),
                })
                .collect::<Result<Vec<_>, ipu_exchange::ExchangeError>>()?;
            Ok(PhysicalExchangePhase {
                id: phase.id,
                rows,
                event_cycles: horizon,
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
    destinations: Vec<(u16, u32)>,
    source_address: u32,
    words: u32,
}

impl PendingTransfer {
    fn tiles(&self) -> impl Iterator<Item = u16> + '_ {
        std::iter::once(self.source).chain(self.destinations.iter().map(|entry| entry.0))
    }
}

/// Colors multicast hyperedges into conflict-free slots without constructing
/// their quadratic conflict graph. Per-tile bit sets and first-free cursors
/// make finding the earliest common slot depend on endpoint pressure rather
/// than the number of pairwise conflicts.
fn conflict_slots(transfers: &[PendingTransfer], tile_count: u16) -> Vec<Vec<usize>> {
    let mut pressure = vec![0usize; usize::from(tile_count)];
    for transfer in transfers {
        for tile in transfer.tiles() {
            pressure[usize::from(tile)] += 1;
        }
    }

    let mut priority = (0..transfers.len()).collect::<Vec<_>>();
    priority.sort_unstable_by_key(|&index| {
        let transfer = &transfers[index];
        let endpoint_pressure = transfer
            .tiles()
            .map(|tile| pressure[usize::from(tile)])
            .sum::<usize>();
        (
            Reverse(endpoint_pressure),
            Reverse(transfer.destinations.len()),
            Reverse(transfer.words),
            transfer.source,
            index,
        )
    });

    let mut slots = Vec::<Vec<usize>>::new();
    let mut used_slots = vec![Vec::<u64>::new(); usize::from(tile_count)];
    let mut first_free = vec![0usize; usize::from(tile_count)];
    for index in priority {
        let transfer = &transfers[index];
        let mut slot = transfer
            .tiles()
            .map(|tile| first_free[usize::from(tile)])
            .max()
            .unwrap_or(0);
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
    }
    slots
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
) -> Result<u32, ExchangeLoweringError> {
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
        let mut plan = topology.multicast(source, &tiles, words, 0)?;
        offset_plan(&mut plan.sender, schedule_offset)?;
        for receiver in &mut plan.receivers {
            offset_plan(receiver, schedule_offset)?;
        }
        plan
    };
    patch_sender_address(&mut plan.sender, source_address)?;
    for (row, (_, address)) in plan.receivers.iter_mut().zip(destinations) {
        patch_receiver_address(row, *address)?;
    }
    builders
        .entry(source)
        .or_default()
        .append_scheduled_row(&plan.sender)?;
    for (&tile, row) in tiles.iter().zip(&plan.receivers) {
        builders
            .entry(tile)
            .or_default()
            .append_scheduled_row(row)?;
    }
    let transfer_end = std::iter::once(&plan.sender)
        .chain(&plan.receivers)
        .map(|row| plan_event_cycles(row))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(schedule_offset);
    *horizon = (*horizon).max(transfer_end);
    Ok(transfer_end)
}

pub fn inactive_exchange_row() -> Vec<u32> {
    let mut row = vec![0; PLAN_WORDS];
    row[0] = SANS_INACTIVE_INSTRUCTION;
    row[1] = SYNC_ANS_INSTRUCTION;
    row[2] = RETURN_M10_INSTRUCTION;
    row
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ComputeGraph, Ipu21CostModel, Layout, PipelineConfig, Precision, TensorFormat, lower,
        lower_to_tiles, place,
    };

    #[test]
    fn randomized_conflict_slots_cover_transfers_without_endpoint_overlap() {
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
                        destinations: receivers.into_iter().map(|tile| (tile, 0)).collect(),
                        source_address: 0,
                        words: random.u32(1..=MAX_TRANSFER_WORDS),
                    }
                })
                .collect::<Vec<_>>();
            let slots = conflict_slots(&transfers, tile_count);
            let mut occurrences = vec![0u8; transfers.len()];
            for slot in slots {
                let mut used_tiles = vec![false; usize::from(tile_count)];
                for index in slot {
                    occurrences[index] += 1;
                    for tile in transfers[index].tiles() {
                        assert!(!std::mem::replace(&mut used_tiles[usize::from(tile)], true));
                    }
                }
            }
            assert!(occurrences.into_iter().all(|count| count == 1));
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
                assert_eq!(phase.rows.len(), usize::from(tiles));
                assert!(phase.event_cycles != 0);
                for row in phase.rows {
                    if row[0] == SANS_INACTIVE_INSTRUCTION {
                        assert_eq!(row[2], RETURN_M10_INSTRUCTION);
                    } else {
                        assert_eq!(row[0], ipu_exchange::SYNC_SUPERVISOR_INSTRUCTION);
                        assert_eq!(row.last(), Some(&RETURN_M10_INSTRUCTION));
                    }
                }
            }
        }
    }
}
