//! Physical exchange programs generated from logical shard transfers.

use crate::{
    ByteSpan, ExchangeOrder, ExchangePhaseId, LogicalExchange, LowProgram, LowShardId, Placement,
    ShardDefinition, logical_view_byte_spans, shard_storage_bytes, view_byte_spans,
};
use ipu_package::ExchangeActivityKind;
use ipu_target::exchange::{
    PhaseProgramBuilder, PhaseTransferTiming, PhysicalTransfer, ResolvedTransfer, TransferEndpoint,
    TransferWidth, patch_sender_instruction, sender_address_instruction_groups,
};
use ipu_target::hardware::HardwareTarget;
use ipu_target::instruction::RETURN_M10_INSTRUCTION;
use ipu_target::memory::{MemoryElement, memory_elements_for_words};
use ipu_target::topology::Topology;
use rayon::prelude::*;
use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

#[cfg(test)]
use ipu_target::exchange::plan_event_cycles;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalExchangePhase {
    pub id: ExchangePhaseId,
    /// Synchronization-free timed supervisor program, or `None` when the
    /// logical tile has no work in this phase.
    pub programs: Vec<Option<Vec<u32>>>,
    /// Per-tile base used by point-to-point receive rows in this phase.
    pub incoming_bases: Vec<u32>,
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
    /// Stable index of the physical transfer within this phase.
    pub transfer: u32,
    pub kind: ExchangeActivityKind,
    pub start_cycle: u32,
    pub end_cycle: u32,
    pub address: u32,
    pub words: u32,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ExchangeLoweringError {
    #[error(transparent)]
    Exchange(#[from] ipu_target::exchange::ExchangeError),
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

pub(crate) fn lower_exchanges(
    program: &LowProgram,
    placement: &Placement,
    target: HardwareTarget,
) -> Result<Vec<PhysicalExchangePhase>, ExchangeLoweringError> {
    let topology = target.topology().prefix(program.tile_count)?;
    let maximum_transfer_words = target.exchange().maximum_transfer_words;
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
                .enumerate()
                .map(|(index, transfer)| {
                    prepare_transfer(program, placement, transfer, maximum_transfer_words)
                        .inspect_err(|error| {
                            tracing::error!(
                                phase = phase.id.index(),
                                transfer = index,
                                provenance = ?phase.provenance,
                                source = ?transfer.source,
                                destinations = ?transfer.destinations,
                                ?error,
                                "failed to prepare logical exchange transfer"
                            );
                        })
                })
                .collect::<Result<Vec<_>, ExchangeLoweringError>>()?
                .into_iter()
                .flatten()
                .collect();
            let mut pending = coalesce_pending_transfers(pending, maximum_transfer_words);
            attach_repeat_source_addresses(&mut pending, &repeat_inputs, placement)?;
            let incoming_bases = incoming_bases(&pending, program.tile_count)?;
            let schedule = materialize_greedy_schedule(
                &topology,
                &pending,
                &incoming_bases,
                program.tile_count,
            )?;
            let MaterializedSchedule {
                builder,
                activities,
                scheduled_sends,
                ..
            } = schedule;
            let horizon = builder.event_cycles();
            let phase_programs = builder.finish()?;
            debug_assert_eq!(phase_programs.event_cycles, horizon);
            let programs = phase_programs.programs;
            let repeat_patches = programs
                .iter()
                .enumerate()
                .map(|(tile, program)| {
                    let Some(program) = program else {
                        return Ok(Vec::new());
                    };
                    let address_groups = sender_address_instruction_groups(program)?;
                    if address_groups.len() != scheduled_sends[tile].len() {
                        return Err(ExchangeLoweringError::IncompatibleRepeatRows);
                    }
                    let mut patches = Vec::new();
                    for (instructions, &(source_shard, source_offset)) in
                        address_groups.into_iter().zip(&scheduled_sends[tile])
                    {
                        let Some(inputs) = repeat_inputs.get(&source_shard) else {
                            continue;
                        };
                        for (word_offset, byte_offset) in instructions {
                            let values = inputs
                                .iter()
                                .map(|input| {
                                    let address = placement
                                        .shard_addresses
                                        .get(input)
                                        .copied()
                                        .ok_or(ExchangeLoweringError::UnplacedShard)?
                                        .checked_add(source_offset)
                                        .and_then(|address| address.checked_add(byte_offset))
                                        .ok_or(ExchangeLoweringError::Overflow)?;
                                    let mut instruction = program[word_offset];
                                    patch_sender_instruction(&mut instruction, address)?;
                                    Ok(instruction)
                                })
                                .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
                            if values.first() != Some(&program[word_offset]) {
                                return Err(ExchangeLoweringError::IncompatibleRepeatRows);
                            }
                            patches.push(ExchangeRowPatch {
                                word_offset: u32::try_from(word_offset)
                                    .map_err(|_| ExchangeLoweringError::Overflow)?,
                                values,
                            });
                        }
                    }
                    Ok(patches)
                })
                .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
            Ok(PhysicalExchangePhase {
                id: phase.id,
                programs,
                incoming_bases: incoming_bases
                    .into_iter()
                    .map(|base| base.unwrap_or(0))
                    .collect(),
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
    maximum_transfer_words: u32,
) -> Result<Vec<PendingTransfer>, ExchangeLoweringError> {
    let source = &program.shards[transfer.source.shard.index() as usize];
    let logical_order = transfer.order == ExchangeOrder::Semantic
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
    let (source_spans, planned_destination_spans) =
        if let ExchangeOrder::Planned(geometry) = transfer.order {
            if geometry.rows == 0
                || geometry.row_bytes == 0
                || geometry.source_offset & 0b11 != 0
                || geometry.destination_offset & 0b11 != 0
                || geometry.row_bytes & 0b11 != 0
            {
                return Err(ExchangeLoweringError::UnalignedPayload);
            }
            let mut source_spans = Vec::with_capacity(geometry.rows as usize);
            let mut destination_spans = Vec::with_capacity(geometry.rows as usize);
            for row in 0..geometry.rows {
                source_spans.push(ByteSpan {
                    offset: geometry
                        .source_offset
                        .checked_add(
                            row.checked_mul(geometry.source_stride)
                                .ok_or(ExchangeLoweringError::Overflow)?,
                        )
                        .ok_or(ExchangeLoweringError::Overflow)?,
                    bytes: geometry.row_bytes,
                });
                destination_spans.push(ByteSpan {
                    offset: geometry
                        .destination_offset
                        .checked_add(
                            row.checked_mul(geometry.destination_stride)
                                .ok_or(ExchangeLoweringError::Overflow)?,
                        )
                        .ok_or(ExchangeLoweringError::Overflow)?,
                    bytes: geometry.row_bytes,
                });
            }
            (source_spans, Some(destination_spans))
        } else if logical_order {
            (logical_view_byte_spans(source, &transfer.source)?, None)
        } else {
            (view_byte_spans(source, &transfer.source)?, None)
        };
    let source_limit = shard_storage_bytes(source)?;
    if source_spans.iter().any(|span| {
        span.offset
            .checked_add(span.bytes)
            .is_none_or(|end| end > source_limit)
    }) {
        return Err(ExchangeLoweringError::UnalignedPayload);
    }
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
                if let Some(destination_spans) = &planned_destination_spans {
                    let destination_limit = shard_storage_bytes(shard)?;
                    if destination_spans.iter().any(|span| {
                        span.offset
                            .checked_add(span.bytes)
                            .is_none_or(|end| end > destination_limit)
                    }) {
                        return Err(ExchangeLoweringError::UnalignedPayload);
                    }
                    Cow::Borrowed(destination_spans.as_slice())
                } else if logical_order {
                    Cow::Owned(logical_view_byte_spans(shard, view)?)
                } else {
                    Cow::Owned(view_byte_spans(shard, view)?)
                },
            ))
        })
        .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
    if destinations.is_empty() {
        return Err(ExchangeLoweringError::SizeMismatch);
    }
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
            maximum_transfer_words
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
            physical: PhysicalTransfer {
                source: source.tile,
                source_addresses: vec![source_address],
                destinations: destination_entries.into_iter().map(Into::into).collect(),
                words: chunk_bytes / 4,
                width: TransferWidth::Word32,
            },
            source_shard: source.id,
            source_offset: source_span
                .offset
                .checked_add(source_offset)
                .ok_or(ExchangeLoweringError::Overflow)?,
            source_elements: memory_elements_for_words(source_address, chunk_bytes / 4).collect(),
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

#[derive(Clone)]
struct PendingTransfer {
    physical: PhysicalTransfer,
    source_shard: LowShardId,
    source_offset: u32,
    source_elements: Vec<MemoryElement>,
}

impl PendingTransfer {
    fn tiles(&self) -> impl Iterator<Item = u16> + '_ {
        std::iter::once(self.physical.source)
            .chain(self.physical.destinations.iter().map(|entry| entry.0))
    }

    fn refresh_source_elements(&mut self) {
        self.source_elements = self
            .physical
            .source_addresses
            .iter()
            .flat_map(|&address| memory_elements_for_words(address, self.physical.words))
            .collect();
        self.source_elements.sort_unstable();
        self.source_elements.dedup();
    }
}

fn attach_repeat_source_addresses(
    pending: &mut [PendingTransfer],
    repeat_inputs: &BTreeMap<LowShardId, Vec<LowShardId>>,
    placement: &Placement,
) -> Result<(), ExchangeLoweringError> {
    for transfer in pending {
        if let Some(inputs) = repeat_inputs.get(&transfer.source_shard) {
            let addresses = inputs
                .iter()
                .map(|input| {
                    placement
                        .shard_addresses
                        .get(input)
                        .copied()
                        .ok_or(ExchangeLoweringError::UnplacedShard)?
                        .checked_add(transfer.source_offset)
                        .ok_or(ExchangeLoweringError::Overflow)
                })
                .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
            if addresses.first().copied() != Some(transfer.physical.source_address()) {
                return Err(ExchangeLoweringError::IncompatibleRepeatRows);
            }
            transfer.physical.source_addresses = addresses;
        }
        transfer.refresh_source_elements();
    }
    Ok(())
}

fn incoming_bases(
    pending: &[PendingTransfer],
    tile_count: u16,
) -> Result<Vec<Option<u32>>, ExchangeLoweringError> {
    let mut receive_counts = vec![0usize; usize::from(tile_count)];
    for transfer in pending {
        for &TransferEndpoint(tile, _) in &transfer.physical.destinations {
            let count = receive_counts
                .get_mut(usize::from(tile))
                .ok_or(ExchangeLoweringError::InvalidDestination)?;
            *count += 1;
        }
    }
    let mut incoming_bases = vec![None::<u32>; usize::from(tile_count)];
    for transfer in pending {
        if let [TransferEndpoint(tile, address)] = transfer.physical.destinations.as_slice()
            && receive_counts[usize::from(*tile)] == 1
        {
            incoming_bases[usize::from(*tile)] = Some(*address);
        }
    }
    Ok(incoming_bases)
}

/// Combines physically contiguous source and destination spans into one
/// hardware message. Separate SEND messages require separate receive events,
/// even when they select the same source tile.
fn coalesce_pending_transfers(
    transfers: Vec<PendingTransfer>,
    maximum_transfer_words: u32,
) -> Vec<PendingTransfer> {
    let mut merged = Vec::<PendingTransfer>::with_capacity(transfers.len());
    for transfer in transfers {
        let Some(previous) = merged.last_mut() else {
            merged.push(transfer);
            continue;
        };
        let previous_bytes = previous.physical.words * 4;
        let combined_words = previous.physical.words.checked_add(transfer.physical.words);
        let contiguous = previous.physical.source == transfer.physical.source
            && previous.physical.width == transfer.physical.width
            && previous.source_shard == transfer.source_shard
            && previous
                .source_offset
                .checked_add(previous_bytes)
                .is_some_and(|end| end == transfer.source_offset)
            && previous.physical.destinations.len() == transfer.physical.destinations.len()
            && previous.physical.source_addresses.len() == transfer.physical.source_addresses.len()
            && previous
                .physical
                .source_addresses
                .iter()
                .zip(&transfer.physical.source_addresses)
                .all(|(&left, &right)| {
                    left.checked_add(previous_bytes)
                        .is_some_and(|end| end == right)
                })
            && previous
                .physical
                .destinations
                .iter()
                .zip(&transfer.physical.destinations)
                .all(
                    |(
                        &TransferEndpoint(left_tile, left_address),
                        &TransferEndpoint(right_tile, right_address),
                    )| {
                        left_tile == right_tile
                            && left_address
                                .checked_add(previous_bytes)
                                .is_some_and(|end| end == right_address)
                    },
                );
        if contiguous && combined_words.is_some_and(|words| words <= maximum_transfer_words) {
            previous.physical.words = combined_words.expect("checked above");
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
    dynamic_word_pressure: bool,
    dependents: Vec<Vec<usize>>,
    indegrees: Vec<usize>,
    dependency_ready: Vec<u32>,
    ready: BinaryHeap<ReadyTransfer>,
    completed: usize,
}

impl<'a> TransferScheduler<'a> {
    fn new(transfers: &'a [PendingTransfer], tile_count: u16) -> Self {
        let mut word_pressure = vec![0u64; usize::from(tile_count)];
        for transfer in transfers {
            let items = u64::from(
                transfer
                    .physical
                    .item_count()
                    .unwrap_or(transfer.physical.words),
            );
            for tile in transfer.tiles() {
                word_pressure[usize::from(tile)] += items;
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
            // Multicast choices release several endpoint queues at once, so
            // their useful priority is the pressure which remains. Stable
            // pressure is a better matching tie-break for point-to-point work.
            dynamic_word_pressure: transfers
                .iter()
                .any(|transfer| transfer.physical.destinations.len() > 1),
            dependents,
            indegrees,
            dependency_ready: vec![0; transfers.len()],
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
            fanout: transfer.physical.destinations.len(),
            words: transfer
                .physical
                .item_count()
                .unwrap_or(transfer.physical.words),
            source: Reverse(transfer.physical.source),
            index: Reverse(index),
        });
    }

    fn next(&mut self, tile_availability: &[TileAvailability]) -> Option<(usize, u32)> {
        loop {
            let candidate = self.ready.pop()?;
            let index = candidate.index.0;
            let transfer = &self.transfers[index];
            let earliest_start =
                std::iter::once(self.dependency_ready[index])
                    .chain(std::iter::once(
                        tile_availability[usize::from(transfer.physical.source)].send,
                    ))
                    .chain(transfer.physical.destinations.iter().map(
                        |&TransferEndpoint(tile, _)| tile_availability[usize::from(tile)].receive,
                    ))
                    .max()
                    .unwrap_or(0);
            if candidate.earliest_start.0 == earliest_start {
                return Some((index, earliest_start));
            }
            self.push_ready(index, earliest_start);
        }
    }

    fn complete(&mut self, index: usize, completion: u32) {
        self.completed += 1;
        let transfer = &self.transfers[index];
        if self.dynamic_word_pressure {
            let items = u64::from(
                transfer
                    .physical
                    .item_count()
                    .unwrap_or(transfer.physical.words),
            );
            for tile in transfer.tiles() {
                self.word_pressure[usize::from(tile)] =
                    self.word_pressure[usize::from(tile)].saturating_sub(items);
            }
        }
        let dependents = std::mem::take(&mut self.dependents[index]);
        for dependent in dependents {
            self.dependency_ready[dependent] = self.dependency_ready[dependent].max(completion);
            self.indegrees[dependent] -= 1;
            if self.indegrees[dependent] == 0 {
                self.push_ready(dependent, self.dependency_ready[dependent]);
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
        let bytes = u64::from(transfer.physical.words) * 4;
        for &address in &transfer.physical.source_addresses {
            accesses[usize::from(transfer.physical.source)].push(TransferAccess {
                transfer: index,
                start: u64::from(address),
                end: u64::from(address) + bytes,
                write: false,
            });
        }
        for &TransferEndpoint(tile, address) in &transfer.physical.destinations {
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

#[derive(Clone, Copy, Debug, Default)]
struct TileAvailability {
    send: u32,
    receive: u32,
}

#[derive(Clone, Debug)]
struct MemoryAccess {
    start: u32,
    end: u32,
    elements: Vec<MemoryElement>,
}

#[derive(Clone, Debug, Default)]
struct TileMemorySchedule {
    sends: Vec<MemoryAccess>,
    receives: Vec<MemoryAccess>,
}

struct MaterializedSchedule {
    builder: PhaseProgramBuilder,
    validate_encoding: bool,
    tile_availability: Vec<TileAvailability>,
    memory_accesses: Vec<TileMemorySchedule>,
    activities: Vec<Vec<ExchangeActivity>>,
    scheduled_sends: Vec<Vec<(LowShardId, u32)>>,
}

impl MaterializedSchedule {
    fn new(tile_count: u16, validate_encoding: bool) -> Self {
        Self {
            builder: PhaseProgramBuilder::new(tile_count),
            validate_encoding,
            tile_availability: vec![TileAvailability::default(); usize::from(tile_count)],
            memory_accesses: (0..tile_count)
                .map(|_| TileMemorySchedule::default())
                .collect(),
            activities: vec![Vec::new(); usize::from(tile_count)],
            scheduled_sends: vec![Vec::new(); usize::from(tile_count)],
        }
    }

    fn append(
        &mut self,
        topology: &Topology,
        pending: &[PendingTransfer],
        incoming_bases: &[Option<u32>],
        index: usize,
        dependency_ready: u32,
    ) -> Result<u32, ExchangeLoweringError> {
        let transfer = &pending[index];
        let physical = &transfer.physical;
        let latest_availability =
            std::iter::once(self.tile_availability[usize::from(physical.source)].send)
                .chain(
                    physical
                        .destinations
                        .iter()
                        .map(|&TransferEndpoint(tile, _)| {
                            self.tile_availability[usize::from(tile)].receive
                        }),
                )
                .max()
                .unwrap_or(0)
                .max(dependency_ready);
        let incoming_base = physical
            .destinations
            .first()
            .and_then(|endpoint| incoming_bases[usize::from(endpoint.0)]);
        let resolved = physical.resolve(topology, incoming_base)?;
        let timing = append_transfer(
            &self.memory_accesses,
            physical,
            &resolved,
            &transfer.source_elements,
            latest_availability,
            &mut self.builder,
            self.validate_encoding,
        )?;
        let payload_end = timing.payload_end;
        self.memory_accesses[usize::from(physical.source)]
            .sends
            .push(MemoryAccess {
                start: timing.payload_start,
                end: timing.sender_horizon,
                elements: transfer.source_elements.clone(),
            });
        for ((&TransferEndpoint(tile, address), &start), &memory_end) in physical
            .destinations
            .iter()
            .zip(&timing.receiver_payload_starts)
            .zip(&timing.receiver_horizons)
        {
            self.memory_accesses[usize::from(tile)]
                .receives
                .push(MemoryAccess {
                    start,
                    end: memory_end,
                    elements: memory_elements_for_words(address, physical.words).collect(),
                });
        }
        self.scheduled_sends[usize::from(physical.source)]
            .push((transfer.source_shard, transfer.source_offset));
        self.activities[usize::from(physical.source)].push(ExchangeActivity {
            transfer: u32::try_from(index).map_err(|_| ExchangeLoweringError::Overflow)?,
            kind: ExchangeActivityKind::Send,
            start_cycle: timing.payload_start,
            end_cycle: payload_end,
            address: physical.source_address(),
            words: physical.words,
        });
        for ((&TransferEndpoint(tile, address), &start_cycle), &end_cycle) in physical
            .destinations
            .iter()
            .zip(&timing.receiver_payload_starts)
            .zip(&timing.receiver_payload_ends)
        {
            self.activities[usize::from(tile)].push(ExchangeActivity {
                transfer: u32::try_from(index).map_err(|_| ExchangeLoweringError::Overflow)?,
                kind: ExchangeActivityKind::Receive,
                start_cycle,
                end_cycle,
                address,
                words: physical.words,
            });
        }
        self.tile_availability[usize::from(physical.source)].send = timing.payload_end;
        for (&TransferEndpoint(tile, _), &receiver_end) in physical
            .destinations
            .iter()
            .zip(&timing.receiver_payload_ends)
        {
            self.tile_availability[usize::from(tile)].receive = receiver_end;
        }
        Ok(timing.payload_completion())
    }
}

fn materialize_greedy_schedule(
    topology: &Topology,
    pending: &[PendingTransfer],
    incoming_bases: &[Option<u32>],
    tile_count: u16,
) -> Result<MaterializedSchedule, ExchangeLoweringError> {
    let schedule =
        materialize_greedy_schedule_impl(topology, pending, incoming_bases, tile_count, false)?;
    // Deferred encoding avoids rebuilding every affected tile row after every
    // insertion. Most completed schedules encode directly; retry the same
    // transfer order with exact validation only when SENDPICP word alignment
    // requires moving an insertion.
    match schedule.builder.clone().finish() {
        Ok(_) => Ok(schedule),
        Err(ipu_target::exchange::ExchangeError::Schedule("SENDPICP instruction alignment")) => {
            materialize_greedy_schedule_impl(topology, pending, incoming_bases, tile_count, true)
        }
        Err(error) => Err(error.into()),
    }
}

fn materialize_greedy_schedule_impl(
    topology: &Topology,
    pending: &[PendingTransfer],
    incoming_bases: &[Option<u32>],
    tile_count: u16,
    validate_encoding: bool,
) -> Result<MaterializedSchedule, ExchangeLoweringError> {
    let mut schedule = MaterializedSchedule::new(tile_count, validate_encoding);
    let mut scheduler = TransferScheduler::new(pending, tile_count);
    while let Some((index, dependency_ready)) = scheduler.next(&schedule.tile_availability) {
        let completion =
            schedule.append(topology, pending, incoming_bases, index, dependency_ready)?;
        scheduler.complete(index, completion);
    }
    debug_assert!(scheduler.is_complete());
    Ok(schedule)
}

fn append_transfer(
    memory_accesses: &[TileMemorySchedule],
    physical: &PhysicalTransfer,
    resolved: &ResolvedTransfer,
    source_elements: &[MemoryElement],
    requested_offset: u32,
    builder: &mut PhaseProgramBuilder,
    validate_encoding: bool,
) -> Result<PhaseTransferTiming, ExchangeLoweringError> {
    let words = physical.words;
    if words == 0 || source_elements.is_empty() {
        return Err(ExchangeLoweringError::UnalignedPayload);
    }
    let mut schedule_offset = requested_offset;
    loop {
        let previous = schedule_offset;
        schedule_offset = if validate_encoding {
            builder.earliest_transfer_offset(resolved, schedule_offset)?
        } else {
            builder.earliest_transfer_offset_deferred(resolved, schedule_offset)?
        };
        let timing = builder.transfer_timing_at(resolved, schedule_offset)?;
        let receiver_intervals = timing
            .receiver_payload_starts
            .iter()
            .copied()
            .zip(timing.receiver_horizons.iter().copied())
            .collect::<Vec<_>>();
        schedule_offset = schedule_offset.max(memory_safe_transfer_offset(
            memory_accesses,
            physical,
            source_elements,
            &timing,
            &receiver_intervals,
            schedule_offset,
        )?);
        if schedule_offset == previous {
            break;
        }
    }
    Ok(builder.append_transfer_at(resolved, schedule_offset)?)
}

fn memory_safe_transfer_offset(
    memory_accesses: &[TileMemorySchedule],
    transfer: &PhysicalTransfer,
    source_elements: &[MemoryElement],
    timing: &PhaseTransferTiming,
    receiver_intervals: &[(u32, u32)],
    schedule_offset: u32,
) -> Result<u32, ExchangeLoweringError> {
    let mut safe_offset = schedule_offset;
    let source_clash = memory_accesses[usize::from(transfer.source)]
        .receives
        .iter()
        .filter(|access| timing.payload_start < access.end && access.start < timing.sender_horizon)
        .filter(|access| {
            access
                .elements
                .iter()
                .any(|element| source_elements.binary_search(element).is_ok())
        })
        .map(|access| access.end)
        .max();
    let receiver_clash = transfer
        .destinations
        .iter()
        .zip(receiver_intervals)
        .flat_map(
            |(&TransferEndpoint(tile, address), &(receive_start, receive_end))| {
                memory_accesses[usize::from(tile)]
                    .sends
                    .iter()
                    .filter(move |access| receive_start < access.end && access.start < receive_end)
                    .filter(move |access| {
                        memory_elements_for_words(address, transfer.words)
                            .any(|element| access.elements.contains(&element))
                    })
                    .map(move |access| access.end.saturating_sub(receive_start))
            },
        )
        .max();
    let source_delay = source_clash.map(|end| end.saturating_sub(timing.payload_start));
    if let Some(delay) = source_delay.into_iter().chain(receiver_clash).max() {
        safe_offset = safe_offset
            .checked_add(delay)
            .ok_or(ExchangeLoweringError::Overflow)?;
    }
    Ok(safe_offset)
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
    fn randomized_transfer_schedules_preserve_hazards_without_same_role_overlap() {
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
                    let words =
                        random.u32(1..=HardwareTarget::Ipu21.exchange().maximum_transfer_words);
                    PendingTransfer {
                        physical: PhysicalTransfer {
                            source,
                            source_addresses: vec![0],
                            destinations: receivers
                                .into_iter()
                                .map(|tile| TransferEndpoint(tile, 0))
                                .collect(),
                            words,
                            width: TransferWidth::Word32,
                        },
                        source_shard: LowShardId::from_index(u32::from(source)),
                        source_offset: 0,
                        source_elements: memory_elements_for_words(0, words).collect(),
                    }
                })
                .collect::<Vec<_>>();
            let dependencies = memory_dependencies(&transfers, tile_count);
            let mut scheduler = TransferScheduler::new(&transfers, tile_count);
            let mut availability = vec![TileAvailability::default(); usize::from(tile_count)];
            let mut occurrences = vec![0u8; transfers.len()];
            let mut intervals = vec![(0u32, 0u32); transfers.len()];
            while let Some((index, dependency_ready)) = scheduler.next(&availability) {
                occurrences[index] += 1;
                let transfer = &transfers[index];
                let start =
                    std::iter::once(dependency_ready)
                        .chain(std::iter::once(
                            availability[usize::from(transfer.physical.source)].send,
                        ))
                        .chain(transfer.physical.destinations.iter().map(
                            |&TransferEndpoint(tile, _)| availability[usize::from(tile)].receive,
                        ))
                        .max()
                        .unwrap_or(0);
                let end = start.saturating_add(transfers[index].physical.words);
                intervals[index] = (start, end);
                availability[usize::from(transfer.physical.source)].send = end;
                for &TransferEndpoint(tile, _) in &transfer.physical.destinations {
                    availability[usize::from(tile)].receive = end;
                }
                scheduler.complete(index, end);
            }
            assert!(scheduler.is_complete());
            assert!(occurrences.into_iter().all(|count| count == 1));
            for &(before, after) in &dependencies {
                assert!(intervals[before].1 <= intervals[after].0);
            }
            for tile in 0..tile_count {
                let mut send_intervals = transfers
                    .iter()
                    .enumerate()
                    .filter(|(_, transfer)| transfer.physical.source == tile)
                    .map(|(index, _)| intervals[index])
                    .collect::<Vec<_>>();
                send_intervals.sort_unstable();
                assert!(send_intervals.windows(2).all(|pair| pair[0].1 <= pair[1].0));
                let mut receive_intervals = transfers
                    .iter()
                    .enumerate()
                    .filter(|(_, transfer)| {
                        transfer
                            .physical
                            .destinations
                            .iter()
                            .any(|&TransferEndpoint(destination, _)| destination == tile)
                    })
                    .map(|(index, _)| intervals[index])
                    .collect::<Vec<_>>();
                receive_intervals.sort_unstable();
                assert!(
                    receive_intervals
                        .windows(2)
                        .all(|pair| pair[0].1 <= pair[1].0)
                );
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
            let phases = lower_exchanges(&low, &placement, HardwareTarget::Ipu21).unwrap();
            assert_eq!(phases.len(), low.exchange_phases.len());
            for phase in phases {
                assert_eq!(phase.programs.len(), usize::from(tiles));
                assert_eq!(phase.activities.len(), usize::from(tiles));
                assert!(phase.event_cycles != 0);
                assert!(phase.activities.iter().flatten().next().is_some());
                for activities in &phase.activities {
                    for activity in activities {
                        assert!(activity.start_cycle < activity.end_cycle);
                        assert!(activity.end_cycle <= phase.event_cycles);
                    }
                }
                for program in &phase.programs {
                    let active = program.is_some();
                    let program = program.as_deref().unwrap_or(&[RETURN_M10_INSTRUCTION]);
                    assert_eq!(program.last(), Some(&RETURN_M10_INSTRUCTION));
                    assert_eq!(active, program.len() > 1);
                    assert_eq!(active, plan_event_cycles(program).unwrap() != 0);
                    assert!(plan_event_cycles(program).unwrap() <= phase.event_cycles);
                    assert!(
                        !program.contains(&ipu_target::instruction::SYNC_SUPERVISOR_INSTRUCTION)
                    );
                }
            }
        }
    }
}
