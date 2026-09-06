//! Physical exchange programs generated from logical shard transfers.

mod diagnostic;
mod order;
use diagnostic::PhaseDiagnostics;
pub use diagnostic::diagnose_exchange_tile;
use order::{critical_neighborhood_order, point_to_point_matching_wave_order};
mod reuse;
mod traffic;
pub(crate) use reuse::ExchangeScheduleCache;
pub(crate) use traffic::MappingTraffic;

use crate::{
    BlockValueId, ExchangePhaseId, LogicalExchange, LowProgram, Placement, ShardDefinition,
    logical_view_byte_spans, view_byte_spans,
};
use ipu_exchange::{
    MAX_TRANSFER_WORDS, MulticastPlan, PhaseProgramBuilder, RETURN_M10_INSTRUCTION, Topology,
    finalize_point_receiver, patch_receiver_address, patch_sender_address,
    patch_sender_instruction, sender_address_instruction_groups,
};
use ipu_package::{
    IPU21_INTERLEAVED_ELEMENT_SIZE, IPU21_INTERLEAVED_MEMORY_BASE, TILE_MEMORY_ELEMENT_SIZE,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};

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
    pub fanout: u16,
    pub paired: bool,
    /// Stable index of the physical transfer within this phase.
    pub transfer: u32,
    pub kind: ExchangeActivityKind,
    pub start_cycle: u32,
    pub end_cycle: u32,
    /// End of the associated SRAM read or write hazard window.
    pub memory_end_cycle: u32,
    pub address: u32,
    pub words: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExchangeActivityKind {
    Send,
    Receive,
    /// This tile's transmit lane is borrowed by its partner; receiving remains available.
    PartnerBusy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExchangeMemoryElement {
    pub interleaved: bool,
    pub index: u32,
}

pub const EXCHANGE_SCHEDULE_SNAPSHOT_VERSION: u32 = 3;

/// Address-resolved transfers captured immediately before physical scheduling.
/// Replaying this data exercises the production scheduler and exchange-row
/// encoder without compiling kernels or loading a device.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExchangeScheduleSnapshot {
    pub schema_version: u32,
    pub tile_count: u16,
    pub phases: Vec<ExchangeScheduleProblem>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExchangeScheduleProblem {
    pub phase: u32,
    pub transfers: Vec<ExchangeScheduleTransfer>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExchangeScheduleTransfer {
    pub source: u16,
    /// Address used by each structured-repeat iteration. Ordinary transfers
    /// contain exactly one entry.
    pub source_addresses: Vec<u32>,
    pub destinations: Vec<ExchangeScheduleDestination>,
    pub words: u32,
    pub width: ExchangeItemWidth,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExchangeItemWidth {
    #[default]
    Word32,
    Paired64,
}

impl ExchangeItemWidth {
    fn item_words(self) -> u32 {
        match self {
            Self::Word32 => 1,
            Self::Paired64 => 2,
        }
    }

    fn item_count(self, words: u32) -> Result<u32, ExchangeLoweringError> {
        let item_words = self.item_words();
        if words == 0 || !words.is_multiple_of(item_words) {
            return Err(ExchangeLoweringError::UnalignedPayload);
        }
        Ok(words / item_words)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExchangeScheduleDestination {
    pub tile: u16,
    pub address: u32,
}

#[derive(Clone, Debug)]
pub struct LoweredExchanges {
    pub phases: Vec<PhysicalExchangePhase>,
    pub schedule_snapshot: ExchangeScheduleSnapshot,
}

#[derive(Clone, Debug)]
pub struct ExchangeScheduleRun {
    pub phase: PhysicalExchangePhase,
    pub initial_horizon: u32,
    pub endpoint_lower_bound: u32,
    pub neighborhood_improvements: usize,
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
    #[error("exchange diagnostic refers to missing tile {0}")]
    DiagnosticTile(u16),
    #[error("invalid exchange-schedule snapshot: {0}")]
    InvalidSnapshot(String),
    #[error("exchange-schedule invariant failed: {0}")]
    Invariant(String),
}

#[cfg(test)]
pub(crate) fn lower_exchanges(
    program: &LowProgram,
    placement: &Placement,
    topology: &Topology,
    enable_diagnostics: bool,
) -> Result<LoweredExchanges, ExchangeLoweringError> {
    lower_exchanges_cached(
        program,
        placement,
        topology,
        enable_diagnostics,
        &mut ExchangeScheduleCache::default(),
    )
}

pub(crate) fn lower_exchanges_cached(
    program: &LowProgram,
    placement: &Placement,
    topology: &Topology,
    enable_diagnostics: bool,
    cache: &mut ExchangeScheduleCache,
) -> Result<LoweredExchanges, ExchangeLoweringError> {
    let mut repeat_inputs = BTreeMap::<BlockValueId, Vec<BlockValueId>>::new();
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
                    prepare_transfer(program, placement, transfer).inspect_err(|error| {
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
            let mut pending = coalesce_pending_transfers(pending);
            attach_repeat_source_addresses(&mut pending, &repeat_inputs, placement)?;
            let ScheduledPending {
                pending,
                receive_counts,
                incoming_bases,
                optimized,
            } = cache.select(phase.id, topology, pending, program.tile_count)?;
            let schedule_problem = schedule_problem(phase.id.index(), &pending);
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
            let OptimizedSchedule {
                schedule,
                initial_horizon,
                endpoint_lower_bound,
                selected_kind,
                neighborhood_improvements,
            } = optimized;
            if enable_diagnostics {
                let repeat_iterations = pending
                    .iter()
                    .map(|transfer| transfer.source_addresses.len())
                    .max()
                    .unwrap_or(1);
                if repeat_iterations > 1 {
                    let mut unsafe_pending = pending.clone();
                    for transfer in &mut unsafe_pending {
                        transfer.source_addresses.truncate(1);
                        transfer.refresh_source_elements();
                    }
                    let unsafe_schedule = optimize_pending_schedule(
                        topology,
                        &unsafe_pending,
                        &incoming_bases,
                        &receive_counts,
                        program.tile_count,
                    )?;
                    tracing::info!(
                        phase = phase.id.index(),
                        repeat_iterations,
                        unsafe_horizon = unsafe_schedule.schedule.horizon,
                        repeat_safe_horizon = schedule.horizon,
                        repeat_safety_cost = schedule
                            .horizon
                            .saturating_sub(unsafe_schedule.schedule.horizon),
                        "compared repeat-safe exchange schedule with first-iteration-only baseline"
                    );
                }
            }
            if pending.len() > 1_000 {
                tracing::info!(
                    phase = phase.id.index(),
                    initial_horizon,
                    selected_horizon = schedule.horizon,
                    endpoint_lower_bound,
                    lower_bound_gap = schedule.horizon.saturating_sub(endpoint_lower_bound),
                    selected_kind,
                    neighborhood_improvements,
                    "optimized physical exchange schedule"
                );
            }
            let MaterializedSchedule {
                builder,
                horizon,
                tile_availability,
                activities,
                scheduled_sends,
                order,
                timings,
                ..
            } = schedule;
            let mut diagnostics =
                enable_diagnostics.then(|| PhaseDiagnostics::new(program.tile_count));
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
                        transfer.source_address(),
                        &transfer.destinations,
                        transfer.words,
                        timing.start,
                        timing.end,
                        timing.blocking_tile,
                    );
                }
            }
            if let Some(diagnostics) = diagnostics {
                diagnostics.emit(
                    phase.id.index(),
                    &phase.provenance,
                    horizon,
                    &tile_availability,
                    &builder,
                );
            }
            let phase_programs = builder.finish()?;
            debug_assert_eq!(phase_programs.event_cycles, horizon);
            let tile_event_cycles = phase_programs.tile_event_cycles;
            let active = phase_programs
                .programs
                .iter()
                .map(Option::is_some)
                .collect::<Vec<_>>();
            let programs = phase_programs
                .programs
                .into_iter()
                .map(|program| program.unwrap_or_else(inactive_exchange_program))
                .collect::<Vec<_>>();
            let repeat_patches = programs
                .iter()
                .enumerate()
                .map(|(tile, program)| {
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
            Ok((
                PhysicalExchangePhase {
                    id: phase.id,
                    active,
                    programs,
                    incoming_bases,
                    tile_event_cycles,
                    event_cycles: horizon,
                    activities,
                    repeat_patches,
                },
                schedule_problem,
            ))
        })
        .collect::<Result<Vec<_>, ExchangeLoweringError>>()
        .map(|lowered| {
            let (phases, schedule_phases) = lowered.into_iter().unzip();
            LoweredExchanges {
                phases,
                schedule_snapshot: ExchangeScheduleSnapshot {
                    schema_version: EXCHANGE_SCHEDULE_SNAPSHOT_VERSION,
                    tile_count: program.tile_count,
                    phases: schedule_phases,
                },
            }
        })
}

fn prepare_transfer(
    program: &LowProgram,
    placement: &Placement,
    transfer: &LogicalExchange,
) -> Result<Vec<PendingTransfer>, ExchangeLoweringError> {
    let source = &program.shards[transfer.source.shard.index() as usize];
    let logical_order = transfer.span_order(&program.shards) == crate::CopyOrder::Semantic;
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
            source_addresses: vec![source_address],
            source_elements: effective_memory_elements(source_address, chunk_bytes / 4),
            words: chunk_bytes / 4,
            width: ExchangeItemWidth::Word32,
            reserved_source: None,
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
    source: u16,
    source_shard: BlockValueId,
    source_offset: u32,
    destinations: Vec<(u16, u32)>,
    source_addresses: Vec<u32>,
    source_elements: Vec<ExchangeMemoryElement>,
    words: u32,
    width: ExchangeItemWidth,
    reserved_source: Option<u16>,
}

impl PendingTransfer {
    fn tiles(&self) -> impl Iterator<Item = u16> + '_ {
        std::iter::once(self.source)
            .chain(self.reserved_source)
            .chain(self.destinations.iter().map(|entry| entry.0))
    }

    fn source_address(&self) -> u32 {
        self.source_addresses[0]
    }

    fn item_count(&self) -> Result<u32, ExchangeLoweringError> {
        self.width.item_count(self.words)
    }

    fn refresh_source_elements(&mut self) {
        self.source_elements = self
            .source_addresses
            .iter()
            .flat_map(|&address| effective_memory_elements(address, self.words))
            .collect();
        self.source_elements.sort_unstable();
        self.source_elements.dedup();
    }
}

fn attach_repeat_source_addresses(
    pending: &mut [PendingTransfer],
    repeat_inputs: &BTreeMap<BlockValueId, Vec<BlockValueId>>,
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
            if addresses.first().copied() != Some(transfer.source_address()) {
                return Err(ExchangeLoweringError::IncompatibleRepeatRows);
            }
            transfer.source_addresses = addresses;
        }
        transfer.refresh_source_elements();
    }
    Ok(())
}

fn paired_transfer_alternatives(
    pending: &[PendingTransfer],
    topology: &Topology,
    tile_count: u16,
) -> Result<Vec<Option<PendingTransfer>>, ExchangeLoweringError> {
    let mut alternatives = Vec::with_capacity(pending.len());
    for transfer in pending {
        if transfer.width != ExchangeItemWidth::Word32
            || transfer.words < 128
            || transfer.words & 1 != 0
            || transfer
                .source_addresses
                .iter()
                .any(|address| address & 0b111 != 0)
        {
            alternatives.push(None);
            continue;
        }
        let source_pair = topology.paired_logical(transfer.source)?;
        if source_pair >= tile_count {
            alternatives.push(None);
            continue;
        }

        let mut by_pair = BTreeMap::<u16, Vec<(u16, u32)>>::new();
        for &(tile, address) in &transfer.destinations {
            by_pair
                .entry(topology.physical(tile)? & !2)
                .or_default()
                .push((tile, address));
        }
        let mut paired_destinations = Vec::with_capacity(transfer.destinations.len());
        let all_destinations_pairable = by_pair.into_values().all(|destinations| {
            let complete_pair = destinations.len() == 2
                && topology
                    .paired_logical(destinations[0].0)
                    .is_ok_and(|paired| paired == destinations[1].0);
            let pairable = complete_pair
                && destinations
                    .iter()
                    .all(|(tile, address)| *tile != source_pair && address & 0b111 == 0)
                && destinations[0].1 == destinations[1].1;
            if pairable {
                paired_destinations.extend(destinations);
            }
            pairable
        });
        if !all_destinations_pairable || paired_destinations.is_empty() {
            alternatives.push(None);
            continue;
        }

        let paired_tiles = paired_destinations
            .iter()
            .map(|&(tile, _)| tile)
            .collect::<Vec<_>>();
        if topology
            .paired_multicast(transfer.source, &paired_tiles, transfer.words / 2)
            .is_err()
        {
            alternatives.push(None);
            continue;
        }
        let mut paired = transfer.clone();
        paired.destinations = paired_destinations;
        paired.width = ExchangeItemWidth::Paired64;
        paired.reserved_source = Some(source_pair);
        alternatives.push(Some(paired));
    }
    Ok(alternatives)
}

fn schedule_problem(phase: u32, pending: &[PendingTransfer]) -> ExchangeScheduleProblem {
    ExchangeScheduleProblem {
        phase,
        transfers: pending
            .iter()
            .map(|transfer| ExchangeScheduleTransfer {
                source: transfer.source,
                source_addresses: transfer.source_addresses.clone(),
                destinations: transfer
                    .destinations
                    .iter()
                    .map(|&(tile, address)| ExchangeScheduleDestination { tile, address })
                    .collect(),
                words: transfer.words,
                width: transfer.width,
            })
            .collect(),
    }
}

fn pending_from_problem(
    tile_count: u16,
    problem: &ExchangeScheduleProblem,
) -> Result<Vec<PendingTransfer>, ExchangeLoweringError> {
    problem
        .transfers
        .iter()
        .enumerate()
        .map(|(index, transfer)| {
            if transfer.source >= tile_count {
                return Err(ExchangeLoweringError::InvalidSnapshot(format!(
                    "phase {} transfer {index} has source tile {} outside 0..{tile_count}",
                    problem.phase, transfer.source
                )));
            }
            if transfer.words == 0 || transfer.words > MAX_TRANSFER_WORDS {
                return Err(ExchangeLoweringError::InvalidSnapshot(format!(
                    "phase {} transfer {index} has invalid word count {}",
                    problem.phase, transfer.words
                )));
            }
            if transfer.width == ExchangeItemWidth::Paired64
                && (transfer.words < 128 || transfer.words & 1 != 0)
            {
                return Err(ExchangeLoweringError::InvalidSnapshot(format!(
                    "phase {} transfer {index} has invalid {}-word paired payload",
                    problem.phase, transfer.words
                )));
            }
            if transfer.source_addresses.is_empty() {
                return Err(ExchangeLoweringError::InvalidSnapshot(format!(
                    "phase {} transfer {index} has no source addresses",
                    problem.phase
                )));
            }
            let bytes = transfer
                .words
                .checked_mul(4)
                .ok_or(ExchangeLoweringError::Overflow)?;
            for &address in &transfer.source_addresses {
                let alignment_mask = match transfer.width {
                    ExchangeItemWidth::Word32 => 0b11,
                    ExchangeItemWidth::Paired64 => 0b111,
                };
                if address & alignment_mask != 0 {
                    return Err(ExchangeLoweringError::InvalidSnapshot(format!(
                        "phase {} transfer {index} has unaligned source address {address:#x}",
                        problem.phase
                    )));
                }
                address
                    .checked_add(bytes)
                    .ok_or(ExchangeLoweringError::Overflow)?;
            }
            if transfer.destinations.is_empty() {
                return Err(ExchangeLoweringError::InvalidSnapshot(format!(
                    "phase {} transfer {index} has no destinations",
                    problem.phase
                )));
            }
            let mut destination_tiles = BTreeSet::new();
            let destinations = transfer
                .destinations
                .iter()
                .map(|destination| {
                    if destination.tile >= tile_count
                        || destination.tile == transfer.source
                        || !destination_tiles.insert(destination.tile)
                    {
                        return Err(ExchangeLoweringError::InvalidSnapshot(format!(
                            "phase {} transfer {index} has invalid destination tile {}",
                            problem.phase, destination.tile
                        )));
                    }
                    let alignment_mask = match transfer.width {
                        ExchangeItemWidth::Word32 => 0b11,
                        ExchangeItemWidth::Paired64 => 0b111,
                    };
                    if destination.address & alignment_mask != 0 {
                        return Err(ExchangeLoweringError::InvalidSnapshot(format!(
                            "phase {} transfer {index} has unaligned destination address {:#x}",
                            problem.phase, destination.address
                        )));
                    }
                    destination
                        .address
                        .checked_add(bytes)
                        .ok_or(ExchangeLoweringError::Overflow)?;
                    Ok((destination.tile, destination.address))
                })
                .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
            let mut pending = PendingTransfer {
                source: transfer.source,
                source_shard: BlockValueId::from_index(
                    u32::try_from(index).map_err(|_| ExchangeLoweringError::Overflow)?,
                ),
                source_offset: 0,
                destinations,
                source_addresses: transfer.source_addresses.clone(),
                source_elements: Vec::new(),
                words: transfer.words,
                width: transfer.width,
                reserved_source: match transfer.width {
                    ExchangeItemWidth::Word32 => None,
                    ExchangeItemWidth::Paired64 => {
                        Some(Topology::c600().paired_logical(transfer.source)?)
                    }
                },
            };
            pending.refresh_source_elements();
            Ok(pending)
        })
        .collect()
}

fn receive_configuration(
    pending: &[PendingTransfer],
    tile_count: u16,
) -> Result<(Vec<usize>, Vec<u32>), ExchangeLoweringError> {
    let mut receive_counts = vec![0usize; usize::from(tile_count)];
    for transfer in pending {
        for &(tile, _) in &transfer.destinations {
            let count = receive_counts
                .get_mut(usize::from(tile))
                .ok_or(ExchangeLoweringError::InvalidDestination)?;
            *count += 1;
        }
    }
    let mut incoming_bases = vec![None::<u32>; usize::from(tile_count)];
    for transfer in pending {
        if let [(tile, address)] = transfer.destinations.as_slice()
            && receive_counts[usize::from(*tile)] == 1
        {
            incoming_bases[usize::from(*tile)] = Some(*address);
        }
    }
    Ok((
        receive_counts,
        incoming_bases
            .into_iter()
            .map(|base| base.unwrap_or(0))
            .collect(),
    ))
}

struct OptimizedSchedule {
    schedule: MaterializedSchedule,
    initial_horizon: u32,
    endpoint_lower_bound: u32,
    selected_kind: &'static str,
    neighborhood_improvements: usize,
}

struct ScheduledPending {
    pending: Vec<PendingTransfer>,
    receive_counts: Vec<usize>,
    incoming_bases: Vec<u32>,
    optimized: OptimizedSchedule,
}

fn optimize_owned_pending(
    topology: &Topology,
    pending: Vec<PendingTransfer>,
    tile_count: u16,
) -> Result<ScheduledPending, ExchangeLoweringError> {
    let (receive_counts, incoming_bases) = receive_configuration(&pending, tile_count)?;
    let optimized = optimize_pending_schedule(
        topology,
        &pending,
        &incoming_bases,
        &receive_counts,
        tile_count,
    )?;
    Ok(ScheduledPending {
        pending,
        receive_counts,
        incoming_bases,
        optimized,
    })
}

/// Compare complete width choices. A single width change can leave another
/// path tied at the horizon, so individually profitable transfers are not a
/// useful prerequisite for pairing. This bounds search to two optimizations.
fn select_transfer_widths(
    phase: u32,
    topology: &Topology,
    pending: Vec<PendingTransfer>,
    tile_count: u16,
) -> Result<ScheduledPending, ExchangeLoweringError> {
    let alternatives = paired_transfer_alternatives(&pending, topology, tile_count)?;
    let candidates = alternatives.iter().flatten().count();
    if candidates == 0 {
        return optimize_owned_pending(topology, pending, tile_count);
    }
    let paired = pending
        .iter()
        .zip(alternatives)
        .map(|(ordinary, paired)| paired.unwrap_or_else(|| ordinary.clone()))
        .collect();
    let ordinary = optimize_owned_pending(topology, pending, tile_count)?;
    let paired = match optimize_owned_pending(topology, paired, tile_count) {
        Ok(paired) => paired,
        Err(error) => {
            tracing::debug!(phase, %error, "paired exchange candidate is not encodable");
            return Ok(ordinary);
        }
    };
    let ordinary_horizon = ordinary.optimized.schedule.horizon;
    let paired_horizon = paired.optimized.schedule.horizon;
    let use_paired = paired_horizon < ordinary_horizon;
    tracing::info!(
        phase,
        candidates,
        ordinary_horizon,
        paired_horizon,
        use_paired,
        "compared ordinary and paired exchange schedules"
    );
    Ok(if use_paired { paired } else { ordinary })
}

fn optimize_pending_schedule(
    topology: &Topology,
    pending: &[PendingTransfer],
    incoming_bases: &[u32],
    receive_counts: &[usize],
    tile_count: u16,
) -> Result<OptimizedSchedule, ExchangeLoweringError> {
    let schedule = materialize_greedy_schedule(
        topology,
        pending,
        incoming_bases,
        receive_counts,
        tile_count,
    )?;
    improve_pending_schedule(
        topology,
        pending,
        incoming_bases,
        receive_counts,
        tile_count,
        schedule,
        "full-duplex",
    )
}

#[allow(clippy::too_many_arguments)]
fn improve_pending_schedule(
    topology: &Topology,
    pending: &[PendingTransfer],
    incoming_bases: &[u32],
    receive_counts: &[usize],
    tile_count: u16,
    mut schedule: MaterializedSchedule,
    initial_kind: &'static str,
) -> Result<OptimizedSchedule, ExchangeLoweringError> {
    let initial_horizon = schedule_score(&schedule);
    let endpoint_lower_bound = endpoint_work_lower_bound(pending, tile_count);
    let mut selected_kind = initial_kind;
    let mut neighborhood_improvements = 0usize;
    if let Some(order) = point_to_point_matching_wave_order(pending, tile_count, &schedule.order) {
        let matching = materialize_schedule_order(
            topology,
            pending,
            incoming_bases,
            receive_counts,
            tile_count,
            &order,
            false,
        );
        if let Ok(matching) = matching
            && schedule_score(&matching) < schedule_score(&schedule)
        {
            schedule = matching;
            selected_kind = "matching-waves";
        }
    }
    loop {
        let repaired_order = critical_neighborhood_order(pending, tile_count, &schedule);
        if repaired_order == schedule.order {
            break;
        }
        let repaired = materialize_schedule_order(
            topology,
            pending,
            incoming_bases,
            receive_counts,
            tile_count,
            &repaired_order,
            false,
        );
        let Ok(repaired) = repaired else {
            break;
        };
        if schedule_score(&repaired) >= schedule_score(&schedule) {
            break;
        }
        schedule = repaired;
        selected_kind = "critical-neighborhood";
        neighborhood_improvements += 1;
    }
    Ok(OptimizedSchedule {
        schedule,
        initial_horizon,
        endpoint_lower_bound,
        selected_kind,
        neighborhood_improvements,
    })
}

impl ExchangeScheduleSnapshot {
    pub fn validate(&self) -> Result<(), ExchangeLoweringError> {
        if self.schema_version != EXCHANGE_SCHEDULE_SNAPSHOT_VERSION {
            return Err(ExchangeLoweringError::InvalidSnapshot(format!(
                "unsupported schema version {} (expected {})",
                self.schema_version, EXCHANGE_SCHEDULE_SNAPSHOT_VERSION
            )));
        }
        if self.tile_count == 0 || usize::from(self.tile_count) > Topology::c600().tile_count() {
            return Err(ExchangeLoweringError::InvalidSnapshot(format!(
                "tile count {} is outside the C600 topology",
                self.tile_count
            )));
        }
        let mut phases = BTreeSet::new();
        for problem in &self.phases {
            if !phases.insert(problem.phase) {
                return Err(ExchangeLoweringError::InvalidSnapshot(format!(
                    "duplicate phase {}",
                    problem.phase
                )));
            }
            pending_from_problem(self.tile_count, problem)?;
        }
        Ok(())
    }
}

/// Runs the same ordering, timing, full-duplex code generation, and row
/// validation used by package lowering on one captured phase.
pub fn schedule_exchange_problem(
    tile_count: u16,
    problem: &ExchangeScheduleProblem,
) -> Result<ExchangeScheduleRun, ExchangeLoweringError> {
    if tile_count == 0 || usize::from(tile_count) > Topology::c600().tile_count() {
        return Err(ExchangeLoweringError::InvalidSnapshot(format!(
            "tile count {tile_count} is outside the C600 topology"
        )));
    }
    let topology = Topology::new(
        (0..tile_count)
            .map(ipu_exchange::c600_logical_to_physical)
            .collect(),
    )?;
    let pending = pending_from_problem(tile_count, problem)?;
    let (receive_counts, incoming_bases) = receive_configuration(&pending, tile_count)?;
    let OptimizedSchedule {
        schedule,
        initial_horizon,
        endpoint_lower_bound,
        neighborhood_improvements,
        ..
    } = optimize_pending_schedule(
        &topology,
        &pending,
        &incoming_bases,
        &receive_counts,
        tile_count,
    )?;
    let MaterializedSchedule {
        builder,
        horizon,
        activities,
        ..
    } = schedule;
    let phase_programs = builder.finish()?;
    if phase_programs.event_cycles != horizon {
        return Err(ExchangeLoweringError::Invariant(format!(
            "phase {} row horizon {} differs from scheduled horizon {horizon}",
            problem.phase, phase_programs.event_cycles
        )));
    }
    let tile_event_cycles = phase_programs.tile_event_cycles;
    let active = phase_programs
        .programs
        .iter()
        .map(Option::is_some)
        .collect::<Vec<_>>();
    let programs = phase_programs
        .programs
        .into_iter()
        .map(|program| program.unwrap_or_else(inactive_exchange_program))
        .collect::<Vec<_>>();
    let phase = PhysicalExchangePhase {
        id: ExchangePhaseId::from_index(problem.phase),
        active,
        programs,
        incoming_bases,
        tile_event_cycles,
        event_cycles: horizon,
        activities,
        repeat_patches: vec![Vec::new(); usize::from(tile_count)],
    };
    Ok(ExchangeScheduleRun {
        phase,
        initial_horizon,
        endpoint_lower_bound,
        neighborhood_improvements,
    })
}

/// Checks that scheduled activities and encoded rows preserve the captured
/// transfer set and obey per-tile bus and SRAM-element hazards.
pub fn validate_exchange_schedule(
    tile_count: u16,
    problem: &ExchangeScheduleProblem,
    phase: &PhysicalExchangePhase,
) -> Result<(), ExchangeLoweringError> {
    let fail = |message| ExchangeLoweringError::Invariant(message);
    let size = usize::from(tile_count);
    if phase.id.index() != problem.phase {
        return Err(fail(format!(
            "phase id {} differs from snapshot phase {}",
            phase.id.index(),
            problem.phase
        )));
    }
    for (name, length) in [
        ("active", phase.active.len()),
        ("programs", phase.programs.len()),
        ("incoming bases", phase.incoming_bases.len()),
        ("tile horizons", phase.tile_event_cycles.len()),
        ("activities", phase.activities.len()),
        ("repeat patches", phase.repeat_patches.len()),
    ] {
        if length != size {
            return Err(fail(format!(
                "phase {} has {length} {name} entries for {tile_count} tiles",
                problem.phase
            )));
        }
    }
    if phase
        .repeat_patches
        .iter()
        .any(|patches| !patches.is_empty())
    {
        return Err(fail(format!(
            "standalone phase {} unexpectedly contains repeat patches",
            problem.phase
        )));
    }
    let maximum_horizon = phase.tile_event_cycles.iter().copied().max().unwrap_or(0);
    if phase.event_cycles != maximum_horizon {
        return Err(fail(format!(
            "phase {} horizon {} differs from maximum tile horizon {maximum_horizon}",
            problem.phase, phase.event_cycles
        )));
    }

    let mut send_counts = vec![0usize; problem.transfers.len()];
    let mut partner_busy_counts = vec![0usize; problem.transfers.len()];
    let mut receive_counts = problem
        .transfers
        .iter()
        .map(|transfer| vec![0usize; transfer.destinations.len()])
        .collect::<Vec<_>>();
    let reserved_paired_sources = problem
        .transfers
        .iter()
        .filter(|transfer| transfer.width == ExchangeItemWidth::Paired64)
        .map(|transfer| Topology::c600().paired_logical(transfer.source))
        .collect::<Result<BTreeSet<_>, _>>()?;
    for tile in 0..size {
        let decoded = ipu_exchange::diagnostic::diagnose_plan_program(&phase.programs[tile], None)?;
        if decoded.event_cycles != phase.tile_event_cycles[tile] {
            return Err(fail(format!(
                "phase {} tile {tile} decoded horizon {} differs from {}",
                problem.phase, decoded.event_cycles, phase.tile_event_cycles[tile]
            )));
        }
        let tile_u16 = u16::try_from(tile).map_err(|_| ExchangeLoweringError::Overflow)?;
        let expected_active =
            !phase.activities[tile].is_empty() || reserved_paired_sources.contains(&tile_u16);
        if phase.active[tile] != expected_active
            || phase.active[tile] != (phase.tile_event_cycles[tile] != 0)
        {
            return Err(fail(format!(
                "phase {} tile {tile} has inconsistent active state",
                problem.phase
            )));
        }
        for activity in &phase.activities[tile] {
            if activity.start_cycle > activity.end_cycle
                || activity.end_cycle > activity.memory_end_cycle
                || activity.memory_end_cycle > phase.tile_event_cycles[tile]
            {
                return Err(fail(format!(
                    "phase {} tile {tile} transfer {} has invalid cycle interval",
                    problem.phase, activity.transfer
                )));
            }
            let transfer_index =
                usize::try_from(activity.transfer).map_err(|_| ExchangeLoweringError::Overflow)?;
            let transfer = problem.transfers.get(transfer_index).ok_or_else(|| {
                fail(format!(
                    "phase {} tile {tile} references missing transfer {}",
                    problem.phase, activity.transfer
                ))
            })?;
            if activity.words != transfer.words {
                return Err(fail(format!(
                    "phase {} tile {tile} transfer {transfer_index} has wrong word count",
                    problem.phase
                )));
            }
            match activity.kind {
                ExchangeActivityKind::Send => {
                    if usize::from(transfer.source) != tile
                        || activity.address != transfer.source_addresses[0]
                    {
                        return Err(fail(format!(
                            "phase {} transfer {transfer_index} has a mismatched send activity",
                            problem.phase
                        )));
                    }
                    send_counts[transfer_index] += 1;
                }
                ExchangeActivityKind::Receive => {
                    let destination = transfer
                        .destinations
                        .iter()
                        .position(|destination| {
                            usize::from(destination.tile) == tile
                                && destination.address == activity.address
                        })
                        .ok_or_else(|| {
                            fail(format!(
                                "phase {} transfer {transfer_index} has an unexpected receive activity on tile {tile}",
                                problem.phase
                            ))
                        })?;
                    receive_counts[transfer_index][destination] += 1;
                }
                ExchangeActivityKind::PartnerBusy => {
                    let expected = (transfer.width == ExchangeItemWidth::Paired64)
                        .then(|| Topology::c600().paired_logical(transfer.source))
                        .transpose()?;
                    if expected != Some(tile_u16)
                        || activity.address != transfer.source_addresses[0]
                    {
                        return Err(fail(format!(
                            "phase {} transfer {transfer_index} has a mismatched partner-busy activity",
                            problem.phase
                        )));
                    }
                    partner_busy_counts[transfer_index] += 1;
                }
            }
        }
        for kind in [ExchangeActivityKind::Send, ExchangeActivityKind::Receive] {
            let mut intervals = phase.activities[tile]
                .iter()
                .filter(|activity| activity.kind == kind)
                .map(|activity| (activity.start_cycle, activity.end_cycle))
                .collect::<Vec<_>>();
            intervals.sort_unstable();
            if intervals.windows(2).any(|pair| pair[1].0 < pair[0].1) {
                return Err(fail(format!(
                    "phase {} tile {tile} has overlapping {kind:?} bus intervals",
                    problem.phase
                )));
            }
        }
        let sends = phase.activities[tile]
            .iter()
            .filter(|activity| activity.kind == ExchangeActivityKind::Send);
        for send in sends {
            let transfer = &problem.transfers[send.transfer as usize];
            for receive in phase.activities[tile]
                .iter()
                .filter(|activity| activity.kind == ExchangeActivityKind::Receive)
            {
                let overlaps = send.start_cycle < receive.memory_end_cycle
                    && receive.start_cycle < send.memory_end_cycle;
                if overlaps
                    && transfer.source_addresses.iter().any(|&address| {
                        spans_share_effective_memory_element(
                            address,
                            send.words,
                            receive.address,
                            receive.words,
                        )
                    })
                {
                    return Err(fail(format!(
                        "phase {} tile {tile} overlaps send/receive access to one SRAM element",
                        problem.phase
                    )));
                }
            }
        }
        for partner_busy in phase.activities[tile]
            .iter()
            .filter(|activity| activity.kind == ExchangeActivityKind::PartnerBusy)
        {
            if phase.activities[tile].iter().any(|activity| {
                activity.transfer != partner_busy.transfer
                    && activity.kind != ExchangeActivityKind::Receive
                    && activity.start_cycle < partner_busy.end_cycle
                    && partner_busy.start_cycle < activity.end_cycle
            }) {
                return Err(fail(format!(
                    "phase {} tile {tile} overlaps borrowed and local transmit intervals",
                    problem.phase
                )));
            }
        }
    }
    for (index, count) in send_counts.into_iter().enumerate() {
        if count != 1 {
            return Err(fail(format!(
                "phase {} transfer {index} has {count} send activities",
                problem.phase
            )));
        }
    }
    for (index, count) in partner_busy_counts.into_iter().enumerate() {
        let expected = usize::from(problem.transfers[index].width == ExchangeItemWidth::Paired64);
        if count != expected {
            return Err(fail(format!(
                "phase {} transfer {index} has {count} partner-busy activities, expected {expected}",
                problem.phase
            )));
        }
    }
    for (transfer, counts) in receive_counts.into_iter().enumerate() {
        if counts.into_iter().any(|count| count != 1) {
            return Err(fail(format!(
                "phase {} transfer {transfer} does not have exactly one activity per destination",
                problem.phase
            )));
        }
    }
    Ok(())
}

/// Combines physically contiguous source and destination spans into one
/// hardware message. Separate SEND messages require separate receive events,
/// even when they select the same source tile.
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
            && previous.width == transfer.width
            && previous.source_shard == transfer.source_shard
            && previous
                .source_offset
                .checked_add(previous_bytes)
                .is_some_and(|end| end == transfer.source_offset)
            && previous.destinations.len() == transfer.destinations.len()
            && previous.source_addresses.len() == transfer.source_addresses.len()
            && previous
                .source_addresses
                .iter()
                .zip(&transfer.source_addresses)
                .all(|(&left, &right)| {
                    left.checked_add(previous_bytes)
                        .is_some_and(|end| end == right)
                })
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
            let items = u64::from(transfer.item_count().unwrap_or(transfer.words));
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
                .any(|transfer| transfer.destinations.len() > 1),
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
            fanout: transfer.destinations.len(),
            words: transfer.item_count().unwrap_or(transfer.words),
            source: Reverse(transfer.source),
            index: Reverse(index),
        });
    }

    fn next(&mut self, tile_availability: &[TileAvailability]) -> Option<(usize, u32)> {
        loop {
            let candidate = self.ready.pop()?;
            let index = candidate.index.0;
            let transfer = &self.transfers[index];
            let earliest_start = std::iter::once(self.dependency_ready[index])
                .chain(std::iter::once(
                    tile_availability[usize::from(transfer.source)].send,
                ))
                .chain(transfer.reserved_source.into_iter().map(|tile| {
                    let availability = tile_availability[usize::from(tile)];
                    availability.send
                }))
                .chain(
                    transfer
                        .destinations
                        .iter()
                        .map(|&(tile, _)| tile_availability[usize::from(tile)].receive),
                )
                .max()
                .unwrap_or(0);
            if candidate.earliest_start.0 == earliest_start {
                // Endpoint availability ranks the ready queue, but is not a
                // dependency on payload arrival. The row builder pipelines
                // source selection and delivery using their actual timings.
                return Some((index, self.dependency_ready[index]));
            }
            self.push_ready(index, earliest_start);
        }
    }

    fn complete(&mut self, index: usize, completion: u32) {
        self.completed += 1;
        let transfer = &self.transfers[index];
        if self.dynamic_word_pressure {
            let items = u64::from(transfer.item_count().unwrap_or(transfer.words));
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
        let bytes = u64::from(transfer.words) * 4;
        for &address in &transfer.source_addresses {
            accesses[usize::from(transfer.source)].push(TransferAccess {
                transfer: index,
                start: u64::from(address),
                end: u64::from(address) + bytes,
                write: false,
            });
        }
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
    source_elements: &'a [ExchangeMemoryElement],
    words: u32,
    width: ExchangeItemWidth,
    schedule_offset: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct TileAvailability {
    send: u32,
    receive: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct TilePredecessor {
    send: Option<usize>,
    receive: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct MaterializedTiming {
    start: u32,
    end: u32,
    blocking_tile: u16,
    predecessor: Option<usize>,
}

#[derive(Clone, Debug)]
struct MemoryAccess {
    start: u32,
    end: u32,
    elements: Vec<ExchangeMemoryElement>,
}

#[derive(Clone, Debug, Default)]
struct TileMemorySchedule {
    sends: Vec<MemoryAccess>,
    receives: Vec<MemoryAccess>,
}

struct MaterializedSchedule {
    builder: PhaseProgramBuilder,
    horizon: u32,
    tile_availability: Vec<TileAvailability>,
    memory_accesses: Vec<TileMemorySchedule>,
    activities: Vec<Vec<ExchangeActivity>>,
    scheduled_sends: Vec<Vec<(BlockValueId, u32)>>,
    order: Vec<usize>,
    timings: Vec<Option<MaterializedTiming>>,
}

impl MaterializedSchedule {
    fn new(tile_count: u16, transfers: &[PendingTransfer]) -> Self {
        let transfer_count = transfers.len();
        Self {
            builder: PhaseProgramBuilder::new(tile_count),
            horizon: 0,
            tile_availability: vec![TileAvailability::default(); usize::from(tile_count)],
            memory_accesses: (0..tile_count)
                .map(|_| TileMemorySchedule::default())
                .collect(),
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
        dependency_ready: u32,
        validate_encoding: bool,
        last_transfer: &mut [TilePredecessor],
    ) -> Result<u32, ExchangeLoweringError> {
        let transfer = &pending[index];
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
            ScheduledTransfer {
                source: transfer.source,
                destinations: &transfer.destinations,
                source_address: transfer.source_address(),
                source_elements: &transfer.source_elements,
                words: transfer.words,
                width: transfer.width,
                // Endpoint constraints are enforced at their actual source,
                // payload and control events by the row builder. Only true
                // memory dependencies constrain the whole transfer's release.
                schedule_offset: dependency_ready,
            },
            &mut self.builder,
            validate_encoding,
        )?;
        let payload_end = timing.sender_end;
        self.memory_accesses[usize::from(transfer.source)]
            .sends
            .push(MemoryAccess {
                start: timing.start,
                end: timing.sender_memory_end,
                elements: transfer.source_elements.clone(),
            });
        for ((&(tile, address), &start), &memory_end) in transfer
            .destinations
            .iter()
            .zip(&timing.receiver_starts)
            .zip(&timing.receiver_memory_ends)
        {
            self.memory_accesses[usize::from(tile)]
                .receives
                .push(MemoryAccess {
                    start,
                    end: memory_end,
                    elements: effective_memory_elements(address, transfer.words),
                });
        }
        self.scheduled_sends[usize::from(transfer.source)]
            .push((transfer.source_shard, transfer.source_offset));
        self.activities[usize::from(transfer.source)].push(ExchangeActivity {
            fanout: transfer.destinations.len() as u16,
            paired: transfer.width == ExchangeItemWidth::Paired64,
            transfer: u32::try_from(index).map_err(|_| ExchangeLoweringError::Overflow)?,
            kind: ExchangeActivityKind::Send,
            start_cycle: timing.start,
            end_cycle: payload_end,
            memory_end_cycle: timing.sender_memory_end,
            address: transfer.source_address(),
            words: transfer.words,
        });
        if let Some(tile) = transfer.reserved_source {
            self.activities[usize::from(tile)].push(ExchangeActivity {
                fanout: transfer.destinations.len() as u16,
                paired: transfer.width == ExchangeItemWidth::Paired64,
                transfer: u32::try_from(index).map_err(|_| ExchangeLoweringError::Overflow)?,
                kind: ExchangeActivityKind::PartnerBusy,
                start_cycle: timing.start,
                end_cycle: timing.sender_memory_end,
                memory_end_cycle: timing.sender_memory_end,
                address: transfer.source_address(),
                words: transfer.words,
            });
        }
        for (((&(tile, address), &start_cycle), &end_cycle), &memory_end_cycle) in transfer
            .destinations
            .iter()
            .zip(&timing.receiver_starts)
            .zip(&timing.receiver_ends)
            .zip(&timing.receiver_memory_ends)
        {
            self.activities[usize::from(tile)].push(ExchangeActivity {
                fanout: transfer.destinations.len() as u16,
                paired: transfer.width == ExchangeItemWidth::Paired64,
                transfer: u32::try_from(index).map_err(|_| ExchangeLoweringError::Overflow)?,
                kind: ExchangeActivityKind::Receive,
                start_cycle,
                end_cycle,
                memory_end_cycle,
                address,
                words: transfer.words,
            });
        }
        self.tile_availability[usize::from(transfer.source)].send = timing.sender_end;
        if let Some(tile) = transfer.reserved_source {
            self.tile_availability[usize::from(tile)].send = timing.sender_memory_end;
            last_transfer[usize::from(tile)].send = Some(index);
        }
        for (&(tile, _), &receiver_end) in transfer.destinations.iter().zip(&timing.receiver_ends) {
            self.tile_availability[usize::from(tile)].receive = receiver_end;
        }
        last_transfer[usize::from(transfer.source)].send = Some(index);
        for &(tile, _) in &transfer.destinations {
            last_transfer[usize::from(tile)].receive = Some(index);
        }
        self.order.push(index);
        self.timings[index] = Some(MaterializedTiming {
            start: timing.start,
            end: timing.end,
            blocking_tile,
            predecessor,
        });
        Ok(timing.end)
    }

    fn finish_horizon(&mut self) {
        self.horizon = self.builder.event_cycles();
    }
}

fn materialize_greedy_schedule(
    topology: &Topology,
    pending: &[PendingTransfer],
    incoming_bases: &[u32],
    receive_counts: &[usize],
    tile_count: u16,
) -> Result<MaterializedSchedule, ExchangeLoweringError> {
    let schedule = materialize_greedy_schedule_impl(
        topology,
        pending,
        incoming_bases,
        receive_counts,
        tile_count,
        false,
    )?;
    if schedule_encoding_is_valid(&schedule)? {
        return Ok(schedule);
    }
    tracing::debug!(
        transfers = pending.len(),
        "retrying exchange schedule with incremental instruction-alignment validation"
    );
    materialize_greedy_schedule_impl(
        topology,
        pending,
        incoming_bases,
        receive_counts,
        tile_count,
        true,
    )
}

fn materialize_greedy_schedule_impl(
    topology: &Topology,
    pending: &[PendingTransfer],
    incoming_bases: &[u32],
    receive_counts: &[usize],
    tile_count: u16,
    validate_encoding: bool,
) -> Result<MaterializedSchedule, ExchangeLoweringError> {
    let mut schedule = MaterializedSchedule::new(tile_count, pending);
    let mut scheduler = TransferScheduler::new(pending, tile_count);
    let mut last_transfer = vec![TilePredecessor::default(); usize::from(tile_count)];
    while let Some((index, dependency_ready)) = scheduler.next(&schedule.tile_availability) {
        let completion = schedule.append(
            topology,
            pending,
            incoming_bases,
            receive_counts,
            index,
            dependency_ready,
            validate_encoding,
            &mut last_transfer,
        )?;
        scheduler.complete(index, completion);
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
    validate_encoding: bool,
) -> Result<MaterializedSchedule, ExchangeLoweringError> {
    if order.len() != pending.len() {
        return Err(ExchangeLoweringError::Overflow);
    }
    let mut schedule = MaterializedSchedule::new(tile_count, pending);
    let mut last_transfer = vec![TilePredecessor::default(); usize::from(tile_count)];
    let dependencies = memory_dependencies(pending, tile_count);
    let mut predecessors = vec![Vec::new(); pending.len()];
    for (before, after) in dependencies {
        predecessors[after].push(before);
    }
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
        Err(ipu_exchange::ExchangeError::Schedule("SENDPICP instruction alignment").into())
    }
}

fn schedule_encoding_is_valid(
    schedule: &MaterializedSchedule,
) -> Result<bool, ExchangeLoweringError> {
    match schedule.builder.clone().finish() {
        Ok(_) => Ok(true),
        Err(ipu_exchange::ExchangeError::Schedule("SENDPICP instruction alignment")) => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn schedule_score(schedule: &MaterializedSchedule) -> u32 {
    schedule.horizon
}

fn endpoint_work_lower_bound(pending: &[PendingTransfer], tile_count: u16) -> u32 {
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

/// Orders a balanced point-to-point phase as maximum-cardinality waves over
/// its send and receive buses. The result remains only a candidate: the exact
/// row builder decides whether it improves the incumbent schedule.
fn append_transfer(
    topology: &Topology,
    memory_accesses: &[TileMemorySchedule],
    incoming_bases: &[u32],
    receive_counts: &[usize],
    transfer: ScheduledTransfer<'_>,
    builder: &mut PhaseProgramBuilder,
    validate_encoding: bool,
) -> Result<ScheduledTransferTiming, ExchangeLoweringError> {
    let ScheduledTransfer {
        source,
        destinations,
        source_address,
        source_elements,
        words,
        width,
        schedule_offset: requested_offset,
    } = transfer;
    if words == 0 || source_elements.is_empty() {
        return Err(ExchangeLoweringError::UnalignedPayload);
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
        topology.paired_multicast(source, &tiles, item_count)?
    } else if point_receiver {
        let point = topology.point_to_point(source, tiles[0], words)?;
        MulticastPlan {
            sender: point.sender,
            receivers: vec![finalize_point_receiver(
                &point.receiver,
                topology.physical(source)?,
            )?],
        }
    } else {
        topology.multicast(source, &tiles, item_count, 0)?
    };
    patch_sender_address(&mut plan.sender, source_address)?;
    if !point_receiver {
        for (row, (_, address)) in plan.receivers.iter_mut().zip(destinations) {
            patch_receiver_address(row, *address)?;
        }
    }
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
        let receiver_intervals = timing
            .receiver_payload_starts
            .iter()
            .copied()
            .zip(timing.receiver_horizons.iter().copied())
            .collect::<Vec<_>>();
        schedule_offset = schedule_offset.max(memory_safe_transfer_offset(
            memory_accesses,
            source,
            destinations,
            source_elements,
            words,
            timing.payload_start,
            timing.sender_horizon,
            &receiver_intervals,
            schedule_offset,
        )?);
        if schedule_offset == previous {
            break;
        }
    }
    let timing = builder.append_transfer_at(
        source,
        reserved_tiles,
        &tiles,
        &plan,
        schedule_offset,
        item_count,
    )?;
    Ok(ScheduledTransferTiming {
        start: timing.payload_start,
        end: timing
            .receiver_payload_ends
            .iter()
            .copied()
            .chain(std::iter::once(timing.payload_end))
            .max()
            .unwrap_or(timing.payload_end),
        sender_end: timing.payload_end,
        sender_memory_end: timing.sender_horizon,
        receiver_starts: timing.receiver_payload_starts,
        receiver_ends: timing.receiver_payload_ends,
        receiver_memory_ends: timing.receiver_horizons,
    })
}

fn memory_safe_transfer_offset(
    memory_accesses: &[TileMemorySchedule],
    source: u16,
    destinations: &[(u16, u32)],
    source_elements: &[ExchangeMemoryElement],
    words: u32,
    payload_start: u32,
    payload_end: u32,
    receiver_intervals: &[(u32, u32)],
    schedule_offset: u32,
) -> Result<u32, ExchangeLoweringError> {
    let mut safe_offset = schedule_offset;
    let source_clash = memory_accesses[usize::from(source)]
        .receives
        .iter()
        .filter(|access| payload_start < access.end && access.start < payload_end)
        .filter(|access| {
            access
                .elements
                .iter()
                .any(|element| source_elements.binary_search(element).is_ok())
        })
        .map(|access| access.end)
        .max();
    let receiver_clash = destinations
        .iter()
        .zip(receiver_intervals)
        .flat_map(|(&(tile, address), &(receive_start, receive_end))| {
            memory_accesses[usize::from(tile)]
                .sends
                .iter()
                .filter(move |access| receive_start < access.end && access.start < receive_end)
                .filter(move |access| {
                    effective_memory_elements(address, words)
                        .iter()
                        .any(|element| access.elements.contains(element))
                })
                .map(move |access| access.end.saturating_sub(receive_start))
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

fn spans_share_effective_memory_element(
    left_address: u32,
    left_words: u32,
    right_address: u32,
    right_words: u32,
) -> bool {
    effective_memory_elements(left_address, left_words)
        .into_iter()
        .any(|left| {
            effective_memory_elements(right_address, right_words)
                .into_iter()
                .any(|right| left == right)
        })
}

pub(crate) fn effective_memory_elements(address: u32, words: u32) -> Vec<ExchangeMemoryElement> {
    let end = address.saturating_add(words.saturating_mul(4));
    let mut elements = Vec::new();
    let mut cursor = address;
    while cursor < end {
        let interleaved = cursor >= IPU21_INTERLEAVED_MEMORY_BASE;
        let (base, size) = if interleaved {
            (
                IPU21_INTERLEAVED_MEMORY_BASE,
                IPU21_INTERLEAVED_ELEMENT_SIZE,
            )
        } else {
            (0, TILE_MEMORY_ELEMENT_SIZE)
        };
        let index = (cursor - base) / size;
        elements.push(ExchangeMemoryElement { interleaved, index });
        let boundary = base.saturating_add((index + 1).saturating_mul(size));
        cursor = boundary.min(end);
    }
    elements
}

struct ScheduledTransferTiming {
    start: u32,
    end: u32,
    sender_end: u32,
    sender_memory_end: u32,
    receiver_starts: Vec<u32>,
    receiver_ends: Vec<u32>,
    receiver_memory_ends: Vec<u32>,
}

pub fn inactive_exchange_program() -> Vec<u32> {
    vec![RETURN_M10_INSTRUCTION]
}

#[cfg(test)]
mod tests;
