//! Physical exchange programs generated from logical shard transfers.

mod diagnostic;
mod hazards;
use hazards::MemoryHistory;
mod order;
use diagnostic::PhaseDiagnostics;
pub use diagnostic::diagnose_exchange_tile;
use order::{critical_neighborhood_order, point_to_point_matching_wave_order};
mod replay;
pub use replay::{
    ExchangeSchedulingPriority, schedule_exchange_problem, schedule_exchange_problem_with_priority,
};
mod reuse;
mod traffic;
pub use reuse::ExchangeScheduleCache;
pub(crate) use traffic::MappingTraffic;

use crate::{
    BlockValueId, ExchangePhaseId, LogicalExchange, LowProgram, Placement, ShardDefinition,
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
    /// Per-tile Repeat argument and byte offset supplying OUTGOING_BASE.
    /// Present only when every sender in the phase has a uniform relocation.
    pub outgoing_bases: Vec<Option<(BlockValueId, u32)>>,
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
    pub reused: bool,
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
    #[error("structured-repeat exchange incompatibility: {0}")]
    IncompatibleRepeatRows(&'static str),
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

fn repeat_inputs(
    program: &LowProgram,
) -> Result<BTreeMap<BlockValueId, Vec<BlockValueId>>, ExchangeLoweringError> {
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
                    return Err(ExchangeLoweringError::IncompatibleRepeatRows(
                        "argument has inconsistent sequence bindings",
                    ));
                }
                std::collections::btree_map::Entry::Occupied(_) => {}
            }
        }
    }
    Ok(repeat_inputs)
}

fn prepare_phase(
    program: &LowProgram,
    placement: &Placement,
    phase: &crate::low::ExchangePhase,
    repeat_inputs: &BTreeMap<BlockValueId, Vec<BlockValueId>>,
) -> Result<Vec<PendingTransfer>, ExchangeLoweringError> {
    let mut pending: Vec<PendingTransfer> = phase
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
    attach_repeat_source_addresses(&mut pending, repeat_inputs, placement)?;
    Ok(coalesce_pending_transfers(pending))
}

pub(crate) fn capture_exchange_schedule(
    program: &LowProgram,
    placement: &Placement,
) -> Result<ExchangeScheduleSnapshot, ExchangeLoweringError> {
    let repeat_inputs = repeat_inputs(program)?;
    let phases = program
        .exchange_phases
        .par_iter()
        .map(|phase| {
            Ok(schedule_problem(
                phase.id.index(),
                &prepare_phase(program, placement, phase, &repeat_inputs)?,
            ))
        })
        .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
    Ok(ExchangeScheduleSnapshot {
        schema_version: EXCHANGE_SCHEDULE_SNAPSHOT_VERSION,
        tile_count: program.tile_count,
        phases,
    })
}

pub(crate) fn lower_exchanges_cached(
    program: &LowProgram,
    placement: &Placement,
    topology: &Topology,
    enable_diagnostics: bool,
    cache: &mut ExchangeScheduleCache,
) -> Result<LoweredExchanges, ExchangeLoweringError> {
    let repeat_inputs = repeat_inputs(program)?;
    // Each barrier-delimited phase has independent scheduling state. Keep its
    // relocation recipe local to the worker, then restore the cache in order.
    let mut phase_caches = program
        .exchange_phases
        .iter()
        .map(|phase| cache.take_phase(phase.id))
        .collect::<Vec<_>>();
    let span = tracing::Span::current();
    let lowered = program
        .exchange_phases
        .par_iter()
        .zip(phase_caches.par_iter_mut())
        .map(|(phase, cache)| {
            let _entered = span.enter();
            let pending = prepare_phase(program, placement, phase, &repeat_inputs)?;
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
                        cache.stream_words,
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
            let mut programs = phase_programs
                .programs
                .into_iter()
                .map(|program| program.unwrap_or_else(inactive_exchange_program))
                .collect::<Vec<_>>();
            let outgoing_bases = repeat_outgoing_bases(&pending, &placement.shard_addresses, program.tile_count);
            let repeat_patches = programs
                .iter_mut()
                .enumerate()
                .map(|(tile, program)| {
                    let sends = activities[tile].iter()
                        .filter(|activity| activity.kind == ExchangeActivityKind::Send)
                        .map(|activity| &pending[activity.transfer as usize])
                        .collect::<Vec<_>>();
                    let address_groups = sender_address_instruction_groups(program)?;
                    if address_groups.len() != sends.len() {
                        tracing::error!(phase = phase.id.index(), tile,
                            groups = address_groups.len(), sends = sends.len(),
                            row = ?program, sources = ?sends.iter().map(|send| (send.source_shard, send.source_offset)).collect::<Vec<_>>(),
                            "exchange send groups differ from scheduled messages");
                        return Err(ExchangeLoweringError::IncompatibleRepeatRows(
                            "send instruction groups do not match scheduled messages",
                        ));
                    }
                    let mut patches = Vec::new();
                    let base = outgoing_bases[tile].map(|(shard, offset)| {
                        placement.shard_addresses[&shard] + offset
                    });
                    for (instructions, transfer) in address_groups.into_iter().zip(sends) {
                        if let Some(base) = base {
                            for (word_offset, byte_offset) in instructions {
                                let offset = transfer.source_address().checked_sub(base)
                                    .and_then(|offset| offset.checked_add(byte_offset))
                                    .ok_or(ExchangeLoweringError::Overflow)?;
                                patch_sender_instruction(&mut program[word_offset], offset)?;
                            }
                            continue;
                        }
                        let source_shard = transfer.source_shard;
                        let source_offset = transfer.source_offset;
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
                                tracing::error!(phase = phase.id.index(), tile,
                                    source = source_shard.index(), source_offset, word_offset, byte_offset,
                                    expected = program[word_offset], actual = ?values.first(), row = ?program,
                                    "Repeat relocation changes the initial exchange row");
                                return Err(ExchangeLoweringError::IncompatibleRepeatRows(
                                    "relocation changes the first iteration",
                                ));
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
                    outgoing_bases,
                },
                schedule_problem,
            ))
        })
        .collect::<Result<Vec<_>, ExchangeLoweringError>>();
    for phase_cache in phase_caches {
        cache.merge(phase_cache);
    }
    lowered.map(|lowered| {
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
    let order = transfer.span_order(&program.shards);
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
                crate::view_byte_traversal(shard, view, order)?,
            ))
        })
        .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
    if destinations.is_empty() {
        return Err(ExchangeLoweringError::SizeMismatch);
    }
    let source_spans = crate::view_byte_traversal(source, &transfer.source, order)?;
    if destinations
        .iter()
        .any(|(_, _, spans)| spans.byte_len() != source_spans.byte_len())
    {
        return Err(ExchangeLoweringError::SizeMismatch);
    }
    let mut receivers = destinations
        .iter()
        .map(|(_, _, spans)| spans.spans().peekable())
        .collect::<Vec<_>>();
    let mut pending = Vec::new();
    for source_span in source_spans.spans() {
        if source_span.bytes == 0 || source_span.offset & 3 != 0 {
            return Err(ExchangeLoweringError::UnalignedPayload);
        }
        let mut offset = 0;
        while offset < source_span.bytes {
            let mut bytes = (source_span.bytes - offset).min(MAX_TRANSFER_WORDS * 4);
            for receiver in &mut receivers {
                let span = receiver.peek().ok_or(ExchangeLoweringError::SizeMismatch)?;
                if span.offset & 3 != 0 {
                    return Err(ExchangeLoweringError::UnalignedPayload);
                }
                bytes = bytes.min(span.bytes);
            }
            if bytes == 0 || bytes & 3 != 0 {
                return Err(ExchangeLoweringError::UnalignedPayload);
            }
            let source_offset = source_span
                .offset
                .checked_add(offset)
                .ok_or(ExchangeLoweringError::Overflow)?;
            let source_address = source_base
                .checked_add(source_offset)
                .ok_or(ExchangeLoweringError::Overflow)?;
            let destination_entries = destinations
                .iter()
                .zip(&mut receivers)
                .map(|((tile, base, _), receiver)| {
                    let span = receiver
                        .peek_mut()
                        .ok_or(ExchangeLoweringError::SizeMismatch)?;
                    let address = base
                        .checked_add(span.offset)
                        .ok_or(ExchangeLoweringError::Overflow)?;
                    span.offset = span
                        .offset
                        .checked_add(bytes)
                        .ok_or(ExchangeLoweringError::Overflow)?;
                    span.bytes -= bytes;
                    if span.bytes == 0 {
                        receiver.next();
                    }
                    Ok((*tile, address))
                })
                .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
            pending.push(PendingTransfer {
                source: source.tile,
                source_shard: source.id,
                source_offset,
                destinations: destination_entries,
                source_addresses: vec![source_address],
                source_elements: effective_memory_elements(source_address, bytes / 4),
                words: bytes / 4,
                width: ExchangeItemWidth::Word32,
                reserved_source: None,
            });
            offset += bytes;
        }
    }
    if receivers.iter_mut().any(|r| r.peek().is_some()) {
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

    fn pressure_resources(&self, directional: bool) -> impl Iterator<Item = usize> + '_ {
        let stride = if directional { 2 } else { 1 };
        std::iter::once(self.source)
            .chain(self.reserved_source)
            .map(move |tile| usize::from(tile) * stride)
            .chain(
                self.destinations
                    .iter()
                    .map(move |&(tile, _)| usize::from(tile) * stride + usize::from(directional)),
            )
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

/// Keep a single base throughout the timed phase, including all stationary sends.
/// Reuse an existing Repeat pointer rather than creating a second relocation ABI.
fn repeat_outgoing_bases(
    pending: &[PendingTransfer],
    addresses: &BTreeMap<BlockValueId, u32>,
    tile_count: u16,
) -> Vec<Option<(BlockValueId, u32)>> {
    let empty = || vec![None; usize::from(tile_count)];
    let mut first = vec![None::<&PendingTransfer>; usize::from(tile_count)];
    let mut paired = vec![false; usize::from(tile_count)];
    for transfer in pending {
        let tile = usize::from(transfer.source);
        paired[tile] |= transfer.width == ExchangeItemWidth::Paired64;
        if let Some(previous) = first[tile] {
            let count = previous
                .source_addresses
                .len()
                .max(transfer.source_addresses.len());
            for i in 0..count {
                let delta = |t: &PendingTransfer| {
                    t.source_addresses
                        .get(i)
                        .copied()
                        .unwrap_or(t.source_address())
                        .wrapping_sub(t.source_address())
                };
                if delta(previous) != delta(transfer) {
                    return empty();
                }
            }
            if previous.source_address() <= transfer.source_address() {
                continue;
            }
        }
        first[tile] = Some(transfer);
    }
    let mut bases = empty();
    for (tile, transfer) in first.into_iter().enumerate() {
        let Some(transfer) = transfer else { continue };
        if transfer
            .source_addresses
            .iter()
            .all(|&a| a == transfer.source_address())
        {
            continue;
        }
        // A paired row's encoded offsets must remain eight-byte aligned.
        if paired[tile] && !transfer.source_address().is_multiple_of(8) {
            return empty();
        }
        let Some(&address) = addresses.get(&transfer.source_shard) else {
            return empty();
        };
        let Some(offset) = transfer.source_address().checked_sub(address) else {
            return empty();
        };
        bases[tile] = Some((transfer.source_shard, offset));
    }
    bases
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
                return Err(ExchangeLoweringError::IncompatibleRepeatRows(
                    "argument address differs from its first sequence member",
                ));
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
            || transfer.words < 2
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
            // Pairing shares the receive stream, not its SRAM pointer. Each
            // receiver row independently programs its destination address.
            let pairable =
                complete_pair && destinations.iter().all(|(_, address)| address & 0b111 == 0);
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
            if transfer.width == ExchangeItemWidth::Paired64 && transfer.words & 1 != 0 {
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
                        || (destination.tile == transfer.source
                            && (transfer.destinations.len() < 2
                                || transfer.source_addresses.iter().any(|&address| {
                                    spans_share_effective_memory_element(
                                        address,
                                        transfer.words,
                                        destination.address,
                                        transfer.words,
                                    )
                                })))
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
    stream_words: Option<std::num::NonZeroU32>,
) -> Result<ScheduledPending, ExchangeLoweringError> {
    let (receive_counts, incoming_bases) = receive_configuration(&pending, tile_count)?;
    let optimized = optimize_pending_schedule(
        topology,
        &pending,
        &incoming_bases,
        &receive_counts,
        tile_count,
        stream_words,
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
    stream_words: Option<std::num::NonZeroU32>,
) -> Result<ScheduledPending, ExchangeLoweringError> {
    let alternatives = paired_transfer_alternatives(&pending, topology, tile_count)?;
    let candidates = alternatives.iter().flatten().count();
    if candidates == 0 {
        return optimize_owned_pending(topology, pending, tile_count, stream_words);
    }
    let paired = pending
        .iter()
        .zip(alternatives)
        .map(|(ordinary, paired)| paired.unwrap_or_else(|| ordinary.clone()))
        .collect();
    let ordinary = optimize_owned_pending(topology, pending, tile_count, stream_words);
    let paired = optimize_owned_pending(topology, paired, tile_count, stream_words);
    let (ordinary, paired) = match (ordinary, paired) {
        (Ok(ordinary), Ok(paired)) => (ordinary, paired),
        (ordinary, Err(error)) => {
            tracing::debug!(phase, %error, "paired exchange candidate is not encodable");
            return ordinary;
        }
        (Err(error), Ok(paired)) => {
            tracing::debug!(phase, %error, "ordinary exchange candidate is not encodable");
            return Ok(paired);
        }
    };
    let ordinary_horizon = ordinary.optimized.schedule.horizon;
    let paired_horizon = paired.optimized.schedule.horizon;
    let use_paired = if stream_words.is_some() {
        let storage_score = |candidate: &ScheduledPending| -> Result<_, ExchangeLoweringError> {
            let (maximum, total) = encoded_row_storage(&candidate.optimized.schedule)?;
            Ok((maximum, total, candidate.optimized.schedule.horizon))
        };
        storage_score(&paired)? < storage_score(&ordinary)?
    } else {
        paired_horizon < ordinary_horizon
    };
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
    stream_words: Option<std::num::NonZeroU32>,
) -> Result<OptimizedSchedule, ExchangeLoweringError> {
    let problem = SchedulingProblem::new(pending, tile_count);
    if let Some(words) = stream_words {
        return replay::balanced_stream_schedule(
            topology,
            &problem,
            incoming_bases,
            receive_counts,
            words.get(),
        );
    }
    let schedule = materialize_greedy_schedule(topology, &problem, incoming_bases, receive_counts)?;
    improve_pending_schedule(
        topology,
        &problem,
        incoming_bases,
        receive_counts,
        schedule,
        "full-duplex",
    )
}

#[allow(clippy::too_many_arguments)]
fn improve_pending_schedule(
    topology: &Topology,
    problem: &SchedulingProblem<'_>,
    incoming_bases: &[u32],
    receive_counts: &[usize],
    mut schedule: MaterializedSchedule,
    initial_kind: &'static str,
) -> Result<OptimizedSchedule, ExchangeLoweringError> {
    let pending = problem.transfers;
    let tile_count = problem.tile_count;
    let initial_horizon = schedule_score(&schedule);
    let endpoint_lower_bound = endpoint_work_lower_bound(pending, tile_count);
    let mut selected_kind = initial_kind;
    let mut neighborhood_improvements = 0usize;
    if let Some(order) = point_to_point_matching_wave_order(problem, &schedule.order) {
        let matching = materialize_schedule_order(
            topology,
            problem,
            incoming_bases,
            receive_counts,
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
        let repaired_order =
            critical_neighborhood_order(problem, &schedule, false).unwrap_or_else(|| {
                critical_neighborhood_order(problem, &schedule, true)
                    .expect("local repair has no work limit")
            });
        if repaired_order == schedule.order {
            break;
        }
        let repaired = materialize_schedule_order(
            topology,
            problem,
            incoming_bases,
            receive_counts,
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
fn finish_exchange_run(
    tile_count: u16,
    phase_id: u32,
    incoming_bases: Vec<u32>,
    optimized: OptimizedSchedule,
) -> Result<ExchangeScheduleRun, ExchangeLoweringError> {
    let OptimizedSchedule {
        schedule,
        initial_horizon,
        endpoint_lower_bound,
        neighborhood_improvements,
        selected_kind,
    } = optimized;
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
            phase_id, phase_programs.event_cycles
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
        id: ExchangePhaseId::from_index(phase_id),
        active,
        programs,
        incoming_bases,
        tile_event_cycles,
        event_cycles: horizon,
        activities,
        repeat_patches: vec![Vec::new(); usize::from(tile_count)],
        outgoing_bases: vec![None; usize::from(tile_count)],
    };
    Ok(ExchangeScheduleRun {
        phase,
        initial_horizon,
        endpoint_lower_bound,
        neighborhood_improvements,
        reused: selected_kind == "reused",
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
        ("outgoing bases", phase.outgoing_bases.len()),
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
        || phase.outgoing_bases.iter().any(Option::is_some)
    {
        return Err(fail(format!(
            "standalone phase {} unexpectedly contains repeat relocation",
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
        // A merged phase may contain receive-then-forward dependencies. Do
        // not absorb a local write into a SEND that also reads those bytes;
        // memory_dependencies must still see the original ordered transfers.
        let independent = contiguous
            && combined_words.is_some_and(|words| {
                let bytes = u64::from(words) * 4;
                previous.destinations.iter().all(|&(tile, address)| {
                    tile != previous.source
                        || previous.source_addresses.iter().all(|&source| {
                            u64::from(address) + bytes <= u64::from(source)
                                || u64::from(source) + bytes <= u64::from(address)
                        })
                })
            });
        if contiguous
            && independent
            && combined_words.is_some_and(|words| words <= MAX_TRANSFER_WORDS)
        {
            previous.words = combined_words.expect("checked above");
            previous.refresh_source_elements();
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
    fanout: u16,
    words: u32,
    source: Reverse<u16>,
    index: Reverse<usize>,
}

/// Address/width-dependent facts shared by every trial for one physical phase.
struct SchedulingProblem<'a> {
    transfers: &'a [PendingTransfer],
    tile_count: u16,
    predecessors: Vec<Vec<usize>>,
    dependents: Vec<Vec<usize>>,
    word_pressure: Vec<u64>,
}

impl<'a> SchedulingProblem<'a> {
    fn new(transfers: &'a [PendingTransfer], tile_count: u16) -> Self {
        let mut predecessors = vec![Vec::new(); transfers.len()];
        let mut dependents = vec![Vec::new(); transfers.len()];
        for (before, after) in memory_dependencies(transfers, tile_count) {
            predecessors[after].push(before);
            dependents[before].push(after);
        }
        let mut word_pressure = vec![0; usize::from(tile_count)];
        for transfer in transfers {
            let words = u64::from(transfer.item_count().unwrap_or(transfer.words));
            for tile in transfer.tiles() {
                word_pressure[usize::from(tile)] += words;
            }
        }
        Self {
            transfers,
            tile_count,
            predecessors,
            dependents,
            word_pressure,
        }
    }
    fn indegrees(&self) -> Vec<usize> {
        self.predecessors.iter().map(Vec::len).collect()
    }
}

/// Incrementally list-schedules dependency-ready multicast hyperedges. Heap
/// keys are lower bounds on their start time and are refreshed lazily as
/// shared endpoints become busy.
struct TransferScheduler<'a> {
    transfers: &'a [PendingTransfer],
    word_pressure: Vec<u64>,
    dynamic_word_pressure: bool,
    directional_pressure: bool,
    dependents: &'a [Vec<usize>],
    indegrees: Vec<usize>,
    dependency_ready: Vec<u32>,
    ready: BinaryHeap<ReadyTransfer>,
    ready_groups: Vec<BinaryHeap<ReadyTransfer>>,
    transfer_group: Vec<usize>,
    completed: usize,
}

impl<'a> TransferScheduler<'a> {
    #[cfg(test)]
    fn new(problem: &'a SchedulingProblem<'_>) -> Self {
        Self::with_priority(problem, ExchangeSchedulingPriority::Combined)
    }

    fn with_priority(
        problem: &'a SchedulingProblem<'_>,
        priority: ExchangeSchedulingPriority,
    ) -> Self {
        let transfers = problem.transfers;
        let priority = match priority {
            ExchangeSchedulingPriority::Automatic => {
                // Point-to-point traffic benefits from draining the remaining
                // send/receive workloads independently. Multicast choices free
                // several receivers together and retain their combined pressure.
                if transfers
                    .iter()
                    .all(|transfer| transfer.destinations.len() == 1)
                {
                    ExchangeSchedulingPriority::RemainingDirectional
                } else {
                    ExchangeSchedulingPriority::Combined
                }
            }
            priority => priority,
        };

        let directional = matches!(
            priority,
            ExchangeSchedulingPriority::Directional
                | ExchangeSchedulingPriority::RemainingDirectional
        );
        let word_pressure = if directional {
            let mut pressure = vec![0; usize::from(problem.tile_count) * 2];
            for transfer in transfers {
                for resource in transfer.pressure_resources(true) {
                    pressure[resource] +=
                        u64::from(transfer.item_count().unwrap_or(transfer.words));
                }
            }
            pressure
        } else {
            problem.word_pressure.clone()
        };
        let mut scheduler = Self {
            transfers,
            word_pressure,
            directional_pressure: directional,
            // Both production policies update remaining work. Static pressure
            // remains available for controlled offline comparisons.
            dynamic_word_pressure: matches!(
                priority,
                ExchangeSchedulingPriority::RemainingCombined
                    | ExchangeSchedulingPriority::RemainingDirectional
            ) || (priority == ExchangeSchedulingPriority::Combined
                && transfers
                    .iter()
                    .any(|transfer| transfer.destinations.len() > 1)),
            dependents: &problem.dependents,
            indegrees: problem.indegrees(),
            dependency_ready: vec![0; transfers.len()],
            ready: BinaryHeap::new(),
            ready_groups: Vec::new(),
            transfer_group: Vec::new(),
            completed: 0,
        };
        for index in 0..transfers.len() {
            if scheduler.indegrees[index] == 0 {
                scheduler.push_ready(index, 0);
            }
        }
        // Initially ready transfers sharing the same endpoint roles have the
        // same changing readiness and pressure. Their relative word/index
        // priority is static, so only the best member needs a global entry.
        // Later dependency releases remain individual entries.
        let mut groups = BTreeMap::new();
        scheduler.transfer_group.resize(transfers.len(), usize::MAX);
        for candidate in std::mem::take(&mut scheduler.ready).into_vec() {
            let index = candidate.index.0;
            let transfer = &transfers[index];
            let receivers = transfer
                .destinations
                .iter()
                .map(|&(tile, _)| tile)
                .collect::<Vec<_>>();
            let group = *groups
                .entry((transfer.source, transfer.reserved_source, receivers))
                .or_insert_with(|| {
                    let group = scheduler.ready_groups.len();
                    scheduler.ready_groups.push(BinaryHeap::new());
                    group
                });
            scheduler.transfer_group[index] = group;
            scheduler.ready_groups[group].push(candidate);
        }
        scheduler.ready.extend(
            scheduler
                .ready_groups
                .iter()
                .filter_map(|queue| queue.peek().copied()),
        );
        scheduler
    }

    fn push_ready(&mut self, index: usize, earliest_start: u32) {
        let transfer = &self.transfers[index];
        let endpoint_pressure = transfer
            .pressure_resources(self.directional_pressure)
            // Bytes, rather than role count, approximate how long selecting
            // this hyperedge frees work on the phase's congested endpoints.
            .map(|tile| self.word_pressure[tile])
            .sum::<u64>();
        self.ready.push(ReadyTransfer {
            earliest_start: Reverse(earliest_start),
            endpoint_pressure,
            fanout: u16::try_from(transfer.destinations.len()).expect("receivers fit tile count"),
            words: transfer.item_count().unwrap_or(transfer.words),
            source: Reverse(transfer.source),
            index: Reverse(index),
        });
    }

    fn refresh(
        &self,
        mut candidate: ReadyTransfer,
        availability: &[TileAvailability],
    ) -> ReadyTransfer {
        let index = candidate.index.0;
        let transfer = &self.transfers[index];
        candidate.earliest_start = Reverse(
            std::iter::once(self.dependency_ready[index])
                .chain(std::iter::once(
                    availability[usize::from(transfer.source)].send,
                ))
                .chain(
                    transfer
                        .reserved_source
                        .into_iter()
                        .map(|tile| availability[usize::from(tile)].send),
                )
                .chain(
                    transfer
                        .destinations
                        .iter()
                        .map(|&(tile, _)| availability[usize::from(tile)].receive),
                )
                .max()
                .unwrap_or(0),
        );
        if self.dynamic_word_pressure {
            candidate.endpoint_pressure = transfer
                .pressure_resources(self.directional_pressure)
                .map(|tile| self.word_pressure[tile])
                .sum();
        }
        candidate
    }

    fn next(&mut self, tile_availability: &[TileAvailability]) -> Option<(usize, u32)> {
        let mut repairs = 0;
        loop {
            let candidate = *self.ready.peek()?;
            let current = self.refresh(candidate, tile_availability);
            if candidate == current {
                let index = candidate.index.0;
                // Readiness ranks the queue; payload dependencies alone gate
                // the row builder, which pipelines source selection/delivery.
                if let Some(group) = self
                    .transfer_group
                    .get(index)
                    .copied()
                    .filter(|&group| group != usize::MAX)
                {
                    let queue = &mut self.ready_groups[group];
                    let head = queue.pop().expect("nonempty ready group");
                    debug_assert_eq!(head.index.0, index);
                    if let Some(mut next) = queue.peek().copied() {
                        next.earliest_start = current.earliest_start;
                        *self.ready.peek_mut().expect("ready head") = next;
                    } else {
                        self.ready.pop();
                    }
                } else {
                    self.ready.pop();
                }
                return Some((index, self.dependency_ready[index]));
            }
            repairs += 1;
            // A wave can invalidate most keys. Once logarithmic root repairs
            // cost a linear scan, refresh/reheapify once instead. Charge heap
            // traversal four times the contiguous scan: the large ViT captures
            // benefit from refreshing sooner than comparison counts suggest.
            // Both readiness
            // and pressure are monotone bounds, so this preserves eager priority.
            if self.ready.len() >= 128
                && 4 * repairs * self.ready.len().ilog2() as usize >= self.ready.len()
            {
                let mut entries = std::mem::take(&mut self.ready).into_vec();
                for entry in &mut entries {
                    *entry = self.refresh(*entry, tile_availability);
                }
                self.ready = BinaryHeap::from(entries);
            } else {
                *self.ready.peek_mut().expect("ready head") = current;
            }
        }
    }

    fn complete(&mut self, index: usize, completion: u32) {
        self.completed += 1;
        let transfer = &self.transfers[index];
        if self.dynamic_word_pressure {
            let items = u64::from(transfer.item_count().unwrap_or(transfer.words));
            for tile in transfer.pressure_resources(self.directional_pressure) {
                self.word_pressure[tile] = self.word_pressure[tile].saturating_sub(items);
            }
        }
        for &dependent in &self.dependents[index] {
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

#[derive(Clone, Debug, Default)]
struct TileMemorySchedule {
    sends: MemoryHistory,
    receives: MemoryHistory,
}

struct MaterializedSchedule {
    builder: PhaseProgramBuilder,
    horizon: u32,
    tile_availability: Vec<TileAvailability>,
    memory_accesses: Vec<TileMemorySchedule>,
    activities: Vec<Vec<ExchangeActivity>>,
    order: Vec<usize>,
    timings: Vec<Option<MaterializedTiming>>,
}

impl MaterializedSchedule {
    fn new(tile_count: u16, transfers: &[PendingTransfer]) -> Self {
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
            .record(
                &transfer.source_elements,
                timing.start,
                timing.sender_memory_end,
            );
        for ((&(tile, address), &start), &memory_end) in transfer
            .destinations
            .iter()
            .zip(&timing.receiver_starts)
            .zip(&timing.receiver_memory_ends)
        {
            self.memory_accesses[usize::from(tile)].receives.record(
                &effective_memory_elements(address, transfer.words),
                start,
                memory_end,
            );
        }
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
        // The row builder can fill earlier gaps. Relocation and profiling must
        // follow execution order, not the order in which transfers were chosen.
        for activities in &mut self.activities {
            activities.sort_by_key(|activity| activity.start_cycle);
        }
    }
}

fn materialize_greedy_schedule(
    topology: &Topology,
    problem: &SchedulingProblem<'_>,
    incoming_bases: &[u32],
    receive_counts: &[usize],
) -> Result<MaterializedSchedule, ExchangeLoweringError> {
    materialize_greedy_schedule_with_priority(
        topology,
        problem,
        incoming_bases,
        receive_counts,
        ExchangeSchedulingPriority::Automatic,
    )
}

fn materialize_greedy_schedule_with_priority(
    topology: &Topology,
    problem: &SchedulingProblem<'_>,
    incoming_bases: &[u32],
    receive_counts: &[usize],
    priority: ExchangeSchedulingPriority,
) -> Result<MaterializedSchedule, ExchangeLoweringError> {
    let pending = problem.transfers;
    let schedule = materialize_greedy_schedule_impl(
        topology,
        problem,
        incoming_bases,
        receive_counts,
        false,
        priority,
    )?;
    if schedule_encoding_is_valid(&schedule)? {
        return Ok(schedule);
    }
    tracing::info!(
        transfers = pending.len(),
        "retrying exchange schedule with incremental instruction-alignment validation"
    );
    let started = std::time::Instant::now();
    let result = materialize_greedy_schedule_impl(
        topology,
        problem,
        incoming_bases,
        receive_counts,
        true,
        priority,
    );
    tracing::info!(
        transfers = pending.len(),
        elapsed_ms = started.elapsed().as_millis(),
        success = result.is_ok(),
        "finished incremental instruction-alignment validation"
    );
    result
}

fn materialize_greedy_schedule_impl(
    topology: &Topology,
    problem: &SchedulingProblem<'_>,
    incoming_bases: &[u32],
    receive_counts: &[usize],
    validate_encoding: bool,
    priority: ExchangeSchedulingPriority,
) -> Result<MaterializedSchedule, ExchangeLoweringError> {
    let pending = problem.transfers;
    let tile_count = problem.tile_count;
    let mut schedule = MaterializedSchedule::new(tile_count, pending);
    let mut scheduler = TransferScheduler::with_priority(problem, priority);
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
        Err(ipu_exchange::ExchangeError::Schedule("SENDPICP instruction alignment").into())
    }
}

fn schedule_encoding_is_valid(
    schedule: &MaterializedSchedule,
) -> Result<bool, ExchangeLoweringError> {
    match schedule.builder.finish() {
        Ok(_) => Ok(true),
        Err(ipu_exchange::ExchangeError::Schedule("SENDPICP instruction alignment")) => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn encoded_row_storage(
    schedule: &MaterializedSchedule,
) -> Result<(usize, usize), ExchangeLoweringError> {
    let encoded = schedule.builder.finish()?;
    Ok(encoded
        .programs
        .iter()
        .fold((0, 0), |(maximum, total), row| {
            let words = row.as_ref().map_or(0, |row| row.len());
            (maximum.max(words), total + words)
        }))
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
    let plan = plan.prepare()?;
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
    let source_clash = memory_accesses[usize::from(source)].receives.conflict_end(
        source_elements,
        payload_start,
        payload_end,
    );
    let receiver_clash = destinations
        .iter()
        .zip(receiver_intervals)
        .filter_map(|(&(tile, address), &(start, end))| {
            memory_accesses[usize::from(tile)]
                .sends
                .conflict_end(&effective_memory_elements(address, words), start, end)
                .map(|conflict| conflict.saturating_sub(start))
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

#[cfg(test)]
mod heap_bench;
