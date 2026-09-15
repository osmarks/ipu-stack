//! Physical exchange programs generated from logical shard transfers.
//!
//! The compiler enters through `lower_exchanges_cached`: resolve shard addresses,
//! schedule each phase, encode its tile rows, and attach Repeat relocations.
//! Hardware probes can instead construct `TransferPlan`s with `point_to_point`,
//! `multicast`, or `paired_multicast` and assemble them with `PhaseProgramBuilder`.
//! The re-exported host packet/program constructors serve the runtime's host
//! exchange protocol; they do not participate in tile-to-tile scheduling.
//!
//! Selection constructs one SchedulingProblem with transfers and dependency edges.
//! `greedy` schedules ready work against live endpoint availability; `matching`,
//! `repair`, and `streams` propose dependency-respecting orders. All use the same
//! MaterializedSchedule append path for memory hazards and the program builder
//! for instruction compatibility. Policy comparison and acceptance stay here.
//! Tests invoke the same algorithms rather than maintaining implementations.

use ipu_target::ipu21::fabric::Topology;
use ipu_target::ipu21::memory::effective_memory_elements;
pub mod diagnostic;
mod program;
pub use program::*;
mod hazards;
use hazards::MemoryHistory;
mod greedy;
use greedy::ExchangeSchedulingPriority;
mod matching;
mod packet;
mod repair;
mod streams;
pub use diagnostic::diagnose_exchange_tile;
mod relocation;
mod reuse;
pub use reuse::ExchangeScheduleCache;

use crate::{
    BlockValueId, ExchangePhaseId, LogicalExchange, LowProgram, Placement, ShardDefinition,
};

use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalExchangePhase {
    pub id: ExchangePhaseId,
    /// Whether each logical tile participates in this phase's timed program.
    pub active: Vec<bool>,
    /// Synchronization-free timed supervisor program indexed by logical tile.
    pub programs: Vec<crate::exchange::EncodedRow>,
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
    /// Used throughout a row or by its moving-source section; retained in m6.
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ExchangeLoweringError {
    #[error(transparent)]
    Topology(#[from] ipu_target::ipu21::fabric::TopologyError),
    #[error(transparent)]
    Instruction(#[from] ipu_target::ipu21::instruction::InstructionError),
    #[error(transparent)]
    Exchange(#[from] crate::exchange::ExchangeError),
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
    #[error("exchange-schedule invariant failed: {0}")]
    Invariant(String),
}

#[cfg(test)]
pub(crate) fn lower_exchanges(
    program: &LowProgram,
    placement: &Placement,
    topology: &Topology,
) -> Result<Vec<PhysicalExchangePhase>, ExchangeLoweringError> {
    lower_exchanges_cached(
        program,
        placement,
        topology,
        None,
        &mut ExchangeScheduleCache::default(),
    )
}

fn repeat_source_bases(
    program: &LowProgram,
    placement: &Placement,
) -> Result<BTreeMap<BlockValueId, Vec<u32>>, ExchangeLoweringError> {
    let mut repeat_inputs = BTreeMap::<BlockValueId, Vec<BlockValueId>>::new();
    for repeat in &program.repeat_runs {
        for iterated in &repeat.binding.iterated {
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
    if repeat_inputs.is_empty() {
        return Ok(BTreeMap::new());
    }
    program
        .shards
        .iter()
        .filter_map(|shard| {
            crate::low::storage::storage_chain(&program.shards, shard.id)
                .find_map(|(source, offset)| {
                    repeat_inputs.get(&source).map(|inputs| (inputs, offset))
                })
                .map(|(inputs, offset)| {
                    let addresses = inputs
                        .iter()
                        .map(|input| {
                            let address = placement
                                .shard_addresses
                                .get(input)
                                .ok_or(ExchangeLoweringError::UnplacedShard)?;
                            u32::try_from(i64::from(*address) + offset)
                                .map_err(|_| ExchangeLoweringError::Overflow)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok((shard.id, addresses))
                })
        })
        .collect()
}

fn prepare_phase(
    program: &LowProgram,
    placement: &Placement,
    phase: &crate::low::ExchangePhase,
    repeat_inputs: &BTreeMap<BlockValueId, Vec<u32>>,
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
                    source_storage = ?program.shards[transfer.source.shard.index() as usize],
                    order = ?transfer.order,
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
    attach_repeat_source_addresses(&mut pending, repeat_inputs)?;
    Ok(coalesce_pending_transfers(pending))
}

pub(crate) fn lower_exchanges_cached(
    program: &LowProgram,
    placement: &Placement,
    topology: &Topology,
    stream_words: Option<std::num::NonZeroU32>,
    cache: &mut ExchangeScheduleCache,
) -> Result<Vec<PhysicalExchangePhase>, ExchangeLoweringError> {
    let repeat_inputs = repeat_source_bases(program, placement)?;
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
                incoming_bases,
                optimized,
            } = select_phase(
                phase.id,
                topology,
                pending,
                program.tile_count,
                stream_words,
                cache,
            )?;
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
            let mut physical = schedule.into_phase(phase.id, incoming_bases)?;
            relocation::relocate_repeat_rows(
                &mut physical,
                &pending,
                &placement.shard_addresses,
                &repeat_inputs,
            )?;
            if pending.len() > 1_000 {
                let (tile, words) = physical
                    .programs
                    .iter()
                    .enumerate()
                    .map(|(tile, row)| (tile, row.words().len()))
                    .max_by_key(|entry| entry.1)
                    .unwrap_or((0, 0));
                tracing::info!(
                    phase = phase.id.index(),
                    tile,
                    row_words = words,
                    horizon = physical.event_cycles,
                    "finished large physical exchange phase"
                );
            }
            Ok(physical)
        })
        .collect::<Result<Vec<_>, ExchangeLoweringError>>();
    for phase_cache in phase_caches {
        cache.merge(phase_cache);
    }

    lowered
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
                view.bind(&program.shards)?.traversal(order)?,
            ))
        })
        .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
    if destinations.is_empty() {
        return Err(ExchangeLoweringError::SizeMismatch);
    }
    let source_spans = transfer.source.bind(&program.shards)?.traversal(order)?;
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
    source_elements: Vec<u32>,
    words: u32,
    width: ExchangeItemWidth,
    reserved_source: Option<u16>,
}

impl PendingTransfer {
    fn moving_source(&self) -> bool {
        self.source_addresses
            .iter()
            .any(|&a| a != self.source_address())
    }

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

    /// Payload eligibility only; topology and route encoding are checked separately.
    fn supports_paired_width(&self) -> bool {
        self.words >= 2
            && self.words.is_multiple_of(2)
            && self
                .source_addresses
                .iter()
                .copied()
                .chain(self.destinations.iter().map(|&(_, address)| address))
                .all(|address| address.is_multiple_of(8))
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

/// Use a common moving source as each tile's base, patching exceptions.
/// All relative addresses must remain representable for every Repeat binding.
fn repeat_outgoing_bases(
    pending: &[PendingTransfer],
    patch_words: &[usize],
    addresses: &BTreeMap<BlockValueId, u32>,
    tile_count: u16,
) -> Vec<Option<(BlockValueId, u32)>> {
    let mut tiles = vec![Vec::new(); usize::from(tile_count)];
    for (transfer, &words) in pending.iter().zip(patch_words) {
        if transfer.moving_source() {
            tiles[usize::from(transfer.source)].push((transfer, words));
        }
    }
    tiles
        .into_iter()
        .map(|transfers| {
            let count = transfers
                .iter()
                .map(|(t, _)| t.source_addresses.len())
                .max()
                .unwrap_or(1);
            let mut patterns = BTreeMap::<Vec<u32>, (usize, &PendingTransfer)>::new();
            for &(transfer, words) in &transfers {
                let deltas = (0..count)
                    .map(|i| {
                        repeat_source_address(transfer, i).wrapping_sub(transfer.source_address())
                    })
                    .collect::<Vec<_>>();
                let entry = patterns.entry(deltas).or_insert((0, transfer));
                entry.0 += words;
                if transfer.source_address() < entry.1.source_address() {
                    entry.1 = transfer;
                }
            }
            let paired = transfers
                .iter()
                .any(|(t, _)| t.width == ExchangeItemWidth::Paired64);
            patterns
                .into_values()
                .filter(|&(_, base)| {
                    (!paired || base.source_address().is_multiple_of(8))
                        && transfers.iter().all(|(t, _)| {
                            (0..count).all(|i| {
                                repeat_source_address(t, i) >= repeat_source_address(base, i)
                            })
                        })
                })
                .max_by_key(|&(uses, base)| (uses, Reverse(base.source_address())))
                .and_then(|(_, base)| {
                    let offset = base
                        .source_address()
                        .checked_sub(*addresses.get(&base.source_shard)?)?;
                    Some((base.source_shard, offset))
                })
        })
        .collect()
}

fn repeat_source_address(transfer: &PendingTransfer, iteration: usize) -> u32 {
    transfer
        .source_addresses
        .get(iteration)
        .copied()
        .unwrap_or(transfer.source_address())
}

fn attach_repeat_source_addresses(
    pending: &mut [PendingTransfer],
    repeat_inputs: &BTreeMap<BlockValueId, Vec<u32>>,
) -> Result<(), ExchangeLoweringError> {
    for transfer in pending {
        if let Some(inputs) = repeat_inputs.get(&transfer.source_shard) {
            let addresses = inputs
                .iter()
                .map(|address| {
                    address
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
        if transfer.width != ExchangeItemWidth::Word32 || !transfer.supports_paired_width() {
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
            // Each receiver independently programs its SRAM pointer; only
            // membership in a complete receive pair is required here.
            let pairable = destinations.len() == 2
                && topology
                    .paired_logical(destinations[0].0)
                    .is_ok_and(|paired| paired == destinations[1].0);
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
        if crate::exchange::paired_multicast(
            &topology,
            transfer.source,
            &paired_tiles,
            transfer.words / 2,
        )
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
        incoming_bases,
        optimized,
    })
}

/// Select under the caller's scheduling policy, replaying only compatible work.
/// Policy is an input to compilation; the cache merely records prior choices.
fn select_phase(
    phase: ExchangePhaseId,
    topology: &Topology,
    pending: Vec<PendingTransfer>,
    tile_count: u16,
    stream_words: Option<std::num::NonZeroU32>,
    cache: &mut ExchangeScheduleCache,
) -> Result<ScheduledPending, ExchangeLoweringError> {
    let pending = packet::split_self_receive_conflicts(topology, pending)?;
    let structure = reuse::structure_fingerprint(&pending, tile_count);
    if let Some(recipe) = cache.phases.get(&phase)
        && recipe.structure == structure
        && recipe.stream_words == stream_words
    {
        match recipe.replay(topology, &pending, tile_count) {
            Ok(Some(schedule)) => {
                tracing::info!(
                    phase = phase.index(),
                    "reused exchange optimization after validating relocated rows"
                );
                return Ok(schedule);
            }
            Ok(None) => {}
            Err(error) => {
                tracing::debug!(phase = phase.index(), %error, "exchange choices require reoptimization after placement")
            }
        }
    }
    let selected =
        select_transfer_widths(phase.index(), topology, pending, tile_count, stream_words)?;
    cache.phases.insert(
        phase,
        std::sync::Arc::new(reuse::ScheduleRecipe {
            stream_words,
            structure,
            widths: selected
                .pending
                .iter()
                .map(|transfer| transfer.width)
                .collect(),
            order: selected.optimized.schedule.order.clone(),
            rows: reuse::normalized_rows(&selected.optimized.schedule)?,
        }),
    );
    Ok(selected)
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
        let schedule = streams::schedule(
            topology,
            &problem,
            incoming_bases,
            receive_counts,
            words.get(),
            true,
        )?;
        return Ok(OptimizedSchedule {
            initial_horizon: schedule.horizon,
            endpoint_lower_bound: endpoint_work_lower_bound(pending, tile_count),
            schedule,
            selected_kind: "balanced-compact-streams",
            neighborhood_improvements: 0,
        });
    }
    let schedule = greedy::schedule(
        topology,
        &problem,
        incoming_bases,
        receive_counts,
        ExchangeSchedulingPriority::Automatic,
    )?;
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
    let initial_horizon = schedule.horizon;
    let endpoint_lower_bound = endpoint_work_lower_bound(pending, tile_count);
    let mut selected_kind = initial_kind;
    let mut neighborhood_improvements = 0usize;
    if let Some(order) = matching::order(problem, &schedule.order) {
        let matching = materialize_schedule_order(
            topology,
            problem,
            incoming_bases,
            receive_counts,
            &order,
            false,
        );
        if let Ok(matching) = matching
            && matching.horizon < schedule.horizon
        {
            schedule = matching;
            selected_kind = "matching-waves";
        }
    }
    loop {
        let repaired_order = repair::order(problem, &schedule, false).unwrap_or_else(|| {
            repair::order(problem, &schedule, true).expect("local repair has no work limit")
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
        if repaired.horizon >= schedule.horizon {
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

/// Combine contiguous source and destination spans into one hardware message.
fn coalesce_pending_transfers(transfers: Vec<PendingTransfer>) -> Vec<PendingTransfer> {
    let mut merged = Vec::<PendingTransfer>::with_capacity(transfers.len());
    for transfer in transfers {
        let Some(previous) = merged.last_mut() else {
            merged.push(transfer);
            continue;
        };
        let previous_bytes = previous.words * 4;
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
        let combined_words = previous
            .words
            .checked_add(transfer.words)
            .filter(|&words| contiguous && words <= MAX_TRANSFER_WORDS)
            .filter(|&words| {
                let bytes = u64::from(words) * 4;
                previous.destinations.iter().all(|&(tile, address)| {
                    tile != previous.source
                        || previous.source_addresses.iter().all(|&source| {
                            u64::from(address) + bytes <= u64::from(source)
                                || u64::from(source) + bytes <= u64::from(address)
                        })
                })
            });
        if let Some(words) = combined_words {
            previous.words = words;
            previous.refresh_source_elements();
        } else {
            merged.push(transfer);
        }
    }
    merged
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
    end: u32,
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
    moving_base: Vec<bool>,
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
            moving_base: {
                let mut moving = vec![false; usize::from(tile_count)];
                for transfer in transfers {
                    moving[usize::from(transfer.source)] |= transfer.moving_source();
                }
                moving
            },
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

    fn into_phase(
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

    fn finish_horizon(&mut self) {
        self.horizon = self.builder.event_cycles();
        // The row builder can fill earlier gaps. Profiles follow execution
        // order; relocation uses the transfer identity retained by encoding.
        for activities in &mut self.activities {
            activities.sort_by_key(|activity| activity.start_cycle);
        }
    }
}

fn materialize_valid_schedule_order(
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
        Err(crate::exchange::ExchangeError::Schedule("SENDPICP instruction alignment").into())
    }
}

fn schedule_encoding_is_valid(
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

fn encoded_row_storage(
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

#[cfg(test)]
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

#[cfg(test)]
mod tests;
#[cfg(test)]
use tests::{transfer, validate_exchange_schedule};
