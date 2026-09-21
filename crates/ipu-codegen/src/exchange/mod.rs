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

use ipu_target::Target;
use ipu_target::ipu21::fabric::Topology;
use ipu_target::ipu21::memory::effective_memory_elements;
pub mod diagnostic;
mod program;
pub use program::{
    EXCHANGE_BASE_WRITE_CYCLES, EncodedRow, ExchangeError, HOST_LONG_MAX_BYTES, HOST_PAGE_BYTES,
    HOST_SHORT_MAX_BYTES, HOST_TO_TILE_WINDOW_BYTES, HostHierarchy, HostPacketHeader,
    HostTransferChunk, MAX_PLAN_OFFSET_CYCLES, MAX_TRANSFER_WORDS, OutgoingBaseWrite, PLAN_WORDS,
    PhaseProgramBuilder, PhasePrograms, PhaseTransferTiming, PlanRow, Receiver,
    ScheduledPayloadTiming, ScheduledReceiverTiming, SendAddress, Sender, TILE_TO_HOST_MAX_BYTES,
    TileToHostProgram, TransferPlan, assemble_host_command_read_program,
    assemble_host_to_tile_target_program, assemble_host_xreq_program,
    assemble_host_xreq_program_for_targets, assemble_tile_to_host_target_program,
    encode_exchange_delay_pic, encode_exchange_delay_xpic, encode_send, host_hierarchy,
    host_mux_for_tile, host_to_tile_packet, multicast, offset_plan, paired_multicast,
    patch_receiver_address, patch_sender_address, patch_sender_instruction, plan_event_cycles,
    plan_host_to_tile, plan_tile_to_host, point_to_point, scheduled_receiver_timing,
    set_sender_control, tile_to_host_packet, wrap_combined_host_operation,
    wrap_host_target_operation, wrap_host_xreq_operation, zero_byte_read_packet,
};
mod greedy;
mod hazards;
mod matching;
mod materialize;
mod packet;
mod prepare;
mod problem;
mod repair;
mod select;
mod streams;
pub use diagnostic::diagnose_exchange_tile;
mod relocation;
mod reuse;
pub use reuse::ExchangeScheduleCache;

use crate::{BlockValueId, ExchangePhaseId, LowGraph, Placement};

use rayon::prelude::*;
use std::collections::BTreeMap;

use materialize::{
    MaterializedSchedule, TileAvailability, TilePredecessor, encoded_row_storage,
    endpoint_work_lower_bound, materialize_schedule_order, materialize_valid_schedule_order,
    schedule_encoding_is_valid,
};
use prepare::{
    PendingTransfer, prepare_phase, repeat_outgoing_bases, repeat_source_address,
    repeat_source_bases,
};
use problem::SchedulingProblem;
use select::{
    OptimizedSchedule, ScheduledPending, paired_transfer_alternatives, receive_configuration,
    select_phase,
};

#[cfg(test)]
use greedy::ExchangeSchedulingPriority;
#[cfg(test)]
use materialize::MaterializedTiming;
#[cfg(test)]
use prepare::coalesce_pending_transfers;
#[cfg(test)]
use problem::memory_dependencies;
#[cfg(test)]
use select::{
    improve_pending_schedule, optimize_owned_pending, optimize_pending_schedule,
    select_transfer_widths,
};
#[cfg(test)]
use std::collections::BTreeSet;

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
    target: Target,
    program: &LowGraph,
    placement: &Placement,
    topology: &Topology,
) -> Result<Vec<PhysicalExchangePhase>, ExchangeLoweringError> {
    lower_exchanges_cached(
        target,
        program,
        placement,
        topology,
        None,
        &mut ExchangeScheduleCache::default(),
    )
}

pub(crate) fn lower_exchanges_cached(
    target: Target,
    program: &LowGraph,
    placement: &Placement,
    topology: &Topology,
    stream_words: Option<std::num::NonZeroU32>,
    cache: &mut ExchangeScheduleCache,
) -> Result<Vec<PhysicalExchangePhase>, ExchangeLoweringError> {
    let Target::Ipu21 = target;

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
            Ok(physical)
        })
        .collect::<Result<Vec<_>, ExchangeLoweringError>>();
    for phase_cache in phase_caches {
        cache.merge(phase_cache);
    }

    lowered
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
