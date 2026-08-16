use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::debug;

pub mod diagnostic;

pub const PLAN_WORDS: usize = 9;
pub const MAX_TRANSFER_WORDS: u32 = 4148;
/// Largest scheduled delay encodable by one exchange delay instruction.
pub const MAX_PLAN_OFFSET_CYCLES: u32 = 0x8_0000;
pub const EXCHANGE_WINDOW_BASE: u32 = 0x50000;
pub const EXCHANGE_WINDOW_BYTES: u32 = 0x8000;
pub const HOST_SHORT_MAX_BYTES: u32 = 60;
pub const HOST_LONG_MAX_BYTES: u32 = 1024;
pub const TILE_TO_HOST_MAX_BYTES: u32 = 256;
pub const HOST_PAGE_BYTES: u32 = 4096;
pub const HOST_TO_TILE_WINDOW_BYTES: u32 = 0x4000;
pub const TILE_MUX_HOST: u32 = 0x600;
pub const TILE_MUX_EXCHANGE: u32 = 0x640;
const XREQ_BITMAP0_BITS: u32 = 24;

const OPCODE_MASK: u32 = 0xfc00_0000;
const LONG_OPCODE_MASK: u32 = 0xf800_0000;
const DELAY_OPCODE_MASK: u32 = 0xfff8_0000;
const DELAY_OPCODE: u32 = 0x40a0_0000;
const DELAY_PIC_OPCODE: u32 = 0x6000_0000;
const DELAY_XPIC_OPCODE: u32 = 0x6400_0000;
const PIC_RECEIVE_ADDRESS_MASK: u32 = 0x3ffff;
const SEND_OPCODE: u32 = 0x7800_0000;
const SEND_ADDRESS_MASK: u32 = 0x001f_fff8;
const SEND_OFF_OPCODE: u32 = 0x7000_0000;
// See docs/EXCHANGE_INSTRUCTION_REFERENCE.md. SENDPICP is an aligned two-word
// supervisor instruction whose following word is inline PIC/XPIC payload.
const SEND_PIC_OPCODE: u32 = 0x7010_0000;
const SEND_PICP_OPCODE: u32 = 0xf000_0000;
const SEND_CONTROL_OPCODE_MASK: u32 = 0xf810_0000;
const SEND_PICP_OPCODE_MASK: u32 = 0xf000_0000;
const SEND_COUNT_MASK: u32 = 0x07e0_0000;
const SYNC_OPCODE: u32 = 0x4180_0000;
const SANS_OPCODE: u32 = 0x40c0_0000;
const BR_M_OPCODE: u32 = 0x4300_0000;
const CALL_M_IMMEDIATE_OPCODE: u32 = 0x1800_0000;
const SETZI_M_OPCODE: u32 = 0x1900_0000;
const PUT_SPECIAL_M_OPCODE: u32 = 0x4300_8000;
const LD32_M_IMMEDIATE_OPCODE: u32 = 0x0100_0000;
const ST32_M_IMMEDIATE_OPCODE: u32 = 0x4f00_0000;
const ADD_M_IMMEDIATE_OPCODE: u32 = 0x2200_0000;
const AND_M_IMMEDIATE_OPCODE: u32 = 0x4200_0000;
const SHL_M_IMMEDIATE_OPCODE: u32 = 0x4200_a000;
const BRZ_M_IMMEDIATE_OPCODE: u32 = 0x1300_0000;
const INCOMING_MUX_REGISTER: u8 = 0xa0;
const INCOMING_DCOUNT_REGISTER: u8 = 0xa6;
// The host hierarchy reserves eighteen exchange events for each tile-to-host
// payload. Short payloads must be padded before the next packet header (or the
// closing zero-byte read); longer payloads provide the interval themselves.
const TILE_TO_HOST_MIN_PAYLOAD_EVENTS: u32 = 18;
const HOST_TO_TILE_STREAM_END_BITS: u32 = 0x0c00_0000;
// Time reserved by the SDK supervisor schedule between receiving a host
// command and injecting that command into the device-side dispatch path.
const HOST_COMMAND_ROUTE_CYCLES: u32 = 73;

pub const SANS_INACTIVE_INSTRUCTION: u32 = sans(0);
pub const SYNC_RECEIVE_INSTRUCTION: u32 = sync(0);
pub const SYNC_ANS_INSTRUCTION: u32 = sync(1);
pub const SYNC_SUPERVISOR_INSTRUCTION: u32 = sync(3);
pub const SYNC_ALL_INSTRUCTION: u32 = sync(7);
pub const SYNC_HOST_INSTRUCTION: u32 = sync(15);
pub const RETURN_M10_INSTRUCTION: u32 = br_m(10);

pub const fn sans(selector: u8) -> u32 {
    SANS_OPCODE | selector as u32
}

pub const fn sync(selector: u8) -> u32 {
    SYNC_OPCODE | selector as u32
}

pub const fn br_m(register: u8) -> u32 {
    BR_M_OPCODE | ((register as u32) << 20)
}

pub fn encode_br_m(register: u8) -> Result<u32, ExchangeError> {
    if register >= 16 {
        return Err(ExchangeError::Schedule("branch register"));
    }
    Ok(br_m(register))
}

pub fn encode_call_m_immediate(
    return_register: u8,
    target_address: u32,
) -> Result<u32, ExchangeError> {
    if return_register >= 16 || target_address & 3 != 0 || target_address >= 1 << 21 {
        return Err(ExchangeError::Schedule("call operand"));
    }
    Ok(CALL_M_IMMEDIATE_OPCODE | (u32::from(return_register) << 20) | (target_address >> 2))
}

pub fn encode_setzi_m(register: u8, immediate: u32) -> Result<u32, ExchangeError> {
    if register >= 16 || immediate >= 1 << 20 {
        return Err(ExchangeError::Schedule("setzi operand"));
    }
    Ok(setzi_m(register, immediate))
}

pub fn encode_put_special_m(special: u8, register: u8) -> Result<u32, ExchangeError> {
    if register >= 16 {
        return Err(ExchangeError::Schedule("put source register"));
    }
    Ok(PUT_SPECIAL_M_OPCODE | (u32::from(register) << 20) | u32::from(special))
}

pub fn encode_ld32_m_immediate(
    destination: u8,
    base: u8,
    delta: u8,
    word_offset: u16,
) -> Result<u32, ExchangeError> {
    if destination >= 16 || base >= 16 || delta >= 16 || word_offset >= 1 << 12 {
        return Err(ExchangeError::Schedule("ld32 operand"));
    }
    Ok(LD32_M_IMMEDIATE_OPCODE
        | (u32::from(base) << 20)
        | (u32::from(destination) << 16)
        | (u32::from(delta) << 12)
        | u32::from(word_offset))
}

pub fn encode_st32_m_immediate(
    source: u8,
    base: u8,
    delta: u8,
    word_offset: u16,
) -> Result<u32, ExchangeError> {
    if source >= 16 || base >= 16 || delta >= 16 || word_offset >= 1 << 12 {
        return Err(ExchangeError::Schedule("st32 operand"));
    }
    Ok(ST32_M_IMMEDIATE_OPCODE
        | (u32::from(base) << 20)
        | (u32::from(source) << 16)
        | (u32::from(delta) << 12)
        | u32::from(word_offset))
}

pub fn encode_add_m_immediate(
    destination: u8,
    source: u8,
    immediate: i32,
) -> Result<u32, ExchangeError> {
    let immediate =
        i16::try_from(immediate).map_err(|_| ExchangeError::Schedule("add immediate operand"))?;
    if destination >= 16 || source >= 16 {
        return Err(ExchangeError::Schedule("add register operand"));
    }
    Ok(ADD_M_IMMEDIATE_OPCODE
        | (u32::from(source) << 20)
        | (u32::from(destination) << 16)
        | u32::from(immediate as u16))
}

pub fn encode_and_m_immediate(
    destination: u8,
    source: u8,
    immediate: u16,
) -> Result<u32, ExchangeError> {
    if destination >= 16 || source >= 16 || immediate >= 1 << 12 {
        return Err(ExchangeError::Schedule("and operand"));
    }
    Ok(AND_M_IMMEDIATE_OPCODE
        | (u32::from(source) << 20)
        | (u32::from(destination) << 16)
        | u32::from(immediate))
}

pub fn encode_shl_m_immediate(
    destination: u8,
    source: u8,
    immediate: u16,
) -> Result<u32, ExchangeError> {
    if destination >= 16 || source >= 16 || immediate >= 1 << 12 {
        return Err(ExchangeError::Schedule("shift-left operand"));
    }
    Ok(SHL_M_IMMEDIATE_OPCODE
        | (u32::from(source) << 20)
        | (u32::from(destination) << 16)
        | u32::from(immediate))
}

pub fn encode_brz_m_immediate(register: u8, target_address: u32) -> Result<u32, ExchangeError> {
    if register >= 16 || target_address & 3 != 0 || target_address >= 1 << 21 {
        return Err(ExchangeError::Schedule("brz operand"));
    }
    Ok(BRZ_M_IMMEDIATE_OPCODE | (u32::from(register) << 20) | (target_address >> 2))
}

/// Encodes a processor delay of `cycles` cycles.
pub fn encode_delay_m(cycles: u32) -> Result<u32, ExchangeError> {
    if !(1..=MAX_PLAN_OFFSET_CYCLES).contains(&cycles) {
        return Err(ExchangeError::Schedule("processor delay range"));
    }
    Ok(delay(cycles - 1))
}

const fn setzi_m(register: u8, immediate: u32) -> u32 {
    SETZI_M_OPCODE | ((register as u32) << 20) | immediate
}

const fn put_special_from_m8(register: u8) -> u32 {
    PUT_SPECIAL_M_OPCODE | (8 << 20) | register as u32
}

pub type PlanRow = [u32; PLAN_WORDS];

/// Returns the plan event horizon measured from the entry synchronization.
///
/// Delay immediates advance to the event `N + 1` cycles later. Send
/// instructions occupy one event per transferred word.
pub fn plan_event_cycles(row: &[u32]) -> Result<u32, ExchangeError> {
    let mut cycles = 0u32;
    let mut cursor = 0;
    while cursor < row.len() {
        let instruction = row[cursor];
        let advance = instruction_advance(instruction);
        cycles = cycles
            .checked_add(advance)
            .ok_or(ExchangeError::Schedule("plan event horizon overflow"))?;
        cursor += if is_send_control_pair(instruction) {
            2
        } else {
            1
        };
    }
    Ok(cycles)
}

#[derive(Clone, Debug, Default)]
struct TileProgramSchedule {
    senders: Vec<ScheduledSenderRow>,
    receive_events: Vec<ReceiveEvent>,
    event_cycles: u32,
    receive_stream: Option<ReceiveStream>,
}

#[derive(Clone, Debug)]
struct ReceiveStream {
    mode: ReceiveMode,
    source_end_cycles: u32,
    pointer_end_cycles: u32,
    next_address: Option<u32>,
}

#[derive(Clone, Debug)]
struct ScheduledSenderRow {
    row: PlanRow,
    start_cycles: u32,
    end_cycles: u32,
}

#[derive(Clone, Debug)]
pub struct PhaseProgramBuilder {
    tile_states: Vec<TileProgramSchedule>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseTransferTiming {
    pub payload_start: u32,
    pub payload_end: u32,
    pub sender_horizon: u32,
    pub receiver_payload_starts: Vec<u32>,
    pub receiver_payload_ends: Vec<u32>,
    pub receiver_horizons: Vec<u32>,
    pub horizon: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhasePrograms {
    pub programs: Vec<Option<Vec<u32>>>,
    pub tile_event_cycles: Vec<u32>,
    pub event_cycles: u32,
}

impl PhaseProgramBuilder {
    pub fn new(tile_count: u16) -> Self {
        Self {
            tile_states: vec![TileProgramSchedule::default(); usize::from(tile_count)],
        }
    }

    pub fn tile_count(&self) -> u16 {
        u16::try_from(self.tile_states.len()).expect("phase tile count was supplied as u16")
    }

    pub fn tile_event_cycles(&self, tile: u16) -> Result<u32, ExchangeError> {
        self.tile_states
            .get(usize::from(tile))
            .map(|schedule| schedule.event_cycles)
            .ok_or(ExchangeError::Tile(tile))
    }

    pub fn event_cycles(&self) -> u32 {
        self.tile_states
            .iter()
            .map(|schedule| schedule.event_cycles)
            .max()
            .unwrap_or(0)
    }

    pub fn active_tile_count(&self) -> usize {
        self.tile_states
            .iter()
            .filter(|schedule| schedule.event_cycles != 0)
            .count()
    }

    /// Finds one absolute offset at which all endpoints can perform the
    /// transfer. Compatibility is evaluated across the complete phase state;
    /// tile programs are not encoded until the phase is finished.
    pub fn earliest_transfer_offset(
        &self,
        source: u16,
        reserved_tiles: &[u16],
        receivers: &[u16],
        plan: &MulticastPlan,
        words: u32,
        requested: u32,
    ) -> Result<u32, ExchangeError> {
        self.earliest_transfer_offset_impl(
            source,
            reserved_tiles,
            receivers,
            plan,
            words,
            requested,
            true,
        )
    }

    /// Finds an endpoint-compatible offset while deferring whole-row encoding
    /// validation until the complete phase is available. Callers must encode
    /// the completed phase and retry with [`Self::earliest_transfer_offset`]
    /// if instruction alignment is not representable.
    pub fn earliest_transfer_offset_deferred(
        &self,
        source: u16,
        reserved_tiles: &[u16],
        receivers: &[u16],
        plan: &MulticastPlan,
        words: u32,
        requested: u32,
    ) -> Result<u32, ExchangeError> {
        self.earliest_transfer_offset_impl(
            source,
            reserved_tiles,
            receivers,
            plan,
            words,
            requested,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn earliest_transfer_offset_impl(
        &self,
        source: u16,
        reserved_tiles: &[u16],
        receivers: &[u16],
        plan: &MulticastPlan,
        words: u32,
        requested: u32,
        validate_encoding: bool,
    ) -> Result<u32, ExchangeError> {
        if receivers.len() != plan.receivers.len() {
            return Err(ExchangeError::Schedule("receiver row count"));
        }
        let source_schedule = self
            .tile_states
            .get(usize::from(source))
            .ok_or(ExchangeError::Tile(source))?;
        let mut offset = requested;
        loop {
            let previous = offset;
            offset = source_schedule.earliest_sender_offset(&plan.sender, offset)?;
            for &tile in reserved_tiles {
                let schedule = self
                    .tile_states
                    .get(usize::from(tile))
                    .ok_or(ExchangeError::Tile(tile))?;
                offset = offset.max(schedule.event_cycles);
            }
            for (&receiver, row) in receivers.iter().zip(&plan.receivers) {
                let schedule = self
                    .tile_states
                    .get(usize::from(receiver))
                    .ok_or(ExchangeError::Tile(receiver))?;
                offset = schedule.earliest_receiver_offset(row, words, offset)?;
            }
            if offset == previous {
                if !validate_encoding {
                    return Ok(offset);
                }
                match self.transfer_is_encodable_at(source, receivers, plan, offset, words) {
                    Ok(()) => return Ok(offset),
                    Err(ExchangeError::Schedule("SENDPICP instruction alignment")) => {
                        offset = offset
                            .checked_add(1)
                            .ok_or(ExchangeError::Schedule("receive offset overflow"))?;
                    }
                    Err(error) => return Err(error),
                }
            }
        }
    }

    fn transfer_is_encodable_at(
        &self,
        source: u16,
        receivers: &[u16],
        plan: &MulticastPlan,
        schedule_offset: u32,
        words: u32,
    ) -> Result<(), ExchangeError> {
        let mut source_schedule = self
            .tile_states
            .get(usize::from(source))
            .ok_or(ExchangeError::Tile(source))?
            .clone();
        source_schedule.append_sender_at(&plan.sender, schedule_offset)?;
        source_schedule.finish()?;
        for (&receiver, row) in receivers.iter().zip(&plan.receivers) {
            let mut receiver_schedule = self
                .tile_states
                .get(usize::from(receiver))
                .ok_or(ExchangeError::Tile(receiver))?
                .clone();
            receiver_schedule.append_receiver_at(row, schedule_offset, words)?;
            receiver_schedule.finish()?;
        }
        Ok(())
    }

    /// Adds one transfer to the phase schedule. The transfer's sender and all
    /// receivers are recorded declaratively and compiled together by
    /// [`Self::finish`].
    pub fn append_transfer_at(
        &mut self,
        source: u16,
        reserved_tiles: &[u16],
        receivers: &[u16],
        plan: &MulticastPlan,
        schedule_offset: u32,
        words: u32,
    ) -> Result<PhaseTransferTiming, ExchangeError> {
        let transfer_timing =
            self.transfer_timing_at(source, receivers, plan, schedule_offset, words)?;
        if receivers.len() != plan.receivers.len() {
            return Err(ExchangeError::Schedule("receiver row count"));
        }
        let mut updates = Vec::with_capacity(receivers.len() + reserved_tiles.len() + 1);
        let mut source_schedule = self
            .tile_states
            .get(usize::from(source))
            .ok_or(ExchangeError::Tile(source))?
            .clone();
        source_schedule.append_sender_at(&plan.sender, schedule_offset)?;
        updates.push((source, source_schedule));

        for &tile in reserved_tiles {
            if tile == source
                || receivers.contains(&tile)
                || updates.iter().any(|(updated, _)| *updated == tile)
            {
                return Err(ExchangeError::DuplicateTile);
            }
            let mut schedule = self
                .tile_states
                .get(usize::from(tile))
                .ok_or(ExchangeError::Tile(tile))?
                .clone();
            schedule.event_cycles = schedule.event_cycles.max(transfer_timing.sender_horizon);
            updates.push((tile, schedule));
        }

        for (&receiver, row) in receivers.iter().zip(&plan.receivers) {
            if receiver == source || updates.iter().any(|(tile, _)| *tile == receiver) {
                return Err(ExchangeError::DuplicateTile);
            }
            let mut receiver_schedule = self
                .tile_states
                .get(usize::from(receiver))
                .ok_or(ExchangeError::Tile(receiver))?
                .clone();
            receiver_schedule.append_receiver_at(row, schedule_offset, words)?;
            updates.push((receiver, receiver_schedule));
        }
        for (tile, schedule) in updates {
            self.tile_states[usize::from(tile)] = schedule;
        }
        Ok(transfer_timing)
    }

    pub fn transfer_timing_at(
        &self,
        source: u16,
        receivers: &[u16],
        plan: &MulticastPlan,
        schedule_offset: u32,
        words: u32,
    ) -> Result<PhaseTransferTiming, ExchangeError> {
        if receivers.len() != plan.receivers.len() {
            return Err(ExchangeError::Schedule("receiver row count"));
        }
        self.tile_states
            .get(usize::from(source))
            .ok_or(ExchangeError::Tile(source))?;
        let sender = scheduled_sender_timing(&plan.sender, schedule_offset)?;
        let receiver_timings = receivers
            .iter()
            .zip(&plan.receivers)
            .map(|(&receiver, row)| {
                if receiver == source {
                    return Err(ExchangeError::DuplicateTile);
                }
                let schedule = self
                    .tile_states
                    .get(usize::from(receiver))
                    .ok_or(ExchangeError::Tile(receiver))?;
                let base = receive_row_timing(row, 0)?;
                scheduled_receive_window(
                    &base,
                    schedule_offset,
                    words,
                    schedule.receive_stream.as_ref(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let horizon = receiver_timings
            .iter()
            .map(|timing| timing.horizon)
            .chain(std::iter::once(sender.horizon))
            .max()
            .unwrap_or(schedule_offset);
        Ok(PhaseTransferTiming {
            payload_start: sender.payload_start,
            payload_end: sender.payload_end,
            sender_horizon: sender.horizon,
            receiver_payload_starts: receiver_timings
                .iter()
                .map(|timing| timing.payload_start)
                .collect(),
            receiver_payload_ends: receiver_timings
                .iter()
                .map(|timing| timing.payload_end)
                .collect(),
            receiver_horizons: receiver_timings
                .iter()
                .map(|timing| timing.horizon)
                .collect(),
            horizon,
        })
    }

    pub fn finish(self) -> Result<PhasePrograms, ExchangeError> {
        // `append_transfer_at` has already merged the authoritative
        // phase-wide schedule into these per-tile sender and receive-event
        // timelines. Encoding them directly avoids replaying every transfer
        // and performing the same conflict checks a second time.
        let tile_states = self.tile_states;
        let event_cycles = tile_states
            .iter()
            .map(|schedule| schedule.event_cycles)
            .max()
            .unwrap_or(0);
        let tile_event_cycles = tile_states
            .iter()
            .map(|schedule| schedule.event_cycles)
            .collect::<Vec<_>>();
        let programs = tile_states
            .iter()
            .enumerate()
            .map(|(tile, schedule)| {
                if schedule.event_cycles == 0 {
                    return Ok(None);
                }
                let program = schedule.finish()?;
                diagnostic::validate_tile_program(tile, schedule, &program)?;
                Ok(Some(program))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PhasePrograms {
            programs,
            tile_event_cycles,
            event_cycles,
        })
    }
}

impl TileProgramSchedule {
    #[cfg(test)]
    fn event_cycles(&self) -> u32 {
        self.event_cycles
    }

    /// Advances a requested transfer offset until its outgoing message does
    /// not overlap another outgoing message on this tile. Receive controls may
    /// be represented by the ISA's composite send/control encodings; this is a
    /// single supervisor instruction, not dual issue from independent lanes.
    fn earliest_sender_offset(&self, row: &PlanRow, requested: u32) -> Result<u32, ExchangeError> {
        let base = sender_row_timing(row, 0)?;
        let mut offset = requested;
        loop {
            let start = base
                .start_cycles
                .checked_add(offset)
                .ok_or(ExchangeError::Schedule("send offset overflow"))?;
            let end = base
                .end_cycles
                .checked_add(offset)
                .ok_or(ExchangeError::Schedule("send offset overflow"))?;
            let conflicting_sender = self
                .senders
                .iter()
                .find(|sender| start < sender.end_cycles && sender.start_cycles < end);
            if let Some(sender) = conflicting_sender {
                offset = sender
                    .end_cycles
                    .checked_sub(base.start_cycles)
                    .ok_or(ExchangeError::Schedule("send offset order"))?;
                continue;
            }
            // A receive control cannot be encoded before the first outgoing
            // word. Leave at least one continuation word so SENDPIC can carry
            // a control at the following event.
            if self
                .receive_events
                .iter()
                .any(|event| event.cycles == start || event.cycles == start.saturating_add(1))
            {
                offset = offset
                    .checked_add(1)
                    .ok_or(ExchangeError::Schedule("send offset overflow"))?;
                continue;
            }
            return Ok(offset);
        }
    }

    /// Advances a requested transfer offset while preserving independent
    /// source-selection and local receive-address streams.
    fn earliest_receiver_offset(
        &self,
        row: &PlanRow,
        received_words: u32,
        requested: u32,
    ) -> Result<u32, ExchangeError> {
        let base = receive_row_timing(row, 0)?;
        let mut offset = requested;
        if let Some(stream) = &self.receive_stream {
            let source_cycles = base
                .source_cycles
                .or(base.format_start_cycles)
                .ok_or(ExchangeError::Schedule("receive source timing"))?;
            offset = offset.max(stream.source_end_cycles.saturating_sub(source_cycles));
            if base.pointer_address != stream.next_address {
                let pointer_cycles = base
                    .pointer_cycles
                    .ok_or(ExchangeError::Schedule("receive pointer event"))?;
                offset = offset.max(stream.pointer_end_cycles.saturating_sub(pointer_cycles));
            }
        }
        loop {
            let timing = scheduled_receive_window(
                &base,
                offset,
                received_words,
                self.receive_stream.as_ref(),
            )?;
            let replaces_neutral = self
                .receive_stream
                .as_ref()
                .is_some_and(|stream| timing.source_start == stream.source_end_cycles);
            let collision = timing.events.iter().any(|new| {
                self.receive_events.iter().any(|existing| {
                    new.cycles == existing.cycles
                        && !(replaces_neutral
                            && new.kind == ReceiveEventKind::OrdinarySource
                            && existing.kind == ReceiveEventKind::OrdinaryNeutral)
                        && !receive_events_can_share_instruction(*new, *existing)
                })
            });
            let sender_boundary = timing.events.iter().any(|event| {
                self.senders.iter().any(|sender| {
                    event.cycles == sender.start_cycles
                        || event.cycles == sender.start_cycles.saturating_add(1)
                })
            });
            if !collision && !sender_boundary {
                return Ok(offset);
            }
            offset = offset
                .checked_add(1)
                .ok_or(ExchangeError::Schedule("receive offset overflow"))?;
        }
    }

    /// Appends a primitive row at an arbitrary absolute phase offset without
    /// requiring that offset to fit in the primitive row's spare words.
    fn append_sender_at(
        &mut self,
        row: &PlanRow,
        schedule_offset: u32,
    ) -> Result<(), ExchangeError> {
        let timing = sender_row_timing(row, schedule_offset)?;
        if self.senders.iter().any(|sender| {
            timing.start_cycles < sender.end_cycles && sender.start_cycles < timing.end_cycles
        }) {
            return Err(ExchangeError::Schedule("overlapping outgoing messages"));
        }
        if self.receive_events.iter().any(|event| {
            event.cycles == timing.start_cycles
                || event.cycles == timing.start_cycles.saturating_add(1)
        }) {
            return Err(ExchangeError::Schedule("unencodable initial send control"));
        }
        self.event_cycles = self.event_cycles.max(timing.horizon_cycles);
        self.senders.push(ScheduledSenderRow {
            row: *row,
            start_cycles: timing.start_cycles,
            end_cycles: timing.end_cycles,
        });
        Ok(())
    }

    /// Appends a receive row, merging its timed control writes with the
    /// current receive stream and replacing an immediately preceding neutral
    /// mux selection with a direct source cutover.
    fn append_receiver_at(
        &mut self,
        row: &PlanRow,
        schedule_offset: u32,
        received_words: u32,
    ) -> Result<ScheduledReceiverWindow, ExchangeError> {
        let base = receive_row_timing(row, 0)?;
        let timing = scheduled_receive_window(
            &base,
            schedule_offset,
            received_words,
            self.receive_stream.as_ref(),
        )?;
        let next_address = timing
            .pointer_address
            .map(|address| {
                received_words
                    .checked_mul(match base.mode {
                        ReceiveMode::Ordinary => 4,
                        ReceiveMode::Paired64 => 8,
                    })
                    .and_then(|bytes| address.checked_add(bytes))
                    .ok_or(ExchangeError::Schedule("receive address overflow"))
            })
            .transpose()?;
        let previous = self.receive_stream.take();
        if let Some(stream) = &previous {
            if timing.source_start < stream.source_end_cycles
                || timing.payload_start < stream.pointer_end_cycles
            {
                return Err(ExchangeError::Schedule("overlapping receive streams"));
            }
            if timing.source_start == stream.source_end_cycles {
                let neutral = self
                    .receive_events
                    .iter()
                    .rposition(|event| {
                        event.kind == ReceiveEventKind::OrdinaryNeutral
                            && event.cycles == stream.source_end_cycles
                    })
                    .filter(|_| {
                        stream.mode == ReceiveMode::Ordinary && base.mode == ReceiveMode::Ordinary
                    });
                if let Some(neutral) = neutral {
                    self.receive_events.remove(neutral);
                }
            }
        }
        // `earliest_receiver_offset` checked the new events against the full
        // existing stream, while `scheduled_receive_window` validated the new
        // group internally. Source and pointer controls are independent and
        // may be inserted on opposite sides of an older teardown event, so
        // keep insertion order here and sort once when encoding the row.
        self.receive_events.extend(timing.events.iter().copied());
        self.event_cycles = self.event_cycles.max(timing.horizon);
        self.receive_stream = Some(ReceiveStream {
            mode: base.mode,
            source_end_cycles: timing.source_end,
            pointer_end_cycles: timing.payload_end,
            next_address,
        });
        Ok(timing)
    }

    pub fn finish(&self) -> Result<Vec<u32>, ExchangeError> {
        build_scheduled_program(&self.senders, &self.receive_events, self.event_cycles)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReceiveEventKind {
    OrdinarySource,
    OrdinaryNeutral,
    PairedSource,
    PairedNeutral,
    Pointer,
    Format,
}

impl ReceiveEventKind {
    fn is_xpic(self) -> bool {
        matches!(
            self,
            Self::OrdinarySource | Self::OrdinaryNeutral | Self::PairedSource | Self::PairedNeutral
        )
    }

    fn is_pic(self) -> bool {
        matches!(self, Self::Pointer | Self::Format)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReceiveMode {
    Ordinary,
    Paired64,
}

#[derive(Clone, Copy, Debug)]
struct ReceiveEvent {
    cycles: u32,
    instruction: u32,
    kind: ReceiveEventKind,
}

fn receive_events_can_share_instruction(left: ReceiveEvent, right: ReceiveEvent) -> bool {
    left.cycles == right.cycles
        && ((left.kind.is_pic() && right.kind.is_xpic())
            || (left.kind.is_xpic() && right.kind.is_pic()))
}

fn validate_receive_events(events: &[ReceiveEvent]) -> Result<(), ExchangeError> {
    let mut cursor = 0;
    while cursor < events.len() {
        let end = cursor
            + events[cursor..].partition_point(|event| event.cycles == events[cursor].cycles);
        let group = &events[cursor..end];
        if group.len() > 2
            || (group.len() == 2 && !receive_events_can_share_instruction(group[0], group[1]))
        {
            return Err(ExchangeError::Schedule(
                "incompatible simultaneous receive controls",
            ));
        }
        cursor = end;
    }
    Ok(())
}

struct ReceiveRowTiming {
    mode: ReceiveMode,
    events: Vec<ReceiveEvent>,
    neutral_cycles: Option<u32>,
    source_cycles: Option<u32>,
    format_start_cycles: Option<u32>,
    format_end_cycles: Option<u32>,
    pointer_cycles: Option<u32>,
    horizon_cycles: u32,
    pointer_address: Option<u32>,
}

#[derive(Clone, Debug)]
struct ScheduledReceiverWindow {
    events: Vec<ReceiveEvent>,
    source_start: u32,
    source_end: u32,
    payload_start: u32,
    payload_end: u32,
    horizon: u32,
    pointer_address: Option<u32>,
}

fn scheduled_receive_window(
    base: &ReceiveRowTiming,
    schedule_offset: u32,
    received_words: u32,
    previous: Option<&ReceiveStream>,
) -> Result<ScheduledReceiverWindow, ExchangeError> {
    let timing = receive_row_timing_from_base(base, schedule_offset)?;
    // The incoming source is occupied for exactly one event per word. The
    // pointer stream has a physical-row phase of its own and must not extend
    // source ownership: SDK full-duplex rows switch XPIC at this boundary
    // while their PIC update and outgoing SEND stream continue independently.
    let (source_start, source_end) = match timing.mode {
        ReceiveMode::Ordinary => {
            let start = timing
                .source_cycles
                .ok_or(ExchangeError::Schedule("receive source timing"))?;
            let end = start
                .checked_add(received_words)
                .ok_or(ExchangeError::Schedule("receive source timing overflow"))?;
            (start, end)
        }
        ReceiveMode::Paired64 => {
            let format_start = timing
                .format_start_cycles
                .ok_or(ExchangeError::Schedule("paired receive format start"))?;
            let format_end = timing
                .format_end_cycles
                .ok_or(ExchangeError::Schedule("paired receive format end"))?;
            let source_start = timing
                .source_cycles
                .map_or(format_start, |source| source.min(format_start));
            let source_end = timing.neutral_cycles.unwrap_or(format_end).max(format_end);
            (source_start, source_end)
        }
    };
    let carries_pointer = timing.mode == ReceiveMode::Ordinary
        && previous.is_some_and(|stream| {
            stream.mode == ReceiveMode::Ordinary && timing.pointer_address == stream.next_address
        });
    // PIC may retain a contiguous destination address, but the payload still
    // follows this transfer's XPIC source cutover through the route pipeline.
    // The primitive row's PIC event gives that arrival offset even when the
    // redundant address write itself can be omitted.
    let payload_start = timing
        .pointer_cycles
        .ok_or(ExchangeError::Schedule("receive pointer event"))?;
    let payload_end = payload_start
        .checked_add(received_words)
        .ok_or(ExchangeError::Schedule("receive payload timing overflow"))?;

    // Primitive rows serialize the independent XPIC and PIC control streams.
    // Their final delay preserves a small guard after both the remote source
    // window and the local SRAM write window have completed. Retain that
    // guard after the two streams are interleaved with other transfers.
    let base_source_end = match base.mode {
        ReceiveMode::Ordinary => base
            .source_cycles
            .ok_or(ExchangeError::Schedule("receive source timing"))?
            .checked_add(received_words)
            .ok_or(ExchangeError::Schedule("receive source timing overflow"))?,
        ReceiveMode::Paired64 => base
            .neutral_cycles
            .unwrap_or(
                base.format_end_cycles
                    .ok_or(ExchangeError::Schedule("paired receive format end"))?,
            )
            .max(
                base.format_end_cycles
                    .ok_or(ExchangeError::Schedule("paired receive format end"))?,
            ),
    };
    let base_payload_end = base
        .pointer_cycles
        .ok_or(ExchangeError::Schedule("receive pointer event"))?
        .checked_add(received_words)
        .ok_or(ExchangeError::Schedule("receive payload timing overflow"))?;
    let guard = base
        .horizon_cycles
        .saturating_sub(base_source_end.max(base_payload_end));
    let horizon = source_end
        .max(payload_end)
        .checked_add(guard)
        .ok_or(ExchangeError::Schedule("receive horizon overflow"))?;

    let mut events = timing
        .events
        .into_iter()
        .filter_map(|mut event| match event.kind {
            ReceiveEventKind::OrdinarySource => Some(event),
            ReceiveEventKind::Pointer if !carries_pointer => Some(event),
            ReceiveEventKind::Pointer => None,
            ReceiveEventKind::OrdinaryNeutral => {
                event.cycles = source_end;
                Some(event)
            }
            ReceiveEventKind::PairedSource
            | ReceiveEventKind::PairedNeutral
            | ReceiveEventKind::Format => Some(event),
        })
        .collect::<Vec<_>>();
    events.sort_by_key(|event| event.cycles);
    validate_receive_events(&events)?;
    Ok(ScheduledReceiverWindow {
        events,
        source_start,
        source_end,
        payload_start,
        payload_end,
        horizon,
        pointer_address: timing.pointer_address,
    })
}

fn receive_row_timing(
    row: &PlanRow,
    schedule_offset: u32,
) -> Result<ReceiveRowTiming, ExchangeError> {
    if row[0] != SYNC_SUPERVISOR_INSTRUCTION {
        return Err(ExchangeError::Schedule("receive row entry"));
    }
    let end = row
        .iter()
        .position(|instruction| *instruction == RETURN_M10_INSTRUCTION)
        .ok_or(ExchangeError::Schedule("receive row return"))?;
    let mut cycles = schedule_offset;
    let mut events = Vec::new();
    let mut source_cycles = None;
    let mut neutral_cycles = None;
    let mut format_start_cycles = None;
    let mut format_end_cycles = None;
    let mut pointer_cycles = None;
    let mut pointer_address = None;
    let mut cursor = 1;
    while cursor < end {
        let instruction = row[cursor];
        if is_send_control_pair(instruction) {
            let payload = *row
                .get(cursor + 1)
                .ok_or(ExchangeError::Schedule("truncated SENDPICP payload"))?;
            let control_cycles = cycles
                .checked_add(1)
                .ok_or(ExchangeError::Schedule("receive event horizon overflow"))?;
            let xpic_value = payload >> 18;
            let paired = xpic_value & (1 << 13) != 0;
            let xpic_kind = if xpic_value & 0x1fff == TILE_MUX_EXCHANGE {
                if paired {
                    ReceiveEventKind::PairedNeutral
                } else {
                    ReceiveEventKind::OrdinaryNeutral
                }
            } else if paired {
                ReceiveEventKind::PairedSource
            } else {
                ReceiveEventKind::OrdinarySource
            };
            events.push(ReceiveEvent {
                cycles: control_cycles,
                instruction: delay_xpic(0, u32::from(paired), xpic_value & 0x1fff),
                kind: xpic_kind,
            });
            let pic_selector = (instruction >> 27) & 1;
            let pic_value = payload & PIC_RECEIVE_ADDRESS_MASK;
            events.push(ReceiveEvent {
                cycles: control_cycles,
                instruction: delay_pic(0, pic_selector, pic_value),
                kind: if pic_selector == 0 {
                    ReceiveEventKind::Pointer
                } else {
                    ReceiveEventKind::Format
                },
            });
            cycles = cycles
                .checked_add(instruction_advance(instruction))
                .ok_or(ExchangeError::Schedule("receive event horizon overflow"))?;
            cursor += 2;
            continue;
        }

        cycles = cycles
            .checked_add(instruction_advance(instruction))
            .ok_or(ExchangeError::Schedule("receive event horizon overflow"))?;
        let kind = if instruction & OPCODE_MASK == DELAY_PIC_OPCODE {
            Some(if instruction & (1 << 18) == 0 {
                ReceiveEventKind::Pointer
            } else {
                ReceiveEventKind::Format
            })
        } else if instruction & OPCODE_MASK == DELAY_XPIC_OPCODE {
            let paired = instruction & (1 << 13) != 0;
            Some(if instruction & 0x1fff == TILE_MUX_EXCHANGE {
                if paired {
                    ReceiveEventKind::PairedNeutral
                } else {
                    ReceiveEventKind::OrdinaryNeutral
                }
            } else if paired {
                ReceiveEventKind::PairedSource
            } else {
                ReceiveEventKind::OrdinarySource
            })
        } else {
            None
        };
        if let Some(kind) = kind {
            events.push(ReceiveEvent {
                cycles,
                instruction,
                kind,
            });
        }
        cursor += 1;
    }

    for event in &events {
        match event.kind {
            ReceiveEventKind::OrdinarySource | ReceiveEventKind::PairedSource => {
                if source_cycles.replace(event.cycles).is_some() {
                    return Err(ExchangeError::Schedule("multiple receive sources"));
                }
            }
            ReceiveEventKind::OrdinaryNeutral | ReceiveEventKind::PairedNeutral => {
                if neutral_cycles.replace(event.cycles).is_some() {
                    return Err(ExchangeError::Schedule("multiple receive teardowns"));
                }
            }
            ReceiveEventKind::Pointer => {
                pointer_cycles = Some(event.cycles);
                if pointer_address
                    .replace((event.instruction & PIC_RECEIVE_ADDRESS_MASK) << 2)
                    .is_some()
                {
                    return Err(ExchangeError::Schedule("multiple receive pointers"));
                }
            }
            ReceiveEventKind::Format => {
                let value = event.instruction & PIC_RECEIVE_ADDRESS_MASK;
                if matches!(value, 1 | 2) {
                    if format_start_cycles.replace(event.cycles).is_some() {
                        return Err(ExchangeError::Schedule("multiple paired format starts"));
                    }
                } else if value == 0 {
                    if format_end_cycles.replace(event.cycles).is_some() {
                        return Err(ExchangeError::Schedule("multiple paired format ends"));
                    }
                } else {
                    return Err(ExchangeError::Schedule("paired receive format value"));
                }
            }
        }
    }
    let mode = if format_start_cycles.is_some()
        || events.iter().any(|event| {
            matches!(
                event.kind,
                ReceiveEventKind::PairedSource | ReceiveEventKind::PairedNeutral
            )
        }) {
        if format_start_cycles.is_none() || format_end_cycles.is_none() {
            return Err(ExchangeError::Schedule("incomplete paired receive format"));
        }
        ReceiveMode::Paired64
    } else {
        if source_cycles.is_none() || neutral_cycles.is_none() {
            return Err(ExchangeError::Schedule("incomplete ordinary receive mux"));
        }
        ReceiveMode::Ordinary
    };
    Ok(ReceiveRowTiming {
        mode,
        events,
        neutral_cycles,
        source_cycles,
        format_start_cycles,
        format_end_cycles,
        pointer_cycles,
        horizon_cycles: cycles,
        pointer_address,
    })
}

struct SenderRowTiming {
    start_cycles: u32,
    end_cycles: u32,
    horizon_cycles: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduledSenderTiming {
    pub payload_start: u32,
    pub payload_end: u32,
    pub horizon: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduledReceiverTiming {
    pub source_event: u32,
    pub pointer_event: Option<u32>,
    pub source_teardown: u32,
    pub horizon: u32,
}

pub fn scheduled_sender_timing(
    row: &PlanRow,
    schedule_offset: u32,
) -> Result<ScheduledSenderTiming, ExchangeError> {
    let timing = sender_row_timing(row, schedule_offset)?;
    Ok(ScheduledSenderTiming {
        payload_start: timing.start_cycles,
        payload_end: timing.end_cycles,
        horizon: timing.horizon_cycles,
    })
}

pub fn scheduled_receiver_timing(
    row: &PlanRow,
    schedule_offset: u32,
) -> Result<ScheduledReceiverTiming, ExchangeError> {
    let timing = receive_row_timing(row, schedule_offset)?;
    Ok(ScheduledReceiverTiming {
        source_event: timing
            .source_cycles
            .or(timing.format_start_cycles)
            .ok_or(ExchangeError::Schedule("receive source timing"))?,
        pointer_event: timing.pointer_cycles,
        source_teardown: timing
            .neutral_cycles
            .or(timing.format_end_cycles)
            .ok_or(ExchangeError::Schedule("receive teardown timing"))?,
        horizon: timing.horizon_cycles,
    })
}

fn sender_row_timing(
    row: &PlanRow,
    schedule_offset: u32,
) -> Result<SenderRowTiming, ExchangeError> {
    if row[0] != SYNC_SUPERVISOR_INSTRUCTION {
        return Err(ExchangeError::Schedule("sender row entry"));
    }
    let end = row
        .iter()
        .position(|instruction| *instruction == RETURN_M10_INSTRUCTION)
        .ok_or(ExchangeError::Schedule("sender row return"))?;
    let mut cycles = schedule_offset;
    let mut start_cycles = None;
    let mut end_cycles = None;
    for &instruction in &row[1..end] {
        if is_payload_send(instruction) {
            start_cycles.get_or_insert(cycles);
            cycles = cycles
                .checked_add(instruction_advance(instruction))
                .ok_or(ExchangeError::Schedule("sender event horizon overflow"))?;
            end_cycles = Some(cycles);
        } else {
            cycles = cycles
                .checked_add(instruction_advance(instruction))
                .ok_or(ExchangeError::Schedule("sender event horizon overflow"))?;
        }
    }
    Ok(SenderRowTiming {
        start_cycles: start_cycles.ok_or(ExchangeError::Schedule("sender payload"))?,
        end_cycles: end_cycles.ok_or(ExchangeError::Schedule("sender payload"))?,
        horizon_cycles: cycles,
    })
}

fn receive_row_timing_from_base(
    base: &ReceiveRowTiming,
    offset: u32,
) -> Result<ReceiveRowTiming, ExchangeError> {
    let shift = |cycles: u32| {
        cycles
            .checked_add(offset)
            .ok_or(ExchangeError::Schedule("receive offset overflow"))
    };
    Ok(ReceiveRowTiming {
        mode: base.mode,
        events: base
            .events
            .iter()
            .map(|event| {
                Ok(ReceiveEvent {
                    cycles: shift(event.cycles)?,
                    ..*event
                })
            })
            .collect::<Result<_, ExchangeError>>()?,
        neutral_cycles: base.neutral_cycles.map(shift).transpose()?,
        source_cycles: base.source_cycles.map(shift).transpose()?,
        format_start_cycles: base.format_start_cycles.map(shift).transpose()?,
        format_end_cycles: base.format_end_cycles.map(shift).transpose()?,
        pointer_cycles: base.pointer_cycles.map(shift).transpose()?,
        horizon_cycles: shift(base.horizon_cycles)?,
        pointer_address: base.pointer_address,
    })
}

fn build_scheduled_program(
    senders: &[ScheduledSenderRow],
    receive_events: &[ReceiveEvent],
    horizon_cycles: u32,
) -> Result<Vec<u32>, ExchangeError> {
    let mut senders = senders.iter().collect::<Vec<_>>();
    senders.sort_by_key(|sender| sender.start_cycles);
    if senders
        .windows(2)
        .any(|pair| pair[0].end_cycles > pair[1].start_cycles)
    {
        return Err(ExchangeError::Schedule("overlapping outgoing messages"));
    }
    let mut events = receive_events.to_vec();
    events.sort_by_key(|event| event.cycles);
    validate_receive_events(&events)?;

    let mut words = Vec::new();
    let mut event_cycles = 0;
    let mut event_index = 0;
    for sender in senders {
        let before_start =
            events[event_index..].partition_point(|event| event.cycles < sender.start_cycles);
        let split = event_index + before_start;
        append_receive_events(
            &mut words,
            &mut event_cycles,
            &events[event_index..split],
            sender.start_cycles,
            true,
        )?;
        event_index = split;
        let through_end =
            events[event_index..].partition_point(|event| event.cycles <= sender.end_cycles);
        let split = event_index + through_end;
        append_sender_message(
            &mut words,
            &mut event_cycles,
            sender,
            &events[event_index..split],
        )?;
        event_index = split;
    }
    append_receive_events(
        &mut words,
        &mut event_cycles,
        &events[event_index..],
        horizon_cycles,
        true,
    )?;
    words.push(RETURN_M10_INSTRUCTION);
    debug_assert_eq!(plan_event_cycles(&words)?, horizon_cycles);
    Ok(words)
}

fn append_sender_message(
    words: &mut Vec<u32>,
    event_cycles: &mut u32,
    sender: &ScheduledSenderRow,
    controls: &[ReceiveEvent],
) -> Result<(), ExchangeError> {
    append_plain_delay(words, event_cycles, sender.start_cycles)?;
    let (initial_instruction, payload_words) = sender_payload(&sender.row)?;
    let direction = initial_instruction & 7;
    let initial_source = (initial_instruction & SEND_ADDRESS_MASK) >> 3;
    let mut remaining = payload_words;
    let mut sent = 0u32;
    let mut started = false;

    let mut cursor = 0;
    while cursor < controls.len() {
        let end = cursor
            + controls[cursor..].partition_point(|event| event.cycles == controls[cursor].cycles);
        let group = &controls[cursor..end];
        let control_start = group[0]
            .cycles
            .checked_sub(1)
            .ok_or(ExchangeError::Schedule("send control at phase entry"))?;
        let distance = control_start
            .checked_sub(*event_cycles)
            .ok_or(ExchangeError::Schedule("send control order"))?;
        if distance >= remaining {
            return Err(ExchangeError::Schedule("send control beyond payload"));
        }
        emit_sender_words(
            words,
            event_cycles,
            initial_instruction,
            direction,
            &mut remaining,
            &mut sent,
            &mut started,
            distance,
            (group.len() == 2).then_some(0),
        )?;

        let next_boundary = controls
            .get(end)
            .map_or(sender.end_cycles, |next| next.cycles.saturating_sub(1));
        let mut available = next_boundary
            .checked_sub(*event_cycles)
            .ok_or(ExchangeError::Schedule("send control order"))?;
        let next_is_pair = controls.get(end + 1).is_some_and(|next| {
            controls
                .get(end)
                .is_some_and(|first| next.cycles == first.cycles)
        });
        let words_after_control = words.len() + group.len();
        if next_is_pair && words_after_control & 1 != 0 && available > 1 {
            // Leave one outgoing word for a SENDOFF instruction that aligns
            // the following two-word SENDPICP without changing its event.
            available -= 1;
        }
        let chunk = available.min(64).min(remaining);
        if chunk == 0 {
            return Err(ExchangeError::Schedule(
                "empty receive-control continuation",
            ));
        }
        if group.len() == 2 {
            let source = initial_source
                .checked_add(sent)
                .ok_or(ExchangeError::Schedule("SENDPICP source overflow"))?;
            let (instruction, payload) =
                encode_send_control_pair(chunk - 1, source, direction, group)?;
            words.extend([instruction, payload]);
        } else {
            words.push(encode_send_control(chunk - 1, group[0])?);
        }
        *event_cycles += chunk;
        remaining -= chunk;
        sent += chunk;
        cursor = end;
    }

    if !started {
        let first = remaining.min(64);
        words.push(resize_send(initial_instruction, first)?);
        *event_cycles += first;
        remaining -= first;
    }
    while remaining != 0 {
        let chunk = remaining.min(4096);
        words.push(send_off(chunk - 1, direction, 0));
        *event_cycles += chunk;
        remaining -= chunk;
    }
    if *event_cycles != sender.end_cycles {
        return Err(ExchangeError::Schedule("sender payload horizon"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_sender_words(
    words: &mut Vec<u32>,
    event_cycles: &mut u32,
    initial_instruction: u32,
    direction: u32,
    remaining: &mut u32,
    sent: &mut u32,
    started: &mut bool,
    count: u32,
    final_word_parity: Option<usize>,
) -> Result<(), ExchangeError> {
    if count > *remaining {
        return Err(ExchangeError::Schedule("sender payload overflow"));
    }
    if count == 0 {
        if !*started {
            return Err(ExchangeError::Schedule("unencodable initial send control"));
        }
        if final_word_parity.is_some_and(|parity| words.len() & 1 != parity) {
            return Err(ExchangeError::Schedule("SENDPICP instruction alignment"));
        }
        return Ok(());
    }

    let first_limit = if *started { 4096 } else { 64 };
    let mut chunks = vec![count.min(first_limit)];
    let mut unscheduled = count - chunks[0];
    while unscheduled != 0 {
        let chunk = unscheduled.min(4096);
        chunks.push(chunk);
        unscheduled -= chunk;
    }
    if final_word_parity.is_some_and(|parity| (words.len() + chunks.len()) & 1 != parity) {
        let split = chunks
            .iter()
            .rposition(|chunk| *chunk >= 2)
            .ok_or(ExchangeError::Schedule("SENDPICP instruction alignment"))?;
        let tail = chunks[split] - 1;
        chunks[split] = 1;
        chunks.insert(split + 1, tail);
    }

    for chunk in chunks {
        if !*started {
            words.push(resize_send(initial_instruction, chunk)?);
            *started = true;
        } else {
            words.push(send_off(chunk - 1, direction, 0));
        }
        *event_cycles += chunk;
        *remaining -= chunk;
        *sent += chunk;
    }
    if final_word_parity.is_some_and(|parity| words.len() & 1 != parity) {
        return Err(ExchangeError::Schedule("SENDPICP instruction alignment"));
    }
    Ok(())
}

fn sender_payload(row: &PlanRow) -> Result<(u32, u32), ExchangeError> {
    let mut initial = None;
    let mut words = 0u32;
    for &instruction in row {
        if instruction & LONG_OPCODE_MASK == SEND_OPCODE {
            if initial.replace(instruction).is_some() {
                return Err(ExchangeError::Schedule("multiple initial sends"));
            }
            words = words
                .checked_add(instruction_advance(instruction))
                .ok_or(ExchangeError::Schedule("sender payload overflow"))?;
        } else if is_send_off(instruction) {
            words = words
                .checked_add(instruction_advance(instruction))
                .ok_or(ExchangeError::Schedule("sender payload overflow"))?;
        }
    }
    Ok((
        initial.ok_or(ExchangeError::Schedule("sender payload"))?,
        words,
    ))
}

fn resize_send(instruction: u32, words: u32) -> Result<u32, ExchangeError> {
    if instruction & LONG_OPCODE_MASK != SEND_OPCODE || !(1..=64).contains(&words) {
        return Err(ExchangeError::Schedule("initial send size"));
    }
    Ok((instruction & !SEND_COUNT_MASK) | ((words - 1) << 21))
}

fn encode_send_control(count_minus_one: u32, event: ReceiveEvent) -> Result<u32, ExchangeError> {
    if count_minus_one > 63 {
        return Err(ExchangeError::Schedule("SENDPIC count"));
    }
    let (selector, operand) = match event.kind {
        ReceiveEventKind::OrdinarySource
        | ReceiveEventKind::OrdinaryNeutral
        | ReceiveEventKind::PairedSource
        | ReceiveEventKind::PairedNeutral => {
            ((event.instruction >> 13) & 1, event.instruction & 0x1fff)
        }
        ReceiveEventKind::Pointer | ReceiveEventKind::Format => (
            2 + ((event.instruction >> 18) & 1),
            event.instruction & PIC_RECEIVE_ADDRESS_MASK,
        ),
    };
    Ok(SEND_PIC_OPCODE | (count_minus_one << 21) | (selector << 18) | operand)
}

fn append_receive_events(
    words: &mut Vec<u32>,
    event_cycles: &mut u32,
    events: &[ReceiveEvent],
    horizon_cycles: u32,
    align_control_pairs: bool,
) -> Result<(), ExchangeError> {
    let mut cursor = 0;
    while cursor < events.len() {
        let end = cursor
            + events[cursor..].partition_point(|event| event.cycles == events[cursor].cycles);
        let group = &events[cursor..end];
        if group.len() == 2 {
            let instruction_start = group[0]
                .cycles
                .checked_sub(1)
                .ok_or(ExchangeError::Schedule("receive control at phase entry"))?;
            if align_control_pairs {
                append_plain_delay_aligned(words, event_cycles, instruction_start, 0)?;
            } else {
                // Primitive rows are a declarative timing representation and
                // are rebuilt phase-wide before execution. Avoid consuming a
                // spare word merely to align an instruction which is never
                // executed at this intermediate address.
                append_plain_delay(words, event_cycles, instruction_start)?;
            }
            let next_start = events
                .get(end)
                .map_or(horizon_cycles, |next| next.cycles.saturating_sub(1));
            let advance = next_start
                .checked_sub(*event_cycles)
                .ok_or(ExchangeError::Schedule("receive control order"))?
                .min(64);
            if advance == 0 {
                return Err(ExchangeError::Schedule("empty SENDPICP interval"));
            }
            let (instruction, payload) = encode_send_control_pair(advance - 1, 0, 0, group)?;
            words.extend([instruction, payload]);
            *event_cycles += advance;
        } else {
            let event = group[0];
            let advance = event
                .cycles
                .checked_sub(*event_cycles)
                .ok_or(ExchangeError::Schedule("receive event order"))?;
            if advance == 0 {
                return Err(ExchangeError::Schedule("receive control issue interval"));
            }
            let mut instruction = event.instruction;
            let maximum_advance = maximum_timed_instruction_advance(instruction)
                .ok_or(ExchangeError::Schedule("receive timed event"))?;
            if advance > maximum_advance {
                append_plain_delay(words, event_cycles, event.cycles - maximum_advance)?;
            }
            set_instruction_advance(&mut instruction, event.cycles - *event_cycles)?;
            words.push(instruction);
            *event_cycles = event.cycles;
        }
        cursor = end;
    }
    append_plain_delay(words, event_cycles, horizon_cycles)
}

fn encode_send_control_pair(
    count_minus_one: u32,
    source_word_address: u32,
    send_control: u32,
    events: &[ReceiveEvent],
) -> Result<(u32, u32), ExchangeError> {
    if count_minus_one > 63
        || source_word_address > SEND_ADDRESS_MASK >> 3
        || send_control > 7
        || events.len() != 2
    {
        return Err(ExchangeError::Schedule("SENDPICP operand"));
    }
    let pic = events
        .iter()
        .find(|event| event.kind.is_pic())
        .ok_or(ExchangeError::Schedule("SENDPICP PIC control"))?;
    let xpic = events
        .iter()
        .find(|event| event.kind.is_xpic())
        .ok_or(ExchangeError::Schedule("SENDPICP XPIC control"))?;
    if pic.cycles != xpic.cycles {
        return Err(ExchangeError::Schedule("SENDPICP event time"));
    }
    // SENDPICP uses the same outgoing fields as SEND: an absolute source word
    // address followed by the three-bit SCTL field. Its fourth operand carries
    // the high PIC selector bit; the inline word holds all fourteen XPIC bits
    // and the remaining eighteen PIC bits.
    let pointer_selector = (pic.instruction >> 18) & 1;
    let instruction = SEND_PICP_OPCODE
        | (pointer_selector << 27)
        | (count_minus_one << 21)
        | ((source_word_address << 3) & SEND_ADDRESS_MASK)
        | send_control;
    let payload =
        ((xpic.instruction & 0x3fff) << 18) | (pic.instruction & PIC_RECEIVE_ADDRESS_MASK);
    Ok((instruction, payload))
}

fn append_plain_delay(
    words: &mut Vec<u32>,
    event_cycles: &mut u32,
    target_cycles: u32,
) -> Result<(), ExchangeError> {
    let mut remaining = target_cycles
        .checked_sub(*event_cycles)
        .ok_or(ExchangeError::Schedule("plan delay order"))?;
    while remaining != 0 {
        let chunk = remaining.min(MAX_PLAN_OFFSET_CYCLES);
        words.push(delay(chunk - 1));
        *event_cycles += chunk;
        remaining -= chunk;
    }
    Ok(())
}

/// Advances with plain delays while arranging the next word at the requested
/// two-word-instruction alignment. Exchange rows are placed at eight-byte
/// boundaries, so `word_parity == 0` aligns a following SENDPICP and payload.
fn append_plain_delay_aligned(
    words: &mut Vec<u32>,
    event_cycles: &mut u32,
    target_cycles: u32,
    word_parity: usize,
) -> Result<(), ExchangeError> {
    let distance = target_cycles
        .checked_sub(*event_cycles)
        .ok_or(ExchangeError::Schedule("plan delay order"))?;
    if distance == 0 {
        return (words.len() & 1 == word_parity)
            .then_some(())
            .ok_or(ExchangeError::Schedule("SENDPICP instruction alignment"));
    }

    let minimum_words = distance.div_ceil(MAX_PLAN_OFFSET_CYCLES);
    let mut delay_words = minimum_words;
    if ((words.len() + delay_words as usize) & 1) != word_parity {
        delay_words += 1;
    }
    if delay_words > distance {
        return Err(ExchangeError::Schedule("SENDPICP instruction alignment"));
    }
    let mut remaining = distance;
    for remaining_words in (1..=delay_words).rev() {
        let chunk = remaining
            .saturating_sub(remaining_words - 1)
            .min(MAX_PLAN_OFFSET_CYCLES);
        words.push(delay(chunk - 1));
        *event_cycles += chunk;
        remaining -= chunk;
    }
    debug_assert_eq!(remaining, 0);
    debug_assert_eq!(words.len() & 1, word_parity);
    Ok(())
}

fn maximum_timed_instruction_advance(instruction: u32) -> Option<u32> {
    match instruction & OPCODE_MASK {
        DELAY_PIC_OPCODE => Some(0x80),
        DELAY_XPIC_OPCODE => Some(0x1000),
        _ => None,
    }
}

#[cfg(test)]
fn is_neutral_mux_teardown(instruction: u32) -> bool {
    instruction & OPCODE_MASK == DELAY_XPIC_OPCODE && instruction & 0x1fff == TILE_MUX_EXCHANGE
}

fn instruction_advance(instruction: u32) -> u32 {
    if instruction & DELAY_OPCODE_MASK == DELAY_OPCODE {
        (instruction & 0x7_ffff) + 1
    } else {
        match instruction & OPCODE_MASK {
            DELAY_PIC_OPCODE => ((instruction >> 19) & 0x7f) + 1,
            DELAY_XPIC_OPCODE => ((instruction >> 14) & 0xfff) + 1,
            _ if instruction & LONG_OPCODE_MASK == SEND_OPCODE => ((instruction >> 21) & 0x3f) + 1,
            _ if is_send_control_pair(instruction) => ((instruction >> 21) & 0x3f) + 1,
            _ if is_send_control(instruction) => ((instruction >> 21) & 0x3f) + 1,
            _ if is_send_off(instruction) => {
                (((instruction >> 21) & 0x3f) | (((instruction >> 14) & 0x3f) << 6)) + 1
            }
            _ => 0,
        }
    }
}

fn is_send_control(instruction: u32) -> bool {
    instruction & SEND_CONTROL_OPCODE_MASK == SEND_PIC_OPCODE
}

fn is_send_control_pair(instruction: u32) -> bool {
    instruction & SEND_PICP_OPCODE_MASK == SEND_PICP_OPCODE
}

fn is_send_off(instruction: u32) -> bool {
    instruction & SEND_CONTROL_OPCODE_MASK == SEND_OFF_OPCODE
}

fn is_payload_send(instruction: u32) -> bool {
    instruction & LONG_OPCODE_MASK == SEND_OPCODE || is_send_off(instruction)
}

fn set_instruction_advance(instruction: &mut u32, advance: u32) -> Result<(), ExchangeError> {
    if advance == 0 {
        return Err(ExchangeError::Schedule("zero event advance"));
    }
    let immediate = advance - 1;
    if *instruction & DELAY_OPCODE_MASK == DELAY_OPCODE {
        if immediate > 0x7_ffff {
            return Err(ExchangeError::Schedule("delay advance overflow"));
        }
        *instruction = (*instruction & !0x7_ffff) | immediate;
    } else if *instruction & OPCODE_MASK == DELAY_PIC_OPCODE {
        if immediate > 0x7f {
            return Err(ExchangeError::Schedule("PIC delay advance overflow"));
        }
        *instruction = (*instruction & !0x03f8_0000) | (immediate << 19);
    } else if *instruction & OPCODE_MASK == DELAY_XPIC_OPCODE {
        if immediate > 0xfff {
            return Err(ExchangeError::Schedule("XPIC delay advance overflow"));
        }
        *instruction = (*instruction & !0x03ff_c000) | (immediate << 14);
    } else {
        return Err(ExchangeError::Schedule(
            if is_payload_send(*instruction) || is_send_control(*instruction) {
                "scheduled offset truncates a send"
            } else {
                "first scheduled event is not a delay"
            },
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostPacketHeader {
    pub word0: u32,
    pub word1: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostTransferChunk {
    pub tile_address: u32,
    pub host_offset: u32,
    pub bytes: u32,
    pub header: HostPacketHeader,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileToHostProgram {
    pub instructions: Vec<u32>,
    pub packet_words: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostHierarchy {
    pub xreq_physical_tile: u16,
    pub target_physical_tile: u16,
}

pub fn host_hierarchy(target_physical_tile: u16) -> Result<HostHierarchy, ExchangeError> {
    validate_host_tile(target_physical_tile)?;
    Ok(HostHierarchy {
        xreq_physical_tile: target_physical_tile & 0x3d,
        target_physical_tile,
    })
}

pub fn assemble_host_xreq_program(
    target_physical_tile: u16,
    packet_address: u32,
) -> Result<TileToHostProgram, ExchangeError> {
    assemble_host_xreq_program_for_targets(&[target_physical_tile], packet_address)
}

pub fn assemble_host_xreq_program_for_targets(
    target_physical_tiles: &[u16],
    packet_address: u32,
) -> Result<TileToHostProgram, ExchangeError> {
    if target_physical_tiles.is_empty() {
        return Err(ExchangeError::HostPacket);
    }
    if packet_address & 7 != 0 {
        return Err(ExchangeError::HostPacket);
    }
    let mut bitmap = [0u32; 2];
    for &target_physical_tile in target_physical_tiles {
        validate_host_tile(target_physical_tile)?;
        let bitmap_index =
            u32::from(target_physical_tile / 64) * 2 + u32::from((target_physical_tile >> 1) & 1);
        if bitmap_index < XREQ_BITMAP0_BITS {
            bitmap[0] |= 1 << bitmap_index;
        } else {
            bitmap[1] |= 1 << (bitmap_index - XREQ_BITMAP0_BITS);
        }
    }
    Ok(TileToHostProgram {
        instructions: vec![
            encode_send(1, 3, packet_address >> 2)?,
            RETURN_M10_INSTRUCTION,
        ],
        packet_words: bitmap.to_vec(),
    })
}

pub fn assemble_host_command_read_program(
    packet_address: u32,
    destination_address: u32,
    host_offset: u32,
) -> Result<TileToHostProgram, ExchangeError> {
    if packet_address & 7 != 0 {
        return Err(ExchangeError::HostPacket);
    }
    let request = host_to_tile_packet(0, destination_address, host_offset, 4)?;
    let mut instructions = vec![
        setzi_m(8, TILE_MUX_HOST),
        put_special_from_m8(INCOMING_MUX_REGISTER),
        SYNC_HOST_INSTRUCTION,
        setzi_m(8, 1),
        put_special_from_m8(INCOMING_DCOUNT_REGISTER),
        encode_send(1, 3, packet_address >> 2)?,
        encode_send(1, 3, (packet_address + 8) >> 2)?,
        SYNC_RECEIVE_INSTRUCTION,
    ];
    append_local_host_completion(&mut instructions);
    instructions.extend([
        SYNC_SUPERVISOR_INSTRUCTION,
        delay(HOST_COMMAND_ROUTE_CYCLES - 1),
        encode_send(0, 3, destination_address >> 2)?,
    ]);
    instructions.push(RETURN_M10_INSTRUCTION);
    Ok(TileToHostProgram {
        instructions,
        packet_words: vec![1, 0, request.word0, request.word1],
    })
}

pub fn assemble_host_to_tile_target_program(
    physical_tile: u16,
    tile_address: u32,
    host_offset: u32,
    bytes: u32,
    packet_address: u32,
) -> Result<TileToHostProgram, ExchangeError> {
    let chunks = plan_host_to_tile(physical_tile, tile_address, host_offset, bytes)?;
    if packet_address & 7 != 0 {
        return Err(ExchangeError::HostPacket);
    }
    let mut instructions = vec![
        setzi_m(8, bytes / 4),
        put_special_from_m8(INCOMING_DCOUNT_REGISTER),
        encode_send(1, 3, packet_address >> 2)?,
    ];
    for _ in 1..chunks.len() {
        instructions.push(send_off(1, 3, 0));
    }
    instructions.push(SYNC_RECEIVE_INSTRUCTION);
    instructions.push(RETURN_M10_INSTRUCTION);
    let packet_words = chunks
        .iter()
        .enumerate()
        .flat_map(|(index, chunk)| {
            let word0 = if index + 1 == chunks.len() {
                chunk.header.word0
            } else {
                chunk.header.word0 & !HOST_TO_TILE_STREAM_END_BITS
            };
            [word0, chunk.header.word1]
        })
        .collect();
    Ok(TileToHostProgram {
        instructions,
        packet_words,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HostPacketSize {
    Short,
    Long,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    pub sender: PlanRow,
    pub receiver: PlanRow,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MulticastPlan {
    pub sender: PlanRow,
    pub receivers: Vec<PlanRow>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology {
    logical_to_physical: Vec<u16>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ExchangeError {
    #[error("logical tile {0} is out of range")]
    Tile(u16),
    #[error("exchange endpoints must be distinct")]
    DuplicateTile,
    #[error("exchange count {0} is outside 1..={MAX_TRANSFER_WORDS}")]
    Count(u32),
    #[error("invalid multicast receiver set")]
    ReceiverSet,
    #[error("exchange schedule is not encodable: {0}")]
    Schedule(&'static str),
    #[error(
        "exchange event offset {cycles} cycles from the phase-entry sync exceeds the encodable maximum of {maximum} cycles"
    )]
    PlanOffsetRange { cycles: u32, maximum: u32 },
    #[error("tile address 0x{0:x} is not encodable")]
    Address(u32),
    #[error("host exchange address or length is not encodable")]
    HostPacket,
}

pub fn host_to_tile_packet(
    physical_tile: u16,
    tile_address: u32,
    host_offset: u32,
    bytes: u32,
) -> Result<HostPacketHeader, ExchangeError> {
    validate_host_tile(physical_tile)?;
    if tile_address < EXCHANGE_WINDOW_BASE || tile_address & 31 != 0 {
        return Err(ExchangeError::HostPacket);
    }
    let exchange_address = (tile_address - EXCHANGE_WINDOW_BASE) >> 5;
    if tile_address >= EXCHANGE_WINDOW_BASE + HOST_TO_TILE_WINDOW_BYTES {
        return Err(ExchangeError::HostPacket);
    }
    let size = host_packet_size(host_offset, bytes)?;
    let opcode = match size {
        HostPacketSize::Short => 0xcc00_0200,
        HostPacketSize::Long => 0xec00_0200,
    };
    Ok(HostPacketHeader {
        word0: opcode | host_route_word0(physical_tile) | exchange_address,
        word1: host_route_word1(physical_tile) | host_address_length(host_offset, bytes, size)?,
    })
}

pub fn tile_to_host_packet(
    physical_tile: u16,
    host_offset: u32,
    bytes: u32,
) -> Result<HostPacketHeader, ExchangeError> {
    validate_host_tile(physical_tile)?;
    let size = host_packet_size(host_offset, bytes)?;
    let opcode = match size {
        HostPacketSize::Short => 0x8000_0000,
        HostPacketSize::Long => 0xa000_0000,
    };
    Ok(HostPacketHeader {
        word0: opcode | host_route_word0(physical_tile),
        word1: host_route_word1(physical_tile) | host_address_length(host_offset, bytes, size)?,
    })
}

pub fn zero_byte_read_packet(
    physical_tile: u16,
    dummy_tile_address: u32,
) -> Result<HostPacketHeader, ExchangeError> {
    validate_host_tile(physical_tile)?;
    if dummy_tile_address < EXCHANGE_WINDOW_BASE || dummy_tile_address & 31 != 0 {
        return Err(ExchangeError::HostPacket);
    }
    let exchange_address = (dummy_tile_address - EXCHANGE_WINDOW_BASE) >> 5;
    if dummy_tile_address >= EXCHANGE_WINDOW_BASE + HOST_TO_TILE_WINDOW_BYTES {
        return Err(ExchangeError::HostPacket);
    }
    Ok(HostPacketHeader {
        word0: 0xcc00_0200 | host_route_word0(physical_tile) | exchange_address,
        word1: host_route_word1(physical_tile),
    })
}

pub fn plan_host_to_tile(
    physical_tile: u16,
    tile_address: u32,
    host_offset: u32,
    bytes: u32,
) -> Result<Vec<HostTransferChunk>, ExchangeError> {
    if bytes == 0 || bytes & 3 != 0 || tile_address & 31 != 0 || host_offset & 3 != 0 {
        return Err(ExchangeError::HostPacket);
    }
    plan_host_transfer(
        tile_address,
        host_offset,
        bytes,
        HOST_LONG_MAX_BYTES,
        |tile, host, count| host_to_tile_packet(physical_tile, tile, host, count),
    )
}

pub fn plan_tile_to_host(
    physical_tile: u16,
    tile_address: u32,
    host_offset: u32,
    bytes: u32,
) -> Result<Vec<HostTransferChunk>, ExchangeError> {
    if bytes == 0 || bytes & 3 != 0 || tile_address & 3 != 0 || host_offset & 3 != 0 {
        return Err(ExchangeError::HostPacket);
    }
    plan_host_transfer(
        tile_address,
        host_offset,
        bytes,
        TILE_TO_HOST_MAX_BYTES,
        |_tile, host, count| tile_to_host_packet(physical_tile, host, count),
    )
}

pub fn assemble_tile_to_host_target_program(
    physical_tile: u16,
    tile_address: u32,
    host_offset: u32,
    bytes: u32,
    packet_address: u32,
    command_address: u32,
) -> Result<TileToHostProgram, ExchangeError> {
    let chunks = plan_tile_to_host(physical_tile, tile_address, host_offset, bytes)?;
    if packet_address & 7 != 0 {
        return Err(ExchangeError::HostPacket);
    }
    let mut packet_words: Vec<u32> = chunks
        .iter()
        .flat_map(|chunk| [chunk.header.word0, chunk.header.word1])
        .collect();
    let close_address = packet_address
        .checked_add(u32::try_from(packet_words.len() * 4).map_err(|_| ExchangeError::HostPacket)?)
        .ok_or(ExchangeError::HostPacket)?;
    let close = zero_byte_read_packet(physical_tile, command_address)?;
    packet_words.extend([close.word0, close.word1]);
    Ok(TileToHostProgram {
        instructions: tile_to_host_target_instructions(&chunks, packet_address, close_address)?,
        packet_words,
    })
}

fn tile_to_host_target_instructions(
    chunks: &[HostTransferChunk],
    packet_address: u32,
    close_address: u32,
) -> Result<Vec<u32>, ExchangeError> {
    if chunks.is_empty() || packet_address & 7 != 0 {
        return Err(ExchangeError::HostPacket);
    }
    let header_base = packet_address;
    let mut instructions = vec![setzi_m(8, 1), put_special_from_m8(INCOMING_DCOUNT_REGISTER)];
    for (index, chunk) in chunks.iter().enumerate() {
        instructions.push(encode_send(1, 3, (header_base + index as u32 * 8) >> 2)?);
        instructions.push(encode_send(
            chunk.bytes / 4 - 1,
            3,
            chunk.tile_address >> 2,
        )?);
        let payload_events = chunk.bytes / 4;
        if payload_events < TILE_TO_HOST_MIN_PAYLOAD_EVENTS {
            instructions.push(delay(TILE_TO_HOST_MIN_PAYLOAD_EVENTS - payload_events - 1));
        }
    }
    instructions.push(encode_send(1, 3, close_address >> 2)?);
    instructions.push(SYNC_RECEIVE_INSTRUCTION);
    instructions.push(RETURN_M10_INSTRUCTION);
    Ok(instructions)
}

fn append_local_host_completion(instructions: &mut Vec<u32>) {
    instructions.extend([
        SYNC_ALL_INSTRUCTION,
        setzi_m(8, TILE_MUX_EXCHANGE),
        put_special_from_m8(INCOMING_MUX_REGISTER),
    ]);
}

fn wrap_host_operation(
    physical_tile: u16,
    operation: &[u32],
    entry_sync: u32,
) -> Result<Vec<u32>, ExchangeError> {
    let Some((&RETURN_M10_INSTRUCTION, body)) = operation.split_last() else {
        return Err(ExchangeError::Schedule("host target operation return"));
    };
    let mut instructions = vec![
        setzi_m(8, host_mux_for_tile(physical_tile)?),
        put_special_from_m8(INCOMING_MUX_REGISTER),
        entry_sync,
    ];
    instructions.extend_from_slice(body);
    instructions.extend([
        SYNC_ALL_INSTRUCTION,
        setzi_m(8, TILE_MUX_EXCHANGE),
        put_special_from_m8(INCOMING_MUX_REGISTER),
        RETURN_M10_INSTRUCTION,
    ]);
    Ok(instructions)
}

pub fn wrap_host_xreq_operation(
    physical_tile: u16,
    operation: &[u32],
) -> Result<Vec<u32>, ExchangeError> {
    wrap_host_operation(physical_tile, operation, SYNC_HOST_INSTRUCTION)
}

pub fn wrap_host_target_operation(
    physical_tile: u16,
    operation: &[u32],
) -> Result<Vec<u32>, ExchangeError> {
    wrap_host_operation(physical_tile, operation, SYNC_ALL_INSTRUCTION)
}

pub fn wrap_combined_host_operation(
    physical_tile: u16,
    operation: &[u32],
    xreq_packet_address: u32,
) -> Result<Vec<u32>, ExchangeError> {
    let Some((&RETURN_M10_INSTRUCTION, body)) = operation.split_last() else {
        return Err(ExchangeError::Schedule("local host operation return"));
    };
    if body.len() < 2 || xreq_packet_address & 7 != 0 {
        return Err(ExchangeError::Schedule("local host operation prefix"));
    }
    let mut instructions = vec![
        setzi_m(8, host_mux_for_tile(physical_tile)?),
        put_special_from_m8(INCOMING_MUX_REGISTER),
        SYNC_HOST_INSTRUCTION,
    ];
    instructions.extend_from_slice(&body[..2]);
    instructions.push(encode_send(1, 3, xreq_packet_address >> 2)?);
    instructions.extend_from_slice(&body[2..]);
    instructions.extend([
        SYNC_ALL_INSTRUCTION,
        setzi_m(8, TILE_MUX_EXCHANGE),
        put_special_from_m8(INCOMING_MUX_REGISTER),
        RETURN_M10_INSTRUCTION,
    ]);
    Ok(instructions)
}

fn plan_host_transfer(
    mut tile_address: u32,
    mut host_offset: u32,
    mut bytes: u32,
    long_max_bytes: u32,
    packet: impl Fn(u32, u32, u32) -> Result<HostPacketHeader, ExchangeError>,
) -> Result<Vec<HostTransferChunk>, ExchangeError> {
    let mut chunks = Vec::new();
    while bytes != 0 {
        let page_bytes = HOST_PAGE_BYTES - host_offset % HOST_PAGE_BYTES;
        let available = bytes.min(page_bytes);
        let count = if host_offset & 63 == 0 && available >= 64 {
            available.min(long_max_bytes) & !63
        } else if available <= HOST_SHORT_MAX_BYTES {
            available
        } else {
            // Keeping intermediate short packets at 32 bytes also preserves
            // the destination alignment required by host-to-tile requests.
            32
        };
        let header = packet(tile_address, host_offset, count)?;
        chunks.push(HostTransferChunk {
            tile_address,
            host_offset,
            bytes: count,
            header,
        });
        tile_address = tile_address
            .checked_add(count)
            .ok_or(ExchangeError::HostPacket)?;
        host_offset = host_offset
            .checked_add(count)
            .ok_or(ExchangeError::HostPacket)?;
        bytes -= count;
    }
    Ok(chunks)
}

fn validate_host_tile(physical_tile: u16) -> Result<(), ExchangeError> {
    if physical_tile > 0xfff {
        return Err(ExchangeError::HostPacket);
    }
    Ok(())
}

pub fn host_mux_for_tile(physical_tile: u16) -> Result<u32, ExchangeError> {
    validate_host_tile(physical_tile)?;
    Ok(TILE_MUX_HOST + u32::from((physical_tile & 0x3f) & !2))
}

fn host_packet_size(host_offset: u32, bytes: u32) -> Result<HostPacketSize, ExchangeError> {
    if (4..=HOST_SHORT_MAX_BYTES).contains(&bytes) && host_offset & 3 == 0 && bytes & 3 == 0 {
        return Ok(HostPacketSize::Short);
    }
    if (64..=HOST_LONG_MAX_BYTES).contains(&bytes) && host_offset & 63 == 0 && bytes & 63 == 0 {
        return Ok(HostPacketSize::Long);
    }
    Err(ExchangeError::HostPacket)
}

fn host_address_length(
    host_offset: u32,
    bytes: u32,
    size: HostPacketSize,
) -> Result<u32, ExchangeError> {
    let shift = match size {
        HostPacketSize::Short => 2,
        HostPacketSize::Long => 6,
    };
    let units = bytes >> shift;
    let length = if size == HostPacketSize::Long && bytes == HOST_LONG_MAX_BYTES {
        0
    } else {
        units
    };
    (u64::from(host_offset >> shift) << 4 | u64::from(length))
        .try_into()
        .ok()
        .filter(|encoded: &u32| *encoded <= 0x7fff_ffff)
        .ok_or(ExchangeError::HostPacket)
}

fn host_route_word0(physical_tile: u16) -> u32 {
    let tile = u32::from(physical_tile);
    ((tile >> 1) << 16) | ((tile & 1) << 15)
}

fn host_route_word1(physical_tile: u16) -> u32 {
    u32::from(physical_tile & 1) << 31
}

impl Topology {
    pub fn new(logical_to_physical: Vec<u16>) -> Result<Self, ExchangeError> {
        let mut physical = HashSet::new();
        if logical_to_physical.is_empty()
            || logical_to_physical
                .iter()
                .any(|tile| !physical.insert(*tile))
        {
            return Err(ExchangeError::ReceiverSet);
        }
        Ok(Self {
            logical_to_physical,
        })
    }

    pub fn c600() -> Self {
        Self {
            logical_to_physical: (0..1472).map(c600_logical_to_physical).collect(),
        }
    }

    pub fn tile_count(&self) -> usize {
        self.logical_to_physical.len()
    }

    pub fn physical(&self, logical: u16) -> Result<u16, ExchangeError> {
        self.logical_to_physical
            .get(usize::from(logical))
            .copied()
            .ok_or(ExchangeError::Tile(logical))
    }

    /// Logical tile that shares this tile's double-width exchange resources.
    pub fn paired_logical(&self, logical: u16) -> Result<u16, ExchangeError> {
        let paired_physical = self.physical(logical)? ^ 2;
        self.logical_to_physical
            .iter()
            .position(|physical| *physical == paired_physical)
            .map(|paired| u16::try_from(paired).expect("logical tile count fits u16"))
            .ok_or(ExchangeError::Tile(logical))
    }

    /// Physical source selected by `INCOMING_MUXPAIR` for a 64-bit send.
    pub fn paired_source_mux(&self, sender_logical: u16) -> Result<u16, ExchangeError> {
        Ok(self.physical(sender_logical)? ^ 2)
    }

    pub fn is_pair_primary(&self, logical: u16) -> Result<bool, ExchangeError> {
        Ok(self.physical(logical)? & 2 == 0)
    }

    /// Direction and width control for one 64-bit route. Both borrowed lanes
    /// follow the same fabric direction; the two members of a receiving pair
    /// are not an ordinary two-direction multicast.
    pub fn paired_send_control(
        &self,
        sender_logical: u16,
        receiver_logical: u16,
    ) -> Result<u8, ExchangeError> {
        let sender = u32::from(self.physical(sender_logical)?);
        let receiver = u32::from(self.physical(receiver_logical)?);
        Ok(u8::try_from(direction(sender, receiver) | 4).expect("send control is three bits"))
    }

    /// Whether this member of a double-width receiving pair owns the paired
    /// XPIC source-selection stream. This matches the SDK architecture
    /// helper `TPair_RxIsEarly`.
    pub fn paired_receiver_is_early(
        &self,
        receiver_logical: u16,
        sender_logical: u16,
    ) -> Result<bool, ExchangeError> {
        let receiver = u32::from(self.physical(receiver_logical)?);
        let sender = u32::from(self.physical(sender_logical)?);
        let local = time_to_mux(sender, receiver);
        let borrowed = time_to_mux(sender, receiver ^ 2);
        Ok(local < borrowed)
    }

    /// Builds an SDK-compatible double-width multicast. Receivers must be
    /// complete physical tile pairs; both members consume the same 64-bit
    /// item stream.
    pub fn paired_multicast(
        &self,
        sender_logical: u16,
        receivers: &[u16],
        count: u32,
    ) -> Result<MulticastPlan, ExchangeError> {
        validate_count(count)?;
        if count < 64 {
            return Err(ExchangeError::Count(count));
        }
        if receivers.is_empty() || receivers.len() & 1 != 0 {
            return Err(ExchangeError::ReceiverSet);
        }
        let receiver_set = receivers.iter().copied().collect::<HashSet<_>>();
        if receiver_set.len() != receivers.len()
            || receiver_set.contains(&sender_logical)
            || receiver_set.contains(&self.paired_logical(sender_logical)?)
            || receivers.iter().any(|&receiver| {
                self.paired_logical(receiver)
                    .map_or(true, |paired| !receiver_set.contains(&paired))
            })
        {
            return Err(ExchangeError::ReceiverSet);
        }

        let mut plan = self.multicast(sender_logical, receivers, count, 0)?;
        let sender_physical = u32::from(self.physical(sender_logical)?);
        let send_control = if receivers.len() == 2 {
            u8::try_from(direction(sender_physical, u32::from(self.physical(receivers[0])?)) | 4)
                .expect("send control is three bits")
        } else {
            7
        };
        set_sender_control(&mut plan.sender, send_control)?;
        let receiver_physical = receivers
            .iter()
            .map(|&receiver| self.physical(receiver).map(u32::from))
            .collect::<Result<Vec<_>, _>>()?;
        let minimum_double_mux = receiver_physical
            .iter()
            .map(|&receiver| paired_time_to_mux(sender_physical, receiver))
            .min()
            .expect("paired multicast has receivers");
        plan.receivers = receivers
            .iter()
            .map(|&receiver| {
                self.paired_receiver_row(sender_logical, receiver, count, minimum_double_mux)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(plan)
    }

    fn paired_receiver_row(
        &self,
        sender_logical: u16,
        receiver_logical: u16,
        count: u32,
        minimum_double_mux: i32,
    ) -> Result<PlanRow, ExchangeError> {
        let sender = u32::from(self.physical(sender_logical)?);
        let receiver = u32::from(self.physical(receiver_logical)?);
        // Route times are already expressed in the exchange epoch's event
        // clock when every receiver is on the positive side of the timing
        // origin. Shift the whole multicast only when its earliest pair would
        // otherwise configure the mux before event one.
        let epoch_shift = (1 - minimum_double_mux).max(0);
        let source_event = u32::try_from(paired_time_to_mux(sender, receiver) + epoch_shift)
            .map_err(|_| ExchangeError::Schedule("paired source timing"))?;
        // MXP is 59 plus twice the physical row. The SDK selects paired mode
        // MXP-8 cycles after this pair's route-specific XPIC source event.
        // Adjacent logical pairs can therefore have distinct format windows
        // when the logical-to-physical mapping turns into another column.
        let format_start = source_event + 51 + 2 * (receiver >> 6);
        let mut events = Vec::with_capacity(5);
        if self.paired_receiver_is_early(receiver_logical, sender_logical)? {
            events.push(ReceiveEvent {
                cycles: source_event,
                instruction: delay_xpic(0, 1, sender ^ (receiver & 2)),
                kind: ReceiveEventKind::PairedSource,
            });
            events.push(ReceiveEvent {
                cycles: source_event + count,
                instruction: delay_xpic(0, 1, TILE_MUX_EXCHANGE),
                kind: ReceiveEventKind::PairedNeutral,
            });
        }
        events.extend([
            ReceiveEvent {
                cycles: format_start,
                instruction: delay_pic(0, 1, direction(sender, receiver)),
                kind: ReceiveEventKind::Format,
            },
            ReceiveEvent {
                cycles: format_start + 2,
                instruction: delay_pic(0, 0, 0),
                kind: ReceiveEventKind::Pointer,
            },
            ReceiveEvent {
                cycles: format_start + count,
                instruction: delay_pic(0, 1, 0),
                kind: ReceiveEventKind::Format,
            },
        ]);
        events.sort_by_key(|event| event.cycles);
        // Directionless SENDPICP can apply ordinary PIC and XPIC controls on
        // one event, but the paired receive path faults if format activation
        // coincides with its source selection or teardown. Reject that row so
        // placement-aware lowering can retain a Word32 transfer instead.
        if events
            .windows(2)
            .any(|pair| pair[0].cycles == pair[1].cycles)
        {
            return Err(ExchangeError::Schedule(
                "paired receive coincident controls",
            ));
        }
        validate_receive_events(&events)?;

        let horizon = events
            .last()
            .expect("paired receive always has format events")
            .cycles
            .checked_add(7)
            .ok_or(ExchangeError::Schedule("paired receive horizon overflow"))?;
        let mut words = vec![SYNC_SUPERVISOR_INSTRUCTION];
        let mut cycles = 0;
        append_receive_events(&mut words, &mut cycles, &events, horizon, false)?;
        words.push(RETURN_M10_INSTRUCTION);
        if words.len() > PLAN_WORDS {
            return Err(ExchangeError::Schedule("paired receive row capacity"));
        }
        let mut row = [0; PLAN_WORDS];
        row[..words.len()].copy_from_slice(&words);
        Ok(row)
    }

    pub fn point_to_point(
        &self,
        sender_logical: u16,
        receiver_logical: u16,
        count: u32,
    ) -> Result<Plan, ExchangeError> {
        validate_count(count)?;
        if sender_logical == receiver_logical {
            return Err(ExchangeError::DuplicateTile);
        }
        let sender = u32::from(self.physical(sender_logical)?);
        let receiver = u32::from(self.physical(receiver_logical)?);
        let direction = direction(sender, receiver);
        let mux_time = time_to_mux(sender, receiver);
        let receiver_phase = 2 * (receiver >> 6);
        let sender_delay = 111 - mux_time;
        if !(-1..=0x7ffff).contains(&sender_delay) {
            return Err(ExchangeError::Schedule("sender delay"));
        }

        let mut sender_row = [0; PLAN_WORDS];
        sender_row[0] = SYNC_SUPERVISOR_INSTRUCTION;
        let mut cursor = 1;
        if sender_delay >= 0 {
            sender_row[cursor] = delay(sender_delay as u32);
            cursor += 1;
        }
        let first_packet = count.min(64);
        sender_row[cursor] = encode_send(first_packet - 1, direction, 0)?;
        cursor += 1;
        if count > 64 {
            sender_row[cursor] = send_off(count - 65, direction, 0);
            cursor += 1;
        }
        let trailing_delay = 4 - sender_delay - count as i32;
        if trailing_delay >= 0 {
            sender_row[cursor] = delay(trailing_delay as u32);
            cursor += 1;
        }
        sender_row[cursor] = RETURN_M10_INSTRUCTION;

        let mut receiver_row = [0; PLAN_WORDS];
        receiver_row[0] = 1;
        receiver_row[1] = SYNC_SUPERVISOR_INSTRUCTION;
        receiver_row[2] = delay_xpic(112, 0, 0);
        if count <= 51 {
            receiver_row[3] = delay_xpic(count - 1, 0, TILE_MUX_EXCHANGE);
            receiver_row[4] = delay_pic(51 - count + receiver_phase, 0, 0);
            receiver_row[5] = delay(count + 4);
            receiver_row[6] = RETURN_M10_INSTRUCTION;
        } else if count == 52 {
            receiver_row[3] = delay_pic(50 + receiver_phase, 0, 0);
            receiver_row[4] = delay_xpic(0, 0, TILE_MUX_EXCHANGE);
            receiver_row[5] = delay(56);
            receiver_row[6] = RETURN_M10_INSTRUCTION;
        } else {
            receiver_row[3] = delay_pic(51 + receiver_phase, 0, 0);
            receiver_row[4] = delay_xpic(count - 53, 0, TILE_MUX_EXCHANGE);
            receiver_row[5] = delay(56);
            receiver_row[6] = RETURN_M10_INSTRUCTION;
        }
        debug!(
            sender_logical,
            receiver_logical, count, "assembled point-to-point exchange"
        );
        Ok(Plan {
            sender: sender_row,
            receiver: receiver_row,
        })
    }

    pub fn multicast(
        &self,
        sender_logical: u16,
        receiver_logical: &[u16],
        count: u32,
        schedule_offset: u32,
    ) -> Result<MulticastPlan, ExchangeError> {
        validate_count(count)?;
        let source_physical = u32::from(self.physical(sender_logical)?);
        let mut used = HashSet::from([sender_logical]);
        if receiver_logical.is_empty()
            || receiver_logical
                .iter()
                .any(|receiver| !used.insert(*receiver) || self.physical(*receiver).is_err())
        {
            return Err(ExchangeError::ReceiverSet);
        }
        let mux_times: Vec<_> = receiver_logical
            .iter()
            .map(|receiver| {
                self.physical(*receiver)
                    .map(|physical| time_to_mux(source_physical, u32::from(physical)))
            })
            .collect::<Result<_, _>>()?;
        let minimum_mux = *mux_times.iter().min().ok_or(ExchangeError::ReceiverSet)?;
        let natural_start = (-minimum_mux).max(0) as u32;
        let start_cycle = natural_start
            .checked_add(schedule_offset)
            .filter(|cycle| *cycle <= 4095)
            .ok_or(ExchangeError::Schedule("multicast start cycle"))?;
        let sender_delay = start_cycle as i32 - 1;

        let mut sender = [0; PLAN_WORDS];
        let mut cursor = 0;
        sender[cursor] = SYNC_SUPERVISOR_INSTRUCTION;
        cursor += 1;
        if sender_delay >= 0 {
            sender[cursor] = delay(sender_delay as u32);
            cursor += 1;
        }
        let send_direction = if receiver_logical.len() == 1 {
            direction(
                source_physical,
                u32::from(self.physical(receiver_logical[0])?),
            )
        } else {
            3
        };
        sender[cursor] = encode_send(count.min(64) - 1, send_direction, 0)?;
        cursor += 1;
        if count > 64 {
            sender[cursor] = send_off(count - 65, send_direction, 0);
            cursor += 1;
        }
        let trailing_delay = 4 - sender_delay - count as i32;
        if trailing_delay >= 0 {
            sender[cursor] = delay(trailing_delay as u32);
            cursor += 1;
        }
        sender[cursor] = RETURN_M10_INSTRUCTION;

        let mut receivers = Vec::with_capacity(receiver_logical.len());
        for (logical, mux_time) in receiver_logical.iter().zip(mux_times) {
            let physical = u32::from(self.physical(*logical)?);
            let receive_cycle = start_cycle as i32 + mux_time;
            if !(0..=4095).contains(&receive_cycle) {
                return Err(ExchangeError::Schedule("multicast receive cycle"));
            }
            let receiver_phase = 2 * (physical >> 6);
            let mut row = [0; PLAN_WORDS];
            row[0] = SYNC_SUPERVISOR_INSTRUCTION;
            row[1] = delay_xpic(receive_cycle as u32, 0, source_physical);
            if count <= 51 {
                // The one-word case still needs this event: without it the
                // tile remains connected to the source after the phase ends.
                row[2] = delay_xpic(count - 1, 0, TILE_MUX_EXCHANGE);
                row[3] = delay_pic(51 - count + receiver_phase, 0, 0) | 0x0001_4000;
                row[4] = delay(count + 4);
                row[5] = RETURN_M10_INSTRUCTION;
            } else if count == 52 {
                row[2] = delay_pic(50 + receiver_phase, 0, 0) | 0x0001_4000;
                row[3] = delay_xpic(0, 0, TILE_MUX_EXCHANGE);
                row[4] = delay(56);
                row[5] = RETURN_M10_INSTRUCTION;
            } else {
                row[2] = delay_pic(51 + receiver_phase, 0, 0) | 0x0001_4000;
                row[3] = delay_xpic(count - 53, 0, TILE_MUX_EXCHANGE);
                row[4] = delay(56);
                row[5] = RETURN_M10_INSTRUCTION;
            }
            receivers.push(row);
        }
        debug!(
            sender_logical,
            receiver_logical = ?receiver_logical,
            count,
            schedule_offset,
            "assembled multicast exchange"
        );
        Ok(MulticastPlan { sender, receivers })
    }
}

/// Selects double-width items in every outgoing instruction in a primitive
/// sender row. Counts and absolute source operands both become 64-bit-item
/// units; use [`patch_sender_address_64`] after selecting this control.
pub fn set_sender_control(row: &mut PlanRow, send_control: u8) -> Result<(), ExchangeError> {
    if !(1..=7).contains(&send_control) {
        return Err(ExchangeError::Schedule("send control"));
    }
    let mut found = false;
    for instruction in row {
        if *instruction & LONG_OPCODE_MASK == SEND_OPCODE
            || is_send_off(*instruction)
            || (is_send_control_pair(*instruction) && *instruction & 3 != 0)
        {
            *instruction = (*instruction & !7) | u32::from(send_control);
            found = true;
        }
    }
    found
        .then_some(())
        .ok_or(ExchangeError::Schedule("sender payload"))
}

/// Removes standalone source-mux writes while retaining their exact event
/// advances. The secondary member of a paired 64-bit receiver uses this when
/// the primary member owns both `INCOMING_MUX` selections.
pub fn replace_xpic_controls_with_delays(program: &mut [u32]) -> Result<(), ExchangeError> {
    for instruction in program {
        if *instruction & OPCODE_MASK == DELAY_XPIC_OPCODE {
            *instruction = delay(instruction_advance(*instruction) - 1);
        } else if is_send_control(*instruction) || is_send_control_pair(*instruction) {
            return Err(ExchangeError::Schedule(
                "paired receiver has fused XPIC control",
            ));
        }
    }
    Ok(())
}

/// Selects the paired XPIC stream for every standalone receive-mux write.
/// Double-width receivers use stream one on the primary tile; stream zero is
/// supplied by the paired incoming-mux register.
pub fn select_paired_xpic_stream(program: &mut [u32]) -> Result<(), ExchangeError> {
    let mut found = false;
    for instruction in program {
        if *instruction & OPCODE_MASK == DELAY_XPIC_OPCODE {
            *instruction |= 1 << 13;
            found = true;
        } else if is_send_control(*instruction) || is_send_control_pair(*instruction) {
            return Err(ExchangeError::Schedule(
                "paired receiver has fused XPIC control",
            ));
        }
    }
    found
        .then_some(())
        .ok_or(ExchangeError::Schedule("paired receiver XPIC control"))
}

/// Replaces the selected physical source in standalone XPIC controls.
pub fn patch_xpic_source(program: &mut [u32], source_physical: u16) -> Result<(), ExchangeError> {
    if u32::from(source_physical) > 0x1fff {
        return Err(ExchangeError::Tile(source_physical));
    }
    let instruction = program
        .iter_mut()
        .find(|instruction| {
            **instruction & OPCODE_MASK == DELAY_XPIC_OPCODE
                && **instruction & 0x1fff != TILE_MUX_EXCHANGE
        })
        .ok_or(ExchangeError::Schedule("receiver source control"))?;
    *instruction = (*instruction & !0x1fff) | u32::from(source_physical);
    Ok(())
}

/// Executable exchange row that reserves a borrowed tile resource without
/// sending or receiving payload data.
pub fn timed_idle_program(cycles: u32) -> Result<Vec<u32>, ExchangeError> {
    let mut words = Vec::new();
    let mut event_cycles = 0;
    append_plain_delay(&mut words, &mut event_cycles, cycles)?;
    words.push(RETURN_M10_INSTRUCTION);
    Ok(words)
}

pub fn c600_logical_to_physical(logical: u16) -> u16 {
    let pair = logical / 2;
    let lane = logical & 1;
    let block = pair / 23;
    let mut row = pair % 23;
    if block & 1 != 0 {
        row = 22 - row;
    }
    let column = (block / 2) * 4 + (block & 1);
    row * 64 + column + lane * 2
}

pub fn patch_sender_address(row: &mut PlanRow, byte_address: u32) -> Result<(), ExchangeError> {
    let instruction = row
        .iter_mut()
        .find(|instruction| **instruction & LONG_OPCODE_MASK == SEND_OPCODE)
        .ok_or(ExchangeError::Address(byte_address))?;
    patch_sender_instruction(instruction, byte_address)
}

/// Replaces the address field of one tile-to-tile SEND instruction.
pub fn patch_sender_instruction(
    instruction: &mut u32,
    byte_address: u32,
) -> Result<(), ExchangeError> {
    if *instruction & LONG_OPCODE_MASK != SEND_OPCODE
        && !(is_send_control_pair(*instruction) && *instruction & 7 != 0)
    {
        return Err(ExchangeError::Address(byte_address));
    }
    let item_shift = if *instruction & 4 != 0 { 3 } else { 2 };
    if byte_address & ((1 << item_shift) - 1) != 0
        || byte_address >> item_shift > SEND_ADDRESS_MASK >> 3
    {
        return Err(ExchangeError::Address(byte_address));
    }
    let item_address = byte_address >> item_shift;
    *instruction = (*instruction & !SEND_ADDRESS_MASK) | ((item_address << 3) & SEND_ADDRESS_MASK);
    Ok(())
}

/// Address-bearing instructions for each outgoing message, in execution
/// order. Each entry is `(word offset, byte offset from the message source)`.
/// SENDPICP restarts the outgoing source stream explicitly after its inline
/// control word, so repeat relocation must patch it as well as the first SEND.
pub fn sender_address_instruction_groups(
    row: &[u32],
) -> Result<Vec<Vec<(usize, u32)>>, ExchangeError> {
    let mut groups = Vec::<Vec<(usize, u32)>>::new();
    let mut sent_words = None;
    let mut cursor = 0;
    while cursor < row.len() {
        let instruction = row[cursor];
        if instruction & LONG_OPCODE_MASK == SEND_OPCODE {
            groups.push(vec![(cursor, 0)]);
            sent_words = Some(instruction_advance(instruction));
        } else if is_send_control_pair(instruction) && instruction & 7 != 0 {
            let sent = sent_words.ok_or(ExchangeError::Schedule(
                "SENDPICP precedes initial outgoing SEND",
            ))?;
            groups
                .last_mut()
                .ok_or(ExchangeError::Schedule("SENDPICP outgoing group"))?
                .push((
                    cursor,
                    sent.checked_mul(if instruction & 4 != 0 { 8 } else { 4 })
                        .ok_or(ExchangeError::Schedule("sender byte offset overflow"))?,
                ));
            sent_words = Some(
                sent.checked_add(instruction_advance(instruction))
                    .ok_or(ExchangeError::Schedule("sender word offset overflow"))?,
            );
        } else if (is_send_off(instruction) || is_send_control(instruction)) && sent_words.is_some()
        {
            sent_words = Some(
                sent_words
                    .unwrap()
                    .checked_add(instruction_advance(instruction))
                    .ok_or(ExchangeError::Schedule("sender word offset overflow"))?,
            );
        }
        cursor += if is_send_control_pair(instruction) {
            2
        } else {
            1
        };
    }
    Ok(groups)
}

/// Removes tile-memory address fields while retaining exchange roles, routes,
/// transfer sizes, and event timing. Rows with the same result can share one
/// executable slot and restore their addresses before invocation.
pub fn normalized_exchange_address_words(row: &[u32]) -> Vec<u32> {
    let mut normalized = row.to_vec();
    let mut cursor = 0;
    while cursor < normalized.len() {
        let instruction = normalized[cursor];
        if is_send_control_pair(instruction) {
            if instruction & 7 != 0 {
                normalized[cursor] &= !SEND_ADDRESS_MASK;
            }
            if instruction & (1 << 27) == 0
                && let Some(payload) = normalized.get_mut(cursor + 1)
            {
                *payload &= !PIC_RECEIVE_ADDRESS_MASK;
            }
            cursor += 2;
            continue;
        }
        normalized[cursor] = if instruction & LONG_OPCODE_MASK == SEND_OPCODE {
            instruction & !SEND_ADDRESS_MASK
        } else if (is_send_control(instruction) && (instruction >> 18) & 3 == 2)
            || (instruction & OPCODE_MASK == DELAY_PIC_OPCODE && instruction & (1 << 18) == 0)
        {
            instruction & !PIC_RECEIVE_ADDRESS_MASK
        } else {
            instruction
        };
        cursor += 1;
    }
    normalized
}

pub fn patch_receiver_address(row: &mut PlanRow, byte_address: u32) -> Result<(), ExchangeError> {
    if byte_address & 3 != 0 || byte_address >> 2 > PIC_RECEIVE_ADDRESS_MASK {
        return Err(ExchangeError::Address(byte_address));
    }
    let word_address = byte_address >> 2;
    let mut cursor = 0;
    while cursor < row.len() {
        let instruction = row[cursor];
        if is_send_control_pair(instruction) {
            if instruction & (1 << 27) == 0 {
                let payload = row
                    .get_mut(cursor + 1)
                    .ok_or(ExchangeError::Schedule("truncated SENDPICP payload"))?;
                *payload = (*payload & !PIC_RECEIVE_ADDRESS_MASK) | word_address;
                return Ok(());
            }
            cursor += 2;
            continue;
        }
        if instruction & OPCODE_MASK == DELAY_PIC_OPCODE && instruction & (1 << 18) == 0 {
            row[cursor] = (instruction & !PIC_RECEIVE_ADDRESS_MASK) | word_address;
            return Ok(());
        }
        cursor += 1;
    }
    Err(ExchangeError::Address(byte_address))
}

/// Delays every timed event in a plan row while preserving route-relative timing.
pub fn offset_plan(row: &mut PlanRow, cycles: u32) -> Result<(), ExchangeError> {
    if cycles == 0 {
        return Ok(());
    }
    if row[0] != SYNC_SUPERVISOR_INSTRUCTION {
        return Err(ExchangeError::Schedule("plan offset entry"));
    }
    let end = row
        .iter()
        .position(|instruction| *instruction == RETURN_M10_INSTRUCTION)
        .ok_or(ExchangeError::Schedule("plan offset return"))?;
    let available = row.len().saturating_sub(end + 1);
    let delay_count = usize::try_from(cycles.div_ceil(MAX_PLAN_OFFSET_CYCLES))
        .map_err(|_| ExchangeError::Schedule("plan offset instruction count"))?;
    if delay_count > available {
        let maximum = u32::try_from(available)
            .unwrap_or(u32::MAX)
            .saturating_mul(MAX_PLAN_OFFSET_CYCLES);
        return Err(ExchangeError::PlanOffsetRange { cycles, maximum });
    }
    if delay_count == 0 {
        return Err(ExchangeError::Schedule("plan offset instruction capacity"));
    }
    row.copy_within(1..=end, 1 + delay_count);
    let mut remaining = cycles;
    for instruction in &mut row[1..=delay_count] {
        let chunk = remaining.min(MAX_PLAN_OFFSET_CYCLES);
        *instruction = delay(chunk - 1);
        remaining -= chunk;
    }
    Ok(())
}

pub fn finalize_point_receiver(
    row: &PlanRow,
    source_physical: u16,
) -> Result<PlanRow, ExchangeError> {
    let patch_index = row[0] as usize;
    if patch_index >= PLAN_WORDS - 1 || u32::from(source_physical) > 0x1fff {
        return Err(ExchangeError::Schedule("point receiver patch index"));
    }
    let mut executable = [0; PLAN_WORDS];
    executable[..PLAN_WORDS - 1].copy_from_slice(&row[1..]);
    executable[patch_index] = (executable[patch_index] & !0x1fff) | u32::from(source_physical);
    Ok(executable)
}

fn validate_count(count: u32) -> Result<(), ExchangeError> {
    if (1..=MAX_TRANSFER_WORDS).contains(&count) {
        Ok(())
    } else {
        Err(ExchangeError::Count(count))
    }
}

fn route_displacement(source: u32, destination: u32) -> i32 {
    let source_raw = ((source >> 2) & 15) as i32;
    let destination_raw = ((destination >> 2) & 15) as i32;
    let source_column = if source_raw > 7 {
        source_raw ^ 15
    } else {
        source_raw
    };
    let destination_column = if destination_raw > 7 {
        destination_raw ^ 15
    } else {
        destination_raw
    };
    let source_mux = source_column + ((source_raw >> 3) ^ (source & 1) as i32);
    let base = (destination_column - source_mux) * 6;
    let destination_lane = destination & 3;
    let destination_half = destination_raw >> 3;
    if destination_lane > 1 {
        base + if destination_half == (destination & 1) as i32 {
            2
        } else {
            4
        }
    } else {
        base + if destination_half == destination_lane as i32 {
            1
        } else {
            5
        }
    }
}

fn direction(source: u32, destination: u32) -> u32 {
    if route_displacement(source, destination) < 1 {
        2
    } else {
        1
    }
}

fn time_to_mux(source: u32, destination: u32) -> i32 {
    let source_raw = ((source >> 2) & 15) as i32;
    let destination_raw = ((destination >> 2) & 15) as i32;
    let source_low = ((source >> 2) & 7) as i32;
    let displacement = route_displacement(source, destination);
    let source_edge = if source_raw > 7 {
        (source_raw * 4) ^ 60
    } else {
        source_raw * 4
    };
    let destination_edge = if destination_raw > 7 {
        (destination_raw * 4) ^ 60
    } else {
        destination_raw * 4
    };
    let local = ((source >> 2) & 8) as i32 | ((source >> 3) & 3) as i32;
    let crossing = local - destination_raw + ((source_low >> 1) ^ 3);
    let same_region = (source ^ destination) & 0x20 == 0;
    let turn = if same_region {
        source_low + 1
    } else {
        16 - source_low
    };
    let group_delta = (((source >> 6) & 31) as i32 - ((destination >> 6) & 31) as i32) * 2;
    crossing + source_edge + turn - destination_edge + group_delta + displacement.abs() - 34
}

/// Route timing when a receiving tile borrows its physical partner's lane.
/// It is monotonic in the opposite direction to the SDK's
/// `ColossusBNET_TTDBL`, but differences between receiver pairs are in event
/// cycles and drive the same paired-source schedule.
fn paired_time_to_mux(source: u32, destination: u32) -> i32 {
    time_to_mux(source, destination).max(time_to_mux(source, destination ^ 2))
}

pub const fn encode_exchange_delay(cycles: u32) -> u32 {
    0x40a0_0000 | (cycles & 0x7ffff)
}

pub const fn encode_exchange_delay_pic(a: u32, b: u32, c: u32) -> u32 {
    0x6000_0000 | ((a << 19) & 0x03f8_0000) | ((b << 18) & 0x0004_0000) | (c & 0x3ffff)
}

pub const fn encode_exchange_delay_xpic(a: u32, b: u32, c: u32) -> u32 {
    0x6400_0000 | ((a << 14) & 0x03ff_c000) | ((b << 13) & 0x0000_2000) | (c & 0x1fff)
}

const fn delay(cycles: u32) -> u32 {
    encode_exchange_delay(cycles)
}

const fn delay_pic(a: u32, b: u32, c: u32) -> u32 {
    encode_exchange_delay_pic(a, b, c)
}

const fn delay_xpic(a: u32, b: u32, c: u32) -> u32 {
    encode_exchange_delay_xpic(a, b, c)
}

pub fn encode_send(
    count_minus_one: u32,
    direction: u32,
    base_word: u32,
) -> Result<u32, ExchangeError> {
    if count_minus_one > 63 || direction > 7 || base_word > 0x3_ffff {
        return Err(ExchangeError::Schedule("send instruction operand"));
    }
    Ok(0x7800_0000
        | ((count_minus_one << 21) & 0x07e0_0000)
        | ((base_word << 3) & 0x001f_fff8)
        | direction)
}

fn send_off(count_minus_one: u32, direction: u32, base_word: u32) -> u32 {
    0x7000_0000
        | ((count_minus_one << 21) & 0x07e0_0000)
        | (((count_minus_one >> 6) << 14) & 0x000f_c000)
        | ((base_word << 3) & 0x0000_3ff8)
        | (direction & 7)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_instruction_encoders_preserve_operands_and_reject_overflow() {
        let setzi = encode_setzi_m(15, (1 << 20) - 1).unwrap();
        assert_eq!((setzi >> 20) & 0xf, 15);
        assert_eq!(setzi & ((1 << 20) - 1), (1 << 20) - 1);

        let put = encode_put_special_m(0xa6, 8).unwrap();
        assert_eq!((put >> 20) & 0xf, 8);
        assert_eq!(put & 0xff, 0xa6);
        assert_eq!((encode_br_m(10).unwrap() >> 20) & 0xf, 10);

        let call = encode_call_m_immediate(10, 0x4c100).unwrap();
        assert_eq!((call >> 20) & 0xf, 10);
        assert_eq!((call & 0x7ffff) << 2, 0x4c100);

        assert!(encode_setzi_m(16, 0).is_err());
        assert!(encode_setzi_m(0, 1 << 20).is_err());
        assert!(encode_put_special_m(0, 16).is_err());
        assert!(encode_br_m(16).is_err());
        assert!(encode_call_m_immediate(16, 0).is_err());
        assert!(encode_call_m_immediate(0, 2).is_err());
        assert!(encode_call_m_immediate(0, 1 << 21).is_err());

        assert_eq!(encode_ld32_m_immediate(8, 11, 15, 1).unwrap(), 0x01b8_f001);
        assert_eq!(encode_st32_m_immediate(2, 11, 15, 0).unwrap(), 0x4fb2_f000);
        assert_eq!(encode_add_m_immediate(11, 11, -32).unwrap(), 0x22bb_ffe0);
        assert_eq!(encode_and_m_immediate(0, 8, 1).unwrap(), 0x4280_0001);
        assert_eq!(encode_shl_m_immediate(10, 7, 2).unwrap(), 0x427a_a002);
        assert!(encode_shl_m_immediate(0, 0, 1 << 12).is_err());
        assert_eq!(encode_brz_m_immediate(0, 0x4c100).unwrap(), 0x1301_3040);
    }

    #[test]
    fn normalizes_all_sender_address_fields() {
        let row = [
            SYNC_SUPERVISOR_INSTRUCTION,
            encode_send(1, 3, 0x1a048).unwrap(),
            SEND_PICP_OPCODE | (7 << 21) | (0x1a04a << 3) | 3,
            0x1901_5000,
            RETURN_M10_INSTRUCTION,
        ];
        let normalized = normalized_exchange_address_words(&row);
        assert_eq!(normalized[1] ^ row[1], row[1] & SEND_ADDRESS_MASK);
        assert_eq!(normalized[2] ^ row[2], row[2] & SEND_ADDRESS_MASK);
        assert_eq!(normalized[0], row[0]);
        assert_eq!(normalized[4], row[4]);
    }

    #[test]
    fn paired_send_matches_sdk_full_duplex_oracle() {
        let events = [
            ReceiveEvent {
                cycles: 189,
                instruction: delay_xpic(0, 0, TILE_MUX_EXCHANGE),
                kind: ReceiveEventKind::OrdinaryNeutral,
            },
            ReceiveEvent {
                cycles: 189,
                instruction: delay_pic(0, 0, 0x15000),
                kind: ReceiveEventKind::Pointer,
            },
        ];
        assert_eq!(
            encode_send_control_pair(42, 0x14021, 1, &events).unwrap(),
            (0xf54a_0109, 0x1901_5000)
        );
    }

    #[test]
    fn paired_multicast64_matches_near_reverse_and_far_sdk_rows() {
        let topology = Topology::c600();
        let cases = [(0, [2, 3], 2), (2, [0, 1], 0), (0, [46, 47], 47)];
        for (source, receivers, early) in cases {
            let mut plan = topology.paired_multicast(source, &receivers, 64).unwrap();
            patch_sender_address(&mut plan.sender, 0x50000).unwrap();
            for row in &mut plan.receivers {
                patch_receiver_address(row, 0x60000).unwrap();
            }
            if source == 0 && receivers == [2, 3] {
                assert_eq!(
                    plan.sender,
                    [
                        SYNC_SUPERVISOR_INSTRUCTION,
                        delay(30),
                        encode_send(63, 5, 0x50000 >> 3).unwrap(),
                        RETURN_M10_INSTRUCTION,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                );
                assert_eq!(
                    plan.receivers[0],
                    [
                        SYNC_SUPERVISOR_INSTRUCTION,
                        delay_xpic(0, 1, 0),
                        delay_pic(52, 1, 1),
                        delay_pic(1, 0, 0x60000 >> 2),
                        delay_xpic(8, 1, TILE_MUX_EXCHANGE),
                        delay_pic(52, 1, 0),
                        delay(6),
                        RETURN_M10_INSTRUCTION,
                        0,
                    ]
                );
                assert_eq!(
                    plan.receivers[1],
                    [
                        SYNC_SUPERVISOR_INSTRUCTION,
                        delay_pic(53, 1, 1),
                        delay_pic(1, 0, 0x60000 >> 2),
                        delay_pic(61, 1, 0),
                        delay(6),
                        RETURN_M10_INSTRUCTION,
                        0,
                        0,
                        0,
                    ]
                );
            }
            assert!(topology.paired_receiver_is_early(early, source).unwrap());
            assert!(
                !topology
                    .paired_receiver_is_early(early ^ 1, source)
                    .unwrap()
            );

            let mut builder = PhaseProgramBuilder::new(1472);
            let source_pair = topology.paired_logical(source).unwrap();
            let offset = builder
                .earliest_transfer_offset(source, &[source_pair], &receivers, &plan, 64, 0)
                .unwrap();
            builder
                .append_transfer_at(source, &[source_pair], &receivers, &plan, offset, 64)
                .unwrap();
            let programs = builder.finish().unwrap();
            assert!(programs.programs[usize::from(source)].is_some());
            assert!(programs.programs[usize::from(source_pair)].is_some());
            assert!(
                receivers
                    .iter()
                    .all(|receiver| programs.programs[usize::from(*receiver)].is_some())
            );
        }
    }

    #[test]
    fn randomized_paired_multicasts_preserve_route_relative_timing() {
        let topology = Topology::c600();
        let tile_count = topology.tile_count() as u16;
        let mut random = fastrand::Rng::with_seed(0x7061_6972_6564_3634);
        let all_pairs = (0..tile_count)
            .filter_map(|tile| {
                let paired = topology.paired_logical(tile).ok()?;
                (tile < paired).then_some([tile, paired])
            })
            .collect::<Vec<_>>();

        for prefer_positive_routes in [false, true] {
            for _ in 0..64 {
                let source = random.u16(0..tile_count);
                let source_pair = topology.paired_logical(source).unwrap();
                let source_physical = u32::from(topology.physical(source).unwrap());
                let mut candidates = all_pairs
                    .iter()
                    .copied()
                    .filter(|pair| !pair.contains(&source) && !pair.contains(&source_pair))
                    .filter(|pair| {
                        let receiver = u32::from(topology.physical(pair[0]).unwrap());
                        let route_time = paired_time_to_mux(source_physical, receiver);
                        (route_time > 1) == prefer_positive_routes
                    })
                    .collect::<Vec<_>>();
                random.shuffle(&mut candidates);
                if candidates.is_empty() {
                    continue;
                }
                let pair_count = random.usize(1..=candidates.len().min(12));
                let receivers = candidates[..pair_count]
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>();
                let count = 512;
                let plan = topology
                    .paired_multicast(source, &receivers, count)
                    .unwrap();
                let minimum_route_time = receivers
                    .iter()
                    .map(|&receiver| {
                        paired_time_to_mux(
                            source_physical,
                            u32::from(topology.physical(receiver).unwrap()),
                        )
                    })
                    .min()
                    .unwrap();
                let epoch_shift = (1 - minimum_route_time).max(0);

                for (&receiver, row) in receivers.iter().zip(&plan.receivers) {
                    let receiver_physical = u32::from(topology.physical(receiver).unwrap());
                    let source_event = u32::try_from(
                        paired_time_to_mux(source_physical, receiver_physical) + epoch_shift,
                    )
                    .unwrap();
                    let format_start = source_event + 51 + 2 * (receiver_physical >> 6);
                    let timing = receive_row_timing(row, 0).unwrap();
                    assert_eq!(timing.mode, ReceiveMode::Paired64);
                    assert_eq!(timing.format_start_cycles, Some(format_start));
                    assert_eq!(timing.pointer_cycles, Some(format_start + 2));
                    assert_eq!(timing.format_end_cycles, Some(format_start + count));
                    if topology.paired_receiver_is_early(receiver, source).unwrap() {
                        assert_eq!(timing.source_cycles, Some(source_event));
                        assert_eq!(timing.neutral_cycles, Some(source_event + count));
                    } else {
                        assert_eq!(timing.source_cycles, None);
                        assert_eq!(timing.neutral_cycles, None);
                    }
                }
            }
        }
    }

    #[test]
    fn randomized_coincident_paired_controls_are_rejected() {
        let topology = Topology::c600();
        let tile_count = topology.tile_count() as u16;
        let mut random = fastrand::Rng::with_seed(0x636f_696e_6369_6465);
        let mut checked = 0;
        for _ in 0..10_000 {
            let source = random.u16(0..tile_count);
            let receiver = random.u16(0..tile_count);
            let paired = topology.paired_logical(receiver).unwrap();
            let source_pair = topology.paired_logical(source).unwrap();
            if receiver > paired
                || [source, source_pair]
                    .iter()
                    .any(|tile| *tile == receiver || *tile == paired)
            {
                continue;
            }
            let receivers = [receiver, paired];
            let source_physical = u32::from(topology.physical(source).unwrap());
            let receiver = receivers
                .iter()
                .copied()
                .find(|receiver| {
                    topology
                        .paired_receiver_is_early(*receiver, source)
                        .unwrap()
                })
                .unwrap();
            let receiver_physical = u32::from(topology.physical(receiver).unwrap());
            let source_event = u32::try_from(
                paired_time_to_mux(source_physical, receiver_physical)
                    + (1 - paired_time_to_mux(source_physical, receiver_physical)).max(0),
            )
            .unwrap();
            let format_start = source_event + 51 + 2 * (receiver_physical >> 6);
            let count = format_start - source_event;
            if count < 64 {
                continue;
            }
            assert!(matches!(
                topology.paired_multicast(source, &receivers, count),
                Err(ExchangeError::Schedule(
                    "paired receive coincident controls"
                ))
            ));
            checked += 1;
            if checked == 64 {
                break;
            }
        }
        assert_eq!(checked, 64);
    }

    #[test]
    fn c600_mapping_is_a_permutation() {
        let topology = Topology::c600();
        let physical: HashSet<_> = (0..topology.tile_count() as u16)
            .map(|logical| topology.physical(logical).unwrap())
            .collect();
        assert_eq!(physical.len(), 1472);
        assert_eq!(topology.physical(46).unwrap(), 1409);
        assert!(physical.iter().all(|tile| *tile < 1472));
    }

    #[test]
    fn point_to_point_matches_cpp_oracle_vectors() {
        let topology = Topology::c600();
        let cases = [
            (
                0,
                1286,
                3,
                [
                    0x41800003, 0x40a00086, 0x78400001, 0x43a00000, 0, 0, 0, 0, 0,
                ],
                [
                    1, 0x41800003, 0x641c0000, 0x64008640, 0x61800000, 0x40a00007, 0x43a00000, 0, 0,
                ],
            ),
            (
                736,
                100,
                52,
                [
                    0x41800003, 0x40a0003e, 0x7e600002, 0x43a00000, 0, 0, 0, 0, 0,
                ],
                [
                    1, 0x41800003, 0x641c0000, 0x61d00000, 0x64000640, 0x40a00038, 0x43a00000, 0, 0,
                ],
            ),
            (
                1286,
                1471,
                65,
                [
                    0x41800003, 0x40a0007d, 0x7fe00002, 0x70000002, 0x43a00000, 0, 0, 0, 0,
                ],
                [
                    1, 0x41800003, 0x641c0000, 0x61980000, 0x64030640, 0x40a00038, 0x43a00000, 0, 0,
                ],
            ),
            (
                32,
                0,
                4148,
                [
                    0x41800003, 0x40a0006c, 0x7fe00001, 0x766fc001, 0x43a00000, 0, 0, 0, 0,
                ],
                [
                    1, 0x41800003, 0x641c0000, 0x61980000, 0x67ffc640, 0x40a00038, 0x43a00000, 0, 0,
                ],
            ),
        ];
        for (sender, receiver, count, expected_sender, expected_receiver) in cases {
            let plan = topology.point_to_point(sender, receiver, count).unwrap();
            assert_eq!(&plan.sender[..expected_sender.len()], &expected_sender);
            assert_eq!(
                &plan.receiver[..expected_receiver.len()],
                &expected_receiver
            );
            assert!(
                plan.sender[expected_sender.len()..]
                    .iter()
                    .all(|word| *word == 0)
            );
            assert!(
                plan.receiver[expected_receiver.len()..]
                    .iter()
                    .all(|word| *word == 0)
            );
        }
    }

    #[test]
    fn multicast_matches_cpp_oracle_vectors() {
        let topology = Topology::c600();
        let plan = topology.multicast(0, &[274, 1286], 3, 0).unwrap();
        assert_eq!(plan.sender[0], 0x41800003);
        assert_eq!(plan.sender[1], 0x40a00016);
        assert_eq!(plan.sender[2], 0x78400003);
        assert_eq!(plan.receivers[0][1], 0x64000000);
        assert_eq!(plan.receivers[0][2], 0x64008640);
        assert_eq!(plan.receivers[0][3], 0x61814000);

        let boundary = topology.multicast(736, &[100, 900], 52, 0).unwrap();
        assert_eq!(boundary.receivers[0][2], 0x61d14000);
        assert_eq!(boundary.receivers[0][3], 0x64000640);
    }

    #[test]
    fn single_receiver_uses_the_directional_route_for_every_payload_instruction() {
        let topology = Topology::c600();
        let mut random = fastrand::Rng::with_seed(0x0072_6f75_7469_6e67);
        for _ in 0..128 {
            let source = random.u16(0..topology.tile_count() as u16);
            let mut destination = random.u16(0..topology.tile_count() as u16);
            while destination == source {
                destination = random.u16(0..topology.tile_count() as u16);
            }
            let words = random.u32(65..=MAX_TRANSFER_WORDS);
            let expected_direction = direction(
                u32::from(topology.physical(source).unwrap()),
                u32::from(topology.physical(destination).unwrap()),
            );
            let plan = topology
                .multicast(source, &[destination], words, 0)
                .unwrap();
            let payload_directions = plan
                .sender
                .iter()
                .filter(|instruction| {
                    matches!(
                        **instruction & LONG_OPCODE_MASK,
                        SEND_OPCODE | SEND_OFF_OPCODE
                    )
                })
                .map(|instruction| instruction & 7)
                .collect::<Vec<_>>();
            assert_eq!(payload_directions, [expected_direction, expected_direction]);
        }
    }

    #[test]
    fn randomized_internal_receives_leave_the_neutral_mux_selected() {
        let topology = Topology::c600();
        let mut random = fastrand::Rng::with_seed(0x006e_6575_7472_616c);
        for _ in 0..128 {
            let source = random.u16(0..topology.tile_count() as u16);
            let receiver_count = random.usize(1..=4);
            let mut receivers = Vec::with_capacity(receiver_count);
            while receivers.len() < receiver_count {
                let receiver = random.u16(0..topology.tile_count() as u16);
                if receiver != source && !receivers.contains(&receiver) {
                    receivers.push(receiver);
                }
            }
            let plan = topology
                .multicast(source, &receivers, random.u32(1..=MAX_TRANSFER_WORDS), 0)
                .unwrap();
            for receiver in plan.receivers {
                assert!(receiver.iter().skip(2).any(|instruction| {
                    instruction & OPCODE_MASK == DELAY_XPIC_OPCODE
                        && instruction & 0x1fff == TILE_MUX_EXCHANGE
                }));
            }
        }
    }

    #[test]
    fn randomized_receiver_streams_schedule_source_and_pointer_cutovers() {
        let topology = Topology::c600();
        let mut random = fastrand::Rng::with_seed(0x6d75_785f_6375_746f);
        for _ in 0..128 {
            let receiver = random.u16(0..topology.tile_count() as u16);
            let transfer_count = random.usize(2..=12);
            let words = random.u32(53..=512);
            let mut address = 0x50000 + random.u32(0..=0x2000) * 4;
            let mut pointer_writes = 1;
            let mut builder = TileProgramSchedule::default();
            for index in 0..transfer_count {
                let source = loop {
                    let candidate = random.u16(0..topology.tile_count() as u16);
                    if candidate != receiver {
                        break candidate;
                    }
                };
                let has_address_gap = index != 0 && random.bool();
                if has_address_gap {
                    address += random.u32(1..=16) * 4;
                    pointer_writes += 1;
                }
                let mut row = topology
                    .multicast(source, &[receiver], words, 0)
                    .unwrap()
                    .receivers[0];
                patch_receiver_address(&mut row, address).unwrap();
                let offset = builder.earliest_receiver_offset(&row, words, 0).unwrap();
                builder.append_receiver_at(&row, offset, words).unwrap();
                address += words * 4;
            }
            let expected_cycles = builder.event_cycles();
            let program = builder.finish().unwrap();
            assert_eq!(plan_event_cycles(&program).unwrap(), expected_cycles);
            assert_eq!(
                program
                    .iter()
                    .filter(|instruction| is_neutral_mux_teardown(**instruction))
                    .count(),
                1
            );
            assert_eq!(
                program
                    .iter()
                    .filter(|instruction| **instruction & OPCODE_MASK == DELAY_XPIC_OPCODE)
                    .count(),
                transfer_count + 1
            );
            assert_eq!(
                program
                    .iter()
                    .filter(|instruction| **instruction & OPCODE_MASK == DELAY_PIC_OPCODE)
                    .count(),
                pointer_writes
            );
        }
    }

    #[test]
    fn randomized_mixed_role_programs_fuse_receive_controls_into_sends() {
        let topology = Topology::c600();
        let mut random = fastrand::Rng::with_seed(0x0073_656e_6470_6963);
        let mut fused_programs = 0;
        for _ in 0..256 {
            let tile = random.u16(0..topology.tile_count() as u16);
            let incoming_source = loop {
                let candidate = random.u16(0..topology.tile_count() as u16);
                if candidate != tile {
                    break candidate;
                }
            };
            let outgoing_destination = loop {
                let candidate = random.u16(0..topology.tile_count() as u16);
                if candidate != tile {
                    break candidate;
                }
            };
            let words = random.u32(2..=256);
            let mut incoming = topology
                .multicast(incoming_source, &[tile], words, 0)
                .unwrap()
                .receivers[0];
            patch_receiver_address(&mut incoming, 0x50000 + random.u32(0..0x1000) * 4).unwrap();
            let outgoing = topology
                .multicast(tile, &[outgoing_destination], words, 0)
                .unwrap()
                .sender;

            let mut builder = TileProgramSchedule::default();
            builder.append_receiver_at(&incoming, 0, words).unwrap();
            let sender_offset = builder.earliest_sender_offset(&outgoing, 0).unwrap();
            builder.append_sender_at(&outgoing, sender_offset).unwrap();
            let expected_horizon = builder.event_cycles();
            let program = builder.finish().unwrap();
            assert_eq!(plan_event_cycles(&program).unwrap(), expected_horizon);
            let outgoing_timing = sender_row_timing(&outgoing, sender_offset).unwrap();

            let base = receive_row_timing(&incoming, 0).unwrap();
            let expected = scheduled_receive_window(&base, 0, words, None)
                .unwrap()
                .events
                .into_iter()
                .map(|event| (event.cycles, receive_control_signature(event.instruction)))
                .collect::<Vec<_>>();
            let mut actual = Vec::new();
            let mut cycles = 0;
            let mut sent_words = 0;
            let mut cursor = 0;
            while cursor < program.len() {
                let instruction = program[cursor];
                let before = cycles;
                let advance = instruction_advance(instruction);
                cycles += advance;
                if instruction & OPCODE_MASK == DELAY_PIC_OPCODE
                    || instruction & OPCODE_MASK == DELAY_XPIC_OPCODE
                {
                    actual.push((cycles, receive_control_signature(instruction)));
                } else if is_send_control(instruction) {
                    fused_programs += 1;
                    actual.push((before + 1, send_control_signature(instruction)));
                    if before >= outgoing_timing.start_cycles && before < outgoing_timing.end_cycles
                    {
                        sent_words += advance;
                    }
                } else if is_send_control_pair(instruction) {
                    fused_programs += 1;
                    let payload = program[cursor + 1];
                    let pointer = payload & PIC_RECEIVE_ADDRESS_MASK;
                    let source = payload >> 18;
                    actual.push((before + 1, (ReceiveEventKind::Pointer, pointer)));
                    actual.push((
                        before + 1,
                        (
                            if source == TILE_MUX_EXCHANGE {
                                ReceiveEventKind::OrdinaryNeutral
                            } else {
                                ReceiveEventKind::OrdinarySource
                            },
                            source,
                        ),
                    ));
                    if before >= outgoing_timing.start_cycles && before < outgoing_timing.end_cycles
                    {
                        sent_words += advance;
                    }
                    cursor += 1;
                } else if is_payload_send(instruction) {
                    sent_words += advance;
                }
                cursor += 1;
            }
            assert_eq!(actual, expected);
            assert_eq!(sent_words, words);
            assert_eq!(
                program
                    .iter()
                    .filter(|instruction| **instruction & LONG_OPCODE_MASK == SEND_OPCODE)
                    .count(),
                1
            );
        }
        assert!(fused_programs > 0);
    }

    fn receive_control_signature(instruction: u32) -> (ReceiveEventKind, u32) {
        if instruction & OPCODE_MASK == DELAY_PIC_OPCODE {
            (
                ReceiveEventKind::Pointer,
                instruction & PIC_RECEIVE_ADDRESS_MASK,
            )
        } else {
            let operand = instruction & 0x1fff;
            (
                if operand == TILE_MUX_EXCHANGE {
                    ReceiveEventKind::OrdinaryNeutral
                } else {
                    ReceiveEventKind::OrdinarySource
                },
                operand,
            )
        }
    }

    fn send_control_signature(instruction: u32) -> (ReceiveEventKind, u32) {
        let selector = (instruction >> 18) & 3;
        let operand = if selector >= 2 {
            instruction & PIC_RECEIVE_ADDRESS_MASK
        } else {
            instruction & 0x1fff
        };
        (
            if selector >= 2 {
                ReceiveEventKind::Pointer
            } else if operand == TILE_MUX_EXCHANGE {
                ReceiveEventKind::OrdinaryNeutral
            } else {
                ReceiveEventKind::OrdinarySource
            },
            operand,
        )
    }

    #[test]
    fn event_horizon_tracks_transfer_size_and_route() {
        let topology = Topology::c600();
        let short = topology.multicast(0, &[736, 1286], 1, 0).unwrap();
        let long = topology.multicast(0, &[736, 1286], 1024, 0).unwrap();
        let horizon = |plan: &MulticastPlan| {
            std::iter::once(&plan.sender)
                .chain(plan.receivers.iter())
                .map(|row| plan_event_cycles(row).unwrap())
                .max()
                .unwrap()
        };

        assert!(horizon(&short) > 0);
        assert!(horizon(&long) > horizon(&short));
        assert_ne!(
            plan_event_cycles(&short.receivers[0]).unwrap(),
            plan_event_cycles(&short.receivers[1]).unwrap()
        );
    }

    #[test]
    fn encoder_places_receive_then_send_on_one_event_timeline() {
        let topology = Topology::c600();
        let first = topology.multicast(0, &[736], 64, 0).unwrap();
        let first_horizon = std::iter::once(&first.sender)
            .chain(first.receivers.iter())
            .map(|row| plan_event_cycles(row).unwrap())
            .max()
            .unwrap();
        let second = topology
            .multicast(736, &[1286], 64, first_horizon + 1)
            .unwrap();
        let horizon = std::iter::once(&second.sender)
            .chain(second.receivers.iter())
            .map(|row| plan_event_cycles(row).unwrap())
            .max()
            .unwrap();

        let mut relay = TileProgramSchedule::default();
        relay
            .append_receiver_at(&first.receivers[0], 0, 64)
            .unwrap();
        let offset = relay.earliest_sender_offset(&second.sender, 0).unwrap();
        relay.append_sender_at(&second.sender, offset).unwrap();
        let relay_horizon = relay.event_cycles();
        let relay = relay.finish().unwrap();

        assert_eq!(relay.last(), Some(&RETURN_M10_INSTRUCTION));
        assert_eq!(
            relay
                .iter()
                .filter(|instruction| **instruction == SYNC_SUPERVISOR_INSTRUCTION)
                .count(),
            0
        );
        assert_eq!(plan_event_cycles(&relay).unwrap(), relay_horizon);
        assert!(relay_horizon <= horizon);
    }

    #[test]
    fn validates_limits_and_patches_addresses() {
        let topology = Topology::c600();
        assert_eq!(
            topology.point_to_point(0, 1, 0),
            Err(ExchangeError::Count(0))
        );
        assert_eq!(
            topology.multicast(0, &[1, 1], 1, 0),
            Err(ExchangeError::ReceiverSet)
        );
        let mut plan = topology.multicast(0, &[274], 65, 0).unwrap();
        patch_sender_address(&mut plan.sender, 0x52040).unwrap();
        patch_receiver_address(&mut plan.receivers[0], 0x53080).unwrap();
        assert_eq!(
            plan.sender[2] & 0x001f_fff8,
            ((0x52040 >> 2) << 3) & 0x001f_fff8
        );
        assert_eq!(plan.receivers[0][2] & 0x3ffff, 0x53080 >> 2);

        patch_receiver_address(&mut plan.receivers[0], 0x8f000).unwrap();
        assert_eq!(plan.receivers[0][2] & 0x3ffff, 0x8f000 >> 2);
        assert_eq!(
            patch_receiver_address(&mut plan.receivers[0], 0x10_0000),
            Err(ExchangeError::Address(0x10_0000))
        );
    }

    #[test]
    fn encodes_supervisor_send_fields() {
        assert_eq!(encode_send(1, 3, 82_041).unwrap(), 0x782a_03cb);
        assert_eq!(encode_send(1, 3, 82_043).unwrap(), 0x782a_03db);
        assert!(encode_send(64, 3, 0).is_err());
        assert!(encode_send(1, 8, 0).is_err());
        assert!(encode_send(1, 3, 0x4_0000).is_err());
    }

    #[test]
    fn host_packets_match_recovered_sdk_vectors() {
        assert_eq!(
            tile_to_host_packet(0, 0x40, 64).unwrap(),
            HostPacketHeader {
                word0: 0xa000_0000,
                word1: 0x0000_0011,
            }
        );
        assert_eq!(
            host_to_tile_packet(0, 0x50120, 0x40, 64).unwrap(),
            HostPacketHeader {
                word0: 0xec00_0209,
                word1: 0x0000_0011,
            }
        );
        assert_eq!(
            zero_byte_read_packet(2, 0x50180).unwrap(),
            HostPacketHeader {
                word0: 0xcc01_020c,
                word1: 0,
            }
        );
        assert_eq!(
            tile_to_host_packet(1409, 0x40, 64).unwrap(),
            HostPacketHeader {
                word0: 0xa2c0_8000,
                word1: 0x8000_0011,
            }
        );
    }

    #[test]
    fn host_packets_validate_both_size_classes() {
        assert!(tile_to_host_packet(0, 4, 4).is_ok());
        assert!(tile_to_host_packet(0, 0x400, 1024).is_ok());
        assert!(tile_to_host_packet(0, 2, 4).is_err());
        assert!(tile_to_host_packet(0, 0, 0).is_err());
        assert!(tile_to_host_packet(0, 0, 1028).is_err());
        assert!(host_to_tile_packet(0, 0x50124, 0x40, 64).is_err());
        assert!(host_to_tile_packet(0, 0x54000, 0x40, 64).is_err());
        assert!(tile_to_host_packet(0x1000, 0, 4).is_err());
    }

    #[test]
    fn host_transfer_planner_covers_unaligned_and_large_ranges() {
        let d2h = plan_tile_to_host(2, 0x60004, 4, 2200).unwrap();
        assert_eq!(d2h.first().unwrap().host_offset, 4);
        assert_eq!(d2h.iter().map(|chunk| chunk.bytes).sum::<u32>(), 2200);
        assert!(d2h.iter().all(|chunk| chunk.bytes <= 1024));
        assert!(
            d2h.windows(2)
                .all(|pair| pair[0].tile_address + pair[0].bytes == pair[1].tile_address)
        );
        assert!(d2h.iter().all(|chunk| {
            chunk.host_offset / HOST_PAGE_BYTES
                == (chunk.host_offset + chunk.bytes - 1) / HOST_PAGE_BYTES
        }));

        let paged = plan_tile_to_host(0, 0x60000, 64, 4096).unwrap();
        assert_eq!(paged.iter().map(|chunk| chunk.bytes).sum::<u32>(), 4096);
        assert!(paged.iter().all(|chunk| {
            chunk.host_offset / HOST_PAGE_BYTES
                == (chunk.host_offset + chunk.bytes - 1) / HOST_PAGE_BYTES
        }));

        let h2d = plan_host_to_tile(1409, 0x50000, 4, 100).unwrap();
        assert_eq!(
            h2d.iter().map(|chunk| chunk.bytes).collect::<Vec<_>>(),
            [32, 32, 36]
        );
        assert_eq!(h2d.last().unwrap().tile_address + 36, 0x50064);
        assert!(plan_host_to_tile(0, 0x50004, 0, 4).is_err());
    }

    #[test]
    fn plan_offsets_extend_beyond_route_timing_fields() {
        let topology = Topology::c600();
        let mut plan = topology.multicast(0, &[1, 2], 4096, 0).unwrap();
        let sender_cycles = plan_event_cycles(&plan.sender).unwrap();
        let receiver_cycles = plan_event_cycles(&plan.receivers[0]).unwrap();
        let offset = MAX_PLAN_OFFSET_CYCLES + 70;

        offset_plan(&mut plan.sender, offset).unwrap();
        offset_plan(&mut plan.receivers[0], offset).unwrap();

        assert_eq!(
            plan_event_cycles(&plan.sender).unwrap(),
            sender_cycles + offset
        );
        assert_eq!(
            plan_event_cycles(&plan.receivers[0]).unwrap(),
            receiver_cycles + offset
        );
    }

    #[test]
    fn tile_to_host_target_preserves_packet_and_payload_addresses() {
        let plan =
            assemble_tile_to_host_target_program(2, 0x50120, 0x40, 64, 0x50160, 0x501a0).unwrap();
        assert_eq!(
            &plan.packet_words[..2],
            &host_packet_words(tile_to_host_packet(2, 0x40, 64).unwrap())
        );
        assert_eq!(plan.packet_words.len(), 4);
        let sends = plan
            .instructions
            .iter()
            .copied()
            .filter(|word| word & LONG_OPCODE_MASK == SEND_OPCODE)
            .collect::<Vec<_>>();
        assert_eq!(sends.len(), 3);
        assert_eq!(send_address(sends[0]), 0x50160);
        assert_eq!(send_address(sends[1]), 0x50120);
        assert_eq!(send_address(sends[2]), 0x50168);
        assert_eq!(instruction_advance(sends[1]), 16);
        assert_eq!(plan.instructions[0], setzi_m(8, 1));
        let payload = plan
            .instructions
            .iter()
            .position(|instruction| *instruction == sends[1])
            .unwrap();
        let close = plan
            .instructions
            .iter()
            .position(|instruction| *instruction == sends[2])
            .unwrap();
        assert_eq!(close, payload + 2);
        assert_eq!(
            plan.instructions[payload + 1],
            delay(TILE_TO_HOST_MIN_PAYLOAD_EVENTS - 64 / 4 - 1)
        );
    }

    #[test]
    fn randomized_tile_to_host_packets_observe_the_payload_interval() {
        let mut random = fastrand::Rng::with_seed(0x686f_7374_5f70_6164);
        for _ in 0..256 {
            let host_offset = random.u32(0..256) * 4;
            let bytes = random.u32(1..=1024) * 4;
            let chunks = plan_tile_to_host(2, 0x52000, host_offset, bytes).unwrap();
            let target = assemble_tile_to_host_target_program(
                2,
                0x52000,
                host_offset,
                bytes,
                0x50160,
                0x501a0,
            )
            .unwrap();

            let mut cursor = 2;
            for _ in 0..chunks.len() {
                cursor += 1; // packet header
                let payload_events = instruction_advance(target.instructions[cursor]);
                cursor += 1;
                let padding_events = if payload_events < TILE_TO_HOST_MIN_PAYLOAD_EVENTS {
                    let events = instruction_advance(target.instructions[cursor]);
                    cursor += 1;
                    events
                } else {
                    0
                };
                assert_eq!(
                    payload_events + padding_events,
                    payload_events.max(TILE_TO_HOST_MIN_PAYLOAD_EVENTS)
                );
            }
            assert_eq!(cursor + 3, target.instructions.len());
        }
    }

    #[test]
    fn tile_to_host_target_has_no_controller_xreq_or_sync_wrapper() {
        let target =
            assemble_tile_to_host_target_program(2, 0x50120, 0x40, 64, 0x50160, 0x501a0).unwrap();

        assert_eq!(target.packet_words.len(), 4);
        assert_eq!(
            &target.packet_words[..2],
            &host_packet_words(tile_to_host_packet(2, 0x40, 64).unwrap())
        );
        assert!(!target.instructions.contains(&SYNC_HOST_INSTRUCTION));
        assert!(!target.instructions.contains(&SYNC_ALL_INSTRUCTION));
        assert_eq!(
            target.instructions[target.instructions.len() - 2],
            SYNC_RECEIVE_INSTRUCTION
        );
    }

    #[test]
    fn groups_multi_packet_tile_to_host_payloads() {
        let chunks = plan_tile_to_host(2, 0x52000, 0x40, 2048).unwrap();
        assert_eq!(chunks.len(), 8);
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.bytes == TILE_TO_HOST_MAX_BYTES)
        );

        let target =
            assemble_tile_to_host_target_program(2, 0x52000, 0x40, 2048, 0x50160, 0x501a0).unwrap();

        assert_eq!(target.packet_words.len(), chunks.len() * 2 + 2);
        assert_eq!(target.instructions[0], setzi_m(8, 1));
        assert!(
            target
                .instructions
                .iter()
                .filter(|word| **word & LONG_OPCODE_MASK == SEND_OPCODE)
                .skip(1)
                .step_by(2)
                .all(|word| instruction_advance(*word) == TILE_TO_HOST_MAX_BYTES / 4)
        );
        assert_eq!(
            target
                .instructions
                .iter()
                .filter(|word| **word & LONG_OPCODE_MASK == SEND_OPCODE)
                .count(),
            chunks.len() * 2 + 1
        );
    }

    #[test]
    fn host_to_tile_target_preserves_packet_and_request_addresses() {
        let plan = assemble_host_to_tile_target_program(2, 0x50120, 0x40, 64, 0x50160).unwrap();
        assert_eq!(
            &plan.packet_words[..],
            &host_packet_words(host_to_tile_packet(2, 0x50120, 0x40, 64).unwrap())
        );
        let sends = plan
            .instructions
            .iter()
            .copied()
            .filter(|word| word & LONG_OPCODE_MASK == SEND_OPCODE)
            .collect::<Vec<_>>();
        assert_eq!(sends.len(), 1);
        assert_eq!(send_address(sends[0]), 0x50160);
        assert_eq!(
            plan.instructions[plan.instructions.len() - 2],
            SYNC_RECEIVE_INSTRUCTION
        );
        assert!(!plan.instructions.contains(&SYNC_HOST_INSTRUCTION));
        assert!(!plan.instructions.contains(&SYNC_ALL_INSTRUCTION));
    }

    #[test]
    fn groups_multi_packet_host_to_tile_as_one_stream_copy() {
        let chunks = plan_host_to_tile(63, 0x50000, 0x40, 4096).unwrap();
        let plan = assemble_host_to_tile_target_program(63, 0x50000, 0x40, 4096, 0x54000).unwrap();
        assert_eq!(plan.packet_words.len(), chunks.len() * 2);
        assert!(
            plan.packet_words[..plan.packet_words.len() - 2]
                .chunks_exact(2)
                .all(|header| header[0] & HOST_TO_TILE_STREAM_END_BITS == 0)
        );
        assert_eq!(
            plan.packet_words[plan.packet_words.len() - 2] & HOST_TO_TILE_STREAM_END_BITS,
            HOST_TO_TILE_STREAM_END_BITS
        );
        assert_eq!(
            plan.instructions
                .iter()
                .filter(|instruction| **instruction & LONG_OPCODE_MASK == SEND_OFF_OPCODE)
                .count(),
            chunks.len() - 1
        );
    }

    #[test]
    fn host_command_read_encoder_preserves_recovered_packet_and_addresses() {
        let plan = assemble_host_command_read_program(0x50160, 0x50180, 0x1000).unwrap();
        assert_eq!(plan.packet_words, [1, 0, 0xcc00_020c, 0x4001]);
        assert_eq!(send_address(plan.instructions[5]), 0x50160);
        assert_eq!(send_address(plan.instructions[6]), 0x50168);
        let command_send = plan.instructions[plan.instructions.len() - 2];
        assert_eq!(send_address(command_send), 0x50180);
    }

    fn host_packet_words(header: HostPacketHeader) -> [u32; 2] {
        [header.word0, header.word1]
    }

    #[test]
    fn host_mux_uses_the_physical_tile_row_endpoint() {
        for (physical_tile, mux) in [(116, 0x634), (582, 0x604), (1173, 0x615)] {
            assert_eq!(host_mux_for_tile(physical_tile).unwrap(), mux);
        }
    }

    #[test]
    fn host_xreq_combines_target_endpoint_bits() {
        let targets = [31, 81, 768, 1471];
        let combined = assemble_host_xreq_program_for_targets(&targets, 0x50120).unwrap();
        let expected = targets
            .into_iter()
            .map(|target| {
                assemble_host_xreq_program(target, 0x50120)
                    .unwrap()
                    .packet_words
            })
            .fold([0u32; 2], |mut bitmap, words| {
                bitmap[0] |= words[0];
                bitmap[1] |= words[1];
                bitmap
            });
        assert_eq!(combined.packet_words, expected);
        assert_eq!(
            combined.instructions,
            assemble_host_xreq_program(31, 0x50120)
                .unwrap()
                .instructions
        );
    }

    #[test]
    fn target_operations_match_sdk_logical_tile_100_oracle() {
        let hierarchy = host_hierarchy(260).unwrap();
        assert_eq!(hierarchy.xreq_physical_tile, 4);
        let xreq = assemble_host_xreq_program(260, 0x50120).unwrap();
        assert_eq!(xreq.instructions, [0x782a_0243, 0x43a0_0000]);
        assert_eq!(xreq.packet_words, [0x100, 0]);
        let first_group = host_hierarchy(31).unwrap();
        assert_eq!(first_group.xreq_physical_tile, 29);
        assert_eq!(
            assemble_host_xreq_program(31, 0x50120)
                .unwrap()
                .packet_words,
            [2, 0]
        );
        assert_eq!(
            assemble_host_xreq_program(81, 0x50120)
                .unwrap()
                .packet_words,
            [4, 0]
        );
        assert_eq!(
            assemble_host_xreq_program(768, 0x50120)
                .unwrap()
                .packet_words,
            [0, 1]
        );
        assert_eq!(
            assemble_host_xreq_program(1471, 0x50120)
                .unwrap()
                .packet_words,
            [0, 1 << 21]
        );
        let wrapped_xreq =
            wrap_host_xreq_operation(hierarchy.xreq_physical_tile, &xreq.instructions).unwrap();
        assert_eq!(&wrapped_xreq[..3], &[0x1980_0604, 0x4380_80a0, 0x4180_000f]);

        let d2h =
            assemble_tile_to_host_target_program(260, 0x50120, 0x40, 64, 0x50160, 0x50180).unwrap();
        assert_eq!(
            d2h.instructions,
            [
                0x1980_0001,
                0x4380_80a6,
                0x782a_02c3,
                0x79ea_0243,
                0x40a0_0001,
                0x782a_02d3,
                0x4180_0000,
                0x43a0_0000,
            ]
        );
        assert_eq!(d2h.packet_words, [0xa082_0000, 0x0000_0011, 0xcc82_020c, 0]);

        let h2d = assemble_host_to_tile_target_program(260, 0x50120, 0x40, 64, 0x50170).unwrap();
        assert_eq!(
            h2d.instructions,
            [
                0x1980_0010,
                0x4380_80a6,
                0x782a_02e3,
                0x4180_0000,
                0x43a0_0000,
            ]
        );
        assert_eq!(h2d.packet_words, [0xec82_0209, 0x0000_0011]);

        let wrapped = wrap_host_target_operation(260, &d2h.instructions).unwrap();
        assert_eq!(&wrapped[..3], &[0x1980_0604, 0x4380_80a0, 0x4180_0007]);
        assert_eq!(
            &wrapped[wrapped.len() - 4..],
            &[0x4180_0007, 0x1980_0640, 0x4380_80a0, 0x43a0_0000]
        );

        let local = assemble_host_to_tile_target_program(0, 0x50120, 0x40, 64, 0x50168).unwrap();
        let wrapped_local = wrap_combined_host_operation(0, &local.instructions, 0x50160).unwrap();
        assert_eq!(
            &wrapped_local[..3],
            &[0x1980_0600, 0x4380_80a0, 0x4180_000f]
        );

        let wrapped_tile_nine =
            wrap_combined_host_operation(9, &local.instructions, 0x50160).unwrap();
        assert_eq!(wrapped_tile_nine[0], 0x1980_0609);
        assert_eq!(send_address(wrapped_local[5]), 0x50160);
    }

    fn send_address(instruction: u32) -> u32 {
        ((instruction & 0x001f_fff8) >> 3) * 4
    }

    #[test]
    fn finalizes_point_receiver_for_direct_execution() {
        let topology = Topology::c600();
        let plan = topology.point_to_point(274, 1286, 64).unwrap();
        let row = finalize_point_receiver(&plan.receiver, topology.physical(274).unwrap()).unwrap();
        assert_eq!(row[0], SYNC_SUPERVISOR_INSTRUCTION);
        assert_eq!(row[1] & 0x1fff, 9);
        assert_eq!(row[5], RETURN_M10_INSTRUCTION);
    }
}
