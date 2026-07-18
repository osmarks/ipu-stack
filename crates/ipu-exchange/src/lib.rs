use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::debug;

pub const PLAN_WORDS: usize = 9;
pub const MAX_TRANSFER_WORDS: u32 = 4148;
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
const PIC_ABSOLUTE_RECEIVE_BITS: u32 = (1 << 16) | (1 << 14);
const PIC_RECEIVE_ADDRESS_MASK: u32 = 0x1fff;
const SEND_OPCODE: u32 = 0x7800_0000;
const SEND_OFF_OPCODE: u32 = 0x7000_0000;
const SYNC_OPCODE: u32 = 0x4180_0000;
const SANS_OPCODE: u32 = 0x40c0_0000;
const BR_M_OPCODE: u32 = 0x4300_0000;
const SETZI_M_OPCODE: u32 = 0x1900_0000;
const PUT_SPECIAL_M_OPCODE: u32 = 0x4300_8000;
const INCOMING_MUX_REGISTER: u8 = 0xa0;
const INCOMING_DCOUNT_REGISTER: u8 = 0xa6;
const TILE_TO_HOST_CLOSE_DELAY_ADVANCE: u32 = 2;
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
    for &instruction in row {
        let advance = instruction_advance(instruction);
        cycles = cycles
            .checked_add(advance)
            .ok_or(ExchangeError::Schedule("plan event horizon overflow"))?;
    }
    Ok(cycles)
}

#[derive(Clone, Debug)]
pub struct PlanProgramBuilder {
    words: Vec<u32>,
    event_cycles: u32,
}

impl Default for PlanProgramBuilder {
    fn default() -> Self {
        Self {
            words: vec![SYNC_SUPERVISOR_INSTRUCTION],
            event_cycles: 0,
        }
    }
}

impl PlanProgramBuilder {
    pub fn event_cycles(&self) -> u32 {
        self.event_cycles
    }

    pub fn append_scheduled_row(&mut self, row: &PlanRow) -> Result<(), ExchangeError> {
        if row[0] != SYNC_SUPERVISOR_INSTRUCTION {
            return Err(ExchangeError::Schedule("exchange row entry"));
        }
        let end = row
            .iter()
            .position(|instruction| *instruction == RETURN_M10_INSTRUCTION)
            .ok_or(ExchangeError::Schedule("exchange row return"))?;
        let mut body = row[1..end].to_vec();
        if self.event_cycles == 0 {
            self.words.extend(body);
            self.event_cycles = plan_event_cycles(&self.words)?;
            return Ok(());
        }
        let first_timed = body
            .iter()
            .position(|instruction| instruction_advance(*instruction) != 0)
            .ok_or(ExchangeError::Schedule("exchange row has no timed event"))?;
        let first_advance = instruction_advance(body[first_timed]);
        if first_advance <= self.event_cycles {
            return Err(ExchangeError::Schedule("overlapping tile exchange roles"));
        }
        set_instruction_advance(&mut body[first_timed], first_advance - self.event_cycles)?;
        self.words.extend(body);
        self.event_cycles = plan_event_cycles(&self.words)?;
        Ok(())
    }

    pub fn finish(mut self, horizon: u32) -> Result<Vec<u32>, ExchangeError> {
        if horizon < self.event_cycles {
            return Err(ExchangeError::Schedule("plan horizon precedes tile events"));
        }
        let padding = horizon - self.event_cycles;
        if padding != 0 {
            self.words.push(delay(padding - 1));
        }
        self.words.push(RETURN_M10_INSTRUCTION);
        Ok(self.words)
    }
}

fn instruction_advance(instruction: u32) -> u32 {
    if instruction & DELAY_OPCODE_MASK == DELAY_OPCODE {
        (instruction & 0x7_ffff) + 1
    } else {
        match instruction & OPCODE_MASK {
            DELAY_PIC_OPCODE => ((instruction >> 19) & 0x7f) + 1,
            DELAY_XPIC_OPCODE => ((instruction >> 14) & 0xfff) + 1,
            _ => match instruction & LONG_OPCODE_MASK {
                SEND_OPCODE => ((instruction >> 21) & 0x3f) + 1,
                SEND_OFF_OPCODE => {
                    (((instruction >> 21) & 0x3f) | (((instruction >> 14) & 0x3f) << 6)) + 1
                }
                _ => 0,
            },
        }
    }
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
            "first scheduled event is not a delay",
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
    validate_host_tile(target_physical_tile)?;
    if packet_address & 7 != 0 {
        return Err(ExchangeError::HostPacket);
    }
    let bitmap_index =
        u32::from(target_physical_tile / 64) * 2 + u32::from((target_physical_tile >> 1) & 1);
    let mut bitmap = [0u32; 2];
    if bitmap_index < XREQ_BITMAP0_BITS {
        bitmap[0] = 1 << bitmap_index;
    } else {
        bitmap[1] = 1 << (bitmap_index - XREQ_BITMAP0_BITS);
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
    }
    instructions.push(delay(TILE_TO_HOST_CLOSE_DELAY_ADVANCE - 1));
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
            sender[cursor] = send_off(count - 65, 3, 0);
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
            if count == 1 {
                row[2] = delay_pic(51 + receiver_phase, 0, 0) | 0x0001_4000;
                row[3] = delay(5);
                row[4] = RETURN_M10_INSTRUCTION;
            } else if count <= 51 {
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
    if byte_address & 3 != 0 || byte_address >> 2 > 0x1f_ffff {
        return Err(ExchangeError::Address(byte_address));
    }
    let word_address = byte_address >> 2;
    for instruction in row {
        if *instruction & 0xf800_0000 == 0x7800_0000 {
            *instruction = (*instruction & !0x001f_fff8) | ((word_address << 3) & 0x001f_fff8);
            return Ok(());
        }
    }
    Err(ExchangeError::Address(byte_address))
}

pub fn patch_receiver_address(row: &mut PlanRow, byte_address: u32) -> Result<(), ExchangeError> {
    if byte_address < EXCHANGE_WINDOW_BASE || byte_address & 3 != 0 {
        return Err(ExchangeError::Address(byte_address));
    }
    let window_word = (byte_address - EXCHANGE_WINDOW_BASE) >> 2;
    if window_word > 0x1fff {
        return Err(ExchangeError::Address(byte_address));
    }
    let instruction = row
        .iter_mut()
        .find(|word| **word & OPCODE_MASK == DELAY_PIC_OPCODE)
        .ok_or(ExchangeError::Address(byte_address))?;
    *instruction = (*instruction & !(PIC_ABSOLUTE_RECEIVE_BITS | PIC_RECEIVE_ADDRESS_MASK))
        | PIC_ABSOLUTE_RECEIVE_BITS
        | window_word;
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

const fn delay(cycles: u32) -> u32 {
    0x40a0_0000 | (cycles & 0x7ffff)
}

fn delay_pic(a: u32, b: u32, c: u32) -> u32 {
    0x6000_0000 | ((a << 19) & 0x03f8_0000) | ((b << 18) & 0x0004_0000) | (c & 0x3ffff)
}

fn delay_xpic(a: u32, b: u32, c: u32) -> u32 {
    0x6400_0000 | ((a << 14) & 0x03ff_c000) | ((b << 13) & 0x0000_2000) | (c & 0x1fff)
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

        assert!(encode_setzi_m(16, 0).is_err());
        assert!(encode_setzi_m(0, 1 << 20).is_err());
        assert!(encode_put_special_m(0, 16).is_err());
        assert!(encode_br_m(16).is_err());
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
            assert_eq!(plan.sender, expected_sender);
            assert_eq!(plan.receiver, expected_receiver);
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
    fn one_word_multicast_omits_zero_length_xpic_stage() {
        let topology = Topology::c600();
        let receivers = (1..topology.tile_count() as u16).collect::<Vec<_>>();
        let plan = topology.multicast(0, &receivers, 1, 0).unwrap();
        for mut receiver in plan.receivers {
            assert_eq!(
                receiver
                    .iter()
                    .filter(|instruction| **instruction & OPCODE_MASK == DELAY_XPIC_OPCODE)
                    .count(),
                1
            );
            let cycles = plan_event_cycles(&receiver).unwrap();
            patch_receiver_address(&mut receiver, EXCHANGE_WINDOW_BASE).unwrap();
            assert_eq!(plan_event_cycles(&receiver).unwrap(), cycles);
            assert_eq!(
                receiver.iter().rfind(|instruction| **instruction != 0),
                Some(&RETURN_M10_INSTRUCTION)
            );
        }
    }

    #[test]
    fn single_receiver_uses_the_directional_send_route() {
        let topology = Topology::c600();
        let unicast = topology.multicast(0, &[1], 16, 0).unwrap();
        let fanout = topology.multicast(0, &[1, 2], 16, 0).unwrap();

        assert_eq!(unicast.sender[2] & 7, 1);
        assert_eq!(fanout.sender[2] & 7, 3);
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

        let mut relay = PlanProgramBuilder::default();
        relay.append_scheduled_row(&first.receivers[0]).unwrap();
        relay.append_scheduled_row(&second.sender).unwrap();
        let relay = relay.finish(horizon).unwrap();

        assert_eq!(relay.first(), Some(&SYNC_SUPERVISOR_INSTRUCTION));
        assert_eq!(relay.last(), Some(&RETURN_M10_INSTRUCTION));
        assert_eq!(
            relay
                .iter()
                .filter(|instruction| **instruction == SYNC_SUPERVISOR_INSTRUCTION)
                .count(),
            1
        );
        assert_eq!(plan_event_cycles(&relay).unwrap(), horizon);
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
        assert_eq!(plan.receivers[0][2] & 0x1fff, (0x53080 - 0x50000) >> 2);
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
            delay(TILE_TO_HOST_CLOSE_DELAY_ADVANCE - 1)
        );
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
