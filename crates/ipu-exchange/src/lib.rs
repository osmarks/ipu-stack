use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::debug;

pub const PLAN_WORDS: usize = 9;
pub const MAX_TRANSFER_WORDS: u32 = 4148;
pub const EXCHANGE_WINDOW_BASE: u32 = 0x50000;
pub const HOST_SHORT_MAX_BYTES: u32 = 60;
pub const HOST_LONG_MAX_BYTES: u32 = 1024;

pub type PlanRow = [u32; PLAN_WORDS];

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
    pub packet_words: [u32; 4],
}

pub fn assemble_tile_to_host_xreq_program(
    packet_address: u32,
) -> Result<TileToHostProgram, ExchangeError> {
    if packet_address & 7 != 0 {
        return Err(ExchangeError::HostPacket);
    }
    Ok(TileToHostProgram {
        instructions: vec![
            0x1980_0600, // setzi $m8, 0x600
            0x4380_80a0, // put $INCOMING_MUX, $m8
            0x4180_000f, // sync 15
            encode_send(1, 3, packet_address >> 2)?,
            0x4180_0007, // sync 7
            0x1980_0640, // setzi $m8, 0x640
            0x4380_80a0, // put $INCOMING_MUX, $m8
            0x43a0_0000, // br $m10
        ],
        packet_words: [2, 0, 0, 0],
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
    Ok(TileToHostProgram {
        instructions: vec![
            0x1980_0600,
            0x4380_80a0,
            0x4180_000f,
            0x1980_0001,
            0x4380_80a6,
            encode_send(1, 3, packet_address >> 2)?,
            encode_send(1, 3, (packet_address + 8) >> 2)?,
            0x4180_0000,
            0x4180_0007,
            0x1980_0640,
            0x4380_80a0,
            0x43a0_0000,
        ],
        packet_words: [1, 0, request.word0, request.word1],
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
    if exchange_address > 0x1ff {
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
    if exchange_address > 0x1ff {
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
    plan_host_transfer(tile_address, host_offset, bytes, |tile, host, count| {
        host_to_tile_packet(physical_tile, tile, host, count)
    })
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
    plan_host_transfer(tile_address, host_offset, bytes, |_tile, host, count| {
        tile_to_host_packet(physical_tile, host, count)
    })
}

pub fn assemble_tile_to_host_program(
    physical_tile: u16,
    tile_address: u32,
    host_offset: u32,
    bytes: u32,
    packet_address: u32,
    dummy_tile_address: u32,
) -> Result<TileToHostProgram, ExchangeError> {
    let chunks = plan_tile_to_host(physical_tile, tile_address, host_offset, bytes)?;
    if chunks.len() != 1 || packet_address & 7 != 0 {
        return Err(ExchangeError::HostPacket);
    }
    let chunk = chunks[0];
    let close = zero_byte_read_packet(physical_tile, dummy_tile_address)?;
    let mut instructions = vec![
        0x1980_0600, // setzi $m8, 0x600
        0x4380_80a0, // put $INCOMING_MUX, $m8
        0x4180_0007, // sync 7
        0x1980_0001, // setzi $m8, 1
        0x4380_80a6, // put $INCOMING_DCOUNT, $m8
        encode_send(1, 3, packet_address >> 2)?,
    ];
    let words = chunk.bytes / 4;
    for offset in (0..words).step_by(64) {
        instructions.push(encode_send(
            (words - offset).min(64) - 1,
            3,
            (chunk.tile_address >> 2) + offset,
        )?);
    }
    instructions.extend([
        delay(1),
        encode_send(1, 3, (packet_address + 8) >> 2)?,
        0x4180_0000, // sync 0
        0x4180_0007, // sync 7
        0x1980_0640, // setzi $m8, 0x640
        0x4380_80a0, // put $INCOMING_MUX, $m8
        0x43a0_0000, // br $m10
    ]);
    Ok(TileToHostProgram {
        instructions,
        packet_words: [
            chunk.header.word0,
            chunk.header.word1,
            close.word0,
            close.word1,
        ],
    })
}

fn plan_host_transfer(
    mut tile_address: u32,
    mut host_offset: u32,
    mut bytes: u32,
    packet: impl Fn(u32, u32, u32) -> Result<HostPacketHeader, ExchangeError>,
) -> Result<Vec<HostTransferChunk>, ExchangeError> {
    let mut chunks = Vec::new();
    while bytes != 0 {
        let count = if host_offset & 63 == 0 && bytes >= 64 {
            bytes.min(HOST_LONG_MAX_BYTES) & !63
        } else if bytes <= HOST_SHORT_MAX_BYTES {
            bytes
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
        sender_row[0] = 0x4180_0003;
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
        sender_row[cursor] = 0x43a0_0000;

        let mut receiver_row = [0; PLAN_WORDS];
        receiver_row[0] = 1;
        receiver_row[1] = 0x4180_0003;
        receiver_row[2] = delay_xpic(112, 0, 0);
        if count <= 51 {
            receiver_row[3] = delay_xpic(count - 1, 0, 0x640);
            receiver_row[4] = delay_pic(51 - count + receiver_phase, 0, 0);
            receiver_row[5] = delay(count + 4);
            receiver_row[6] = 0x43a0_0000;
        } else if count == 52 {
            receiver_row[3] = delay_pic(50 + receiver_phase, 0, 0);
            receiver_row[4] = delay_xpic(0, 0, 0x640);
            receiver_row[5] = delay(56);
            receiver_row[6] = 0x43a0_0000;
        } else {
            receiver_row[3] = delay_pic(51 + receiver_phase, 0, 0);
            receiver_row[4] = delay_xpic(count - 53, 0, 0x640);
            receiver_row[5] = delay(56);
            receiver_row[6] = 0x43a0_0000;
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
        sender[cursor] = 0x4180_0003;
        cursor += 1;
        if sender_delay >= 0 {
            sender[cursor] = delay(sender_delay as u32);
            cursor += 1;
        }
        sender[cursor] = encode_send(count.min(64) - 1, 3, 0)?;
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
        sender[cursor] = 0x43a0_0000;

        let mut receivers = Vec::with_capacity(receiver_logical.len());
        for (logical, mux_time) in receiver_logical.iter().zip(mux_times) {
            let physical = u32::from(self.physical(*logical)?);
            let receive_cycle = start_cycle as i32 + mux_time;
            if !(0..=4095).contains(&receive_cycle) {
                return Err(ExchangeError::Schedule("multicast receive cycle"));
            }
            let receiver_phase = 2 * (physical >> 6);
            let mut row = [0; PLAN_WORDS];
            row[0] = 0x4180_0003;
            row[1] = delay_xpic(receive_cycle as u32, 0, source_physical);
            if count <= 51 {
                row[2] = delay_xpic(count - 1, 0, 0x640);
                row[3] = delay_pic(51 - count + receiver_phase, 0, 0) | 0x0001_4000;
                row[4] = delay(count + 4);
                row[5] = 0x43a0_0000;
            } else if count == 52 {
                row[2] = delay_pic(50 + receiver_phase, 0, 0) | 0x0001_4000;
                row[3] = delay_xpic(0, 0, 0x640);
                row[4] = delay(56);
                row[5] = 0x43a0_0000;
            } else {
                row[2] = delay_pic(51 + receiver_phase, 0, 0) | 0x0001_4000;
                row[3] = delay_xpic(count - 53, 0, 0x640);
                row[4] = delay(56);
                row[5] = 0x43a0_0000;
            }
            receivers.push(row);
        }
        debug!(
            sender_logical,
            receivers = receivers.len(),
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

pub fn patch_multicast_receiver_address(
    row: &mut PlanRow,
    byte_address: u32,
) -> Result<(), ExchangeError> {
    if byte_address < EXCHANGE_WINDOW_BASE || byte_address & 3 != 0 {
        return Err(ExchangeError::Address(byte_address));
    }
    let window_word = (byte_address - EXCHANGE_WINDOW_BASE) >> 2;
    if window_word > 0x1fff {
        return Err(ExchangeError::Address(byte_address));
    }
    let instruction = row
        .iter_mut()
        .find(|word| **word & 0xfc01_4000 == 0x6001_4000)
        .ok_or(ExchangeError::Address(byte_address))?;
    *instruction = (*instruction & !0x1fff) | window_word;
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

fn delay(cycles: u32) -> u32 {
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
        patch_multicast_receiver_address(&mut plan.receivers[0], 0x53080).unwrap();
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

        let h2d = plan_host_to_tile(1409, 0x50000, 4, 100).unwrap();
        assert_eq!(
            h2d.iter().map(|chunk| chunk.bytes).collect::<Vec<_>>(),
            [32, 32, 36]
        );
        assert_eq!(h2d.last().unwrap().tile_address + 36, 0x50064);
        assert!(plan_host_to_tile(0, 0x50004, 0, 4).is_err());
    }

    #[test]
    fn assembles_tile_to_host_program() {
        let plan = assemble_tile_to_host_program(2, 0x50120, 0x40, 64, 0x50160, 0x50180).unwrap();
        assert_eq!(plan.packet_words, [0xa001_0000, 0x11, 0xcc01_020c, 0]);
        assert_eq!(
            plan.instructions[0..3],
            [0x1980_0600, 0x4380_80a0, 0x4180_0007]
        );
        assert_eq!(
            plan.instructions[3..6],
            [0x1980_0001, 0x4380_80a6, 0x782a_02c3]
        );
        assert_eq!(plan.instructions[6], 0x79ea_0243);
        assert_eq!(
            plan.instructions[7..],
            [
                0x40a0_0001,
                0x782a_02d3,
                0x4180_0000,
                0x4180_0007,
                0x1980_0640,
                0x4380_80a0,
                0x43a0_0000
            ]
        );
    }

    #[test]
    fn assembles_tile_to_host_xreq_program() {
        let plan = assemble_tile_to_host_xreq_program(0x50170).unwrap();
        assert_eq!(plan.packet_words, [2, 0, 0, 0]);
        assert_eq!(plan.instructions[2], 0x4180_000f);
        assert_eq!(plan.instructions[3], 0x782a_02e3);
        assert_eq!(plan.instructions.last(), Some(&0x43a0_0000));
    }

    #[test]
    fn assembles_host_command_read_program() {
        let plan = assemble_host_command_read_program(0x50160, 0x50180, 0x1000).unwrap();
        assert_eq!(plan.packet_words, [1, 0, 0xcc00_020c, 0x4001]);
        assert_eq!(
            plan.instructions[5..8],
            [0x782a_02c3, 0x782a_02d3, 0x4180_0000]
        );
    }

    #[test]
    fn finalizes_point_receiver_for_direct_execution() {
        let topology = Topology::c600();
        let plan = topology.point_to_point(274, 1286, 64).unwrap();
        let row = finalize_point_receiver(&plan.receiver, topology.physical(274).unwrap()).unwrap();
        assert_eq!(row[0], 0x4180_0003);
        assert_eq!(row[1] & 0x1fff, 9);
        assert_eq!(row[5], 0x43a0_0000);
    }
}
