use super::*;
use serde::{Deserialize, Serialize};

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
    window_base: u32,
    packet_address: u32,
    destination_address: u32,
    host_offset: u32,
) -> Result<TileToHostProgram, ExchangeError> {
    if packet_address & 7 != 0 {
        return Err(ExchangeError::HostPacket);
    }
    let request = host_to_tile_packet(window_base, 0, destination_address, host_offset, 4)?;
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
    window_base: u32,
    physical_tile: u16,
    tile_address: u32,
    host_offset: u32,
    bytes: u32,
    packet_address: u32,
) -> Result<TileToHostProgram, ExchangeError> {
    let chunks = plan_host_to_tile(window_base, physical_tile, tile_address, host_offset, bytes)?;
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

/// One routed transfer, ready for scheduling without encoding or decoding rows.
fn host_exchange_address(window_base: u32, address: u32) -> Result<u32, ExchangeError> {
    let offset = address
        .checked_sub(window_base)
        .ok_or(ExchangeError::HostPacket)?;
    if window_base & 31 != 0 || offset & 31 != 0 || offset >= HOST_TO_TILE_WINDOW_BYTES {
        return Err(ExchangeError::HostPacket);
    }
    Ok(offset >> 5)
}

pub fn host_to_tile_packet(
    window_base: u32,
    physical_tile: u16,
    tile_address: u32,
    host_offset: u32,
    bytes: u32,
) -> Result<HostPacketHeader, ExchangeError> {
    validate_host_tile(physical_tile)?;
    let exchange_address = host_exchange_address(window_base, tile_address)?;
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
    window_base: u32,
    physical_tile: u16,
    dummy_tile_address: u32,
) -> Result<HostPacketHeader, ExchangeError> {
    validate_host_tile(physical_tile)?;
    let exchange_address = host_exchange_address(window_base, dummy_tile_address)?;
    Ok(HostPacketHeader {
        word0: 0xcc00_0200 | host_route_word0(physical_tile) | exchange_address,
        word1: host_route_word1(physical_tile),
    })
}

pub fn plan_host_to_tile(
    window_base: u32,
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
        |tile, host, count| host_to_tile_packet(window_base, physical_tile, tile, host, count),
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
    window_base: u32,
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
    let close = zero_byte_read_packet(window_base, physical_tile, command_address)?;
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
