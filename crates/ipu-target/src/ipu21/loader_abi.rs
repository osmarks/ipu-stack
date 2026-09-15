//! SDK secondary-loader transport contract shared by package validation and
//! the driver. Loadable SRAM ends before architectural SRAM; frame counts and
//! the startup handoff are properties of this loader, not of IPU21 instructions.
use super::memory::TILE_MEMORY_BASE;

// The secondary loader installs framed application payload at the SDK image's
// launch slot. Applications reserve that word and enter at the following word.
pub const APPLICATION_LOAD_BASE: u32 = TILE_MEMORY_BASE + 0x10;
pub const HSP_MARK_MASK: u32 = 0xffff;
// The secondary loader consumes 23 tile batches before handing HSP ownership
// to the resident host-exchange program.
pub const HOST_EXCHANGE_HANDOFF_MARK: u32 = 23;
pub const TILES_PER_BATCH: usize = 64;
pub const FRAME_SIZE: usize = 1024;
pub const FRAME_HEADER_SIZE: usize = 16;
pub const FRAME_TRAILER_SIZE: usize = 16;
pub const FRAME_PAYLOAD_SIZE: usize = FRAME_SIZE - FRAME_HEADER_SIZE - FRAME_TRAILER_SIZE;
pub const DESCRIPTOR_AREA_SIZE: usize = 4096;
pub const TRANSPORT_SIZE: usize = 0x2842000;
// The Graphcore secondary loader does not acknowledge a one-frame application.
// Pad transport payloads to the smallest established working envelope.
pub const SECONDARY_LOADER_MIN_PAYLOAD_SIZE: usize = 0x4134;
/// Maximum frame count accepted by the IPU21 SDK secondary bootloader.
pub const SECONDARY_LOADER_MAX_FRAMES: usize = 0x283;
/// Exclusive upper address that can be represented by that bootloader when
/// loading an application from [`APPLICATION_LOAD_BASE`].
pub const APPLICATION_LOAD_LIMIT: u32 =
    APPLICATION_LOAD_BASE + (SECONDARY_LOADER_MAX_FRAMES * FRAME_PAYLOAD_SIZE) as u32;
