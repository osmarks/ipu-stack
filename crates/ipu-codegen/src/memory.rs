/// Runtime completion word followed by the supervisor and worker stack state.
pub const RUNTIME_STATE_BASE: u32 =
    ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES;
pub const WORKER_STACK_HEADROOM: u32 = 0xe0;
pub const WORKER_SYNC_STRIDE: u32 = 0x100;
pub const WORKER_CONTEXTS: u32 = 6;
pub const RUNTIME_STATE_BYTES: u32 = WORKER_STACK_HEADROOM + WORKER_CONTEXTS * WORKER_SYNC_STRIDE;
pub const PROFILE_START_CYCLE: u32 = RUNTIME_STATE_BASE + 4;
pub const PROFILE_END_CYCLE: u32 = RUNTIME_STATE_BASE + 8;

/// First byte after the permanently reserved runtime state.
pub const IPU21_DATA_BASE: u32 = RUNTIME_STATE_BASE + RUNTIME_STATE_BYTES;
pub const IPU21_INTERLEAVED_BYTES: u32 =
    ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT - ipu_package::IPU21_INTERLEAVED_MEMORY_BASE;
/// All of architectural region 1 can back interleaved data. The smaller
/// `IPU21_INTERLEAVED_BYTES` value is only the default scratch reservation.
pub const IPU21_INTERLEAVED_REGION_BYTES: u32 =
    ipu_package::IPU21_INTERLEAVED_REGION_LIMIT - ipu_package::IPU21_INTERLEAVED_MEMORY_BASE;
/// Standard-addressable storage which is not borrowed from region 1.
pub const IPU21_STANDARD_FIXED_BYTES: u32 =
    ipu_package::IPU21_INTERLEAVED_MEMORY_BASE - IPU21_DATA_BASE;
pub const IPU21_STANDARD_DATA_BYTES: u32 = ipu_package::IPU21_INTERLEAVED_MEMORY_BASE
    - IPU21_DATA_BASE
    + ipu_package::TILE_MEMORY_BASE
    + ipu_package::TILE_MEMORY_SIZE
    - ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT;
/// Total tile SRAM available to planned values after permanent runtime state.
pub const IPU21_PLANNED_DATA_BYTES: u32 =
    IPU21_STANDARD_FIXED_BYTES + IPU21_INTERLEAVED_REGION_BYTES;
