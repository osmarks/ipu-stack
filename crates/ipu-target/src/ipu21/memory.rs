//! Tile SRAM and instruction-fetch geometry. Region 1 interleaves pairs of
//! physical memory elements; its base is architectural, not a placement choice.
pub const TILE_MEMORY_BASE: u32 = 0x4c000;
pub const TILE_MEMORY_SIZE: u32 = 624 * 1024;
/// TMEM_ELEMSIZE: instruction/data contention granularity.
pub const TILE_MEMORY_ELEMENT_SIZE: u32 = 0x4000;
/// Maximum supervisor instruction-fetch lookahead in bytes.
pub const IPU21_SUPERVISOR_FETCH_LOOKAHEAD: u32 = 8 * 8;
/// Region 0 is the only region supporting instruction fetch.
pub const IPU21_EXECUTABLE_MEMORY_LIMIT: u32 = 0x80000;
/// Region 1 begins after region 0 and interleaves two physical elements.
pub const IPU21_INTERLEAVED_MEMORY_BASE: u32 = IPU21_EXECUTABLE_MEMORY_LIMIT;
pub const IPU21_INTERLEAVED_REGION_LIMIT: u32 = TILE_MEMORY_BASE + TILE_MEMORY_SIZE;
pub const IPU21_INTERLEAVED_ELEMENT_SIZE: u32 = 2 * TILE_MEMORY_ELEMENT_SIZE;
