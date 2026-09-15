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

// Memory available under the resident runtime and SDK loader contracts.
/// First byte after the permanently reserved runtime state.
pub const IPU21_DATA_BASE: u32 =
    super::runtime_layout::RUNTIME_STATE_BASE + super::runtime_layout::RUNTIME_STATE_BYTES;
/// Loader-populatable region 1 storage available to interleaved data.
pub const IPU21_INTERLEAVED_REGION_BYTES: u32 =
    super::loader_abi::APPLICATION_LOAD_LIMIT - IPU21_INTERLEAVED_MEMORY_BASE;
/// Standard-addressable storage which is not borrowed from region 1.
pub const IPU21_STANDARD_FIXED_BYTES: u32 = IPU21_INTERLEAVED_MEMORY_BASE - IPU21_DATA_BASE;
/// Total tile SRAM available to planned values after permanent runtime state.
pub const IPU21_PLANNED_DATA_BYTES: u32 =
    IPU21_STANDARD_FIXED_BYTES + IPU21_INTERLEAVED_REGION_BYTES;
/// Default standard-memory allowance for linked support, exchange rows and profiling.
pub const IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES: u32 = 3 * TILE_MEMORY_ELEMENT_SIZE;

/// Base addresses of SRAM contention regions touched by a word range.
/// Interleaved accesses occupy pairs of physical elements.
pub fn effective_memory_elements(address: u32, words: u32) -> Vec<u32> {
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
        elements.push(base + index * size);
        let boundary = base.saturating_add((index + 1).saturating_mul(size));
        cursor = boundary.min(end);
    }
    elements
}
