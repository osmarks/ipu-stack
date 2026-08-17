use ipu_package::AddressRegion;

/// Runtime completion word followed by the supervisor and worker stack state.
pub const RUNTIME_STATE_BASE: u32 =
    ipu_target::EXCHANGE_WINDOW_BASE + ipu_target::EXCHANGE_WINDOW_BYTES;
pub const WORKER_STACK_HEADROOM: u32 = 0xe0;
pub const WORKER_SYNC_STRIDE: u32 = 0x100;
pub const WORKER_CONTEXTS: u32 = 6;
pub const RUNTIME_STATE_BYTES: u32 = WORKER_STACK_HEADROOM + WORKER_CONTEXTS * WORKER_SYNC_STRIDE;
pub const PROFILE_START_CYCLE: u32 = RUNTIME_STATE_BASE + 4;
pub const PROFILE_END_CYCLE: u32 = RUNTIME_STATE_BASE + 8;

/// First byte after the permanently reserved runtime state.
pub const IPU21_DATA_BASE: u32 = RUNTIME_STATE_BASE + RUNTIME_STATE_BYTES;
/// Loader-populatable region 1 storage available to interleaved data.
pub const IPU21_INTERLEAVED_REGION_BYTES: u32 =
    ipu_package::IPU21_APPLICATION_MEMORY_LIMIT - ipu_package::IPU21_INTERLEAVED_MEMORY_BASE;
/// Standard-addressable storage which is not borrowed from region 1.
pub const IPU21_STANDARD_FIXED_BYTES: u32 =
    ipu_package::IPU21_INTERLEAVED_MEMORY_BASE - IPU21_DATA_BASE;
/// Total tile SRAM available to planned values after permanent runtime state.
pub const IPU21_PLANNED_DATA_BYTES: u32 =
    IPU21_STANDARD_FIXED_BYTES + IPU21_INTERLEAVED_REGION_BYTES;
/// Baseline standard-memory budget for package support data that is only
/// materialized after operator planning. Exchange tables are element-aligned,
/// so three standard memory elements preserve one contiguous allocation once
/// host-command and generated-program data are placed around the interleaved
/// region.
pub const IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES: u32 = 3 * ipu_package::TILE_MEMORY_ELEMENT_SIZE;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MemoryAllocation {
    pub name: &'static str,
    pub range: AddressRegion,
    reserved: AddressRegion,
}

#[derive(Clone, Debug)]
pub(crate) struct TileMemoryMap {
    free: Vec<AddressRegion>,
    allocations: Vec<MemoryAllocation>,
}

#[derive(Clone, Debug)]
pub(crate) struct MemoryRequest {
    pub name: &'static str,
    pub bytes: u32,
    pub alignment: u32,
    pub bounds: AddressRegion,
    /// Aligns the first following allocation and reserves any resulting gap.
    pub end_alignment: u32,
    /// Additional inaccessible bytes after the payload, before end alignment.
    pub guard_after: u32,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum MemoryLayoutError {
    #[error("invalid tile-memory request for {0}")]
    Invalid(&'static str),
    #[error("tile-memory region {name} at 0x{start:x}..0x{end:x} overlaps another allocation")]
    Overlap {
        name: &'static str,
        start: u32,
        end: u32,
    },
    #[error("no tile-memory range can satisfy {name} ({bytes} bytes)")]
    OutOfMemory { name: &'static str, bytes: u32 },
}

impl TileMemoryMap {
    pub(crate) fn new() -> Self {
        let free = std::iter::once(AddressRegion::new(
            ipu_package::TILE_MEMORY_BASE,
            ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
        ))
        .collect();
        Self {
            free,
            allocations: Vec::new(),
        }
    }

    pub(crate) fn reserve(
        &mut self,
        name: &'static str,
        range: AddressRegion,
    ) -> Result<MemoryAllocation, MemoryLayoutError> {
        if range.start >= range.end {
            return Err(MemoryLayoutError::Invalid(name));
        }
        let Some(index) = self
            .free
            .iter()
            .position(|free| free.start <= range.start && range.end <= free.end)
        else {
            return Err(MemoryLayoutError::Overlap {
                name,
                start: range.start,
                end: range.end,
            });
        };
        let free = self.free.remove(index);
        if free.start < range.start {
            self.free.push(AddressRegion::new(free.start, range.start));
        }
        if range.end < free.end {
            self.free.push(AddressRegion::new(range.end, free.end));
        }
        self.free.sort_by_key(|range| range.start);
        let allocation = MemoryAllocation {
            name,
            range,
            reserved: range,
        };
        self.allocations.push(allocation.clone());
        Ok(allocation)
    }

    pub(crate) fn allocate(
        &mut self,
        request: MemoryRequest,
    ) -> Result<MemoryAllocation, MemoryLayoutError> {
        if request.bytes == 0
            || !request.alignment.is_power_of_two()
            || !request.end_alignment.is_power_of_two()
            || request.bounds.start >= request.bounds.end
        {
            return Err(MemoryLayoutError::Invalid(request.name));
        }
        for free in self.free.clone() {
            let start = align_up(
                free.start.max(request.bounds.start),
                request.alignment,
                request.name,
            )?;
            let Some(payload_end) = start.checked_add(request.bytes) else {
                continue;
            };
            let Some(guarded_end) = payload_end.checked_add(request.guard_after) else {
                continue;
            };
            let reserved_end = align_up(guarded_end, request.end_alignment, request.name)?;
            if reserved_end <= free.end.min(request.bounds.end) {
                let payload = AddressRegion::new(start, payload_end);
                let reserved =
                    self.reserve(request.name, AddressRegion::new(start, reserved_end))?;
                let allocation = MemoryAllocation {
                    name: request.name,
                    range: payload,
                    reserved: reserved.reserved,
                };
                *self
                    .allocations
                    .last_mut()
                    .expect("reserve records allocation") = allocation.clone();
                return Ok(allocation);
            }
        }
        Err(MemoryLayoutError::OutOfMemory {
            name: request.name,
            bytes: request.bytes,
        })
    }

    pub(crate) fn free_ranges(&self, bounds: AddressRegion) -> Vec<AddressRegion> {
        self.free
            .iter()
            .filter_map(|free| {
                let start = free.start.max(bounds.start);
                let end = free.end.min(bounds.end);
                (start < end).then_some(AddressRegion::new(start, end))
            })
            .collect()
    }

    pub(crate) fn next_free(
        &self,
        start: u32,
        bounds: AddressRegion,
        alignment: u32,
        name: &'static str,
    ) -> Result<u32, MemoryLayoutError> {
        if !alignment.is_power_of_two() {
            return Err(MemoryLayoutError::Invalid(name));
        }
        self.free_ranges(bounds)
            .into_iter()
            .find_map(|free| {
                let address = align_up(free.start.max(start), alignment, name).ok()?;
                (address < free.end).then_some(address)
            })
            .ok_or(MemoryLayoutError::OutOfMemory { name, bytes: 1 })
    }
}

fn align_up(value: u32, alignment: u32, name: &'static str) -> Result<u32, MemoryLayoutError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(MemoryLayoutError::Invalid(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn randomized_memory_requests_are_aligned_bounded_and_disjoint() {
        let mut random = fastrand::Rng::with_seed(0x6d65_6d6f_7279);
        for _ in 0..128 {
            let mut map = TileMemoryMap::new();
            map.reserve("fixed", AddressRegion::new(0x50000, 0x58000))
                .unwrap();
            for _ in 0..random.usize(1..=24) {
                let alignment = 1 << random.u32(2..=14);
                let bytes = 4 * random.u32(1..=1024);
                let result = map.allocate(MemoryRequest {
                    name: "random",
                    bytes,
                    alignment,
                    bounds: AddressRegion::new(
                        ipu_package::TILE_MEMORY_BASE,
                        ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
                    ),
                    end_alignment: 1 << random.u32(0..=14),
                    guard_after: random.u32(0..=64),
                });
                if let Ok(allocation) = result {
                    assert!(allocation.range.start.is_multiple_of(alignment));
                    assert_eq!(allocation.range.size(), bytes);
                }
            }
            let mut ranges = map
                .allocations
                .iter()
                .map(|allocation| allocation.reserved.clone())
                .collect::<Vec<_>>();
            ranges.sort_by_key(|range| range.start);
            assert!(ranges.windows(2).all(|pair| pair[0].end <= pair[1].start));
        }
    }
}
