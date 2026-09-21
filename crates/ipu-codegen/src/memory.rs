use ipu_target::Target;
use std::ops::Range;

pub use ipu_target::ipu21::runtime_layout::{PROFILE_END_CYCLE, PROFILE_START_CYCLE};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MemoryAllocation {
    pub name: &'static str,
    pub range: Range<u32>,
    /// Entire reservation, including end alignment and any trailing guard.
    pub reserved: Range<u32>,
}

#[derive(Clone, Debug)]
pub(crate) struct TileMemoryMap {
    free: Vec<Range<u32>>,
    allocations: Vec<MemoryAllocation>,
}

#[derive(Clone, Debug)]
pub(crate) struct MemoryRequest {
    pub name: &'static str,
    pub bytes: u32,
    pub alignment: u32,
    pub bounds: Range<u32>,
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
    pub(crate) fn allocations(&self) -> &[MemoryAllocation] {
        &self.allocations
    }

    pub(crate) fn new(target: Target) -> Self {
        let free = std::iter::once(target.tile_memory()).collect();
        Self {
            free,
            allocations: Vec::new(),
        }
    }

    pub(crate) fn reserve(
        &mut self,
        name: &'static str,
        range: Range<u32>,
    ) -> Result<&mut MemoryAllocation, MemoryLayoutError> {
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
        let free = &self.free[index];
        let remaining = [free.start..range.start, range.end..free.end];
        self.free.splice(
            index..=index,
            remaining.into_iter().filter(|range| !range.is_empty()),
        );
        self.allocations.push(MemoryAllocation {
            name,
            range: range.clone(),
            reserved: range,
        });
        Ok(self.allocations.last_mut().unwrap())
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
        for free in &self.free {
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
                let allocation = self.reserve(request.name, start..reserved_end)?;
                allocation.range.end = payload_end;
                return Ok(allocation.clone());
            }
        }
        Err(MemoryLayoutError::OutOfMemory {
            name: request.name,
            bytes: request.bytes,
        })
    }

    pub(crate) fn free_ranges(&self, bounds: Range<u32>) -> Vec<(u32, u32)> {
        self.free
            .iter()
            .filter_map(|free| {
                let start = free.start.max(bounds.start);
                let end = free.end.min(bounds.end);
                (start < end).then_some((start, end))
            })
            .collect()
    }

    pub(crate) fn next_free(
        &self,
        start: u32,
        bounds: Range<u32>,
        alignment: u32,
        name: &'static str,
    ) -> Result<u32, MemoryLayoutError> {
        if !alignment.is_power_of_two() {
            return Err(MemoryLayoutError::Invalid(name));
        }
        self.free_ranges(bounds)
            .into_iter()
            .find_map(|(free_start, free_end)| {
                let address = align_up(free_start.max(start), alignment, name).ok()?;
                (address < free_end).then_some(address)
            })
            .ok_or(MemoryLayoutError::OutOfMemory { name, bytes: 1 })
    }
}

/// Complement within `start..end` of occupied ranges sorted by start address.
pub(crate) fn uncovered_ranges(start: u32, end: u32, occupied: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut cursor = start;
    let mut result = Vec::new();
    for &(a, b) in occupied {
        if a >= end {
            break;
        }
        if b <= cursor {
            continue;
        }
        if cursor < a {
            result.push((cursor, a));
        }
        cursor = cursor.max(b);
    }
    if cursor < end {
        result.push((cursor, end));
    }
    result
}

/// Union occupied ranges, including aliases and adjacent allocations.
pub(crate) fn merge_ranges(mut ranges: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    ranges.sort_unstable();
    let mut merged = Vec::<(u32, u32)>::new();
    for (start, end) in ranges {
        if let Some((_, previous_end)) = merged.last_mut()
            && start <= *previous_end
        {
            *previous_end = (*previous_end).max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}

fn align_up(value: u32, alignment: u32, name: &'static str) -> Result<u32, MemoryLayoutError> {
    value
        .checked_next_multiple_of(alignment)
        .ok_or(MemoryLayoutError::Invalid(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn randomized_memory_requests_are_aligned_bounded_and_disjoint() {
        let mut random = fastrand::Rng::with_seed(0x6d65_6d6f_7279);
        for _ in 0..128 {
            let mut map = TileMemoryMap::new(Target::Ipu21);
            map.reserve("fixed", 0x50000..0x58000).unwrap();
            for _ in 0..random.usize(1..=24) {
                let alignment = 1 << random.u32(2..=14);
                let bytes = 4 * random.u32(1..=1024);
                let result = map.allocate(MemoryRequest {
                    name: "random",
                    bytes,
                    alignment,
                    bounds: ipu_target::ipu21::memory::TILE_MEMORY_BASE
                        ..ipu_target::ipu21::memory::TILE_MEMORY_BASE
                            + ipu_target::ipu21::memory::TILE_MEMORY_SIZE,
                    end_alignment: 1 << random.u32(0..=14),
                    guard_after: random.u32(0..=64),
                });
                if let Ok(allocation) = result {
                    assert!(allocation.range.start.is_multiple_of(alignment));
                    assert_eq!(allocation.range.len(), bytes as usize);
                }
            }
            let mut ranges = map
                .allocations
                .iter()
                .map(|allocation| allocation.reserved.clone())
                .collect::<Vec<_>>();
            ranges.sort_by_key(|range| range.start);
            assert!(ranges.windows(2).all(|pair| pair[0].end <= pair[1].start));
            assert!(map.free.windows(2).all(|pair| pair[0].end < pair[1].start));
            ranges.extend(map.free.clone());
            ranges.sort_by_key(|range| range.start);
            assert_eq!(
                ranges.first().unwrap().start,
                ipu_target::ipu21::memory::TILE_MEMORY_BASE
            );
            assert_eq!(
                ranges.last().unwrap().end,
                ipu_target::ipu21::memory::TILE_MEMORY_BASE
                    + ipu_target::ipu21::memory::TILE_MEMORY_SIZE
            );
            assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
        }
    }
}
