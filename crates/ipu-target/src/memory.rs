//! IPU21 tile-memory geometry.

pub const TILE_MEMORY_BASE: u32 = 0x4c000;
pub const TILE_MEMORY_SIZE: u32 = 624 * 1024;
/// IPU21 `TMEM_ELEMSIZE`. Instruction fetch and data access contend at this
/// granularity even when placement policy is supplied by another crate.
pub const TILE_MEMORY_ELEMENT_SIZE: u32 = 0x4000;
/// Maximum supervisor instruction-fetch lookahead used when checking whether
/// executable and data ranges share a memory element.
pub const IPU21_SUPERVISOR_FETCH_LOOKAHEAD: u32 = 8 * 8;
/// End of IPU21 region 0, the only tile-memory region supporting instruction fetch.
pub const IPU21_EXECUTABLE_MEMORY_LIMIT: u32 = 0x80000;
/// First logical address of the commonly used interleaved operand window.
pub const IPU21_INTERLEAVED_MEMORY_BASE: u32 = TILE_MEMORY_BASE + 0x34000;
/// End of the commonly used interleaved operand window.
pub const IPU21_INTERLEAVED_MEMORY_LIMIT: u32 = TILE_MEMORY_BASE + 0x3c000;
/// End of architectural region 1, whose interleave factor is two on IPU21.
pub const IPU21_INTERLEAVED_REGION_LIMIT: u32 = TILE_MEMORY_BASE + TILE_MEMORY_SIZE;
/// Exclusive end of SRAM which the SDK secondary loader can populate.
///
/// The final 0x450 bytes are architectural tile memory, but lie beyond the
/// loader's 643 frames of 992 payload bytes starting at `TILE_MEMORY_BASE +
/// 0x10`. Runtime-loaded packages must place no segment there.
pub const IPU21_APPLICATION_MEMORY_LIMIT: u32 = 0xe7bb0;
/// Logical bytes covered by a pair of physical elements in interleaved region 1.
pub const IPU21_INTERLEAVED_ELEMENT_SIZE: u32 = 2 * TILE_MEMORY_ELEMENT_SIZE;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MemoryElement {
    pub interleaved: bool,
    pub index: u32,
}

/// Iterates the physical SRAM elements touched by a word-addressed span.
pub fn memory_elements_for_words(address: u32, words: u32) -> MemoryElements {
    MemoryElements {
        cursor: address,
        end: address.saturating_add(words.saturating_mul(4)),
    }
}

#[derive(Clone, Debug)]
pub struct MemoryElements {
    cursor: u32,
    end: u32,
}

impl Iterator for MemoryElements {
    type Item = MemoryElement;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.end {
            return None;
        }
        let interleaved = self.cursor >= IPU21_INTERLEAVED_MEMORY_BASE;
        let (base, size) = if interleaved {
            (
                IPU21_INTERLEAVED_MEMORY_BASE,
                IPU21_INTERLEAVED_ELEMENT_SIZE,
            )
        } else {
            (0, TILE_MEMORY_ELEMENT_SIZE)
        };
        let index = (self.cursor - base) / size;
        let boundary = base.saturating_add((index + 1).saturating_mul(size));
        self.cursor = boundary.min(self.end);
        Some(MemoryElement { interleaved, index })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn randomized_spans_visit_exactly_their_physical_elements() {
        let mut random = fastrand::Rng::with_seed(0x6d65_6d6f_7279_5f65);
        for _ in 0..512 {
            let address = TILE_MEMORY_BASE + random.u32(0..TILE_MEMORY_SIZE / 4) * 4;
            let available_words = (TILE_MEMORY_BASE + TILE_MEMORY_SIZE - address) / 4;
            let words = random.u32(0..=available_words.min(16_384));
            let actual = memory_elements_for_words(address, words).collect::<BTreeSet<_>>();
            let expected = (0..words)
                .map(|word| {
                    let address = address + word * 4;
                    if address >= IPU21_INTERLEAVED_MEMORY_BASE {
                        MemoryElement {
                            interleaved: true,
                            index: (address - IPU21_INTERLEAVED_MEMORY_BASE)
                                / IPU21_INTERLEAVED_ELEMENT_SIZE,
                        }
                    } else {
                        MemoryElement {
                            interleaved: false,
                            index: address / TILE_MEMORY_ELEMENT_SIZE,
                        }
                    }
                })
                .collect::<BTreeSet<_>>();
            assert_eq!(actual, expected);
        }
    }
}
