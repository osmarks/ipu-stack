use std::ops::Range;

#[derive(Clone, Debug)]
pub(crate) struct AddressSpace {
    bounds: Range<u32>,
    reserved: Vec<Range<u32>>,
}

impl AddressSpace {
    pub(crate) fn new(bounds: Range<u32>) -> Self {
        Self {
            bounds,
            reserved: Vec::new(),
        }
    }

    pub(crate) fn reserve(&mut self, range: Range<u32>) {
        if range.start < range.end {
            self.reserved.push(range);
        }
    }

    pub(crate) fn reserve_all(&mut self, ranges: impl IntoIterator<Item = (u32, u32)>) {
        for (start, end) in ranges {
            self.reserve(start..end);
        }
    }

    /// Returns free regions after expanding every reservation to the access
    /// granularity. Instruction placement uses a full tile-memory element;
    /// ordinary data placement uses byte granularity.
    pub(crate) fn free_regions(mut self, granularity: u32) -> Vec<(u32, u32)> {
        assert!(granularity.is_power_of_two());
        for range in &mut self.reserved {
            range.start &= !(granularity - 1);
            range.end = align_up(range.end, granularity);
        }
        self.reserved.sort_unstable_by_key(|range| range.start);

        let mut free = Vec::new();
        let mut cursor = align_up(self.bounds.start, granularity);
        for range in self.reserved {
            if range.end <= cursor || range.start >= self.bounds.end {
                continue;
            }
            if cursor < range.start {
                free.push((cursor, range.start.min(self.bounds.end)));
            }
            cursor = cursor.max(range.end);
        }
        if cursor < self.bounds.end {
            free.push((cursor, self.bounds.end));
        }
        free
    }
}

fn align_up(value: u32, alignment: u32) -> u32 {
    value.saturating_add(alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservations_are_clipped_merged_and_granularity_aligned() {
        let mut space = AddressSpace::new(0x1000..0x2000);
        space.reserve_all([
            (0x0800, 0x1101),
            (0x11f0, 0x1300),
            (0x1200, 0x1401),
            (0x1f80, 0x2800),
        ]);

        assert_eq!(space.free_regions(0x100), vec![(0x1500, 0x1f00)]);
    }
}
