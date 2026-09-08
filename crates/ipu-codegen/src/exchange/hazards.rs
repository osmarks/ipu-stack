//! Per-memory-element interval queries for simultaneous send/receive hazards.
use super::ExchangeMemoryElement;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default)]
pub(super) struct MemoryHistory {
    elements: BTreeMap<ExchangeMemoryElement, Vec<Interval>>,
}

#[derive(Clone, Debug)]
struct Interval {
    start: u32,
    end: u32,
    prefix_end: u32,
}

impl MemoryHistory {
    pub(super) fn record(&mut self, elements: &[ExchangeMemoryElement], start: u32, end: u32) {
        for element in elements {
            let intervals = self.elements.entry(*element).or_default();
            let index = intervals.partition_point(|interval| interval.start <= start);
            let mut prefix_end = intervals
                .get(index.wrapping_sub(1))
                .map_or(0, |i| i.prefix_end);
            intervals.insert(
                index,
                Interval {
                    start,
                    end,
                    prefix_end: 0,
                },
            );
            // Usually an append. Retain exact queries even when a scheduler
            // inserts an earlier transfer or intervals overlap/nest.
            for interval in &mut intervals[index..] {
                prefix_end = prefix_end.max(interval.end);
                interval.prefix_end = prefix_end;
            }
        }
    }

    pub(super) fn conflict_end(
        &self,
        elements: &[ExchangeMemoryElement],
        start: u32,
        end: u32,
    ) -> Option<u32> {
        elements
            .iter()
            .filter_map(|element| {
                let intervals = self.elements.get(element)?;
                let count = intervals.partition_point(|interval| interval.start < end);
                let latest_end = intervals.get(count.wrapping_sub(1))?.prefix_end;
                (latest_end > start).then_some(latest_end)
            })
            .max()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_hazards_match_exhaustive_queries_after_arbitrary_insertions() {
        let mut random = fastrand::Rng::with_seed(0x6861_7a61_7264);
        let elements = (0..8)
            .map(|index| ExchangeMemoryElement {
                interleaved: index % 2 == 0,
                index,
            })
            .collect::<Vec<_>>();
        let mut history = MemoryHistory::default();
        let mut reference = Vec::new();
        for _ in 0..2048 {
            let selected = elements
                .iter()
                .copied()
                .filter(|_| random.bool())
                .collect::<Vec<_>>();
            let start = random.u32(0..4096);
            let end = start + random.u32(1..512);
            history.record(&selected, start, end);
            reference.push((selected, start, end));
            for _ in 0..8 {
                let selected = elements
                    .iter()
                    .copied()
                    .filter(|_| random.bool())
                    .collect::<Vec<_>>();
                let start = random.u32(0..4608);
                let end = start + random.u32(1..512);
                let expected = reference
                    .iter()
                    .filter(|(elements, before, after)| {
                        *before < end
                            && start < *after
                            && elements.iter().any(|element| selected.contains(element))
                    })
                    .map(|(_, _, after)| *after)
                    .max();
                assert_eq!(history.conflict_end(&selected, start, end), expected);
            }
        }
        let mut history = MemoryHistory::default();
        history.record(&elements, 10, 20);
        history.record(&elements, 20, 30);
        assert_eq!(history.conflict_end(&elements, 0, 10), None);
        assert_eq!(history.conflict_end(&elements, 15, 20), Some(20));
        assert_eq!(history.conflict_end(&elements, 30, 40), None);
    }
}
