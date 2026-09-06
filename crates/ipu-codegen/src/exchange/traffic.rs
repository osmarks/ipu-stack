//! Resource-load model for tile placement; no timed rows or schedules.
use super::*;

pub(crate) struct MappingTraffic {
    phases: Vec<Vec<PendingTransfer>>,
    tile_count: usize,
}

impl MappingTraffic {
    pub(crate) fn new(
        program: &LowProgram,
        placement: &Placement,
    ) -> Result<Self, ExchangeLoweringError> {
        let phases = program
            .exchange_phases
            .iter()
            .map(|phase| {
                let mut transfers = Vec::new();
                for transfer in &phase.transfers {
                    transfers.extend(prepare_transfer(program, placement, transfer)?);
                }
                let mut transfers = coalesce_pending_transfers(transfers);
                for transfer in &mut transfers {
                    transfer.destinations.sort_unstable();
                }
                Ok(transfers)
            })
            .collect::<Result<_, ExchangeLoweringError>>()?;
        Ok(Self {
            phases,
            tile_count: usize::from(program.tile_count),
        })
    }

    /// Bottleneck cycles, then load-weighted mean pressure as a balancing tie-break.
    /// Active C600 execution indices pair adjacent tiles. Width eligibility is
    /// optimistic about route encoding; exact scheduling verifies finalists.
    pub(crate) fn score(&self, mapping: &[u16], multiplicities: &[u64]) -> (u64, u128) {
        self.phase_scores(mapping)
            .into_iter()
            .zip(multiplicities)
            .fold((0u64, 0u128), |total, ((cycles, pressure), &count)| {
                (
                    total.0.saturating_add(cycles.saturating_mul(count)),
                    total
                        .1
                        .saturating_add(pressure.saturating_mul(u128::from(count))),
                )
            })
    }

    pub(crate) fn phase_cycles(&self, mapping: &[u16]) -> Vec<u64> {
        self.phase_scores(mapping)
            .into_iter()
            .map(|score| {
                score
                    .0
                    .saturating_add(crate::IPU21_TARGET_COSTS.exchange_phase_cycles)
            })
            .collect()
    }

    fn phase_scores(&self, mapping: &[u16]) -> Vec<(u64, u128)> {
        let mut inverse = vec![0u16; self.tile_count];
        for (old, &new) in mapping.iter().enumerate() {
            inverse[usize::from(new)] = old as u16;
        }
        self.phases
            .iter()
            .map(|phase| {
                let score = |pairing: bool| {
                    let mut buses = vec![0u64; self.tile_count.div_ceil(2)];
                    let mut receives = vec![0u64; self.tile_count];
                    let mut elements = BTreeMap::<(u16, ExchangeMemoryElement), u64>::new();
                    for transfer in phase {
                        let source = mapping[usize::from(transfer.source)];
                        let partner = source ^ 1;
                        let paired = pairing
                            && usize::from(partner) < self.tile_count
                            && transfer.words >= 128
                            && transfer.words.is_multiple_of(2)
                            && transfer
                                .source_addresses
                                .iter()
                                .all(|address| address.is_multiple_of(8))
                            && transfer.destinations.iter().all(|&(tile, address)| {
                                let mapped = mapping[usize::from(tile)];
                                mapped != partner
                                    && usize::from(mapped ^ 1) < self.tile_count
                                    && address.is_multiple_of(8)
                                    && transfer
                                        .destinations
                                        .binary_search(&(inverse[usize::from(mapped ^ 1)], address))
                                        .is_ok()
                            });
                        let cycles = u64::from(transfer.words) / if paired { 2 } else { 1 };
                        buses[usize::from(source / 2)] += cycles;
                        for element in &transfer.source_elements {
                            *elements.entry((source, *element)).or_default() += cycles;
                        }
                        for &(tile, address) in &transfer.destinations {
                            let tile = mapping[usize::from(tile)];
                            receives[usize::from(tile)] += cycles;
                            for element in effective_memory_elements(address, transfer.words) {
                                *elements.entry((tile, element)).or_default() += cycles;
                            }
                        }
                    }
                    let maximum = buses
                        .iter()
                        .chain(&receives)
                        .chain(elements.values())
                        .copied()
                        .max()
                        .unwrap_or(0);
                    let sum = buses
                        .iter()
                        .chain(&receives)
                        .map(|&load| u128::from(load))
                        .sum::<u128>();
                    let squares = buses
                        .iter()
                        .chain(&receives)
                        .map(|&load| u128::from(load).pow(2))
                        .sum::<u128>();
                    (maximum, squares / sum.max(1))
                };
                score(false).min(score(true))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transfer(source: u16, destinations: &[u16], words: u32) -> PendingTransfer {
        PendingTransfer {
            source,
            source_shard: BlockValueId::from_index(u32::from(source)),
            source_offset: 0,
            destinations: destinations.iter().map(|&tile| (tile, 0x50000)).collect(),
            source_addresses: vec![0x40000],
            source_elements: effective_memory_elements(0x40000, words),
            words,
            width: ExchangeItemWidth::Word32,
            reserved_source: None,
        }
    }

    #[test]
    fn mapping_prices_shared_buses_and_preserves_paired_receive_opportunities() {
        let traffic = MappingTraffic {
            phases: vec![vec![transfer(0, &[2], 100), transfer(1, &[3], 100)]],
            tile_count: 4,
        };
        assert_eq!(traffic.score(&[0, 1, 2, 3], &[1]).0, 200);
        assert_eq!(traffic.score(&[0, 2, 1, 3], &[1]).0, 100);
        assert_eq!(traffic.score(&[0, 2, 1, 3], &[3]).0, 300);
        let multicast = MappingTraffic {
            phases: vec![vec![transfer(0, &[2, 3], 128)]],
            tile_count: 4,
        };
        assert_eq!(multicast.score(&[0, 1, 2, 3], &[1]).0, 64);
        assert_eq!(multicast.score(&[0, 2, 1, 3], &[1]).0, 128);
    }
}
