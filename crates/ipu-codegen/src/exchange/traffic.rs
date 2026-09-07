//! Resource-load model for tile placement; no timed rows or schedules.
use super::*;

pub(crate) struct MappingTraffic {
    phases: Vec<MappingPhase>,
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
                Ok(MappingPhase::new(
                    transfers,
                    usize::from(program.tile_count),
                ))
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
            .map(|phase| phase.score(mapping, &inverse))
            .collect()
    }
}

/// Resource indices use original tile IDs: a bijective mapping only renames
/// resources, preserving their maximum load and the ordinary-transfer score.
struct MappingPhase {
    transfers: Vec<MappingTransfer>,
    loads: Vec<u64>,
    tile_count: usize,
    ordinary_score: (u64, u128),
}

struct MappingTransfer {
    source: u16,
    destinations: Vec<(u16, u32)>,
    words: u32,
    resources: Vec<usize>,
}

impl MappingPhase {
    fn new(transfers: Vec<PendingTransfer>, tile_count: usize) -> Self {
        let mut loads = vec![0; tile_count * 2];
        let mut elements = BTreeMap::new();
        let mut pairable = Vec::new();
        for transfer in transfers {
            let mut resources = vec![usize::from(transfer.source)];
            let mut add_element = |tile, element| {
                resources.push(*elements.entry((tile, element)).or_insert_with(|| {
                    loads.push(0);
                    loads.len() - 1
                }));
            };
            for &element in &transfer.source_elements {
                add_element(transfer.source, element);
            }
            for &(tile, address) in &transfer.destinations {
                for element in effective_memory_elements(address, transfer.words) {
                    add_element(tile, element);
                }
            }
            resources.extend(
                transfer
                    .destinations
                    .iter()
                    .map(|&(tile, _)| tile_count + usize::from(tile)),
            );
            for &resource in &resources {
                loads[resource] += u64::from(transfer.words);
            }
            if transfer.words >= 128
                && transfer.words.is_multiple_of(2)
                && transfer
                    .source_addresses
                    .iter()
                    .all(|address| address.is_multiple_of(8))
                && transfer
                    .destinations
                    .iter()
                    .all(|&(_, address)| address.is_multiple_of(8))
            {
                pairable.push(MappingTransfer {
                    source: transfer.source,
                    destinations: transfer.destinations,
                    words: transfer.words,
                    resources,
                });
            }
        }
        let ordinary_score = Self::load_score(&loads, tile_count);
        Self {
            transfers: pairable,
            loads,
            tile_count,
            ordinary_score,
        }
    }

    fn load_score(loads: &[u64], tile_count: usize) -> (u64, u128) {
        let maximum = loads.iter().copied().max().unwrap_or(0);
        let lanes = &loads[..tile_count * 2];
        let sum = lanes.iter().map(|&load| u128::from(load)).sum::<u128>();
        let squares = lanes
            .iter()
            .map(|&load| u128::from(load).pow(2))
            .sum::<u128>();
        (maximum, squares / sum.max(1))
    }

    fn score(&self, mapping: &[u16], inverse: &[u16]) -> (u64, u128) {
        let mut loads = self.loads.clone();
        for transfer in &self.transfers {
            let partner = mapping[usize::from(transfer.source)] ^ 1;
            let paired = usize::from(partner) < self.tile_count
                && transfer.destinations.iter().all(|&(tile, address)| {
                    let mapped = mapping[usize::from(tile)];
                    mapped != partner
                        && usize::from(mapped ^ 1) < self.tile_count
                        && transfer
                            .destinations
                            .binary_search(&(inverse[usize::from(mapped ^ 1)], address))
                            .is_ok()
                });
            if paired {
                let saving = u64::from(transfer.words) / 2;
                for &resource in &transfer.resources {
                    loads[resource] -= saving;
                }
                loads[usize::from(inverse[usize::from(partner)])] += saving;
            }
        }
        self.ordinary_score
            .min(Self::load_score(&loads, self.tile_count))
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
    fn resource_scores_preserve_all_four_tile_pairings() {
        let mut transfers = Vec::new();
        for source in 0..4 {
            for mask in 1u16..16 {
                let destinations = (0..4)
                    .filter(|tile| mask & (1 << tile) != 0)
                    .collect::<Vec<_>>();
                for (address, words) in [
                    (0x50000, 100),
                    (0x50000, 128),
                    (0x53ff8, 256),
                    (0x50004, 256),
                    (0x90000, 512),
                ] {
                    let mut transfer =
                        transfer(source, &destinations, words + u32::from(source * mask * 2));
                    for (_, destination) in &mut transfer.destinations {
                        *destination = address;
                    }
                    transfers.push(transfer);
                }
            }
        }
        let traffic = MappingTraffic {
            phases: vec![MappingPhase::new(transfers, 4)],
            tile_count: 4,
        };
        // Reference scores from direct per-transfer resource accounting. All
        // 24 permutations cover the three distinct physical partner assignments.
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        let mapping = [a, b, c, d];
                        let mut sorted = mapping;
                        sorted.sort_unstable();
                        if sorted == [0, 1, 2, 3] {
                            assert_eq!(
                                traffic.phase_scores(&mapping),
                                vec![match mapping
                                    .iter()
                                    .position(|&tile| tile == mapping[0] ^ 1)
                                    .unwrap()
                                {
                                    1 => (44652, 36274),
                                    2 => (44558, 36201),
                                    3 => (44539, 36195),
                                    _ => unreachable!(),
                                }],
                                "{mapping:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn mapping_prices_independent_lanes_and_borrowed_pairing() {
        let traffic = MappingTraffic {
            phases: vec![MappingPhase::new(
                vec![transfer(0, &[2], 100), transfer(1, &[3], 100)],
                4,
            )],
            tile_count: 4,
        };
        let scheduled = schedule_exchange_problem(
            4,
            &schedule_problem(0, &[transfer(0, &[2], 100), transfer(1, &[3], 100)]),
        )
        .unwrap();
        let send = |tile: usize| {
            scheduled.phase.activities[tile]
                .iter()
                .find(|activity| activity.kind == ExchangeActivityKind::Send)
                .unwrap()
        };
        assert!(
            send(0)
                .end_cycle
                .min(send(1).end_cycle)
                .saturating_sub(send(0).start_cycle.max(send(1).start_cycle))
                > 50
        );
        assert_eq!(traffic.score(&[0, 1, 2, 3], &[1]).0, 100);
        assert_eq!(traffic.score(&[0, 2, 1, 3], &[1]).0, 100);
        assert_eq!(traffic.score(&[0, 2, 1, 3], &[3]).0, 300);
        let multicast = MappingTraffic {
            phases: vec![MappingPhase::new(vec![transfer(0, &[2, 3], 128)], 4)],
            tile_count: 4,
        };
        assert_eq!(multicast.score(&[0, 1, 2, 3], &[1]).0, 64);
        assert_eq!(multicast.score(&[0, 2, 1, 3], &[1]).0, 128);
    }
}
