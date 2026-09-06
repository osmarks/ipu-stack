//! Address-only shortlist score; exact exchange scheduling decides acceptance.

use super::*;
use crate::{ByteSpan, view_byte_spans};

struct Access {
    shard: BlockValueId,
    span: ByteSpan,
}

#[derive(Default)]
struct Traffic {
    sends: Vec<Access>,
    receives: Vec<Access>,
}

pub(crate) struct ExchangeConflicts(Vec<Traffic>);

impl ExchangeConflicts {
    pub(crate) fn new(program: &LowProgram) -> Result<Self, StorageError> {
        let mut traffic = Vec::new();
        for phase in &program.exchange_phases {
            let mut tiles: Vec<Traffic> = (0..program.tile_count)
                .map(|_| Traffic::default())
                .collect();
            for transfer in &phase.transfers {
                for (view, sending) in std::iter::once((&transfer.source, true))
                    .chain(transfer.destinations.iter().map(|view| (view, false)))
                {
                    let shard = &program.shards[view.shard.index() as usize];
                    let tile = &mut tiles[usize::from(shard.tile)];
                    let accesses = if sending {
                        &mut tile.sends
                    } else {
                        &mut tile.receives
                    };
                    accesses.extend(
                        view_byte_spans(shard, view)?
                            .into_iter()
                            .map(|span| Access {
                                shard: view.shard,
                                span,
                            }),
                    );
                }
            }
            traffic.extend(
                tiles
                    .into_iter()
                    .filter(|tile| !tile.sends.is_empty() && !tile.receives.is_empty()),
            );
        }
        Ok(Self(traffic))
    }

    /// Product of send/receive payload sizes sharing an SRAM element. This
    /// estimates contention under independent arrival times, not schedule
    /// length. Pairing, multicast coupling and actual ordering are resolved
    /// only for finalists. Count a pair once even if it shares several elements.
    pub(crate) fn score(&self, placement: &Placement) -> u128 {
        let aggregate = |accesses: &[Access]| {
            let mut grouped = BTreeMap::<Vec<crate::exchange::ExchangeMemoryElement>, u128>::new();
            for access in accesses {
                let address = placement.shard_addresses[&access.shard] + access.span.offset;
                let elements = crate::exchange::effective_memory_elements(
                    address,
                    access.span.bytes.div_ceil(4),
                );
                *grouped.entry(elements).or_default() += u128::from(access.span.bytes);
            }
            grouped
        };
        self.0
            .iter()
            .map(|tile| {
                // Many fragments touch the same element set. Aggregate first to
                // avoid quadratic work in the number of transfer fragments.
                let sends = aggregate(&tile.sends);
                let receives = aggregate(&tile.receives);
                sends
                    .iter()
                    .map(|(elements, bytes)| {
                        receives
                            .iter()
                            .filter(|(other, _)| {
                                elements.iter().any(|element| other.contains(element))
                            })
                            .map(|(_, received)| bytes * received)
                            .sum::<u128>()
                    })
                    .sum::<u128>()
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relocation_removes_a_weight_broadcast_bank_conflict() {
        let source = BlockValueId::from_index(0);
        let destination = BlockValueId::from_index(1);
        let conflicts = ExchangeConflicts(vec![Traffic {
            sends: vec![Access {
                shard: source,
                span: ByteSpan {
                    offset: 0,
                    bytes: 4148 * 4,
                },
            }],
            receives: vec![Access {
                shard: destination,
                span: ByteSpan {
                    offset: 0,
                    bytes: 4148 * 4,
                },
            }],
        }]);
        let placement = |shift: u32| Placement {
            shard_addresses: BTreeMap::from([
                (source, 0x80000 + shift),
                (destination, 0x87d40 + shift),
            ]),
            tile_auxiliary_ranges: Vec::new(),
        };
        assert_eq!(conflicts.score(&placement(0)), (4148_u128 * 4).pow(2));
        assert_eq!(conflicts.score(&placement(4096)), 0);
        // Both accesses can cross multiple elements; charge the pair once.
        assert_eq!(conflicts.score(&placement(24576)), (4148_u128 * 4).pow(2));
    }
}
