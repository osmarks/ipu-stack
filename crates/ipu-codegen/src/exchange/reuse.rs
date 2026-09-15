//! Reuse optimization choices across placement; rebuild and check physical rows.

use super::*;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Clone, Default)]
pub struct ExchangeScheduleCache {
    pub(super) phases: BTreeMap<ExchangePhaseId, Arc<ScheduleRecipe>>,
}

#[derive(Clone)]
pub(super) struct ScheduleRecipe {
    pub(super) stream_words: Option<std::num::NonZeroU32>,
    pub(super) structure: u64,
    pub(super) widths: Vec<ExchangeItemWidth>,
    pub(super) order: Vec<usize>,
    pub(super) rows: Vec<Vec<u32>>,
}

// This is only a cheap rejection filter. Address-dependent dependencies and
// timing are still rebuilt, validated, and compared against normalized rows.
pub(super) fn structure_fingerprint(pending: &[PendingTransfer], tile_count: u16) -> u64 {
    let mut hash = std::hash::DefaultHasher::new();
    tile_count.hash(&mut hash);
    pending.len().hash(&mut hash);
    for transfer in pending {
        transfer.source.hash(&mut hash);
        transfer.words.hash(&mut hash);
        transfer.source_addresses.len().hash(&mut hash);
        transfer.moving_source().hash(&mut hash);
        transfer.destinations.len().hash(&mut hash);
        for &(tile, _) in &transfer.destinations {
            tile.hash(&mut hash);
        }
    }
    hash.finish()
}

pub(super) fn normalized_rows(
    schedule: &MaterializedSchedule,
) -> Result<Vec<Vec<u32>>, ExchangeLoweringError> {
    Ok(schedule
        .builder
        .finish()?
        .programs
        .into_iter()
        .map(|program| {
            program
                .unwrap_or_else(ipu_exchange::EncodedRow::inactive)
                .normalized_words()
        })
        .collect())
}

impl ExchangeScheduleCache {
    pub(super) fn take_phase(&mut self, phase: ExchangePhaseId) -> Self {
        Self {
            phases: self
                .phases
                .remove(&phase)
                .map(|recipe| (phase, recipe))
                .into_iter()
                .collect(),
        }
    }

    pub(super) fn merge(&mut self, other: Self) {
        self.phases.extend(other.phases);
    }
}

impl ScheduleRecipe {
    pub(super) fn replay(
        &self,
        topology: &Topology,
        ordinary: &[PendingTransfer],
        tile_count: u16,
    ) -> Result<Option<ScheduledPending>, ExchangeLoweringError> {
        if ordinary.len() != self.widths.len() {
            return Ok(None);
        }
        let alternatives = paired_transfer_alternatives(ordinary, topology, tile_count)?;
        let Some(pending) = ordinary
            .iter()
            .zip(alternatives)
            .zip(&self.widths)
            .map(|((ordinary, paired), width)| match width {
                ExchangeItemWidth::Word32 => Some(ordinary.clone()),
                ExchangeItemWidth::Paired64 => paired,
            })
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        let (receive_counts, incoming_bases) = receive_configuration(&pending, tile_count)?;
        // Recompute alias dependencies, SRAM-element hazards, receiver setup,
        // repeat-source hazards, and instruction alignment at the final addresses.
        // The original greedy schedule may have needed incremental encoding
        // validation. Replay must use the same fallback before comparing rows.
        let problem = SchedulingProblem::new(&pending, tile_count);
        let schedule = materialize_valid_schedule_order(
            topology,
            &problem,
            &incoming_bases,
            &receive_counts,
            &self.order,
        )?;
        // Identical normalized rows preserve compact table sharing and the
        // provisional code-size reservation, not just the total cycle count.
        if normalized_rows(&schedule)? != self.rows {
            return Ok(None);
        }
        let optimized = OptimizedSchedule {
            initial_horizon: schedule.horizon,
            endpoint_lower_bound: endpoint_work_lower_bound(&pending, tile_count),
            schedule,
            selected_kind: "reused",
            neighborhood_improvements: 0,
        };
        Ok(Some(ScheduledPending {
            pending,
            receive_counts,
            incoming_bases,
            optimized,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transfers() -> Vec<PendingTransfer> {
        pending_from_problem(
            4,
            &ExchangeScheduleProblem {
                phase: 0,
                transfers: (0..2)
                    .map(|source| ExchangeScheduleTransfer {
                        source,
                        source_addresses: vec![0x60000],
                        destinations: vec![ExchangeScheduleDestination {
                            tile: source + 2,
                            address: 0x64000,
                        }],
                        words: 64,
                        width: ExchangeItemWidth::Word32,
                    })
                    .collect(),
            },
        )
        .unwrap()
    }

    #[test]
    fn reuses_rows_after_relocation_and_reoptimizes_changed_transfers() {
        for words in [
            None,
            std::num::NonZeroU32::new(64),
            std::num::NonZeroU32::new(256),
        ] {
            let topology = Topology::c600();
            let phase = ExchangePhaseId::from_index(0);
            let mut cache = ExchangeScheduleCache::default();
            let mut child = cache.take_phase(phase);
            let original = transfers();
            let first =
                select_phase(phase, &topology, original.clone(), 4, words, &mut child).unwrap();
            if words.is_some() {
                assert_eq!(first.optimized.selected_kind, "balanced-compact-streams");
            }
            cache.merge(child);
            let mut relocated = original;
            for transfer in &mut relocated {
                transfer.source_addresses[0] += 0x4000;
                transfer.destinations[0].1 += 0x4000;
                transfer.refresh_source_elements();
            }
            let second =
                select_phase(phase, &topology, relocated.clone(), 4, words, &mut cache).unwrap();
            assert_eq!(second.optimized.selected_kind, "reused");
            assert_eq!(
                normalized_rows(&first.optimized.schedule).unwrap(),
                normalized_rows(&second.optimized.schedule).unwrap()
            );
            assert_eq!(second.pending[0].source_address(), 0x64000);
            relocated[0].words = 32;
            let changed = select_phase(phase, &topology, relocated, 4, words, &mut cache).unwrap();
            assert_ne!(changed.optimized.selected_kind, "reused");
        }
    }

    #[test]
    fn changed_policy_cannot_replay_another_policy_schedule() {
        let topology = Topology::c600();
        let phase = ExchangePhaseId::from_index(0);
        let mut cache = ExchangeScheduleCache::default();
        for words in [
            None,
            std::num::NonZeroU32::new(64),
            std::num::NonZeroU32::new(256),
            None,
        ] {
            let selected =
                select_phase(phase, &topology, transfers(), 4, words, &mut cache).unwrap();
            assert_ne!(selected.optimized.selected_kind, "reused");
            let reused = select_phase(phase, &topology, transfers(), 4, words, &mut cache).unwrap();
            assert_eq!(reused.optimized.selected_kind, "reused");
            assert_eq!(
                normalized_rows(&selected.optimized.schedule).unwrap(),
                normalized_rows(&reused.optimized.schedule).unwrap()
            );
        }
    }

    #[test]
    fn replay_rejects_orders_that_violate_new_alias_dependencies() {
        let topology = Topology::c600();
        let mut pending = transfers();
        let (counts, bases) = receive_configuration(&pending, 4).unwrap();
        assert!(
            materialize_schedule_order(
                &topology,
                &SchedulingProblem::new(&pending, 4),
                &bases,
                &counts,
                &[1, 0],
                false
            )
            .is_ok()
        );
        // Transfer 1 now reads bytes written by transfer 0. The old order is
        // no longer legal even though transfer lengths and tile counts match.
        pending[1].source = pending[0].destinations[0].0;
        pending[1].source_addresses[0] = pending[0].destinations[0].1;
        pending[1].refresh_source_elements();
        let (counts, bases) = receive_configuration(&pending, 4).unwrap();
        assert!(
            materialize_schedule_order(
                &topology,
                &SchedulingProblem::new(&pending, 4),
                &bases,
                &counts,
                &[0, 1],
                false
            )
            .is_ok()
        );
        for order in [[1, 0], [0, 0], [0, 2]] {
            assert!(
                materialize_schedule_order(
                    &topology,
                    &SchedulingProblem::new(&pending, 4),
                    &bases,
                    &counts,
                    &order,
                    false
                )
                .is_err()
            );
        }
    }
    #[test]
    fn replay_preserves_incrementally_aligned_rows() {
        let topology = Topology::c600();
        // Reduced multicast fixture: the fast construction violates SENDPICP
        // alignment; incremental validation produces valid, reusable rows.
        let transfers: [(u16, u32, &[u16], u32); 8] = [
            (3, 3, &[5, 6], 10),
            (0, 7, &[4, 5, 6, 7], 4),
            (2, 8, &[4, 6], 9),
            (2, 9, &[6, 7], 13),
            (0, 10, &[5, 6, 7], 15),
            (2, 12, &[4, 5, 6, 7], 10),
            (1, 17, &[6], 18),
            (2, 18, &[5, 6, 7], 2),
        ];
        let problem = ExchangeScheduleProblem {
            phase: 0,
            transfers: transfers
                .into_iter()
                .map(
                    |(source, offset, destinations, words)| ExchangeScheduleTransfer {
                        source,
                        source_addresses: vec![0x10000 + offset * 128],
                        destinations: destinations
                            .iter()
                            .map(|&tile| ExchangeScheduleDestination {
                                tile,
                                address: 0x40000 + offset * 128,
                            })
                            .collect(),
                        words,
                        width: ExchangeItemWidth::Word32,
                    },
                )
                .collect(),
        };
        let pending = pending_from_problem(8, &problem).unwrap();
        let (counts, bases) = receive_configuration(&pending, 8).unwrap();
        let aligned = materialize_greedy_schedule(
            &topology,
            &SchedulingProblem::new(&pending, 8),
            &bases,
            &counts,
        )
        .unwrap();
        assert!(
            materialize_schedule_order(
                &topology,
                &SchedulingProblem::new(&pending, 8),
                &bases,
                &counts,
                &aligned.order,
                false
            )
            .is_err()
        );
        let recipe = ScheduleRecipe {
            stream_words: None,
            structure: structure_fingerprint(&pending, 8),
            widths: vec![ExchangeItemWidth::Word32; pending.len()],
            order: aligned.order.clone(),
            rows: normalized_rows(&aligned).unwrap(),
        };
        assert!(recipe.replay(&topology, &pending, 8).unwrap().is_some());
    }
}
