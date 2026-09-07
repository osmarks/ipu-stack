//! Reuse optimization choices across placement; rebuild and check physical rows.

use super::*;

#[derive(Clone, Default)]
pub(crate) struct ExchangeScheduleCache {
    phases: BTreeMap<ExchangePhaseId, ScheduleRecipe>,
}

#[derive(Clone)]
struct ScheduleRecipe {
    widths: Vec<ExchangeItemWidth>,
    order: Vec<usize>,
    rows: Vec<Vec<u32>>,
}

fn normalized_rows(
    schedule: &MaterializedSchedule,
) -> Result<Vec<Vec<u32>>, ExchangeLoweringError> {
    Ok(schedule
        .builder
        .finish()?
        .programs
        .into_iter()
        .map(|program| {
            ipu_exchange::normalized_exchange_address_words(
                &program.unwrap_or_else(inactive_exchange_program),
            )
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

    pub(super) fn select(
        &mut self,
        phase: ExchangePhaseId,
        topology: &Topology,
        pending: Vec<PendingTransfer>,
        tile_count: u16,
    ) -> Result<ScheduledPending, ExchangeLoweringError> {
        if let Some(recipe) = self.phases.get(&phase) {
            match recipe.replay(topology, &pending, tile_count) {
                Ok(Some(schedule)) => {
                    tracing::info!(
                        phase = phase.index(),
                        "reused exchange optimization after validating relocated rows"
                    );
                    return Ok(schedule);
                }
                Ok(None) => {}
                Err(error) => {
                    tracing::debug!(phase = phase.index(), %error, "exchange choices require reoptimization after placement")
                }
            }
        }
        let selected = select_transfer_widths(phase.index(), topology, pending, tile_count)?;
        self.phases.insert(
            phase,
            ScheduleRecipe {
                widths: selected
                    .pending
                    .iter()
                    .map(|transfer| transfer.width)
                    .collect(),
                order: selected.optimized.schedule.order.clone(),
                rows: normalized_rows(&selected.optimized.schedule)?,
            },
        );
        Ok(selected)
    }
}

impl ScheduleRecipe {
    fn replay(
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
        let replay = |validate_encoding| {
            materialize_schedule_order(
                topology,
                &pending,
                &incoming_bases,
                &receive_counts,
                tile_count,
                &self.order,
                validate_encoding,
            )
        };
        let schedule = match replay(false) {
            Err(ExchangeLoweringError::Exchange(ipu_exchange::ExchangeError::Schedule(
                "SENDPICP instruction alignment",
            ))) => replay(true)?,
            result => result?,
        };
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
        let topology = Topology::c600();
        let phase = ExchangePhaseId::from_index(0);
        let mut cache = ExchangeScheduleCache::default();
        let original = transfers();
        let first = cache.select(phase, &topology, original.clone(), 4).unwrap();
        let mut relocated = original;
        for transfer in &mut relocated {
            transfer.source_addresses[0] += 0x4000;
            transfer.destinations[0].1 += 0x4000;
            transfer.refresh_source_elements();
        }
        let second = cache
            .select(phase, &topology, relocated.clone(), 4)
            .unwrap();
        assert_eq!(second.optimized.selected_kind, "reused");
        assert_eq!(
            normalized_rows(&first.optimized.schedule).unwrap(),
            normalized_rows(&second.optimized.schedule).unwrap()
        );
        assert_eq!(second.pending[0].source_address(), 0x64000);
        relocated[0].words = 32;
        let changed = cache.select(phase, &topology, relocated, 4).unwrap();
        assert_ne!(changed.optimized.selected_kind, "reused");
    }

    #[test]
    fn replay_rejects_orders_that_violate_new_alias_dependencies() {
        let topology = Topology::c600();
        let mut pending = transfers();
        let (counts, bases) = receive_configuration(&pending, 4).unwrap();
        assert!(
            materialize_schedule_order(&topology, &pending, &bases, &counts, 4, &[1, 0], false)
                .is_ok()
        );
        // Transfer 1 now reads bytes written by transfer 0. The old order is
        // no longer legal even though transfer lengths and tile counts match.
        pending[1].source = pending[0].destinations[0].0;
        pending[1].source_addresses[0] = pending[0].destinations[0].1;
        pending[1].refresh_source_elements();
        let (counts, bases) = receive_configuration(&pending, 4).unwrap();
        assert!(
            materialize_schedule_order(&topology, &pending, &bases, &counts, 4, &[0, 1], false)
                .is_ok()
        );
        for order in [[1, 0], [0, 0], [0, 2]] {
            assert!(
                materialize_schedule_order(&topology, &pending, &bases, &counts, 4, &order, false)
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
        let aligned = materialize_greedy_schedule(&topology, &pending, &bases, &counts, 8).unwrap();
        assert!(
            materialize_schedule_order(
                &topology,
                &pending,
                &bases,
                &counts,
                8,
                &aligned.order,
                false
            )
            .is_err()
        );
        let recipe = ScheduleRecipe {
            widths: vec![ExchangeItemWidth::Word32; pending.len()],
            order: aligned.order.clone(),
            rows: normalized_rows(&aligned).unwrap(),
        };
        assert!(recipe.replay(&topology, &pending, 8).unwrap().is_some());
    }
}
