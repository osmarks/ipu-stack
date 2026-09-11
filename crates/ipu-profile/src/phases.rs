//! Compute preparation grouped by its next global exchange, not local step IDs.
use super::*;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PhaseWork {
    pub epoch: u32,
    pub next_exchange_phase: Option<u32>,
    pub start: u64,
    pub end: u64,
    pub span_cycles: u64,
    pub work_cycles: u64,
    pub active_tiles: usize,
    pub occupancy: f64,
    /// Gap between the preceding phase's latest exit and this phase's latest
    /// recorded entry. Preparation outside this gap overlaps prior exchange.
    /// This excludes the cost of retaining a barrier: hidden preparation can
    /// still prevent adjacent exchanges from fusing. It is not a removal saving.
    pub exposed_preparation_cycles: Option<u64>,
    pub kernels: Vec<KernelWork>,
    pub exchange: Option<ExchangeBoundary>,
    /// Scheduled TX/RX coverage; excludes idle time and partner reservation.
    pub exchange_occupancy: Option<f64>,
    pub late_tile_kernels: Vec<KernelWork>,
    /// Time on the last-arriving tile outside detailed compute samples since
    /// its preceding exchange. Includes setup/patching, not necessarily a stall.
    pub unattributed_before_barrier_cycles: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelWork {
    pub operation: String,
    pub kernel: String,
    pub samples: usize,
    pub tiles: usize,
    pub work_cycles: u64,
    pub maximum_cycles: u64,
    pub start: u64,
    pub end: u64,
}

#[derive(Default)]
struct Work<'a> {
    samples: Vec<(u32, u64, u64, &'a CycleSample)>,
    last_compute: BTreeMap<u32, u64>,
    preceding_exchange_end: BTreeMap<u32, u64>,
    preceding_boundaries: BTreeSet<(u32, Option<u32>)>,
    exchange_work: u64,
    exchange_described: bool,
}

fn kernels(samples: &[(u32, u64, u64, &CycleSample)]) -> Vec<KernelWork> {
    let mut groups = BTreeMap::new();
    for &(tile, start, end, sample) in samples {
        let key = (&sample.step.operation, &sample.step.kernel);
        let (work, tiles) = groups.entry(key).or_insert_with(|| {
            (
                KernelWork {
                    operation: key.0.clone(),
                    kernel: key.1.clone(),
                    samples: 0,
                    tiles: 0,
                    work_cycles: 0,
                    maximum_cycles: 0,
                    start,
                    end,
                },
                BTreeSet::new(),
            )
        });
        work.samples += 1;
        work.work_cycles += end - start;
        work.maximum_cycles = work.maximum_cycles.max(end - start);
        work.start = work.start.min(start);
        work.end = work.end.max(end);
        tiles.insert(tile);
    }
    let mut groups = groups
        .into_values()
        .map(|(mut work, tiles)| {
            work.tiles = tiles.len();
            work
        })
        .collect::<Vec<_>>();
    groups.sort_by_key(|g| (std::cmp::Reverse(g.work_cycles), g.start));
    groups
}

/// Spans may overlap while some tiles compute and others finish the previous
/// exchange. Occupancy is measured compute work / (span * device tile count),
/// not useful-work efficiency. Empty preparation rounds are retained. Opaque
/// unprofiled Repeat remainders are excluded and break preparation grouping.
pub fn phase_work(report: &ProfileReport, shared_clock: bool) -> Vec<PhaseWork> {
    let base = cycle_origin(report);
    let crop = if shared_clock {
        0
    } else {
        initial_sample_entry_span(report, base)
    };
    let mut boundaries = BTreeMap::new();
    let mut work = BTreeMap::new();
    for mut boundary in exchange_boundaries(report) {
        let key = (boundary.epoch, Some(boundary.phase));
        boundary.first_entry = boundary.first_entry.saturating_sub(crop);
        boundary.last_entry = boundary.last_entry.saturating_sub(crop);
        boundary.last_exit = boundary.last_exit.saturating_sub(crop);
        boundary.arrival_spread_cycles = boundary.last_entry - boundary.first_entry;
        boundary.after_last_arrival_cycles = boundary.last_exit - boundary.last_entry;
        boundaries.insert(key, boundary);
        work.insert(key, Work::default());
    }
    for tile in &report.tiles {
        let mut next = None;
        for sample in tile.samples.iter().rev() {
            let start = u64::from(sample.start_cycle.wrapping_sub(base));
            let end = (start + u64::from(duration(sample))).saturating_sub(crop);
            let start = start.saturating_sub(crop);
            if is_exchange_boundary(sample) {
                if let Some(key) = next {
                    let group = work.entry(key).or_default();
                    group.preceding_exchange_end.insert(tile.physical_tile, end);
                    group
                        .preceding_boundaries
                        .insert((sample.step.epoch, Some(sample.step.phase)));
                }
                let key = (sample.step.epoch, Some(sample.step.phase));
                next = Some(key);
                let group = work.entry(key).or_default();
                let horizon = u64::from(sample.step.exchange_event_cycles);
                if horizon != 0 {
                    let roles = exchange_role_cycles(&sample.step.exchange_activities, horizon);
                    group.exchange_work += roles.send + roles.receive - roles.simultaneous;
                    group.exchange_described = true;
                }
            } else if sample.step.kernel == "repeat-remainder" {
                // Do not attribute the previous profiled iteration to a later
                // outer-graph barrier across an opaque sequence of iterations.
                next = None;
            } else if sample.step.kind == ProfileStepKind::Compute
                && !sample
                    .step
                    .metadata
                    .iter()
                    .any(|m| m.name == "active" && m.value == "false")
                && start < end
            {
                let key = next.unwrap_or((sample.step.epoch, None));
                let group = work.entry(key).or_default();
                group.last_compute.entry(tile.physical_tile).or_insert(end);
                group.samples.push((tile.physical_tile, start, end, sample));
            }
        }
    }
    let mut result = work
        .into_iter()
        .map(|(key, work)| {
            let boundary = boundaries.get(&key).cloned();
            let fallback = boundary.as_ref().map_or(0, |b| b.last_entry);
            let start = work.samples.iter().map(|s| s.1).min().unwrap_or(fallback);
            let end = work.samples.iter().map(|s| s.2).max().unwrap_or(fallback);
            let span = end - start;
            let cycles = work.samples.iter().map(|s| s.2 - s.1).sum::<u64>();
            let late = boundary.as_ref().map(|b| b.last_arriving_tile);
            let late_samples = work
                .samples
                .iter()
                .copied()
                .filter(|s| Some(s.0) == late)
                .collect::<Vec<_>>();
            let unattributed = boundary
                .as_ref()
                .and_then(|b| {
                    work.last_compute
                        .get(&b.last_arriving_tile)
                        .or_else(|| work.preceding_exchange_end.get(&b.last_arriving_tile))
                        .map(|end| b.last_entry.saturating_sub(*end))
                })
                .unwrap_or(0);
            let exchange_occupancy = boundary
                .as_ref()
                .and_then(|b| b.scheduled_event_cycles)
                .filter(|&n| n != 0 && work.exchange_described && !report.tiles.is_empty())
                .map(|n| work.exchange_work as f64 / (n as f64 * report.tiles.len() as f64));
            let exposed_preparation_cycles = boundary.as_ref().and_then(|b| {
                if work.preceding_boundaries.len() != 1 {
                    return None;
                }
                let previous = boundaries.get(work.preceding_boundaries.first()?)?;
                Some(b.last_entry.saturating_sub(previous.last_exit))
            });
            PhaseWork {
                epoch: key.0,
                next_exchange_phase: key.1,
                start,
                end,
                span_cycles: span,
                work_cycles: cycles,
                active_tiles: work.last_compute.len(),
                occupancy: if span == 0 || report.tiles.is_empty() {
                    0.0
                } else {
                    cycles as f64 / (span as f64 * report.tiles.len() as f64)
                },
                exposed_preparation_cycles,
                kernels: kernels(&work.samples),
                exchange: boundary,
                exchange_occupancy,
                late_tile_kernels: kernels(&late_samples),
                unattributed_before_barrier_cycles: unattributed,
            }
        })
        .collect::<Vec<_>>();
    result.sort_by_key(|g| (g.start, g.epoch, g.next_exchange_phase));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::sample;
    use ipu_package::TileProfile;

    #[test]
    fn groups_different_local_indices_at_the_same_barrier() {
        let c = ProfileStepKind::Compute;
        let e = ProfileStepKind::Exchange;
        let report = ProfileReport {
            clock_hz: 1,
            tiles: vec![
                TileProfile {
                    physical_tile: 0,
                    samples: vec![
                        sample(9, c, "copy", 0, 10),
                        sample(80, e, "exchange", 12, 20),
                    ],
                },
                TileProfile {
                    physical_tile: 1,
                    samples: vec![
                        sample(44, c, "fill", 2, 6),
                        sample(80, e, "exchange", 9, 20),
                    ],
                },
                TileProfile {
                    physical_tile: 2,
                    samples: vec![sample(80, e, "exchange", 0, 20)],
                },
            ],
        };
        let groups = phase_work(&report, true);
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert_eq!((g.span_cycles, g.work_cycles, g.active_tiles), (10, 14, 2));
        assert_eq!(g.occupancy, 14.0 / 30.0);
        assert_eq!(g.late_tile_kernels[0].kernel, "copy");
        assert_eq!(g.unattributed_before_barrier_cycles, 2);
        let cropped = phase_work(&report, false);
        assert_eq!(
            (cropped[0].start, cropped[0].end, cropped[0].work_cycles),
            (0, 8, 12)
        );
    }

    #[test]
    fn synchronization_only_tiles_can_determine_barrier_arrival() {
        let mut exchange = sample(80, ProfileStepKind::Exchange, "exchange", 10, 100);
        exchange.step.exchange_event_cycles = 30;
        let mut sync = sample(80, ProfileStepKind::Synchronization, "sync", 60, 70);
        sync.step.metadata.push(ipu_package::ProfileMetadata {
            name: "reason".into(),
            value: "ExchangeBarrier".into(),
        });
        let report = ProfileReport {
            clock_hz: 1,
            tiles: vec![
                TileProfile {
                    physical_tile: 0,
                    samples: vec![exchange],
                },
                TileProfile {
                    physical_tile: 1,
                    samples: vec![
                        sample(2, ProfileStepKind::Compute, "copy", 0, 60),
                        sync,
                        sample(81, ProfileStepKind::Exchange, "exchange", 80, 120),
                    ],
                },
            ],
        };
        let groups = phase_work(&report, true);
        let g = groups
            .iter()
            .find(|g| g.next_exchange_phase == Some(80))
            .unwrap();
        assert_eq!(g.exchange.as_ref().unwrap().last_entry, 60);
        assert_eq!(g.exchange.as_ref().unwrap().last_arriving_tile, 1);
        assert_eq!(g.late_tile_kernels[0].kernel, "copy");
        assert_eq!(g.work_cycles, 60);
        assert_eq!(
            groups
                .iter()
                .find(|g| g.next_exchange_phase == Some(81))
                .unwrap()
                .work_cycles,
            0
        );
    }

    #[test]
    fn preserves_empty_rounds_epochs_and_opaque_repeat_boundaries() {
        let c = ProfileStepKind::Compute;
        let e = ProfileStepKind::Exchange;
        let mut inner = sample(9, e, "exchange", 10, 20);
        inner.step.epoch = 1;
        let mut end = sample(7, c, "copy", 20, 24);
        end.step.epoch = 1;
        let report = ProfileReport {
            clock_hz: 1,
            tiles: vec![TileProfile {
                physical_tile: 0,
                samples: vec![
                    sample(9, e, "exchange", 0, 10),
                    inner,
                    end,
                    sample(4, c, "repeat-remainder", 24, 1000),
                    sample(5, c, "norm", 1000, 1010),
                    sample(10, e, "exchange", 1012, 1020),
                ],
            }],
        };
        let groups = phase_work(&report, true);
        assert_eq!(groups.len(), 4);
        assert_eq!(groups.iter().map(|g| g.work_cycles).sum::<u64>(), 14);
        assert_eq!(
            groups
                .iter()
                .filter(|g| g.next_exchange_phase == Some(9))
                .count(),
            2
        );
        let outer = groups
            .iter()
            .find(|g| g.next_exchange_phase == Some(10))
            .unwrap();
        assert_eq!(outer.kernels.len(), 1);
        assert_eq!(outer.kernels[0].kernel, "norm");
        assert!(
            groups
                .iter()
                .any(|g| g.epoch == 1 && g.next_exchange_phase.is_none())
        );
    }
}
