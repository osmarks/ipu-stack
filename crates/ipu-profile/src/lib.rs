use ipu_package::{CycleSample, ProfileReport, ProfileStepKind};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GroupBy {
    Kind,
    #[default]
    Kernel,
    Operation,
    Phase,
    Tile,
    Metadata,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SortBy {
    #[default]
    PhaseCycles,
    WorkCycles,
    MaximumCycles,
    Samples,
    Name,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepKind {
    Exchange,
    Compute,
    Synchronization,
    Idle,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetadataFilter {
    pub name: String,
    pub value: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Query {
    pub group_by: GroupBy,
    pub sort_by: SortBy,
    pub kind: Option<StepKind>,
    pub kernel: Option<String>,
    pub operation_contains: Option<String>,
    pub tiles: BTreeSet<u32>,
    pub phases: BTreeSet<u32>,
    pub metadata: Vec<MetadataFilter>,
    pub metadata_key: Option<String>,
    pub at_offset: Option<u64>,
    pub limit: Option<usize>,
    pub sample_limit: usize,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct QueryReport {
    pub clock_hz: u64,
    pub tile_count: usize,
    pub sample_count: usize,
    pub matched_sample_count: usize,
    pub profile_span_cycles: u64,
    pub profile_span_ms: f64,
    pub at_offset: Option<u64>,
    pub groups: Vec<GroupSummary>,
    pub samples: Vec<SampleRecord>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GroupSummary {
    pub name: String,
    pub dimensions: BTreeMap<String, String>,
    pub sample_count: usize,
    pub tile_count: usize,
    pub phase_count: usize,
    pub phase_cycles: u64,
    pub phase_ms: f64,
    pub work_cycles: u64,
    pub average_active_tiles: f64,
    pub mean_cycles: f64,
    pub p50_cycles: u32,
    pub p95_cycles: u32,
    pub maximum_cycles: u32,
    pub first_offset: u64,
    pub last_offset: u64,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SampleRecord {
    pub physical_tile: u32,
    pub offset: u64,
    pub duration: u32,
    pub phase: u32,
    pub epoch: u32,
    pub kind: String,
    pub operation: String,
    pub kernel: String,
    pub metadata: Vec<MetadataEntry>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct MetadataEntry {
    pub name: String,
    pub value: String,
}

#[derive(Default)]
struct Accumulator {
    dimensions: BTreeMap<String, String>,
    durations: Vec<u32>,
    tiles: BTreeSet<u32>,
    phases: BTreeMap<(u32, u32), Vec<(u64, u64)>>,
    work_cycles: u64,
    first_offset: u64,
    last_offset: u64,
}

struct Candidate<'a> {
    tile: u32,
    offset: u64,
    duration: u32,
    sample: &'a CycleSample,
}

pub fn query(report: &ProfileReport, query: &Query) -> QueryReport {
    let sample_count = report.tiles.iter().map(|tile| tile.samples.len()).sum();
    let profile_span_cycles = report
        .tiles
        .iter()
        .filter_map(|tile| {
            let base = tile.samples.first()?.start_cycle;
            tile.samples
                .iter()
                .map(|sample| {
                    u64::from(sample.start_cycle.wrapping_sub(base)) + u64::from(duration(sample))
                })
                .max()
        })
        .max()
        .unwrap_or(0);
    let mut groups = HashMap::<String, Accumulator>::new();
    let mut candidates = Vec::new();
    let mut matched_sample_count = 0;

    for tile in &report.tiles {
        let base = tile.samples.first().map_or(0, |sample| sample.start_cycle);
        for sample in &tile.samples {
            let offset = u64::from(sample.start_cycle.wrapping_sub(base));
            let sample_duration = duration(sample);
            if !matches_query(query, tile.physical_tile, sample, offset, sample_duration) {
                continue;
            }
            matched_sample_count += 1;
            let (name, dimensions) = group_key(query, tile.physical_tile, sample);
            let accumulator = groups.entry(name).or_insert_with(|| Accumulator {
                dimensions,
                first_offset: offset,
                ..Accumulator::default()
            });
            accumulator.durations.push(sample_duration);
            accumulator.tiles.insert(tile.physical_tile);
            accumulator
                .phases
                .entry((sample.step.phase, sample.step.epoch))
                .or_default()
                .push((offset, offset + u64::from(sample_duration)));
            accumulator.work_cycles += u64::from(sample_duration);
            accumulator.first_offset = accumulator.first_offset.min(offset);
            accumulator.last_offset = accumulator
                .last_offset
                .max(offset + u64::from(sample_duration));
            if query.sample_limit > 0 {
                candidates.push(Candidate {
                    tile: tile.physical_tile,
                    offset,
                    duration: sample_duration,
                    sample,
                });
            }
        }
    }

    let mut summaries = groups
        .into_iter()
        .map(|(name, mut accumulator)| {
            accumulator.durations.sort_unstable();
            let phase_cycles = accumulator
                .phases
                .values()
                .map(|intervals| union_length(intervals))
                .sum::<u64>();
            let mean_cycles = if accumulator.durations.is_empty() {
                0.0
            } else {
                accumulator.work_cycles as f64 / accumulator.durations.len() as f64
            };
            GroupSummary {
                name,
                dimensions: accumulator.dimensions,
                sample_count: accumulator.durations.len(),
                tile_count: accumulator.tiles.len(),
                phase_count: accumulator.phases.len(),
                phase_cycles,
                phase_ms: cycles_to_ms(phase_cycles, report.clock_hz),
                work_cycles: accumulator.work_cycles,
                average_active_tiles: if phase_cycles == 0 {
                    0.0
                } else {
                    accumulator.work_cycles as f64 / phase_cycles as f64
                },
                mean_cycles,
                p50_cycles: percentile(&accumulator.durations, 50),
                p95_cycles: percentile(&accumulator.durations, 95),
                maximum_cycles: accumulator.durations.last().copied().unwrap_or(0),
                first_offset: accumulator.first_offset,
                last_offset: accumulator.last_offset,
            }
        })
        .collect::<Vec<_>>();
    summaries.sort_by(|left, right| compare_groups(query.sort_by, left, right));
    if let Some(limit) = query.limit {
        summaries.truncate(limit);
    }

    candidates.sort_by(|left, right| {
        right
            .duration
            .cmp(&left.duration)
            .then_with(|| left.tile.cmp(&right.tile))
            .then_with(|| left.offset.cmp(&right.offset))
    });
    let samples = candidates
        .into_iter()
        .take(query.sample_limit)
        .map(sample_record)
        .collect();

    QueryReport {
        clock_hz: report.clock_hz,
        tile_count: report.tiles.len(),
        sample_count,
        matched_sample_count,
        profile_span_cycles,
        profile_span_ms: cycles_to_ms(profile_span_cycles, report.clock_hz),
        at_offset: query.at_offset,
        groups: summaries,
        samples,
    }
}

fn matches_query(
    query: &Query,
    tile: u32,
    sample: &CycleSample,
    offset: u64,
    sample_duration: u32,
) -> bool {
    if query.kind.is_some_and(|kind| kind != kind_of(sample))
        || query
            .kernel
            .as_ref()
            .is_some_and(|kernel| sample.step.kernel != *kernel)
        || query
            .operation_contains
            .as_ref()
            .is_some_and(|operation| !sample.step.operation.contains(operation))
        || (!query.tiles.is_empty() && !query.tiles.contains(&tile))
        || (!query.phases.is_empty() && !query.phases.contains(&sample.step.phase))
        || query.at_offset.is_some_and(|at| {
            at < offset || at >= offset.saturating_add(u64::from(sample_duration))
        })
    {
        return false;
    }
    query.metadata.iter().all(|filter| {
        sample.step.metadata.iter().any(|entry| {
            entry.name == filter.name
                && filter
                    .value
                    .as_ref()
                    .is_none_or(|value| entry.value == *value)
        })
    })
}

fn group_key(query: &Query, tile: u32, sample: &CycleSample) -> (String, BTreeMap<String, String>) {
    let mut dimensions = BTreeMap::new();
    let name = match query.group_by {
        GroupBy::Kind => kind_name(sample).into(),
        GroupBy::Kernel => {
            if sample.step.kernel.is_empty() {
                "(no kernel)".into()
            } else {
                sample.step.kernel.clone()
            }
        }
        GroupBy::Operation => sample.step.operation.clone(),
        GroupBy::Phase => {
            dimensions.insert("phase".into(), sample.step.phase.to_string());
            dimensions.insert("epoch".into(), sample.step.epoch.to_string());
            format!("phase {}/{}", sample.step.phase, sample.step.epoch)
        }
        GroupBy::Tile => {
            dimensions.insert("physicalTile".into(), tile.to_string());
            format!("tile {tile}")
        }
        GroupBy::Metadata => {
            let key = query.metadata_key.as_deref().unwrap_or_default();
            let value = sample
                .step
                .metadata
                .iter()
                .find(|entry| entry.name == key)
                .map_or("(missing)", |entry| entry.value.as_str());
            dimensions.insert("metadataKey".into(), key.into());
            dimensions.insert("metadataValue".into(), value.into());
            value.into()
        }
    };
    (name, dimensions)
}

fn kind_of(sample: &CycleSample) -> StepKind {
    match sample.step.kind {
        ProfileStepKind::Exchange => StepKind::Exchange,
        ProfileStepKind::Compute => StepKind::Compute,
        ProfileStepKind::Synchronization => StepKind::Synchronization,
        ProfileStepKind::Idle => StepKind::Idle,
    }
}

fn kind_name(sample: &CycleSample) -> &'static str {
    match kind_of(sample) {
        StepKind::Exchange => "exchange",
        StepKind::Compute => "compute",
        StepKind::Synchronization => "synchronization",
        StepKind::Idle => "idle",
    }
}

fn duration(sample: &CycleSample) -> u32 {
    sample.end_cycle.wrapping_sub(sample.start_cycle)
}

fn percentile(sorted: &[u32], percentage: usize) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    sorted[(sorted.len() - 1) * percentage / 100]
}

fn union_length(intervals: &[(u64, u64)]) -> u64 {
    let mut intervals = intervals.to_vec();
    intervals.sort_unstable();
    let Some(&(mut start, mut end)) = intervals.first() else {
        return 0;
    };
    let mut total = 0;
    for &(next_start, next_end) in &intervals[1..] {
        if next_start <= end {
            end = end.max(next_end);
        } else {
            total += end - start;
            start = next_start;
            end = next_end;
        }
    }
    total + end - start
}

fn cycles_to_ms(cycles: u64, clock_hz: u64) -> f64 {
    if clock_hz == 0 {
        0.0
    } else {
        cycles as f64 * 1_000.0 / clock_hz as f64
    }
}

fn compare_groups(
    sort_by: SortBy,
    left: &GroupSummary,
    right: &GroupSummary,
) -> std::cmp::Ordering {
    let ordering = match sort_by {
        SortBy::PhaseCycles => right.phase_cycles.cmp(&left.phase_cycles),
        SortBy::WorkCycles => right.work_cycles.cmp(&left.work_cycles),
        SortBy::MaximumCycles => right.maximum_cycles.cmp(&left.maximum_cycles),
        SortBy::Samples => right.sample_count.cmp(&left.sample_count),
        SortBy::Name => left.name.cmp(&right.name),
    };
    ordering
        .then_with(|| left.first_offset.cmp(&right.first_offset))
        .then_with(|| left.name.cmp(&right.name))
}

fn sample_record(candidate: Candidate<'_>) -> SampleRecord {
    SampleRecord {
        physical_tile: candidate.tile,
        offset: candidate.offset,
        duration: candidate.duration,
        phase: candidate.sample.step.phase,
        epoch: candidate.sample.step.epoch,
        kind: kind_name(candidate.sample).into(),
        operation: candidate.sample.step.operation.clone(),
        kernel: candidate.sample.step.kernel.clone(),
        metadata: candidate
            .sample
            .step
            .metadata
            .iter()
            .map(|entry| MetadataEntry {
                name: entry.name.clone(),
                value: entry.value.clone(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ipu_package::{CycleSample, ProfileMetadata, ProfileStep, TileProfile};

    fn sample(
        phase: u32,
        kind: ProfileStepKind,
        kernel: &str,
        start: u32,
        end: u32,
    ) -> CycleSample {
        CycleSample {
            step: ProfileStep {
                local_index: phase,
                phase,
                epoch: 0,
                operation: format!("operation {phase}"),
                kind,
                kernel: kernel.into(),
                metadata: vec![ProfileMetadata {
                    name: "block".into(),
                    value: phase.to_string(),
                }],
            },
            start_cycle: start,
            end_cycle: end,
        }
    }

    #[test]
    fn groups_parallel_samples_by_kernel() {
        let report = ProfileReport {
            clock_hz: 1_000_000_000,
            tiles: vec![
                TileProfile {
                    physical_tile: 2,
                    samples: vec![sample(0, ProfileStepKind::Compute, "add", 100, 120)],
                },
                TileProfile {
                    physical_tile: 3,
                    samples: vec![sample(0, ProfileStepKind::Compute, "add", 200, 230)],
                },
            ],
        };
        let result = query(&report, &Query::default());

        assert_eq!(result.groups.len(), 1);
        assert_eq!(result.groups[0].phase_cycles, 30);
        assert_eq!(result.groups[0].work_cycles, 50);
        assert_eq!(result.groups[0].average_active_tiles, 50.0 / 30.0);
        assert_eq!(result.profile_span_cycles, 30);
    }

    #[test]
    fn phase_cycles_include_sequential_repeated_operations() {
        let report = ProfileReport {
            clock_hz: 1_000_000_000,
            tiles: vec![
                TileProfile {
                    physical_tile: 2,
                    samples: vec![
                        sample(0, ProfileStepKind::Compute, "gemm", 100, 110),
                        sample(0, ProfileStepKind::Compute, "gemm", 110, 120),
                    ],
                },
                TileProfile {
                    physical_tile: 3,
                    samples: vec![
                        sample(0, ProfileStepKind::Compute, "gemm", 200, 208),
                        sample(0, ProfileStepKind::Compute, "gemm", 208, 216),
                    ],
                },
            ],
        };
        let result = query(&report, &Query::default());

        assert_eq!(result.groups[0].phase_cycles, 20);
        assert_eq!(result.groups[0].work_cycles, 36);
        assert_eq!(result.groups[0].average_active_tiles, 1.8);
    }

    #[test]
    fn filters_by_normalized_time_and_metadata_across_counter_wrap() {
        let report = ProfileReport {
            clock_hz: 1_500_000_000,
            tiles: vec![TileProfile {
                physical_tile: 9,
                samples: vec![
                    sample(0, ProfileStepKind::Exchange, "", u32::MAX - 4, 5),
                    sample(1, ProfileStepKind::Compute, "add", 5, 15),
                ],
            }],
        };
        let query = Query {
            at_offset: Some(12),
            metadata: vec![MetadataFilter {
                name: "block".into(),
                value: Some("1".into()),
            }],
            sample_limit: 1,
            ..Query::default()
        };
        let result = super::query(&report, &query);

        assert_eq!(result.matched_sample_count, 1);
        assert_eq!(result.samples[0].phase, 1);
        assert_eq!(result.samples[0].offset, 10);
    }

    #[test]
    fn groups_by_semantic_metadata_value() {
        let report = ProfileReport {
            clock_hz: 1_000_000_000,
            tiles: vec![TileProfile {
                physical_tile: 1,
                samples: vec![
                    sample(0, ProfileStepKind::Compute, "add", 0, 10),
                    sample(1, ProfileStepKind::Compute, "add", 10, 30),
                ],
            }],
        };
        let result = query(
            &report,
            &Query {
                group_by: GroupBy::Metadata,
                metadata_key: Some("block".into()),
                ..Query::default()
            },
        );

        assert_eq!(result.groups.len(), 2);
        assert_eq!(result.groups[0].name, "1");
        assert_eq!(result.groups[1].name, "0");
    }
}
