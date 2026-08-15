use ipu_package::{
    CycleSample, ProfileExchangeActivity, ProfileExchangeActivityKind, ProfileReport,
    ProfileStepKind,
};
use serde::{Deserialize, Serialize};
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
    /// Preserve the complete device-wide timeline instead of cropping the
    /// common leading interval before every tile has emitted its first sample.
    pub shared_clock: bool,
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

#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ExchangeActivitySummary {
    pub exchange_samples: usize,
    pub described_samples: usize,
    pub send_intervals: usize,
    pub receive_intervals: usize,
    pub partner_busy_intervals: usize,
    pub estimated_send_work_cycles: u64,
    pub estimated_receive_work_cycles: u64,
    pub estimated_simultaneous_work_cycles: u64,
    pub estimated_partner_busy_work_cycles: u64,
    pub estimated_idle_work_cycles: u64,
    pub measured_phase_cycles: u64,
    pub scheduled_event_cycles: u64,
    pub arrival_wait_cycles: u64,
    pub phase_boundary_cycles: u64,
}

/// Machine-readable local-work measurements suitable for generating a target
/// cost table. Exchange and synchronization samples are deliberately omitted.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationDatabase {
    pub schema_version: u32,
    pub target: String,
    pub build_id: String,
    pub clock_hz: u64,
    pub measurements: Vec<CalibrationMeasurement>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationKey {
    pub kernel: String,
    pub dimensions: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationMeasurement {
    pub key: CalibrationKey,
    pub samples: u64,
    pub minimum_cycles: u32,
    pub median_cycles: u32,
    pub p95_cycles: u32,
    pub maximum_cycles: u32,
}

/// Collates profiled kernel and local-copy calls. Consecutive calls merged by
/// profiling are divided by their recorded invocation count before aggregation.
pub fn calibrate_profiles(
    reports: &[ProfileReport],
    target: impl Into<String>,
    build_id: impl Into<String>,
) -> Result<CalibrationDatabase, &'static str> {
    let Some(clock_hz) = reports.first().map(|report| report.clock_hz) else {
        return Err("at least one profile is required");
    };
    if reports.iter().any(|report| report.clock_hz != clock_hz) {
        return Err("profile clock frequencies differ");
    }
    let mut groups = BTreeMap::<CalibrationKey, Vec<u32>>::new();
    for report in reports {
        for tile in &report.tiles {
            for sample in &tile.samples {
                if sample.step.kind != ProfileStepKind::Compute || sample.step.kernel.is_empty() {
                    continue;
                }
                let invocations = sample
                    .step
                    .metadata
                    .iter()
                    .find(|entry| entry.name == "invocations")
                    .and_then(|entry| entry.value.parse::<u32>().ok())
                    .unwrap_or(1)
                    .max(1);
                let dimensions = sample
                    .step
                    .metadata
                    .iter()
                    .filter(|entry| {
                        !matches!(entry.name.as_str(), "reason" | "value" | "invocations")
                    })
                    .map(|entry| (entry.name.clone(), entry.value.clone()))
                    .collect();
                let cycles = duration(sample).div_ceil(invocations);
                groups
                    .entry(CalibrationKey {
                        kernel: sample.step.kernel.clone(),
                        dimensions,
                    })
                    .or_default()
                    .push(cycles);
            }
        }
    }
    let measurements = groups
        .into_iter()
        .map(|(key, mut cycles)| {
            cycles.sort_unstable();
            CalibrationMeasurement {
                key,
                samples: cycles.len() as u64,
                minimum_cycles: cycles[0],
                median_cycles: percentile(&cycles, 50),
                p95_cycles: percentile(&cycles, 95),
                maximum_cycles: *cycles.last().unwrap(),
            }
        })
        .collect();
    Ok(CalibrationDatabase {
        schema_version: 1,
        target: target.into(),
        build_id: build_id.into(),
        clock_hz,
        measurements,
    })
}

/// Returns a common device-cycle origin while preserving short offsets across
/// the wrapping 32-bit cycle counter.
pub fn cycle_origin(report: &ProfileReport) -> u32 {
    let Some(reference) = report
        .tiles
        .iter()
        .find_map(|tile| tile.samples.first().map(|sample| sample.start_cycle))
    else {
        return 0;
    };
    let minimum_delta = report
        .tiles
        .iter()
        .filter_map(|tile| tile.samples.first())
        .map(|sample| sample.start_cycle.wrapping_sub(reference) as i32)
        .min()
        .unwrap_or(0);
    reference.wrapping_add_signed(minimum_delta)
}

/// Summarizes statically scheduled exchange roles, counting a device-wide
/// exchange phase from the last participating tile's entry.
pub fn exchange_activity_summary(report: &ProfileReport) -> ExchangeActivitySummary {
    let mut summary = ExchangeActivitySummary::default();
    let base = cycle_origin(report);
    let mut phases = BTreeMap::<(u32, u32), (u64, u64, u64, u64)>::new();
    for tile in &report.tiles {
        for sample in tile
            .samples
            .iter()
            .filter(|sample| sample.step.kind == ProfileStepKind::Exchange)
        {
            summary.exchange_samples += 1;
            let event_cycles = u64::from(sample.step.exchange_event_cycles);
            if event_cycles == 0 {
                continue;
            }
            let start = u64::from(sample.start_cycle.wrapping_sub(base));
            let end = start + u64::from(duration(sample));
            let phase = phases
                .entry((sample.step.epoch, sample.step.phase))
                .or_insert((start, start, end, event_cycles));
            phase.0 = phase.0.min(start);
            phase.1 = phase.1.max(start);
            phase.2 = phase.2.max(end);
            phase.3 = phase.3.max(event_cycles);
            if sample.step.exchange_activities.is_empty() {
                continue;
            }
            summary.described_samples += 1;
            for activity in &sample.step.exchange_activities {
                match activity.kind {
                    ProfileExchangeActivityKind::Send => {
                        summary.send_intervals += 1;
                    }
                    ProfileExchangeActivityKind::Receive => {
                        summary.receive_intervals += 1;
                    }
                    ProfileExchangeActivityKind::PartnerBusy => {
                        summary.partner_busy_intervals += 1;
                    }
                }
            }
            let roles = exchange_role_cycles(&sample.step.exchange_activities, event_cycles);
            summary.estimated_send_work_cycles += roles.send;
            summary.estimated_receive_work_cycles += roles.receive;
            summary.estimated_simultaneous_work_cycles += roles.simultaneous;
            summary.estimated_partner_busy_work_cycles += roles.partner_busy;
            summary.estimated_idle_work_cycles += roles.idle;
        }
    }
    for (minimum_start, maximum_start, maximum_end, scheduled) in phases.into_values() {
        let measured = maximum_end.saturating_sub(maximum_start);
        summary.measured_phase_cycles += measured;
        summary.scheduled_event_cycles += scheduled;
        summary.arrival_wait_cycles += maximum_start.saturating_sub(minimum_start);
        summary.phase_boundary_cycles += measured.saturating_sub(scheduled);
    }
    summary
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ExchangeRoleCycles {
    send: u64,
    receive: u64,
    simultaneous: u64,
    partner_busy: u64,
    idle: u64,
}

fn exchange_role_cycles(
    activities: &[ProfileExchangeActivity],
    event_cycles: u64,
) -> ExchangeRoleCycles {
    let mut events = Vec::with_capacity(activities.len() * 2);
    for activity in activities {
        let start = u64::from(activity.start_cycle).min(event_cycles);
        let end = u64::from(activity.end_cycle).min(event_cycles);
        if end <= start {
            continue;
        }
        let (send, receive, partner_busy) = match activity.kind {
            ProfileExchangeActivityKind::Send => (1i32, 0i32, 0i32),
            ProfileExchangeActivityKind::Receive => (0, 1, 0),
            ProfileExchangeActivityKind::PartnerBusy => (0, 0, 1),
        };
        events.push((start, send, receive, partner_busy));
        events.push((end, -send, -receive, -partner_busy));
    }
    events.sort_unstable_by_key(|event| event.0);

    let mut send_depth = 0i32;
    let mut receive_depth = 0i32;
    let mut partner_busy_depth = 0i32;
    let mut cursor = 0u64;
    let mut send_cycles = 0u64;
    let mut receive_cycles = 0u64;
    let mut simultaneous_cycles = 0u64;
    let mut partner_busy_cycles = 0u64;
    let mut idle_cycles = 0u64;
    let mut index = 0usize;
    while index < events.len() {
        let time = events[index].0;
        let cycles = time.saturating_sub(cursor);
        send_cycles += cycles * u64::from(send_depth > 0);
        receive_cycles += cycles * u64::from(receive_depth > 0);
        simultaneous_cycles += cycles * u64::from(send_depth > 0 && receive_depth > 0);
        partner_busy_cycles += cycles * u64::from(partner_busy_depth > 0);
        idle_cycles +=
            cycles * u64::from(send_depth == 0 && receive_depth == 0 && partner_busy_depth == 0);
        while index < events.len() && events[index].0 == time {
            send_depth += events[index].1;
            receive_depth += events[index].2;
            partner_busy_depth += events[index].3;
            index += 1;
        }
        cursor = time;
    }
    idle_cycles += event_cycles.saturating_sub(cursor);
    ExchangeRoleCycles {
        send: send_cycles,
        receive: receive_cycles,
        simultaneous: simultaneous_cycles,
        partner_busy: partner_busy_cycles,
        idle: idle_cycles,
    }
}

#[derive(Default)]
struct Accumulator {
    dimensions: BTreeMap<String, String>,
    durations: Vec<u32>,
    tiles: BTreeSet<u32>,
    phases: BTreeSet<(u32, u32)>,
    intervals: Vec<(u64, u64)>,
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
    let shared_base = cycle_origin(report);
    let crop = if query.shared_clock {
        0
    } else {
        initial_sample_entry_span(report, shared_base)
    };
    let profile_span_cycles = report
        .tiles
        .iter()
        .flat_map(|tile| &tile.samples)
        .map(|sample| {
            u64::from(sample.start_cycle.wrapping_sub(shared_base))
                .saturating_add(u64::from(duration(sample)))
                .saturating_sub(crop)
        })
        .max()
        .unwrap_or(0);
    let mut groups = HashMap::<String, Accumulator>::new();
    let mut candidates = Vec::new();
    let mut matched_sample_count = 0;

    for tile in &report.tiles {
        for sample in &tile.samples {
            let raw_offset = u64::from(sample.start_cycle.wrapping_sub(shared_base));
            let raw_end = raw_offset.saturating_add(u64::from(duration(sample)));
            let offset = raw_offset.saturating_sub(crop);
            let end = raw_end.saturating_sub(crop);
            let sample_duration = u32::try_from(end.saturating_sub(offset)).unwrap_or(u32::MAX);
            if sample_duration == 0 {
                continue;
            }
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
                .insert((sample.step.phase, sample.step.epoch));
            accumulator
                .intervals
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
            // Compute phase numbers are local schedule indices. Distinct
            // numbers can overlap on different tiles, so union the group's
            // complete wall-clock coverage instead of summing phase unions.
            let phase_cycles = union_length(&accumulator.intervals);
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

/// Amount of the shared-clock timeline before the last tile emits its first
/// sample. Cropping this common prefix preserves all cross-tile relationships
/// after entry into the initial global boundary.
fn initial_sample_entry_span(report: &ProfileReport, shared_base: u32) -> u64 {
    report
        .tiles
        .iter()
        .filter_map(|tile| tile.samples.first())
        .map(|sample| u64::from(sample.start_cycle.wrapping_sub(shared_base)))
        .max()
        .unwrap_or(0)
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
                exchange_activities: Vec::new(),
                exchange_event_cycles: 0,
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
                    samples: vec![sample(1, ProfileStepKind::Compute, "add", 100, 130)],
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
    fn randomized_queries_crop_one_common_initial_entry_span() {
        let mut random = fastrand::Rng::with_seed(0x7072_6f67_7261_6d);
        for case in 0..64 {
            let tile_count = random.usize(2..=32);
            let starts = (0..tile_count)
                .map(|_| random.u32(1_000..=10_000))
                .collect::<Vec<_>>();
            let latest_start = *starts.iter().max().unwrap();
            let boundary = latest_start + random.u32(1..=2_000);
            let work = random.u32(1..=2_000);
            let report = ProfileReport {
                clock_hz: 1_000_000_000,
                tiles: starts
                    .iter()
                    .enumerate()
                    .map(|(tile, &start)| TileProfile {
                        physical_tile: tile as u32,
                        samples: vec![
                            sample(0, ProfileStepKind::Synchronization, "", start, boundary),
                            sample(
                                1,
                                ProfileStepKind::Compute,
                                "work",
                                boundary,
                                boundary + work,
                            ),
                        ],
                    })
                    .collect(),
            };
            let aligned = query(
                &report,
                &Query {
                    group_by: GroupBy::Kind,
                    ..Query::default()
                },
            );
            let shared = query(
                &report,
                &Query {
                    shared_clock: true,
                    ..Query::default()
                },
            );

            let retained_wait = u64::from(boundary - latest_start);
            assert_eq!(
                aligned.profile_span_cycles,
                retained_wait + u64::from(work),
                "case {case}"
            );
            assert!(
                shared.profile_span_cycles >= aligned.profile_span_cycles,
                "case {case}"
            );
            let sync = aligned
                .groups
                .iter()
                .find(|group| group.name == "synchronization")
                .unwrap();
            assert_eq!(sync.first_offset, 0, "case {case}");
            assert_eq!(sync.maximum_cycles, retained_wait as u32, "case {case}");
            let compute = aligned
                .groups
                .iter()
                .find(|group| group.name == "compute")
                .unwrap();
            assert_eq!(compute.first_offset, retained_wait, "case {case}");
        }
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
                        sample(0, ProfileStepKind::Compute, "gemm", 100, 108),
                        sample(0, ProfileStepKind::Compute, "gemm", 108, 116),
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
    fn common_origin_does_not_turn_first_exchange_skew_into_later_skew() {
        let report = ProfileReport {
            clock_hz: 1_000_000_000,
            tiles: vec![
                TileProfile {
                    physical_tile: 2,
                    samples: vec![
                        sample(0, ProfileStepKind::Exchange, "", 100, 150),
                        sample(1, ProfileStepKind::Compute, "gemm", 150, 160),
                    ],
                },
                TileProfile {
                    physical_tile: 3,
                    samples: vec![
                        sample(0, ProfileStepKind::Exchange, "", 120, 150),
                        sample(1, ProfileStepKind::Compute, "gemm", 150, 160),
                    ],
                },
            ],
        };
        let result = query(
            &report,
            &Query {
                kind: Some(StepKind::Compute),
                sample_limit: 2,
                shared_clock: true,
                ..Query::default()
            },
        );

        assert_eq!(result.samples.len(), 2);
        assert!(result.samples.iter().all(|sample| sample.offset == 50));
        assert_eq!(result.groups[0].phase_cycles, 10);
    }

    #[test]
    fn randomized_exchange_role_accounting_tracks_simultaneous_intervals() {
        let mut random = fastrand::Rng::with_seed(0x6578_6368_726f_6c65);
        for _ in 0..256 {
            let event_cycles = random.u32(1..=200);
            let activities = (0..random.usize(0..=40))
                .map(|_| {
                    let start_cycle = random.u32(0..event_cycles);
                    ProfileExchangeActivity {
                        kind: match random.u8(0..3) {
                            0 => ProfileExchangeActivityKind::Send,
                            1 => ProfileExchangeActivityKind::Receive,
                            _ => ProfileExchangeActivityKind::PartnerBusy,
                        },
                        start_cycle,
                        end_cycle: random.u32(start_cycle + 1..=event_cycles),
                    }
                })
                .collect::<Vec<_>>();
            let expected =
                (0..event_cycles).fold(ExchangeRoleCycles::default(), |mut totals, cycle| {
                    let send = activities.iter().any(|activity| {
                        activity.kind == ProfileExchangeActivityKind::Send
                            && (activity.start_cycle..activity.end_cycle).contains(&cycle)
                    });
                    let receive = activities.iter().any(|activity| {
                        activity.kind == ProfileExchangeActivityKind::Receive
                            && (activity.start_cycle..activity.end_cycle).contains(&cycle)
                    });
                    let partner_busy = activities.iter().any(|activity| {
                        activity.kind == ProfileExchangeActivityKind::PartnerBusy
                            && (activity.start_cycle..activity.end_cycle).contains(&cycle)
                    });
                    totals.send += u64::from(send);
                    totals.receive += u64::from(receive);
                    totals.simultaneous += u64::from(send && receive);
                    totals.partner_busy += u64::from(partner_busy);
                    totals.idle += u64::from(!send && !receive && !partner_busy);
                    totals
                });
            assert_eq!(
                exchange_role_cycles(&activities, u64::from(event_cycles)),
                expected
            );
        }
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

    #[test]
    fn calibration_normalizes_merged_invocations_and_excludes_graph_metadata() {
        let mut first = sample(0, ProfileStepKind::Compute, "gemm-f16", 10, 50);
        first.step.metadata.extend([
            ProfileMetadata {
                name: "kernelSpec".into(),
                value: "Gemm".into(),
            },
            ProfileMetadata {
                name: "invocations".into(),
                value: "2".into(),
            },
            ProfileMetadata {
                name: "reason".into(),
                value: "OperatorKernel".into(),
            },
        ]);
        let mut second = first.clone();
        second.start_cycle = 100;
        second.end_cycle = 160;
        let database = calibrate_profiles(
            &[ProfileReport {
                clock_hz: 1_500_000_000,
                tiles: vec![TileProfile {
                    physical_tile: 0,
                    samples: vec![first, second],
                }],
            }],
            "ipu21",
            "test-build",
        )
        .unwrap();

        assert_eq!(database.measurements.len(), 1);
        let measurement = &database.measurements[0];
        assert_eq!(measurement.samples, 2);
        assert_eq!(measurement.minimum_cycles, 20);
        assert_eq!(measurement.maximum_cycles, 30);
        assert!(!measurement.key.dimensions.contains_key("reason"));
        assert!(!measurement.key.dimensions.contains_key("invocations"));
    }
}
