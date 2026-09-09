use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use ipu_codegen::{
    ExchangeScheduleCache, ExchangeScheduleSnapshot, ExchangeSchedulingPriority,
    schedule_exchange_problem_with_priority, validate_exchange_schedule,
};
use ipu_exchange::diagnostic::diagnose_plan_program;
use std::collections::BTreeSet;
use std::fs::File;
use std::hint::black_box;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Default, ValueEnum)]
enum Priority {
    #[default]
    Automatic,
    Combined,
    Directional,
    RemainingCombined,
    RemainingDirectional,
}

impl From<Priority> for ExchangeSchedulingPriority {
    fn from(value: Priority) -> Self {
        match value {
            Priority::Automatic => Self::Automatic,
            Priority::Combined => Self::Combined,
            Priority::Directional => Self::Directional,
            Priority::RemainingCombined => Self::RemainingCombined,
            Priority::RemainingDirectional => Self::RemainingDirectional,
        }
    }
}

#[derive(Parser)]
#[command(
    version,
    about = "Replay and validate production exchange scheduling without IPU hardware"
)]
struct Arguments {
    /// JSON snapshot written by ipu-trivial-test --export-exchange-schedule.
    snapshot: PathBuf,
    /// Offline queue priority experiment, with unchanged timing and validation.
    #[arg(long, value_enum, default_value_t = Priority::Automatic, conflicts_with_all = ["select_widths", "replay_cache"])]
    priority: Priority,
    /// Offline address-ordered stream waves, measured in 32-bit words.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..), conflicts_with_all = ["priority", "select_widths", "replay_cache"])]
    stream_words: Option<u32>,
    /// Restrict the benchmark to these physical exchange phase IDs.
    #[arg(long = "phase")]
    phases: Vec<u32>,
    /// Report storage estimates without running the scheduler.
    #[arg(long)]
    footprint_only: bool,
    /// Untimed scheduler/codegen runs before measurement.
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    /// Timed scheduler/codegen runs per selected phase.
    #[arg(long, default_value_t = 1)]
    iterations: usize,
    /// Ignore sender addresses after the first Repeat iteration. This
    /// reproduces the unsafe scheduler behavior used before Repeat-aware
    /// memory-element hazard checking.
    #[arg(long)]
    first_iteration_only: bool,
    /// Include the production comparison of ordinary and paired transfers.
    #[arg(long)]
    select_widths: bool,
    /// Save the selected ordinary/paired transfers for controlled order comparisons.
    #[arg(long, requires = "select_widths")]
    write_selected_snapshot: Option<PathBuf>,
    /// Report the busiest directional endpoints and gaps between their payloads.
    #[arg(long)]
    slack_report: bool,
    /// Populate the production recipe cache before timing, then measure replay.
    #[arg(long, requires = "select_widths")]
    replay_cache: bool,
    /// Relocate every source/destination by this many bytes after cache warmup.
    #[arg(long, default_value_t = 0, requires = "replay_cache")]
    relocate_by: u32,
    /// Decode the generated exchange program for this logical tile.
    #[arg(long)]
    dump_tile: Option<usize>,
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    if std::env::var_os("RUST_LOG").is_some() {
        ipu_runtime::init_tracing();
    }
    if arguments.iterations == 0 {
        bail!("--iterations must be nonzero");
    }
    let input = File::open(&arguments.snapshot)
        .with_context(|| format!("open {}", arguments.snapshot.display()))?;
    let mut snapshot: ExchangeScheduleSnapshot = serde_json::from_reader(BufReader::new(input))
        .with_context(|| format!("parse {}", arguments.snapshot.display()))?;
    snapshot.validate()?;
    if arguments.first_iteration_only {
        for transfer in snapshot
            .phases
            .iter_mut()
            .flat_map(|phase| &mut phase.transfers)
        {
            transfer.source_addresses.truncate(1);
        }
    }

    let selected = arguments.phases.iter().copied().collect::<BTreeSet<_>>();
    if selected.len() != arguments.phases.len() {
        bail!("--phase contains a duplicate phase ID");
    }
    for &phase in &selected {
        if !snapshot.phases.iter().any(|problem| problem.phase == phase) {
            bail!("snapshot does not contain phase {phase}");
        }
    }

    let problems = snapshot
        .phases
        .iter()
        .filter(|problem| selected.is_empty() || selected.contains(&problem.phase))
        .collect::<Vec<_>>();
    println!(
        "snapshot={} tiles={} phases={} warmup={} iterations={} repeatAware={}",
        arguments.snapshot.display(),
        snapshot.tile_count,
        problems.len(),
        arguments.warmup,
        arguments.iterations,
        !arguments.first_iteration_only,
    );

    let mut selected_problems = Vec::new();
    let total_start = Instant::now();
    let mut estimated_totals = vec![0u64; usize::from(snapshot.tile_count)];
    let mut storage = ipu_codegen::ExchangeStorageEstimator::new(snapshot.tile_count);
    for captured in problems {
        let start = Instant::now();
        let estimate = storage.add_phase(captured);
        let estimate_time = start.elapsed();
        for (total, bytes) in estimated_totals.iter_mut().zip(&estimate) {
            *total += bytes;
        }
        println!(
            "phase={} estimatedMaximumRowBytes={} footprintMs={:.3}",
            captured.phase,
            estimate.iter().max().unwrap_or(&0),
            milliseconds(estimate_time)
        );
        if arguments.footprint_only {
            continue;
        }
        let mut cache = ExchangeScheduleCache::default();
        if arguments.replay_cache {
            let (selected, run) = cache.schedule_problem(snapshot.tile_count, captured)?;
            validate_exchange_schedule(snapshot.tile_count, &selected, &run.phase)?;
        }
        let mut relocated;
        let captured = if arguments.relocate_by != 0 {
            relocated = captured.clone();
            for transfer in &mut relocated.transfers {
                for address in transfer.source_addresses.iter_mut().chain(
                    transfer
                        .destinations
                        .iter_mut()
                        .map(|destination| &mut destination.address),
                ) {
                    *address = address
                        .checked_add(arguments.relocate_by)
                        .context("relocated address overflow")?;
                }
            }
            &relocated
        } else {
            captured
        };
        let mut schedule = || {
            if arguments.select_widths {
                if !arguments.replay_cache {
                    cache = ExchangeScheduleCache::default();
                }
                cache
                    .schedule_problem(snapshot.tile_count, captured)
                    .map(|(problem, run)| (std::borrow::Cow::Owned(problem), run))
            } else {
                schedule_exchange_problem_with_priority(
                    snapshot.tile_count,
                    captured,
                    arguments
                        .stream_words
                        .map(ExchangeSchedulingPriority::Streams)
                        .unwrap_or_else(|| arguments.priority.into()),
                )
                .map(|run| (std::borrow::Cow::Borrowed(captured), run))
            }
        };
        for _ in 0..arguments.warmup {
            let (problem, run) = black_box(schedule()?);
            validate_exchange_schedule(snapshot.tile_count, &problem, &run.phase)?;
        }
        let mut baseline = None;
        let mut durations = Vec::with_capacity(arguments.iterations);
        let mut validation_durations = Vec::with_capacity(arguments.iterations);
        for _ in 0..arguments.iterations {
            let start = Instant::now();
            let (problem, run) = black_box(schedule()?);
            durations.push(start.elapsed());
            let validation_start = Instant::now();
            validate_exchange_schedule(snapshot.tile_count, &problem, &run.phase)?;
            validation_durations.push(validation_start.elapsed());
            if let Some(expected) = &baseline {
                if &run.phase != expected {
                    bail!(
                        "phase {} scheduler/codegen output changed between identical runs",
                        problem.phase
                    );
                }
            } else {
                baseline = Some(run.phase.clone());
                selected_problems.push(problem.as_ref().clone());
                if arguments.slack_report {
                    report_slack(&run.phase);
                }
            }
            if durations.len() == 1
                && let Some(tile) = arguments.dump_tile
            {
                let words = run
                    .phase
                    .programs
                    .get(tile)
                    .with_context(|| format!("logical tile {tile} is out of range"))?;
                let activities = run
                    .phase
                    .activities
                    .get(tile)
                    .with_context(|| format!("logical tile {tile} is out of range"))?;
                for activity in activities {
                    println!(
                        "phase={} logicalTile={} transfer={} {:?} cycles={}..{} memoryEnd={} address=0x{:x} words={}",
                        problem.phase,
                        tile,
                        activity.transfer,
                        activity.kind,
                        activity.start_cycle,
                        activity.end_cycle,
                        activity.memory_end_cycle,
                        activity.address,
                        activity.words,
                    );
                }
                println!(
                    "phase={} logicalTile={}\n{}",
                    problem.phase,
                    tile,
                    diagnose_plan_program(words, None)?.render()
                );
            }
            let destination_count = problem
                .transfers
                .iter()
                .map(|transfer| transfer.destinations.len())
                .sum::<usize>();
            let row_words = run.phase.programs.iter().map(Vec::len).sum::<usize>();
            let maximum_row_words = run.phase.programs.iter().map(Vec::len).max().unwrap_or(0);
            if durations.len() == arguments.iterations {
                durations.sort_unstable();
                validation_durations.sort_unstable();
                println!(
                    "phase={} transfers={} destinations={} initialHorizonCycles={} horizonCycles={} endpointLowerBoundCycles={} lowerBoundGapCycles={} neighborhoodImprovements={} rowWords={} maximumRowWords={} scheduleCodegenMinMs={:.3} scheduleCodegenMedianMs={:.3} scheduleCodegenP95Ms={:.3} scheduleCodegenMaxMs={:.3} validationMedianMs={:.3} reused={} rowFingerprint={:016x} invariants=PASS",
                    problem.phase,
                    problem.transfers.len(),
                    destination_count,
                    run.initial_horizon,
                    run.phase.event_cycles,
                    run.endpoint_lower_bound,
                    run.phase
                        .event_cycles
                        .saturating_sub(run.endpoint_lower_bound),
                    run.neighborhood_improvements,
                    row_words,
                    maximum_row_words,
                    milliseconds(durations[0]),
                    milliseconds(percentile(&durations, 50)),
                    milliseconds(percentile(&durations, 95)),
                    milliseconds(*durations.last().expect("iterations is nonzero")),
                    milliseconds(percentile(&validation_durations, 50)),
                    run.reused,
                    {
                        use std::hash::{Hash, Hasher};
                        let mut hash = std::hash::DefaultHasher::new();
                        run.phase.programs.hash(&mut hash);
                        hash.finish()
                    },
                );
            }
        }
    }
    if let Some(path) = &arguments.write_selected_snapshot {
        serde_json::to_writer(
            File::create(path)?,
            &ExchangeScheduleSnapshot {
                schema_version: snapshot.schema_version,
                tile_count: snapshot.tile_count,
                phases: selected_problems,
            },
        )?;
    }
    println!(
        "estimatedMaximumTableBytes={} unsharedMaximumTableBytes={}",
        storage.maximum_bytes(),
        estimated_totals.iter().max().unwrap_or(&0)
    );
    println!("totalMs={:.3}", milliseconds(total_start.elapsed()));
    Ok(())
}

fn percentile(samples: &[Duration], percentile: usize) -> Duration {
    let rank = (samples.len() * percentile).div_ceil(100);
    samples[rank.clamp(1, samples.len()) - 1]
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn report_slack(phase: &ipu_codegen::PhysicalExchangePhase) {
    use ipu_codegen::ExchangeActivityKind::{PartnerBusy, Receive, Send};
    let mut endpoints = Vec::new();
    for (tile, activities) in phase.activities.iter().enumerate() {
        for receive in [false, true] {
            let mut events = activities
                .iter()
                .filter(|event| {
                    if receive {
                        event.kind == Receive
                    } else {
                        matches!(event.kind, Send | PartnerBusy)
                    }
                })
                .collect::<Vec<_>>();
            events.sort_unstable_by_key(|event| event.start_cycle);
            let Some(last) = events.last() else {
                continue;
            };
            let end = last.end_cycle;
            let busy = events
                .iter()
                .map(|event| u64::from(event.end_cycle - event.start_cycle))
                .sum::<u64>();
            let gaps = events
                .windows(2)
                .map(|pair| pair[1].start_cycle.saturating_sub(pair[0].end_cycle))
                .filter(|&gap| gap != 0)
                .collect::<Vec<_>>();
            let discontinuities = events
                .windows(2)
                .filter(|pair| {
                    pair[0].address.checked_add(pair[0].words * 4) != Some(pair[1].address)
                })
                .count();
            endpoints.push((
                end,
                tile,
                receive,
                busy,
                events.len(),
                gaps.iter().map(|&n| u64::from(n)).sum::<u64>(),
                gaps.into_iter().max().unwrap_or(0),
                discontinuities,
            ));
        }
    }
    endpoints.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.0));
    for (end, tile, receive, busy, transfers, gaps, largest_gap, discontinuities) in
        endpoints.into_iter().take(8)
    {
        println!(
            "slack phase={} tile={} port={} end={} payloadCycles={} transfers={} internalGapCycles={} largestGap={} addressDiscontinuities={}",
            phase.id.index(),
            tile,
            if receive { "rx" } else { "tx" },
            end,
            busy,
            transfers,
            gaps,
            largest_gap,
            discontinuities
        );
    }
}
