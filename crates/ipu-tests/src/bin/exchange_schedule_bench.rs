use anyhow::{Context, Result, bail};
use clap::Parser;
use ipu_codegen::{
    ExchangeScheduleCache, ExchangeScheduleSnapshot, schedule_exchange_problem,
    validate_exchange_schedule,
};
use ipu_exchange::diagnostic::diagnose_plan_program;
use std::collections::BTreeSet;
use std::fs::File;
use std::hint::black_box;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(
    version,
    about = "Replay and validate production exchange scheduling without IPU hardware"
)]
struct Arguments {
    /// JSON snapshot written by ipu-trivial-test --export-exchange-schedule.
    snapshot: PathBuf,
    /// Restrict the benchmark to these physical exchange phase IDs.
    #[arg(long = "phase")]
    phases: Vec<u32>,
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

    let total_start = Instant::now();
    for captured in problems {
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
                schedule_exchange_problem(snapshot.tile_count, captured)
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
