use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use half::f16;
use ipu_codegen::{
    AmpOrder, AttentionStrategy, BlockMajorOrder, CompiledPackage, ComputeGraph, DiagnosticTensor,
    GemmOrientation, GemmPlanConstraint, Layout, LocalOperandStaging, MemoryClass, OperatorClass,
    PackageConfig, PipelineConfig, PlannerSearchDomain, Precision, ProfilingMode, ReductionStaging,
    TensorFormat, amp_matrix_coordinates, block_major_matrix_coordinates, build_diagnostic_package,
    build_package,
};
use ipu_driver::DriverError;
use ipu_elf::Toolchain;
use ipu_package::{Application, Binding};
use ipu_runtime::Runtime;
use rand_distr::{Distribution, StandardNormal};
use rand_xoshiro::{SplitMix64, rand_core::SeedableRng};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::time::{Duration, Instant};

mod diagnostic;
mod exchange_stress;

#[derive(Parser)]
#[command(version, about = "Build, load, and diagnose the trivial IPU21 package")]
struct Arguments {
    /// Captured IPUCFG1 device configuration.
    configuration: PathBuf,
    #[arg(long, env = "POPLAR_SDK_ENABLED")]
    sdk: PathBuf,
    #[arg(long)]
    bootloader: Option<PathBuf>,
    #[arg(long, default_value = "/dev/ipu0")]
    device: String,
    #[arg(long, default_value = "/tmp/ipu-trivial.ipuexe")]
    package: PathBuf,
    /// Load an existing --package without rebuilding it.
    #[arg(long)]
    reuse_package: bool,
    /// Stop after each semantic operator, sample its optimized device tensor,
    /// and compare it with a host reference before resuming.
    #[arg(long, conflicts_with = "reuse_package")]
    diagnostic_run: bool,
    /// Maximum number of logical elements checked per operator result.
    #[arg(long, default_value_t = 256)]
    diagnostic_samples: usize,
    #[arg(long, default_value_t = 0.03)]
    diagnostic_atol: f32,
    #[arg(long, default_value_t = 0.03)]
    diagnostic_rtol: f32,
    /// Write per-tile kernel and exchange cycle samples for benchmark runs.
    #[arg(long)]
    profile_output: Option<PathBuf>,
    /// Profiling instrumentation: none, overall start/end timing, or full step traces.
    #[arg(long, default_value = "full")]
    profiling: ProfilingMode,
    /// Log exchange scheduling lower bounds and critical dependency chains.
    #[arg(long)]
    exchange_diagnostics: bool,
    /// Decode every active supervisor row for this exchange-stress case.
    #[arg(long, requires = "exchange_diagnostics")]
    exchange_diagnostic_case: Option<u32>,
    /// Replace the compiled workload with a tokenized replay of one exact
    /// physical exchange phase and verify every touched word after execution.
    #[arg(long, conflicts_with_all = ["reuse_package", "diagnostic_run"])]
    exchange_replay_phase: Option<usize>,
    /// Replay a phase from an exported address-resolved schedule without
    /// recompiling the model.
    #[arg(
        long,
        requires = "exchange_replay_phase",
        conflicts_with = "export_exchange_schedule"
    )]
    replay_exchange_schedule: Option<PathBuf>,
    /// Maximum number of systematically distributed words read back by an
    /// exact exchange-phase replay.
    #[arg(long, default_value_t = 8192)]
    exchange_replay_samples: usize,
    /// Replay only this prefix of the selected phase's transfer list.
    #[arg(long, requires = "exchange_replay_phase")]
    exchange_replay_transfer_limit: Option<usize>,
    /// Skip this many transfers before applying the replay transfer limit.
    #[arg(long, default_value_t = 0, requires = "exchange_replay_phase")]
    exchange_replay_first_transfer: usize,
    /// Summarize packaged exchange rows and exit before loading hardware.
    #[arg(long)]
    inspect_exchanges: bool,
    /// Write the address-resolved pre-scheduling exchange input and exit.
    #[arg(long, conflicts_with_all = ["reuse_package", "diagnostic_run", "exchange_replay_phase"])]
    export_exchange_schedule: Option<PathBuf>,
    /// Include complete decoded rows for one physical tile in the inspection.
    #[arg(long, requires = "inspect_exchanges")]
    inspect_exchange_tile: Option<u32>,
    /// Force eligible one-use layout conversions to stream into consumer slices.
    #[arg(long, conflicts_with = "materialize_conversions")]
    stream_conversions: bool,
    /// Materialize eligible layout conversions before their consumers.
    #[arg(long, conflicts_with = "stream_conversions")]
    materialize_conversions: bool,
    /// Constrain planning as though only this much SRAM per tile were free.
    #[arg(long, conflicts_with = "reuse_package")]
    tile_memory_budget_kib: Option<u64>,
    /// Rank this many complete planner finalists with physical exchange scheduling.
    #[arg(long, default_value_t = 1, conflicts_with = "reuse_package")]
    exchange_schedule_finalists: usize,
    /// Retain an exact GEMM family: OP:RxCxK:RRxRC:C:MEMORY:ORIENTATION:REDUCTION:LOCAL.
    #[arg(
        long,
        value_parser = parse_gemm_plan_constraint,
        conflicts_with = "reuse_package"
    )]
    gemm_plan_constraint: Vec<GemmPlanConstraint>,
    #[arg(long, default_value_t = c600_tile_count())]
    tiles: u32,
    #[arg(long)]
    runtime_source: Option<PathBuf>,
    #[arg(long, default_value_t = 10)]
    timeout_seconds: u64,
    /// Program and validation workload to build and run.
    #[arg(long, value_enum, default_value_t = Workload::Diagnostic)]
    workload: Workload,
    #[arg(long, default_value_t = SIGLIP_MLP_BATCH)]
    mlp_batch: u32,
    #[arg(long, default_value_t = SIGLIP_MLP_TOKENS)]
    mlp_tokens: u32,
    #[arg(long, default_value_t = SIGLIP_MLP_DIMENSION)]
    mlp_dim: u32,
    /// Independent attention batches, flattened into the head axis.
    #[arg(long, default_value_t = 1)]
    attention_batch: u32,
    /// Restrict attention planning for controlled strategy comparisons.
    #[arg(long, value_enum, default_value_t = AttentionMode::Auto)]
    attention_strategy: AttentionMode,
    #[arg(long, default_value_t = SIGLIP_ATTENTION_HEADS)]
    attention_heads: u32,
    /// Defaults to SigLIP's 4304-wide intermediate for the canonical 1152D
    /// benchmark, and otherwise to four times --mlp-dim.
    #[arg(long)]
    mlp_hidden_dim: Option<u32>,
    /// Number of sequential MLP blocks, represented by one structured repeat.
    #[arg(long, default_value_t = 1)]
    mlp_blocks: u32,
    /// Global GEMM row count, independent of the configured tile count.
    #[arg(long, default_value_t = GEMM_BENCHMARK_ROWS)]
    benchmark_rows: u32,
    #[arg(long, default_value_t = GEMM_BENCHMARK_INNER)]
    benchmark_inner: u32,
    #[arg(long, default_value_t = GEMM_BENCHMARK_COLUMNS)]
    benchmark_columns: u32,
    #[arg(long, default_value_t = 1_500_000_000)]
    clock_hz: u64,
    /// Reproducible random seed for --workload exchange-stress.
    #[arg(long, default_value_t = 0x4950_5532_3100_0001)]
    exchange_seed: u64,
    /// Schedule family exercised by --workload exchange-stress.
    #[arg(long, value_enum, default_value_t = ExchangeStressPattern::Random)]
    exchange_pattern: ExchangeStressPattern,
    /// Number of randomized exchange epochs in one stress package.
    #[arg(long, default_value_t = 128)]
    exchange_cases: u32,
    /// First bank/class combination selected by the paired-width diagnostic.
    #[arg(long, default_value_t = 0)]
    exchange_wide_first_case: u32,
    /// Maximum words in one randomized transfer.
    #[arg(long, default_value_t = 256)]
    exchange_max_words: u32,
    /// Maximum independently scheduled transfers in one randomized epoch.
    #[arg(long, default_value_t = 8)]
    exchange_max_transfers: u32,
    /// Maximum worker-loop iterations between randomized exchange epochs.
    #[arg(long, default_value_t = 2048)]
    exchange_compute_delay: u32,
    /// Omit paired-width result readback while retaining the selected rows.
    #[arg(long)]
    exchange_skip_validation: bool,
    /// Bit mask selecting the primary (bit 0) and secondary (bit 1) rows in
    /// the paired-width diagnostic.
    #[arg(long, default_value_t = 3)]
    exchange_wide_receiver_mask: u8,
    /// Explicitly program the otherwise self-describing paired receive CSRs.
    #[arg(long)]
    exchange_wide_explicit_config: bool,
    /// Make every tile join the paired-width sync with a minimal idle row.
    #[arg(long)]
    exchange_wide_all_active: bool,
    /// Number of complete physical receiver pairs in the paired-width diagnostic.
    #[arg(long, default_value_t = 1)]
    exchange_wide_pairs: u16,
    /// Logical source tile used by the paired-width diagnostic.
    #[arg(long, default_value_t = 0)]
    exchange_wide_source: u16,
    /// First logical receiver tile; it must begin a complete physical pair.
    #[arg(long, default_value_t = 2)]
    exchange_wide_first_destination: u16,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Workload {
    /// Load the minimal runtime and report tile completion diagnostics.
    Diagnostic,
    /// Numerically verify one blocked F16 GEMM.
    GemmSmoke,
    /// Numerically verify an F16 GEMM with batched activations.
    BatchedGemmSmoke,
    /// Numerically verify GEMM-GeLU-GEMM-GeLU with Gaussian data.
    MlpSmoke,
    /// Numerically verify exact non-causal FP16 FlashAttention.
    AttentionSmoke,
    /// Profile one compute-dense F16 GEMM.
    GemmBenchmark,
    /// Profile the canonical batched SigLIP Dense-GeLU-Dense workload.
    SiglipMlpBenchmark,
    /// Profile SigLIP self-attention (16 heads, 729 tokens, width 72).
    SiglipAttentionBenchmark,
    /// Run reproducible randomized small-group tile exchanges.
    ExchangeStress,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ExchangeStressPattern {
    /// Random small-group transfers, including repeated and chained payloads.
    Random,
    /// Controlled full-duplex cases with one tile sending and receiving.
    Overlap,
    /// Paired 64-bit sends across the standard/interleaved SRAM bank matrix.
    Wide,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum AttentionMode {
    Auto,
    Flash,
    Materialized,
}

impl From<AttentionMode> for AttentionStrategy {
    fn from(value: AttentionMode) -> Self {
        match value {
            AttentionMode::Auto => Self::Automatic,
            AttentionMode::Flash => Self::Flash,
            AttentionMode::Materialized => Self::Materialized,
        }
    }
}

impl Workload {
    fn is_benchmark(self) -> bool {
        matches!(
            self,
            Self::GemmBenchmark
                | Self::SiglipMlpBenchmark
                | Self::SiglipAttentionBenchmark
                | Self::AttentionSmoke
        )
    }
}

const SIGLIP_MLP_BATCH: u32 = 4;
const SIGLIP_MLP_TOKENS: u32 = 729;
const SIGLIP_MLP_DIMENSION: u32 = 1152;
const SIGLIP_MLP_HIDDEN_DIMENSION: u32 = 4304;
const SIGLIP_ATTENTION_HEADS: u32 = 16;
const SIGLIP_ATTENTION_TOKENS: u32 = 729;
const SIGLIP_ATTENTION_HEAD_DIMENSION: u32 = 72;
const GEMM_BENCHMARK_ROWS: u32 = 131_072;
const GEMM_BENCHMARK_INNER: u32 = 64;
const GEMM_BENCHMARK_COLUMNS: u32 = 64;

fn mlp_weight_name(blocks: u32, block: u32, projection: u32) -> String {
    if blocks == 1 {
        format!("right.{projection}")
    } else {
        format!("right.{block}.{projection}")
    }
}

fn parse_gemm_plan_constraint(value: &str) -> Result<GemmPlanConstraint, String> {
    let fields = value.split(':').collect::<Vec<_>>();
    let [
        operation,
        grid,
        result_grid,
        columns,
        memory,
        orientation,
        reduction,
        local,
    ] = fields.as_slice()
    else {
        return Err("expected OP:RxCxK:RRxRC:C:MEMORY:ORIENTATION:REDUCTION:LOCAL".into());
    };
    let grid = grid
        .split('x')
        .map(|field| field.parse::<u16>().map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let [row_partitions, column_partitions, inner_partitions] = grid.as_slice() else {
        return Err("GEMM grid must be ROWSxCOLUMNSxINNER".into());
    };
    let result_grid = result_grid
        .split('x')
        .map(|field| field.parse::<u16>().map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let [result_row_partitions, result_column_partitions] = result_grid.as_slice() else {
        return Err("result grid must be ROWSxCOLUMNS".into());
    };
    let nonzero = [
        *row_partitions,
        *column_partitions,
        *inner_partitions,
        *result_row_partitions,
        *result_column_partitions,
    ];
    if nonzero.contains(&0) {
        return Err("GEMM and result grid partitions must be nonzero".into());
    }
    let output_column_block = columns.parse::<u32>().map_err(|error| error.to_string())?;
    if output_column_block == 0 {
        return Err("GEMM output column block must be nonzero".into());
    }
    Ok(GemmPlanConstraint {
        source_operation: operation
            .parse::<u32>()
            .map_err(|error| error.to_string())?,
        orientation: match *orientation {
            "normal" => GemmOrientation::Normal,
            "swapped" => GemmOrientation::Swapped,
            _ => return Err("orientation must be normal or swapped".into()),
        },
        row_partitions: *row_partitions,
        column_partitions: *column_partitions,
        inner_partitions: *inner_partitions,
        result_row_partitions: *result_row_partitions,
        result_column_partitions: *result_column_partitions,
        output_column_block,
        weight_memory_class: match *memory {
            "standard" => MemoryClass::Standard,
            "interleaved" => MemoryClass::Interleaved,
            _ => return Err("memory must be standard or interleaved".into()),
        },
        reduction_staging: match *reduction {
            "complete" => ReductionStaging::Complete,
            "streamed" => ReductionStaging::Streamed,
            _ => return Err("reduction must be complete or streamed".into()),
        },
        local_weight_staging: match *local {
            "direct" => LocalOperandStaging::Direct,
            "match-remote" => LocalOperandStaging::MatchRemote,
            _ => return Err("local staging must be direct or match-remote".into()),
        },
    })
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    ipu_runtime::init_tracing();
    let arguments = Arguments::parse();
    if arguments.diagnostic_run && matches!(arguments.workload, Workload::Diagnostic) {
        bail!("--diagnostic-run requires a computational workload");
    }
    if arguments.diagnostic_run && arguments.diagnostic_samples == 0 {
        bail!("--diagnostic-samples must be nonzero");
    }
    if arguments.exchange_replay_phase.is_some() && arguments.exchange_replay_samples == 0 {
        bail!("--exchange-replay-samples must be nonzero");
    }
    if arguments.diagnostic_run && arguments.profile_output.is_some() {
        bail!("--diagnostic-run cannot be combined with --profile-output");
    }
    if !arguments.workload.is_benchmark() && arguments.profile_output.is_some() {
        bail!("--profile-output requires a benchmark workload");
    }
    if arguments.profile_output.is_some() && !arguments.profiling.records_steps() {
        bail!("--profile-output requires --profiling full");
    }
    if arguments.exchange_diagnostic_case.is_some()
        && !matches!(arguments.workload, Workload::ExchangeStress)
    {
        bail!("--exchange-diagnostic-case requires --workload exchange-stress");
    }
    if !matches!(arguments.workload, Workload::SiglipMlpBenchmark)
        && (arguments.mlp_batch != SIGLIP_MLP_BATCH
            || arguments.mlp_tokens != SIGLIP_MLP_TOKENS
            || arguments.mlp_dim != SIGLIP_MLP_DIMENSION
            || arguments.mlp_hidden_dim.is_some()
            || arguments.mlp_blocks != 1)
    {
        bail!("--mlp-* shape options require --workload siglip-mlp-benchmark");
    }
    if !matches!(arguments.workload, Workload::GemmBenchmark)
        && (arguments.benchmark_rows != GEMM_BENCHMARK_ROWS
            || arguments.benchmark_inner != GEMM_BENCHMARK_INNER
            || arguments.benchmark_columns != GEMM_BENCHMARK_COLUMNS)
    {
        bail!("--benchmark-* shape options require --workload gemm-benchmark");
    }
    if !matches!(arguments.workload, Workload::SiglipAttentionBenchmark)
        && (arguments.attention_batch != 1 || arguments.attention_heads != SIGLIP_ATTENTION_HEADS)
    {
        bail!("--attention-* shape options require --workload siglip-attention-benchmark");
    }
    let active_tiles = u16::try_from(arguments.tiles).context("tile count exceeds u16")?;
    let attention_streams = arguments
        .attention_batch
        .checked_mul(arguments.attention_heads)
        .context("attention batch/head count overflow")?;
    if attention_streams == 0 {
        bail!("--attention-batch and --attention-heads must be nonzero");
    }
    let benchmark_rows = arguments.benchmark_rows;
    let mlp_hidden_dim = arguments.mlp_hidden_dim.map_or_else(
        || {
            if arguments.mlp_dim == SIGLIP_MLP_DIMENSION {
                return Ok(SIGLIP_MLP_HIDDEN_DIMENSION);
            }
            arguments
                .mlp_dim
                .checked_mul(4)
                .context("MLP dimension overflow")
        },
        Ok,
    )?;
    let runtime_source = arguments.runtime_source.clone().unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../device/static_runtime.S")
    });
    let bootloader = arguments
        .bootloader
        .clone()
        .unwrap_or_else(|| arguments.sdk.join("bin/ipu/tile_bootloader_cc_ipu21.elf"));
    if matches!(arguments.workload, Workload::ExchangeStress) {
        if arguments.reuse_package {
            bail!(
                "--reuse-package is not supported by exchange-stress because its manifest is generated with the package"
            );
        }
        let toolchain = Toolchain::from_sdk(&arguments.sdk);
        let stress = if matches!(arguments.exchange_pattern, ExchangeStressPattern::Wide) {
            exchange_stress::build_wide(
                active_tiles,
                arguments.exchange_wide_first_case,
                arguments.exchange_cases,
                arguments.exchange_max_words,
                !arguments.exchange_skip_validation,
                arguments.exchange_wide_receiver_mask,
                arguments.exchange_wide_explicit_config,
                arguments.exchange_wide_all_active,
                arguments.exchange_wide_pairs,
                arguments.exchange_wide_source,
                arguments.exchange_wide_first_destination,
                &toolchain,
                &runtime_source,
            )?
        } else {
            exchange_stress::build(
                arguments.exchange_seed,
                active_tiles,
                arguments.exchange_cases,
                arguments.exchange_max_words,
                arguments.exchange_max_transfers,
                arguments.exchange_compute_delay,
                matches!(arguments.exchange_pattern, ExchangeStressPattern::Overlap),
                &toolchain,
                &runtime_source,
            )?
        };
        if arguments.exchange_diagnostics {
            eprintln!(
                "{}",
                stress.static_diagnostic(arguments.exchange_diagnostic_case.unwrap_or(0))?
            );
        }
        write_package(&stress.application, &arguments.package)?;
        let configuration = fs::read(&arguments.configuration)
            .with_context(|| format!("read {}", arguments.configuration.display()))?;
        let bootloader_bytes =
            fs::read(&bootloader).with_context(|| format!("read {}", bootloader.display()))?;
        retry_after_reset(&arguments.sdk, || {
            let runtime = open_and_load_once(
                &arguments.device,
                &configuration,
                &stress.application,
                &bootloader_bytes,
                stress.application.host_exchange.startup_mark,
            )?;
            let mut session = runtime.host_session(&stress.application)?;
            session.start()?;
            std::thread::scope(|scope| {
                let (finished_tx, finished_rx) = mpsc::channel();
                let watchdog_timeout = Duration::from_secs(arguments.timeout_seconds);
                let stress_ref = &stress;
                let runtime_ref = &runtime;
                let watchdog = scope.spawn(move || {
                    if finished_rx.recv_timeout(watchdog_timeout).is_err() {
                        eprintln!(
                            "exchange watchdog: {}",
                            stress_ref.live_exchange_state(runtime_ref)
                        );
                    }
                });
                let result = (|| {
                    let call = session.invoke_streaming_deferred("run", &[0; 4])?;
                    let output = session.collect(&call)?;
                    diagnose_completion(
                        &runtime,
                        &stress.application,
                        Duration::from_secs(arguments.timeout_seconds),
                    )?;
                    stress.validate_readbacks(&output)
                })();
                let _ = finished_tx.send(());
                watchdog
                    .join()
                    .map_err(|_| anyhow::anyhow!("exchange watchdog panicked"))?;
                result
            })
            .with_context(|| stress.failure_context(&runtime))
        })?;
        println!(
            "package={} seed={:#x} exchangeCases={} hardwareTest=PASS",
            arguments.package.display(),
            arguments.exchange_seed,
            arguments.exchange_cases
        );
        return Ok(());
    }
    if let Some(path) = &arguments.replay_exchange_schedule {
        let phase = arguments
            .exchange_replay_phase
            .expect("clap requires a replay phase");
        let input = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let snapshot: ipu_codegen::ExchangeScheduleSnapshot =
            serde_json::from_reader(std::io::BufReader::new(input))
                .with_context(|| format!("read {}", path.display()))?;
        if snapshot.schema_version != ipu_codegen::EXCHANGE_SCHEDULE_SNAPSHOT_VERSION {
            bail!(
                "exchange schedule schema {} does not match supported schema {}",
                snapshot.schema_version,
                ipu_codegen::EXCHANGE_SCHEDULE_SNAPSHOT_VERSION
            );
        }
        let replay = exchange_stress::build_schedule_phase_replay(
            &snapshot,
            phase,
            arguments.exchange_replay_first_transfer,
            arguments.exchange_replay_transfer_limit,
            &Toolchain::from_sdk(&arguments.sdk),
            &runtime_source,
        )?;
        execute_exchange_replay(&arguments, &replay, &bootloader)?;
        return Ok(());
    }
    let mut graph = ComputeGraph::default();
    let mut search_domain =
        PlannerSearchDomain::default().with_attention_strategy(arguments.attention_strategy.into());
    for constraint in &arguments.gemm_plan_constraint {
        search_domain = search_domain.with_gemm_plan_constraint(*constraint);
    }
    if matches!(arguments.workload, Workload::SiglipAttentionBenchmark) {
        search_domain =
            search_domain.with_operator_precisions(OperatorClass::Gemm, [Precision::F16]);
    }
    let mut pipeline = PipelineConfig::new(active_tiles).with_search_domain(search_domain);
    pipeline.profiling = arguments.profiling;
    pipeline = pipeline.with_exchange_schedule_finalists(arguments.exchange_schedule_finalists);
    if let Some(kib) = arguments.tile_memory_budget_kib {
        let bytes = kib.checked_mul(1024).context("tile SRAM budget overflow")?;
        pipeline = pipeline.with_tile_memory_budget(bytes);
    }
    pipeline.exchange_diagnostics = arguments.exchange_diagnostics;
    if arguments.stream_conversions {
        pipeline.conversion_streaming = ipu_codegen::ConversionStreamingPolicy::Always;
    } else if arguments.materialize_conversions {
        pipeline.conversion_streaming = ipu_codegen::ConversionStreamingPolicy::Never;
    }
    if matches!(
        arguments.workload,
        Workload::GemmSmoke | Workload::BatchedGemmSmoke
    ) {
        let batch = if matches!(arguments.workload, Workload::BatchedGemmSmoke) {
            3
        } else {
            1
        };
        let left = graph.host_input("left", [batch, u32::from(active_tiles), 64])?;
        let right = graph.parameter("right", [1, 64, u32::from(active_tiles) * 64])?;
        let output = graph.gemm(left, right)?;
        graph.set_outputs([output])?;
        pipeline = pipeline
            .with_input(
                left,
                TensorFormat {
                    precision: Precision::F16,
                    layout: Layout::amp_left(64, active_tiles),
                },
            )
            .with_input(
                right,
                TensorFormat {
                    precision: Precision::F16,
                    layout: Layout::block_major_matrix(64, active_tiles),
                },
            );
    } else if matches!(arguments.workload, Workload::MlpSmoke) {
        let left = graph.host_input("left", [u32::from(active_tiles), 64])?;
        let right0 = graph.parameter("right.0", [64, 64])?;
        let right1 = graph.parameter("right.1", [64, 64])?;
        let hidden = graph.gemm(left, right0)?;
        let hidden = graph.gelu(hidden)?;
        let output = graph.gemm(hidden, right1)?;
        let output = graph.gelu(output)?;
        graph.set_outputs([output])?;
        let left_format = TensorFormat {
            precision: Precision::F16,
            layout: Layout::amp_left(64, active_tiles),
        };
        let right_format = TensorFormat {
            precision: Precision::F16,
            layout: Layout::block_major_matrix(64, active_tiles),
        };
        pipeline = pipeline
            .with_input(left, left_format)
            .with_input(right0, right_format.clone())
            .with_input(right1, right_format);
    } else if matches!(
        arguments.workload,
        Workload::AttentionSmoke | Workload::SiglipAttentionBenchmark
    ) {
        if matches!(arguments.workload, Workload::SiglipAttentionBenchmark) {
            let model_width = arguments
                .attention_heads
                .checked_mul(SIGLIP_ATTENTION_HEAD_DIMENSION)
                .context("attention model width overflow")?;
            let input = graph.host_input(
                "input",
                [
                    arguments.attention_batch,
                    SIGLIP_ATTENTION_TOKENS,
                    model_width,
                ],
            )?;
            let query_weights = graph.parameter("query.weight", [model_width, model_width])?;
            let key_weights = graph.parameter("key.weight", [model_width, model_width])?;
            let value_weights = graph.parameter("value.weight", [model_width, model_width])?;
            let query_projection = graph.gemm(input, query_weights)?;
            let key_projection = graph.gemm(input, key_weights)?;
            let value_projection = graph.gemm(input, value_weights)?;
            let query = graph.split_heads(query_projection, arguments.attention_heads)?;
            let key = graph.split_heads(key_projection, arguments.attention_heads)?;
            let value = graph.split_heads(value_projection, arguments.attention_heads)?;
            let output = graph.flash_attention(query, key, value)?;
            graph.set_outputs([output])?;
            pipeline = pipeline
                .with_automatic_input(input, Precision::F16)
                .with_automatic_input(query_weights, Precision::F16)
                .with_automatic_input(key_weights, Precision::F16)
                .with_automatic_input(value_weights, Precision::F16);
        } else {
            let (heads, query_rows, key_rows) = (4, 17, 19);
            let query_dimension = SIGLIP_ATTENTION_HEAD_DIMENSION;
            let value_dimension = SIGLIP_ATTENTION_HEAD_DIMENSION;
            if active_tiles < heads as u16 {
                bail!("attention workload requires at least {heads} tiles");
            }
            let query = graph.host_input("query", [heads, query_rows, query_dimension])?;
            let key = graph.host_input("key", [heads, key_rows, query_dimension])?;
            let value = graph.host_input("value", [heads, key_rows, value_dimension])?;
            let output = graph.flash_attention(query, key, value)?;
            graph.set_outputs([output])?;
            let heads = u16::try_from(heads)?;
            let query_partitions = u16::try_from(query_rows)?.min(active_tiles / heads).max(1);
            let key_partitions = u16::try_from(key_rows.div_ceil(64))?;
            pipeline = pipeline
                .with_input(
                    query,
                    TensorFormat {
                        precision: Precision::F16,
                        layout: Layout::attention_output(heads, query_partitions),
                    },
                )
                .with_input(
                    key,
                    TensorFormat {
                        precision: Precision::F16,
                        layout: Layout::attention_output(heads, key_partitions),
                    },
                )
                .with_input(
                    value,
                    TensorFormat {
                        precision: Precision::F16,
                        layout: Layout::attention_output(heads, key_partitions),
                    },
                );
            pipeline.conversion_streaming = ipu_codegen::ConversionStreamingPolicy::Never;
        }
    } else if matches!(arguments.workload, Workload::SiglipMlpBenchmark) {
        validate_mlp_benchmark_shape(
            arguments.mlp_batch,
            arguments.mlp_tokens,
            arguments.mlp_dim,
            mlp_hidden_dim,
        )?;
        if arguments.mlp_blocks == 0 {
            bail!("--mlp-blocks must be nonzero");
        }
        let left = graph.host_input(
            "left",
            [arguments.mlp_batch, arguments.mlp_tokens, arguments.mlp_dim],
        )?;
        let mut right0 = Vec::with_capacity(arguments.mlp_blocks as usize);
        let mut right1 = Vec::with_capacity(arguments.mlp_blocks as usize);
        for block in 0..arguments.mlp_blocks {
            right0.push(graph.parameter(
                mlp_weight_name(arguments.mlp_blocks, block, 0),
                [1, arguments.mlp_dim, mlp_hidden_dim],
            )?);
            right1.push(graph.parameter(
                mlp_weight_name(arguments.mlp_blocks, block, 1),
                [1, mlp_hidden_dim, arguments.mlp_dim],
            )?);
        }
        let output = if arguments.mlp_blocks == 1 {
            let hidden = graph.gemm(left, right0[0])?;
            let hidden = graph.gelu(hidden)?;
            graph.gemm(hidden, right1[0])?
        } else {
            let right0_sequence = graph.value_sequence("MLP up weights", right0.clone())?;
            let right1_sequence = graph.value_sequence("MLP down weights", right1.clone())?;
            graph.repeat(
                arguments.mlp_blocks,
                [left],
                [],
                [right0_sequence, right1_sequence],
                |body, arguments| {
                    let hidden = body.gemm(arguments.carried[0], arguments.iterated[0])?;
                    let hidden = body.gelu(hidden)?;
                    Ok(vec![body.gemm(hidden, arguments.iterated[1])?])
                },
            )?[0]
        };
        graph.set_outputs([output])?;
        pipeline = pipeline.with_automatic_input(left, Precision::F16);
        for weight in right0.into_iter().chain(right1) {
            pipeline = pipeline.with_automatic_input(weight, Precision::F16);
        }
    } else if matches!(arguments.workload, Workload::GemmBenchmark) {
        validate_benchmark_shape(
            benchmark_rows,
            arguments.benchmark_inner,
            arguments.benchmark_columns,
        )?;
        let left = graph.host_input("left", [benchmark_rows, arguments.benchmark_inner])?;
        let right = graph.parameter(
            "right",
            [arguments.benchmark_inner, arguments.benchmark_columns],
        )?;
        let output = graph.gemm(left, right)?;
        graph.set_outputs([output])?;
        pipeline = pipeline
            .with_automatic_input(left, Precision::F16)
            .with_automatic_input(right, Precision::F16);
    }
    let package_config = PackageConfig {
        toolchain: Toolchain::from_sdk(&arguments.sdk),
        kernel_source_directory: runtime_source
            .parent()
            .expect("runtime source has no parent directory")
            .to_owned(),
        runtime_source,
        pipeline,
    };
    let mut compiled_package = None;
    let diagnostic_package = if arguments.diagnostic_run {
        let package = build_diagnostic_package(&graph, &package_config)?;
        write_package(&package.application, &arguments.package)?;
        Some(package)
    } else {
        if !arguments.reuse_package {
            let package = build_package(&graph, &package_config)?;
            write_package(&package.application, &arguments.package)?;
            compiled_package = Some(package);
        }
        None
    };
    if let Some(path) = &arguments.export_exchange_schedule {
        let compiled = compiled_package
            .as_ref()
            .context("--export-exchange-schedule requires a newly compiled package")?;
        let output =
            fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
        serde_json::to_writer(std::io::BufWriter::new(output), &compiled.exchange_schedule)
            .with_context(|| format!("write {}", path.display()))?;
        let transfers = compiled
            .exchange_schedule
            .phases
            .iter()
            .map(|phase| phase.transfers.len())
            .sum::<usize>();
        println!(
            "exchangeSchedule={} tiles={} phases={} transfers={}",
            path.display(),
            compiled.exchange_schedule.tile_count,
            compiled.exchange_schedule.phases.len(),
            transfers
        );
        return Ok(());
    }
    if let Some(phase) = arguments.exchange_replay_phase {
        let compiled = compiled_package
            .as_ref()
            .context("--exchange-replay-phase requires a newly compiled package")?;
        let replay = exchange_stress::build_phase_replay(
            compiled,
            phase,
            &package_config.toolchain,
            &package_config.runtime_source,
        )?;
        execute_exchange_replay(&arguments, &replay, &bootloader)?;
        return Ok(());
    }
    let application = Application::read(
        fs::File::open(&arguments.package)
            .with_context(|| format!("open {}", arguments.package.display()))?,
    )?;
    if arguments.inspect_exchanges {
        println!(
            "{}",
            inspect_exchange_rows(&application, arguments.inspect_exchange_tile)?
        );
        return Ok(());
    }

    let configuration = fs::read(&arguments.configuration)
        .with_context(|| format!("read {}", arguments.configuration.display()))?;
    let bootloader_bytes =
        fs::read(&bootloader).with_context(|| format!("read {}", bootloader.display()))?;
    let startup_mark = if matches!(arguments.workload, Workload::Diagnostic) {
        0
    } else {
        application.host_exchange.startup_mark
    };
    retry_after_reset(&arguments.sdk, || {
        let runtime = open_and_load_once(
            &arguments.device,
            &configuration,
            &application,
            &bootloader_bytes,
            startup_mark,
        )?;
        if let Some(package) = &diagnostic_package {
            diagnostic::run(
                &runtime,
                &graph,
                package,
                arguments.diagnostic_samples,
                arguments.diagnostic_atol,
                arguments.diagnostic_rtol,
                Duration::from_secs(arguments.timeout_seconds),
            )?;
        } else if !matches!(arguments.workload, Workload::Diagnostic) {
            if matches!(
                arguments.workload,
                Workload::GemmSmoke | Workload::BatchedGemmSmoke
            ) {
                run_gemm(
                    &runtime,
                    &application,
                    active_tiles,
                    if matches!(arguments.workload, Workload::BatchedGemmSmoke) {
                        3
                    } else {
                        1
                    },
                    arguments.timeout_seconds,
                )?;
            } else if matches!(arguments.workload, Workload::MlpSmoke) {
                run_mlp_chain(
                    &runtime,
                    &application,
                    active_tiles,
                    arguments.timeout_seconds,
                )?;
            } else if matches!(arguments.workload, Workload::SiglipAttentionBenchmark) {
                run_projected_attention_benchmark(
                    &runtime,
                    &application,
                    arguments.attention_batch,
                    arguments.attention_heads,
                    arguments.clock_hz,
                    arguments.timeout_seconds,
                    arguments.profile_output.as_deref(),
                    arguments.profiling,
                )?;
            } else if matches!(arguments.workload, Workload::AttentionSmoke) {
                let (heads, query_rows, key_rows) = (4, 17, 19);
                let query_dimension = SIGLIP_ATTENTION_HEAD_DIMENSION;
                let value_dimension = SIGLIP_ATTENTION_HEAD_DIMENSION;
                run_attention_smoke(
                    &runtime,
                    &application,
                    heads,
                    query_rows,
                    key_rows,
                    query_dimension,
                    value_dimension,
                    query_rows,
                    arguments.clock_hz,
                    arguments.timeout_seconds,
                    arguments.profile_output.as_deref(),
                    arguments.profiling,
                )?;
            } else if matches!(arguments.workload, Workload::SiglipMlpBenchmark) {
                run_siglip_mlp_benchmark(
                    &runtime,
                    &application,
                    &graph,
                    compiled_package.as_ref().context(
                        "large MLP numerical validation requires rebuilding the package without --reuse-package",
                    )?,
                    active_tiles,
                    arguments.mlp_batch,
                    arguments.mlp_tokens,
                    arguments.mlp_dim,
                    mlp_hidden_dim,
                    arguments.mlp_blocks,
                    arguments.clock_hz,
                    arguments.timeout_seconds,
                    arguments.profile_output.as_deref(),
                    arguments.profiling,
                )?;
            } else {
                run_gemm_benchmark(
                    &runtime,
                    &application,
                    active_tiles,
                    benchmark_rows,
                    arguments.benchmark_inner,
                    arguments.benchmark_columns,
                    arguments.clock_hz,
                    arguments.timeout_seconds,
                    arguments.profile_output.as_deref(),
                    arguments.profiling,
                )?;
            }
        } else {
            diagnose_completion(
                &runtime,
                &application,
                Duration::from_secs(arguments.timeout_seconds),
            )?;
        }
        Ok(())
    })?;
    println!(
        "package={} tiles={} hardwareTest=PASS",
        arguments.package.display(),
        application.tiles.len()
    );
    Ok(())
}

fn open_and_load_once(
    device: &str,
    configuration: &[u8],
    application: &Application,
    bootloader: &[u8],
    final_mark: u32,
) -> Result<Runtime> {
    let runtime = Runtime::open(device, configuration)?;
    runtime.load(application, bootloader, final_mark)?;
    Ok(runtime)
}

fn retry_after_reset<T>(sdk: &Path, mut attempt: impl FnMut() -> Result<T>) -> Result<T> {
    match attempt() {
        Ok(value) => Ok(value),
        Err(error)
            if error
                .chain()
                .any(|cause| matches!(cause.downcast_ref(), Some(DriverError::Timeout(_)))) =>
        {
            tracing::warn!(%error, "device timed out; resetting and retrying once");
            let reset = sdk.join("bin/gc-reset");
            let status = Command::new(&reset)
                .arg("-m")
                .status()
                .with_context(|| format!("run {} -m", reset.display()))?;
            if !status.success() {
                bail!("{} -m exited with {status}", reset.display());
            }
            attempt().context("hardware execution failed after gc-reset -m")
        }
        Err(error) => Err(error),
    }
}

fn execute_exchange_replay(
    arguments: &Arguments,
    replay: &exchange_stress::PhaseReplayPackage,
    bootloader: &Path,
) -> Result<()> {
    write_package(&replay.application, &arguments.package)?;
    let configuration = fs::read(&arguments.configuration)
        .with_context(|| format!("read {}", arguments.configuration.display()))?;
    let bootloader_bytes =
        fs::read(bootloader).with_context(|| format!("read {}", bootloader.display()))?;
    retry_after_reset(&arguments.sdk, || {
        let runtime = open_and_load_once(
            &arguments.device,
            &configuration,
            &replay.application,
            &bootloader_bytes,
            replay.application.host_exchange.startup_mark,
        )?;
        let mut session = runtime.host_session(&replay.application)?;
        session.start()?;
        let mut serviced = false;
        let executed = session
            .invoke_streaming_deferred_with_poll("run", &[0; 4], |device| {
                replay
                    .service_readback(device, arguments.exchange_replay_samples, &mut serviced)
                    .map_err(|error| DriverError::Invalid(error.to_string()))
            })
            .inspect_err(|error| {
                tracing::error!(
                    %error,
                    device = %device_failure_diagnostics(&runtime, &replay.application),
                    "exchange replay failed"
                );
            })?;
        if !serviced {
            bail!("exchange replay completed without reaching its readback trap");
        }
        let _ = session.collect(&executed)?;
        diagnose_completion(
            &runtime,
            &replay.application,
            Duration::from_secs(arguments.timeout_seconds),
        )
    })?;
    println!(
        "package={} exchangeReplayPhase={} hardwareTest=PASS",
        arguments.package.display(),
        replay.phase
    );
    Ok(())
}

fn run_gemm(
    runtime: &Runtime,
    application: &Application,
    active_tiles: u16,
    batch: u32,
    timeout_seconds: u64,
) -> Result<()> {
    let left = application
        .inputs
        .iter()
        .find(|binding| binding.name == "left")
        .cloned()
        .context("GEMM package has no left input binding")?;
    let right = application
        .weights
        .iter()
        .find(|binding| binding.name == "right")
        .cloned()
        .context("GEMM package has no right weight binding")?;
    let left_bytes = packed_binding(&left, |logical_tile, linear, elements| {
        let (batch_index, inner) = amp_matrix_coordinates(
            AmpOrder::Left,
            Precision::F16,
            batch,
            elements / batch,
            linear,
        )?;
        let selected_inner = (batch_index * 7 + u32::from(logical_tile)) % 64;
        Ok(if selected_inner == inner { 0x3c00 } else { 0 })
    })?;
    let right_bytes = packed_binding(&right, |logical_tile, linear, elements| {
        let (inner, column) = block_major_matrix_coordinates(
            BlockMajorOrder::Matrix {
                row_block: 64,
                column_block: 16,
            },
            Precision::F16,
            64,
            elements / 64,
            linear,
        )?;
        Ok(gemm_right_value(
            inner,
            u32::from(logical_tile) * 64 + column,
        ))
    })?;
    if left.slices.len() != usize::from(active_tiles)
        || right.slices.len() != usize::from(active_tiles)
    {
        bail!("GEMM bindings do not cover every active tile");
    }
    let output = run_initialized_program(
        runtime,
        application,
        &right_bytes,
        &left_bytes,
        timeout_seconds,
    )?;
    verify_gemm_output(application, active_tiles, batch, &output)
}

fn run_mlp_chain(
    runtime: &Runtime,
    application: &Application,
    active_tiles: u16,
    timeout_seconds: u64,
) -> Result<()> {
    let binding = |name: &str, bindings: &[Binding]| {
        bindings
            .iter()
            .find(|binding| binding.name == name)
            .cloned()
            .with_context(|| format!("MLP package has no {name} binding"))
    };
    let left = binding("left", &application.inputs)?;
    let right0 = binding("right.0", &application.weights)?;
    let right1 = binding("right.1", &application.weights)?;
    let left_bytes = packed_binding(&left, |logical_tile, linear, elements| {
        let (_, inner) =
            amp_matrix_coordinates(AmpOrder::Left, Precision::F16, 1, elements, linear)?;
        Ok(mlp_smoke_value(
            MLP_INPUT_SEED,
            u64::from(logical_tile) * u64::from(MLP_SMOKE_WIDTH) + u64::from(inner),
            MLP_INPUT_STANDARD_DEVIATION,
        ))
    })?;
    let right0_bytes = packed_binding(&right0, |logical_tile, linear, elements| {
        let (inner, column) = block_major_matrix_coordinates(
            BlockMajorOrder::Matrix {
                row_block: 64,
                column_block: 16,
            },
            Precision::F16,
            64,
            elements / 64,
            linear,
        )?;
        let column = u32::from(logical_tile) * 64 + column;
        Ok(if column < 64 {
            mlp_smoke_value(
                MLP_FIRST_WEIGHT_SEED,
                u64::from(inner) * u64::from(MLP_SMOKE_WIDTH) + u64::from(column),
                MLP_WEIGHT_STANDARD_DEVIATION,
            )
        } else {
            0
        })
    })?;
    let right1_bytes = packed_binding(&right1, |logical_tile, linear, elements| {
        let (inner, column) = block_major_matrix_coordinates(
            BlockMajorOrder::Matrix {
                row_block: 64,
                column_block: 16,
            },
            Precision::F16,
            64,
            elements / 64,
            linear,
        )?;
        let column = u32::from(logical_tile) * 64 + column;
        Ok(if column < 64 {
            mlp_smoke_value(
                MLP_SECOND_WEIGHT_SEED,
                u64::from(inner) * u64::from(MLP_SMOKE_WIDTH) + u64::from(column),
                MLP_WEIGHT_STANDARD_DEVIATION,
            )
        } else {
            0
        })
    })?;
    let mut weights = Vec::with_capacity(right0_bytes.len() + right1_bytes.len());
    weights.extend_from_slice(&right0_bytes);
    weights.extend_from_slice(&right1_bytes);

    let output =
        run_initialized_program(runtime, application, &weights, &left_bytes, timeout_seconds)?;
    verify_mlp_output(application, active_tiles, &output)
}

fn run_projected_attention_benchmark(
    runtime: &Runtime,
    application: &Application,
    batch: u32,
    heads: u32,
    clock_hz: u64,
    timeout_seconds: u64,
    profile_output: Option<&Path>,
    profiling: ProfilingMode,
) -> Result<()> {
    let streams = batch
        .checked_mul(heads)
        .context("attention stream overflow")?;
    let model_width = heads
        .checked_mul(SIGLIP_ATTENTION_HEAD_DIMENSION)
        .context("attention model width overflow")?;
    let input = application
        .inputs
        .iter()
        .find(|binding| binding.name == "input")
        .context("projected attention package has no input binding")?;
    let input_bytes = filled_f16_binding(input, f32_to_half(1.0))?;
    let weight_bits = f32_to_half(1.0 / model_width as f32);
    let mut weights = Vec::new();
    for name in ["query.weight", "key.weight", "value.weight"] {
        let binding = application
            .weights
            .iter()
            .find(|binding| binding.name == name)
            .with_context(|| format!("projected attention package has no {name} binding"))?;
        weights.extend_from_slice(&filled_f16_binding(binding, weight_bits)?);
    }
    let actual = run_initialized_program(
        runtime,
        application,
        &weights,
        &input_bytes,
        timeout_seconds,
    )?;
    if profiling.records_steps() {
        write_profile(application, &actual, clock_hz, profile_output)?;
    }

    let output = application
        .outputs
        .iter()
        .find(|binding| binding.name == "output.0")
        .context("projected attention package has no output binding")?;
    let populated = output.slices.iter().filter(|slice| slice.size != 0).count();
    let query_partitions = u32::try_from(populated)? / streams;
    if query_partitions == 0 {
        bail!("projected attention output has no populated shards");
    }
    let expected = expected_projection_value(model_width);
    let padded_width = padded_attention_width();
    let mut maximum_error = 0.0f32;
    let mut checks = 0usize;
    let mut first_mismatch = None;
    let mut mismatches = 0usize;
    let mut mismatches_by_column = vec![0usize; SIGLIP_ATTENTION_HEAD_DIMENSION as usize];
    for stream in 0..streams {
        for partition in 0..query_partitions {
            let tile = partition * streams + stream;
            let slice = &output.slices[usize::try_from(tile)?];
            let (_, rows) = balanced_range(SIGLIP_ATTENTION_TOKENS, query_partitions, partition);
            for row in 0..rows {
                for column in 0..SIGLIP_ATTENTION_HEAD_DIMENSION {
                    let linear = u64::from(row * padded_width + column);
                    let offset = usize::try_from(slice.file_offset + linear * 4)?;
                    let observed = f32::from_le_bytes(
                        actual
                            .get(offset..offset + 4)
                            .context("projected attention output exceeds host data")?
                            .try_into()
                            .unwrap(),
                    );
                    maximum_error = maximum_error.max((observed - expected).abs());
                    if (observed - expected).abs() > 0.02 {
                        mismatches += 1;
                        mismatches_by_column[column as usize] += 1;
                        first_mismatch.get_or_insert((stream, partition, row, column, observed));
                    }
                    checks += 1;
                }
            }
        }
    }
    if maximum_error > 0.02 {
        bail!(
            "projected attention numerical verification failed: checks={checks} mismatches={mismatches} expected={expected:.6} maximumError={maximum_error:.6} firstMismatch={first_mismatch:?} mismatchesByColumn={mismatches_by_column:?}"
        );
    }
    println!(
        "attentionNumericalChecks={checks} expected={expected:.6} maxError={maximum_error:.6} numericalTest=PASS"
    );
    Ok(())
}

fn expected_projection_value(model_width: u32) -> f32 {
    half_to_f32(f32_to_half(1.0 / model_width as f32)) * model_width as f32
}

fn padded_attention_width() -> u32 {
    SIGLIP_ATTENTION_HEAD_DIMENSION.div_ceil(16) * 16
}

fn run_attention_smoke(
    runtime: &Runtime,
    application: &Application,
    heads: u32,
    query_rows: u32,
    key_rows: u32,
    query_dimension: u32,
    value_dimension: u32,
    checked_query_rows: u32,
    clock_hz: u64,
    timeout_seconds: u64,
    profile_output: Option<&Path>,
    profiling: ProfilingMode,
) -> Result<()> {
    const QUERY_SEED: u64 = 0x6174_746e_5f71;
    const KEY_SEED: u64 = 0x6174_746e_5f6b;
    const VALUE_SEED: u64 = 0x6174_746e_5f76;
    const STANDARD_DEVIATION: f32 = 0.25;

    let binding = |name: &str, bindings: &[Binding]| {
        bindings
            .iter()
            .find(|binding| binding.name == name)
            .cloned()
            .with_context(|| format!("attention package has no {name} binding"))
    };
    let query_binding = binding("query", &application.inputs)?;
    let key_binding = binding("key", &application.inputs)?;
    let value_binding = binding("value", &application.inputs)?;
    let populated = query_binding
        .slices
        .iter()
        .filter(|slice| slice.size != 0)
        .count();
    let query_partitions = u32::try_from(populated)? / heads;
    if query_partitions == 0 {
        bail!("attention query has no populated shards");
    }
    let padded_query_dimension = query_dimension.div_ceil(16) * 16;
    let padded_value_dimension = value_dimension.div_ceil(16) * 16;
    let query_bytes = packed_binding(&query_binding, |logical_tile, linear, _elements| {
        let tile = u32::from(logical_tile);
        let head = tile % heads;
        let partition = tile / heads;
        let local_row = linear / padded_query_dimension;
        let column = linear % padded_query_dimension;
        let (row_start, _) = balanced_range(query_rows, query_partitions, partition);
        Ok(if column < query_dimension {
            mlp_smoke_value(
                QUERY_SEED,
                u64::from(head * query_rows * query_dimension)
                    + u64::from((row_start + local_row) * query_dimension + column),
                STANDARD_DEVIATION,
            )
        } else {
            0
        })
    })?;
    let key_bytes = packed_binding(&key_binding, |logical_tile, linear, elements| {
        let tile = u32::from(logical_tile);
        let head = tile % heads;
        let partition = tile / heads;
        let (local_row, column) = if elements == 64 * padded_query_dimension {
            amp_matrix_coordinates(
                AmpOrder::TransposedRight,
                Precision::F16,
                64,
                padded_query_dimension,
                linear,
            )?
        } else {
            (
                linear / padded_query_dimension,
                linear % padded_query_dimension,
            )
        };
        let (row_start, _) = balanced_range(key_rows, key_rows.div_ceil(64), partition);
        let row = row_start + local_row;
        Ok(if row < key_rows && column < query_dimension {
            mlp_smoke_value(
                KEY_SEED,
                u64::from(head * key_rows * query_dimension + row * query_dimension + column),
                STANDARD_DEVIATION,
            )
        } else {
            0
        })
    })?;
    let value_bytes = packed_binding(&value_binding, |logical_tile, linear, elements| {
        let tile = u32::from(logical_tile);
        let head = tile % heads;
        let partition = tile / heads;
        let (local_row, column) = if elements == 64 * padded_value_dimension {
            block_major_matrix_coordinates(
                BlockMajorOrder::Matrix {
                    row_block: 64,
                    column_block: 16,
                },
                Precision::F16,
                64,
                padded_value_dimension,
                linear,
            )?
        } else {
            (
                linear / padded_value_dimension,
                linear % padded_value_dimension,
            )
        };
        let (row_start, _) = balanced_range(key_rows, key_rows.div_ceil(64), partition);
        let row = row_start + local_row;
        Ok(if row < key_rows && column < value_dimension {
            mlp_smoke_value(
                VALUE_SEED,
                u64::from(head * key_rows * value_dimension + row * value_dimension + column),
                STANDARD_DEVIATION,
            )
        } else {
            0
        })
    })?;
    let mut inputs = Vec::with_capacity(query_bytes.len() + key_bytes.len() + value_bytes.len());
    inputs.extend_from_slice(&query_bytes);
    inputs.extend_from_slice(&key_bytes);
    inputs.extend_from_slice(&value_bytes);
    let actual = run_initialized_program(runtime, application, &[], &inputs, timeout_seconds)?;
    if profiling.records_steps() {
        write_profile(application, &actual, clock_hz, profile_output)?;
    }
    let output = binding("output.0", &application.outputs)?;
    if output.slices.len() < usize::try_from(heads * query_partitions)? {
        bail!("attention output has fewer shards than its query tiling");
    }

    let sample = |seed, index, width| half_to_f32(mlp_smoke_value(seed, index, width));
    let scale = 1.0 / (query_dimension as f32).sqrt();
    let mut maximum_error = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut checks = 0usize;
    let mut first_mismatch = None;
    for head in 0..heads {
        for query_row in 0..checked_query_rows.min(query_rows) {
            let mut scores = vec![0.0f32; key_rows as usize];
            for key_row in 0..key_rows {
                let mut dot = 0.0f32;
                for column in 0..query_dimension {
                    let query_index = u64::from(
                        head * query_rows * query_dimension + query_row * query_dimension + column,
                    );
                    let key_index = u64::from(
                        head * key_rows * query_dimension + key_row * query_dimension + column,
                    );
                    dot += sample(QUERY_SEED, query_index, STANDARD_DEVIATION)
                        * sample(KEY_SEED, key_index, STANDARD_DEVIATION);
                }
                scores[key_row as usize] = dot * scale;
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = scores.iter().map(|score| (*score - maximum).exp()).sum();
            for column in 0..value_dimension {
                let expected = scores
                    .iter()
                    .enumerate()
                    .map(|(key_row, score)| {
                        let value_index = u64::from(
                            head * key_rows * value_dimension
                                + key_row as u32 * value_dimension
                                + column,
                        );
                        ((*score - maximum).exp() / denominator)
                            * sample(VALUE_SEED, value_index, STANDARD_DEVIATION)
                    })
                    .sum::<f32>();
                let partition = partition_for_index(query_rows, query_partitions, query_row);
                let (row_start, _) = balanced_range(query_rows, query_partitions, partition);
                let tile = partition * heads + head;
                let slice = &output.slices[tile as usize];
                let linear = u64::from((query_row - row_start) * padded_value_dimension + column);
                let offset = usize::try_from(slice.file_offset + linear * 4)?;
                let observed = f32::from_le_bytes(actual[offset..offset + 4].try_into().unwrap());
                let error = (observed - expected).abs();
                maximum_error = maximum_error.max(error);
                squared_error += f64::from(error) * f64::from(error);
                if error > 0.012 {
                    first_mismatch.get_or_insert((head, query_row, column, expected, observed));
                }
                checks += 1;
            }
        }
    }
    let rms_error = (squared_error / checks as f64).sqrt();
    if maximum_error > 0.012 || rms_error > 0.003 {
        bail!(
            "FlashAttention numerical verification failed: checks={checks} maxError={maximum_error:.6} rmsError={rms_error:.6} firstMismatch={first_mismatch:?}"
        );
    }
    println!(
        "attentionNumericalChecks={checks} maxError={maximum_error:.6} rmsError={rms_error:.6} numericalTest=PASS"
    );
    Ok(())
}

fn balanced_range(extent: u32, partitions: u32, partition: u32) -> (u32, u32) {
    let base = extent / partitions;
    let remainder = extent % partitions;
    let start = partition * base + partition.min(remainder);
    (start, base + u32::from(partition < remainder))
}

fn partition_for_index(extent: u32, partitions: u32, index: u32) -> u32 {
    (0..partitions)
        .find(|&partition| {
            let (start, size) = balanced_range(extent, partitions, partition);
            index >= start && index < start + size
        })
        .expect("index belongs to one balanced partition")
}

fn run_initialized_program(
    runtime: &Runtime,
    application: &Application,
    weights: &[u8],
    input: &[u8],
    timeout_seconds: u64,
) -> Result<Vec<u8>> {
    let mut session = runtime.host_session(application)?;
    session.start().inspect_err(|_| {
        eprintln!(
            "startFailureDiagnostics={}",
            device_failure_diagnostics(runtime, application)
        );
    })?;
    if !application.weights.is_empty() {
        let initialized = session.invoke_streaming_deferred("initialize", weights)?;
        session.collect(&initialized)?;
    }
    let executed = session
        .invoke_streaming_deferred("run", input)
        .inspect_err(|_| {
            eprintln!(
                "runFailureDiagnostics={}",
                device_failure_diagnostics(runtime, application)
            );
        })?;
    runtime
        .device()
        .write_sync_mark(ipu_driver::pci::HSP_GS2_CONTROL, 1)?;
    diagnose_completion(runtime, application, Duration::from_secs(timeout_seconds)).with_context(
        || {
            format!(
                "deviceFailureDiagnostics={}",
                device_failure_diagnostics(runtime, application)
            )
        },
    )?;
    Ok(session.collect(&executed)?)
}

#[allow(clippy::too_many_arguments)]
fn run_gemm_benchmark(
    runtime: &Runtime,
    application: &Application,
    execution_tiles: u16,
    rows: u32,
    inner: u32,
    columns: u32,
    clock_hz: u64,
    timeout_seconds: u64,
    profile_output: Option<&Path>,
    profiling: ProfilingMode,
) -> Result<()> {
    validate_benchmark_shape(rows, inner, columns)?;
    if clock_hz == 0 {
        bail!("benchmark clock must be nonzero");
    }
    let left = application
        .inputs
        .iter()
        .find(|binding| binding.name == "left")
        .context("benchmark package has no left binding")?;
    let right = application
        .weights
        .iter()
        .find(|binding| binding.name == "right")
        .context("benchmark package has no right binding")?;
    let left_bytes = filled_f16_binding(left, 0x3c00)?;
    let right_bytes = filled_f16_binding(right, 0x3400)?;
    let output = run_initialized_program(
        runtime,
        application,
        &right_bytes,
        &left_bytes,
        timeout_seconds,
    )?;
    let maximum_absolute_error = verify_benchmark_output(application, &output, inner)?;
    if !profiling.records_overall_time() {
        println!(
            "workload=gemm-f16-r{rows}-k{inner}-c{columns} benchmark=gemm-f16 rows={rows} inner={inner} columns={columns} profiling=false maximumAbsoluteError={maximum_absolute_error:.6}"
        );
        return Ok(());
    }
    let (cycles, minimum_cycles) = benchmark_cycles(application, &output, execution_tiles)?;
    if profiling.records_steps() {
        write_profile(application, &output, clock_hz, profile_output)?;
    }
    let active_tiles = binding_tile_count(application, "output.0")?;
    let rows = u64::from(rows);
    let flops = 2.0 * rows as f64 * f64::from(inner) * f64::from(columns);
    let seconds = f64::from(cycles) / clock_hz as f64;
    let tflops = flops / seconds / 1.0e12;
    let peak_tflops = clock_hz as f64 * f64::from(execution_tiles) * 128.0 / 1.0e12;
    println!(
        "workload=gemm-f16-r{rows}-k{inner}-c{columns} benchmark=gemm-f16 rows={rows} inner={inner} columns={columns} activeTiles={active_tiles} executionTiles={execution_tiles} inputBytes={} weightBytes={} cycles={cycles} minimumTileCycles={minimum_cycles} deviceMicroseconds={:.3} tflops={tflops:.3} peakTflops={peak_tflops:.3} efficiencyPercent={:.2} maximumAbsoluteError={maximum_absolute_error:.6}",
        left_bytes.len(),
        right_bytes.len(),
        seconds * 1.0e6,
        tflops / peak_tflops * 100.0,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_siglip_mlp_benchmark(
    runtime: &Runtime,
    application: &Application,
    graph: &ComputeGraph,
    package: &CompiledPackage,
    execution_tiles: u16,
    batch: u32,
    tokens: u32,
    dimension: u32,
    hidden_dimension: u32,
    blocks: u32,
    clock_hz: u64,
    timeout_seconds: u64,
    profile_output: Option<&Path>,
    profiling: ProfilingMode,
) -> Result<()> {
    validate_mlp_benchmark_shape(batch, tokens, dimension, hidden_dimension)?;
    if clock_hz == 0 {
        bail!("benchmark clock must be nonzero");
    }
    let (host_inputs, weights, left_bytes) =
        diagnostic::prepare_inputs(graph, application, &package.inputs)?;
    let references = diagnostic::evaluate(graph, host_inputs, &package.precisions)?;

    let output =
        run_initialized_program(runtime, application, &weights, &left_bytes, timeout_seconds)?;

    let output_metadata = package
        .outputs
        .iter()
        .find(|tensor| tensor.name.as_deref() == Some("output.0"))
        .context("MLP benchmark package has no logical output storage map")?;
    let expected = references
        .get(&output_metadata.value)
        .context("MLP host reference has no graph output")?;
    let maximum_absolute_error =
        verify_logical_f16_output(application, output_metadata, &output, &expected.values)?;
    if !profiling.records_overall_time() {
        println!(
            "workload=siglip-mlp-f16-b{batch}-t{tokens}-d{dimension}-h{hidden_dimension}-n{blocks} benchmark=siglip-mlp-f16 batch={batch} tokens={tokens} dimension={dimension} hiddenDimension={hidden_dimension} blocks={blocks} biases=false profiling=false maximumAbsoluteError={maximum_absolute_error:.6}"
        );
        return Ok(());
    }
    let (cycles, minimum_cycles) = benchmark_cycles(application, &output, execution_tiles)?;
    if profiling.records_steps() {
        write_profile(application, &output, clock_hz, profile_output)?;
    }
    let active_tiles = binding_tile_count(application, "output.0")?;
    let rows = u64::from(batch) * u64::from(tokens);
    let flops =
        4.0 * rows as f64 * f64::from(dimension) * f64::from(hidden_dimension) * f64::from(blocks);
    let seconds = f64::from(cycles) / clock_hz as f64;
    let tflops = flops / seconds / 1.0e12;
    let peak_tflops = clock_hz as f64 * f64::from(execution_tiles) * 128.0 / 1.0e12;
    println!(
        "workload=siglip-mlp-f16-b{batch}-t{tokens}-d{dimension}-h{hidden_dimension}-n{blocks} benchmark=siglip-mlp-f16 batch={batch} tokens={tokens} rows={rows} dimension={dimension} hiddenDimension={hidden_dimension} blocks={blocks} biases=false activeTiles={active_tiles} executionTiles={execution_tiles} inputBytes={} weightBytes={} cycles={cycles} minimumTileCycles={minimum_cycles} deviceMicroseconds={:.3} effectiveGemmTflops={tflops:.3} peakTflops={peak_tflops:.3} efficiencyPercent={:.2} maximumAbsoluteError={maximum_absolute_error:.6}",
        left_bytes.len(),
        weights.len(),
        seconds * 1.0e6,
        tflops / peak_tflops * 100.0,
    );
    Ok(())
}

fn write_profile(
    application: &Application,
    output: &[u8],
    clock_hz: u64,
    path: Option<&Path>,
) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    let report = application.profile_report(output, clock_hz)?;
    report.write(fs::File::create(path).with_context(|| format!("create {}", path.display()))?)?;
    println!(
        "profile={} tiles={} samples={}",
        path.display(),
        report.tiles.len(),
        report
            .tiles
            .iter()
            .map(|tile| tile.samples.len())
            .sum::<usize>()
    );
    Ok(())
}

fn benchmark_cycles(
    application: &Application,
    output: &[u8],
    active_tiles: u16,
) -> Result<(u32, u32)> {
    let starts = binding_u32_values(application, output, "profile.start-cycle")?;
    let ends = binding_u32_values(application, output, "profile.end-cycle")?;
    if starts.len() != usize::from(active_tiles) || ends.len() != starts.len() {
        bail!(
            "benchmark profile bindings have {} start and {} end samples for {active_tiles} tiles",
            starts.len(),
            ends.len()
        );
    }
    let durations = starts
        .into_iter()
        .zip(ends)
        .map(|(start, end)| end.wrapping_sub(start))
        .collect::<Vec<_>>();
    let maximum = durations.iter().copied().max().unwrap_or(0);
    if maximum == 0 {
        bail!("benchmark cycle interval is zero");
    }
    Ok((maximum, durations.iter().copied().min().unwrap_or(maximum)))
}

fn validate_benchmark_shape(rows: u32, inner: u32, columns: u32) -> Result<()> {
    if rows == 0
        || inner == 0
        || columns == 0
        || !inner.is_multiple_of(64)
        || !columns.is_multiple_of(ipu_codegen::mid::AMP_COLUMN_MICRO)
    {
        bail!(
            "benchmark rows must be nonzero, inner must be a multiple of 64, and columns must be a multiple of 16"
        );
    }
    Ok(())
}

fn validate_mlp_benchmark_shape(
    batch: u32,
    tokens: u32,
    dimension: u32,
    hidden_dimension: u32,
) -> Result<()> {
    if batch == 0
        || tokens == 0
        || dimension == 0
        || hidden_dimension == 0
        || !dimension.is_multiple_of(64)
    {
        bail!(
            "MLP batch/tokens and hidden dimension must be nonzero, and model dimension must be a nonzero multiple of 64"
        );
    }
    Ok(())
}

fn verify_benchmark_output(application: &Application, bytes: &[u8], inner: u32) -> Result<f32> {
    let (binding, base) = output_binding(application, "output.0")?;
    let size = binding_size(binding);
    if size == 0 || !size.is_multiple_of(2) {
        bail!("benchmark graph output is not a nonempty F16 binding");
    }
    let start = usize::try_from(base)?;
    let end = usize::try_from(
        base.checked_add(size)
            .context("benchmark output overflow")?,
    )?;
    let output = bytes
        .get(start..end)
        .context("benchmark graph output exceeds host output")?;
    let expected = inner as f32 * 0.25;
    let mut maximum = 0.0f32;
    for raw in output.chunks_exact(2) {
        let actual = half_to_f32(u16::from_le_bytes(raw.try_into().unwrap()));
        maximum = maximum.max((actual - expected).abs());
    }
    if maximum > expected.abs() * 0.002 + 0.01 {
        bail!(
            "benchmark numerical output differs from {expected}: maximum absolute error {maximum}"
        );
    }
    Ok(maximum)
}

fn verify_logical_f16_output(
    application: &Application,
    tensor: &DiagnosticTensor,
    bytes: &[u8],
    expected: &[f32],
) -> Result<f32> {
    let (binding, base) = output_binding(application, "output.0")?;
    if tensor.precision != Precision::F16
        || expected.len() != usize::try_from(tensor.shape.elements())?
    {
        bail!("MLP benchmark output metadata is inconsistent with its reference");
    }
    let mut covered = vec![false; expected.len()];
    let mut maximum = 0.0f32;
    let mut mismatches = Vec::new();
    let mut mismatch_count = 0usize;
    let mut checked = 0usize;
    for shard in &tensor.shards {
        let slice = binding
            .slices
            .iter()
            .find(|slice| {
                slice.tile == u32::from(shard.physical_tile) && slice.tile_address == shard.address
            })
            .context("MLP output binding slice is missing")?;
        for (index, offset) in diagnostic::shard_elements(tensor, shard)? {
            let start = usize::try_from(base + slice.file_offset + u64::from(offset))?;
            let raw = bytes
                .get(start..start + 2)
                .context("MLP logical output exceeds host output")?;
            let actual = half_to_f32(u16::from_le_bytes(raw.try_into().unwrap()));
            let reference = expected[index];
            let error = (actual - reference).abs();
            checked += 1;
            maximum = maximum.max(error);
            covered[index] = true;
            if !actual.is_finite() || error > 0.03 + 0.05 * reference.abs() {
                mismatch_count += 1;
                if mismatches.len() < 16 {
                    mismatches.push((index, reference, actual, error));
                }
            }
        }
    }
    if let Some(missing) = covered.iter().position(|covered| !covered) {
        bail!("MLP output does not contain logical element {missing}");
    }
    if !mismatches.is_empty() {
        bail!(
            "MLP benchmark numerical comparison failed for {mismatch_count}/{} logical shard values (maximum absolute error {maximum}): {mismatches:?}",
            checked
        );
    }
    Ok(maximum)
}

fn binding_u32_values(application: &Application, bytes: &[u8], name: &str) -> Result<Vec<u32>> {
    let (binding, base) = output_binding(application, name)?;
    let mut values = Vec::with_capacity(binding.slices.len());
    for slice in &binding.slices {
        if slice.size != 4 {
            bail!("profile binding {name} contains a non-u32 slice");
        }
        let start = usize::try_from(
            base.checked_add(slice.file_offset)
                .context("profile offset overflow")?,
        )?;
        let end = start.checked_add(4).context("profile offset overflow")?;
        let raw = bytes
            .get(start..end)
            .with_context(|| format!("profile binding {name} exceeds host output"))?;
        values.push(u32::from_le_bytes(raw.try_into().unwrap()));
    }
    Ok(values)
}

fn output_binding<'a>(application: &'a Application, name: &str) -> Result<(&'a Binding, u64)> {
    let mut base = 0u64;
    for binding in &application.outputs {
        if binding.name == name {
            return Ok((binding, base));
        }
        base = base
            .checked_add(binding_size(binding))
            .context("host output binding offset overflow")?;
    }
    bail!("package has no {name} output binding")
}

fn binding_tile_count(application: &Application, name: &str) -> Result<u16> {
    let (binding, _) = output_binding(application, name)?;
    let tiles = binding
        .slices
        .iter()
        .map(|slice| slice.tile)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    u16::try_from(tiles).context("binding tile count exceeds u16")
}

fn binding_size(binding: &Binding) -> u64 {
    binding
        .slices
        .iter()
        .map(|slice| slice.file_offset + slice.size)
        .max()
        .unwrap_or(0)
}

fn filled_f16_binding(binding: &Binding, bits: u16) -> Result<Vec<u8>> {
    let size = binding
        .slices
        .iter()
        .map(|slice| slice.file_offset + slice.size)
        .max()
        .context("binding has no slices")?;
    if size & 1 != 0 {
        bail!("binding {} has an odd byte count", binding.name);
    }
    let pair = bits.to_le_bytes();
    let mut bytes = vec![0; usize::try_from(size)?];
    for chunk in bytes.chunks_exact_mut(2) {
        chunk.copy_from_slice(&pair);
    }
    Ok(bytes)
}

fn verify_mlp_output(application: &Application, active_tiles: u16, bytes: &[u8]) -> Result<()> {
    let output = application
        .outputs
        .iter()
        .find(|binding| binding.name == "output.0")
        .context("MLP package has no output binding")?;
    let expected_bytes = output
        .slices
        .iter()
        .map(|slice| slice.file_offset + slice.size)
        .max()
        .context("MLP output has no slices")?;
    if bytes.len() != usize::try_from(expected_bytes)? {
        bail!(
            "MLP returned {} bytes, expected {expected_bytes}",
            bytes.len()
        );
    }
    let mut maximum_error = 0.0f32;
    let mut mismatches = Vec::new();
    let mut checked = 0usize;
    if output.slices.len() != usize::from(active_tiles) {
        bail!("MLP output does not cover every logical tile");
    }
    for (row, slice) in output.slices.iter().enumerate() {
        let row = u16::try_from(row)?;
        let expected_row = mlp_smoke_reference(row);
        let elements = u32::try_from(slice.size / 2)?;
        for linear in 0..elements {
            let (_, column) =
                amp_matrix_coordinates(AmpOrder::Left, Precision::F16, 1, elements, linear)?;
            if column >= 64 {
                continue;
            }
            let offset = usize::try_from(slice.file_offset + u64::from(linear) * 2)?;
            let actual = half_to_f32(u16::from_le_bytes(
                bytes[offset..offset + 2].try_into().unwrap(),
            ));
            let expected = expected_row[column as usize];
            let error = (actual - expected).abs();
            maximum_error = maximum_error.max(error);
            checked += 1;
            if error > 0.02 && mismatches.len() < 16 {
                mismatches.push((row, column, expected, actual, error));
            }
        }
    }
    if !mismatches.is_empty() {
        bail!("MLP numerical verification failed after {checked} checks: {mismatches:?}");
    }
    println!(
        "mlpNumericalChecks={checked} maximumAbsoluteError={maximum_error:.6} numericalTest=PASS"
    );
    Ok(())
}

fn gelu_reference(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value.powi(3))).tanh())
}

const MLP_SMOKE_WIDTH: u32 = 64;
const MLP_INPUT_SEED: u64 = 0x6d6c_705f_696e_7075;
const MLP_FIRST_WEIGHT_SEED: u64 = 0x6d6c_705f_7730_5f5f;
const MLP_SECOND_WEIGHT_SEED: u64 = 0x6d6c_705f_7731_5f5f;
const MLP_INPUT_STANDARD_DEVIATION: f32 = 0.5;
const MLP_WEIGHT_STANDARD_DEVIATION: f32 = 0.125;

fn mlp_smoke_value(seed: u64, index: u64, standard_deviation: f32) -> u16 {
    f32_to_half(gaussian(seed, index) * standard_deviation)
}

fn mlp_smoke_reference(row: u16) -> [f32; MLP_SMOKE_WIDTH as usize] {
    let mut hidden = [0.0; MLP_SMOKE_WIDTH as usize];
    for column in 0..MLP_SMOKE_WIDTH {
        let mut sum = 0.0;
        for inner in 0..MLP_SMOKE_WIDTH {
            let input = half_to_f32(mlp_smoke_value(
                MLP_INPUT_SEED,
                u64::from(row) * u64::from(MLP_SMOKE_WIDTH) + u64::from(inner),
                MLP_INPUT_STANDARD_DEVIATION,
            ));
            let weight = half_to_f32(mlp_smoke_value(
                MLP_FIRST_WEIGHT_SEED,
                u64::from(inner) * u64::from(MLP_SMOKE_WIDTH) + u64::from(column),
                MLP_WEIGHT_STANDARD_DEVIATION,
            ));
            sum += input * weight;
        }
        hidden[column as usize] = gelu_reference(half_to_f32(f32_to_half(sum)));
    }
    let mut output = [0.0; MLP_SMOKE_WIDTH as usize];
    for column in 0..MLP_SMOKE_WIDTH {
        let mut sum = 0.0;
        for inner in 0..MLP_SMOKE_WIDTH {
            let weight = half_to_f32(mlp_smoke_value(
                MLP_SECOND_WEIGHT_SEED,
                u64::from(inner) * u64::from(MLP_SMOKE_WIDTH) + u64::from(column),
                MLP_WEIGHT_STANDARD_DEVIATION,
            ));
            sum += half_to_f32(f32_to_half(hidden[inner as usize])) * weight;
        }
        output[column as usize] = gelu_reference(half_to_f32(f32_to_half(sum)));
    }
    output
}

fn gaussian(seed: u64, index: u64) -> f32 {
    let mut random = SplitMix64::seed_from_u64(seed ^ index);
    StandardNormal.sample(&mut random)
}

fn f32_to_half(value: f32) -> u16 {
    f16::from_f32(value).to_bits()
}

fn half_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

fn packed_binding(
    binding: &Binding,
    mut value: impl FnMut(u16, u32, u32) -> Result<u16>,
) -> Result<Vec<u8>> {
    let total = binding
        .slices
        .iter()
        .map(|slice| slice.file_offset + slice.size)
        .max()
        .context("binding has no slices")?;
    let mut bytes = vec![0; usize::try_from(total)?];
    for (logical_tile, slice) in binding.slices.iter().enumerate() {
        if slice.size == 0 {
            continue;
        }
        if slice.size & 1 != 0 {
            bail!("binding {} has a non-F16 slice", binding.name);
        }
        let elements = u32::try_from(slice.size / 2)?;
        let logical_tile = u16::try_from(logical_tile)?;
        for linear in 0..elements {
            let bits = value(logical_tile, linear, elements)?;
            let offset = usize::try_from(slice.file_offset + u64::from(linear) * 2)?;
            bytes[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    Ok(bytes)
}

fn verify_gemm_output(
    application: &Application,
    active_tiles: u16,
    batch: u32,
    bytes: &[u8],
) -> Result<()> {
    let output = application
        .outputs
        .iter()
        .find(|binding| binding.name == "output.0")
        .context("GEMM package has no output binding")?;
    let expected_bytes = output
        .slices
        .iter()
        .map(|slice| slice.file_offset + slice.size)
        .max()
        .context("GEMM output has no slices")?;
    if bytes.len() != usize::try_from(expected_bytes)? {
        bail!(
            "GEMM returned {} bytes, expected {expected_bytes}",
            bytes.len()
        );
    }
    let mut mismatches = Vec::new();
    let mut checked = 0usize;
    if output.slices.len() != usize::from(active_tiles) {
        bail!("GEMM output does not cover every logical tile");
    }
    for (row, slice) in output.slices.iter().enumerate() {
        let row = u16::try_from(row)?;
        let elements = u32::try_from(slice.size / 2)?;
        for linear in 0..elements {
            let (batch_index, column) = amp_matrix_coordinates(
                AmpOrder::Output,
                Precision::F16,
                batch,
                u32::from(active_tiles) * 64,
                linear,
            )?;
            let offset = usize::try_from(slice.file_offset + u64::from(linear) * 2)?;
            let actual = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            let selected_inner = (batch_index * 7 + u32::from(row)) % 64;
            let expected = gemm_right_value(selected_inner, column);
            checked += 1;
            if actual != expected && mismatches.len() < 16 {
                mismatches.push((row, batch_index, column, expected, actual));
            }
        }
    }
    if !mismatches.is_empty() {
        bail!("GEMM numerical verification failed after {checked} checks: {mismatches:?}");
    }
    println!("gemmNumericalChecks={checked} numericalTest=PASS");
    Ok(())
}

fn gemm_right_value(inner: u32, column: u32) -> u16 {
    const VALUES: [u16; 5] = [0xbc00, 0xb800, 0x0000, 0x3800, 0x3c00];
    VALUES[((inner * 3 + column) % VALUES.len() as u32) as usize]
}

fn c600_tile_count() -> u32 {
    ipu_exchange::Topology::c600().tile_count() as u32
}

fn write_package(application: &Application, path: &Path) -> Result<()> {
    let mut output =
        fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    application.write(&mut output)?;
    Ok(())
}

fn inspect_exchange_rows(application: &Application, selected_tile: Option<u32>) -> Result<String> {
    use ipu_exchange::diagnostic::{PlanOperation, SendEncoding};

    let mut summaries = Vec::new();
    let mut selected = String::new();
    let mut totals = [0usize; 6];
    for region in application
        .debug_regions
        .iter()
        .filter(|region| region.name == "exchange row")
    {
        let tile = application
            .tiles
            .iter()
            .find(|tile| tile.physical_tile == region.physical_tile)
            .context("exchange debug region refers to a missing tile")?;
        let segment = tile
            .segments
            .iter()
            .find(|segment| {
                segment.address == region.address
                    && segment.memory_size == region.size
                    && segment.data.len() == region.size as usize
            })
            .context("exchange debug region has no exact package segment")?;
        let words = segment
            .data
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte exchange word")))
            .collect::<Vec<_>>();
        let diagnostic =
            ipu_exchange::diagnostic::diagnose_plan_program(&words, Some(region.address))?;
        let mut counts = [0usize; 6];
        for instruction in &diagnostic.instructions {
            match &instruction.operation {
                PlanOperation::Send { encoding, .. } => {
                    counts[match encoding {
                        SendEncoding::Explicit => 0,
                        SendEncoding::Offset => 1,
                        SendEncoding::Pic => 2,
                        SendEncoding::PicPair => 3,
                    }] += 1;
                }
                PlanOperation::IncomingControl(_) => counts[4] += 1,
                PlanOperation::Unknown(_) => counts[5] += 1,
                _ => {}
            }
        }
        for (total, count) in totals.iter_mut().zip(counts) {
            *total += count;
        }
        summaries.push((
            words.len(),
            diagnostic.event_cycles,
            region.physical_tile,
            region.address,
            counts,
        ));
        if selected_tile == Some(region.physical_tile) {
            selected.push_str(&format!(
                "tile={} row=0x{:x} words={} events={} counts={counts:?}\n{}",
                region.physical_tile,
                region.address,
                words.len(),
                diagnostic.event_cycles,
                diagnostic.render(),
            ));
        }
    }
    summaries.sort_unstable_by_key(|summary| std::cmp::Reverse((summary.0, summary.1)));
    let longest = summaries.iter().take(16).collect::<Vec<_>>();
    Ok(format!(
        "exchangeRows={} counts=[send, sendoff, sendpic, sendpicp, standaloneControl, unknown]={totals:?} longest={longest:?}\n{selected}",
        summaries.len(),
    ))
}

fn diagnose_completion(
    runtime: &Runtime,
    application: &Application,
    timeout: Duration,
) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        let states = supervisor_states(runtime, application)?;
        if states.iter().all(|&(_, state)| state == 0) {
            break;
        }
        if Instant::now() >= deadline {
            bail!("supervisors did not halt: {}", summarize_states(&states));
        }
        std::thread::sleep(Duration::from_micros(100));
    }

    let mut active_workers = Vec::new();
    for tile in &application.tiles {
        let physical = u16::try_from(tile.physical_tile)?;
        for context in 1..=6 {
            let state = runtime.device().tile_context_state(physical, context)?;
            if state != 0 && active_workers.len() < 16 {
                active_workers.push((physical, context, state));
            }
        }
    }
    if !active_workers.is_empty() {
        bail!("workers did not halt: {active_workers:?}");
    }
    Ok(())
}

fn supervisor_states(runtime: &Runtime, application: &Application) -> Result<Vec<(u16, u32)>> {
    application
        .tiles
        .iter()
        .map(|tile| {
            let physical = u16::try_from(tile.physical_tile)?;
            Ok((physical, runtime.device().tile_context_state(physical, 0)?))
        })
        .collect()
}

fn summarize_states(states: &[(u16, u32)]) -> String {
    let mut counts = [0usize; 4];
    let mut unexpected = Vec::new();
    for &(tile, state) in states {
        if let Some(count) = counts.get_mut(state as usize) {
            *count += 1;
        }
        if state != 0 && unexpected.len() < 16 {
            unexpected.push((tile, state));
        }
    }
    format!("counts={counts:?} firstUnexpected={unexpected:?}")
}

fn exchange_row_failure_diagnostic(
    runtime: &Runtime,
    application: &Application,
    physical: u16,
    program_counter: u32,
) -> Option<String> {
    let region = application.debug_regions.iter().find(|region| {
        region.physical_tile == u32::from(physical)
            && region.name == "exchange row"
            && (region.address..region.address.saturating_add(region.size))
                .contains(&program_counter)
    })?;
    let segment = application
        .tiles
        .iter()
        .find(|tile| tile.physical_tile == u32::from(physical))?
        .segments
        .iter()
        .find(|segment| {
            segment.address == region.address
                && segment.memory_size == region.size
                && segment.data.len() == region.size as usize
        })?;
    let expected = segment
        .data
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte exchange word")))
        .collect::<Vec<_>>();
    let actual = runtime
        .device()
        .read_tile_words_from_inactive_context(physical, 1, region.address, region.size / 4)
        .ok()?;
    let differences = expected
        .iter()
        .zip(&actual)
        .enumerate()
        .filter(|(_, (expected, actual))| expected != actual)
        .map(|(offset, (&expected, &actual))| (offset, expected, actual))
        .collect::<Vec<_>>();
    let expected_decode =
        ipu_exchange::diagnostic::diagnose_plan_program(&expected, Some(region.address))
            .map(|row| row.render_around_address(program_counter, 12));
    let actual_decode = (actual != expected).then(|| {
        ipu_exchange::diagnostic::diagnose_plan_program(&actual, Some(region.address))
            .map(|row| row.render_around_address(program_counter, 12))
    });
    Some(format!(
        "exchangeRow=0x{:x}..0x{:x} pc=0x{program_counter:x} readbackDifferences={} firstDifferences={:?} expectedDecode={expected_decode:?} actualDecode={actual_decode:?}",
        region.address,
        region.address + region.size,
        differences.len(),
        differences.iter().take(16).collect::<Vec<_>>(),
    ))
}

fn device_failure_diagnostics(runtime: &Runtime, application: &Application) -> String {
    let states = match supervisor_states(runtime, application) {
        Ok(states) => states,
        Err(error) => return format!("supervisor state read failed: {error}"),
    };
    let mut contexts = Vec::new();
    for &(physical, state) in states.iter().filter(|(_, state)| *state != 0).take(16) {
        let program_counter = runtime.device().read_tile_program_counter(physical, 0);
        let program_counter_symbol = program_counter
            .as_ref()
            .ok()
            .and_then(|&pc| application.symbolize_pc(u32::from(physical), pc));
        let segment = program_counter.as_ref().ok().and_then(|&pc| {
            application
                .tiles
                .iter()
                .find(|tile| tile.physical_tile == u32::from(physical))?
                .segments
                .iter()
                .find(|segment| {
                    (segment.address..segment.address.saturating_add(segment.memory_size))
                        .contains(&pc)
                })
                .map(|segment| {
                    let offset = usize::try_from(pc - segment.address).ok()?;
                    let start = offset.saturating_sub(8) & !3;
                    let end = (offset + 12).min(segment.data.len()) & !3;
                    let words = segment
                        .data
                        .get(start..end)?
                        .chunks_exact(4)
                        .map(|word| {
                            u32::from_le_bytes(word.try_into().expect("four-byte instruction"))
                        })
                        .collect::<Vec<_>>();
                    Some((segment.address, segment.memory_size, start, words))
                })?
        });
        let memory_readback =
            segment
                .as_ref()
                .and_then(|&(address, memory_size, byte_offset, ref expected)| {
                    let words = memory_size / 4;
                    let actual = runtime
                        .device()
                        .read_tile_words_from_inactive_context(physical, 1, address, words)
                        .ok()?;
                    let actual_near =
                        actual.get(byte_offset / 4..byte_offset / 4 + expected.len())?;
                    let differences = expected
                        .iter()
                        .zip(actual_near)
                        .enumerate()
                        .filter(|(_, (expected, actual))| expected != actual)
                        .map(|(offset, (&expected, &actual))| {
                            (byte_offset / 4 + offset, expected, actual)
                        })
                        .collect::<Vec<_>>();
                    Some((
                        actual.len(),
                        differences.len(),
                        differences.into_iter().take(16).collect::<Vec<_>>(),
                    ))
                });
        let exchange_row = program_counter
            .as_ref()
            .ok()
            .and_then(|&pc| exchange_row_failure_diagnostic(runtime, application, physical, pc));
        let supervisor_registers = (0..16)
            .map(|register| runtime.device().read_tile_m_register(physical, 0, register))
            .collect::<Result<Vec<_>, _>>();
        let exchange_state = runtime.device().tile_exchange_state(physical);
        let exchange_receive_error = runtime.device().tile_exchange_receive_error(physical);
        let workers = (1..=6)
            .map(|context| runtime.device().tile_context_state(physical, context))
            .collect::<Result<Vec<_>, _>>();
        let worker_program_counters = workers
            .as_ref()
            .ok()
            .map(|states| {
                states
                    .iter()
                    .enumerate()
                    .filter(|(_, state)| **state != 0)
                    .map(|(worker, _)| {
                        let context = u32::try_from(worker + 1).expect("worker context fits u32");
                        (
                            context,
                            runtime
                                .device()
                                .read_tile_program_counter(physical, context)
                                .map(|pc| (pc, application.symbolize_pc(u32::from(physical), pc))),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        contexts.push((
            physical,
            state,
            program_counter,
            program_counter_symbol,
            segment,
            memory_readback,
            exchange_row,
            supervisor_registers,
            exchange_state,
            exchange_receive_error,
            workers,
            worker_program_counters,
        ));
    }
    format!("{} contexts={contexts:?}", summarize_states(&states))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn randomized_binding_values_follow_logical_slice_order() -> Result<()> {
        let mut random = fastrand::Rng::with_seed(0x6c6f_6769_6361_6c5f);
        for _ in 0..256 {
            let count = random.usize(1..=128);
            let mut physical = (0..count as u32).collect::<Vec<_>>();
            random.shuffle(&mut physical);
            let binding = Binding {
                name: "mapped".into(),
                dtype: "f16".into(),
                shape: vec![count as u32],
                slices: physical
                    .into_iter()
                    .enumerate()
                    .map(|(logical, tile)| ipu_package::RegionSlice {
                        tile,
                        tile_address: 0x10_0000,
                        file_offset: (logical * 2) as u64,
                        size: 2,
                    })
                    .collect(),
            };

            let bytes = packed_binding(&binding, |logical, _, _| Ok(logical))?;
            for (logical, raw) in bytes.chunks_exact(2).enumerate() {
                assert_eq!(u16::from_le_bytes(raw.try_into().unwrap()), logical as u16);
            }
        }
        Ok(())
    }
}
