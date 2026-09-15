//! Exercise the actual compiler driver, including support reservations and
//! rejected packages. No fake finalization callback stands in for evaluation.
use super::*;
use crate::PipelineConfig;
use crate::Precision;
use crate::package::PackageBuildError;
use std::path::PathBuf;

fn mlp() -> (ComputeGraph, PipelineConfig) {
    let mut graph = ComputeGraph::new();
    let x = graph.host_input("x", [32, 64]).unwrap();
    let up = graph.parameter("up", [64, 128]).unwrap();
    let down = graph.parameter("down", [128, 64]).unwrap();
    let hidden = graph.gemm(x, up).unwrap();
    let hidden = graph.gelu(hidden).unwrap();
    let y = graph.gemm(hidden, down).unwrap();
    graph.set_outputs([y]).unwrap();
    let config = PipelineConfig::new(64)
        .with_active_tile_counts([8])
        .with_automatic_input(x, Precision::F16)
        .with_automatic_input(up, Precision::F16)
        .with_automatic_input(down, Precision::F16);
    (graph, config)
}

#[test]
#[ignore = "requires IPU_TEST_SDK; compiles actual IPU packages without a device"]
fn saved_search_preserves_budget_aliases_and_order() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let sdk = std::env::var_os("IPU_TEST_SDK").expect("set IPU_TEST_SDK to the Poplar SDK");
    let (graph, mut pipeline) = mlp();
    pipeline.optimization_steps = 4;
    let mut config = PackageConfig {
        invocations: 1,
        toolchain: ipu_elf::Toolchain::from_sdk(sdk),
        runtime_source: root.join("device/static_runtime.S"),
        kernel_source_directory: root.join("device"),
        tile_mapping: Some((0..pipeline.tile_count).collect()),
        pipeline,
    };
    let run = |config: &PackageConfig, threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| compile_graph(&graph, config))
            .unwrap()
    };
    let uninterrupted = run(&config, 1);
    let path = std::env::temp_dir().join(format!("ipu-search-{}.json", fastrand::u64(..)));
    config.pipeline.optimization_steps = 2;
    config.pipeline.save_search_state = Some(path.clone());
    run(&config, 4);
    config.pipeline.load_search_state = Some(path.clone());
    let before =
        checkpoint::State::load(&graph, &config.pipeline, config.tile_mapping.as_deref()).unwrap();
    assert!(
        before.attempts > 0 && before.attempts <= 2,
        "attempts={}",
        before.attempts
    );
    let resumed = run(&config, 4);
    assert_eq!(resumed.program, uninterrupted.program);
    assert_eq!(resumed.placement, uninterrupted.placement);
    assert_eq!(resumed.exchanges.phases, uninterrupted.exchanges.phases);
    assert_eq!(resumed.cycles, uninterrupted.cycles);
    let state =
        checkpoint::State::load(&graph, &config.pipeline, config.tile_mapping.as_deref()).unwrap();
    assert!(state.attempts <= 4);
    assert!(
        state.visited.len() > state.attempts,
        "equivalent recipes travel together"
    );
    config.pipeline.optimization_steps = 0;
    let rebuilt = run(&config, 2);
    assert_eq!(resumed.program, rebuilt.program);
    assert_eq!(resumed.placement, rebuilt.placement);
    assert_eq!(resumed.exchanges.phases, rebuilt.exchanges.phases);
    let after =
        checkpoint::State::load(&graph, &config.pipeline, config.tile_mapping.as_deref()).unwrap();
    assert_eq!(after.attempts, state.attempts);
    assert!(after.visited == state.visited);
    std::fs::remove_file(&path).unwrap();

    // An infeasible baseline returns its failure before saving an incumbent.
    config.pipeline.load_search_state = None;
    config.pipeline.optimization_steps = 4;
    config.pipeline.exchange_transfer_limit_per_tile = 0;
    assert!(matches!(
        compile_graph(&graph, &config),
        Err(PackageBuildError::ExchangeTransferLimitExceeded { .. })
    ));
    assert!(!path.exists());
}
