//! Small LN -> two independent FP8 projections, without device compilation.
use ipu_codegen::optimistic::{SearchOptions, plan_graph};
use ipu_codegen::{
    ComputeGraph, Layout, MidOperator, OperatorCandidate, PipelineConfig, Precision, TensorFormat,
};
use std::collections::BTreeMap;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = ComputeGraph::new();
    let x = graph.host_input("x", [1, 8, 128])?;
    let scale = graph.parameter("scale", [1, 1, 128])?;
    let bias = graph.parameter("bias", [1, 1, 128])?;
    let normalized = graph.layer_norm(x, scale, bias)?;
    let w0 = graph.parameter("w0", [128, 128])?;
    let w1 = graph.parameter("w1", [128, 128])?;
    let q = graph.gemm(normalized, w0)?;
    let k = graph.gemm(normalized, w1)?;
    graph.set_outputs([q, k])?;
    let mut config = PipelineConfig::new(64);
    for input in graph.inputs() {
        config = config.with_automatic_input(input.value, Precision::F16);
    }
    config
        .operator_candidates
        .retain(|c| !matches!(c.operator(), MidOperator::Gemm { .. }));
    config
        .operator_candidates
        .push(OperatorCandidate::fp8_gemm(64, -4));
    let output = TensorFormat {
        precision: Precision::F16,
        layout: Layout::row_sharded(64),
    };
    let report = plan_graph(
        &graph,
        &config,
        BTreeMap::from([(q, output.clone()), (k, output)]),
        &SearchOptions::default(),
    )?;
    println!(
        "expanded={} truncated={} candidates={} opportunities={}",
        report.expanded,
        report.truncated,
        report.candidates.len(),
        report.opportunities().count()
    );
    if let Some(reference) = report.reference() {
        println!("referenceCycles={}", reference.cycles.conservative);
    }
    for (index, candidate) in report.candidates.iter().take(5).enumerate() {
        println!(
            "candidate={index} cycles={:?} memory={:?} assumptions={:?}",
            candidate.cycles, candidate.memory, candidate.assumptions
        );
    }
    if let Some(path) = std::env::args_os().nth(1) {
        std::fs::write(path, report.candidates[0].to_dot())?;
    }
    Ok(())
}
