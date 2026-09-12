//! External logical tensors and independent reference outputs for trained-model checks.
use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result, ensure};
use ipu_codegen::{CompiledPackage, ComputeGraph, GraphInputKind};
use ipu_package::Application;
use ipu_runtime::Runtime;
use serde_json::Value;

use crate::diagnostic::{HostTensor, pack_bindings};

fn tensor_path(root: &Path, entry: &Value, shape: &[u32]) -> Result<std::path::PathBuf> {
    let declared: Vec<u32> = serde_json::from_value(entry["shape"].clone())?;
    ensure!(
        declared == shape,
        "fixture shape {declared:?}, expected {shape:?}"
    );
    let path = root.join(
        entry["file"]
            .as_str()
            .context("fixture tensor has no file")?,
    );
    let bytes = shape
        .iter()
        .try_fold(4u64, |n, &d| n.checked_mul(u64::from(d)))
        .context("fixture tensor size overflow")?;
    ensure!(
        fs::metadata(&path)?.len() == bytes,
        "wrong tensor length: {}",
        path.display()
    );
    Ok(path)
}

fn tensor(root: &Path, entry: &Value, shape: &[u32]) -> Result<HostTensor> {
    let path = tensor_path(root, entry, shape)?;
    let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let values: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite fixture: {}",
        path.display()
    );
    Ok(HostTensor {
        shape: shape.to_vec(),
        values,
    })
}

/// Fail before planning if the checkpoint does not match this graph.
pub(crate) fn validate(root: &Path, graph: &ComputeGraph, count: u32) -> Result<()> {
    let manifest: Value = serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
    let cases = manifest["cases"]
        .as_array()
        .context("fixture has no cases")?;
    ensure!(
        cases.len() == count as usize,
        "fixture has {} cases; set --reference-inferences accordingly",
        cases.len()
    );
    for input in graph.inputs() {
        let entries: Vec<_> = if input.kind == GraphInputKind::Parameter {
            vec![&manifest["parameters"][&input.name]]
        } else {
            cases.iter().map(|c| &c["inputs"][&input.name]).collect()
        };
        for entry in entries {
            tensor_path(root, entry, &input.shape.0)
                .with_context(|| format!("fixture tensor {}", input.name))?;
        }
    }
    ensure!(
        graph.outputs().len() == 1,
        "fixture requires one graph output"
    );
    let shape = graph
        .value_shape(graph.outputs()[0])
        .context("missing graph output")?;
    for case in cases {
        tensor_path(root, &case["expected"], &shape.0).context("fixture expected output")?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    runtime: &Runtime,
    application: &Application,
    graph: &ComputeGraph,
    package: &CompiledPackage,
    root: &Path,
    timeout: u64,
    count: u32,
) -> Result<()> {
    let manifest: Value = serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
    let cases = manifest["cases"]
        .as_array()
        .context("fixture has no cases")?;
    ensure!(
        cases.len() == count as usize,
        "fixture has {} cases; set --reference-inferences accordingly",
        cases.len()
    );
    let mut values = BTreeMap::new();
    for input in graph
        .inputs()
        .iter()
        .filter(|i| i.kind == GraphInputKind::Parameter)
    {
        values.insert(
            input.value,
            tensor(root, &manifest["parameters"][&input.name], &input.shape.0)
                .with_context(|| format!("parameter {}", input.name))?,
        );
    }
    let weights = pack_bindings(&application.weights, &package.inputs, &values)?;
    // Release FP32 parameters before hardware execution. Every image reuses the same resident weights.
    values.clear();
    let output = package
        .outputs
        .iter()
        .find(|t| t.name.as_deref() == Some("output.0"))
        .context("fixture requires output.0 storage metadata")?;
    let mut failures = Vec::new();
    crate::run_checked_inferences(
        runtime,
        application,
        &weights,
        |index| {
            for input in graph
                .inputs()
                .iter()
                .filter(|i| i.kind == GraphInputKind::Host)
            {
                values.insert(
                    input.value,
                    tensor(
                        root,
                        &cases[index as usize]["inputs"][&input.name],
                        &input.shape.0,
                    )?,
                );
            }
            pack_bindings(&application.inputs, &package.inputs, &values)
        },
        timeout,
        count,
        |index, bytes| {
            let case = &cases[index as usize];
            println!(
                "referenceFixtureCase={}",
                case["name"].as_str().unwrap_or("unnamed")
            );
            let expected = tensor(root, &case["expected"], &output.shape.0)?;
            // Save raw output even on numerical failure, so a failed model can be investigated.
            fs::write(root.join(format!("device-{index}.bin")), bytes)?;
            match crate::verify_logical_output(
                application,
                output,
                bytes,
                &expected.values,
                crate::ReferenceCheck::Cosine(0.99),
            ) {
                Ok(error) => {
                    println!("referenceMaximumAbsoluteError={error:.6} numericalTest=PASS")
                }
                Err(error) => {
                    eprintln!("numericalTest=FAIL {error:#}");
                    failures.push(index);
                }
            }
            Ok(())
        },
    )?;
    ensure!(
        failures.is_empty(),
        "fixture cases failed cosine >0.99: {failures:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rejects_incomplete_or_mismatched_fixture_before_planning() -> Result<()> {
        let root = std::env::temp_dir().join(format!("ipu-fixture-{}", fastrand::u64(..)));
        fs::create_dir(&root)?;
        let result = (|| -> Result<()> {
            let mut graph = ComputeGraph::new();
            let x = graph.host_input("image", [1, 2])?;
            let w = graph.parameter("weight", [1, 2])?;
            let output = graph.add(x, w)?;
            graph.set_outputs([output])?;
            fs::write(
                root.join("values.f32"),
                [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat(),
            )?;
            let entry = json!({"file": "values.f32", "shape": [1, 2]});
            let manifest = json!({"parameters": {"weight": entry}, "cases": [{
                "name": "sample", "inputs": {"image": entry}, "expected": entry
            }]});
            fs::write(root.join("manifest.json"), serde_json::to_vec(&manifest)?)?;
            validate(&root, &graph, 1)?;
            assert_eq!(tensor(&root, &entry, &[1, 2])?.values, [1.0, 2.0]);
            assert!(validate(&root, &graph, 2).is_err());
            assert!(tensor(&root, &entry, &[2, 1]).is_err());
            fs::write(root.join("values.f32"), [0; 4])?;
            assert!(validate(&root, &graph, 1).is_err());
            fs::write(
                root.join("values.f32"),
                [f32::NAN.to_le_bytes(), 0.0f32.to_le_bytes()].concat(),
            )?;
            assert!(tensor(&root, &entry, &[1, 2]).is_err());
            fs::remove_file(root.join("values.f32"))?;
            assert!(validate(&root, &graph, 1).is_err());
            Ok(())
        })();
        fs::remove_dir_all(root)?;
        result
    }
}

/// Apply calibrated fixed scales to graph parameters and their GEMM consumers.
/// Repeat binds parameter sequences to body arguments; every instance must agree.
pub(crate) fn configure(
    graph: &ComputeGraph,
    pipeline: &mut ipu_codegen::PipelineConfig,
    path: &Path,
) -> Result<()> {
    use ipu_codegen::{
        MidOperator, Operation, OperationKind, OperatorCandidate, Precision, ValueId,
    };
    let report: Value = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(
        report["shared_operand_scale"] == true && report["fp16_embedding"] == true,
        "hardware calibration requires shared operand scales and FP16 embedding"
    );
    ensure!(
        report["nearest"].is_array() && report.get("gptq").is_none(),
        "use nearest calibration with original fixture parameters; reconstructed weights require a separate fixture"
    );
    let mut parameters = BTreeMap::new();
    for input in graph
        .inputs()
        .iter()
        .filter(|i| i.kind == GraphInputKind::Parameter && i.name.ends_with(".weight"))
    {
        let name = input
            .name
            .strip_prefix("vit.")
            .context("calibration expects named ViT parameters")?
            .strip_suffix(".weight")
            .unwrap();
        let precision = if name == "embedding" {
            Precision::F16
        } else {
            let scales = &report["scales"][name];
            let exponent = scales["weight"]
                .as_i64()
                .with_context(|| format!("missing scale for {name}"))?;
            ensure!(
                scales["activation"].as_i64() == Some(exponent),
                "different operand scales for {name}"
            );
            ensure!(
                (-16..=15).contains(&exponent),
                "GEMM product scale out of range for {name}"
            );
            Precision::F8F143 {
                scale_exponent: exponent as i8,
            }
        };
        pipeline.automatic_inputs.insert(input.value, precision);
        parameters.insert(input.value, precision);
    }
    fn visit(
        operations: &[Operation],
        graph: &ComputeGraph,
        parameters: &mut BTreeMap<ValueId, Precision>,
        choices: &mut BTreeMap<ipu_codegen::OperationId, Precision>,
    ) -> Result<()> {
        for operation in operations {
            match &operation.kind {
                OperationKind::Gemm(_) => {
                    let precision = parameters.get(&operation.inputs[1]).with_context(|| {
                        format!("GEMM {:?} has no calibrated weight", operation.id)
                    })?;
                    choices.insert(operation.id, *precision);
                }
                OperationKind::Repeat(repeat) => {
                    for (&argument, &input) in repeat.body.arguments.iter().zip(&operation.inputs) {
                        if let Some(&precision) = parameters.get(&input) {
                            parameters.insert(argument, precision);
                        }
                    }
                    for (&argument, sequence) in repeat
                        .body
                        .arguments
                        .iter()
                        .skip(repeat.carried_inputs + repeat.invariant_inputs)
                        .zip(&repeat.iterated_inputs)
                    {
                        let values = &graph.sequences()[sequence.index() as usize].values;
                        if let Some(&precision) = parameters.get(&values[0]) {
                            ensure!(
                                values.iter().all(|v| parameters.get(v) == Some(&precision)),
                                "calibrated scales vary within a repeated parameter sequence"
                            );
                            parameters.insert(argument, precision);
                        }
                    }
                    visit(&repeat.body.operations, graph, parameters, choices)?;
                }
                _ => {}
            }
        }
        Ok(())
    }
    visit(
        graph.operations(),
        graph,
        &mut parameters,
        &mut pipeline.gemm_precisions,
    )?;
    pipeline
        .operator_candidates
        .retain(|c| !matches!(c.operator(), MidOperator::Gemm { .. }));
    pipeline
        .operator_candidates
        .push(OperatorCandidate::parallel_gemm(pipeline.tile_count));
    Ok(())
}
