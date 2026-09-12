//! External logical tensors and independent reference outputs for trained-model checks.
use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result, ensure};
use ipu_codegen::{CompiledPackage, ComputeGraph, GraphInputKind};
use ipu_package::Application;
use ipu_runtime::Runtime;
use serde_json::Value;

use crate::diagnostic::{HostTensor, pack_bindings};

fn tensor(root: &Path, entry: &Value, shape: &[u32]) -> Result<HostTensor> {
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
    let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let elements = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
        .context("fixture tensor size overflow")?;
    ensure!(
        bytes.len() == elements * 4,
        "wrong tensor length: {}",
        path.display()
    );
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
        shape: declared,
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
            let shape: Vec<u32> = serde_json::from_value(entry["shape"].clone())
                .with_context(|| format!("fixture tensor {}", input.name))?;
            ensure!(
                shape == input.shape.0,
                "fixture shape mismatch for {}",
                input.name
            );
            let path = root.join(
                entry["file"]
                    .as_str()
                    .context("fixture tensor has no file")?,
            );
            ensure!(
                fs::metadata(&path)?.len() == input.shape.elements() * 4,
                "wrong tensor length: {}",
                path.display()
            );
        }
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
