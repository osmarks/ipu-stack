//! Fuse compatible whole-device primitives before physical expansion.
use super::*;

impl MidProgram {
    pub(super) fn with_elementwise_fusions(&self) -> Option<Self> {
        let mut result = self.clone();
        if !fuse_region(&mut result.operations, &result.values, &result.outputs) {
            return None;
        }
        let (before, _) = crate::estimate::analyze_mid(self, &BTreeMap::new())?;
        let (after, peak) = crate::estimate::analyze_mid(&result, &BTreeMap::new())?;
        if after.total >= before.total {
            return None;
        }
        result.estimated_cycles = after.total;
        result.estimated_exchange_cycles = after.exchange;
        result.peak_memory = peak;
        Some(result)
    }
}

fn fuse_region(
    operations: &mut Vec<MidOperation>,
    values: &[MidValue],
    required: &[MidValueId],
) -> bool {
    let mut changed = false;
    for operation in &mut *operations {
        if let MidOperationKind::Repeat(repeat) = &mut operation.kind {
            changed |= fuse_region(&mut repeat.body.operations, values, &repeat.body.yields);
        }
    }
    let mut users = BTreeMap::<MidValueId, BTreeSet<usize>>::new();
    let mut producers = BTreeMap::new();
    for (index, operation) in operations.iter().enumerate() {
        for &input in operation.read_values() {
            users.entry(input).or_default().insert(index);
        }
        for &output in &operation.results {
            producers.insert(output, index);
        }
    }
    for &output in required {
        users.entry(output).or_default().insert(usize::MAX);
    }
    let mut removed = BTreeSet::new();
    for index in 0..operations.len() {
        let current = &operations[index];
        let MidOperationKind::Primitive(Primitive::Compute {
            kernel,
            operands,
            product: None,
            ..
        }) = &current.kind
        else {
            continue;
        };
        let fused = match kernel {
            TileKernelSpec::Gelu => TileKernelSpec::BiasGelu,
            TileKernelSpec::LayerNorm => TileKernelSpec::AddLayerNorm,
            _ => continue,
        };
        if operands.iter().any(|window| !window.0.is_empty()) {
            continue;
        }
        let Some(&input) = current.inputs.first() else {
            continue;
        };
        let Some(&previous) = producers.get(&input) else {
            continue;
        };
        if previous >= index || users.get(&input).is_none_or(|uses| uses.len() != 1) {
            continue;
        }
        let add = &operations[previous];
        let MidOperationKind::Primitive(Primitive::Compute {
            kernel: TileKernelSpec::Add,
            operands: add_operands,
            product: None,
            ..
        }) = &add.kind
        else {
            continue;
        };
        if add_operands.len() != 2 || add_operands.iter().any(|window| !window.0.is_empty()) {
            continue;
        }
        let [output] = current.results.as_slice() else {
            continue;
        };
        let output_value = &values[output.index() as usize];
        let output_type = &output_value.tensor_type;
        let left = &values[add.inputs[0].index() as usize];
        let right = &values[add.inputs[1].index() as usize];
        if &left.tensor_type != output_type
            || left.tile_offset != output_value.tile_offset
            || output_type.format.precision != Precision::F16
            || output_type.format.layout.order != ElementOrder::RowMajor
            || values[input.index() as usize].tensor_type != *output_type
            || values[input.index() as usize].tile_offset != output_value.tile_offset
        {
            continue;
        }
        if !compatible_fusion(&fused, &left.tensor_type, &right.tensor_type, output_type)
            || (fused == TileKernelSpec::AddLayerNorm
                && right.tile_offset != output_value.tile_offset)
        {
            continue;
        }
        // Parameter copies may intervene. Delay the add only if none can
        // overwrite either source; other computation retains its ordering.
        let groups = [
            left.storage_group,
            right.storage_group,
            values[input.index() as usize].storage_group,
        ];
        if operations[previous + 1..index].iter().any(|step| {
            !matches!(
                step.kind,
                MidOperationKind::Primitive(Primitive::Copy { .. })
            ) || step
                .results
                .iter()
                .any(|id| groups.contains(&values[id.index() as usize].storage_group))
        }) {
            continue;
        }
        let mut inputs = add.inputs[..2].to_vec();
        inputs.extend_from_slice(&current.inputs[1..operands.len()]);
        let reuse_input = inputs.iter().position(|id| {
            let value = &values[id.index() as usize];
            value.storage_group == output_value.storage_group && value.tensor_type == *output_type
        });
        let arity = inputs.len();
        let mut replacement = current.clone();
        replacement.inputs = inputs;
        replacement.kind = MidOperationKind::Primitive(Primitive::Compute {
            kernel: fused,
            operands: vec![OperandWindow::default(); arity],
            product: None,
            reuse_input,
        });
        let prices = crate::estimate::operation_cost(add, values)
            .zip(crate::estimate::operation_cost(current, values))
            .zip(crate::estimate::operation_cost(&replacement, values));
        if prices.is_none_or(|(((a, _, _), (b, _, _)), (fused, _, _))| {
            fused.total >= a.total.saturating_add(b.total)
        }) {
            continue;
        }
        operations[index] = replacement;
        removed.insert(previous);
        changed = true;
    }
    let mut index = 0;
    operations.retain(|_| {
        let keep = !removed.contains(&index);
        index += 1;
        keep
    });
    changed
}

// Shared with the optimistic regional search: known fusion is not a missing kernel.
pub(super) fn compatible_fusion(
    kernel: &TileKernelSpec,
    left: &TensorType,
    right: &TensorType,
    output: &TensorType,
) -> bool {
    if left != output
        || output.format.precision != Precision::F16
        || right.format.precision != Precision::F16
        || output.format.layout.order != ElementOrder::RowMajor
    {
        return false;
    }
    match kernel {
        TileKernelSpec::BiasGelu => {
            right.shape.0.last() == output.shape.0.last()
                && !right.shape.0.is_empty()
                && right.shape.0[..right.shape.0.len() - 1]
                    .iter()
                    .all(|&size| size == 1)
        }
        TileKernelSpec::AddLayerNorm => right == output,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fusion_preserves_live_add_results_and_matches_kernel_contracts() {
        for norm in [false, true] {
            for keep_sum in [false, true] {
                let mut graph = ComputeGraph::new();
                let x = graph.host_input("x", [2, 4, 384]).unwrap();
                let rhs = graph
                    .host_input(
                        "rhs",
                        if norm {
                            vec![2, 4, 384]
                        } else {
                            vec![1, 1, 384]
                        },
                    )
                    .unwrap();
                let sum = graph.add(x, rhs).unwrap();
                let mut config = PipelineConfig::new(4)
                    .with_automatic_input(x, Precision::F16)
                    .with_automatic_input(rhs, Precision::F16);
                let y = if norm {
                    let gamma = graph.parameter("gamma", [1, 1, 384]).unwrap();
                    let beta = graph.parameter("beta", [1, 1, 384]).unwrap();
                    config = config
                        .with_automatic_input(gamma, Precision::F16)
                        .with_automatic_input(beta, Precision::F16);
                    graph.layer_norm(sum, gamma, beta).unwrap()
                } else {
                    graph.gelu(sum).unwrap()
                };
                graph
                    .set_outputs(if keep_sum { vec![sum, y] } else { vec![y] })
                    .unwrap();
                let mid = implementation::resolve(lower(&graph, &config, &Ipu21CostModel).unwrap())
                    .unwrap();
                let fused = mid.with_elementwise_fusions();
                assert_eq!(fused.is_some(), !keep_sum, "norm={norm}");
                if let Some(fused) = fused {
                    assert!(
                        fused.estimated_cycles
                            < crate::estimate::analyze_mid(&mid, &BTreeMap::new())
                                .unwrap()
                                .0
                                .total
                    );
                    let low = crate::lower_to_tiles(
                        &crate::low::expand::expand_tiles(&fused, false).unwrap(),
                        false,
                    );
                    let mut count = 0;
                    for run in &low.kernel_runs {
                        if matches!(
                            run.kernel,
                            TileKernelSpec::BiasGelu | TileKernelSpec::AddLayerNorm
                        ) {
                            crate::validate_kernel_run(run).unwrap();
                            count += 1;
                        }
                    }
                    assert_eq!(count, 4);
                }
            }
        }
    }
}
