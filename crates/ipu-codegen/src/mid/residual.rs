//! Preserve live residuals while fusing their addition with local statistics.
use super::*;

pub(super) fn fuse(
    operations: &mut Vec<MidOperation>,
    values: &mut Vec<MidValue>,
    required: &[MidValueId],
) -> bool {
    let mut changed = false;
    for operation in &mut *operations {
        if let MidOperationKind::Repeat(repeat) = &mut operation.kind {
            changed |= fuse(&mut repeat.body.operations, values, &repeat.body.yields);
        }
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
        let ordinary = *kernel == TileKernelSpec::LayerNorm;
        if (!ordinary && *kernel != TileKernelSpec::LayerNormMoments)
            || operands.iter().any(|window| !window.0.is_empty())
        {
            continue;
        }
        let Some(&input) = current.inputs.first() else {
            continue;
        };
        let Some((sum, previous, identity_copies)) =
            super::elementwise::producer_through_identity_copies(
                input,
                &operations[..index],
                values,
            )
        else {
            continue;
        };
        if removed.contains(&previous) {
            continue;
        }
        let add = &operations[previous];
        let MidOperationKind::Primitive(Primitive::Compute {
            kernel: TileKernelSpec::Add,
            operands,
            product: None,
            output_aliases,
        }) = &add.kind
        else {
            continue;
        };
        if operands.len() != 2 || operands.iter().any(|window| !window.0.is_empty()) {
            continue;
        }
        let value = &values[sum.index() as usize];
        let tensor = &value.tensor_type;
        if tensor.format.precision != Precision::F16
            || tensor.format.layout.order != ElementOrder::RowMajor
            || add.inputs[..2]
                .iter()
                .any(|v| !super::elementwise::same_storage(&values[v.index() as usize], value))
        {
            continue;
        }
        // The statistics must describe the version written by this add.
        if operations[previous + 1..index].iter().any(|op| {
            !matches!(
                op.kind,
                MidOperationKind::Primitive(Primitive::Copy { .. }) | MidOperationKind::Convert(_)
            )
        }) || operations[previous + 1..index]
            .iter()
            .enumerate()
            .filter(|(i, _)| !identity_copies.contains(&(previous + 1 + i)))
            .flat_map(|(_, op)| &op.results)
            .any(|v| values[v.index() as usize].storage_group == value.storage_group)
        {
            continue;
        }
        let mut stats_value = None;
        let stats = if ordinary {
            if current.results.len() != 1
                || !super::elementwise::same_storage(
                    &values[current.results[0].index() as usize],
                    value,
                )
            {
                continue;
            }
            let rank = tensor.shape.0.len();
            let mut stats_type = tensor.clone();
            stats_type.shape.0.pop();
            stats_type.shape.0.extend([1, 2]);
            stats_type.format.precision = Precision::F32;
            let mut valid = true;
            for axis in &mut stats_type.format.layout.tiling.axes {
                let Ok(index) = axis.axis.resolve(rank) else {
                    valid = false;
                    break;
                };
                axis.axis = TensorAxis::FromStart(index as u16);
                if index + 1 == rank {
                    if axis.partitions != 1 {
                        valid = false;
                        break;
                    }
                    *axis = AxisTiling::new(axis.axis, 1, 1, Padding::Reject);
                }
            }
            if !valid {
                continue;
            }
            let id = MidValueId(values.len() as u32);
            stats_value = Some(MidValue {
                id,
                tensor_type: stats_type,
                storage_group: id,
                ..value.clone()
            });
            id
        } else {
            current.results[0]
        };
        let mut fused = add.clone();
        fused.results = vec![stats, sum];
        fused.kind = MidOperationKind::Primitive(Primitive::Compute {
            kernel: TileKernelSpec::AddLayerNormMoments,
            operands: vec![OperandWindow::default(); 2],
            product: None,
            output_aliases: output_aliases
                .iter()
                .map(|&(_, input)| (1, input))
                .collect(),
        });
        let mut apply = current.clone();
        if ordinary {
            apply.inputs[0] = sum;
            apply.inputs.truncate(3);
            apply.inputs.push(stats);
            apply.kind = MidOperationKind::Primitive(Primitive::Compute {
                kernel: TileKernelSpec::LayerNormApply { parts: 1 },
                operands: vec![OperandWindow::default(); 4],
                product: None,
                output_aliases: Vec::new(),
            });
        }
        if let Some(value) = stats_value {
            values.push(value);
        }
        let cost = |op: &MidOperation| {
            crate::estimate::operation_cost(op, values).map(|(cost, _, _)| cost.total)
        };
        let before = cost(add)
            .zip(cost(current))
            .map(|(a, b)| a.saturating_add(b));
        let after = cost(&fused)
            .zip(if ordinary { cost(&apply) } else { Some(0) })
            .map(|(a, b)| a.saturating_add(b));
        if before
            .zip(after)
            .is_none_or(|(before, after)| after >= before)
        {
            if ordinary {
                values.pop();
            }
            continue;
        }
        for copy in identity_copies {
            let result = operations[copy].results[0];
            if !required.contains(&result)
                && operations
                    .iter()
                    .filter(|op| op.read_values().any(|v| *v == result))
                    .count()
                    == 1
            {
                removed.insert(copy);
            }
        }
        operations[previous] = fused;
        if ordinary {
            operations[index] = apply;
        } else {
            removed.insert(index);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_fusion_keeps_repeat_carried_results() {
        let mut graph = ComputeGraph::new();
        let x = graph.host_input("x", [4, 1152]).unwrap();
        let r = graph.host_input("residual", [4, 1152]).unwrap();
        let gamma = graph.parameter("gamma", [1, 1152]).unwrap();
        let beta = graph.parameter("beta", [1, 1152]).unwrap();
        let outputs = graph
            .repeat(3, [x, r], [gamma, beta], [], |body, args| {
                let sum = body.add(args.carried[0], args.carried[1])?;
                let norm = body.layer_norm(sum, args.invariants[0], args.invariants[1])?;
                Ok(vec![sum, norm])
            })
            .unwrap();
        graph.set_outputs(outputs).unwrap();
        let config = PipelineConfig::new(4)
            .with_automatic_input(x, Precision::F16)
            .with_automatic_input(r, Precision::F16)
            .with_automatic_input(gamma, Precision::F16)
            .with_automatic_input(beta, Precision::F16);
        let mid =
            implementation::resolve(lower(&graph, &config, &Ipu21CostModel).unwrap()).unwrap();
        let fused = mid.with_elementwise_fusions().unwrap();
        let low = crate::lower_to_tiles(&crate::expand_tiles(&fused).unwrap(), false);
        assert!(
            low.kernel_runs
                .iter()
                .any(|run| run.kernel == TileKernelSpec::AddLayerNormMoments)
        );
        assert!(!low.repeat_runs.is_empty());
        let placement = crate::place(&low).unwrap();
        let kernels = crate::KernelBuildPlan::from_program(&low).unwrap();
        for run in &low.kernel_runs {
            crate::materialize_kernel_run(
                run,
                &low.shards,
                &placement.shard_addresses,
                &kernels,
                &BTreeMap::new(),
            )
            .unwrap();
        }
    }

    #[test]
    fn live_residual_and_statistics_have_independent_allocations() {
        let tensor = TensorType::new([4, 1152], Precision::F16, Layout::row_sharded(4));
        let mut program = MidProgram {
            tile_count: 4,
            ..MidProgram::default()
        };
        for index in 0..6 {
            let id = MidValueId(index);
            program.values.push(MidValue {
                id,
                tile_offset: 0,
                tensor_type: tensor.clone(),
                origin: ValueId::from_index(index),
                storage_group: id,
            });
            if index < 4 {
                program.inputs.push(MidInput {
                    name: index.to_string(),
                    kind: GraphInputKind::Host,
                    value: id,
                });
            }
        }
        for (kernel, inputs, result) in [
            (
                TileKernelSpec::Add,
                vec![MidValueId(0), MidValueId(1)],
                MidValueId(4),
            ),
            (
                TileKernelSpec::LayerNorm,
                vec![MidValueId(4), MidValueId(2), MidValueId(3)],
                MidValueId(5),
            ),
        ] {
            program.operations.push(MidOperation {
                source: None,
                results: vec![result],
                kind: MidOperationKind::Primitive(Primitive::Compute {
                    kernel,
                    operands: vec![OperandWindow::default(); inputs.len()],
                    product: None,
                    output_aliases: Vec::new(),
                }),
                inputs,
                estimated_cycles: 0,
                estimated_exchange_cycles: 0,
            });
        }
        program.outputs = vec![MidValueId(4), MidValueId(5)];
        let fused = program.with_elementwise_fusions().unwrap();
        assert_eq!(fused.outputs, program.outputs);
        assert_eq!(
            fused.operations[0].results,
            vec![MidValueId(6), MidValueId(4)]
        );
        assert_eq!(fused.operations[1].inputs[0], MidValueId(4));
        let low = crate::lower_to_tiles(&crate::expand_tiles(&fused).unwrap(), false);
        let placement = crate::place(&low).unwrap();
        let kernels = crate::KernelBuildPlan::from_program(&low).unwrap();
        let mut sums = 0;
        for run in &low.kernel_runs {
            crate::validate_kernel_run(run).unwrap();
            let call = crate::materialize_kernel_run(
                run,
                &low.shards,
                &placement.shard_addresses,
                &kernels,
                &BTreeMap::new(),
            )
            .unwrap();
            if run.kernel == TileKernelSpec::AddLayerNormMoments {
                sums += 1;
                assert_eq!(run.additional_outputs.len(), 1);
                assert_eq!(call.input_addresses.len(), 3);
                assert_ne!(call.output_address, call.input_addresses[2]);
                assert_eq!(call.arguments, vec![1, 1152]);
                assert!(
                    call.input_addresses[..2]
                        .iter()
                        .all(|a| a != &call.input_addresses[2])
                );
            }
        }
        assert_eq!(sums, 4);
    }
}
