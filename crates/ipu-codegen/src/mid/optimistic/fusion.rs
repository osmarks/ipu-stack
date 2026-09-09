use super::search::refresh;
use super::*;

pub(super) fn fuse(
    graph: &DiagnosticMidGraph,
    ops: &[Operation],
    limit: usize,
) -> Option<DiagnosticMidGraph> {
    if limit < 2 {
        return None;
    }
    let mut result = graph.clone();
    let mut steps = Vec::new();
    let mut at = 0;
    let mut changed = false;
    while at < graph.steps.len() {
        let mut end = at;
        let mut consumed = at;
        let mut operations = Vec::new();
        let mut group: Vec<&Step> = Vec::new();
        let mut lifted = Vec::new();
        let mut produced = BTreeSet::new();
        while end < graph.steps.len() && group.len() < limit {
            let step = &graph.steps[end];
            if matches!(step.kind, StepKind::Transform(_)) {
                if at == end || step.inputs.iter().any(|id| produced.contains(id)) {
                    break;
                }
                lifted.push((end, step.clone()));
                end += 1;
                continue;
            }
            let StepKind::Algorithm { .. } = step.kind else {
                break;
            };
            let Some(op) = ops.iter().find(|op| step.sources == [op.id]) else {
                break;
            };
            if !matches!(
                op.kind,
                OperationKind::Add | OperationKind::Gelu | OperationKind::LayerNorm
            ) {
                break;
            }
            if let Some(previous) = group.last()
                && (!step.inputs.iter().any(|id| previous.outputs.contains(id))
                    || graph.values[step.outputs[0]].tensor
                        != graph.values[previous.outputs[0]].tensor)
            {
                break;
            }
            operations.push(op.clone());
            produced.extend(step.outputs.iter().copied());
            group.push(step);
            end += 1;
            consumed = end;
        }
        if operations.len() < 2 {
            steps.push(graph.steps[at].clone());
            at += 1;
            continue;
        }
        let inputs = group
            .iter()
            .flat_map(|s| s.inputs.iter().copied())
            .filter(|id| !produced.contains(id))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let used_later = graph.steps[consumed..]
            .iter()
            .flat_map(|s| s.inputs.iter())
            .chain(&graph.outputs)
            .copied()
            .collect::<BTreeSet<_>>();
        let outputs = produced
            .intersection(&used_later)
            .copied()
            .collect::<Vec<_>>();
        let original = group
            .iter()
            .fold(CycleEstimate::default(), |sum, s| sum.plus(s.cycles));
        let saved_bytes = produced
            .difference(&used_later)
            .map(|&id| {
                crate::estimate::maximum_shard_bytes(&graph.values[id].tensor).saturating_mul(2)
            })
            .sum::<u64>();
        let savings = saved_bytes
            .div_ceil(8)
            .saturating_add((operations.len() as u64 - 1) * 330);
        let known = known_fusion_cost(graph, &group, &operations, &outputs);
        if known.is_some_and(|price| price >= original.conservative) {
            steps.push(graph.steps[at].clone());
            at += 1;
            continue;
        }
        let (cycles, assumptions) = if let Some(price) = known {
            (
                CycleEstimate {
                    optimistic: price,
                    conservative: price,
                },
                BTreeSet::new(),
            )
        } else {
            (
                CycleEstimate {
                    optimistic: original.optimistic.saturating_sub(savings).max(330),
                    conservative: original.conservative,
                },
                BTreeSet::from([Assumption::MissingKernel(format!(
                    "fused {:?} with {} live outputs",
                    operations.iter().map(|op| &op.kind).collect::<Vec<_>>(),
                    outputs.len()
                ))]),
            )
        };
        steps.extend(
            lifted
                .into_iter()
                .filter(|(i, _)| *i < consumed)
                .map(|(_, step)| step),
        );
        steps.push(Step {
            sources: operations.iter().map(|op| op.id).collect(),
            inputs,
            outputs,
            kind: StepKind::FusedElementwise { operations },
            cycles,
            assumptions,
        });
        changed = true;
        at = consumed;
    }
    if !changed {
        return None;
    }
    result.steps = steps;
    refresh(&mut result);
    Some(result)
}

fn known_fusion_cost(
    graph: &DiagnosticMidGraph,
    group: &[&Step],
    operations: &[Operation],
    outputs: &[usize],
) -> Option<u64> {
    if group.len() != 2
        || outputs.len() != 1
        || operations[0].kind != OperationKind::Add
        || group[1].inputs.first() != group[0].outputs.first()
    {
        return None;
    }
    let kernel = match operations[1].kind {
        OperationKind::Gelu => TileKernelSpec::BiasGelu,
        OperationKind::LayerNorm => TileKernelSpec::AddLayerNorm,
        _ => return None,
    };
    if !super::super::elementwise::compatible_fusion(
        &kernel,
        &graph.values[group[0].inputs[0]].tensor,
        &graph.values[group[0].inputs[1]].tensor,
        &graph.values[outputs[0]].tensor,
    ) {
        return None;
    }
    let mut inputs = group[0].inputs.clone();
    inputs.extend_from_slice(&group[1].inputs[1..]);
    let values = graph
        .values
        .iter()
        .enumerate()
        .map(|(i, value)| {
            let id = MidValueId::from_index(i as u32);
            MidValue {
                id,
                tensor_type: value.tensor.clone(),
                origin: value.origin,
                storage_group: id,
                tile_offset: 0,
            }
        })
        .collect::<Vec<_>>();
    let op = MidOperation {
        source: None,
        results: vec![MidValueId::from_index(outputs[0] as u32)],
        kind: MidOperationKind::Primitive(Primitive::Compute {
            kernel,
            operands: vec![OperandWindow::default(); inputs.len()],
            product: None,
            output_aliases: Vec::new(),
        }),
        inputs: inputs
            .into_iter()
            .map(|id| MidValueId::from_index(id as u32))
            .collect(),
        estimated_cycles: 0,
        estimated_exchange_cycles: 0,
    };
    crate::estimate::operation_cost(&op, &values).map(|(cost, _, _)| cost.total)
}
