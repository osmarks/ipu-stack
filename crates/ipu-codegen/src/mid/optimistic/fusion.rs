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
        let assumptions = BTreeSet::from([Assumption::MissingKernel(format!(
            "fused {:?} with {} live outputs",
            operations.iter().map(|op| &op.kind).collect::<Vec<_>>(),
            outputs.len()
        ))]);
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
            cycles: CycleEstimate {
                optimistic: original.optimistic.saturating_sub(savings).max(330),
                conservative: original.conservative,
            },
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
