//! Elementwise implementations require a common operand/output layout. The
//! catalogue supplies layouts from both sides of a chain; DP connects operands.

use super::candidates::{BoundaryValue, Candidate};
use super::{BoundaryLayouts, PlanningError, PlanningResult};
use crate::config::PipelineConfig;
use crate::graph::{HighGraph, OperationKind, ValueId};
use crate::mid::{MidOperation, MidOperationKind, OperandIndexing};
use crate::tensor::{Layout, OwnerMap, Precision, TensorType};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn generate(
    high: &HighGraph,
    position: usize,
    tensors: &BTreeMap<ValueId, TensorType>,
    choices: &BoundaryLayouts,
    settings: &PipelineConfig,
    kind: MidOperationKind,
    offers: &BTreeMap<ValueId, BTreeSet<Layout>>,
    existing: &[Candidate],
) -> PlanningResult<Vec<Candidate>> {
    let op = &high.operations()[position];
    let result = op.results[0];
    let shape = high.value_shape(result).unwrap();
    if op
        .inputs
        .iter()
        .any(|id| tensors[id].shape != *shape || tensors[id].format.precision != Precision::F16)
    {
        return Err(PlanningError::Unimplemented(
            "broadcast or non-FP16 elementwise candidates",
        ));
    }
    // The fused intermediate must neither escape nor have a required layout.
    let fused = high.operations().get(position + 1).filter(|next| {
        matches!(op.kind, OperationKind::Add)
            && matches!(next.kind, OperationKind::Gelu)
            && next.inputs == [result]
            && !high.outputs().contains(&result)
            && choices.get(&result).and_then(Option::as_ref).is_none()
            && !high.operations()[position + 2..]
                .iter()
                .any(|later| high.operation_inputs(later).any(|v| v == result))
    });
    let existing = existing
        .iter()
        .map(|candidate| {
            (
                candidate.end,
                candidate.graph.values[candidate.graph.outputs[0].index() as usize]
                    .tensor_type
                    .format
                    .layout
                    .clone(),
            )
        })
        .collect::<BTreeSet<_>>();
    let mut candidates = Vec::new();
    for (end, output, kind) in std::iter::once((position + 1, result, kind))
        .chain(fused.map(|next| (position + 2, next.results[0], MidOperationKind::BiasGelu)))
    {
        let layouts = choices
            .get(&output)
            .and_then(Option::as_ref)
            .map(|layout| BTreeSet::from([layout.clone()]))
            .unwrap_or_else(|| {
                op.inputs
                    .iter()
                    .chain([&output])
                    .filter_map(|id| offers.get(id))
                    .flatten()
                    .cloned()
                    .chain([Layout::row_sharded(settings.tile_count)])
                    .collect()
            });
        for layout in layouts {
            if existing.contains(&(end, layout.clone())) {
                continue;
            }
            if layout.validate_tile_count(settings.tile_count).is_err()
                || layout.resolve(shape).is_err()
            {
                continue;
            }
            let tensor = TensorType::new(shape.0.clone(), Precision::F16, layout);
            let operands = op
                .inputs
                .iter()
                .map(|&id| {
                    (
                        id,
                        BoundaryValue {
                            tensor: tensor.clone(),
                            owners: OwnerMap::default(),
                        },
                    )
                })
                .collect();
            let mut candidate = Candidate::inputs(high, &operands, settings.tile_count, end);
            let inputs = op
                .inputs
                .iter()
                .map(|id| candidate.bindings[id])
                .collect::<Vec<_>>();
            let value = super::construction::value(
                &mut candidate.graph,
                output,
                tensor,
                OwnerMap::default(),
            );
            candidate.graph.operations.push(MidOperation {
                source: Some(high.operations()[end - 1].id),
                operands: vec![OperandIndexing::Elementwise { result: 0 }; inputs.len()],
                inputs,
                results: vec![value],
                kind: kind.clone(),
                output_aliases: Vec::new(),
                output_windows: Vec::new(),
            });
            candidate.bindings.insert(output, value);
            candidate.graph.outputs = vec![value];
            candidates.push(candidate);
        }
    }
    Ok(candidates)
}
