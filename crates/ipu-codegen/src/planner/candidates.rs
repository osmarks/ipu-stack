//! Construct complete local alternatives, including conversions, as mid graphs.
//! Only elementwise construction is implemented initially. GEMM/attention/Repeat
//! must get their own direct constructors, not adapters to the parked planner.

use super::{BoundaryLayouts, PlanningError, PlanningResult};
use crate::config::PipelineConfig;
use crate::graph::{GraphInputKind, HighGraph, OperationKind, ValueId};
use crate::mid::{
    CoordinateMapping, MidGraph, MidInput, MidOperation, MidOperationKind, MidValue, MidValueId,
    OperandIndexing, PackingPolicy, default_copy_policy,
};
use crate::tensor::{Layout, OwnerMap, Precision, TensorType};
use std::collections::BTreeMap;

/// No allocation IDs from a particular history belong in a boundary key.
/// Cross-boundary aliases are deliberately unsupported until their backing
/// identity and access requirements can be included in this key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct BoundaryValue {
    pub tensor: TensorType,
    pub owners: OwnerMap,
}

pub(super) type LiveValues = BTreeMap<ValueId, BoundaryValue>;

#[derive(Debug)]
pub(super) struct Candidate {
    /// First high operation not implemented by this fragment.
    pub end: usize,
    pub graph: MidGraph,
    /// High boundary identity -> fragment-local value. Private values have no entry.
    pub bindings: BTreeMap<ValueId, MidValueId>,
}

impl Candidate {
    /// Import the whole live boundary. Unrelated activations become pass-through
    /// outputs when extending the path, so ordinary mid liveness accounts for
    /// residual storage without a separate approximation of its lifetime.
    pub fn inputs(high: &HighGraph, live: &LiveValues, tile_count: u16, end: usize) -> Self {
        let mut candidate = Self {
            end,
            graph: MidGraph {
                tile_count,
                ..MidGraph::default()
            },
            bindings: BTreeMap::new(),
        };
        for (&origin, boundary) in live {
            let id = candidate.value(origin, boundary.tensor.clone(), boundary.owners.clone());
            let input = high.inputs().iter().find(|input| input.value == origin);
            candidate.graph.inputs.push(MidInput {
                name: input.map_or_else(|| format!("value.{}", origin.index()), |i| i.name.clone()),
                kind: input.map_or(GraphInputKind::Host, |i| i.kind),
                value: id,
            });
            candidate.bindings.insert(origin, id);
        }
        candidate
    }

    fn value(&mut self, origin: ValueId, tensor_type: TensorType, owners: OwnerMap) -> MidValueId {
        let id =
            MidValueId::from_index(self.graph.values.len().try_into().expect("mid value count"));
        self.graph.values.push(MidValue {
            id,
            origin,
            tensor_type,
            owners,
            storage_group: id,
        });
        id
    }
}

pub(super) fn generate(
    high: &HighGraph,
    position: usize,
    live: &LiveValues,
    choices: &BoundaryLayouts,
    settings: &PipelineConfig,
) -> PlanningResult<Vec<Candidate>> {
    let op = &high.operations()[position];
    let kind = match op.kind {
        OperationKind::Gelu => MidOperationKind::Gelu,
        OperationKind::Add => MidOperationKind::Add,
        OperationKind::Gemm(_) => return Err(PlanningError::Unimplemented("GEMM candidates")),
        OperationKind::LayerNorm => {
            return Err(PlanningError::Unimplemented("layernorm candidates"));
        }
        OperationKind::View(_) | OperationKind::Slice(_) => {
            return Err(PlanningError::Unimplemented("view/slice candidates"));
        }
        OperationKind::FlashAttention(_) => {
            return Err(PlanningError::Unimplemented("attention candidates"));
        }
        OperationKind::Repeat(_) => return Err(PlanningError::Unimplemented("Repeat candidates")),
    };
    let result = op.results[0];
    let shape = high.value_shape(result).unwrap();
    for input in &op.inputs {
        let boundary = live
            .get(input)
            .ok_or(PlanningError::InvalidFragment("missing high input"))?;
        if boundary.tensor.shape != *shape || boundary.tensor.format.precision != Precision::F16 {
            return Err(PlanningError::Unimplemented(
                "broadcast or non-FP16 elementwise candidates",
            ));
        }
    }

    // A fused edge consumes the same high operations as the two unfused edges.
    // The intermediate must not escape or have a separately constrained layout.
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
    let mut candidates = Vec::new();
    for (end, output, kind) in std::iter::once((position + 1, result, kind))
        .chain(fused.map(|next| (position + 2, next.results[0], MidOperationKind::BiasGelu)))
    {
        let layouts = if let Some(layout) = choices.get(&output).and_then(Option::as_ref) {
            vec![layout.clone()]
        } else {
            let mut layouts = op
                .inputs
                .iter()
                .map(|id| live[id].tensor.format.layout.clone())
                .collect::<Vec<_>>();
            layouts.push(Layout::row_sharded(settings.tile_count));
            layouts.sort();
            layouts.dedup();
            layouts
        };
        for layout in layouts {
            // A proposed distribution need not support this rank or shape.
            // Reject it before construction without losing the other alternatives.
            if layout.validate_tile_count(settings.tile_count).is_err()
                || layout.resolve(shape).is_err()
            {
                continue;
            }
            let mut candidate = Candidate::inputs(high, live, settings.tile_count, end);
            let tensor = TensorType {
                shape: shape.clone(),
                format: crate::TensorFormat {
                    precision: Precision::F16,
                    layout,
                },
            };
            let owners = OwnerMap::default();
            let mut inputs = Vec::new();
            let mut prepared = BTreeMap::new();
            for &input in &op.inputs {
                if let Some(&value) = prepared.get(&input) {
                    inputs.push(value);
                    continue;
                }
                let original = candidate.bindings[&input];
                let from = &candidate.graph.values[original.index() as usize];
                let value = if from.tensor_type == tensor && from.owners == owners {
                    original
                } else {
                    let policy =
                        default_copy_policy(&from.tensor_type.format.layout, &tensor.format.layout);
                    let copied = candidate.value(input, tensor.clone(), owners.clone());
                    candidate.graph.operations.push(MidOperation {
                        source: Some(op.id),
                        inputs: vec![original],
                        results: vec![copied],
                        kind: MidOperationKind::Copy {
                            mapping: CoordinateMapping::default(),
                            policy,
                            packing: PackingPolicy::Staged,
                        },
                        operands: Vec::new(),
                        output_aliases: Vec::new(),
                        output_windows: Vec::new(),
                    });
                    copied
                };
                inputs.push(value);
                prepared.insert(input, value);
            }
            let value = candidate.value(output, tensor, owners);
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
            candidates.push(candidate);
        }
    }
    Ok(candidates)
}
