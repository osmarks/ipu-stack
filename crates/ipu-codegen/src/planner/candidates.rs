//! Search edges are complete local mid graphs. Operator families choose their
//! layouts, algorithms and conversions; this module owns the common boundary
//! representation and dispatches construction.

use super::{BoundaryLayouts, PlanningError, PlanningResult, elementwise};
use crate::config::PipelineConfig;
use crate::graph::{GraphInputKind, HighGraph, OperationKind, ValueId};
use crate::mid::{MidGraph, MidInput, MidOperationKind, MidValue, MidValueId};
use crate::tensor::{OwnerMap, TensorType};
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

    pub(super) fn value(
        &mut self,
        origin: ValueId,
        tensor_type: TensorType,
        owners: OwnerMap,
    ) -> MidValueId {
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
    match high.operations()[position].kind {
        OperationKind::Add => elementwise::generate(
            high,
            position,
            live,
            choices,
            settings,
            MidOperationKind::Add,
        ),
        OperationKind::Gelu => elementwise::generate(
            high,
            position,
            live,
            choices,
            settings,
            MidOperationKind::Gelu,
        ),
        OperationKind::Gemm(_) => Err(PlanningError::Unimplemented("GEMM candidates")),
        OperationKind::LayerNorm => Err(PlanningError::Unimplemented("layernorm candidates")),
        OperationKind::View(_) | OperationKind::Slice(_) => {
            Err(PlanningError::Unimplemented("view/slice candidates"))
        }
        OperationKind::FlashAttention(_) => {
            Err(PlanningError::Unimplemented("attention candidates"))
        }
        OperationKind::Repeat(_) => Err(PlanningError::Unimplemented("Repeat candidates")),
    }
}
