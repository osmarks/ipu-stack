//! A catalogue of implementations, independent of search histories. Inputs name
//! required operand representations; outputs name produced representations.
//! Search supplies boundary conversions and accounts for unrelated live values.

use super::{BoundaryLayouts, PlanningError, PlanningResult, elementwise};
use crate::config::PipelineConfig;
use crate::graph::{GraphInputKind, HighGraph, OperationKind, ValueId};
use crate::mid::{MidGraph, MidInput, MidOperationKind, MidValueId};
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

#[derive(Clone, Debug)]
pub(super) struct Candidate {
    /// First high operation not implemented by this fragment.
    pub end: usize,
    pub graph: MidGraph,
    /// High boundary identity -> fragment-local value. Private values have no entry.
    pub bindings: BTreeMap<ValueId, MidValueId>,
}

impl Candidate {
    /// Import exactly the supplied representations. Generators supply operands;
    /// search supplies its live boundary when connecting an implementation.
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
            let id = super::construction::value(
                &mut candidate.graph,
                origin,
                boundary.tensor.clone(),
                boundary.owners.clone(),
            );
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
}

pub(super) fn catalogue(
    high: &HighGraph,
    choices: &BoundaryLayouts,
    settings: &PipelineConfig,
) -> PlanningResult<Vec<Vec<Candidate>>> {
    let mut tensors = BTreeMap::new();
    let mut offers = BTreeMap::<ValueId, std::collections::BTreeSet<crate::Layout>>::new();
    for input in high.inputs() {
        if input.kind == GraphInputKind::Host
            && !high.outputs().contains(&input.value)
            && !high
                .operations()
                .iter()
                .any(|op| high.operation_inputs(op).any(|id| id == input.value))
        {
            continue;
        }
        let format = super::parameters::initial_format(
            high,
            input,
            choices.get(&input.value).and_then(Option::as_ref),
            settings,
        )?;
        offers
            .entry(input.value)
            .or_default()
            .insert(format.layout.clone());
        tensors.insert(
            input.value,
            TensorType {
                shape: input.shape.clone(),
                format,
            },
        );
    }
    let mut catalogue = vec![Vec::new(); high.operations().len()];
    // Candidate ports supply a finite vocabulary of boundary layouts. Revisit
    // construction when a new port layout becomes available; each family adds
    // only implementations it has not already offered. No search state is used.
    loop {
        let mut changed = false;
        for (position, alternatives) in catalogue.iter_mut().enumerate() {
            let additions = match high.operations()[position].kind {
                OperationKind::Add => elementwise::generate(
                    high,
                    position,
                    &tensors,
                    choices,
                    settings,
                    MidOperationKind::Add,
                    &offers,
                    alternatives,
                )?,
                OperationKind::Gelu => elementwise::generate(
                    high,
                    position,
                    &tensors,
                    choices,
                    settings,
                    MidOperationKind::Gelu,
                    &offers,
                    alternatives,
                )?,
                OperationKind::Gemm(options) => super::gemm::generate(
                    high,
                    position,
                    &tensors,
                    choices,
                    settings,
                    options,
                    alternatives,
                )?,
                _ => return Err(PlanningError::Unimplemented("operator candidates")),
            };
            for candidate in additions {
                for id in candidate
                    .graph
                    .inputs
                    .iter()
                    .map(|i| i.value)
                    .chain(candidate.graph.outputs.iter().copied())
                {
                    let value = &candidate.graph.values[id.index() as usize];
                    changed |= offers
                        .entry(value.origin)
                        .or_default()
                        .insert(value.tensor_type.format.layout.clone());
                    tensors
                        .entry(value.origin)
                        .or_insert_with(|| value.tensor_type.clone());
                }
                alternatives.push(candidate);
            }
            if alternatives.is_empty() {
                return Err(PlanningError::NoPlan(position + 1));
            }
        }
        if !changed {
            break;
        }
    }
    for alternatives in &mut catalogue {
        let mut retained = Vec::<Candidate>::new();
        let mut buckets =
            std::collections::HashMap::<_, Vec<usize>, foldhash::fast::FixedState>::default();
        for mut candidate in alternatives.drain(..) {
            // Compare complete graphs only within identical boundary signatures.
            // Different private implementations remain distinct here; their
            // memory tradeoffs depend on the state to which search connects them.
            let ports = candidate
                .graph
                .inputs
                .iter()
                .map(|input| input.value)
                .chain(candidate.graph.outputs.iter().copied())
                .map(|id| {
                    let value = &candidate.graph.values[id.index() as usize];
                    (
                        value.origin,
                        value.tensor_type.clone(),
                        value.owners.clone(),
                    )
                })
                .collect::<Vec<_>>();
            let bucket = buckets
                .entry((candidate.end, candidate.graph.inputs.len(), ports))
                .or_default();
            // Check kernel/cost-model support once, before search. Memory
            // dominance is evaluated after connection: an imported parameter
            // requirement may become either resident storage or scratch.
            if candidate.graph.refresh_estimates(settings.target).is_none() {
                continue;
            }
            if !bucket.iter().any(|&i| {
                retained[i].graph == candidate.graph && retained[i].bindings == candidate.bindings
            }) {
                bucket.push(retained.len());
                retained.push(candidate);
            }
        }
        *alternatives = retained;
    }
    Ok(catalogue)
}
