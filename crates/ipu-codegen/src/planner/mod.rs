//! Search a fixed high-operation order using executable mid fragments as edges.
//! Boundary state is shared by histories; cycles and memory remain Pareto labels.
//! Construction never calls back into package building, placement or scheduling.

pub(crate) mod budget;
mod candidates;
mod construction;
mod elementwise;
mod gemm;
mod parameters;
mod search;

use crate::Layout;
use crate::config::PipelineConfig;
use crate::graph::{HighGraph, Operation, OperationKind, ValueId};
use crate::mid::MidGraph;
pub(crate) use budget::SearchLimits;
#[cfg(test)]
use ipu_target::Target;
use search::Search;
use std::collections::BTreeMap;

/// One choice per high value, shared by its producer and every consumer.
/// None means not selected yet; it is not an instruction to lowering.
pub(crate) type BoundaryLayouts = BTreeMap<ValueId, Option<Layout>>;

/// Includes Repeat's external inputs (also iterated parameters), body arguments,
/// body operation boundaries and yields. Value IDs are unique across regions.
pub(crate) fn boundary_layouts(graph: &HighGraph, settings: &PipelineConfig) -> BoundaryLayouts {
    fn collect(graph: &HighGraph, operations: &[Operation], layouts: &mut BoundaryLayouts) {
        for operation in operations {
            for value in graph
                .operation_inputs(operation)
                .chain(operation.results.iter().copied())
            {
                layouts.entry(value).or_insert(None);
            }
            if let OperationKind::Repeat(repeat) = &operation.kind {
                for &value in repeat.body.arguments.iter().chain(&repeat.body.yields) {
                    layouts.entry(value).or_insert(None);
                }
                collect(graph, &repeat.body.operations, layouts);
            }
        }
    }
    let mut layouts = BoundaryLayouts::new();
    for input in graph.inputs() {
        layouts.insert(
            input.value,
            settings.inputs.get(&input.value).map(|f| f.layout.clone()),
        );
    }
    for &value in graph.outputs() {
        layouts.entry(value).or_insert(None);
    }
    collect(graph, graph.operations(), &mut layouts);
    layouts
}

#[derive(Debug, thiserror::Error)]
pub enum PlanningError {
    #[error("no layout selected for high value {0:?}")]
    UnassignedLayout(ValueId),
    #[error("no fitting path through high operation boundary {0}")]
    NoPlan(usize),
    #[error("invalid planner fragment: {0}")]
    InvalidFragment(&'static str),
    #[error("{0} construction is not implemented")]
    Unimplemented(&'static str),
    #[error("memory profile: {0}")]
    MemoryProfile(String),
}

pub type PlanningResult<T> = Result<T, PlanningError>;

/// Layout entries containing Some constrain the corresponding high boundary.
/// None leaves it to candidate generation. Host inputs need explicit formats;
/// unspecified parameters start in compact FP16 homes and can change at first use.
pub(crate) fn plan(
    graph: &HighGraph,
    layouts: &BoundaryLayouts,
    settings: &PipelineConfig,
    limits: SearchLimits,
) -> PlanningResult<MidGraph> {
    let mut search = Search::new(graph, layouts, settings, limits)?;
    let catalogue = candidates::catalogue(graph, layouts, settings)?;
    for (position, candidates) in catalogue.into_iter().enumerate() {
        for state in search.take_states(position) {
            search.extend(position, &state, &candidates)?;
        }
    }
    search.finish()
}

#[cfg(test)]
mod search_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Precision, TensorFormat, TensorTiling};

    pub(super) fn gelu(x: f64) -> f64 {
        0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh())
    }

    #[test]
    fn boundaries_share_branch_values_and_include_repeat_parameters_and_body() {
        let mut graph = HighGraph::new();
        let input = graph.host_input("input", [8, 8]).unwrap();
        let weights = (0..2)
            .map(|i| graph.parameter(format!("weights.{i}"), [8, 8]).unwrap())
            .collect::<Vec<_>>();
        let sequence = graph.value_sequence("weights", weights.clone()).unwrap();
        let mut body_values = Vec::new();
        let repeated = graph
            .repeat(2, [input], [], [sequence], |body, arguments| {
                let product = body.gemm(arguments.carried[0], arguments.iterated[0])?;
                let residual = body.add(product, arguments.carried[0])?;
                body_values.extend(&arguments.carried);
                body_values.extend(&arguments.iterated);
                body_values.extend([product, residual]);
                Ok(vec![residual])
            })
            .unwrap()[0];
        let output = graph.add(input, repeated).unwrap();
        graph.set_outputs([output]).unwrap();
        let fixed = Layout::row_major(TensorTiling::replicated(2));
        let settings = PipelineConfig::new(Target::Ipu21, 2).with_input(
            input,
            TensorFormat {
                precision: Precision::F16,
                layout: fixed.clone(),
            },
        );
        let layouts = boundary_layouts(&graph, &settings);
        let expected = [input, repeated, output]
            .into_iter()
            .chain(weights)
            .chain(body_values)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            layouts
                .keys()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            expected
        );
        assert_eq!(layouts[&input], Some(fixed));
        assert!(
            layouts
                .iter()
                .all(|(&value, layout)| value == input || layout.is_none())
        );
    }
}
