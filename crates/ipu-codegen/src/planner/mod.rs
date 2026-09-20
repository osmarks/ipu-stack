//! Choose layouts at high-operation boundaries, then construct executable mid work.
//! Private temporaries belong to each component's implementation, not this table.

pub(crate) mod budget;

use crate::Layout;
use crate::config::PipelineConfig;
use crate::graph::{HighGraph, Operation, OperationId, OperationKind, ValueId};
use crate::mid::{MidGraph, MidValueId};
use budget::OperationBudget;
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
    #[error("no budget assigned to high operation {0:?}")]
    UnassignedBudget(OperationId),
    #[error("{0} construction is not implemented")]
    Unimplemented(&'static str),
}

pub type PlanningResult<T> = Result<T, PlanningError>;

/// No search or implicit layout selection here. Each component must honor its
/// boundary choices and append its private copies, computation and temporaries.
pub(crate) fn plan(
    graph: &HighGraph,
    layouts: &BoundaryLayouts,
    budgets: &BTreeMap<OperationId, OperationBudget>,
    settings: &PipelineConfig,
) -> PlanningResult<MidGraph> {
    for value in boundary_layouts(graph, settings).keys() {
        if layouts.get(value).and_then(Option::as_ref).is_none() {
            return Err(PlanningError::UnassignedLayout(*value));
        }
    }
    let mut construction = Construction {
        graph,
        layouts,
        budgets,
        settings,
        mid: MidGraph {
            tile_count: settings.tile_count,
            ..MidGraph::default()
        },
        values: BTreeMap::new(),
    };
    construction.inputs()?;
    construction.operations(graph.operations())?;
    construction.mid.outputs = graph
        .outputs()
        .iter()
        .map(|id| construction.values[id])
        .collect();
    Ok(construction.mid)
}

/// Shared construction state: helpers bind high boundary IDs to actual mid
/// values. Algorithm-private temporaries need no high ID or global layout choice.
struct Construction<'a> {
    graph: &'a HighGraph,
    layouts: &'a BoundaryLayouts,
    budgets: &'a BTreeMap<OperationId, OperationBudget>,
    settings: &'a PipelineConfig,
    mid: MidGraph,
    values: BTreeMap<ValueId, MidValueId>,
}

impl Construction<'_> {
    fn operations(&mut self, operations: &[Operation]) -> PlanningResult<()> {
        for operation in operations {
            let budget = self
                .budgets
                .get(&operation.id)
                .ok_or(PlanningError::UnassignedBudget(operation.id))?;
            match &operation.kind {
                OperationKind::Gemm(_) => self.gemm(operation, budget),
                OperationKind::Gelu => self.gelu(operation, budget),
                OperationKind::LayerNorm => self.layer_norm(operation, budget),
                OperationKind::Add => self.add(operation, budget),
                OperationKind::View(_) => self.view(operation, budget),
                OperationKind::Slice(_) => self.slice(operation, budget),
                OperationKind::FlashAttention(_) => self.attention(operation, budget),
                OperationKind::Repeat(_) => self.repeat(operation, budget),
            }?;
        }
        Ok(())
    }

    fn inputs(&mut self) -> PlanningResult<()> {
        Err(PlanningError::Unimplemented("graph inputs"))
    }

    fn gemm(&mut self, _operation: &Operation, _budget: &OperationBudget) -> PlanningResult<()> {
        Err(PlanningError::Unimplemented("GEMM"))
    }

    fn gelu(&mut self, _operation: &Operation, _budget: &OperationBudget) -> PlanningResult<()> {
        Err(PlanningError::Unimplemented("GeLU"))
    }

    fn layer_norm(
        &mut self,
        _operation: &Operation,
        _budget: &OperationBudget,
    ) -> PlanningResult<()> {
        Err(PlanningError::Unimplemented("layernorm"))
    }

    fn add(&mut self, _operation: &Operation, _budget: &OperationBudget) -> PlanningResult<()> {
        Err(PlanningError::Unimplemented("add"))
    }

    fn view(&mut self, _operation: &Operation, _budget: &OperationBudget) -> PlanningResult<()> {
        Err(PlanningError::Unimplemented("view"))
    }

    fn slice(&mut self, _operation: &Operation, _budget: &OperationBudget) -> PlanningResult<()> {
        Err(PlanningError::Unimplemented("slice"))
    }

    fn attention(
        &mut self,
        _operation: &Operation,
        _budget: &OperationBudget,
    ) -> PlanningResult<()> {
        Err(PlanningError::Unimplemented("attention"))
    }

    // Will bind carried/invariant/iterated arguments, recurse through operations,
    // and bind yields. It must preserve Repeat rather than unroll its body.
    fn repeat(&mut self, _operation: &Operation, _budget: &OperationBudget) -> PlanningResult<()> {
        Err(PlanningError::Unimplemented("Repeat"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Precision, TensorFormat, TensorTiling};

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
        let settings = PipelineConfig::new(2).with_input(
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
