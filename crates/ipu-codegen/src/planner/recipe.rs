//! Selected operator implementations and boundaries reopened by local search.
use crate::PipelineConfig;
use crate::graph::{OperationId, ValueId};
use crate::mid::MidProgram;
use crate::planner::operator::OperatorPlan;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Default, PartialEq, Eq)]
pub(crate) struct Recipe {
    pub plans: BTreeMap<OperationId, OperatorPlan>,
    pub open_boundaries: BTreeSet<ValueId>,
}
impl Recipe {
    pub(crate) fn changes<'a>(&'a self, before: &'a Self) -> impl std::fmt::Debug + 'a {
        (
            self.plans
                .keys()
                .chain(before.plans.keys())
                .copied()
                .filter(|id| self.plans.get(id) != before.plans.get(id))
                .collect::<BTreeSet<_>>(),
            self.open_boundaries
                .symmetric_difference(&before.open_boundaries)
                .copied()
                .collect::<Vec<_>>(),
        )
    }
}

pub(crate) struct Candidate {
    pub program: MidProgram,
    pub config: PipelineConfig,
    pub recipe: Recipe,
    pub alternatives: BTreeMap<OperationId, Vec<OperatorPlan>>,
}
