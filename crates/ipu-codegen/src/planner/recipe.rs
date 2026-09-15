//! Operator layouts and whole-program optimization settings, separate from mid.
use crate::PipelineConfig;
use crate::graph::{OperationId, ValueId};
use crate::mid::MidProgram;
use crate::planner::operator::OperatorPlan;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq)]
pub(crate) struct Options {
    pub cast_before_copies: bool,
    pub reuse_cast_inputs: bool,
    pub packing_rows: u16,
    pub parallel_reductions: usize,
    pub disjoint_copy_sources: bool,
    pub tile_mapping: Vec<u16>,
}

#[derive(Serialize, Deserialize, Clone, Default, PartialEq, Eq)]
pub(crate) struct Recipe {
    pub plans: BTreeMap<OperationId, OperatorPlan>,
    pub open_boundaries: BTreeSet<ValueId>,
    pub options: Options,
}
impl Recipe {
    pub(crate) fn baseline(config: &PipelineConfig) -> Self {
        Self {
            options: Options {
                reuse_cast_inputs: config.capacity_baseline,
                ..Options::default()
            },
            ..Self::default()
        }
    }
    pub(crate) fn remapped(
        &self,
        mapping: &[u16],
        tile_count: u16,
    ) -> Result<Self, crate::mid::ProgramError> {
        let mut recipe = self.clone();
        let mut owners = crate::tensor::OwnerMap::default();
        if !recipe.options.tile_mapping.is_empty() {
            crate::tensor::remap_owners(
                std::iter::once(&mut owners),
                &recipe.options.tile_mapping,
                tile_count,
            )?;
        }
        crate::tensor::remap_owners(std::iter::once(&mut owners), mapping, tile_count)?;
        recipe.options.tile_mapping = (0..tile_count)
            .map(|tile| owners.tile(tile, tile_count).unwrap())
            .collect();
        Ok(recipe)
    }
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
            (&before.options, &self.options),
        )
    }
}

pub(crate) struct Candidate {
    pub program: MidProgram,
    pub recipe: Recipe,
    pub alternatives: BTreeMap<OperationId, Vec<OperatorPlan>>,
}
