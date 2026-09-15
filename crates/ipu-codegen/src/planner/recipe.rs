//! Selected family and boundary/rewrite choices, separate from executable mid.
use crate::compile::PipelineConfig;
use crate::graph::{OperationId, ValueId};
use crate::mid::MidProgram;
use crate::mid::cast::{CastStorage, CastStoragePolicy};

use crate::planner::operator::OperatorPlan;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Serialize, Deserialize, Clone, Default, PartialEq, Eq)]
pub(crate) struct Recipe {
    pub plans: BTreeMap<OperationId, OperatorPlan>,
    #[serde(default)]
    pub owners: crate::mid::OwnerChoices,
    pub open_boundaries: BTreeSet<ValueId>,
    /// Initial capacity-baseline requests; converted to
    /// individual cast sites after expansion and never saved in this form.
    #[serde(default, skip_serializing)]
    pub early_casts: BTreeSet<OperationId>,
    #[serde(default)]
    pub cast_before_copies: BTreeSet<crate::mid::WorkSite>,
    #[serde(default, with = "crate::mid::site_map")]
    pub packing: BTreeMap<crate::mid::WorkSite, crate::mid::PanelPacking>,
    #[serde(default)]
    pub reduction_groups: Vec<crate::mid::ReductionGroup>,
    #[serde(default)]
    pub cast_storage: Option<CastStoragePolicy>,
}

impl Recipe {
    pub(crate) fn remapped(
        &self,
        graph: &crate::ComputeGraph,
        mapping: &[u16],
        tile_count: u16,
    ) -> Result<Self, crate::mid::ProgramError> {
        let mut recipe = self.clone();
        recipe.owners.remap_tiles(
            graph.inputs().iter().map(|input| input.value),
            graph.walk_operations().map(|operation| operation.id),
            mapping,
            tile_count,
        )?;
        crate::tensor::remap_owners(
            recipe
                .packing
                .values_mut()
                .map(|choice| &mut choice.workspace),
            mapping,
            tile_count,
        )?;
        Ok(recipe)
    }

    pub(crate) fn resolve_cast_choices(
        &mut self,
        available: &BTreeSet<crate::mid::WorkSite>,
    ) -> crate::planner::error::LoweringResult<()> {
        self.cast_before_copies.extend(
            available
                .iter()
                .filter(|site| self.early_casts.contains(&site.source))
                .cloned(),
        );
        self.early_casts.clear();
        if let Some(site) = self.cast_before_copies.difference(available).next() {
            return Err(crate::planner::error::LoweringError::UnavailableCastChoice(
                site.clone(),
            ));
        }
        Ok(())
    }

    pub(crate) fn normalize(&mut self, config: &PipelineConfig) {
        self.cast_storage.get_or_insert_with(|| {
            CastStoragePolicy::new(if config.capacity_baseline {
                CastStorage::ReuseIfSmaller
            } else {
                CastStorage::Separate
            })
        });
    }

    pub(crate) fn changes<'a>(&'a self, before: &'a Self) -> impl std::fmt::Debug + 'a {
        #[derive(Debug)]
        #[allow(dead_code)] // Fields are consumed by Debug only.
        struct Changes<'a> {
            plans: BTreeSet<OperationId>,
            boundaries: Vec<ValueId>,
            casts: Vec<crate::mid::WorkSite>,
            early_casts: Vec<OperationId>,
            packing: BTreeSet<crate::mid::WorkSite>,
            reductions: (
                &'a [crate::mid::ReductionGroup],
                &'a [crate::mid::ReductionGroup],
            ),
            cast_storage: (&'a Option<CastStoragePolicy>, &'a Option<CastStoragePolicy>),
            owners: (&'a crate::mid::OwnerChoices, &'a crate::mid::OwnerChoices),
        }
        Changes {
            plans: self
                .plans
                .keys()
                .chain(before.plans.keys())
                .copied()
                .filter(|id| self.plans.get(id) != before.plans.get(id))
                .collect(),
            boundaries: self
                .open_boundaries
                .symmetric_difference(&before.open_boundaries)
                .copied()
                .collect(),
            casts: self
                .cast_before_copies
                .symmetric_difference(&before.cast_before_copies)
                .cloned()
                .collect(),
            early_casts: self
                .early_casts
                .symmetric_difference(&before.early_casts)
                .copied()
                .collect(),
            packing: self
                .packing
                .keys()
                .chain(before.packing.keys())
                .filter(|site| self.packing.get(*site) != before.packing.get(*site))
                .cloned()
                .collect(),
            reductions: (&before.reduction_groups, &self.reduction_groups),
            cast_storage: (&before.cast_storage, &self.cast_storage),
            owners: (&before.owners, &self.owners),
        }
    }
}

pub(crate) struct Candidate {
    pub program: MidProgram,
    pub recipe: Recipe,
    pub alternatives: BTreeMap<OperationId, Vec<OperatorPlan>>,
    pub cast_sites: BTreeSet<crate::mid::WorkSite>,
    pub cast_storage_sites: BTreeSet<crate::mid::WorkSite>,
    pub packing_choices: BTreeMap<crate::mid::WorkSite, Vec<crate::mid::PanelPacking>>,
    pub grouping_choices: Vec<crate::mid::GroupProposal>,
}
