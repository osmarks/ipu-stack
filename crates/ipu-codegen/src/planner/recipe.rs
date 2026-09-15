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
    /// Initial capacity-baseline or legacy checkpoint requests; converted to
    /// individual cast sites after expansion and never saved in this form.
    #[serde(default, skip_serializing)]
    pub early_casts: BTreeSet<OperationId>,
    #[serde(default)]
    pub cast_before_copies: BTreeSet<crate::mid::WorkSite>,
    /// Version-one checkpoint ordinals, resolved once against the constructed family.
    #[serde(default, skip_serializing)]
    pub legacy_cast_sites: BTreeSet<(Option<OperationId>, u32)>,
    pub packing_rows: Option<u16>,
    pub parallel_reductions: usize,
    pub disjoint_copy_sources: bool,
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
        Ok(recipe)
    }

    /// Legacy ordinals are interpreted once, at the checkpoint/construction
    /// boundary. Executable mid and subsequent proposals use named sites only.
    pub(crate) fn resolve_cast_choices(
        &mut self,
        program: &MidProgram,
        available: &BTreeSet<crate::mid::WorkSite>,
    ) -> crate::planner::error::LoweringResult<()> {
        if !self.legacy_cast_sites.is_empty() {
            let mut ordinals = BTreeMap::new();
            for operation in program.walk_operations() {
                if crate::mid::rewrite::fp8_cast(operation, &program.values).is_none() {
                    continue;
                }
                let ordinal = ordinals.entry(operation.source).or_insert(0);
                if self.legacy_cast_sites.remove(&(operation.source, *ordinal)) {
                    self.cast_before_copies.insert(operation.work_site().ok_or(
                        crate::planner::error::LoweringError::UnknownLegacyCastChoice(
                            operation.source,
                            *ordinal,
                        ),
                    )?);
                }
                *ordinal += 1;
            }
            if let Some(&(source, ordinal)) = self.legacy_cast_sites.first() {
                return Err(
                    crate::planner::error::LoweringError::UnknownLegacyCastChoice(source, ordinal),
                );
            }
        }
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
            packing_rows: (Option<u16>, Option<u16>),
            parallel_reductions: (usize, usize),
            disjoint_copy_sources: (bool, bool),
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
            packing_rows: (before.packing_rows, self.packing_rows),
            parallel_reductions: (before.parallel_reductions, self.parallel_reductions),
            disjoint_copy_sources: (before.disjoint_copy_sources, self.disjoint_copy_sources),
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
}
