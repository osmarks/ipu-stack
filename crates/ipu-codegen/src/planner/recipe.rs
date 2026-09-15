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
    /// Checkpoint-only result bases, resolved against constructor rotations.
    #[serde(default, skip_serializing, with = "crate::mid::site_map")]
    pub legacy_result_bases: BTreeMap<crate::mid::ResultSite, crate::tensor::OwnerMap>,
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
    #[serde(default, with = "crate::mid::site_map")]
    pub packing: BTreeMap<crate::mid::WorkSite, crate::mid::PanelPacking>,
    /// Checkpoint-only global preference, resolved once against eligible copies.
    #[serde(default, skip_serializing)]
    pub legacy_packing_rows: Option<std::num::NonZeroU16>,
    #[serde(default)]
    pub reduction_groups: Vec<crate::mid::ReductionGroup>,
    #[serde(default, skip_serializing)]
    pub legacy_parallel_reductions: usize,
    #[serde(default, skip_serializing)]
    pub legacy_disjoint_copy_sources: bool,
    #[serde(default)]
    pub cast_storage: Option<CastStoragePolicy>,
}

impl Recipe {
    pub(crate) fn has_legacy_choices(&self) -> bool {
        !self.early_casts.is_empty()
            || !self.legacy_cast_sites.is_empty()
            || self.legacy_packing_rows.is_some()
            || !self.legacy_result_bases.is_empty()
            || self.legacy_parallel_reductions != 0
            || self.legacy_disjoint_copy_sources
    }

    /// Compatibility requests are resolved once against executable work. New
    /// recipes name each group and use the ordinary result-home policy.
    pub(crate) fn resolve_legacy_grouping(
        &mut self,
        program: &mut MidProgram,
        checkpoints: bool,
    ) -> Result<(), crate::mid::ProgramError> {
        let limit = std::mem::take(&mut self.legacy_parallel_reductions);
        let separate = std::mem::take(&mut self.legacy_disjoint_copy_sources);
        if limit < 2 && !separate {
            return Ok(());
        }
        if !self.reduction_groups.is_empty() {
            return Err(crate::mid::ProgramError::Invalid(
                "recipe mixes global and named reduction grouping".into(),
            ));
        }
        if limit > 1 && !checkpoints {
            let proposal = program.propose_reduction_groups(limit);
            program.apply_ownership(&crate::mid::OwnerChoices {
                results: proposal.homes.clone(),
                ..Default::default()
            })?;
            self.owners.results.extend(proposal.homes);
            self.reduction_groups = proposal.reductions;
        }
        program.group_reductions(&self.reduction_groups)?;
        if separate {
            let homes = program.propose_preparation_homes(checkpoints);
            program.apply_ownership(&crate::mid::OwnerChoices {
                results: homes.clone(),
                ..Default::default()
            })?;
            self.owners.results.extend(homes);
        }
        Ok(())
    }

    pub(crate) fn resolve_result_homes(
        &mut self,
        program: &MidProgram,
    ) -> Result<(), crate::mid::ProgramError> {
        if self.legacy_result_bases.is_empty() {
            return Ok(());
        }
        let values = program.named_results().collect::<BTreeMap<_, _>>();
        for (site, base) in std::mem::take(&mut self.legacy_result_bases) {
            base.validate(1, program.tile_count)?;
            let value = values.get(&site).ok_or_else(|| {
                crate::mid::ProgramError::Invalid(format!(
                    "legacy result home is unavailable at {site:?}"
                ))
            })?;
            let home = base
                .shifted(
                    i32::from(program.values[value.index() as usize].owners.rotation()),
                    program.tile_count,
                )
                .ok_or_else(|| {
                    crate::mid::ProgramError::Invalid("invalid legacy result owner domain".into())
                })?;
            if self.owners.results.insert(site, home).is_some() {
                return Err(crate::mid::ProgramError::Invalid(
                    "recipe mixes legacy and explicit result homes".into(),
                ));
            }
        }
        Ok(())
    }

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

    pub(crate) fn resolve_packing_choices(
        &mut self,
        program: &MidProgram,
    ) -> Result<(), crate::mid::ProgramError> {
        if let Some(rows) = self.legacy_packing_rows.take() {
            if !self.packing.is_empty() {
                return Err(crate::mid::ProgramError::Invalid(
                    "recipe mixes global and scoped packing choices".into(),
                ));
            }
            self.packing = program.legacy_packing_choices(rows)?;
        }
        Ok(())
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
