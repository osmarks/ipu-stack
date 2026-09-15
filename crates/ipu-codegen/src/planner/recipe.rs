//! Selected family and boundary/rewrite choices, separate from executable mid.
use crate::compile::PipelineConfig;
use crate::graph::{OperationId, ValueId};
use crate::mid::MidProgram;

use crate::planner::operator::OperatorPlan;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Serialize, Deserialize, Clone, Default, PartialEq, Eq)]
pub(crate) struct Recipe {
    pub plans: BTreeMap<OperationId, OperatorPlan>,
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
    pub in_place_casts: Option<bool>,
}

impl Recipe {
    /// Legacy ordinals are interpreted once, at the checkpoint/construction
    /// boundary. Executable mid and subsequent proposals use named sites only.
    pub(crate) fn resolve_cast_choices(
        &mut self,
        program: &MidProgram,
        available: &BTreeSet<crate::mid::WorkSite>,
    ) -> crate::planner::error::LoweringResult<()> {
        fn visit(
            operations: &[crate::mid::MidOperation],
            values: &[crate::mid::MidValue],
            ordinals: &mut BTreeMap<Option<OperationId>, u32>,
            requested: &mut BTreeSet<(Option<OperationId>, u32)>,
            selected: &mut BTreeSet<crate::mid::WorkSite>,
        ) -> crate::planner::error::LoweringResult<()> {
            for operation in operations {
                if let crate::mid::MidOperationKind::Repeat(repeat) = &operation.kind {
                    visit(
                        &repeat.body.operations,
                        values,
                        ordinals,
                        requested,
                        selected,
                    )?;
                } else if crate::mid::rewrite::fp8_cast(operation, values).is_some() {
                    let ordinal = ordinals.entry(operation.source).or_default();
                    if requested.remove(&(operation.source, *ordinal)) {
                        selected.insert(operation.work_site().ok_or(
                            crate::planner::error::LoweringError::UnknownLegacyCastChoice(
                                operation.source,
                                *ordinal,
                            ),
                        )?);
                    }
                    *ordinal += 1;
                }
            }
            Ok(())
        }
        if !self.legacy_cast_sites.is_empty() {
            visit(
                &program.operations,
                &program.values,
                &mut BTreeMap::new(),
                &mut self.legacy_cast_sites,
                &mut self.cast_before_copies,
            )?;
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
        self.in_place_casts = Some(self.in_place_casts.unwrap_or(config.capacity_baseline));
    }

    pub(crate) fn changes(&self, before: &Self) -> impl std::fmt::Debug {
        #[derive(Debug)]
        #[allow(dead_code)] // Fields are consumed by Debug only.
        struct Changes {
            plans: BTreeSet<OperationId>,
            boundaries: Vec<ValueId>,
            casts: Vec<crate::mid::WorkSite>,
            early_casts: Vec<OperationId>,
            packing_rows: (Option<u16>, Option<u16>),
            parallel_reductions: (usize, usize),
            disjoint_copy_sources: (bool, bool),
            in_place_casts: (Option<bool>, Option<bool>),
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
            in_place_casts: (before.in_place_casts, self.in_place_casts),
        }
    }
}

pub(crate) struct Candidate {
    pub program: MidProgram,
    pub recipe: Recipe,
    pub alternatives: BTreeMap<OperationId, Vec<OperatorPlan>>,
    pub cast_sites: BTreeSet<crate::mid::WorkSite>,
}
