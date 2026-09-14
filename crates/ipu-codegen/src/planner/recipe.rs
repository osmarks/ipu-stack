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
    pub cast_before_copies: BTreeSet<crate::mid::cast_order::CastSite>,
    pub packing_rows: Option<u16>,
    pub parallel_reductions: usize,
    pub disjoint_copy_sources: bool,
    #[serde(default)]
    pub in_place_casts: Option<bool>,
}

impl Recipe {
    pub(crate) fn normalize(&mut self, config: &PipelineConfig) {
        self.in_place_casts = Some(self.in_place_casts.unwrap_or(config.capacity_baseline));
    }

    pub(crate) fn changes(&self, before: &Self) -> impl std::fmt::Debug {
        #[derive(Debug)]
        #[allow(dead_code)] // Fields are consumed by Debug only.
        struct Changes {
            plans: BTreeSet<OperationId>,
            boundaries: Vec<ValueId>,
            casts: Vec<crate::mid::cast_order::CastSite>,
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
                .copied()
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
    pub cast_sites: BTreeSet<crate::mid::cast_order::CastSite>,
}
