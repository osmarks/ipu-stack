//! Parametric whole-device work selected by mid-level planning.

use crate::{
    AttentionPlan, BlockedGemmPlan, MidOperator, OperatorRequirements, ReductionStaging,
    TileKernelSpec,
};

/// A value consumed or produced by an operator schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScheduleValue {
    Input(u16),
    Output,
}

/// How a mapped kernel obtains one input for each output shard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleAccess {
    /// Select every logical intersection, including singleton broadcasting.
    LogicalOverlap,
    /// Consume the corresponding shard already resident on the kernel tile.
    TileLocal,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelMap {
    pub kernel: TileKernelSpec,
    pub inputs: Vec<(ScheduleValue, ScheduleAccess)>,
    pub output: ScheduleValue,
}

/// Ordered, symbolic whole-device work. Placement binds values to concrete
/// shards and tiles; low lowering only materializes the selected steps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScheduleStep {
    KernelMap(KernelMap),
    BlockedGemm(BlockedGemmPlan),
    Reduce { staging: ReductionStaging },
    Attention(AttentionPlan),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatorSchedule {
    pub operator: MidOperator,
    pub steps: Vec<ScheduleStep>,
    pub requirements: OperatorRequirements,
}

impl OperatorSchedule {
    pub(crate) fn blocked_gemm(
        operator: MidOperator,
        plan: BlockedGemmPlan,
        requirements: OperatorRequirements,
    ) -> Self {
        let mut steps = vec![ScheduleStep::BlockedGemm(plan)];
        if plan.geometry.compute.inner > 1 {
            steps.push(ScheduleStep::Reduce {
                staging: ReductionStaging::Complete,
            });
        }
        Self {
            operator,
            steps,
            requirements,
        }
    }

    pub(crate) fn attention(
        operator: MidOperator,
        plan: AttentionPlan,
        requirements: OperatorRequirements,
    ) -> Self {
        Self {
            operator,
            steps: vec![ScheduleStep::Attention(plan)],
            requirements,
        }
    }

    pub(crate) fn gemm_plan(&self) -> Option<&BlockedGemmPlan> {
        match self.steps.first() {
            Some(ScheduleStep::BlockedGemm(plan)) => Some(plan),
            _ => None,
        }
    }

    pub(crate) fn gemm_plan_mut(&mut self) -> Option<&mut BlockedGemmPlan> {
        match self.steps.first_mut() {
            Some(ScheduleStep::BlockedGemm(plan)) => Some(plan),
            _ => None,
        }
    }

    pub(crate) fn attention_plan(&self) -> Option<&AttentionPlan> {
        match self.steps.as_slice() {
            [ScheduleStep::Attention(plan)] => Some(plan),
            _ => None,
        }
    }

    pub(crate) fn reduction_staging(&self) -> Option<ReductionStaging> {
        self.steps.iter().find_map(|step| match step {
            ScheduleStep::Reduce { staging } => Some(*staging),
            _ => None,
        })
    }

    pub(crate) fn set_reduction_staging(&mut self, staging: ReductionStaging) -> bool {
        if self
            .gemm_plan()
            .is_none_or(|plan| plan.geometry.compute.inner < 2)
        {
            return false;
        }
        if let Some(ScheduleStep::Reduce { staging: scheduled }) = self.steps.get_mut(1) {
            *scheduled = staging;
        } else if self.steps.len() == 1 {
            self.steps.push(ScheduleStep::Reduce { staging });
        } else {
            return false;
        }
        true
    }
}
