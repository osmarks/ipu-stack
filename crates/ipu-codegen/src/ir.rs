//! Layout-aware mid-level graph records.

use crate::graph::{GraphInputKind, OperationId, ValueId};
use crate::layout::{Layout, TensorType};
use crate::metrics::{OperationMetrics, RegionMetrics};
use crate::operator::{ConversionPlan, MidOperator, OperatorPlan, Precision};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MidValueId(u32);

impl MidValueId {
    pub const fn index(self) -> u32 {
        self.0
    }

    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidValue {
    pub id: MidValueId,
    pub tensor_type: TensorType,
    /// Semantic value represented by this value; conversions retain the same
    /// origin. Region arguments also refer to their high-level argument ID.
    pub origin: ValueId,
    /// Values in the same group use the same logical-to-physical tile mapping.
    /// Structured iteration uses this to keep successive parameter blocks
    /// addressable by a single advancing base pointer.
    pub storage_group: MidValueId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MidOperationKind {
    Operator(OperatorPlan),
    CastPrecision { from: Precision, to: Precision },
    Rearrange { from: Layout, to: Layout },
    Repeat(MidRepeat),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidOperation {
    pub source: Option<OperationId>,
    pub inputs: Vec<MidValueId>,
    pub results: Vec<MidValueId>,
    pub kind: MidOperationKind,
    pub conversion_plan: Option<ConversionPlan>,
    pub metrics: OperationMetrics,
}

impl MidOperation {
    pub fn operator_plan(&self) -> Option<&OperatorPlan> {
        match &self.kind {
            MidOperationKind::Operator(plan) => Some(plan),
            MidOperationKind::CastPrecision { .. }
            | MidOperationKind::Rearrange { .. }
            | MidOperationKind::Repeat(_) => None,
        }
    }

    pub fn operator_plan_mut(&mut self) -> Option<&mut OperatorPlan> {
        match &mut self.kind {
            MidOperationKind::Operator(plan) => Some(plan),
            MidOperationKind::CastPrecision { .. }
            | MidOperationKind::Rearrange { .. }
            | MidOperationKind::Repeat(_) => None,
        }
    }

    pub fn operator(&self) -> Option<MidOperator> {
        self.operator_plan().map(|plan| plan.operator)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidRegion {
    pub arguments: Vec<MidValueId>,
    pub operations: Vec<MidOperation>,
    pub yields: Vec<MidValueId>,
    pub metrics: RegionMetrics,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidRepeat {
    pub count: u32,
    pub carried_inputs: usize,
    pub invariant_inputs: usize,
    /// One normalized value list for each iterated body argument. Keeping the
    /// lists on the structured operation avoids unrolling layer parameters.
    pub iterated_inputs: Vec<Vec<MidValueId>>,
    pub body: MidRegion,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidInput {
    pub name: String,
    pub kind: GraphInputKind,
    pub value: MidValueId,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MidGraph {
    pub inputs: Vec<MidInput>,
    pub values: Vec<MidValue>,
    pub operations: Vec<MidOperation>,
    pub outputs: Vec<MidValueId>,
    pub metrics: RegionMetrics,
}
