//! Estimates share one vocabulary: tensor geometry and traffic describe work;
//! memory evaluates allocations/liveness, and cycles applies IPU21 prices.

mod cycles;
mod memory;
mod tensor;
mod traffic;

use crate::graph::TensorShape;
use crate::mid::{
    AMP_COLUMN_MICRO, AMP_INNER_BLOCK, ElementOrder, GemmDistribution, Layout, MemoryClass,
    MemoryOperand, MidOperation, MidOperationKind, MidValue, MidValueId, OperandMaterialization,
    OperatorDispatch, Precision, StorageRequirements, TensorAxis, TensorType,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

pub use cycles::{
    CostModel, ExchangeFootprint, IPU21_TARGET_COSTS, Ipu21CostModel, RearrangementCost,
};
pub(crate) use cycles::{IPU21_LOGICAL_FRAGMENT_CYCLES, MemoizedCostModel, row_major_pack_cycles};
pub(crate) use memory::*;
pub use memory::{MemoryEstimate, MemoryPeaks, MemoryUsage};
pub(crate) use tensor::*;
pub(crate) use traffic::*;

#[cfg(test)]
mod tests;
