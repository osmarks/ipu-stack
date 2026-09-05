//! Estimates share one vocabulary: tensor geometry and traffic describe work;
//! memory evaluates allocations/liveness, and cycles applies IPU21 prices.

mod cycles;
mod implementation;
pub(crate) use implementation::implementation_estimate;
mod memory;
mod program;
pub(crate) use program::{ProgramCycles, program_cycles};
mod tensor;
mod traffic;

use crate::graph::TensorShape;
use crate::mid::{
    ElementOrder, Layout, MemoryClass, MidOperation, MidValue, MidValueId, Precision, TensorType,
};
use std::collections::{BTreeMap, HashMap, HashSet};

pub use cycles::{
    CostModel, ExchangeFootprint, IPU21_TARGET_COSTS, Ipu21CostModel, RearrangementCost,
};
pub(crate) use cycles::{IPU21_LOGICAL_FRAGMENT_CYCLES, MemoizedCostModel, row_major_pack_cycles};
pub(crate) use memory::*;
pub use memory::{MemoryPeaks, MemoryUsage};
pub(crate) use tensor::*;
pub(crate) use traffic::*;

#[cfg(test)]
mod tests;
