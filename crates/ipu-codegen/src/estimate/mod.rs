//! Estimates share one vocabulary: tensor geometry and traffic describe work;
//! memory evaluates allocations/liveness, and cycles applies IPU21 prices.

mod cycles;
use cycles::exchange_fragment_price;
mod exchange_storage;
use exchange_storage::ExchangeStoragePhase;
pub use exchange_storage::{ExchangeStorageEstimator, estimate_exchange_phase_storage};
mod memory;
mod mid;
mod primitive;
mod program;
pub(crate) use mid::{
    analyze as analyze_mid, operation_cost, region_estimate, region_peak_memory,
    region_peak_memory_with_multiplicity, unavailable_memory,
};
pub(crate) use program::{
    ProgramCycles, program_cycles, program_footprint, scheduled_program_cycles,
};
mod movement;
mod tensor;
mod traffic;

use crate::graph::TensorShape;
use crate::{
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
