//! Estimates share one vocabulary: tensor geometry and traffic describe work;
//! memory evaluates allocations/liveness, and cycles applies IPU21 prices.

use crate::storage::GeometryCache;
mod cycles;
use cycles::exchange_fragment_price;
mod exchange_storage;
use exchange_storage::ExchangeStoragePhase;
pub use exchange_storage::{ExchangeStorageEstimator, estimate_exchange_phase_storage};
mod memory;
pub(crate) mod memory_profile;
mod mid;
#[cfg(test)]
pub(crate) use mid::operation_cost;
mod primitive;
mod program;
pub(crate) use mid::{
    analyze as analyze_mid, analyze_with_budget, operation_cycles, region_program,
};
#[cfg(test)]
pub(crate) use program::program_cycles;
pub(crate) use program::{
    ProgramCycles, exchange_phase_estimate, program_cycles_analyzed, program_footprint_analyzed,
    scheduled_program_cycles,
};
mod movement;
mod tensor;
mod traffic;

use crate::graph::TensorShape;
use crate::{
    ElementOrder, Layout, MemoryClass, MidOperation, MidValue, MidValueId, Precision, TensorType,
};
use std::collections::{BTreeMap, HashMap, HashSet};

#[cfg(test)]
pub(crate) use cycles::RearrangementCost;
pub use cycles::{CostModel, ExchangeFootprint, IPU21_TARGET_COSTS, Ipu21CostModel};
pub(crate) use cycles::{
    EXCHANGE_FRAGMENT_CONTROLS, MemoizedCostModel, exchange_work_cycles, row_major_pack_cycles,
};
pub(crate) use memory::*;
pub use memory::{MemoryPeaks, MemoryUsage};
pub(crate) use tensor::*;
pub(crate) use traffic::*;

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use program::program_footprint;
