//! Estimates share one vocabulary: tensor geometry and traffic describe work;
//! memory evaluates allocations/liveness, and cycles applies target prices.

use crate::storage::GeometryCache;
mod cycles;
#[cfg(test)]
use cycles::ExchangeFootprint;
use cycles::exchange_fragment_price;
mod exchange_storage;
#[cfg(test)]
use exchange_storage::ExchangeStorageEstimator;
use exchange_storage::ExchangeStoragePhase;
mod memory;
pub(crate) mod memory_profile;
mod mid;
#[cfg(test)]
pub(crate) use mid::operation_cost;
mod program;
pub(crate) use mid::{
    analyze as analyze_mid, analyze_with_budget, operation_cycles, region_program,
};
#[cfg(test)]
pub(crate) use program::program_cycles;
pub(crate) use program::{ProgramCycles, scheduled_program_cycles};
mod movement;
mod tensor;
mod traffic;

use crate::graph::TensorShape;
use crate::{
    ElementOrder, Layout, MemoryClass, MidOperation, MidValue, MidValueId, Precision, TensorType,
};
#[cfg(test)]
use std::collections::HashSet;
use std::collections::{BTreeMap, HashMap};

pub(crate) use memory::*;
pub use memory::{MemoryPeaks, MemoryUsage};
pub(crate) use tensor::*;
pub(crate) use traffic::*;

#[cfg(test)]
mod tests;
