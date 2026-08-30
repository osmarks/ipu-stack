//! Resource estimation and target-specific planning costs.

mod exchange;
mod kernel;
mod memoization;
mod parallel_reduction;
mod resources;

pub(crate) use kernel::row_major_pack_cycles;
pub use kernel::{CostModel, Ipu21CostModel};
pub(crate) use memoization::MemoizedCostModel;
pub(crate) use parallel_reduction::parallel_reduction_preselection_metrics;
pub(crate) use resources::{
    conversion_memory_estimate, operator_memory_estimate, region_peak_memory,
    region_peak_memory_with_multiplicity,
};
#[cfg(test)]
pub(crate) use resources::{maximum_shard_bytes, physical_elements};
