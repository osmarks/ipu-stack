//! Kernel ABI, specialization recipes, and placed call materialization.

pub mod abi;
pub(crate) mod copy;
pub(crate) use copy::CopyRun;
mod attention;
mod binding;
mod build;
pub(crate) mod cast;
pub(crate) mod cost;
mod gemm;
mod inventory;
mod normalization;
mod output;
mod pointwise;
pub(crate) mod rearrange;
mod reduce;
mod spec;
pub use spec::*;
#[cfg(test)]
mod tests;
use attention::AttentionKernelShape;
pub(crate) use attention::softmax_workspaces;
pub(crate) use binding::*;
pub(crate) use build::*;
pub(crate) use gemm::gemm_rows;
use inventory::*;
use rearrange::{RearrangeTarget, UnpackSource};

use crate::{AMP_COLUMN_MICRO, AMP_INNER_BLOCK};

use crate::{
    AmpOrder, BlockMajorOrder, BlockOperation, BlockValue, BlockValueId, ComputeStep, ElementOrder,
    KernelRun, Precision, StepProfile, StorageError, TileAddress,
};

use std::collections::{BTreeMap, BTreeSet};

/// Resolves one scheduled call after placement has assigned each shard base.
/// Layout conversion supplies the byte offset; the build plan supplies the
/// linked specialization and ABI scalar values.
pub fn materialize_kernel_run(
    run: &KernelRun,
    shards: &[BlockValue],
    shard_addresses: &BTreeMap<BlockValueId, u32>,
    plan: &KernelBuildPlan,
    overrides: &BTreeMap<BlockValueId, TileAddress>,
) -> Result<ComputeStep, KernelError> {
    let call = run.call()?;
    let symbol = plan.symbol(&call.implementation)?.to_owned();
    let resolve = |operand: MemoryOperand| -> Result<TileAddress, KernelError> {
        let view = run
            .operand_view(operand)
            .ok_or(KernelAbiError::RequirementMismatch)?;
        let offset = view_offset(run, operand, shards)?;
        Ok(crate::low::storage::resolve_address(
            shards,
            shard_addresses,
            overrides,
            view.shard,
            offset,
        )?)
    };
    let output_address = resolve(MemoryOperand::Output(0))?;
    // The worker ABI puts result zero in R2, followed by inputs and then the
    // remaining results. This register order does not distinguish result storage.
    let input_addresses = (0..run.inputs.len())
        .map(|index| MemoryOperand::Input(index as u16))
        .chain((1..run.outputs.len()).map(|index| MemoryOperand::Output(index as u16)))
        .map(resolve)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ComputeStep {
        symbol,
        output_address,
        input_addresses,
        arguments: call.arguments,
        profile: StepProfile::default(),
    })
}
