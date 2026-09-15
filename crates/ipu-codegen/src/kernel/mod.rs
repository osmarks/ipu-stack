//! Kernel ABI, specialization recipes, and placed call materialization.

mod abi;
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
pub(crate) use abi::*;
use attention::AttentionKernelShape;
pub(crate) use attention::softmax_workspaces;
pub(crate) use binding::*;
pub(crate) use build::*;
pub(crate) use gemm::gemm_rows;
use inventory::*;
use rearrange::{RearrangeTarget, UnpackSource};

use crate::{AMP_COLUMN_MICRO, AMP_INNER_BLOCK};
use crate::{
    AmpOrder, BlockMajorOrder, BlockValue, BlockValueId, ComputeStep, ElementOrder, KernelRun,
    LowProgram, Precision, StepProfile, StorageError, TileAddress, TileWorkList, TileWorkRef,
    view_byte_traversal,
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
    let resolve = |operand: MemoryOperand| {
        let view = run
            .operand_view(operand)
            .ok_or(KernelAbiError::RequirementMismatch)?;
        let offset = view_offset(run, operand, shards)?;
        let base = resolve_shard_address(shards, shard_addresses, overrides, view.shard)?;
        add_address_offset(base, offset)
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

pub(crate) fn add_address_offset(
    address: TileAddress,
    offset: u32,
) -> Result<TileAddress, KernelError> {
    add_address_displacement(address, i64::from(offset))
}

pub(crate) fn resolve_shard_address(
    shards: &[BlockValue],
    addresses: &BTreeMap<BlockValueId, u32>,
    overrides: &BTreeMap<BlockValueId, TileAddress>,
    shard: BlockValueId,
) -> Result<TileAddress, KernelError> {
    if !overrides.is_empty()
        && let Some((base, displacement)) = crate::storage_chain(shards, shard)
            .find_map(|(source, offset)| overrides.get(&source).map(|&base| (base, offset)))
    {
        return add_address_displacement(base, displacement);
    }
    addresses
        .get(&shard)
        .copied()
        .map(TileAddress::Absolute)
        .ok_or(KernelError::UnplacedShard(shard.index()))
}

fn add_address_displacement(
    address: TileAddress,
    displacement: i64,
) -> Result<TileAddress, KernelError> {
    let offset = |base: i64| {
        base.checked_add(displacement)
            .ok_or(KernelError::AddressOverflow)
    };
    Ok(match address {
        TileAddress::Absolute(address) => TileAddress::Absolute(
            u32::try_from(offset(i64::from(address))?).map_err(|_| KernelError::AddressOverflow)?,
        ),
        TileAddress::RepeatPointer {
            index,
            offset: existing,
        } => TileAddress::RepeatPointer {
            index,
            offset: i32::try_from(offset(i64::from(existing))?)
                .map_err(|_| KernelError::AddressOverflow)?,
        },
    })
}
