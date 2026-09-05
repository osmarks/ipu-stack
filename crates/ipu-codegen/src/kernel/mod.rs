//! Kernel ABI, specialization recipes, and placed call materialization.

mod abi;
mod attention;
mod build;
mod gemm;
mod rearrange;
mod specialization;
#[cfg(test)]
mod tests;
pub(crate) use abi::*;
pub(crate) use build::*;
use specialization::*;

use crate::{
    AMP_COLUMN_MICRO, AMP_INNER_BLOCK, AttentionKernelShape, KernelAbiError, attention_shape,
    gemm_rows, input_matrix_extent, matrix_count, matrix_extent,
};
use crate::{
    AmpOrder, BlockMajorOrder, BlockValue, BlockValueId, ComputeStep, ElementOrder, GemmKernelMode,
    GemmWeightLoad, KernelRequirements, KernelRun, LowProgram, Precision, StepProfile,
    StorageError, TileAddress, TileKernelSpec, TileWorkList, TileWorkRef, view_byte_spans,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedKernelCall {
    pub symbol: String,
    pub arguments: Vec<u32>,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum KernelMaterializationError {
    #[error(transparent)]
    Abi(#[from] KernelAbiError),
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error("shard {0} has no assigned address")]
    UnplacedShard(u32),
    #[error("kernel operand view of shard {shard} has {spans} physical byte spans")]
    FragmentedView { shard: u32, spans: usize },
    #[error("placed kernel address overflowed")]
    AddressOverflow,
}

/// Resolves one scheduled call after placement has assigned each shard base.
/// Layout conversion supplies the byte offset; the build plan supplies the
/// linked specialization and ABI scalar values.
pub fn materialize_kernel_run(
    run: &KernelRun,
    shards: &[BlockValue],
    shard_addresses: &BTreeMap<BlockValueId, u32>,
    plan: &KernelBuildPlan,
    overrides: &BTreeMap<BlockValueId, TileAddress>,
) -> Result<ComputeStep, KernelMaterializationError> {
    let call = plan.call(run)?;
    let resolve = |view: &crate::ShardView| {
        let shard = shards.get(view.shard.index() as usize).ok_or(
            KernelMaterializationError::UnplacedShard(view.shard.index()),
        )?;
        let spans = view_byte_spans(shard, view)?;
        let [span] = spans.as_slice() else {
            return Err(KernelMaterializationError::FragmentedView {
                shard: view.shard.index(),
                spans: spans.len(),
            });
        };
        let base = overrides.get(&view.shard).copied().unwrap_or_else(|| {
            TileAddress::Absolute(
                shard_addresses
                    .get(&view.shard)
                    .copied()
                    .unwrap_or_default(),
            )
        });
        if !overrides.contains_key(&view.shard) && !shard_addresses.contains_key(&view.shard) {
            return Err(KernelMaterializationError::UnplacedShard(
                view.shard.index(),
            ));
        }
        add_address_offset(base, span.offset)
    };
    let output_address = resolve(&run.output)?;
    let input_addresses = run
        .inputs
        .iter()
        .map(|operand| resolve(&operand.views[0]))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ComputeStep {
        symbol: call.symbol,
        output_address,
        input_addresses,
        arguments: call.arguments,
        profile: StepProfile::default(),
    })
}

fn add_address_offset(
    address: TileAddress,
    offset: u32,
) -> Result<TileAddress, KernelMaterializationError> {
    Ok(match address {
        TileAddress::Absolute(address) => TileAddress::Absolute(
            address
                .checked_add(offset)
                .ok_or(KernelMaterializationError::AddressOverflow)?,
        ),
        TileAddress::RepeatPointer {
            index,
            offset: existing,
        } => TileAddress::RepeatPointer {
            index,
            offset: existing
                .checked_add(offset)
                .ok_or(KernelMaterializationError::AddressOverflow)?,
        },
    })
}
