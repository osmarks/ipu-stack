//! Kernel ABI, specialization recipes, and placed call materialization.

mod abi;
mod attention;
mod build;
pub(crate) mod cost;
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
    let packed_group = run
        .requirements
        .output
        .format
        .layout
        .order
        .gemm_output_group()
        .filter(|_| {
            matches!(
                run.kernel,
                TileKernelSpec::Gemm {
                    multiply: Precision::F16,
                    ..
                }
            )
        });
    if let Some(group) = packed_group {
        let shard = &shards[run.output.shard.index() as usize];
        let column = run.output.extents.len()
            - if run
                .requirements
                .output
                .format
                .layout
                .order
                .gemm_output_transposed()
            {
                2
            } else {
                1
            };
        let row = if column + 1 == run.output.extents.len() {
            column - 1
        } else {
            column + 1
        };
        let extent = run.output.extents[column];
        let start = extent.start - shard.extents[column].start;
        let end = extent.physical_end - shard.extents[column].start;
        if run.output.extents[row] != shard.extents[row]
            || !gemm_rows(run)?.is_multiple_of(16)
            || !start.is_multiple_of(16)
            || end <= start
            || start / group != (end - 1) / group
        {
            return Err(KernelAbiError::RequirementMismatch.into());
        }
    }
    let resolve = |view: &crate::ShardView, packed: bool| {
        let shard = shards.get(view.shard.index() as usize).ok_or(
            KernelMaterializationError::UnplacedShard(view.shard.index()),
        )?;
        let spans = view_byte_spans(shard, view)?;
        let span = if spans.len() == 1 || packed {
            spans.first()
        } else {
            None
        };
        let Some(span) = span else {
            return Err(KernelMaterializationError::FragmentedView {
                shard: view.shard.index(),
                spans: spans.len(),
            });
        };
        let base = overrides
            .get(&view.shard)
            .copied()
            .or_else(|| {
                shard_addresses
                    .get(&view.shard)
                    .copied()
                    .map(TileAddress::Absolute)
            })
            .ok_or(KernelMaterializationError::UnplacedShard(
                view.shard.index(),
            ))?;
        add_address_offset(base, span.offset)
    };
    let mut output_address = resolve(&run.output, packed_group.is_some())?;
    if let TileKernelSpec::FillZero { offset, bytes, .. } = run.kernel {
        let output_spans =
            view_byte_spans(&shards[run.output.shard.index() as usize], &run.output)?;
        let allocation_bytes = output_spans[0].bytes;
        if !offset.is_multiple_of(8)
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > allocation_bytes)
        {
            return Err(StorageError::InvalidView.into());
        }
        output_address = add_address_offset(output_address, offset)?;
    }
    let input_addresses = run
        .inputs
        .iter()
        .map(|operand| resolve(&operand.views[0], false))
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
