use super::{KernelBuildPlan, KernelMaterializationError};
use crate::{KernelRun, LowShard, LowShardId, view_byte_spans};
use ipu_target::program::{ComputeStep, StepProfile, TileAddress};
use std::collections::BTreeMap;

/// Resolves one scheduled call after placement has assigned each shard base.
/// Layout conversion supplies the byte offset; the build plan supplies the
/// linked specialization and ABI scalar values.
pub fn materialize_kernel_run(
    run: &KernelRun,
    shards: &[LowShard],
    shard_addresses: &BTreeMap<LowShardId, u32>,
    plan: &KernelBuildPlan,
    overrides: &BTreeMap<LowShardId, TileAddress>,
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
