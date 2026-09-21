//! Each family constructs its ABI, cost and padding contract together.
//! `binding` shares operand checks; `build` collects object definitions and
//! supplies compilation/wrapper helpers. Placement only supplies addresses.
//! Family implementations currently emit IPU21 code; `KernelCall::select`
//! dispatches the caller's target before entering them.

use crate::storage::TensorStorage;
use ipu_target::Target;

pub mod abi;
pub mod copy;
pub(crate) use copy::CopyRun;
pub(crate) mod attention;
mod binding;
mod build;
pub(crate) mod cast;
pub(crate) mod gemm;
pub(crate) mod normalization;
pub(crate) mod pointwise;
pub(crate) mod rearrange;
pub(crate) mod reduce;
pub use gemm::{AccumulationPrecision, GemmAxes, GemmKernelMode, GemmWeightLoad};
#[cfg(test)]
mod tests;
pub(crate) use attention::softmax_workspaces;
pub(crate) use binding::*;
pub(crate) use build::*;
pub(crate) use gemm::gemm_rows;

use crate::{AMP_COLUMN_MICRO, AMP_INNER_BLOCK};

use crate::{
    AmpOrder, BlockMajorOrder, BlockOperation, BlockValue, BlockValueId, ComputeStep, ElementOrder,
    KernelRun, Precision, StepProfile, StorageError, TileAddress,
};

use std::collections::{BTreeMap, BTreeSet};

impl crate::mid::MidOperationKind {
    /// Address requirements for one operand and the family’s distinct-element
    /// group. Storage consumers evaluate the contract against the current output;
    /// shifted storage affects the cast contract, not the allocator.
    pub(super) fn access(
        &self,
        operand: MemoryOperand,
        output: &BlockValue,
    ) -> (crate::low::storage::StorageAccess, &'static [MemoryOperand]) {
        match *self {
            Self::Gemm { multiply, .. } => gemm::access(multiply, operand),
            Self::Cast { from, to } => cast::access(from, to, output),
            _ => (
                crate::low::storage::StorageAccess {
                    alignment: 8,
                    access_tail_bytes: 0,
                },
                &[],
            ),
        }
    }
}

impl KernelCall {
    /// The only operation-to-kernel dispatch. Both estimation and bound calls
    /// select a concrete ABI from local geometry, before addresses exist.
    pub(crate) fn select(
        target: Target,
        kernel: &crate::mid::MidOperationKind,
        inputs: &[TensorStorage<'_>],
        outputs: &[TensorStorage<'_>],
        mut build: Option<&mut KernelObjects>,
    ) -> Result<Self, KernelError> {
        // The family implementations below emit IPU21 worker code. Exhaustive
        // dispatch prevents another architecture from falling through to it.
        let Target::Ipu21 = target;
        use crate::mid::MidOperationKind::*;
        let call = match kernel {
            Gemm { .. } => gemm::call(kernel, inputs, outputs, build.as_deref_mut()),
            Gelu | BiasGelu | Add => pointwise::call(kernel, inputs, outputs, build.as_deref_mut()),
            LayerNorm
            | AddLayerNorm
            | LayerNormMoments
            | AddLayerNormMoments
            | LayerNormApply { .. } => {
                normalization::call(kernel, inputs, outputs, build.as_deref_mut())
            }
            FlashAttention { .. } | AttentionSoftmax { .. } | AttentionMerge { .. } => {
                attention::call(kernel, inputs, outputs, build.as_deref_mut())
            }
            Cast { .. } => cast::call(kernel, inputs, outputs, build.as_deref_mut()),
            Rearrange { .. } => rearrange::call(kernel, inputs, outputs, build.as_deref_mut()),
            ReductionSum { .. } => reduce::call(kernel, inputs, outputs, build.as_deref_mut()),
            FillZero { .. } => copy::fill_call(kernel, inputs, outputs),
            _ => Err(KernelError::RequirementMismatch),
        }?;
        if let Some(build) = build {
            build.symbols.insert(call.symbol.clone());
        }
        Ok(call)
    }
}

/// Resolves one scheduled call after placement has assigned each shard base.
/// Layout conversion supplies byte offsets; family construction supplies the
/// entry-point name and ABI scalar values independently of object collection.
pub fn materialize_kernel_run(
    target: Target,
    run: &KernelRun,
    shards: &[BlockValue],
    shard_addresses: &BTreeMap<BlockValueId, u32>,
    overrides: &BTreeMap<BlockValueId, TileAddress>,
) -> Result<ComputeStep, KernelError> {
    let call = run.call(target, None)?;
    let symbol = call.symbol;
    let resolve = |operand: MemoryOperand| -> Result<TileAddress, KernelError> {
        let view = run
            .operand_view(operand)
            .ok_or(KernelError::RequirementMismatch)?;
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
