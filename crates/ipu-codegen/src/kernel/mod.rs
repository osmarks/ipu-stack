//! Each family owns its specs, call geometry, costs, and device build rules.
//! `binding` shares ABI and operand checks; `build` collects implementations
//! and supplies compilation/wrapper helpers. Placement only supplies addresses.

use crate::storage::TensorStorage;

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
use attention::AttentionKernelShape;
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
    /// group. Binding applies this same contract when interning and constructing
    /// metadata; shifted storage affects the cast contract, not the allocator.
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
        kernel: &crate::mid::MidOperationKind,
        inputs: &[TensorStorage<'_>],
        outputs: &[TensorStorage<'_>],
    ) -> Result<Self, KernelAbiError> {
        use crate::mid::MidOperationKind::*;
        match kernel {
            Gemm { .. } => gemm::call(kernel, inputs, outputs),
            Gelu | BiasGelu | Add => pointwise::call(kernel, inputs, outputs),
            LayerNorm
            | AddLayerNorm
            | LayerNormMoments
            | AddLayerNormMoments
            | LayerNormApply { .. } => normalization::call(kernel, inputs, outputs),
            FlashAttention { .. } | AttentionSoftmax { .. } | AttentionMerge { .. } => {
                attention::call(kernel, inputs, outputs)
            }
            Cast { .. } => cast::call(kernel, inputs, outputs),
            Rearrange { .. } => rearrange::call(kernel, inputs, outputs),
            ReductionSum { .. } => reduce::call(kernel, inputs, outputs),
            FillZero { .. } => copy::fill_call(kernel, inputs, outputs),
            _ => Err(KernelAbiError::RequirementMismatch),
        }
    }

    /// Price the selected implementation and its actual scalar arguments.
    pub(crate) fn cycles(&self) -> u64 {
        match &self.implementation {
            KernelImplementation::Gemm(..) => gemm::cycles(&self.implementation),
            KernelImplementation::Attention(_)
            | KernelImplementation::Softmax(..)
            | KernelImplementation::Merge(..) => attention::cycles(self),
            KernelImplementation::Rearrange(_) | KernelImplementation::Unpack(_) => {
                rearrange::cycles(self)
            }
            // Decode the exact ABI once; family helpers price typed dimensions.
            KernelImplementation::Exact(symbol) => match (*symbol, self.arguments.as_slice()) {
                ("gelu_tanh_approx_f16", &[count]) => {
                    pointwise::gelu_row_cycles(count.into(), false)
                }
                ("bias_gelu_f16", &[rows, width]) => {
                    pointwise::f16_bias_gelu_cycles(rows.into(), width.into())
                }
                ("add_f16", &[count, left, right]) => {
                    pointwise::f16_add_cycles(count.into(), left.into(), right.into())
                }
                ("gelu_f8", &[rows, width, ..]) => {
                    pointwise::fp8_gelu_cycles(rows.into(), width.into())
                }
                ("bias_gelu_f8", &[rows, width, _, packed, ..]) => {
                    pointwise::fp8_bias_gelu_cycles(rows.into(), width.into(), packed != 0)
                }
                ("layer_norm_f16" | "add_layer_norm_f16", &[rows, width]) => {
                    normalization::layernorm_cycles(
                        rows.into(),
                        width.into(),
                        *symbol == "add_layer_norm_f16",
                        true,
                    )
                }
                ("layer_norm_f8", &[rows, width, _, packed]) => {
                    normalization::fp8_layernorm_cycles(rows.into(), width.into(), packed != 0)
                }
                ("layer_norm_moments" | "add_layer_norm_moments", &[rows, width]) => {
                    normalization::f16_layernorm_moments_cycles(
                        rows.into(),
                        width.into(),
                        *symbol == "add_layer_norm_moments",
                    )
                }
                ("layer_norm_apply", &[rows, width, parts]) => {
                    normalization::f16_layernorm_apply_cycles(
                        rows.into(),
                        width.into(),
                        parts as u16,
                    )
                }
                ("cast_f16_f8", &[count, _, _, panel_rows, ..]) => {
                    cast::f16_fp8_cycles(count.into(), panel_rows.into())
                }
                ("cast_f32_f16", &[count, ..]) => cast::stream_cycles(count.into(), 4, 2),
                ("cast_f8_f8", &[count, ..]) => cast::stream_cycles(count.into(), 1, 1),
                ("cast_f8_f16", &[count, ..]) => cast::stream_cycles(count.into(), 1, 2),
                ("cast_f8_f32", &[count, ..]) => cast::stream_cycles(count.into(), 1, 4),
                ("cast_f32_f8", &[count, ..]) => cast::stream_cycles(count.into(), 4, 1),
                ("reduce_sum_f16", &[partials, count]) => {
                    reduce::f16_reduction_cycles(count.into(), u64::from(partials) + 1)
                }
                _ => copy::cycles(symbol, &self.arguments),
            },
        }
    }
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
