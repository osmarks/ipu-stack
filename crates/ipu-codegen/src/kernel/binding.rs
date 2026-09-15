//! Address-independent kernel access contracts and call geometry.

use super::*;
use crate::low::{KernelRun, ShardView};
use crate::low::{KernelRunMetadata, WorkProvenance};
use crate::mid::MidOperationKind;
use crate::tensor::TensorFormat;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryOperand {
    Output(u16),
    Input(u16),
}

/// Access contract of an actual kernel buffer, without candidate planning policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelAccess {
    pub format: TensorFormat,
    pub storage: crate::low::storage::StorageAccess,
}

impl KernelAccess {
    pub fn new(format: TensorFormat, alignment: u32) -> Self {
        Self {
            format,
            storage: crate::low::storage::StorageAccess {
                alignment,
                access_tail_bytes: 0,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelRequirements {
    pub inputs: Vec<KernelAccess>,
    pub outputs: Vec<KernelAccess>,
    pub distinct_elements: Vec<Vec<MemoryOperand>>,
}

impl KernelRequirements {
    pub fn new(
        kernel: &MidOperationKind,
        inputs: impl IntoIterator<Item = TensorFormat>,
        outputs: impl IntoIterator<Item = TensorFormat>,
    ) -> Self {
        let alignment = match kernel {
            MidOperationKind::Gemm { .. } => 32,
            // Includes rearrangement fast paths: these use 64-bit accesses
            // even when the tensor elements are F16.
            _ => 8,
        };
        let mut requirements = Self {
            inputs: inputs
                .into_iter()
                .map(|format| KernelAccess::new(format, alignment))
                .collect(),
            outputs: outputs
                .into_iter()
                .map(|format| KernelAccess::new(format, alignment))
                .collect(),
            distinct_elements: Vec::new(),
        };
        if let MidOperationKind::Gemm { multiply, .. } = kernel
            && let Some(left) = requirements.inputs.first_mut()
        {
            left.storage.access_tail_bytes = 8 * multiply.bytes() as u32;
            requirements
                .distinct_elements
                .push(vec![MemoryOperand::Output(0), MemoryOperand::Input(0)]);
        }
        requirements
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum KernelAbiError {
    #[error("kernel requirements do not match the tile-kernel family")]
    RequirementMismatch,
    #[error("FP8 hardware scale {0} is outside -32..=31")]
    Fp8Scale(i32),
    #[error("kernel run has {actual} pointer operands, ABI requires {expected}")]
    PointerArity { expected: usize, actual: usize },
    #[error("kernel {0:?} has no device implementation")]
    Unavailable(MidOperationKind),
    #[error("GEMM output view does not have a matrix row axis")]
    MissingGemmRows,
    #[error("kernel element count overflowed")]
    ElementCountOverflow,
    #[error("kernel {symbol} requires an element count divisible by {divisor}, got {count}")]
    UnsupportedElementCount {
        symbol: &'static str,
        count: u32,
        divisor: u32,
    },
}

pub(crate) fn matrix_extent(
    view: &ShardView,
    logical: bool,
    columns: bool,
) -> Result<u32, KernelAbiError> {
    let rank = view.extents.len();
    let axis = rank
        .checked_sub(if columns { 1 } else { 2 })
        .ok_or(KernelAbiError::RequirementMismatch)?;
    let extent = &view.extents[axis];
    Ok(if logical {
        extent.logical_end - extent.start
    } else {
        extent.physical_end - extent.start
    })
}

pub(crate) fn input_matrix_extent(
    run: &KernelRun,
    logical: bool,
    columns: bool,
) -> Result<u32, KernelAbiError> {
    let view = run
        .inputs
        .first()
        .ok_or(KernelAbiError::RequirementMismatch)?;
    matrix_extent(view, logical, columns)
}

pub(crate) fn matrix_count(run: &KernelRun) -> Result<u32, KernelAbiError> {
    let view = run
        .inputs
        .first()
        .ok_or(KernelAbiError::RequirementMismatch)?;
    view.extents[..view.extents.len().saturating_sub(2)]
        .iter()
        .try_fold(1u32, |product, extent| {
            product
                .checked_mul(extent.physical_end - extent.start)
                .ok_or(KernelAbiError::ElementCountOverflow)
        })
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum KernelError {
    #[error(transparent)]
    Abi(#[from] KernelAbiError),
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error(transparent)]
    Address(#[from] crate::low::storage::AddressError),
    #[error("kernel operand view of shard {shard} has {spans} physical byte spans")]
    FragmentedView { shard: u32, spans: usize },
}

impl KernelRun {
    /// Bind a complete local invocation. The low builder supplies resolved views;
    /// the kernel owner establishes all address-independent call requirements.
    /// Metadata interning shares formats and access contracts across tile calls.
    pub(crate) fn bind(
        provenance: WorkProvenance,
        kernel: MidOperationKind,
        inputs: Vec<ShardView>,
        outputs: Vec<ShardView>,
        shards: &[BlockValue],
        metadata: &mut Vec<Arc<KernelRunMetadata>>,
    ) -> Result<Self, KernelError> {
        let output = outputs.first().ok_or(KernelAbiError::RequirementMismatch)?;
        let tile = output.bind(shards)?.shard.tile;
        for view in inputs.iter().chain(&outputs) {
            if view.bind(shards)?.shard.tile != tile {
                return Err(KernelAbiError::RequirementMismatch.into());
            }
        }
        let format = |view: &ShardView| &shards[view.shard.index() as usize].tensor_type.format;
        let shared = metadata.iter().find(|metadata| {
            metadata.provenance == provenance
                && metadata.kernel == kernel
                && metadata.requirements.inputs.len() == inputs.len()
                && metadata.requirements.outputs.len() == outputs.len()
                && inputs
                    .iter()
                    .zip(&metadata.requirements.inputs)
                    .chain(outputs.iter().zip(&metadata.requirements.outputs))
                    .all(|(view, requirement)| *format(view) == requirement.format)
        });
        let shared = if let Some(shared) = shared {
            Arc::clone(shared)
        } else {
            let shared = Arc::new(KernelRunMetadata {
                provenance,
                requirements: KernelRequirements::new(
                    &kernel,
                    inputs.iter().map(|view| format(view).clone()),
                    outputs.iter().map(|view| format(view).clone()),
                ),
                kernel,
            });
            metadata.push(Arc::clone(&shared));
            shared
        };
        let mut run = Self {
            product_flops: None,
            metadata: shared,
            inputs,
            outputs,
        };
        run.call()?;
        if let MidOperationKind::Gemm {
            axes, inner_block, ..
        } = run.kernel
        {
            let axis = |a: crate::TensorAxis, rank| {
                a.resolve(rank)
                    .map_err(|_| KernelAbiError::RequirementMismatch)
            };
            let li = axis(axes.left_inner, run.inputs[0].extents.len())?;
            let ri = axis(axes.right_inner, run.inputs[1].extents.len())?;
            let oc = axis(axes.output_column, run.outputs[0].extents.len())?;
            let rc = if ri + 1 == run.inputs[1].extents.len() {
                ri - 1
            } else {
                ri + 1
            };
            let size = |e: crate::ShardExtent, bound: Option<u32>| {
                u64::from(
                    e.logical_end
                        .min(bound.unwrap_or(u32::MAX))
                        .saturating_sub(e.start),
                )
            };
            let rows: u64 = run.outputs[0]
                .extents
                .iter()
                .enumerate()
                .filter(|(a, _)| *a != oc)
                .map(|(_, &e)| size(e, None))
                .product();
            let cols = size(run.outputs[0].extents[oc], axes.valid_columns)
                .min(size(run.inputs[1].extents[rc], axes.valid_columns));
            let inner = size(run.inputs[0].extents[li], axes.valid_inner)
                .min(size(run.inputs[1].extents[ri], axes.valid_inner));
            let physical: u64 = run.outputs[0]
                .extents
                .iter()
                .map(|e| u64::from(e.physical_end - e.start))
                .product();
            run.product_flops = Some([
                2 * rows * cols * inner,
                2 * physical * u64::from(inner_block),
            ]);
        }
        for operand in (0..run.inputs.len())
            .map(|i| MemoryOperand::Input(i as u16))
            .chain((0..run.outputs.len()).map(|i| MemoryOperand::Output(i as u16)))
        {
            view_offset(&run, operand, shards)?;
        }
        Ok(run)
    }
}

/// Physical addressing within a bound operand. Placement adds the allocation
/// base later; it does not reinterpret shapes, strides or packed GEMM stores.
pub(super) fn view_offset(
    run: &KernelRun,
    operand: MemoryOperand,
    shards: &[BlockValue],
) -> Result<u32, KernelError> {
    let view = run
        .operand_view(operand)
        .ok_or(KernelAbiError::RequirementMismatch)?;
    let bound = view.bind(shards)?;
    let spans = bound.traversal(crate::CopyOrder::Physical)?;
    let packed = operand == MemoryOperand::Output(0) && gemm::packed_output(run, bound.shard)?;
    let span = if packed {
        spans.spans().next()
    } else {
        spans.contiguous_span()
    }
    .ok_or_else(|| KernelError::FragmentedView {
        shard: view.shard.index(),
        spans: spans.span_count() as usize,
    })?;
    if operand == MemoryOperand::Output(0)
        && let MidOperationKind::FillZero { offset, bytes, .. } = run.kernel
    {
        if !offset.is_multiple_of(8) || offset.checked_add(bytes).is_none_or(|end| end > span.bytes)
        {
            return Err(StorageError::InvalidView.into());
        }
        return span
            .offset
            .checked_add(offset)
            .ok_or_else(|| crate::low::storage::AddressError::Overflow.into());
    }
    Ok(span.offset)
}

pub(super) fn element_count(extents: &[crate::ShardExtent]) -> Result<u32, KernelAbiError> {
    extents.iter().try_fold(1u32, |product, extent| {
        product
            .checked_mul(extent.physical_end - extent.start)
            .ok_or(KernelAbiError::ElementCountOverflow)
    })
}

pub(super) fn output_byte_count(run: &KernelRun) -> Result<u32, KernelAbiError> {
    if let MidOperationKind::FillZero { bytes, .. } = run.kernel {
        return Ok(bytes);
    }
    let precision = run.requirements.outputs[0].format.precision;
    element_count(&run.outputs[0].extents)?
        .checked_mul(
            u32::try_from(precision.bytes()).map_err(|_| KernelAbiError::ElementCountOverflow)?,
        )
        .ok_or(KernelAbiError::ElementCountOverflow)
}

pub(super) fn fp8_scale_argument(scale: i32) -> Result<u32, KernelAbiError> {
    if (-32..=31).contains(&scale) {
        Ok(u32::from_ne_bytes(scale.to_ne_bytes()))
    } else {
        Err(KernelAbiError::Fp8Scale(scale))
    }
}

/// The same key selects a build recipe and resolves its eventual call.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum KernelImplementation {
    Exact(&'static str),
    Gemm(
        Precision,
        GemmWeightLoad,
        u32,
        u32,
        GemmKernelMode,
        u32,
        u32,
    ),
    Attention(AttentionKernelShape),
    Softmax(u32, u32, u32, Precision),
    Merge(u32, u32, Precision),
    Rearrange((RearrangeTarget, u32, u32, u32, u32)),
    Unpack((UnpackSource, u32, u32, u32, u32)),
}

/// Derived facts for one bound invocation. The implementation key is complete
/// before build collection; only its linked symbol and operand addresses remain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct KernelCall {
    pub(super) implementation: KernelImplementation,
    pub arguments: Vec<u32>,
}

impl KernelCall {
    pub(super) fn exact(symbol: &'static str, arguments: Vec<u32>) -> Self {
        Self {
            implementation: KernelImplementation::Exact(symbol),
            arguments,
        }
    }
}

impl KernelRun {
    pub(crate) fn call(&self) -> Result<KernelCall, KernelAbiError> {
        if self.inputs.len() != self.requirements.inputs.len()
            || self.outputs.len() != self.requirements.outputs.len()
            || self
                .requirements
                .distinct_elements
                .iter()
                .flatten()
                .any(|&operand| self.operand_view(operand).is_none())
        {
            return Err(KernelAbiError::RequirementMismatch);
        }
        match self.kernel {
            MidOperationKind::Gemm { .. } => gemm::call(self),
            MidOperationKind::FlashAttention { .. }
            | MidOperationKind::AttentionSoftmax { .. }
            | MidOperationKind::AttentionMerge { .. } => attention::call(self),
            MidOperationKind::Cast { .. } => cast::call(self),
            MidOperationKind::Rearrange { .. } => rearrange::call(self),
            MidOperationKind::Gelu | MidOperationKind::BiasGelu | MidOperationKind::Add => {
                pointwise::call(self)
            }
            MidOperationKind::LayerNorm
            | MidOperationKind::AddLayerNorm
            | MidOperationKind::LayerNormMoments
            | MidOperationKind::AddLayerNormMoments
            | MidOperationKind::LayerNormApply { .. } => normalization::call(self),
            MidOperationKind::ReductionSum { .. } => reduce::call(self),
            MidOperationKind::FillZero { .. } => fill_call(self),
            _ => Err(KernelAbiError::RequirementMismatch),
        }
    }

    pub(super) fn check_arity(&self, inputs: usize, outputs: usize) -> Result<(), KernelAbiError> {
        if self.inputs.len() != inputs {
            return Err(KernelAbiError::PointerArity {
                expected: inputs,
                actual: self.inputs.len(),
            });
        }
        if self.outputs.len() != outputs {
            return Err(KernelAbiError::RequirementMismatch);
        }
        Ok(())
    }
}

fn fill_call(run: &KernelRun) -> Result<KernelCall, KernelAbiError> {
    run.check_arity(0, 1)?;
    let bytes = output_byte_count(run)?;
    if !bytes.is_multiple_of(8) {
        return Err(KernelAbiError::UnsupportedElementCount {
            symbol: crate::kernel::abi::FILL_ZERO_U64_SYMBOL,
            count: bytes,
            divisor: 8,
        });
    }
    Ok(KernelCall::exact(
        crate::kernel::abi::FILL_ZERO_U64_SYMBOL,
        vec![bytes / 8 / 6, bytes / 8 % 6],
    ))
}
