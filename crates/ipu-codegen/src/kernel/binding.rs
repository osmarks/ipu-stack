//! Address-independent kernel access contracts and call geometry.

use super::*;
use crate::low::{KernelRun, ShardView};
use crate::low::{KernelRunMetadata, WorkProvenance};
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
    pub alignment: u32,
    pub access_tail_bytes: u32,
}

impl KernelAccess {
    pub fn new(format: TensorFormat, alignment: u32) -> Self {
        Self {
            format,
            alignment,
            access_tail_bytes: 0,
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
        kernel: &TileKernelSpec,
        inputs: impl IntoIterator<Item = TensorFormat>,
        outputs: impl IntoIterator<Item = TensorFormat>,
    ) -> Self {
        let alignment = match kernel {
            TileKernelSpec::Gemm { .. } => 32,
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
        if let TileKernelSpec::Gemm { multiply, .. } = kernel
            && let Some(left) = requirements.inputs.first_mut()
        {
            left.access_tail_bytes = 8 * multiply.bytes() as u32;
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
    Unavailable(TileKernelSpec),
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
    #[error("kernel references missing shard {0}")]
    UnknownShard(u32),
    #[error("shard {0} has no assigned address")]
    UnplacedShard(u32),
    #[error("kernel operand view of shard {shard} has {spans} physical byte spans")]
    FragmentedView { shard: u32, spans: usize },
    #[error("placed kernel address overflowed")]
    AddressOverflow,
}

impl KernelRun {
    /// Bind a complete local invocation. The low builder supplies resolved views;
    /// the kernel owner establishes all address-independent call requirements.
    /// Metadata interning shares formats and access contracts across tile calls.
    pub(crate) fn bind(
        provenance: WorkProvenance,
        kernel: TileKernelSpec,
        inputs: Vec<ShardView>,
        outputs: Vec<ShardView>,
        shards: &[BlockValue],
        metadata: &mut Vec<Arc<KernelRunMetadata>>,
    ) -> Result<Self, KernelError> {
        let output = outputs.first().ok_or(KernelAbiError::RequirementMismatch)?;
        let tile = shards
            .get(output.shard.index() as usize)
            .ok_or(KernelError::UnknownShard(output.shard.index()))?
            .tile;
        for view in inputs.iter().chain(&outputs) {
            let shard = shards
                .get(view.shard.index() as usize)
                .ok_or(KernelError::UnknownShard(view.shard.index()))?;
            if view.shard != shard.id || shard.tile != tile {
                return Err(KernelAbiError::RequirementMismatch.into());
            }
            crate::storage::validate_view(shard.storage(), &view.extents)?;
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
        let run = Self {
            product_flops: None,
            metadata: shared,
            inputs,
            outputs,
        };
        let abi = validate_kernel_run(&run)?;
        scalar_values(&run, &abi)?;
        if abi.symbols == KernelSymbols::Specialized {
            KernelSpecialization::from_run(&run)?;
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
    let shard = shards
        .get(view.shard.index() as usize)
        .ok_or(KernelError::UnknownShard(view.shard.index()))?;
    let spans = view_byte_traversal(shard, view, crate::CopyOrder::Physical)?;
    let packed = operand == MemoryOperand::Output(0) && gemm::packed_output(run, shard)?;
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
        && let TileKernelSpec::FillZero { offset, bytes, .. } = run.kernel
    {
        if !offset.is_multiple_of(8) || offset.checked_add(bytes).is_none_or(|end| end > span.bytes)
        {
            return Err(StorageError::InvalidView.into());
        }
        return span
            .offset
            .checked_add(offset)
            .ok_or(KernelError::AddressOverflow);
    }
    Ok(span.offset)
}
