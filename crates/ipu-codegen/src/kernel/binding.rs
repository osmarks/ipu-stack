//! Address-independent kernel access contracts and call geometry.

use super::*;
use crate::low::{KernelRun, ShardView};
use crate::low::{KernelRunMetadata, WorkProvenance};
use crate::mid::MidOperationKind;
use crate::tensor::TensorFormat;
use std::sync::Arc;

// Kernel-specific arithmetic on address-free local storage geometry.
impl TensorStorage<'_> {
    pub(super) fn logical_elements(self) -> Result<u32, KernelError> {
        self.extents.iter().try_fold(1u32, |count, extent| {
            count
                .checked_mul(extent.logical_end - extent.start)
                .ok_or(KernelError::ElementCountOverflow)
        })
    }
    pub(super) fn count(self) -> Result<u32, KernelError> {
        u32::try_from(self.elements()).map_err(|_| KernelError::ElementCountOverflow)
    }
    pub(super) fn matrix_extent(self, logical: bool, columns: bool) -> Result<u32, KernelError> {
        let axis = self
            .extents
            .len()
            .checked_sub(if columns { 1 } else { 2 })
            .ok_or(KernelError::RequirementMismatch)?;
        Ok(if logical {
            self.logical_dimension(axis)
        } else {
            self.dimension(axis)
        })
    }
    pub(crate) fn dimension(self, axis: usize) -> u32 {
        self.extents[axis].physical_end - self.extents[axis].start
    }
    pub(crate) fn trailing_dimension(self, offset: usize) -> Option<u32> {
        self.extents
            .len()
            .checked_sub(offset + 1)
            .map(|axis| self.dimension(axis))
    }
    pub(crate) fn widths(self) -> impl DoubleEndedIterator<Item = u32> + ExactSizeIterator {
        self.extents.iter().map(|e| e.physical_end - e.start)
    }
    pub(crate) fn elements(self) -> u64 {
        self.widths()
            .fold(1u64, |n, width| n.saturating_mul(u64::from(width)))
    }
    pub(super) fn logical_dimension(self, axis: usize) -> u32 {
        self.extents[axis].logical_end - self.extents[axis].start
    }
    pub(super) fn rows(self) -> u64 {
        self.widths()
            .take(self.extents.len().saturating_sub(1))
            .fold(1u64, |n, width| n.saturating_mul(u64::from(width)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryOperand {
    Output(u16),
    Input(u16),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelRequirements {
    pub inputs: Vec<TensorFormat>,
    pub outputs: Vec<TensorFormat>,
    pub distinct_elements: Vec<Vec<MemoryOperand>>,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum KernelError {
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

    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error(transparent)]
    Address(#[from] crate::low::storage::AddressError),
    #[error("kernel operand view of shard {shard} has {spans} physical byte spans")]
    FragmentedView { shard: u32, spans: usize },
}

impl KernelRun {
    pub(crate) fn accesses<'a>(
        &'a self,
        shards: &'a [BlockValue],
    ) -> impl Iterator<Item = (crate::BlockValueId, crate::low::storage::StorageAccess)> + 'a {
        let output = &shards[self.outputs[0].shard.index() as usize];
        self.inputs
            .iter()
            .enumerate()
            .map(|(i, view)| (MemoryOperand::Input(i as u16), view))
            .chain(
                self.outputs
                    .iter()
                    .enumerate()
                    .map(|(i, view)| (MemoryOperand::Output(i as u16), view)),
            )
            .map(move |(operand, view)| (view.shard, self.kernel.access(operand, output).0))
    }

    pub(crate) fn cycles(&self) -> u64 {
        self.call(None).map_or(u64::MAX, |call| call.cycles)
    }

    pub(super) fn geometry(&self, operand: MemoryOperand) -> TensorStorage<'_> {
        let (view, access) = match operand {
            MemoryOperand::Input(index) => (
                &self.inputs[usize::from(index)],
                &self.requirements.inputs[usize::from(index)],
            ),
            MemoryOperand::Output(index) => (
                &self.outputs[usize::from(index)],
                &self.requirements.outputs[usize::from(index)],
            ),
        };
        TensorStorage {
            format: access,
            extents: &view.extents,
        }
    }

    /// Bind a complete local invocation. The low builder supplies resolved views;
    /// the kernel owner establishes all address-independent call requirements.
    /// Metadata interning shares operand formats and element constraints across tile calls.
    pub(crate) fn bind(
        provenance: WorkProvenance,
        kernel: MidOperationKind,
        inputs: Vec<ShardView>,
        outputs: Vec<ShardView>,
        shards: &[BlockValue],
        metadata: &mut Vec<Arc<KernelRunMetadata>>,
    ) -> Result<Self, KernelError> {
        let output = outputs.first().ok_or(KernelError::RequirementMismatch)?;
        let tile = output.bind(shards)?.shard.tile;
        for view in inputs.iter().chain(&outputs) {
            if view.bind(shards)?.shard.tile != tile {
                return Err(KernelError::RequirementMismatch);
            }
        }
        let format = |view: &ShardView| &shards[view.shard.index() as usize].tensor_type.format;
        let output_shard = &shards[output.shard.index() as usize];
        let shared = metadata.iter().find(|metadata| {
            metadata.provenance == provenance
                && metadata.kernel == kernel
                && metadata
                    .requirements
                    .inputs
                    .iter()
                    .eq(inputs.iter().map(format))
                && metadata
                    .requirements
                    .outputs
                    .iter()
                    .eq(outputs.iter().map(format))
        });
        let shared = if let Some(shared) = shared {
            Arc::clone(shared)
        } else {
            let distinct = kernel.access(MemoryOperand::Output(0), output_shard).1;
            let requirements = KernelRequirements {
                inputs: inputs.iter().map(|view| format(view).clone()).collect(),
                outputs: outputs.iter().map(|view| format(view).clone()).collect(),
                distinct_elements: if distinct.is_empty() {
                    Vec::new()
                } else {
                    vec![distinct.to_vec()]
                },
            };
            let shared = Arc::new(KernelRunMetadata {
                provenance,
                requirements,
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
        run.call(None)?;
        if matches!(run.kernel, MidOperationKind::Gemm { .. }) {
            run.product_flops = Some(gemm::product_flops(&run)?);
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
        .ok_or(KernelError::RequirementMismatch)?;
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

pub(super) fn fp8_scale_argument(scale: i32) -> Result<u32, KernelError> {
    if (-32..=31).contains(&scale) {
        Ok(u32::from_ne_bytes(scale.to_ne_bytes()))
    } else {
        Err(KernelError::Fp8Scale(scale))
    }
}

/// Results of family selection. Costs and access proofs use the same local
/// dimensions as the ABI; no later pass decodes argument words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct KernelCall {
    pub symbol: String,
    pub arguments: Vec<u32>,
    pub cycles: u64,
    pub padding: PaddingRequirement,
}

impl KernelCall {
    pub(super) fn new(symbol: impl Into<String>, arguments: Vec<u32>, cycles: u64) -> Self {
        Self {
            symbol: symbol.into(),
            arguments,
            cycles,
            padding: PaddingRequirement::Required,
        }
    }
}

impl KernelRun {
    pub(crate) fn call(
        &self,
        build: Option<&mut KernelObjects>,
    ) -> Result<KernelCall, KernelError> {
        if self.inputs.len() != self.requirements.inputs.len()
            || self.outputs.len() != self.requirements.outputs.len()
            || self
                .requirements
                .distinct_elements
                .iter()
                .flatten()
                .any(|&operand| self.operand_view(operand).is_none())
        {
            return Err(KernelError::RequirementMismatch);
        }
        let inputs = (0..self.inputs.len())
            .map(|index| self.geometry(MemoryOperand::Input(index as u16)))
            .collect::<Vec<_>>();
        let outputs = (0..self.outputs.len())
            .map(|index| self.geometry(MemoryOperand::Output(index as u16)))
            .collect::<Vec<_>>();
        KernelCall::select(&self.kernel, &inputs, &outputs, build)
    }
}

pub(super) fn check_arity(
    inputs: &[TensorStorage<'_>],
    outputs: &[TensorStorage<'_>],
    expected_inputs: usize,
    expected_outputs: usize,
) -> Result<(), KernelError> {
    if inputs.len() != expected_inputs {
        return Err(KernelError::PointerArity {
            expected: expected_inputs,
            actual: inputs.len(),
        });
    }
    if outputs.len() != expected_outputs {
        return Err(KernelError::RequirementMismatch);
    }
    Ok(())
}

pub(crate) struct OutputCapability {
    /// Operand zero follows the output coordinates; the remaining operands
    /// are row parameters placed with the ordinary broadcast tiling rules.
    pub complete_rows: bool,
    pub input_order: ElementOrder,
    pub output_orders: &'static [ElementOrder],
    pub column_multiple: u32,
}

impl MidOperationKind {
    pub(crate) fn output_capability(&self, precision: Precision) -> Option<OutputCapability> {
        if !matches!(precision, Precision::F8F143 { .. }) {
            return None;
        }
        match self {
            Self::Gelu | Self::BiasGelu => Some(pointwise::GELU_FP8_OUTPUT),
            Self::LayerNorm => Some(normalization::FP8_OUTPUT),
            _ => None,
        }
    }
}

pub(super) fn fp8_arguments(
    kernel: &MidOperationKind,
    inputs: &[TensorStorage<'_>],
    output: TensorStorage<'_>,
) -> Result<Option<Vec<u32>>, KernelError> {
    let Some(capability) = kernel.output_capability(output.format.precision) else {
        return Ok(None);
    };
    let input = inputs[0];
    let width = input.matrix_extent(false, true)?;
    let columns = output.matrix_extent(false, true)?;
    let packed = output.format.layout.order == ElementOrder::Amp(AmpOrder::Left);
    if width == 0
        || !width.is_multiple_of(capability.column_multiple)
        || input.matrix_extent(true, true)? != width
        || !capability
            .output_orders
            .contains(&output.format.layout.order)
        || columns
            != if packed {
                width.next_multiple_of(32)
            } else {
                width
            }
        || input.extents.len() != output.extents.len()
        || input.extents[..input.extents.len() - 1] != output.extents[..output.extents.len() - 1]
        || inputs.iter().any(|input| {
            input.format.precision != Precision::F16
                || input.format.layout.order != capability.input_order
        })
    {
        return Err(KernelError::RequirementMismatch);
    }
    let Precision::F8F143 { scale_exponent } = output.format.precision else {
        unreachable!();
    };
    Ok(Some(vec![
        input.count()? / width,
        width,
        fp8_scale_argument(i32::from(scale_exponent))?,
        u32::from(packed),
    ]))
}

/// The bias and normalization codelets consume complete, dense FP16 rows.
pub(super) fn f16_row_width(
    kernel: &MidOperationKind,
    inputs: &[TensorStorage<'_>],
    output: TensorStorage<'_>,
) -> Result<u32, KernelError> {
    if output.format.precision != Precision::F16 {
        return Err(KernelError::Unavailable(kernel.clone()));
    }
    let width = output.matrix_extent(true, true)?;
    if width == 0
        || !width.is_multiple_of(2)
        || output.format.layout.order != ElementOrder::RowMajor
        || inputs
            .iter()
            .any(|input| input.format.precision != Precision::F16)
    {
        return Err(KernelError::RequirementMismatch);
    }
    Ok(width)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PaddingRequirement {
    Required,
    Unread(Vec<Vec<crate::ShardExtent>>),
    FiniteIfZero {
        region: Vec<crate::ShardExtent>,
        zero: Vec<crate::ShardExtent>,
    },
}
