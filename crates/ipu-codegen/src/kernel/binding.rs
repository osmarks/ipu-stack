//! Address-independent kernel access contracts and call geometry.

use super::*;
use crate::low::{KernelRun, ShardView};
use crate::low::{KernelRunMetadata, WorkProvenance};
use crate::mid::MidOperationKind;
use crate::tensor::TensorFormat;
use std::sync::Arc;

/// Local geometry: planned tensors already contain physical shard dimensions.
/// Bound views additionally retain logical tails; estimates without those tails
/// treat the whole physical extent as logical.
#[derive(Clone, Copy)]
pub(crate) enum Geometry<'a> {
    Tensor(&'a crate::TensorType),
    Storage(crate::storage::TensorStorage<'a>),
}
impl<'a> Geometry<'a> {
    pub(super) fn logical_elements(self) -> Result<u32, KernelAbiError> {
        self.extents().try_fold(1u32, |count, extent| {
            count
                .checked_mul(extent.logical_end - extent.start)
                .ok_or(KernelAbiError::ElementCountOverflow)
        })
    }
    pub(super) fn extents(
        self,
    ) -> impl ExactSizeIterator<Item = crate::ShardExtent> + DoubleEndedIterator {
        (0..self.rank()).map(move |axis| match self {
            Self::Tensor(t) => crate::ShardExtent {
                axis: axis as u16,
                start: 0,
                logical_end: t.shape.0[axis],
                physical_end: t.shape.0[axis],
            },
            Self::Storage(s) => s.extents[axis],
        })
    }

    pub(super) fn count(self) -> Result<u32, KernelAbiError> {
        u32::try_from(self.elements()).map_err(|_| KernelAbiError::ElementCountOverflow)
    }

    pub(super) fn matrix_extent(self, logical: bool, columns: bool) -> Result<u32, KernelAbiError> {
        let axis = self
            .rank()
            .checked_sub(if columns { 1 } else { 2 })
            .ok_or(KernelAbiError::RequirementMismatch)?;
        Ok(if logical {
            self.logical_dimension(axis)
        } else {
            self.dimension(axis)
        })
    }
    pub(crate) fn format(self) -> &'a TensorFormat {
        match self {
            Self::Tensor(t) => &t.format,
            Self::Storage(s) => s.format,
        }
    }
    pub(crate) fn rank(self) -> usize {
        match self {
            Self::Tensor(t) => t.shape.0.len(),
            Self::Storage(s) => s.extents.len(),
        }
    }
    pub(crate) fn dimension(self, axis: usize) -> u32 {
        match self {
            Self::Tensor(t) => t.shape.0[axis],
            Self::Storage(s) => s.extents[axis].physical_end - s.extents[axis].start,
        }
    }
    pub(crate) fn trailing_dimension(self, offset: usize) -> Option<u32> {
        self.rank()
            .checked_sub(offset + 1)
            .map(|axis| self.dimension(axis))
    }
    pub(crate) fn widths(self) -> impl DoubleEndedIterator<Item = u32> + ExactSizeIterator {
        (0..self.rank()).map(move |axis| self.dimension(axis))
    }
    pub(crate) fn elements(self) -> u64 {
        self.widths()
            .fold(1u64, |n, width| n.saturating_mul(u64::from(width)))
    }

    pub(super) fn logical_dimension(self, axis: usize) -> u32 {
        match self {
            Self::Tensor(t) => t.shape.0[axis],
            Self::Storage(s) => s.extents[axis].logical_end - s.extents[axis].start,
        }
    }

    pub(super) fn rows(self) -> u64 {
        self.widths()
            .take(self.rank().saturating_sub(1))
            .fold(1u64, |n, width| n.saturating_mul(u64::from(width)))
    }
}

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
    pub(crate) fn cycles(&self) -> u64 {
        self.call().map_or(u64::MAX, |call| call.cycles())
    }

    pub(super) fn geometry(&self, operand: MemoryOperand) -> Geometry<'_> {
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
        Geometry::Storage(crate::storage::TensorStorage {
            format: &access.format,
            extents: &view.extents,
        })
    }

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
        let output_shard = &shards[output.shard.index() as usize];
        let operands = || {
            (0..inputs.len())
                .map(|index| (MemoryOperand::Input(index as u16), &inputs[index]))
                .chain(
                    (0..outputs.len())
                        .map(|index| (MemoryOperand::Output(index as u16), &outputs[index])),
                )
        };
        let shared = metadata.iter().find(|metadata| {
            metadata.provenance == provenance
                && metadata.kernel == kernel
                && metadata.requirements.inputs.len() == inputs.len()
                && metadata.requirements.outputs.len() == outputs.len()
                && operands()
                    .zip(
                        metadata
                            .requirements
                            .inputs
                            .iter()
                            .chain(&metadata.requirements.outputs),
                    )
                    .all(|((operand, view), requirement)| {
                        *format(view) == requirement.format
                            && kernel.access(operand, output_shard).0 == requirement.storage
                    })
        });
        let shared = if let Some(shared) = shared {
            Arc::clone(shared)
        } else {
            let mut accesses = operands().map(|(operand, view)| KernelAccess {
                format: format(view).clone(),
                storage: kernel.access(operand, output_shard).0,
            });
            let distinct = kernel.access(MemoryOperand::Output(0), output_shard).1;
            let requirements = KernelRequirements {
                inputs: accesses.by_ref().take(inputs.len()).collect(),
                outputs: accesses.collect(),
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
        run.call()?;
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
    Rearrange((ElementOrder, u32, u32, u32, u32)),
    Unpack((ElementOrder, u32, u32, u32, u32)),
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
        let inputs = (0..self.inputs.len())
            .map(|index| self.geometry(MemoryOperand::Input(index as u16)))
            .collect::<Vec<_>>();
        let outputs = (0..self.outputs.len())
            .map(|index| self.geometry(MemoryOperand::Output(index as u16)))
            .collect::<Vec<_>>();
        KernelCall::select(&self.kernel, &inputs, &outputs)
    }
}

pub(super) fn check_arity(
    inputs: &[Geometry<'_>],
    outputs: &[Geometry<'_>],
    expected_inputs: usize,
    expected_outputs: usize,
) -> Result<(), KernelAbiError> {
    if inputs.len() != expected_inputs {
        return Err(KernelAbiError::PointerArity {
            expected: expected_inputs,
            actual: inputs.len(),
        });
    }
    if outputs.len() != expected_outputs {
        return Err(KernelAbiError::RequirementMismatch);
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
    inputs: &[Geometry<'_>],
    output: Geometry<'_>,
) -> Result<Option<Vec<u32>>, KernelAbiError> {
    let Some(capability) = kernel.output_capability(output.format().precision) else {
        return Ok(None);
    };
    let input = inputs[0];
    let width = input.matrix_extent(false, true)?;
    let columns = output.matrix_extent(false, true)?;
    let packed = output.format().layout.order == ElementOrder::Amp(AmpOrder::Left);
    if width == 0
        || !width.is_multiple_of(capability.column_multiple)
        || input.matrix_extent(true, true)? != width
        || !capability
            .output_orders
            .contains(&output.format().layout.order)
        || columns
            != if packed {
                width.next_multiple_of(32)
            } else {
                width
            }
        || input.rank() != output.rank()
        || !input
            .extents()
            .take(input.rank() - 1)
            .eq(output.extents().take(output.rank() - 1))
        || inputs.iter().any(|input| {
            input.format().precision != Precision::F16
                || input.format().layout.order != capability.input_order
        })
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let Precision::F8F143 { scale_exponent } = output.format().precision else {
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
    inputs: &[Geometry<'_>],
    output: Geometry<'_>,
) -> Result<u32, KernelAbiError> {
    if output.format().precision != Precision::F16 {
        return Err(KernelAbiError::Unavailable(kernel.clone()));
    }
    let width = output.matrix_extent(true, true)?;
    if width == 0
        || !width.is_multiple_of(2)
        || output.format().layout.order != ElementOrder::RowMajor
        || inputs
            .iter()
            .any(|input| input.format().precision != Precision::F16)
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    Ok(width)
}

pub(crate) enum PaddingRequirement {
    Required,
    Unread(Vec<ShardView>),
    FiniteIfZero { region: ShardView, zero: ShardView },
}

impl KernelCall {
    pub(crate) fn input_padding(
        &self,
        run: &KernelRun,
        operand: usize,
    ) -> Result<PaddingRequirement, KernelAbiError> {
        if operand != 0 {
            return Ok(PaddingRequirement::Required);
        }
        match self.implementation {
            KernelImplementation::Exact("cast_f16_f8") => cast::input_padding(self, run),
            KernelImplementation::Gemm(Precision::F16, _, inner, _, _, _, _) => {
                gemm::input_padding(run, inner)
            }
            _ => Ok(PaddingRequirement::Required),
        }
    }
}
