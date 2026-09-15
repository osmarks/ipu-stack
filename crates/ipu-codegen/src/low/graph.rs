//! Expanded tile storage, calls and movement, before physical placement.
//! Distributed algorithms and intermediate tensor layouts were selected in mid.

use crate::graph::OperationId;
use crate::kernel::TileKernelSpec;
use crate::kernel::{KernelRequirements, MemoryOperand};
use crate::mid::{MidInput, MidValue, MidValueId};
use crate::storage::CopyOrder;
use crate::tensor::{ShardExtent, TensorType};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockValueId(pub(crate) u32);

impl BlockValueId {
    pub const fn index(self) -> u32 {
        self.0
    }

    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExchangePhaseId(pub(crate) u32);

impl ExchangePhaseId {
    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }

    pub const fn index(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelRunId(pub(crate) u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalCopyId(pub(crate) u32);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShardView {
    pub shard: BlockValueId,
    pub extents: Vec<ShardExtent>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardDefinition {
    Value(MidValueId),
    /// Transient storage populated by one or more exchange phases.
    ExchangeStaging,
    LocalCopy(BlockValueId),
    /// Persistent scratch allocation populated by local copies or exchanges.
    Staging,
    Alias(BlockValueId),
    /// Alias intentionally used as an in-place operation destination.
    WritableAlias(BlockValueId),
    /// Writable storage donation at a signed byte displacement from its source.
    /// Placement reserves the union and preserves this offset through aliases.
    ShiftedAlias {
        source: BlockValueId,
        offset: i32,
    },
    /// Canonical format placeholder replaced by dispatch-local staging.
    Unmaterialized,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockValue {
    pub id: BlockValueId,
    pub tile: u16,
    pub tensor_type: TensorType,
    pub extents: Vec<ShardExtent>,
    pub definition: ShardDefinition,
}

/// One source view may populate arbitrary corresponding views on several
/// tiles. Sequential phases may reuse transient destinations after consumers
/// have run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalExchange {
    pub source: ShardView,
    pub destinations: Vec<ShardView>,
    pub order: CopyOrder,
}

impl LogicalExchange {
    /// Equal storage orders can traverse physical spans while preserving the
    /// requested coordinates. Share this decision with pricing and encoding.
    pub(crate) fn span_order(&self, shards: &[BlockValue]) -> CopyOrder {
        if self.order == CopyOrder::Panels {
            return CopyOrder::Panels;
        }
        let source = &shards[self.source.shard.index() as usize];
        if self.order == CopyOrder::Semantic
            && self.destinations.iter().any(|view| {
                shards[view.shard.index() as usize]
                    .tensor_type
                    .format
                    .layout
                    .order
                    != source.tensor_type.format.layout.order
            })
        {
            CopyOrder::Semantic
        } else {
            CopyOrder::Physical
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExchangePhase {
    pub id: ExchangePhaseId,
    pub provenance: WorkProvenance,
    pub transfers: Vec<LogicalExchange>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkReason {
    OperatorKernel,
    OperatorInputs,
    PrecisionCast,
    LayoutRearrangement,
    Repeat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkProvenance {
    pub operation: Option<OperationId>,
    pub value: Option<MidValueId>,
    pub reason: WorkReason,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelRunMetadata {
    pub provenance: WorkProvenance,
    pub kernel: TileKernelSpec,
    pub requirements: KernelRequirements,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelRun {
    /// Useful and physical product FLOPs, before axes are erased by the ABI.
    pub product_flops: Option<[u64; 2]>,
    pub(crate) metadata: Arc<KernelRunMetadata>,
    pub inputs: Vec<ShardView>,
    /// All result bindings in the family's declared order.
    pub outputs: Vec<ShardView>,
}

impl KernelRun {
    pub(crate) fn operand_view(&self, operand: MemoryOperand) -> Option<&ShardView> {
        match operand {
            MemoryOperand::Input(index) => self.inputs.get(usize::from(index)),
            MemoryOperand::Output(index) => self.outputs.get(usize::from(index)),
        }
    }

    #[cfg(test)]
    pub fn new(
        provenance: WorkProvenance,
        kernel: TileKernelSpec,
        inputs: Vec<ShardView>,
        outputs: Vec<ShardView>,
        requirements: KernelRequirements,
    ) -> Self {
        Self {
            product_flops: None,
            metadata: Arc::new(KernelRunMetadata {
                provenance,
                kernel,
                requirements,
            }),
            inputs,
            outputs,
        }
    }
}

impl std::ops::Deref for KernelRun {
    type Target = KernelRunMetadata;

    fn deref(&self) -> &Self::Target {
        &self.metadata
    }
}

pub type LocalCopy = crate::CopyOperation<BlockValueId>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepeatCarried {
    pub initial: BlockValueId,
    pub argument: BlockValueId,
    pub yielded: BlockValueId,
    pub result: BlockValueId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepeatInvariant {
    pub input: BlockValueId,
    pub argument: BlockValueId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepeatIterated {
    /// Placement derives a uniform physical stride from the body's access contracts.
    pub inputs: Vec<BlockValueId>,
    pub argument: BlockValueId,
}

/// A device-wide region preserving each tile's operation order. Entries on
/// distinct tiles may overlap; exchange markers provide global synchronization.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockRegion {
    pub operations: Vec<BlockOperation>,
}

impl BlockRegion {
    /// Live operations in structural order, entering each Repeat body once.
    /// Arena entries alone do not imply execution; removed calls may remain interned.
    pub(crate) fn walk(&self) -> impl Iterator<Item = &BlockOperation> {
        let mut regions = vec![self.operations.iter()];
        std::iter::from_fn(move || {
            loop {
                let Some(operation) = regions.last_mut()?.next() else {
                    regions.pop();
                    continue;
                };
                if let BlockOperation::Repeat(repeat) = operation {
                    regions.push(repeat.body.operations.iter());
                }
                return Some(operation);
            }
        })
    }

    pub(crate) fn retain(&mut self, keep: &mut impl FnMut(&BlockOperation) -> bool) {
        self.operations.retain_mut(|operation| {
            if !keep(operation) {
                return false;
            }
            if let BlockOperation::Repeat(repeat) = operation {
                repeat.body.retain(keep);
            }
            true
        });
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockOperation {
    Exchange(ExchangePhaseId),
    Copy { tile: u16, copy: LocalCopyId },
    Compute { tile: u16, run: KernelRunId },
    Repeat(Box<BlockRepeat>),
    Checkpoint(OperationId, u8),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockRepeat {
    pub provenance: WorkProvenance,
    pub count: u32,
    pub bindings: Vec<BlockRepeatBinding>,
    pub body: BlockRegion,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockRepeatBinding {
    pub tile: u16,
    pub carried: Vec<RepeatCarried>,
    pub invariants: Vec<RepeatInvariant>,
    pub iterated: Vec<RepeatIterated>,
}

impl BlockRepeatBinding {
    /// Storage exposed by bindings on either side of the body boundary.
    pub(crate) fn bound_shards(&self) -> impl Iterator<Item = BlockValueId> + '_ {
        self.carried
            .iter()
            .flat_map(|binding| {
                [
                    binding.initial,
                    binding.argument,
                    binding.yielded,
                    binding.result,
                ]
            })
            .chain(
                self.invariants
                    .iter()
                    .flat_map(|binding| [binding.input, binding.argument]),
            )
            .chain(self.iterated.iter().flat_map(|binding| {
                std::iter::once(binding.argument).chain(binding.inputs.iter().copied())
            }))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileGraph {
    pub tile_count: u16,
    /// Padding clears were elided using the finite-F16 arena invariant.
    /// Such storage cannot borrow memory containing host protocol words.
    pub requires_finite_scratch: bool,
    pub shards: Vec<BlockValue>,
    pub exchange_phases: Vec<ExchangePhase>,
    pub inputs: Vec<MidInput>,
    pub body: BlockRegion,
    pub kernel_runs: Vec<KernelRun>,
    pub local_copies: Vec<LocalCopy>,
    /// Concrete reads indexed by MidValueId, including borrowed selections.
    /// Logical shapes remain in logical_values; backing strides belong to shards.
    pub value_views: Vec<Vec<ShardView>>,
    pub outputs: Vec<MidValueId>,
    pub logical_values: Vec<MidValue>,
    pub checkpoints: Vec<(OperationId, Vec<MidValueId>)>,
}

impl TileGraph {
    /// Exchange phases are whole-device work. Count their executions in the
    /// authoritative region tree, without visiting every tile's projection.
    pub(crate) fn exchange_multiplicities(&self) -> Vec<u64> {
        fn count(region: &BlockRegion, repetitions: u64, counts: &mut [u64]) {
            for operation in &region.operations {
                match operation {
                    BlockOperation::Exchange(id) => {
                        counts[id.index() as usize] =
                            counts[id.index() as usize].saturating_add(repetitions);
                    }
                    BlockOperation::Repeat(repeat) => count(
                        &repeat.body,
                        repetitions.saturating_mul(u64::from(repeat.count)),
                        counts,
                    ),
                    _ => {}
                }
            }
        }
        let mut counts = vec![0; self.exchange_phases.len()];
        count(&self.body, 1, &mut counts);
        counts
    }

    pub fn value_views(&self, value: MidValueId) -> &[ShardView] {
        &self.value_views[value.index() as usize]
    }

    pub(crate) fn kernel_calls(&self) -> impl Iterator<Item = &KernelRun> {
        self.body.walk().filter_map(|operation| match operation {
            BlockOperation::Compute { run, .. } => Some(&self.kernel_runs[run.0 as usize]),
            _ => None,
        })
    }
}

pub(crate) fn storage_root(shards: &[BlockValue], shard: BlockValueId) -> BlockValueId {
    storage_location(shards, shard).0
}

/// Byte origin relative to the ultimate backing value, before placement.
pub(crate) fn storage_location(shards: &[BlockValue], shard: BlockValueId) -> (BlockValueId, i64) {
    storage_chain(shards, shard).last().unwrap()
}

/// Follow backing storage while retaining the byte displacement at each binding.
/// Repeat arguments may acquire other shard IDs through views and reductions.
pub(crate) fn storage_chain(
    shards: &[BlockValue],
    shard: BlockValueId,
) -> impl Iterator<Item = (BlockValueId, i64)> + '_ {
    std::iter::successors(Some((shard, 0i64)), |&(shard, offset)| {
        match shards[shard.index() as usize].definition {
            ShardDefinition::Alias(source) | ShardDefinition::WritableAlias(source) => {
                Some((source, offset))
            }
            ShardDefinition::ShiftedAlias {
                source,
                offset: delta,
            } => Some((source, offset + i64::from(delta))),
            _ => None,
        }
    })
    .take(shards.len().saturating_add(2))
}
