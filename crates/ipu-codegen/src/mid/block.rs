//! Executable whole-device mid operations and their ordinary tensor blocks.
//! Layout choices, copies, GEMM blocks, reductions, and attention stages are
//! explicit here. Tile-list projection and placement do not expand operators.

use super::*;
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramInput {
    pub name: String,
    pub kind: GraphInputKind,
    pub value: MidValueId,
    pub shards: Vec<BlockValueId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueBlocks {
    pub value: MidValueId,
    pub shards: Vec<BlockValueId>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExchangePhase {
    pub id: ExchangePhaseId,
    pub provenance: WorkProvenance,
    pub transfers: Vec<LogicalExchange>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkReason {
    OperatorKernel,
    OperatorInput { input: u16 },
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
pub struct KernelOperand {
    /// Views resident on the execution tile which form this ABI operand.
    pub views: Vec<ShardView>,
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
    pub output: KernelAccess,
    pub distinct_elements: Vec<Vec<MemoryOperand>>,
}

impl KernelRequirements {
    pub fn new(
        kernel: &TileKernelSpec,
        inputs: impl IntoIterator<Item = TensorFormat>,
        output: TensorFormat,
    ) -> Self {
        let alignment = match kernel {
            TileKernelSpec::Gemm { .. } => 32,
            TileKernelSpec::Rearrange { .. } => 2,
            _ => 8,
        };
        let mut requirements = Self {
            inputs: inputs
                .into_iter()
                .map(|format| KernelAccess::new(format, alignment))
                .collect(),
            output: KernelAccess::new(output, alignment),
            distinct_elements: Vec::new(),
        };
        if let TileKernelSpec::Gemm { multiply, .. } = kernel
            && let Some(left) = requirements.inputs.first_mut()
        {
            left.access_tail_bytes = 8 * multiply.bytes() as u32;
            requirements
                .distinct_elements
                .push(vec![MemoryOperand::Output, MemoryOperand::Input(0)]);
        }
        requirements
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelRunMetadata {
    pub provenance: WorkProvenance,
    pub kernel: TileKernelSpec,
    pub requirements: KernelRequirements,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelRun {
    pub(crate) metadata: Arc<KernelRunMetadata>,
    pub inputs: Vec<KernelOperand>,
    pub output: ShardView,
}

impl KernelRun {
    pub fn new(
        provenance: WorkProvenance,
        kernel: TileKernelSpec,
        inputs: Vec<KernelOperand>,
        output: ShardView,
        requirements: KernelRequirements,
    ) -> Self {
        Self {
            metadata: Arc::new(KernelRunMetadata {
                provenance,
                kernel,
                requirements,
            }),
            inputs,
            output,
        }
    }
}

impl std::ops::Deref for KernelRun {
    type Target = KernelRunMetadata;

    fn deref(&self) -> &Self::Target {
        &self.metadata
    }
}

pub type LocalCopy = crate::mid::CopyOperation<BlockValueId>;

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
    pub inputs: Vec<BlockValueId>,
    pub argument: BlockValueId,
    /// Placement must assign entries consecutively at this byte stride.
    pub stride_bytes: u32,
    pub alignment: u32,
}

/// A device-wide region preserving each tile's operation order. Entries on
/// distinct tiles may overlap; exchange markers provide global synchronization.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockRegion {
    pub operations: Vec<BlockOperation>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidProgram {
    pub tile_count: u16,
    pub shards: Vec<BlockValue>,
    pub exchange_phases: Vec<ExchangePhase>,
    pub inputs: Vec<ProgramInput>,
    pub body: BlockRegion,
    pub kernel_runs: Vec<KernelRun>,
    pub local_copies: Vec<LocalCopy>,
    pub values: Vec<ValueBlocks>,
    pub outputs: Vec<ValueBlocks>,
    pub logical_values: Vec<MidValue>,
    pub checkpoints: Vec<(OperationId, Vec<MidValueId>)>,
    pub estimated_cycles: u64,
    pub estimated_exchange_cycles: u64,
}

pub(crate) fn storage_root(shards: &[BlockValue], mut shard: BlockValueId) -> BlockValueId {
    let mut remaining = shards.len().saturating_add(1);
    while remaining != 0 {
        remaining -= 1;
        shard = match shards[shard.index() as usize].definition {
            ShardDefinition::Alias(source) | ShardDefinition::WritableAlias(source) => source,
            _ => return shard,
        };
    }
    shard
}
