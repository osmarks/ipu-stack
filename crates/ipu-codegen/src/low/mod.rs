//! Logical per-tile schedule produced from the layout-aware mid-level IR.
//!
//! Tensor shards have tile identities and rectangular physical extents, and
//! work is ordered per tile. Exchanges still refer to logical shards rather
//! than SRAM addresses; kernel runs still name a selected kernel kind rather
//! than a linked symbol. Placement and final code generation resolve those
//! remaining choices.

mod attention;
mod conversion;
mod copies;
mod gemm;
use crate::graph::{GraphInputKind, OperationId};
use crate::mid::{
    AMP_COLUMN_MICRO, AMP_INNER_BLOCK, AmpOrder, AxisFactorView, BlockMajorOrder,
    ConversionStrategy, CopyPattern, ElementOrder, GemmDistribution, Layout, LayoutError,
    MemoryClass, MemoryOperand, MidGraph, MidOperation, MidOperationKind, MidRepeat, MidValueId,
    OperandRequirement, OperatorDispatch, OperatorRequirements, OutputAliasing, PipelineConfig,
    PointwiseInputMapping, Precision, ShardExtent, TensorTiling, TensorType, TileKernelSpec,
};
use crate::storage::{ByteSpan, StorageError};
use conversion::*;
use copies::*;
pub use copies::{logical_view_byte_spans, shard_storage_bytes, view_byte_spans};
use gemm::*;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LowShardId(u32);

impl LowShardId {
    pub const fn index(self) -> u32 {
        self.0
    }

    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExchangePhaseId(u32);

impl ExchangePhaseId {
    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }

    pub const fn index(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelRunId(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalCopyId(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RepeatRunId(u32);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShardView {
    pub shard: LowShardId,
    pub extents: Vec<ShardExtent>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardDefinition {
    Value(MidValueId),
    /// Transient storage populated by one or more exchange phases.
    ExchangeStaging,
    LocalCopy(LowShardId),
    /// Persistent scratch allocation populated by local copies or exchanges.
    Staging,
    Alias(LowShardId),
    /// Alias intentionally used as an in-place operation destination.
    WritableAlias(LowShardId),
    /// Canonical format placeholder replaced by dispatch-local staging.
    Unmaterialized,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LowShard {
    pub id: LowShardId,
    pub tile: u16,
    pub tensor_type: TensorType,
    pub extents: Vec<ShardExtent>,
    pub definition: ShardDefinition,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LowInput {
    pub name: String,
    pub kind: GraphInputKind,
    pub value: MidValueId,
    pub shards: Vec<LowShardId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LowValue {
    pub value: MidValueId,
    pub shards: Vec<LowShardId>,
}

/// One source view may populate arbitrary corresponding views on several
/// tiles. Sequential phases may reuse transient destinations after consumers
/// have run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalExchange {
    pub source: ShardView,
    pub destinations: Vec<ShardView>,
    pub order: ExchangeOrder,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExchangeOrder {
    /// Preserve tensor coordinates, converting between physical layouts.
    #[default]
    Semantic,
    /// Preserve allocation order, treating both views as packed byte spans.
    Physical,
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
pub enum TileKernel {
    Planned(TileKernelSpec),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelRequirements {
    Operator(OperatorRequirements),
    Conversion {
        input: OperandRequirement,
        output: OperandRequirement,
        distinct_elements: Vec<Vec<MemoryOperand>>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelOperand {
    /// Views resident on the execution tile which form this ABI operand.
    pub views: Vec<ShardView>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelRunMetadata {
    pub provenance: WorkProvenance,
    pub kernel: TileKernel,
    pub requirements: KernelRequirements,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelRun {
    metadata: Arc<KernelRunMetadata>,
    pub inputs: Vec<KernelOperand>,
    pub output: ShardView,
}

impl KernelRun {
    pub fn new(
        provenance: WorkProvenance,
        kernel: TileKernel,
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

pub type LocalCopy = crate::mid::CopyOperation<LowShardId>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepeatCarried {
    pub initial: LowShardId,
    pub argument: LowShardId,
    pub yielded: LowShardId,
    pub result: LowShardId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepeatInvariant {
    pub input: LowShardId,
    pub argument: LowShardId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepeatIterated {
    pub inputs: Vec<LowShardId>,
    pub argument: LowShardId,
    /// Placement must assign entries consecutively at this byte stride.
    pub stride_bytes: u32,
    pub alignment: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepeatRun {
    pub provenance: WorkProvenance,
    pub count: u32,
    pub carried: Vec<RepeatCarried>,
    pub invariants: Vec<RepeatInvariant>,
    pub iterated: Vec<RepeatIterated>,
    pub body: Box<TileWorkList>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileWork {
    /// All tiles encounter a phase marker, including tiles without transfers.
    Exchange(ExchangePhaseId),
    LocalCopy(LocalCopyId),
    Kernel(KernelRunId),
    Repeat(RepeatRunId),
    Checkpoint(OperationId, u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileWorkRef<'a> {
    Exchange(ExchangePhaseId),
    LocalCopy(&'a LocalCopy),
    Kernel(&'a KernelRun),
    Repeat(&'a RepeatRun),
    Checkpoint(OperationId, u8),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileWorkList {
    pub tile: u16,
    pub work: Vec<TileWork>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LowProgram {
    pub tile_count: u16,
    pub shards: Vec<LowShard>,
    pub exchange_phases: Vec<ExchangePhase>,
    pub inputs: Vec<LowInput>,
    /// Compact per-tile ordering. Non-exchange entries index the arenas below.
    pub tiles: Vec<TileWorkList>,
    /// Tile-specific kernel operands and outputs, with shared call metadata.
    pub kernel_runs: Vec<KernelRun>,
    pub local_copies: Vec<LocalCopy>,
    pub repeat_runs: Vec<RepeatRun>,
    /// Canonical materialization of every mid-level value that reaches tile
    /// lowering. Diagnostic metadata uses this without adding device copies.
    pub values: Vec<LowValue>,
    pub outputs: Vec<LowValue>,
}

impl LowProgram {
    /// Resolves compact schedule entries as they are consumed, without
    /// constructing a second per-tile work list.
    pub fn work<'a>(
        &'a self,
        tile: &'a TileWorkList,
    ) -> impl Iterator<Item = TileWorkRef<'a>> + 'a {
        tile.work.iter().map(|work| match *work {
            TileWork::Exchange(id) => TileWorkRef::Exchange(id),
            TileWork::LocalCopy(id) => TileWorkRef::LocalCopy(&self.local_copies[id.0 as usize]),
            TileWork::Kernel(id) => TileWorkRef::Kernel(&self.kernel_runs[id.0 as usize]),
            TileWork::Repeat(id) => TileWorkRef::Repeat(&self.repeat_runs[id.0 as usize]),
            TileWork::Checkpoint(operation, breakpoint) => {
                TileWorkRef::Checkpoint(operation, breakpoint)
            }
        })
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LowLoweringError {
    #[error("low-level lowering requires a nonzero tile count")]
    EmptyTileGroup,
    #[error("value {value:?} declares {declared} tiles, but the schedule capacity is {scheduled}")]
    TileCountMismatch {
        value: MidValueId,
        declared: u16,
        scheduled: u16,
    },
    #[error("value {0:?} does not exist")]
    UnknownValue(MidValueId),
    #[error("operation must have exactly one result")]
    ResultArity,
    #[error("operator plan is incompatible with its values or block dimensions")]
    InvalidOperatorPlan,
    #[error("conversion plan is incompatible with its input or output")]
    InvalidConversionPlan,
    #[error("repeat structure is inconsistent with its inputs, arguments, yields, or results")]
    InvalidRepeat,
    #[error("repeat carried value {0} cannot alias its body argument")]
    RepeatRequiresInPlace(usize),
    #[error("repeat iterated input {0} cannot be represented as equal contiguous blocks")]
    InvalidIteratedBlocks(usize),
    #[error("too many logical shards or exchange phases")]
    IdOverflow,
    #[error("invalid tensor layout: {0}")]
    Layout(#[from] LayoutError),
    #[error("invalid tensor storage view: {0}")]
    Storage(#[from] StorageError),
}

pub type LowLoweringResult<T> = Result<T, LowLoweringError>;

fn append_checkpoint(tiles: &mut [TileWorkList], operation: OperationId, breakpoint: u8) {
    for tile in tiles {
        tile.work.push(TileWork::Checkpoint(operation, breakpoint));
    }
}

/// Produces a logical per-tile schedule by expanding selected operator plans.
/// Conversions without plans still use a conservative gather fallback.
#[tracing::instrument(
    name = "ipu_codegen.low.lower_to_tiles",
    skip(graph, config),
    fields(
        tile_count = config.tile_count,
        operations = graph.operations.len(),
        profiling = config.profiling
    )
)]
pub fn lower_to_tiles(graph: &MidGraph, config: &PipelineConfig) -> LowLoweringResult<LowProgram> {
    if config.tile_count == 0 {
        return Err(LowLoweringError::EmptyTileGroup);
    }
    let mut state = LoweringState::new(graph, config.tile_count)?;
    let tiles = state.lower_region(&graph.operations, config.diagnostic_checkpoints)?;
    let inputs = graph
        .inputs
        .iter()
        .map(|input| {
            Ok(LowInput {
                name: input.name.clone(),
                kind: input.kind,
                value: input.value,
                shards: state.value_shards(input.value)?.to_vec(),
            })
        })
        .collect::<LowLoweringResult<_>>()?;
    let outputs = graph
        .outputs
        .iter()
        .map(|value| {
            Ok(LowValue {
                value: *value,
                shards: state.value_shards(*value)?.to_vec(),
            })
        })
        .collect::<LowLoweringResult<_>>()?;
    let values = graph
        .values
        .iter()
        .filter_map(|value| {
            let shards = &state.canonical[value.id.index() as usize];
            (!shards.is_empty()).then(|| LowValue {
                value: value.id,
                shards: shards.clone(),
            })
        })
        .collect();
    tracing::info!(
        shards = state.shards.len(),
        exchange_phases = state.phases.len(),
        "built logical tile schedule"
    );
    Ok(LowProgram {
        tile_count: config.tile_count,
        shards: state.shards,
        exchange_phases: state.phases,
        inputs,
        tiles,
        kernel_runs: state.kernel_runs,
        local_copies: state.local_copies,
        repeat_runs: state.repeat_runs,
        values,
        outputs,
    })
}

type ShardIntersections = Vec<(Vec<ShardExtent>, Vec<LowShardId>)>;

struct LoweringState {
    tile_count: u16,
    shards: Vec<LowShard>,
    canonical: Vec<Vec<LowShardId>>,
    phases: Vec<ExchangePhase>,
    kernel_runs: Vec<KernelRun>,
    local_copies: Vec<LocalCopy>,
    repeat_runs: Vec<RepeatRun>,
    kernel_metadata: Vec<Arc<KernelRunMetadata>>,
    deferred_conversions: BTreeMap<MidValueId, MidValueId>,
    deferred_values: BTreeMap<MidValueId, DeferredValue>,
    intersection_cache: BTreeMap<(MidValueId, Vec<ShardExtent>), ShardIntersections>,
}

impl LoweringState {
    fn storage_root(&self, mut shard: LowShardId) -> LowShardId {
        let mut remaining = self.shards.len().saturating_add(1);
        while remaining != 0 {
            remaining -= 1;
            shard = match self.shards[shard.index() as usize].definition {
                ShardDefinition::Alias(source) | ShardDefinition::WritableAlias(source) => source,
                _ => return shard,
            };
        }
        shard
    }

    fn new(graph: &MidGraph, tile_count: u16) -> LowLoweringResult<Self> {
        let mut state = Self {
            tile_count,
            shards: Vec::new(),
            canonical: vec![Vec::new(); graph.values.len()],
            phases: Vec::new(),
            kernel_runs: Vec::new(),
            local_copies: Vec::new(),
            repeat_runs: Vec::new(),
            kernel_metadata: Vec::new(),
            deferred_conversions: BTreeMap::new(),
            deferred_values: BTreeMap::new(),
            intersection_cache: BTreeMap::new(),
        };
        let parameter_origins = graph
            .inputs
            .iter()
            .filter(|input| input.kind == GraphInputKind::Parameter)
            .map(|input| graph.values[input.value.index() as usize].origin)
            .collect::<BTreeSet<_>>();
        let parameter_values = graph
            .values
            .iter()
            .filter(|value| parameter_origins.contains(&value.origin))
            .map(|value| value.id)
            .collect::<BTreeSet<_>>();
        let parameter_groups = parameter_values
            .iter()
            .map(|value| graph.values[value.index() as usize].storage_group)
            .collect::<BTreeSet<_>>();
        let mut parameter_bytes = vec![0u64; usize::from(tile_count)];
        let mut parameter_offsets = BTreeMap::<MidValueId, u16>::new();
        for value in &graph.values {
            let declared_tiles = value.tensor_type.format.layout.tiling.tile_count;
            if declared_tiles == 0 || declared_tiles > tile_count {
                return Err(LowLoweringError::TileCountMismatch {
                    value: value.id,
                    declared: declared_tiles,
                    scheduled: tile_count,
                });
            }
            let extents = value
                .tensor_type
                .format
                .layout
                .shard_extents(&value.tensor_type.shape)?;
            let is_parameter = parameter_values.contains(&value.id);
            let placement_group = value.storage_group;
            let rotate_parameter = is_parameter || parameter_groups.contains(&placement_group);
            let parameter_shard_bytes = if rotate_parameter {
                extents
                    .iter()
                    .map(|(_, extents)| {
                        crate::storage::storage_bytes(crate::storage::TensorStorage {
                            format: &value.tensor_type.format,
                            extents,
                        })
                        .map(u64::from)
                        .map_err(LowLoweringError::from)
                    })
                    .collect::<LowLoweringResult<Vec<_>>>()?
            } else {
                Vec::new()
            };
            let parameter_offset = if rotate_parameter {
                if let Some(&offset) = parameter_offsets.get(&placement_group) {
                    offset
                } else {
                    let offset = (0..tile_count)
                        .min_by_key(|&offset| {
                            let mut loads = parameter_bytes.clone();
                            for (logical, &bytes) in parameter_shard_bytes.iter().enumerate() {
                                let tile =
                                    (logical + usize::from(offset)) % usize::from(tile_count);
                                loads[tile] = loads[tile].saturating_add(bytes);
                            }
                            (loads.into_iter().max().unwrap_or(u64::MAX), offset)
                        })
                        .ok_or(LowLoweringError::EmptyTileGroup)?;
                    tracing::debug!(
                        ?placement_group,
                        offset,
                        shards = parameter_shard_bytes.len(),
                        "assigned parameter storage group to tiles"
                    );
                    parameter_offsets.insert(placement_group, offset);
                    offset
                }
            } else {
                0
            };
            let mut value_shards = Vec::with_capacity(extents.len());
            for (logical_shard, (owner_tile, extents)) in extents.into_iter().enumerate() {
                let mut shard = LowShard {
                    id: LowShardId(0),
                    tile: 0,
                    tensor_type: value.tensor_type.clone(),
                    extents,
                    definition: ShardDefinition::Value(value.id),
                };
                shard.tile = if rotate_parameter {
                    let tile = (usize::from(owner_tile) + usize::from(parameter_offset))
                        % usize::from(tile_count);
                    let bytes = parameter_shard_bytes[logical_shard];
                    parameter_bytes[tile] = parameter_bytes[tile]
                        .checked_add(bytes)
                        .ok_or(LowLoweringError::IdOverflow)?;
                    u16::try_from(tile).map_err(|_| LowLoweringError::IdOverflow)?
                } else {
                    owner_tile
                };
                let id = state.push_shard(shard)?;
                value_shards.push(id);
            }
            state.canonical[value.id.index() as usize] = value_shards;
        }
        Ok(state)
    }

    fn push_shard(&mut self, mut shard: LowShard) -> LowLoweringResult<LowShardId> {
        let id =
            LowShardId(u32::try_from(self.shards.len()).map_err(|_| LowLoweringError::IdOverflow)?);
        shard.id = id;
        self.shards.push(shard);
        Ok(id)
    }

    fn interleaved_capacity_available(
        &self,
        tile: u16,
        bytes: u32,
        access_tail: u32,
    ) -> LowLoweringResult<bool> {
        let used = self
            .shards
            .iter()
            .filter(|shard| {
                shard.tile == tile
                    && shard.tensor_type.format.layout.memory_class
                        == crate::MemoryClass::Ipu21Interleaved
                    && !matches!(
                        shard.definition,
                        ShardDefinition::Alias(_)
                            | ShardDefinition::WritableAlias(_)
                            | ShardDefinition::ExchangeStaging
                    )
            })
            .try_fold(0u32, |total, shard| {
                total
                    .checked_add(crate::shard_storage_bytes(shard)?)
                    .and_then(|total| total.checked_add(access_tail))
                    .ok_or(LowLoweringError::IdOverflow)
            })?;
        Ok(used
            .checked_add(bytes)
            .and_then(|total| total.checked_add(access_tail))
            .is_some_and(|total| total <= crate::memory::IPU21_INTERLEAVED_REGION_BYTES))
    }

    fn value_shards(&self, value: MidValueId) -> LowLoweringResult<&[LowShardId]> {
        self.canonical
            .get(value.index() as usize)
            .filter(|shards| !shards.is_empty())
            .map(Vec::as_slice)
            .ok_or(LowLoweringError::UnknownValue(value))
    }

    fn local_shard(&self, value: MidValueId, tile: u16) -> LowLoweringResult<LowShardId> {
        let shards = self.value_shards(value)?;
        if let Some(&shard) = shards.get(usize::from(tile))
            && self.shards[shard.index() as usize].tile == tile
        {
            return Ok(shard);
        }
        shards
            .iter()
            .copied()
            .find(|shard| self.shards[shard.index() as usize].tile == tile)
            .ok_or(LowLoweringError::UnknownValue(value))
    }

    fn intersecting_shards(
        &mut self,
        source: MidValueId,
        target: &[ShardExtent],
        local_tile: u16,
    ) -> LowLoweringResult<Vec<(Vec<ShardExtent>, LowShardId)>> {
        let key = (source, target.to_vec());
        if !self.intersection_cache.contains_key(&key) {
            let mut groups = BTreeMap::<Vec<ShardExtent>, Vec<LowShardId>>::new();
            for shard in self.value_shards(source)?.to_vec() {
                if let Some(extents) =
                    intersect_extents(&self.shards[shard.index() as usize].extents, target)
                {
                    groups.entry(extents).or_default().push(shard);
                }
            }
            self.intersection_cache
                .insert(key.clone(), groups.into_iter().collect());
        }
        Ok(self.intersection_cache[&key]
            .iter()
            .map(|(extents, candidates)| {
                let selected = candidates
                    .iter()
                    .copied()
                    .find(|shard| self.shards[shard.index() as usize].tile == local_tile)
                    .unwrap_or(candidates[0]);
                (extents.clone(), selected)
            })
            .collect())
    }

    fn intersecting_shard_set(
        &self,
        sources: &[LowShardId],
        target: &[ShardExtent],
        local_tile: u16,
    ) -> Vec<(Vec<ShardExtent>, LowShardId)> {
        let mut groups = BTreeMap::<Vec<ShardExtent>, Vec<LowShardId>>::new();
        for &source in sources {
            if let Some(extents) =
                intersect_extents(&self.shards[source.index() as usize].extents, target)
            {
                groups.entry(extents).or_default().push(source);
            }
        }
        groups
            .into_iter()
            .map(|(extents, candidates)| {
                let selected = candidates
                    .iter()
                    .copied()
                    .find(|source| self.shards[source.index() as usize].tile == local_tile)
                    .unwrap_or(candidates[0]);
                (extents, selected)
            })
            .collect()
    }

    fn lower_region(
        &mut self,
        operations: &[MidOperation],
        checkpoints: bool,
    ) -> LowLoweringResult<Vec<TileWorkList>> {
        let mut tiles = (0..self.tile_count)
            .map(|tile| TileWorkList {
                tile,
                work: Vec::new(),
            })
            .collect::<Vec<_>>();
        let mut checkpoint = 0u8;
        for (index, operation) in operations.iter().enumerate() {
            let started = Instant::now();
            if self.lower_deferred_output(operation, &mut tiles)? {
                tracing::info!(
                    operation = index,
                    source = ?operation.source.map(OperationId::index),
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    "deferred fused-operator materialization"
                );
                continue;
            }
            let lowered = match &operation.kind {
                MidOperationKind::Repeat(repeat) => {
                    self.lower_repeat(operation, repeat, &mut tiles)
                }
                MidOperationKind::Operator { plan, .. } => {
                    self.lower_operator(operation, plan, &mut tiles)
                }
                MidOperationKind::Convert(plan) => {
                    self.lower_conversion(operation, plan, &mut tiles)
                }
            };
            if let Err(error) = lowered {
                tracing::error!(
                    operation = index,
                    source = ?operation.source.map(OperationId::index),
                    kind = ?operation.kind,
                    inputs = ?operation.inputs,
                    results = ?operation.results,
                    ?error,
                    "failed to lower mid operation to tile work"
                );
                return Err(error);
            }
            if checkpoints
                && matches!(
                    operation.kind,
                    MidOperationKind::Operator { .. } | MidOperationKind::Repeat(_)
                )
                && let Some(source) = operation.source
            {
                append_checkpoint(&mut tiles, source, checkpoint);
                checkpoint ^= 1;
            }
            tracing::info!(
                operation = index,
                source = ?operation.source.map(OperationId::index),
                elapsed_ms = started.elapsed().as_millis() as u64,
                shards = self.shards.len(),
                exchange_phases = self.phases.len(),
                "lowered mid operation to tile work"
            );
        }
        Ok(tiles)
    }

    fn lower_operator(
        &mut self,
        operation: &MidOperation,
        plan: &crate::OperatorPlan,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        match &plan.dispatch {
            OperatorDispatch::Pointwise {
                kernel,
                input_mapping,
            } => self.lower_pointwise(
                operation,
                kernel.clone(),
                *input_mapping,
                &plan.requirements,
                tiles,
            ),
            OperatorDispatch::BlockedGemm {
                inner_block,
                output_column_block,
                distribution,
                orientation,
            } => {
                let crate::MidOperator::Gemm {
                    multiply,
                    accumulate,
                    ..
                } = plan.operator
                else {
                    return Err(LowLoweringError::InvalidOperatorPlan);
                };
                let kernel = |mode| TileKernelSpec::Gemm {
                    multiply,
                    accumulate,
                    mode,
                    weights: crate::GemmWeightLoad::Standard,
                    inner_block: *inner_block,
                    output_columns: *output_column_block,
                };
                self.lower_blocked_gemm(
                    operation,
                    kernel(crate::GemmKernelMode::Initialize),
                    kernel(crate::GemmKernelMode::Accumulate),
                    *inner_block,
                    *output_column_block,
                    *orientation,
                    *distribution,
                    &plan.requirements,
                    tiles,
                )
            }
            OperatorDispatch::BlockedAttention {
                query_key,
                probability_value,
                query_block_rows,
                key_block_rows,
                padded_query_dimension,
                padded_value_dimension,
            } => self.lower_blocked_attention(
                operation,
                query_key.clone(),
                probability_value.clone(),
                *query_block_rows,
                *key_block_rows,
                *padded_query_dimension,
                *padded_value_dimension,
                &plan.requirements,
                tiles,
            ),
            OperatorDispatch::MaterializedAttention {
                query_key,
                probability_value,
                query_block_rows,
                padded_key_rows,
                padded_query_dimension,
                padded_value_dimension,
            } => self.lower_materialized_attention(
                operation,
                query_key.clone(),
                probability_value.clone(),
                *query_block_rows,
                *padded_key_rows,
                *padded_query_dimension,
                *padded_value_dimension,
                &plan.requirements,
                tiles,
            ),
            OperatorDispatch::View => self.lower_view(operation, &plan.operator, tiles),
        }
    }

    fn shard_has_padding(&self, shard: LowShardId) -> bool {
        self.shards[shard.index() as usize]
            .extents
            .iter()
            .any(|extent| extent.logical_end < extent.physical_end)
    }

    fn append_fill_zero(
        &mut self,
        tiles: &mut [TileWorkList],
        shard: LowShardId,
        provenance: WorkProvenance,
    ) -> LowLoweringResult<()> {
        let shard_data = &self.shards[shard.index() as usize];
        let tile = shard_data.tile;
        let output = OperandRequirement::new(shard_data.tensor_type.format.clone(), 8);
        self.append_kernel(
            tiles,
            tile,
            KernelRun::new(
                provenance,
                TileKernel::Planned(TileKernelSpec::FillZero),
                Vec::new(),
                self.full_view(shard),
                KernelRequirements::Operator(OperatorRequirements {
                    inputs: Vec::new(),
                    output,
                    output_aliasing: crate::OutputAliasing::Fresh,
                    distinct_elements: Vec::new(),
                }),
            ),
        )
    }

    fn lower_pointwise(
        &mut self,
        operation: &MidOperation,
        kernel: TileKernelSpec,
        input_mapping: PointwiseInputMapping,
        requirements: &OperatorRequirements,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [result] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        let outputs = self.value_shards(*result)?.to_vec();
        let mut wave_transfers = Vec::<BTreeMap<ShardView, Vec<ShardView>>>::new();
        let mut wave_runs = Vec::<Vec<(u16, KernelRun)>>::new();
        for output in outputs {
            if self.shards[output.index() as usize]
                .extents
                .iter()
                .any(|extent| extent.start == extent.physical_end)
            {
                continue;
            }
            let tile = self.shards[output.index() as usize].tile;
            let sources = operation
                .inputs
                .iter()
                .map(|input| {
                    Ok(match input_mapping {
                        PointwiseInputMapping::BroadcastToOutput => self
                            .value_shards(*input)?
                            .iter()
                            .find_map(|source| self.broadcast_view(*source, output))
                            .ok_or(LowLoweringError::InvalidOperatorPlan)?,
                        PointwiseInputMapping::TileLocal => {
                            let output_extents = &self.shards[output.index() as usize].extents;
                            let source = self
                                .value_shards(*input)?
                                .iter()
                                .copied()
                                .find(|source| {
                                    let source = &self.shards[source.index() as usize];
                                    source.tile == tile && source.extents == *output_extents
                                })
                                .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                            self.full_view(source)
                        }
                    })
                })
                .collect::<LowLoweringResult<Vec<_>>>()?;
            let chunks = vec![self.shards[output.index() as usize].extents.clone()];
            for (wave, output_extents) in chunks.into_iter().enumerate() {
                if wave_transfers.len() <= wave {
                    wave_transfers.push(BTreeMap::new());
                    wave_runs.push(Vec::new());
                }
                let inputs = sources
                    .iter()
                    .map(|source| {
                        let source_view = match input_mapping {
                            PointwiseInputMapping::BroadcastToOutput => self
                                .broadcast_view_for_extents(source.shard, output, &output_extents)
                                .ok_or(LowLoweringError::InvalidOperatorPlan)?,
                            PointwiseInputMapping::TileLocal => source.clone(),
                        };
                        let view = if self.shards[source_view.shard.index() as usize].tile == tile {
                            source_view
                        } else {
                            let copy = self.push_shard(LowShard {
                                id: LowShardId(0),
                                tile,
                                tensor_type: self.shards[source_view.shard.index() as usize]
                                    .tensor_type
                                    .clone(),
                                extents: source_view.extents.clone(),
                                definition: ShardDefinition::ExchangeStaging,
                            })?;
                            wave_transfers[wave]
                                .entry(source_view)
                                .or_default()
                                .push(self.full_view(copy));
                            self.full_view(copy)
                        };
                        Ok(KernelOperand { views: vec![view] })
                    })
                    .collect::<LowLoweringResult<_>>()?;
                wave_runs[wave].push((
                    tile,
                    KernelRun::new(
                        WorkProvenance {
                            operation: operation.source,
                            value: operation.results.first().copied(),
                            reason: WorkReason::OperatorKernel,
                        },
                        TileKernel::Planned(kernel.clone()),
                        inputs,
                        ShardView {
                            shard: output,
                            extents: output_extents,
                        },
                        KernelRequirements::Operator(requirements.clone()),
                    ),
                ));
            }
        }
        for (transfers, runs) in wave_transfers.into_iter().zip(wave_runs) {
            self.append_phase(
                transfers,
                WorkProvenance {
                    operation: operation.source,
                    value: None,
                    reason: WorkReason::OperatorInputs,
                },
                tiles,
            )?;
            for (tile, run) in runs {
                self.append_kernel(tiles, tile, run)?;
            }
        }
        Ok(())
    }

    fn broadcast_view(&self, source: LowShardId, output: LowShardId) -> Option<ShardView> {
        self.broadcast_view_for_extents(
            source,
            output,
            &self.shards[output.index() as usize].extents,
        )
    }

    fn broadcast_view_for_extents(
        &self,
        source: LowShardId,
        output: LowShardId,
        output_extents: &[ShardExtent],
    ) -> Option<ShardView> {
        let source_shard = &self.shards[source.index() as usize];
        let output_shard = &self.shards[output.index() as usize];
        let source_rank = source_shard.extents.len();
        let output_rank = output_shard.extents.len();
        if source_rank > output_rank {
            return None;
        }
        let offset = output_rank - source_rank;
        let mut extents = source_shard.extents.clone();
        for (axis, extent) in extents.iter_mut().enumerate() {
            let dimension = source_shard.tensor_type.shape.0[axis];
            if dimension == 1 {
                if extent.start != 0 || extent.logical_end == 0 {
                    return None;
                }
                extent.start = 0;
                extent.logical_end = 1;
                extent.physical_end = 1;
            } else {
                let required = output_extents[offset + axis];
                if extent.start > required.start || extent.logical_end < required.logical_end {
                    return None;
                }
                extent.start = required.start;
                extent.logical_end = required.logical_end;
                extent.physical_end = required.logical_end;
            }
        }
        Some(ShardView {
            shard: source,
            extents,
        })
    }

    fn dispatch_input_view(
        &mut self,
        value: MidValueId,
        tile: u16,
        ranges: &[(usize, u32, u32)],
        transfers: &mut BTreeMap<ShardView, Vec<ShardView>>,
        local_copies: &mut Vec<(u16, LocalCopy)>,
    ) -> LowLoweringResult<ShardView> {
        let target = self.local_shard(value, tile)?;
        let target_view = self.narrow_view(target, ranges)?;
        let Some(source_value) = self.deferred_conversions.get(&value).copied() else {
            return Ok(target_view);
        };

        let staging = self.push_shard(LowShard {
            id: LowShardId(0),
            tile,
            tensor_type: self.shards[target.index() as usize].tensor_type.clone(),
            extents: target_view.extents.clone(),
            definition: ShardDefinition::ExchangeStaging,
        })?;
        let intersections = self.intersecting_shards(source_value, &target_view.extents, tile)?;
        if intersections.is_empty() {
            return Err(LowLoweringError::InvalidConversionPlan);
        }
        for (extents, source) in intersections {
            let source_view = ShardView {
                shard: source,
                extents: extents.clone(),
            };
            let destination_view = ShardView {
                shard: staging,
                extents,
            };
            if self.shards[source.index() as usize].tile == tile {
                append_logical_span_copies(
                    &self.shards,
                    &source_view,
                    &destination_view,
                    tile,
                    local_copies,
                )?;
            } else {
                transfers
                    .entry(source_view)
                    .or_default()
                    .push(destination_view);
            }
        }
        Ok(self.full_view(staging))
    }

    fn append_phase(
        &mut self,
        transfers: BTreeMap<ShardView, Vec<ShardView>>,
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        self.append_ordered_phase(transfers, provenance, ExchangeOrder::Semantic, tiles)
    }

    fn append_physical_phase(
        &mut self,
        transfers: BTreeMap<ShardView, Vec<ShardView>>,
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        self.append_ordered_phase(transfers, provenance, ExchangeOrder::Physical, tiles)
    }

    fn append_ordered_phase(
        &mut self,
        transfers: BTreeMap<ShardView, Vec<ShardView>>,
        provenance: WorkProvenance,
        order: ExchangeOrder,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let transfers = transfers
            .into_iter()
            .map(|(source, mut destinations)| {
                destinations.sort_unstable();
                destinations.dedup();
                LogicalExchange {
                    source,
                    destinations,
                    order,
                }
            })
            .collect::<Vec<_>>();
        self.append_exchange_phase(transfers, provenance, tiles)
    }

    fn append_mixed_phase(
        &mut self,
        semantic: BTreeMap<ShardView, Vec<ShardView>>,
        physical: BTreeMap<ShardView, Vec<ShardView>>,
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let mut transfers = Vec::with_capacity(semantic.len().saturating_add(physical.len()));
        for (order, mappings) in [
            (ExchangeOrder::Semantic, semantic),
            (ExchangeOrder::Physical, physical),
        ] {
            transfers.extend(mappings.into_iter().map(|(source, mut destinations)| {
                destinations.sort_unstable();
                destinations.dedup();
                LogicalExchange {
                    source,
                    destinations,
                    order,
                }
            }));
        }
        self.append_exchange_phase(transfers, provenance, tiles)
    }

    fn append_exchange_phase(
        &mut self,
        mut transfers: Vec<LogicalExchange>,
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        if transfers.is_empty() {
            return Ok(());
        }
        if let Some(previous) = self.phases.last().map(|phase| phase.id)
            && self.phases[previous.index() as usize]
                .provenance
                .operation
                .is_some()
            && self.phases[previous.index() as usize].provenance.operation == provenance.operation
        {
            let touched = transfers
                .iter()
                .flat_map(|transfer| {
                    std::iter::once(transfer.source.shard)
                        .chain(transfer.destinations.iter().map(|view| view.shard))
                })
                .map(|shard| self.storage_root(shard))
                .collect::<BTreeSet<_>>();
            let previous_touched = self.phases[previous.index() as usize]
                .transfers
                .iter()
                .flat_map(|transfer| {
                    std::iter::once(transfer.source.shard)
                        .chain(transfer.destinations.iter().map(|view| view.shard))
                })
                .map(|shard| self.storage_root(shard))
                .collect::<BTreeSet<_>>();
            let disjoint_transfers = touched.is_disjoint(&previous_touched);
            let only_independent_copies_between = tiles.iter().all(|tile| {
                let Some(boundary) = tile
                    .work
                    .iter()
                    .rposition(|work| *work == TileWork::Exchange(previous))
                else {
                    return false;
                };
                tile.work[boundary + 1..].iter().all(|work| {
                    let TileWork::LocalCopy(copy) = *work else {
                        return false;
                    };
                    let copy = &self.local_copies[copy.0 as usize];
                    !touched.contains(&self.storage_root(copy.source))
                        && !touched.contains(&self.storage_root(copy.destination))
                })
            });
            if disjoint_transfers && only_independent_copies_between {
                let phase = &mut self.phases[previous.index() as usize];
                phase.transfers.append(&mut transfers);
                if phase.provenance != provenance {
                    phase.provenance = WorkProvenance {
                        operation: provenance.operation,
                        value: None,
                        reason: WorkReason::OperatorInputs,
                    };
                }
                tracing::debug!(
                    phase = previous.index(),
                    operation = ?provenance.operation.map(OperationId::index),
                    "consolidated independent exchange transfers"
                );
                return Ok(());
            }
        }
        let id = ExchangePhaseId(
            u32::try_from(self.phases.len()).map_err(|_| LowLoweringError::IdOverflow)?,
        );
        self.phases.push(ExchangePhase {
            id,
            provenance,
            transfers,
        });
        tracing::debug!(
            phase = id.index(),
            operation = ?provenance.operation.map(OperationId::index),
            value = ?provenance.value.map(MidValueId::index),
            reason = ?provenance.reason,
            "scheduled exchange phase"
        );
        for tile in tiles {
            tile.work.push(TileWork::Exchange(id));
        }
        Ok(())
    }

    fn append_kernel(
        &mut self,
        tiles: &mut [TileWorkList],
        tile: u16,
        run: KernelRun,
    ) -> LowLoweringResult<()> {
        let output_flattens_outer_rows = self
            .shards
            .get(run.output.shard.index() as usize)
            .is_some_and(|shard| {
                matches!(
                    shard.tensor_type.format.layout.order,
                    ElementOrder::Amp(AmpOrder::Left | AmpOrder::Output)
                )
            });
        if matches!(run.kernel, TileKernel::Planned(TileKernelSpec::Gemm { .. }))
            && run.output.extents.len() > 2
            && !output_flattens_outer_rows
        {
            let matrix_axes = run.output.extents.len() - 2;
            let mut coordinates = vec![0; matrix_axes];
            let mut matrix_runs = Vec::new();
            split_gemm_matrices(&run, 0, &mut coordinates, &mut matrix_runs)?;
            if matrix_runs.len() > 1 {
                for matrix_run in matrix_runs {
                    self.append_single_kernel(tiles, tile, matrix_run)?;
                }
                return Ok(());
            }
        }
        self.append_single_kernel(tiles, tile, run)
    }

    fn append_single_kernel(
        &mut self,
        tiles: &mut [TileWorkList],
        tile: u16,
        mut run: KernelRun,
    ) -> LowLoweringResult<()> {
        if let Some(metadata) = self
            .kernel_metadata
            .iter()
            .find(|metadata| metadata.as_ref() == run.metadata.as_ref())
        {
            run.metadata = Arc::clone(metadata);
        } else {
            self.kernel_metadata.push(Arc::clone(&run.metadata));
        }
        let id = KernelRunId(
            u32::try_from(self.kernel_runs.len()).map_err(|_| LowLoweringError::IdOverflow)?,
        );
        self.kernel_runs.push(run);
        tiles[usize::from(tile)].work.push(TileWork::Kernel(id));
        Ok(())
    }

    fn append_local_copy(
        &mut self,
        tiles: &mut [TileWorkList],
        tile: u16,
        copy: LocalCopy,
    ) -> LowLoweringResult<()> {
        let id = LocalCopyId(
            u32::try_from(self.local_copies.len()).map_err(|_| LowLoweringError::IdOverflow)?,
        );
        self.local_copies.push(copy);
        tiles[usize::from(tile)].work.push(TileWork::LocalCopy(id));
        Ok(())
    }

    fn append_repeat(
        &mut self,
        tiles: &mut [TileWorkList],
        tile: u16,
        repeat: RepeatRun,
    ) -> LowLoweringResult<()> {
        let id = RepeatRunId(
            u32::try_from(self.repeat_runs.len()).map_err(|_| LowLoweringError::IdOverflow)?,
        );
        self.repeat_runs.push(repeat);
        tiles[usize::from(tile)].work.push(TileWork::Repeat(id));
        Ok(())
    }

    fn full_view(&self, shard: LowShardId) -> ShardView {
        ShardView {
            shard,
            extents: self.shards[shard.index() as usize].extents.clone(),
        }
    }

    fn narrow_view(
        &self,
        shard: LowShardId,
        ranges: &[(usize, u32, u32)],
    ) -> LowLoweringResult<ShardView> {
        let mut view = self.full_view(shard);
        for &(axis, start, end) in ranges {
            let extent = view
                .extents
                .get_mut(axis)
                .ok_or(LowLoweringError::InvalidOperatorPlan)?;
            if start < extent.start || end > extent.physical_end || start >= end {
                return Err(LowLoweringError::InvalidOperatorPlan);
            }
            extent.start = start;
            extent.physical_end = end;
            extent.logical_end = end.min(extent.logical_end).max(start);
        }
        Ok(view)
    }

    fn lower_repeat(
        &mut self,
        operation: &MidOperation,
        repeat: &MidRepeat,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let expected_inputs = repeat.carried_inputs + repeat.invariant_inputs;
        let expected_arguments = expected_inputs + repeat.iterated_inputs.len();
        if operation.inputs.len() != expected_inputs
            || operation.results.len() != repeat.carried_inputs
            || repeat.body.arguments.len() != expected_arguments
            || repeat.body.yields.len() != repeat.carried_inputs
            || repeat
                .iterated_inputs
                .iter()
                .any(|values| values.len() != repeat.count as usize)
        {
            return Err(LowLoweringError::InvalidRepeat);
        }
        for index in 0..repeat.carried_inputs {
            if !repeat_yield_can_alias(
                repeat.body.yields[index],
                repeat.body.arguments[index],
                &repeat.body.operations,
            ) {
                return Err(LowLoweringError::RepeatRequiresInPlace(index));
            }
        }
        let iterated_requirements = repeat
            .iterated_inputs
            .iter()
            .enumerate()
            .map(|(index, _)| {
                body_storage_requirement(
                    repeat.body.arguments[expected_inputs + index],
                    &repeat.body.operations,
                )
            })
            .collect::<Vec<_>>();
        let body = self.lower_region(&repeat.body.operations, false)?;
        for tile in 0..self.tile_count {
            let mut carried = Vec::with_capacity(repeat.carried_inputs);
            for index in 0..repeat.carried_inputs {
                let Some(argument) = self.find_local_shard(repeat.body.arguments[index], tile)?
                else {
                    continue;
                };
                let initial = self.corresponding_shard(operation.inputs[index], argument)?;
                let yielded = self.corresponding_shard(repeat.body.yields[index], argument)?;
                let result = self.corresponding_shard(operation.results[index], argument)?;
                self.alias_shard(argument, initial);
                if yielded != argument {
                    self.shards[yielded.index() as usize].definition =
                        ShardDefinition::WritableAlias(argument);
                }
                self.alias_shard(result, initial);
                carried.push(RepeatCarried {
                    initial,
                    argument,
                    yielded,
                    result,
                });
            }
            let invariants = (0..repeat.invariant_inputs)
                .filter_map(|index| {
                    let input_index = repeat.carried_inputs + index;
                    let argument =
                        match self.find_local_shard(repeat.body.arguments[input_index], tile) {
                            Ok(Some(argument)) => argument,
                            Ok(None) => return None,
                            Err(error) => return Some(Err(error)),
                        };
                    Some(
                        self.corresponding_shard(operation.inputs[input_index], argument)
                            .map(|input| RepeatInvariant { input, argument }),
                    )
                })
                .collect::<LowLoweringResult<_>>()?;
            let iterated = repeat
                .iterated_inputs
                .iter()
                .enumerate()
                .filter_map(|(index, values)| {
                    let argument = match self
                        .find_local_shard(repeat.body.arguments[expected_inputs + index], tile)
                    {
                        Ok(Some(argument)) => argument,
                        Ok(None) => return None,
                        Err(error) => return Some(Err(error)),
                    };
                    let inputs = values
                        .iter()
                        .map(|value| self.corresponding_shard(*value, argument))
                        .collect::<LowLoweringResult<Vec<_>>>();
                    let inputs = match inputs {
                        Ok(inputs) => inputs,
                        Err(error) => return Some(Err(error)),
                    };
                    let (alignment, access_tail) = iterated_requirements[index];
                    let strides = inputs
                        .iter()
                        .map(|shard| self.shard_stride(*shard, alignment, access_tail))
                        .collect::<LowLoweringResult<Vec<_>>>();
                    let strides = match strides {
                        Ok(strides) => strides,
                        Err(error) => return Some(Err(error)),
                    };
                    let Some(&stride_bytes) = strides.first() else {
                        return Some(Err(LowLoweringError::InvalidIteratedBlocks(index)));
                    };
                    if strides.iter().any(|stride| *stride != stride_bytes) {
                        return Some(Err(LowLoweringError::InvalidIteratedBlocks(index)));
                    }
                    Some(Ok(RepeatIterated {
                        inputs,
                        argument,
                        stride_bytes,
                        alignment,
                    }))
                })
                .collect::<LowLoweringResult<_>>()?;
            self.append_repeat(
                tiles,
                tile,
                RepeatRun {
                    provenance: WorkProvenance {
                        operation: operation.source,
                        value: operation.results.first().copied(),
                        reason: WorkReason::Repeat,
                    },
                    count: repeat.count,
                    carried,
                    invariants,
                    iterated,
                    body: Box::new(body[usize::from(tile)].clone()),
                },
            )?;
        }
        Ok(())
    }

    fn alias_shard(&mut self, shard: LowShardId, target: LowShardId) {
        self.shards[shard.index() as usize].definition = ShardDefinition::Alias(target);
    }

    fn find_local_shard(
        &self,
        value: MidValueId,
        tile: u16,
    ) -> LowLoweringResult<Option<LowShardId>> {
        Ok(self
            .value_shards(value)?
            .iter()
            .copied()
            .find(|shard| self.shards[shard.index() as usize].tile == tile))
    }

    fn corresponding_shard(
        &self,
        value: MidValueId,
        target: LowShardId,
    ) -> LowLoweringResult<LowShardId> {
        let target = &self.shards[target.index() as usize];
        self.value_shards(value)?
            .iter()
            .copied()
            .filter(|shard| self.shards[shard.index() as usize].extents == target.extents)
            .min_by_key(|shard| u8::from(self.shards[shard.index() as usize].tile != target.tile))
            .ok_or(LowLoweringError::UnknownValue(value))
    }

    fn shard_stride(
        &self,
        shard: LowShardId,
        alignment: u32,
        access_tail: u32,
    ) -> LowLoweringResult<u32> {
        let shard = &self.shards[shard.index() as usize];
        let elements = shard
            .extents
            .iter()
            .try_fold(1_u64, |elements, extent| {
                elements.checked_mul(u64::from(extent.physical_end - extent.start))
            })
            .ok_or(LowLoweringError::IdOverflow)?;
        let bytes = elements
            .checked_mul(shard.tensor_type.format.precision.bytes())
            .and_then(|bytes| bytes.checked_add(u64::from(access_tail)))
            .ok_or(LowLoweringError::IdOverflow)?;
        let alignment = u64::from(alignment.max(1));
        let stride = bytes
            .checked_add(alignment - 1)
            .map(|bytes| bytes / alignment * alignment)
            .ok_or(LowLoweringError::IdOverflow)?;
        u32::try_from(stride).map_err(|_| LowLoweringError::IdOverflow)
    }
}

fn value_can_alias(value: MidValueId, target: MidValueId, operations: &[MidOperation]) -> bool {
    if value == target {
        return true;
    }
    let Some(operation) = operations
        .iter()
        .find(|operation| operation.results.contains(&value))
    else {
        return false;
    };
    let Some(plan) = operation.operator_plan() else {
        return false;
    };
    let indices = match &plan.requirements.output_aliasing {
        OutputAliasing::Fresh => return false,
        OutputAliasing::MayAliasInputs(indices) => indices.as_slice(),
    };
    indices.iter().any(|index| {
        operation
            .inputs
            .get(usize::from(*index))
            .is_some_and(|input| value_can_alias(*input, target, operations))
    })
}

fn repeat_yield_can_alias(
    value: MidValueId,
    carried: MidValueId,
    operations: &[MidOperation],
) -> bool {
    if value_can_alias(value, carried, operations) {
        return true;
    }
    let Some(definition) = operations
        .iter()
        .position(|operation| operation.results.contains(&value))
    else {
        return false;
    };
    // A repeat reuses the carried allocation on its next iteration. A fresh
    // yield may overwrite it when every read of the previous iteration's
    // value has completed before the yielding operation begins.
    !operations[definition..]
        .iter()
        .any(|operation| operation.inputs.contains(&carried))
}

fn body_storage_requirement(value: MidValueId, operations: &[MidOperation]) -> (u32, u32) {
    let mut alignment = 8;
    let mut access_tail = 0;
    for operation in operations {
        for (index, input) in operation.inputs.iter().enumerate() {
            if *input != value {
                continue;
            }
            let requirement = operation
                .operator_plan()
                .and_then(|plan| plan.requirements.inputs.get(index))
                .or_else(|| operation.conversion_plan().map(|plan| &plan.input));
            if let Some(requirement) = requirement {
                alignment = alignment.max(requirement.alignment);
                access_tail = access_tail.max(requirement.access_tail_bytes);
            }
        }
    }
    (alignment, access_tail)
}

fn operation_provenance(operation: &MidOperation) -> WorkProvenance {
    WorkProvenance {
        operation: operation.source,
        value: operation.results.first().copied(),
        reason: match &operation.kind {
            MidOperationKind::Convert(plan)
                if plan.input.format.precision != plan.output.format.precision =>
            {
                WorkReason::PrecisionCast
            }
            MidOperationKind::Convert(_) => WorkReason::LayoutRearrangement,
            MidOperationKind::Operator { .. } => WorkReason::OperatorKernel,
            MidOperationKind::Repeat(_) => WorkReason::Repeat,
        },
    }
}

#[cfg(test)]
mod tests;
