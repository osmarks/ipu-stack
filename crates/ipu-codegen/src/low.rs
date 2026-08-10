//! Logical per-tile schedule produced from the layout-aware mid-level IR.
//!
//! Tensor shards have tile identities and rectangular physical extents, and
//! work is ordered per tile. Exchanges still refer to logical shards rather
//! than SRAM addresses; kernel runs still name a selected kernel kind rather
//! than a linked symbol. Placement and final code generation resolve those
//! remaining choices.

use crate::graph::{GraphInputKind, OperationId};
use crate::mid::{
    ConversionDispatch, LayoutError, MidGraph, MidOperation, MidOperationKind, MidRepeat,
    MidValueId, OperandRequirement, OperatorDispatch, OperatorRequirements, OutputAliasing,
    PipelineConfig, PointwiseInputMapping, TensorType, TileKernelSpec,
};
use crate::storage::{ByteSpan, StorageError, logical_view_byte_spans, view_byte_spans};
use std::collections::BTreeMap;
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
    pub const fn index(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelRunId(u32);

impl KernelRunId {
    pub const fn index(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalCopyId(u32);

impl LocalCopyId {
    pub const fn index(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RepeatRunId(u32);

impl RepeatRunId {
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Half-open bounds along one tensor axis. `physical_end` includes any zero
/// padding while `logical_end` never exceeds the semantic tensor shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShardExtent {
    pub axis: u16,
    pub start: u32,
    pub logical_end: u32,
    pub physical_end: u32,
}

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalCopy {
    pub source: LowShardId,
    pub source_offset: u32,
    pub destination: LowShardId,
    pub destination_offset: u32,
    pub bytes: u32,
}

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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileWorkRef<'a> {
    Exchange(ExchangePhaseId),
    LocalCopy(&'a LocalCopy),
    Kernel(&'a KernelRun),
    Repeat(&'a RepeatRun),
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
    #[error("operator operation is missing its selected whole-device plan")]
    MissingOperatorPlan,
    #[error("operator plan is incompatible with its values or block dimensions")]
    InvalidOperatorPlan,
    #[error("conversion operation is missing its selected plan")]
    MissingConversionPlan,
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

/// Produces a logical per-tile schedule by expanding selected operator plans.
/// Conversions without plans still use a conservative gather fallback.
#[tracing::instrument(
    name = "ipu_codegen.low.lower_to_tiles",
    skip(graph, config),
    fields(
        tile_count = config.tile_count,
        operations = graph.operations.len(),
        profiling = config.profiling.enabled
    )
)]
pub fn lower_to_tiles(graph: &MidGraph, config: &PipelineConfig) -> LowLoweringResult<LowProgram> {
    if config.tile_count == 0 {
        return Err(LowLoweringError::EmptyTileGroup);
    }
    let mut state = LoweringState::new(graph, config.tile_count)?;
    let tiles = state.lower_region(&graph.operations, &graph.outputs)?;
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
    intersection_cache: BTreeMap<(MidValueId, Vec<ShardExtent>), ShardIntersections>,
}

impl LoweringState {
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
            intersection_cache: BTreeMap::new(),
        };
        for value in &graph.values {
            let declared_tiles = value.tensor_type.format.layout.tiling.tile_count;
            if declared_tiles == 0 || declared_tiles > tile_count {
                return Err(LowLoweringError::TileCountMismatch {
                    value: value.id,
                    declared: declared_tiles,
                    scheduled: tile_count,
                });
            }
            let extents = shard_extents(&value.tensor_type)?;
            let mut value_shards = Vec::with_capacity(usize::from(tile_count));
            for (tile, extents) in extents.into_iter().enumerate() {
                let id = state.push_shard(LowShard {
                    id: LowShardId(0),
                    tile: u16::try_from(tile).map_err(|_| LowLoweringError::IdOverflow)?,
                    tensor_type: value.tensor_type.clone(),
                    extents,
                    definition: ShardDefinition::Value(value.id),
                })?;
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
                        ShardDefinition::Alias(_) | ShardDefinition::ExchangeStaging
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

    fn right_shard_for_block(
        &self,
        right_shards: &[LowShardId],
        tile: u16,
        column_start: u32,
        column_end: u32,
        inner_start: u32,
        inner_end: u32,
    ) -> Option<LowShardId> {
        self.right_shards_for_block(
            right_shards,
            column_start,
            column_end,
            inner_start,
            inner_end,
        )
        .min_by_key(|shard| u8::from(self.shards[shard.index() as usize].tile != tile))
    }

    fn right_shards_for_block<'a>(
        &'a self,
        right_shards: &'a [LowShardId],
        column_start: u32,
        column_end: u32,
        inner_start: u32,
        inner_end: u32,
    ) -> impl Iterator<Item = LowShardId> + 'a {
        right_shards.iter().copied().filter(move |shard| {
            let extents = &self.shards[shard.index() as usize].extents;
            let columns = extents[extents.len() - 1];
            let inner = extents[extents.len() - 2];
            columns.start <= column_start
                && columns.physical_end >= column_end
                && inner.start <= inner_start
                && inner.physical_end >= inner_end
        })
    }

    fn prefer_local_shard(&self, shards: &[LowShardId], tile: u16) -> Option<LowShardId> {
        shards
            .iter()
            .copied()
            .min_by_key(|shard| u8::from(self.shards[shard.index() as usize].tile != tile))
    }

    fn use_uniform_interleaved_gemm_staging(
        &self,
        output_shards: &[LowShardId],
        right_shards: &[LowShardId],
        columns: Range<u32>,
        inner: Range<u32>,
        access_tail: u32,
    ) -> LowLoweringResult<bool> {
        let mut candidates = Vec::with_capacity(output_shards.len());
        for output in output_shards {
            let tile = self.shards[output.index() as usize].tile;
            let Some(right) = self.right_shard_for_block(
                right_shards,
                tile,
                columns.start,
                columns.end,
                inner.start,
                inner.end,
            ) else {
                return Ok(false);
            };
            let shard = &self.shards[right.index() as usize];
            if shard.tile != tile || shard.tensor_type.format.precision != crate::Precision::F16 {
                return Ok(false);
            }
            let rank = shard.extents.len();
            if rank < 2 {
                return Ok(false);
            }
            let view = self.narrow_view(
                right,
                &[
                    (rank - 2, inner.start, inner.end),
                    (rank - 1, columns.start, columns.end),
                ],
            )?;
            let spans = view_byte_spans(shard, &view)?;
            if spans.len() <= 1 {
                return Ok(false);
            }
            let bytes = spans.iter().try_fold(0u32, |total, span| {
                total
                    .checked_add(span.bytes)
                    .ok_or(LowLoweringError::IdOverflow)
            })?;
            candidates.push((tile, bytes));
        }
        for (tile, bytes) in candidates {
            if !self.interleaved_capacity_available(tile, bytes, access_tail)? {
                return Ok(false);
            }
        }
        Ok(true)
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

    fn lower_region(
        &mut self,
        operations: &[MidOperation],
        retained_values: &[MidValueId],
    ) -> LowLoweringResult<Vec<TileWorkList>> {
        let mut tiles = (0..self.tile_count)
            .map(|tile| TileWorkList {
                tile,
                work: Vec::new(),
            })
            .collect::<Vec<_>>();
        for (index, operation) in operations.iter().enumerate() {
            let started = Instant::now();
            if self.defer_conversion(
                operation,
                operations.get(index + 1),
                operations,
                retained_values,
            )? {
                tracing::info!(
                    operation = index,
                    source = ?operation.source.map(OperationId::index),
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    "deferred low-level conversion"
                );
                continue;
            }
            match &operation.kind {
                MidOperationKind::Repeat(repeat) => {
                    self.lower_repeat(operation, repeat, &mut tiles)?;
                }
                MidOperationKind::Operator(_) => self.lower_operator(operation, &mut tiles)?,
                kind => self.lower_conversion(operation, kind, &mut tiles)?,
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

    fn defer_conversion(
        &mut self,
        operation: &MidOperation,
        next: Option<&MidOperation>,
        operations: &[MidOperation],
        retained_values: &[MidValueId],
    ) -> LowLoweringResult<bool> {
        let (MidOperationKind::Rearrange { .. }, Some(next)) = (&operation.kind, next) else {
            return Ok(false);
        };
        let Some(plan) = &operation.conversion_plan else {
            return Err(LowLoweringError::MissingConversionPlan);
        };
        if plan.dispatch != ConversionDispatch::Intersections {
            return Ok(false);
        }
        if plan.output.materialization != crate::OperandMaterialization::DispatchSlices {
            return Ok(false);
        }
        let ([source], [result]) = (operation.inputs.as_slice(), operation.results.as_slice())
        else {
            return Ok(false);
        };
        let uses = operations
            .iter()
            .flat_map(|operation| &operation.inputs)
            .chain(retained_values)
            .filter(|value| **value == *result)
            .count();
        if uses != 1 {
            return Ok(false);
        }
        let Some(input_index) = next.inputs.iter().position(|input| input == result) else {
            return Ok(false);
        };
        let streamable = next
            .operator_plan
            .as_ref()
            .and_then(|plan| plan.requirements.inputs.get(input_index))
            .is_some_and(|requirement| {
                requirement.materialization == crate::OperandMaterialization::DispatchSlices
            });
        if !streamable {
            return Ok(false);
        }
        self.deferred_conversions.insert(*result, *source);
        for shard in self.value_shards(*result)?.to_vec() {
            self.shards[shard.index() as usize].definition = ShardDefinition::Unmaterialized;
        }
        Ok(true)
    }

    fn lower_conversion(
        &mut self,
        operation: &MidOperation,
        kind: &MidOperationKind,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let plan = operation
            .conversion_plan
            .as_ref()
            .ok_or(LowLoweringError::MissingConversionPlan)?;
        match plan.dispatch {
            ConversionDispatch::Local => self.lower_local_conversion(operation, kind, plan, tiles),
            ConversionDispatch::Intersections => {
                self.lower_intersection_conversion(operation, kind, plan, tiles)
            }
        }
    }

    fn lower_local_conversion(
        &mut self,
        operation: &MidOperation,
        kind: &MidOperationKind,
        plan: &crate::ConversionPlan,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [input] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidConversionPlan);
        };
        let [result] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        for output in self.value_shards(*result)?.to_vec() {
            let tile = self.shards[output.index() as usize].tile;
            let input = self.local_shard(*input, tile)?;
            self.append_kernel(
                tiles,
                tile,
                KernelRun::new(
                    operation_provenance(operation, kind),
                    TileKernel::Planned(plan.kernel.clone()),
                    vec![KernelOperand {
                        views: vec![self.full_view(input)],
                    }],
                    self.full_view(output),
                    KernelRequirements::Conversion {
                        input: plan.input.clone(),
                        output: plan.output.clone(),
                    },
                ),
            )?;
        }
        Ok(())
    }

    fn lower_intersection_conversion(
        &mut self,
        operation: &MidOperation,
        kind: &MidOperationKind,
        plan: &crate::ConversionPlan,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [input] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidConversionPlan);
        };
        let [result] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        let inputs = self.value_shards(*input)?.to_vec();
        let outputs = self.value_shards(*result)?.to_vec();
        let direct_retile = plan.input.format.layout.order == plan.output.format.layout.order;
        let mut remote_items =
            Vec::<(ShardView, ShardView, u16, Vec<LocalCopy>, Option<KernelRun>)>::new();
        for output in outputs {
            let tile = self.shards[output.index() as usize].tile;
            let mut unique_intersections = BTreeMap::<Vec<ShardExtent>, LowShardId>::new();
            for source in &inputs {
                let Some(extents) = intersect_extents(
                    &self.shards[source.index() as usize].extents,
                    &self.shards[output.index() as usize].extents,
                ) else {
                    continue;
                };
                let selected = unique_intersections.entry(extents).or_insert(*source);
                if self.shards[source.index() as usize].tile == tile {
                    *selected = *source;
                }
            }
            for (extents, source) in unique_intersections {
                let remote = self.shards[source.index() as usize].tile != tile;
                for extents in [extents] {
                    let source_view = ShardView {
                        shard: source,
                        extents: extents.clone(),
                    };
                    let output_view = ShardView {
                        shard: output,
                        extents: extents.clone(),
                    };
                    let (resident, exchange) = if remote {
                        (
                            output_view.clone(),
                            Some((source_view.clone(), output_view.clone())),
                        )
                    } else {
                        (source_view.clone(), None)
                    };
                    let (local_copies, run) = if remote {
                        (Vec::new(), None)
                    } else {
                        let mut copies = Vec::new();
                        if direct_retile {
                            append_span_copies(
                                &self.shards,
                                &resident,
                                &output_view,
                                tile,
                                &mut copies,
                            )?;
                        } else {
                            append_logical_span_copies(
                                &self.shards,
                                &resident,
                                &output_view,
                                tile,
                                &mut copies,
                            )?;
                        }
                        (copies.into_iter().map(|(_, copy)| copy).collect(), None)
                    };
                    if let Some((source_view, destination_view)) = exchange {
                        remote_items.push((source_view, destination_view, tile, local_copies, run));
                    } else {
                        for copy in local_copies {
                            self.append_local_copy(tiles, tile, copy)?;
                        }
                        if let Some(run) = run {
                            self.append_kernel(tiles, tile, run)?;
                        }
                    }
                }
            }
        }
        let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
        let mut consumers = Vec::<(u16, Vec<LocalCopy>, Option<KernelRun>)>::new();
        for (source, destination, tile, copies, run) in remote_items {
            transfers.entry(source).or_default().push(destination);
            consumers.push((tile, copies, run));
        }
        self.flush_conversion_phase(
            &mut transfers,
            &mut consumers,
            operation_provenance(operation, kind),
            tiles,
        )?;
        Ok(())
    }

    fn flush_conversion_phase(
        &mut self,
        transfers: &mut BTreeMap<ShardView, Vec<ShardView>>,
        consumers: &mut Vec<(u16, Vec<LocalCopy>, Option<KernelRun>)>,
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        if transfers.is_empty() {
            return Ok(());
        }
        self.append_phase(std::mem::take(transfers), provenance, tiles)?;
        for (tile, copies, run) in consumers.drain(..) {
            for copy in copies {
                self.append_local_copy(tiles, tile, copy)?;
            }
            if let Some(run) = run {
                self.append_kernel(tiles, tile, run)?;
            }
        }
        Ok(())
    }

    fn lower_operator(
        &mut self,
        operation: &MidOperation,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let plan = operation
            .operator_plan
            .as_ref()
            .ok_or(LowLoweringError::MissingOperatorPlan)?;
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
                initialize,
                accumulate,
                inner_block,
                output_column_block,
            } => self.lower_blocked_gemm(
                operation,
                initialize.clone(),
                accumulate.clone(),
                *inner_block,
                *output_column_block,
                &plan.requirements,
                tiles,
            ),
        }
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
                            self.full_view(self.local_shard(*input, tile)?)
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

    #[allow(clippy::too_many_arguments)]
    fn lower_blocked_gemm(
        &mut self,
        operation: &MidOperation,
        initialize: TileKernelSpec,
        accumulate: TileKernelSpec,
        inner_block: u32,
        output_column_block: u32,
        requirements: &OperatorRequirements,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [left_value, right_value] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidOperatorPlan);
        };
        let [output_value] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        if requirements.inputs[1]
            .format
            .layout
            .tiling
            .axes
            .iter()
            .any(|axis| axis.axis == crate::TensorAxis::FromEnd(2) && axis.partitions > 1)
        {
            return self.lower_streamed_blocked_gemm(
                operation,
                initialize,
                accumulate,
                inner_block,
                output_column_block,
                requirements,
                tiles,
            );
        }
        if inner_block == 0 || output_column_block == 0 {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let left_shards = self.value_shards(*left_value)?.to_vec();
        let right_shards = self.value_shards(*right_value)?.to_vec();
        let output_shards = self.value_shards(*output_value)?.to_vec();
        let left_type = &self.shards[left_shards[0].index() as usize].tensor_type;
        let output_type = &self.shards[output_shards[0].index() as usize].tensor_type;
        let left_rank = left_type.shape.0.len();
        let output_rank = output_type.shape.0.len();
        if left_rank < 2 || output_rank < 2 {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let inner_extent = left_type.format.layout.padded_shape(&left_type.shape)?.0[left_rank - 1];
        let column_extent = output_type
            .format
            .layout
            .padded_shape(&output_type.shape)?
            .0[output_rank - 1];
        if !inner_extent.is_multiple_of(inner_block)
            || !column_extent.is_multiple_of(output_column_block)
        {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }

        let panels_per_phase = column_extent / output_column_block;
        let phase_column_width = output_column_block
            .checked_mul(panels_per_phase)
            .ok_or(LowLoweringError::IdOverflow)?;
        let mut local_right_staging = BTreeMap::<(u16, u32), LowShardId>::new();
        let mut remote_right_staging = vec![
            vec![
                None;
                usize::try_from(panels_per_phase)
                    .map_err(|_| LowLoweringError::IdOverflow)?
            ];
            usize::from(self.tile_count)
        ];
        for phase_column_start in (0..column_extent).step_by(phase_column_width as usize) {
            let phase_column_end = phase_column_start
                .saturating_add(phase_column_width)
                .min(column_extent);
            for inner_start in (0..inner_extent).step_by(inner_block as usize) {
                let inner_end = inner_start + inner_block;
                let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
                let mut local_copies = Vec::<(u16, LocalCopy)>::new();
                let mut runs = Vec::new();
                let mut left_views = BTreeMap::<u16, ShardView>::new();
                for column_start in
                    (phase_column_start..phase_column_end).step_by(output_column_block as usize)
                {
                    let column_end = column_start + output_column_block;
                    let right_candidates = self
                        .right_shards_for_block(
                            &right_shards,
                            column_start,
                            column_end,
                            inner_start,
                            inner_end,
                        )
                        .collect::<Vec<_>>();
                    if right_candidates.is_empty() {
                        return Err(LowLoweringError::InvalidOperatorPlan);
                    }
                    let column_outputs = output_shards
                        .iter()
                        .copied()
                        .filter(|output| {
                            let extents = &self.shards[output.index() as usize].extents;
                            let columns = extents[extents.len() - 1];
                            columns.start <= column_start && columns.physical_end >= column_end
                        })
                        .collect::<Vec<_>>();
                    if column_outputs.is_empty() {
                        return Err(LowLoweringError::InvalidOperatorPlan);
                    }
                    let use_interleaved_staging = self.use_uniform_interleaved_gemm_staging(
                        &column_outputs,
                        &right_shards,
                        column_start..column_end,
                        0..inner_block,
                        requirements.inputs[1].access_tail_bytes,
                    )?;
                    let staging_slot =
                        usize::try_from((column_start - phase_column_start) / output_column_block)
                            .map_err(|_| LowLoweringError::IdOverflow)?;
                    for output in &column_outputs {
                        let tile = self.shards[output.index() as usize].tile;
                        let left_view = if let Some(view) = left_views.get(&tile) {
                            view.clone()
                        } else {
                            let view = self.dispatch_input_view(
                                *left_value,
                                tile,
                                &[(left_rank - 1, inner_start, inner_end)],
                                &mut transfers,
                                &mut local_copies,
                            )?;
                            left_views.insert(tile, view.clone());
                            view
                        };
                        let right = self
                            .prefer_local_shard(&right_candidates, tile)
                            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                        let right_rank = self.shards[right.index() as usize].extents.len();
                        let right_view = self.narrow_view(
                            right,
                            &[
                                (right_rank - 2, inner_start, inner_end),
                                (right_rank - 1, column_start, column_end),
                            ],
                        )?;
                        let resident_right = if self.shards[right.index() as usize].tile == tile {
                            let spans =
                                view_byte_spans(&self.shards[right.index() as usize], &right_view)?;
                            if spans.len() == 1
                                && (!use_interleaved_staging
                                    || requirements.inputs[1].local_staging
                                        == crate::LocalOperandStaging::Direct)
                            {
                                right_view
                            } else {
                                let existing_staging =
                                    local_right_staging.get(&(tile, column_start)).copied();
                                let definition = existing_staging
                                    .map(ShardDefinition::Alias)
                                    .unwrap_or(ShardDefinition::LocalCopy(right));
                                let mut tensor_type =
                                    self.shards[right.index() as usize].tensor_type.clone();
                                if use_interleaved_staging {
                                    tensor_type.format.layout.memory_class =
                                        crate::MemoryClass::Ipu21Interleaved;
                                }
                                let copy = self.push_shard(LowShard {
                                    id: LowShardId(0),
                                    tile,
                                    tensor_type,
                                    extents: right_view.extents.clone(),
                                    definition,
                                })?;
                                local_right_staging
                                    .entry((tile, column_start))
                                    .or_insert(copy);
                                let mut destination_offset = 0u32;
                                for span in spans {
                                    local_copies.push((
                                        tile,
                                        LocalCopy {
                                            source: right,
                                            source_offset: span.offset,
                                            destination: copy,
                                            destination_offset,
                                            bytes: span.bytes,
                                        },
                                    ));
                                    destination_offset = destination_offset
                                        .checked_add(span.bytes)
                                        .ok_or(LowLoweringError::IdOverflow)?;
                                }
                                self.full_view(copy)
                            }
                        } else {
                            let slot = &mut remote_right_staging[usize::from(tile)][staging_slot];
                            let copy = if let Some(copy) = *slot {
                                copy
                            } else {
                                let copy = self.push_shard(LowShard {
                                    id: LowShardId(0),
                                    tile,
                                    tensor_type: self.shards[right.index() as usize]
                                        .tensor_type
                                        .clone(),
                                    extents: right_view.extents.clone(),
                                    definition: ShardDefinition::ExchangeStaging,
                                })?;
                                *slot = Some(copy);
                                copy
                            };
                            transfers
                                .entry(right_view.clone())
                                .or_default()
                                .push(self.full_view(copy));
                            self.full_view(copy)
                        };
                        let output_view = self
                            .narrow_view(*output, &[(output_rank - 1, column_start, column_end)])?;
                        let mut selected_kernel = if inner_start == 0 {
                            initialize.clone()
                        } else {
                            accumulate.clone()
                        };
                        if self.shards[resident_right.shard.index() as usize]
                            .tensor_type
                            .format
                            .layout
                            .memory_class
                            == crate::MemoryClass::Ipu21Interleaved
                            && let TileKernelSpec::Gemm { weights, .. } = &mut selected_kernel
                        {
                            *weights = crate::GemmWeightLoad::Interleaved;
                        }
                        runs.push((
                            tile,
                            KernelRun::new(
                                WorkProvenance {
                                    operation: operation.source,
                                    value: Some(*output_value),
                                    reason: WorkReason::OperatorKernel,
                                },
                                TileKernel::Planned(selected_kernel),
                                vec![
                                    KernelOperand {
                                        views: vec![left_view],
                                    },
                                    KernelOperand {
                                        views: vec![resident_right],
                                    },
                                ],
                                output_view,
                                KernelRequirements::Operator(requirements.clone()),
                            ),
                        ));
                    }
                }
                self.append_phase(
                    transfers,
                    WorkProvenance {
                        operation: operation.source,
                        value: (!self.deferred_conversions.contains_key(left_value))
                            .then_some(*right_value),
                        reason: if self.deferred_conversions.contains_key(left_value) {
                            WorkReason::OperatorInputs
                        } else {
                            WorkReason::OperatorInput { input: 1 }
                        },
                    },
                    tiles,
                )?;
                for (tile, copy) in local_copies {
                    self.append_local_copy(tiles, tile, copy)?;
                }
                for (tile, run) in runs {
                    self.append_kernel(tiles, tile, run)?;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_streamed_blocked_gemm(
        &mut self,
        operation: &MidOperation,
        initialize: TileKernelSpec,
        accumulate: TileKernelSpec,
        inner_block: u32,
        output_column_block: u32,
        requirements: &OperatorRequirements,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [left_value, right_value] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidOperatorPlan);
        };
        let [output_value] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        if inner_block == 0 || output_column_block == 0 {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let left_shards = self.value_shards(*left_value)?.to_vec();
        let right_shards = self.value_shards(*right_value)?.to_vec();
        let output_shards = self.value_shards(*output_value)?.to_vec();
        let left_type = &self.shards[left_shards[0].index() as usize].tensor_type;
        let output_type = &self.shards[output_shards[0].index() as usize].tensor_type;
        let left_rank = left_type.shape.0.len();
        let output_rank = output_type.shape.0.len();
        if left_rank < 2 || output_rank < 2 {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let inner_extent = left_type.format.layout.padded_shape(&left_type.shape)?.0[left_rank - 1];
        let column_extent = output_type
            .format
            .layout
            .padded_shape(&output_type.shape)?
            .0[output_rank - 1];
        if !inner_extent.is_multiple_of(inner_block)
            || !column_extent.is_multiple_of(output_column_block)
        {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let staging_bytes = inner_block
            .checked_mul(output_column_block)
            .and_then(|elements| {
                elements.checked_mul(requirements.inputs[1].format.precision.bytes() as u32)
            })
            .ok_or(LowLoweringError::IdOverflow)?;
        let use_interleaved_staging = requirements.inputs[1].format.precision
            == crate::Precision::F16
            && output_shards.iter().try_fold(true, |available, output| {
                let tile = self.shards[output.index() as usize].tile;
                Ok::<_, LowLoweringError>(
                    available
                        && self.interleaved_capacity_available(
                            tile,
                            staging_bytes,
                            requirements.inputs[1].access_tail_bytes,
                        )?,
                )
            })?;
        let mut staging = BTreeMap::<(u16, u32), LowShardId>::new();
        let mut local_staging = BTreeMap::<(u16, u32), LowShardId>::new();
        let columns_per_phase = column_extent / output_column_block;
        let column_phase_width = output_column_block
            .checked_mul(columns_per_phase)
            .ok_or(LowLoweringError::IdOverflow)?;

        for inner_start in (0..inner_extent).step_by(inner_block as usize) {
            let inner_end = inner_start + inner_block;
            for phase_column_start in (0..column_extent).step_by(column_phase_width as usize) {
                let phase_column_end = phase_column_start
                    .saturating_add(column_phase_width)
                    .min(column_extent);
                let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
                let mut local_copies = Vec::<(u16, LocalCopy)>::new();
                let mut runs = Vec::with_capacity(output_shards.len());
                let mut left_views = BTreeMap::<u16, ShardView>::new();
                for column_start in
                    (phase_column_start..phase_column_end).step_by(output_column_block as usize)
                {
                    let column_end = column_start + output_column_block;
                    let right_candidates = self
                        .right_shards_for_block(
                            &right_shards,
                            column_start,
                            column_end,
                            inner_start,
                            inner_end,
                        )
                        .collect::<Vec<_>>();
                    if right_candidates.is_empty() {
                        return Err(LowLoweringError::InvalidOperatorPlan);
                    }
                    let column_outputs = output_shards
                        .iter()
                        .copied()
                        .filter(|output| {
                            let extents = &self.shards[output.index() as usize].extents;
                            let columns = extents[extents.len() - 1];
                            columns.start <= column_start && columns.physical_end >= column_end
                        })
                        .collect::<Vec<_>>();
                    for output in column_outputs {
                        let tile = self.shards[output.index() as usize].tile;
                        let left_view = if let Some(view) = left_views.get(&tile) {
                            view.clone()
                        } else {
                            let view = self.dispatch_input_view(
                                *left_value,
                                tile,
                                &[(left_rank - 1, inner_start, inner_end)],
                                &mut transfers,
                                &mut local_copies,
                            )?;
                            left_views.insert(tile, view.clone());
                            view
                        };
                        let right = self
                            .prefer_local_shard(&right_candidates, tile)
                            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                        let right_rank = self.shards[right.index() as usize].extents.len();
                        let right_view = self.narrow_view(
                            right,
                            &[
                                (right_rank - 2, inner_start, inner_end),
                                (right_rank - 1, column_start, column_end),
                            ],
                        )?;
                        let local = self.shards[right.index() as usize].tile == tile;
                        let local_spans = local
                            .then(|| {
                                view_byte_spans(&self.shards[right.index() as usize], &right_view)
                            })
                            .transpose()?;
                        let resident_view =
                            if local_spans.as_ref().is_some_and(|spans| spans.len() == 1)
                                && (!use_interleaved_staging
                                    || requirements.inputs[1].local_staging
                                        == crate::LocalOperandStaging::Direct)
                            {
                                right_view.clone()
                            } else {
                                let selected_staging = if local {
                                    &mut local_staging
                                } else {
                                    &mut staging
                                };
                                let resident = if let Some(resident) =
                                    selected_staging.get(&(tile, column_start)).copied()
                                {
                                    resident
                                } else {
                                    let mut tensor_type =
                                        self.shards[right.index() as usize].tensor_type.clone();
                                    if use_interleaved_staging {
                                        tensor_type.format.layout.memory_class =
                                            crate::MemoryClass::Ipu21Interleaved;
                                    }
                                    let resident = self.push_shard(LowShard {
                                        id: LowShardId(0),
                                        tile,
                                        tensor_type,
                                        extents: right_view.extents.clone(),
                                        definition: if local {
                                            ShardDefinition::Staging
                                        } else {
                                            ShardDefinition::ExchangeStaging
                                        },
                                    })?;
                                    selected_staging.insert((tile, column_start), resident);
                                    resident
                                };
                                if let Some(spans) = local_spans {
                                    let mut destination_offset = 0u32;
                                    for span in spans {
                                        local_copies.push((
                                            tile,
                                            LocalCopy {
                                                source: right,
                                                source_offset: span.offset,
                                                destination: resident,
                                                destination_offset,
                                                bytes: span.bytes,
                                            },
                                        ));
                                        destination_offset = destination_offset
                                            .checked_add(span.bytes)
                                            .ok_or(LowLoweringError::IdOverflow)?;
                                    }
                                } else {
                                    transfers
                                        .entry(right_view.clone())
                                        .or_default()
                                        .push(self.full_view(resident));
                                }
                                self.full_view(resident)
                            };
                        let output_view = self
                            .narrow_view(output, &[(output_rank - 1, column_start, column_end)])?;
                        let mut kernel = if inner_start == 0 {
                            initialize.clone()
                        } else {
                            accumulate.clone()
                        };
                        if self.shards[resident_view.shard.index() as usize]
                            .tensor_type
                            .format
                            .layout
                            .memory_class
                            == crate::MemoryClass::Ipu21Interleaved
                            && let TileKernelSpec::Gemm { weights, .. } = &mut kernel
                        {
                            *weights = crate::GemmWeightLoad::Interleaved;
                        }
                        runs.push((
                            tile,
                            KernelRun::new(
                                WorkProvenance {
                                    operation: operation.source,
                                    value: Some(*output_value),
                                    reason: WorkReason::OperatorKernel,
                                },
                                TileKernel::Planned(kernel),
                                vec![
                                    KernelOperand {
                                        views: vec![left_view],
                                    },
                                    KernelOperand {
                                        views: vec![resident_view],
                                    },
                                ],
                                output_view,
                                KernelRequirements::Operator(requirements.clone()),
                            ),
                        ));
                    }
                }
                self.append_phase(
                    transfers,
                    WorkProvenance {
                        operation: operation.source,
                        value: (!self.deferred_conversions.contains_key(left_value))
                            .then_some(*right_value),
                        reason: if self.deferred_conversions.contains_key(left_value) {
                            WorkReason::OperatorInputs
                        } else {
                            WorkReason::OperatorInput { input: 1 }
                        },
                    },
                    tiles,
                )?;
                for (tile, copy) in local_copies {
                    self.append_local_copy(tiles, tile, copy)?;
                }
                for (tile, run) in runs {
                    self.append_kernel(tiles, tile, run)?;
                }
            }
        }
        Ok(())
    }

    fn append_phase(
        &mut self,
        transfers: BTreeMap<ShardView, Vec<ShardView>>,
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        if transfers.is_empty() {
            return Ok(());
        }
        let id = ExchangePhaseId(
            u32::try_from(self.phases.len()).map_err(|_| LowLoweringError::IdOverflow)?,
        );
        self.phases.push(ExchangePhase {
            id,
            provenance,
            transfers: transfers
                .into_iter()
                .map(|(source, destinations)| LogicalExchange {
                    source,
                    destinations,
                })
                .collect(),
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
            if !value_can_alias(
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
        let body = self.lower_region(&repeat.body.operations, &repeat.body.yields)?;
        for tile in 0..self.tile_count {
            let mut carried = Vec::with_capacity(repeat.carried_inputs);
            for index in 0..repeat.carried_inputs {
                let initial = self.local_shard(operation.inputs[index], tile)?;
                let argument = self.local_shard(repeat.body.arguments[index], tile)?;
                let yielded = self.local_shard(repeat.body.yields[index], tile)?;
                let result = self.local_shard(operation.results[index], tile)?;
                self.alias_shard(argument, initial);
                if yielded != argument {
                    self.alias_shard(yielded, argument);
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
                .map(|index| {
                    let input_index = repeat.carried_inputs + index;
                    Ok(RepeatInvariant {
                        input: self.local_shard(operation.inputs[input_index], tile)?,
                        argument: self.local_shard(repeat.body.arguments[input_index], tile)?,
                    })
                })
                .collect::<LowLoweringResult<_>>()?;
            let iterated = repeat
                .iterated_inputs
                .iter()
                .enumerate()
                .map(|(index, values)| {
                    let inputs = values
                        .iter()
                        .map(|value| self.local_shard(*value, tile))
                        .collect::<LowLoweringResult<Vec<_>>>()?;
                    let (alignment, access_tail) = iterated_requirements[index];
                    let strides = inputs
                        .iter()
                        .map(|shard| self.shard_stride(*shard, alignment, access_tail))
                        .collect::<LowLoweringResult<Vec<_>>>()?;
                    let Some(&stride_bytes) = strides.first() else {
                        return Err(LowLoweringError::InvalidIteratedBlocks(index));
                    };
                    if strides.iter().any(|stride| *stride != stride_bytes) {
                        return Err(LowLoweringError::InvalidIteratedBlocks(index));
                    }
                    Ok(RepeatIterated {
                        inputs,
                        argument: self
                            .local_shard(repeat.body.arguments[expected_inputs + index], tile)?,
                        stride_bytes,
                        alignment,
                    })
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

fn append_span_copies(
    shards: &[LowShard],
    source: &ShardView,
    destination: &ShardView,
    tile: u16,
    copies: &mut Vec<(u16, LocalCopy)>,
) -> LowLoweringResult<()> {
    let source_spans = view_byte_spans(&shards[source.shard.index() as usize], source)?;
    let destination_spans =
        view_byte_spans(&shards[destination.shard.index() as usize], destination)?;
    append_byte_span_copies(
        source,
        destination,
        tile,
        &source_spans,
        &destination_spans,
        copies,
    )
}

fn append_byte_span_copies(
    source: &ShardView,
    destination: &ShardView,
    tile: u16,
    source_spans: &[ByteSpan],
    destination_spans: &[ByteSpan],
    copies: &mut Vec<(u16, LocalCopy)>,
) -> LowLoweringResult<()> {
    let mut source_index = 0usize;
    let mut destination_index = 0usize;
    let mut source_offset = 0u32;
    let mut destination_offset = 0u32;
    while source_index < source_spans.len() && destination_index < destination_spans.len() {
        let source_span = source_spans[source_index];
        let destination_span = destination_spans[destination_index];
        let bytes =
            (source_span.bytes - source_offset).min(destination_span.bytes - destination_offset);
        copies.push((
            tile,
            LocalCopy {
                source: source.shard,
                source_offset: source_span.offset + source_offset,
                destination: destination.shard,
                destination_offset: destination_span.offset + destination_offset,
                bytes,
            },
        ));
        source_offset += bytes;
        destination_offset += bytes;
        if source_offset == source_span.bytes {
            source_index += 1;
            source_offset = 0;
        }
        if destination_offset == destination_span.bytes {
            destination_index += 1;
            destination_offset = 0;
        }
    }
    if source_index != source_spans.len() || destination_index != destination_spans.len() {
        return Err(LowLoweringError::InvalidConversionPlan);
    }
    Ok(())
}

fn append_logical_span_copies(
    shards: &[LowShard],
    source: &ShardView,
    destination: &ShardView,
    tile: u16,
    copies: &mut Vec<(u16, LocalCopy)>,
) -> LowLoweringResult<()> {
    let source_spans = logical_view_byte_spans(&shards[source.shard.index() as usize], source)?;
    let destination_spans =
        logical_view_byte_spans(&shards[destination.shard.index() as usize], destination)?;
    append_byte_span_copies(
        source,
        destination,
        tile,
        &source_spans,
        &destination_spans,
        copies,
    )
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
    let Some(plan) = &operation.operator_plan else {
        return false;
    };
    let indices = match &plan.requirements.output_aliasing {
        OutputAliasing::Fresh => return false,
        OutputAliasing::MayAliasInputs(indices) => indices.as_slice(),
        OutputAliasing::MustAliasInput(index) => std::slice::from_ref(index),
    };
    indices.iter().any(|index| {
        operation
            .inputs
            .get(usize::from(*index))
            .is_some_and(|input| value_can_alias(*input, target, operations))
    })
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
                .operator_plan
                .as_ref()
                .and_then(|plan| plan.requirements.inputs.get(index))
                .or_else(|| operation.conversion_plan.as_ref().map(|plan| &plan.input));
            if let Some(requirement) = requirement {
                alignment = alignment.max(requirement.alignment);
                access_tail = access_tail.max(requirement.access_tail_bytes);
            }
        }
    }
    (alignment, access_tail)
}

fn operation_provenance(operation: &MidOperation, kind: &MidOperationKind) -> WorkProvenance {
    WorkProvenance {
        operation: operation.source,
        value: operation.results.first().copied(),
        reason: match kind {
            MidOperationKind::CastPrecision { .. } => WorkReason::PrecisionCast,
            MidOperationKind::Rearrange { .. } => WorkReason::LayoutRearrangement,
            MidOperationKind::Operator(_) => WorkReason::OperatorKernel,
            MidOperationKind::Repeat(_) => WorkReason::Repeat,
        },
    }
}

fn intersect_extents(left: &[ShardExtent], right: &[ShardExtent]) -> Option<Vec<ShardExtent>> {
    if left.len() != right.len() {
        return None;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let start = left.start.max(right.start);
            let end = left.logical_end.min(right.logical_end);
            (start < end).then_some(ShardExtent {
                axis: left.axis,
                start,
                logical_end: end,
                physical_end: end,
            })
        })
        .collect()
}

fn shard_extents(tensor_type: &TensorType) -> LowLoweringResult<Vec<Vec<ShardExtent>>> {
    let layout = &tensor_type.format.layout;
    let padded = layout.padded_shape(&tensor_type.shape)?;
    let rank = tensor_type.shape.0.len();
    let strides = layout.tiling.axis_strides()?;
    let axes = layout
        .tiling
        .axes
        .iter()
        .zip(strides)
        .map(|(tiling, stride)| Ok((tiling.axis.resolve(rank)?, tiling, stride)))
        .collect::<LowLoweringResult<Vec<_>>>()?;
    let mut all = Vec::with_capacity(usize::from(layout.tiling.tile_count));
    for tile in 0..layout.tiling.tile_count {
        let mut extents = Vec::with_capacity(rank);
        for axis in 0..rank {
            let (start, physical_end) = if let Some((_, tiling, stride)) =
                axes.iter().find(|(index, _, _)| *index == axis)
            {
                let coordinate = (u32::from(tile) / *stride) % u32::from(tiling.partitions);
                let blocks = padded.0[axis] / tiling.block_size;
                let partitions = u32::from(tiling.partitions);
                let short_size = blocks / partitions;
                let long_shards = blocks % partitions;
                let start_blocks = coordinate * short_size + coordinate.min(long_shards);
                let shard_blocks = short_size + u32::from(coordinate < long_shards);
                let start = start_blocks * tiling.block_size;
                (start, start + shard_blocks * tiling.block_size)
            } else {
                (0, padded.0[axis])
            };
            extents.push(ShardExtent {
                axis: u16::try_from(axis).map_err(|_| LowLoweringError::IdOverflow)?,
                start,
                logical_end: physical_end.min(tensor_type.shape.0[axis]).max(start),
                physical_end,
            });
        }
        all.push(extents);
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AxisTiling, ComputeGraph, ElementOrder, Ipu21CostModel, Layout, MemoryClass, MidOperator,
        OperandRequirement, OperatorCandidate, Padding, PipelineConfig, Precision, TensorAxis,
        TensorFormat, TensorTiling, lower,
    };
    use std::collections::BTreeSet;

    const CASES: usize = 32;

    fn format(tiles: u16) -> TensorFormat {
        TensorFormat {
            precision: Precision::F16,
            layout: Layout::row_sharded(tiles),
        }
    }

    #[test]
    fn randomized_pointwise_dispatch_skips_empty_output_shards() {
        let mut random = fastrand::Rng::with_seed(0x656d_7074);
        for case in 0..CASES {
            let tiles = random.u16(2..=32);
            let rows = random.u32(1..u32::from(tiles));
            let columns = random.u32(1..=32) * 2;
            let tensor_format = format(tiles);
            let mut graph = ComputeGraph::new();
            let input = graph.host_input("input", [rows, columns]).unwrap();
            let output = graph.gelu(input).unwrap();
            graph.set_outputs([output]).unwrap();
            let mut config = PipelineConfig::new(tiles).with_input(input, tensor_format.clone());
            config.operator_candidates = vec![OperatorCandidate::new(
                MidOperator::Gelu,
                [OperandRequirement::new(tensor_format.clone(), 8)],
                OperandRequirement::new(tensor_format, 8),
            )];

            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();
            let runs = low
                .tiles
                .iter()
                .flat_map(|tile| low.work(tile))
                .filter_map(|work| match work {
                    TileWorkRef::Kernel(run) => Some(run),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(runs.len(), rows as usize, "random case {case}");
            assert!(runs.iter().all(|run| {
                run.output
                    .extents
                    .iter()
                    .all(|extent| extent.start < extent.physical_end)
            }));
        }
    }

    #[test]
    fn randomized_dispatch_streaming_defers_one_use_rearrangements() {
        let mut random = fastrand::Rng::with_seed(0x7374_7265_616d);
        for case in 0..8 {
            let batch = random.u32(1..=4);
            let tokens = 16;
            let mut graph = ComputeGraph::new();
            let input = graph.host_input("input", [batch, tokens, 64]).unwrap();
            let up = graph.parameter("up", [1, 64, 256]).unwrap();
            let down = graph.parameter("down", [1, 256, 64]).unwrap();
            let hidden = graph.gemm(input, up).unwrap();
            let hidden = graph.gelu(hidden).unwrap();
            let output = graph.gemm(hidden, down).unwrap();
            graph.set_outputs([output]).unwrap();
            let mut config = PipelineConfig::new(16)
                .with_active_tile_counts([16])
                .with_automatic_input(input, Precision::F16)
                .with_automatic_input(up, Precision::F16)
                .with_automatic_input(down, Precision::F16);
            config.conversion_streaming = crate::ConversionStreamingPolicy::Always;

            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let deferred = mid
                .operations
                .iter()
                .filter_map(|operation| {
                    operation.conversion_plan.as_ref().and_then(|plan| {
                        (plan.output.materialization
                            == crate::OperandMaterialization::DispatchSlices)
                            .then(|| operation.results[0])
                    })
                })
                .collect::<BTreeSet<_>>();
            assert!(!deferred.is_empty(), "case {case}");
            let consumers = mid
                .operations
                .iter()
                .filter(|operation| {
                    operation
                        .inputs
                        .iter()
                        .any(|input| deferred.contains(input))
                })
                .filter_map(|operation| operation.source)
                .collect::<BTreeSet<_>>();

            let low = lower_to_tiles(&mid, &config).unwrap();
            assert!(
                low.exchange_phases
                    .iter()
                    .all(|phase| phase.provenance.reason != WorkReason::LayoutRearrangement),
                "case {case}"
            );
            assert!(
                low.exchange_phases
                    .iter()
                    .any(|phase| phase.provenance.reason == WorkReason::OperatorInputs),
                "case {case}"
            );
            assert!(
                low.shards
                    .iter()
                    .filter(|shard| shard.definition == ShardDefinition::Unmaterialized)
                    .count()
                    >= 16 * deferred.len(),
                "case {case}"
            );
            for run in &low.kernel_runs {
                if run
                    .provenance
                    .operation
                    .is_some_and(|operation| consumers.contains(&operation))
                {
                    let input = &run.inputs[0].views[0];
                    assert_eq!(
                        low.shards[input.shard.index() as usize].definition,
                        ShardDefinition::ExchangeStaging,
                        "case {case}"
                    );
                    let inner = input.extents.last().unwrap();
                    assert!(inner.physical_end - inner.start <= 64, "case {case}");
                }
            }
        }
    }

    #[test]
    fn randomized_tile_local_gelu_reorders_without_exchange() {
        let mut random = fastrand::Rng::with_seed(0x6765_6c75);
        for case in 0..CASES {
            let row_partitions = 1_u16 << random.u32(0..=3);
            let column_partitions = 1_u16 << random.u32(0..=3);
            let tiles = row_partitions * column_partitions;
            let rows = u32::from(row_partitions) * random.u32(1..=8);
            let columns = u32::from(column_partitions) * 64 * random.u32(1..=4);
            let input_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::amp_output_replicated_grid(
                    tiles,
                    row_partitions,
                    column_partitions,
                ),
            };
            let output_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::amp_left_grid(64, tiles, row_partitions, column_partitions),
            };
            let mut graph = ComputeGraph::new();
            let input = graph.host_input("input", [rows, columns]).unwrap();
            let output = graph.gelu(input).unwrap();
            graph.set_outputs([output]).unwrap();
            let mut config = PipelineConfig::new(tiles).with_input(input, input_format.clone());
            config.operator_candidates = vec![OperatorCandidate::new(
                MidOperator::Gelu,
                [OperandRequirement::new(input_format, 8)],
                OperandRequirement::new(output_format, 8),
            )];

            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();
            assert!(low.exchange_phases.is_empty(), "random case {case}");
            for tile in &low.tiles {
                for work in low.work(tile) {
                    let TileWorkRef::Kernel(run) = work else {
                        continue;
                    };
                    assert_eq!(
                        low.shards[run.inputs[0].views[0].shard.index() as usize].tile,
                        tile.tile
                    );
                    assert_eq!(
                        low.shards[run.output.shard.index() as usize].tile,
                        tile.tile
                    );
                }
            }
        }
    }

    #[test]
    fn randomized_same_order_retiles_exchange_into_final_values() {
        let mut random = fastrand::Rng::with_seed(0x6469_7265_6374_7265);
        for case in 0..CASES {
            let source_rows = 1_u16 << random.u32(0..=3);
            let source_columns = 1_u16 << random.u32(0..=3);
            let tiles = source_rows * source_columns;
            let rows = u32::from(source_rows.max(source_columns)) * random.u32(1..=4);
            let columns = u32::from(source_rows.max(source_columns)) * random.u32(1..=4) * 4;
            let layout = |row_partitions, column_partitions| Layout {
                order: ElementOrder::RowMajor,
                tiling: TensorTiling {
                    tile_count: tiles,
                    replicas: 1,
                    axes: vec![
                        AxisTiling::new(TensorAxis::FromEnd(2), row_partitions, 1, Padding::Reject),
                        AxisTiling::new(
                            TensorAxis::FromEnd(1),
                            column_partitions,
                            4,
                            Padding::Reject,
                        ),
                    ],
                },
                memory_class: MemoryClass::Ipu21Standard,
            };
            let input_format = TensorFormat {
                precision: Precision::F16,
                layout: layout(source_rows, source_columns),
            };
            let target_format = TensorFormat {
                precision: Precision::F16,
                layout: layout(source_columns, source_rows),
            };
            let mut graph = ComputeGraph::new();
            let input = graph.host_input("input", [rows, columns]).unwrap();
            let output = graph.gelu(input).unwrap();
            graph.set_outputs([output]).unwrap();
            let mut config = PipelineConfig::new(tiles).with_input(input, input_format);
            config.operator_candidates = vec![OperatorCandidate::new(
                MidOperator::Gelu,
                [OperandRequirement::new(target_format.clone(), 8)],
                OperandRequirement::new(target_format, 8),
            )];

            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();
            let conversion_phases = low
                .exchange_phases
                .iter()
                .filter(|phase| phase.provenance.reason == WorkReason::LayoutRearrangement)
                .collect::<Vec<_>>();
            for phase in conversion_phases {
                for destination in phase
                    .transfers
                    .iter()
                    .flat_map(|transfer| &transfer.destinations)
                {
                    assert!(matches!(
                        low.shards[destination.shard.index() as usize].definition,
                        ShardDefinition::Value(_)
                    ));
                }
            }
            assert!(
                low.shards
                    .iter()
                    .all(|shard| !matches!(shard.definition, ShardDefinition::ExchangeStaging)),
                "case {case}"
            );
            assert!(low.local_copies.iter().all(|copy| {
                low.shards[copy.source.index() as usize].tile
                    == low.shards[copy.destination.index() as usize].tile
            }));
        }
    }

    #[test]
    fn randomized_multiaxis_shards_cover_padded_extents_in_whole_blocks() {
        let mut random = fastrand::Rng::with_seed(0x7368_6172);
        for case in 0..CASES {
            let row_partitions = random.u16(1..=4);
            let column_partitions = random.u16(1..=4);
            let replicas = random.u16(1..=3);
            let row_block = 1_u32 << random.u32(0..=3);
            let column_block = 1_u32 << random.u32(0..=3);
            let tile_count = row_partitions * column_partitions * replicas;
            let layout = Layout {
                order: ElementOrder::RowMajor,
                tiling: TensorTiling {
                    tile_count,
                    replicas,
                    axes: vec![
                        AxisTiling::new(
                            TensorAxis::FromEnd(1),
                            column_partitions,
                            column_block,
                            Padding::Zero,
                        ),
                        AxisTiling::new(
                            TensorAxis::FromEnd(2),
                            row_partitions,
                            row_block,
                            Padding::Zero,
                        ),
                    ],
                },
                memory_class: MemoryClass::Ipu21Standard,
            };
            let tensor_type = TensorType::new(
                [
                    u32::from(row_partitions) * row_block + random.u32(0..=65),
                    u32::from(column_partitions) * column_block + random.u32(0..=65),
                ],
                Precision::F16,
                layout.clone(),
            );
            let padded = layout.padded_shape(&tensor_type.shape).unwrap();
            let shards = shard_extents(&tensor_type).unwrap();
            assert_eq!(shards.len(), usize::from(tile_count), "case {case}");

            for (axis, partitions, block) in [
                (0, row_partitions, row_block),
                (1, column_partitions, column_block),
            ] {
                let ranges = shards
                    .iter()
                    .map(|extents| (extents[axis].start, extents[axis].physical_end))
                    .collect::<std::collections::BTreeSet<_>>();
                assert_eq!(ranges.len(), usize::from(partitions), "case {case}");
                let mut cursor = 0;
                for (start, end) in ranges {
                    assert_eq!(start, cursor, "case {case}");
                    assert_eq!(start % block, 0, "case {case}");
                    assert_eq!(end % block, 0, "case {case}");
                    cursor = end;
                }
                assert_eq!(cursor, padded.0[axis], "case {case}");
            }
        }
    }

    #[test]
    fn randomized_schedules_make_kernel_operands_resident() {
        let mut random = fastrand::Rng::with_seed(0x6c6f_7721);
        for case in 0..CASES {
            let tiles = 1_u16 << random.u32(0..=3);
            let rows = u32::from(tiles) * random.u32(1..=8) * 16;
            let columns = random.u32(1..=8) * 16;
            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", [rows, columns]).unwrap();
            let right = graph.host_input("right", [rows, columns]).unwrap();
            let output = graph.add(left, right).unwrap();
            graph.set_outputs([output]).unwrap();
            let config = PipelineConfig::new(tiles)
                .with_active_tile_counts([tiles])
                .with_input(left, format(tiles))
                .with_input(right, format(tiles));
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();

            assert_eq!(low.tiles.len(), usize::from(tiles), "case {case}");
            for tile in &low.tiles {
                for work in low.work(tile) {
                    if let TileWorkRef::Kernel(run) = work {
                        crate::validate_kernel_run(run).unwrap();
                        assert_eq!(
                            low.shards[run.output.shard.index() as usize].tile,
                            tile.tile
                        );
                        assert!(
                            run.inputs
                                .iter()
                                .flat_map(|operand| &operand.views)
                                .all(|view| {
                                    low.shards[view.shard.index() as usize].tile == tile.tile
                                })
                        );
                    }
                }
            }
            for phase in &low.exchange_phases {
                assert!(
                    low.tiles
                        .iter()
                        .all(|tile| contains_phase(&low, tile, phase.id))
                );
                for transfer in &phase.transfers {
                    assert!(transfer.destinations.iter().all(|destination| matches!(
                        low.shards[destination.shard.index() as usize].definition,
                        ShardDefinition::Value(_)
                            | ShardDefinition::ExchangeStaging
                            | ShardDefinition::LocalCopy(_)
                            | ShardDefinition::Staging
                    )));
                }
            }
        }
    }

    #[test]
    fn randomized_broadcast_adds_schedule_remote_singleton_views() {
        let mut random = fastrand::Rng::with_seed(0x6272_6463);
        for case in 0..CASES {
            let tiles = 1_u16 << random.u32(1..=3);
            let rows = u32::from(tiles) * random.u32(1..=8);
            let columns = random.u32(1..=8) * 16;
            let mut graph = ComputeGraph::new();
            let bias = graph.host_input("bias", [1, columns]).unwrap();
            let tensor = graph.host_input("tensor", [rows, columns]).unwrap();
            let output = graph.add(bias, tensor).unwrap();
            graph.set_outputs([output]).unwrap();
            let config = PipelineConfig::new(tiles)
                .with_input(bias, format(tiles))
                .with_input(tensor, format(tiles));
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();

            assert!(
                low.exchange_phases
                    .iter()
                    .any(|phase| matches!(phase.provenance.reason, WorkReason::OperatorInputs)),
                "case {case}"
            );
            for tile in &low.tiles {
                let add = low
                    .work(tile)
                    .find_map(|work| match work {
                        TileWorkRef::Kernel(run)
                            if matches!(run.kernel, TileKernel::Planned(TileKernelSpec::Add)) =>
                        {
                            Some(run)
                        }
                        _ => None,
                    })
                    .unwrap();
                assert_eq!(add.inputs[0].views[0].extents[0].logical_end, 1);
                assert_eq!(
                    low.shards[add.inputs[0].views[0].shard.index() as usize].tile,
                    tile.tile
                );
            }
        }
    }

    #[test]
    fn randomized_blocked_gemms_expand_to_tile_kernel_phases() {
        let mut random = fastrand::Rng::with_seed(0x6765_6d6d);
        for case in 0..CASES {
            let tiles = 1_u16 << random.u32(0..=3);
            let rows = u32::from(tiles) * random.u32(1..=4) * 8;
            let inner_blocks = random.u32(1..=4);
            let column_blocks = random.u32(1..=4);
            let inner = inner_blocks * 64;
            let columns = column_blocks * 64;
            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", [rows, inner]).unwrap();
            let right = graph.parameter("right", [inner, columns]).unwrap();
            let output = graph.gemm(left, right).unwrap();
            graph.set_outputs([output]).unwrap();
            let config = PipelineConfig::new(tiles)
                .with_active_tile_counts([tiles])
                .with_input(left, format(tiles))
                .with_input(right, format(tiles));
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();

            assert!(std::mem::size_of::<TileWork>() <= 8);
            let mut metadata = Vec::<&Arc<KernelRunMetadata>>::new();
            for run in &low.kernel_runs {
                if let Some(existing) = metadata
                    .iter()
                    .find(|existing| existing.as_ref() == run.metadata.as_ref())
                {
                    assert!(Arc::ptr_eq(existing, &run.metadata), "case {case}");
                } else {
                    metadata.push(&run.metadata);
                }
            }

            let gemm_destinations = low
                .exchange_phases
                .iter()
                .filter(|phase| phase.provenance.reason == WorkReason::OperatorInput { input: 1 })
                .flat_map(|phase| &phase.transfers)
                .flat_map(|transfer| &transfer.destinations)
                .map(|destination| destination.shard)
                .collect::<Vec<_>>();
            let unique_gemm_staging = gemm_destinations
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>();
            let panels_per_phase = column_blocks;
            assert!(
                unique_gemm_staging.len()
                    <= usize::from(tiles)
                        * usize::try_from(column_blocks.min(panels_per_phase)).unwrap(),
                "case {case}"
            );
            let gemm_phases = low
                .exchange_phases
                .iter()
                .filter(|phase| phase.provenance.reason == WorkReason::OperatorInput { input: 1 })
                .count();
            if gemm_phases > 1 && !unique_gemm_staging.is_empty() {
                assert!(
                    gemm_destinations.len() > unique_gemm_staging.len(),
                    "case {case}"
                );
            }
            assert!(low.exchange_phases.iter().all(|phase| {
                phase.provenance.operation.is_some() && phase.provenance.value.is_some()
            }));
            let weight_modes = |tile: &TileWorkList| {
                low.work(tile)
                    .filter_map(|work| match work {
                        TileWorkRef::Kernel(run) => match &run.kernel {
                            TileKernel::Planned(TileKernelSpec::Gemm { weights, .. }) => {
                                Some(*weights)
                            }
                            _ => None,
                        },
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            };
            let expected_weight_modes = weight_modes(&low.tiles[0]);
            assert!(
                low.tiles
                    .iter()
                    .all(|tile| weight_modes(tile) == expected_weight_modes),
                "case {case}"
            );

            for tile in &low.tiles {
                let gemms = low
                    .work(tile)
                    .filter_map(|work| match work {
                        TileWorkRef::Kernel(run)
                            if matches!(
                                run.kernel,
                                TileKernel::Planned(TileKernelSpec::Gemm { .. })
                            ) =>
                        {
                            Some(run)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let output = low.shards.iter().find(|shard| {
                    shard.tile == tile.tile
                        && shard.tensor_type.shape.0.last() == Some(&columns)
                        && shard.tensor_type.format.layout.memory_class
                            == crate::MemoryClass::Ipu21Interleaved
                });
                let local_column_blocks = match output {
                    Some(shard) => {
                        let output_columns = match gemms.first().map(|run| &run.kernel) {
                            Some(TileKernel::Planned(TileKernelSpec::Gemm {
                                output_columns,
                                ..
                            })) => *output_columns,
                            _ => panic!("case {case}: output tile has no GEMM kernel"),
                        };
                        let extent = shard.extents.last().unwrap();
                        (extent.physical_end - extent.start) / output_columns
                    }
                    None => {
                        assert!(gemms.is_empty(), "case {case}");
                        0
                    }
                };
                assert_eq!(
                    gemms.len(),
                    (inner_blocks * local_column_blocks) as usize,
                    "case {case}"
                );
                let mut initialized_columns = std::collections::BTreeSet::new();
                for run in gemms {
                    assert_eq!(run.provenance.reason, WorkReason::OperatorKernel);
                    assert!(run.provenance.operation.is_some());
                    assert!(run.provenance.value.is_some());
                    let TileKernel::Planned(TileKernelSpec::Gemm {
                        mode,
                        output_columns: kernel_columns,
                        ..
                    }) = run.kernel
                    else {
                        unreachable!()
                    };
                    let columns = run.output.extents.last().unwrap();
                    assert_eq!(
                        mode,
                        if initialized_columns.insert((columns.start, columns.physical_end)) {
                            crate::GemmKernelMode::Initialize
                        } else {
                            crate::GemmKernelMode::Accumulate
                        },
                        "case {case}"
                    );
                    assert_eq!(run.inputs.len(), 2);
                    assert!(run.inputs.iter().all(|operand| operand.views.len() == 1));
                    let left_inner = run.inputs[0].views[0].extents.last().unwrap();
                    assert_eq!(left_inner.physical_end - left_inner.start, 64);
                    let output_columns = run.output.extents.last().unwrap();
                    assert_eq!(
                        output_columns.physical_end - output_columns.start,
                        kernel_columns
                    );
                }
            }
        }
    }

    #[test]
    fn randomized_automatic_gemms_choose_local_two_axis_grids() {
        let mut random = fastrand::Rng::with_seed(0x6772_6964_6765_6d6d);
        for case in 0..CASES {
            let tiles = 1_u16 << random.u32(0..=3);
            let rows = u32::from(tiles) * random.u32(1..=8);
            let inner_blocks = random.u32(1..=4);
            let inner = inner_blocks * 64;
            let columns = u32::from(tiles) * 64;
            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", [rows, inner]).unwrap();
            let right = graph.parameter("right", [inner, columns]).unwrap();
            let output = graph.gemm(left, right).unwrap();
            graph.set_outputs([output]).unwrap();
            let config = PipelineConfig::new(tiles)
                .with_automatic_input(left, Precision::F16)
                .with_automatic_input(right, Precision::F16);
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();

            assert!(low.exchange_phases.is_empty(), "case {case}");
            for tile in &low.tiles {
                let gemms = low
                    .work(tile)
                    .filter(|work| {
                        matches!(
                            work,
                            TileWorkRef::Kernel(run)
                                if matches!(run.kernel, TileKernel::Planned(TileKernelSpec::Gemm { .. }))
                        )
                    })
                    .count();
                assert_eq!(gemms, inner_blocks as usize, "case {case}");
            }
        }
    }

    #[test]
    fn randomized_odd_capacities_use_regular_active_tile_subsets() {
        let mut random = fastrand::Rng::with_seed(0x7375_6273_6574);
        for case in 0..16 {
            let active_tiles = 1_u16 << random.u32(2..=5);
            let capacity = active_tiles + random.u16(1..active_tiles);
            let rows = u32::from(active_tiles);
            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", [rows, 64]).unwrap();
            let right = graph.parameter("right", [64, 64]).unwrap();
            let output = graph.gemm(left, right).unwrap();
            graph.set_outputs([output]).unwrap();
            let config = PipelineConfig::new(capacity)
                .with_automatic_input(left, Precision::F16)
                .with_automatic_input(right, Precision::F16);

            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let result = mid.operations.last().unwrap().results[0];
            assert_eq!(
                mid.values[result.index() as usize]
                    .tensor_type
                    .format
                    .layout
                    .tiling
                    .tile_count,
                active_tiles,
                "case {case}"
            );

            let low = lower_to_tiles(&mid, &config).unwrap();
            assert_eq!(low.tile_count, capacity, "case {case}");
            assert_eq!(low.outputs[0].shards.len(), usize::from(active_tiles));
            for tile in &low.tiles[usize::from(active_tiles)..] {
                assert!(
                    low.work(tile)
                        .all(|work| !matches!(work, TileWorkRef::Kernel(_))),
                    "case {case}"
                );
            }
        }
    }

    #[test]
    fn randomized_resident_k64_weights_lower_without_panel_copies() {
        let mut random = fastrand::Rng::with_seed(0x7265_7369);
        for _ in 0..48 {
            let tiles = 1_u16 << random.u32(0..=3);
            let rows = u32::from(tiles) * random.u32(1..=4);
            let inner = 64 * random.u32(2..=4);
            let columns = 64 * random.u32(1..=4);
            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", [rows, inner]).unwrap();
            let right = graph.parameter("right", [inner, columns]).unwrap();
            let output = graph.gemm(left, right).unwrap();
            graph.set_outputs([output]).unwrap();
            let config = PipelineConfig::new(tiles)
                .with_automatic_input(left, Precision::F16)
                .with_automatic_input(right, Precision::F16);
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let operation = mid
                .operations
                .iter()
                .find(|operation| matches!(operation.kind, MidOperationKind::Operator(_)))
                .unwrap();
            let right_type = &mid.values[operation.inputs[1].index() as usize].tensor_type;
            assert!(matches!(
                right_type.format.layout.order,
                crate::ElementOrder::Amp(crate::AmpOrder::RightK64)
            ));
            let low = lower_to_tiles(&mid, &config).unwrap();
            assert!(low.tiles.iter().all(|tile| low.work(tile).all(|work| {
                !matches!(work, TileWorkRef::LocalCopy(_))
                    && !matches!(work, TileWorkRef::Exchange(_))
            })));
            assert!(
                low.tiles
                    .iter()
                    .flat_map(|tile| low.work(tile))
                    .any(|work| {
                        matches!(
                            work,
                            TileWorkRef::Kernel(run)
                                if matches!(run.kernel, TileKernel::Planned(TileKernelSpec::Gemm {
                                    weights: crate::GemmWeightLoad::Interleaved,
                                    ..
                                }))
                        )
                    })
            );
        }
    }

    #[test]
    fn randomized_streamed_weight_grids_share_one_phase_per_inner_block() {
        let mut random = fastrand::Rng::with_seed(0x7374_726d_6765_6d6d);
        for case in 0..32 {
            let row_partitions = 1_u16 << random.u32(1..=2);
            let column_partitions = 1_u16 << random.u32(0..=2);
            let tiles = row_partitions * column_partitions;
            let rows = u32::from(row_partitions) * random.u32(1..=4);
            let inner_blocks = u32::from(row_partitions) * random.u32(1..=2);
            let inner = inner_blocks * 64;
            let columns = u32::from(column_partitions) * 64;
            let local_staging = if random.bool() {
                crate::LocalOperandStaging::Direct
            } else {
                crate::LocalOperandStaging::MatchRemote
            };
            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", [rows, inner]).unwrap();
            let right = graph.parameter("right", [inner, columns]).unwrap();
            let output = graph.gemm(left, right).unwrap();
            graph.set_outputs([output]).unwrap();
            let left_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::amp_left_grid(64, tiles, row_partitions, column_partitions),
            };
            let right_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::amp_right_k64_streamed_grid(
                    tiles,
                    row_partitions,
                    column_partitions,
                    crate::MemoryClass::Ipu21Standard,
                ),
            };
            let output_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::amp_output_grid(tiles, row_partitions, column_partitions),
            };
            let mut config = PipelineConfig::new(tiles)
                .with_input(left, left_format.clone())
                .with_input(right, right_format.clone());
            config.operator_candidates = vec![crate::OperatorCandidate::new(
                crate::MidOperator::Gemm {
                    options: crate::GemmOptions::default(),
                    multiply: Precision::F16,
                    accumulate: crate::AccumulationPrecision::F32,
                },
                [
                    crate::OperandRequirement::new(left_format, 32).with_access_tail(16),
                    crate::OperandRequirement::new(right_format, 32)
                        .with_local_staging(local_staging),
                ],
                crate::OperandRequirement::new(output_format, 32),
            )];
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();

            assert_eq!(
                low.exchange_phases.len(),
                inner_blocks as usize,
                "case {case}"
            );
            let copies = low
                .tiles
                .iter()
                .flat_map(|tile| low.work(tile))
                .filter_map(|work| match work {
                    TileWorkRef::LocalCopy(copy) => Some(copy),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                copies.is_empty(),
                local_staging == crate::LocalOperandStaging::Direct,
                "case {case}: {copies:?}"
            );
            let weight_loads = low
                .kernel_runs
                .iter()
                .filter_map(|run| match run.kernel {
                    TileKernel::Planned(TileKernelSpec::Gemm { weights, .. }) => Some(weights),
                    _ => None,
                })
                .collect::<BTreeSet<_>>();
            let expected_weight_loads = match local_staging {
                crate::LocalOperandStaging::Direct => BTreeSet::from([
                    crate::GemmWeightLoad::Standard,
                    crate::GemmWeightLoad::Interleaved,
                ]),
                crate::LocalOperandStaging::MatchRemote => {
                    BTreeSet::from([crate::GemmWeightLoad::Interleaved])
                }
            };
            assert_eq!(weight_loads, expected_weight_loads, "case {case}");
            for phase in &low.exchange_phases {
                for destination in phase
                    .transfers
                    .iter()
                    .flat_map(|transfer| &transfer.destinations)
                {
                    let shard = &low.shards[destination.shard.index() as usize];
                    assert_eq!(shard.definition, ShardDefinition::ExchangeStaging);
                    assert!(low.kernel_runs.iter().any(|run| {
                        run.inputs
                            .iter()
                            .flat_map(|operand| &operand.views)
                            .any(|view| view.shard == destination.shard)
                    }));
                }
            }
            for tile in &low.tiles {
                assert_eq!(
                    low.work(tile)
                        .filter(|work| matches!(work, TileWorkRef::Kernel(_)))
                        .count(),
                    inner_blocks as usize,
                    "case {case}"
                );
            }
        }
    }

    #[test]
    fn randomized_repeats_remain_structured_per_tile() {
        let mut random = fastrand::Rng::with_seed(0x7265_706c);
        for case in 0..CASES {
            let tiles = 1_u16 << random.u32(0..=3);
            let count = random.u32(1..=8);
            let width = u32::from(tiles) * random.u32(1..=8);
            let mut graph = ComputeGraph::new();
            let carried = graph.host_input("carried", [width, 16]).unwrap();
            let parameters = (0..count)
                .map(|index| graph.parameter(format!("parameter.{index}"), [width, 16]))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let sequence = graph
                .value_sequence("parameters", parameters.clone())
                .unwrap();
            let result = graph
                .repeat(count, [carried], [], [sequence], |body, arguments| {
                    Ok(vec![body.add(arguments.carried[0], arguments.iterated[0])?])
                })
                .unwrap()[0];
            graph.set_outputs([result]).unwrap();
            let mut config = PipelineConfig::new(tiles).with_input(carried, format(tiles));
            for parameter in parameters {
                config.inputs.insert(parameter, format(tiles));
            }
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();

            for tile in &low.tiles {
                let repeats = low
                    .work(tile)
                    .filter_map(|work| match work {
                        TileWorkRef::Repeat(repeat) => Some(repeat),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                assert_eq!(repeats.len(), 1, "case {case}");
                assert_eq!(repeats[0].count, count);
                assert_eq!(repeats[0].iterated[0].inputs.len(), count as usize);
                assert!(repeats[0].iterated[0].stride_bytes > 0);
                assert!(
                    repeats[0].iterated[0]
                        .stride_bytes
                        .is_multiple_of(repeats[0].iterated[0].alignment)
                );
                let carried = &repeats[0].carried[0];
                assert_eq!(
                    low.shards[carried.argument.index() as usize].definition,
                    ShardDefinition::Alias(carried.initial)
                );
                assert_eq!(
                    low.shards[carried.yielded.index() as usize].definition,
                    ShardDefinition::Alias(carried.argument)
                );
                assert_eq!(
                    low.shards[carried.result.index() as usize].definition,
                    ShardDefinition::Alias(carried.initial)
                );
                assert!(
                    low.work(&repeats[0].body)
                        .any(|work| matches!(work, TileWorkRef::Kernel(_)))
                );
            }
        }
    }

    #[test]
    fn randomized_repeats_reject_fresh_carried_results() {
        let mut random = fastrand::Rng::with_seed(0x696e_706c);
        for case in 0..CASES {
            let tiles = 1_u16 << random.u32(0..=3);
            let count = random.u32(1..=4);
            let rows = u32::from(tiles) * random.u32(1..=4) * 8;
            let mut graph = ComputeGraph::new();
            let carried = graph.host_input("carried", [rows, 64]).unwrap();
            let weights = (0..count)
                .map(|index| graph.parameter(format!("weight.{index}"), [64, 64]))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let sequence = graph.value_sequence("weights", weights.clone()).unwrap();
            let result = graph
                .repeat(count, [carried], [], [sequence], |body, arguments| {
                    Ok(vec![
                        body.gemm(arguments.carried[0], arguments.iterated[0])?,
                    ])
                })
                .unwrap()[0];
            graph.set_outputs([result]).unwrap();
            let mut config = PipelineConfig::new(tiles).with_input(carried, format(tiles));
            for weight in weights {
                config.inputs.insert(weight, format(tiles));
            }
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            assert_eq!(
                lower_to_tiles(&mid, &config),
                Err(LowLoweringError::RepeatRequiresInPlace(0)),
                "case {case}"
            );
        }
    }

    fn contains_phase(program: &LowProgram, list: &TileWorkList, phase: ExchangePhaseId) -> bool {
        program.work(list).any(|work| match work {
            TileWorkRef::Exchange(candidate) => candidate == phase,
            TileWorkRef::Repeat(repeat) => contains_phase(program, &repeat.body, phase),
            TileWorkRef::Kernel(_) | TileWorkRef::LocalCopy(_) => false,
        })
    }
}
