//! Logical per-tile schedule produced from the layout-aware mid-level IR.
//!
//! Tensor shards have tile identities and rectangular physical extents, and
//! work is ordered per tile. Exchanges still refer to logical shards rather
//! than SRAM addresses; kernel runs still name a selected kernel kind rather
//! than a linked symbol. Placement and final code generation resolve those
//! remaining choices.

use crate::graph::{GraphInputKind, OperationId};
use crate::layout::ShardExtent;
use crate::mid::{
    AMP_COLUMN_MICRO, AMP_INNER_BLOCK, AmpOrder, BlockMajorOrder, ConversionStrategy,
    DeferredTransform, ElementOrder, GemmDistribution, Layout, LayoutError, MemoryClass,
    MemoryOperand, MemoryRelation, MidGraph, MidOperation, MidOperationKind, MidRepeat, MidValueId,
    OperandRequirement, OperatorDispatch, OperatorRequirements, OutputAliasing, PipelineConfig,
    PointwiseInputMapping, Precision, TensorTiling, TensorType, TileKernelSpec,
};
use crate::storage::{ByteSpan, StorageError, logical_view_byte_spans, view_byte_spans};
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
        memory_relations: Vec<MemoryRelation>,
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
    pub pattern: LocalCopyPattern,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LocalCopyPattern {
    #[default]
    Contiguous,
    Strided {
        rows: u32,
        row_bytes: u32,
        source_stride: u32,
        destination_stride: u32,
    },
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

fn split_gemm_matrices(
    run: &KernelRun,
    axis: usize,
    coordinates: &mut [u32],
    runs: &mut Vec<KernelRun>,
) -> LowLoweringResult<()> {
    if axis < coordinates.len() {
        let extent = run
            .output
            .extents
            .get(axis)
            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
        if extent.logical_end != extent.physical_end {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        for coordinate in extent.start..extent.physical_end {
            coordinates[axis] = coordinate;
            split_gemm_matrices(run, axis + 1, coordinates, runs)?;
        }
        return Ok(());
    }

    let mut matrix = run.clone();
    narrow_gemm_matrix_view(&mut matrix.output, coordinates)?;
    for operand in &mut matrix.inputs {
        for view in &mut operand.views {
            narrow_gemm_matrix_view(view, coordinates)?;
        }
    }
    runs.push(matrix);
    Ok(())
}

fn narrow_gemm_matrix_view(
    view: &mut ShardView,
    output_coordinates: &[u32],
) -> LowLoweringResult<()> {
    let input_axes = view.extents.len().saturating_sub(2);
    if input_axes > output_coordinates.len() {
        return Err(LowLoweringError::InvalidOperatorPlan);
    }
    let output_axis_offset = output_coordinates.len() - input_axes;
    for (axis, extent) in view.extents[..input_axes].iter_mut().enumerate() {
        if extent.physical_end - extent.start == 1 {
            continue;
        }
        let coordinate = output_coordinates[output_axis_offset + axis];
        if coordinate < extent.start || coordinate >= extent.physical_end {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        extent.start = coordinate;
        extent.logical_end = coordinate + 1;
        extent.physical_end = coordinate + 1;
    }
    Ok(())
}

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
        profiling = config.profiling.enabled
    )
)]
pub fn lower_to_tiles(graph: &MidGraph, config: &PipelineConfig) -> LowLoweringResult<LowProgram> {
    if config.tile_count == 0 {
        return Err(LowLoweringError::EmptyTileGroup);
    }
    let mut state = LoweringState::new(graph, config.tile_count)?;
    let tiles = state.lower_region(
        &graph.operations,
        &graph.outputs,
        config.diagnostic_checkpoints,
    )?;
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

struct DeferredValue {
    transform: DeferredTransform,
    shards: Vec<LowShardId>,
}

struct PreparedDistributedPanel {
    panel: u32,
    row_major: Option<LowShardId>,
    packed: LowShardId,
    tile: u16,
    destinations: Vec<LowShardId>,
}

struct PreparedAttentionBlock {
    row_start: u32,
    valid_rows: u32,
    key_panels: Vec<PreparedDistributedPanel>,
    value_panels: Vec<PreparedDistributedPanel>,
}

struct AttentionTask {
    tile: u16,
    head: u32,
    query_row_start: u32,
    query_rows: u32,
    query_dimension: u32,
    value_dimension: u32,
    query: LowShardId,
    query_receive: Option<LowShardId>,
    output: LowShardId,
    scratch: LowShardId,
    weights: LowShardId,
    key_staging: LowShardId,
    value_staging: LowShardId,
}

#[derive(Clone, Copy)]
struct AttentionBufferShape {
    query_block_rows: u32,
    logical_staging_rows: u32,
    physical_staging_rows: u32,
    scratch_columns: u32,
    state_columns: u32,
    padded_query_dimension: u32,
    padded_value_dimension: u32,
    reuse_key_staging_for_state: bool,
}

#[derive(Clone, Copy)]
enum AttentionOperand {
    Key,
    Value,
}

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
            let extents = shard_extents(&value.tensor_type)?;
            let is_parameter = parameter_values.contains(&value.id);
            let placement_group = value.storage_group;
            let rotate_parameter = is_parameter || parameter_groups.contains(&placement_group);
            let parameter_shard_bytes = if rotate_parameter {
                extents
                    .iter()
                    .map(|(_, extents)| {
                        crate::shard_storage_bytes(&LowShard {
                            id: LowShardId(0),
                            tile: 0,
                            tensor_type: value.tensor_type.clone(),
                            extents: extents.clone(),
                            definition: ShardDefinition::Value(value.id),
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

    fn matrix_shards_for_block<'a>(
        &'a self,
        shards: &'a [LowShardId],
        column_axis: usize,
        inner_axis: usize,
        column_start: u32,
        column_end: u32,
        inner_start: u32,
        inner_end: u32,
    ) -> impl Iterator<Item = LowShardId> + 'a {
        shards.iter().copied().filter(move |shard| {
            let extents = &self.shards[shard.index() as usize].extents;
            let columns = extents[column_axis];
            let inner = extents[inner_axis];
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

    fn deferred_root(&self, mut value: MidValueId) -> Option<MidValueId> {
        let mut remaining = self.deferred_conversions.len().saturating_add(1);
        while !self.deferred_values.contains_key(&value) {
            value = *self.deferred_conversions.get(&value)?;
            remaining = remaining.checked_sub(1)?;
        }
        Some(value)
    }

    fn has_deferred_value(&self, value: MidValueId) -> bool {
        self.deferred_root(value).is_some()
    }

    fn deferred_supports_physical_exchange(
        &self,
        value: MidValueId,
        destination: LowShardId,
    ) -> bool {
        let Some(root) = self.deferred_root(value) else {
            return false;
        };
        let Some(source) = self
            .deferred_values
            .get(&root)
            .and_then(|deferred| deferred.shards.first())
        else {
            return false;
        };
        self.value_shards(value)
            .ok()
            .and_then(|shards| shards.first())
            .and_then(|shard| {
                self.shards[shard.index() as usize]
                    .tensor_type
                    .shape
                    .0
                    .last()
            })
            .is_some_and(|width| width.is_multiple_of(2))
            && self.shards[source.index() as usize]
                .tensor_type
                .format
                .supports_f16_micro_panel_exchange(
                    &self.shards[destination.index() as usize].tensor_type.format,
                )
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
        retained_values: &[MidValueId],
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
            if self.defer_fused_output(operation, operations, retained_values, &mut tiles)? {
                tracing::info!(
                    operation = index,
                    source = ?operation.source.map(OperationId::index),
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    "deferred fused-operator materialization"
                );
                continue;
            }
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
            let lowered = match &operation.kind {
                MidOperationKind::Repeat(repeat) => {
                    self.lower_repeat(operation, repeat, &mut tiles)
                }
                MidOperationKind::Operator(_) => self.lower_operator(operation, &mut tiles),
                kind => self.lower_conversion(operation, kind, &mut tiles),
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
                    MidOperationKind::Operator(_) | MidOperationKind::Repeat(_)
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

    fn unpack_amp_to_row_major(
        &mut self,
        source: MidValueId,
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<Option<Vec<LowShardId>>> {
        let sources = self.value_shards(source)?.to_vec();
        for &source_shard in &sources {
            let source = &self.shards[source_shard.index() as usize];
            let compatible = source.extents.len() == 3
                && source.tensor_type.format.precision == Precision::F16
                && match source.tensor_type.format.layout.order {
                    ElementOrder::Amp(AmpOrder::Output) => {
                        let columns = source.extents[2];
                        (columns.physical_end - columns.start).is_multiple_of(AMP_COLUMN_MICRO)
                    }
                    ElementOrder::Amp(AmpOrder::TransposedLeft) => {
                        let rows = source.extents[1];
                        (rows.physical_end - rows.start).is_multiple_of(AMP_COLUMN_MICRO)
                    }
                    _ => false,
                };
            if !compatible {
                tracing::debug!(
                    shard = source_shard.index(),
                    rank = source.extents.len(),
                    precision = ?source.tensor_type.format.precision,
                    order = ?source.tensor_type.format.layout.order,
                    extents = ?source.extents,
                    "cannot unpack source storage into row-major order"
                );
                return Ok(None);
            }
        }

        let mut staging_shards = Vec::with_capacity(sources.len());
        for source_shard in sources {
            let source = self.shards[source_shard.index() as usize].clone();
            let mut staging_type = source.tensor_type.clone();
            staging_type.format.layout = Layout::row_major(TensorTiling::replicated(1));
            let staging = self.push_shard(LowShard {
                id: LowShardId(0),
                tile: source.tile,
                tensor_type: staging_type,
                extents: source.extents.clone(),
                definition: ShardDefinition::Staging,
            })?;
            self.append_kernel(
                tiles,
                source.tile,
                KernelRun::new(
                    provenance,
                    TileKernel::Planned(TileKernelSpec::Rearrange {
                        from: source.tensor_type.format.layout.clone(),
                        to: self.shards[staging.index() as usize]
                            .tensor_type
                            .format
                            .layout
                            .clone(),
                    }),
                    vec![KernelOperand {
                        views: vec![self.full_view(source_shard)],
                    }],
                    self.full_view(staging),
                    KernelRequirements::Conversion {
                        input: OperandRequirement::new(source.tensor_type.format, 4),
                        output: OperandRequirement::new(
                            self.shards[staging.index() as usize]
                                .tensor_type
                                .format
                                .clone(),
                            4,
                        ),
                        memory_relations: Vec::new(),
                    },
                ),
            )?;
            staging_shards.push(staging);
        }
        Ok(Some(staging_shards))
    }

    fn defer_fused_output(
        &mut self,
        operation: &MidOperation,
        operations: &[MidOperation],
        retained_values: &[MidValueId],
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<bool> {
        let Some(offered) = operation
            .operator_plan
            .as_ref()
            .and_then(|plan| plan.deferred_output)
        else {
            return Ok(false);
        };
        let [result] = operation.results.as_slice() else {
            return Ok(false);
        };
        let Some(source) = operation.inputs.get(offered.source_input) else {
            return Ok(false);
        };
        let uses = operations
            .iter()
            .flat_map(|operation| &operation.inputs)
            .chain(retained_values)
            .filter(|value| **value == *result)
            .count();
        let claimed = operations.iter().any(|candidate| {
            candidate.operator_plan.as_ref().is_some_and(|plan| {
                plan.deferred_inputs.iter().flatten().any(|input| {
                    input.producer == *result
                        && input.source == *source
                        && input.transform == offered.transform
                })
            })
        });
        if uses != 1 || !claimed {
            tracing::debug!(
                source = ?operation.source.map(OperationId::index),
                ?result,
                uses,
                claimed,
                "cannot defer fused output because its consumer contract is not exclusive"
            );
            return Ok(false);
        }
        let DeferredTransform::SplitLastAxisIntoLeading { parts } = offered.transform;
        let columns = self.shards[self.value_shards(*source)?[0].index() as usize]
            .tensor_type
            .shape
            .0[2];
        if parts == 0 || !columns.is_multiple_of(parts) {
            return Ok(false);
        }
        let source_shards = self.value_shards(*source)?.to_vec();
        let source_format = &self.shards[source_shards[0].index() as usize]
            .tensor_type
            .format;
        let result_format = &self.shards[self.value_shards(*result)?[0].index() as usize]
            .tensor_type
            .format;
        let direct_panel_exchange = source_format.supports_f16_micro_panel_exchange(result_format);
        tracing::debug!(
            source = ?operation.source.map(OperationId::index),
            source_order = ?source_format.layout.order,
            result_order = ?result_format.layout.order,
            direct_panel_exchange,
            "selected deferred-output storage"
        );
        let staging_shards = if direct_panel_exchange
            || !matches!(
                source_format.layout.order,
                ElementOrder::Amp(AmpOrder::Output | AmpOrder::TransposedLeft)
            ) {
            source_shards
        } else {
            let Some(staging) = self.unpack_amp_to_row_major(
                *source,
                WorkProvenance {
                    operation: operation.source,
                    value: Some(*result),
                    reason: WorkReason::OperatorKernel,
                },
                tiles,
            )?
            else {
                return Ok(false);
            };
            staging
        };
        self.deferred_values.insert(
            *result,
            DeferredValue {
                transform: offered.transform,
                shards: staging_shards,
            },
        );
        for shard in self.value_shards(*result)?.to_vec() {
            self.shards[shard.index() as usize].definition = ShardDefinition::Unmaterialized;
        }
        Ok(true)
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
        if !plan.strategy.uses_intersections() {
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
        match plan.strategy {
            ConversionStrategy::LocalKernel => {
                self.lower_local_conversion(operation, kind, plan, tiles)
            }
            ConversionStrategy::DirectRetile | ConversionStrategy::StageLogicalThenTransform => {
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
                        memory_relations: Vec::new(),
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
        let logical_order = match plan.strategy {
            ConversionStrategy::DirectRetile => false,
            ConversionStrategy::StageLogicalThenTransform => true,
            ConversionStrategy::LocalKernel => {
                return Err(LowLoweringError::InvalidConversionPlan);
            }
        };
        let mut mappings = Vec::new();
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
                mappings.push((
                    ShardView {
                        shard: source,
                        extents: extents.clone(),
                    },
                    ShardView {
                        shard: output,
                        extents,
                    },
                ));
            }
        }
        self.lower_mapped_views(
            mappings,
            logical_order,
            ExchangeOrder::Semantic,
            operation_provenance(operation, kind),
            tiles,
        )
    }

    fn lower_mapped_views(
        &mut self,
        mappings: Vec<(ShardView, ShardView)>,
        logical_order: bool,
        exchange_order: ExchangeOrder,
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
        let mut before_exchange = Vec::new();
        let mut after_exchange = Vec::new();
        let mut after_exchange_kernels = Vec::new();
        let mut grouped = BTreeMap::<LowShardId, Vec<(ShardView, ShardView)>>::new();
        for mapping in mappings {
            grouped.entry(mapping.1.shard).or_default().push(mapping);
        }
        for (destination_shard, mut mappings) in grouped {
            let destination_format = &self.shards[destination_shard.index() as usize]
                .tensor_type
                .format;
            let destination_unaligned = mappings.iter().try_fold(
                false,
                |unaligned, (_, destination)| -> LowLoweringResult<bool> {
                    let spans = logical_view_byte_spans(
                        &self.shards[destination.shard.index() as usize],
                        destination,
                    )?;
                    Ok(unaligned
                        || spans
                            .iter()
                            .any(|span| span.offset & 0b11 != 0 || span.bytes & 0b11 != 0))
                },
            )?;
            let requires_materialization = logical_order
                && (mappings.iter().any(|(source, _)| {
                    self.shards[source.shard.index() as usize]
                        .tensor_type
                        .format
                        .layout
                        .order
                        != destination_format.layout.order
                }) || destination_unaligned);
            let direct_logical = requires_materialization
                && self.mappings_benefit_from_word_exchange(&mappings, destination_shard)?;
            let stage_destination = requires_materialization && !direct_logical;
            if direct_logical && self.shard_has_padding(destination_shard) {
                self.append_fill_zero(tiles, destination_shard, provenance.clone())?;
            }
            let staging = if stage_destination {
                Some(self.push_conversion_staging(destination_shard)?)
            } else {
                None
            };
            for (mut source, mut destination) in mappings.drain(..) {
                if let Some(staging) = staging {
                    destination.shard = staging;
                    for extent in &mut source.extents {
                        extent.physical_end = extent.logical_end;
                    }
                    for extent in &mut destination.extents {
                        extent.physical_end = extent.logical_end;
                    }
                }
                let source_tile = self.shards[source.shard.index() as usize].tile;
                let destination_tile = self.shards[destination.shard.index() as usize].tile;
                if source_tile == destination_tile {
                    let copies = if staging.is_some() {
                        &mut before_exchange
                    } else {
                        &mut after_exchange
                    };
                    if logical_order {
                        append_logical_span_copies(
                            &self.shards,
                            &source,
                            &destination,
                            destination_tile,
                            copies,
                        )?;
                    } else {
                        append_span_copies(
                            &self.shards,
                            &source,
                            &destination,
                            destination_tile,
                            copies,
                        )?;
                    }
                } else {
                    transfers.entry(source).or_default().push(destination);
                }
            }
            if let Some(staging) = staging {
                let destination = self.logical_view(destination_shard);
                let staging = self.full_view(staging);
                let source_format = self.shards[staging.shard.index() as usize]
                    .tensor_type
                    .format
                    .clone();
                let destination_format = self.shards[destination_shard.index() as usize]
                    .tensor_type
                    .format
                    .clone();
                let tile = self.shards[destination_shard.index() as usize].tile;
                if source_format.precision == crate::Precision::F16
                    && source_format.layout.order == ElementOrder::RowMajor
                    && matches!(
                        destination_format.layout.order,
                        ElementOrder::Amp(AmpOrder::Left | AmpOrder::TransposedRight)
                            | ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. })
                    )
                {
                    after_exchange_kernels.push((
                        tile,
                        KernelRun::new(
                            provenance.clone(),
                            TileKernel::Planned(TileKernelSpec::Rearrange {
                                from: source_format.layout.clone(),
                                to: destination_format.layout.clone(),
                            }),
                            vec![KernelOperand {
                                views: vec![staging],
                            }],
                            self.full_view(destination_shard),
                            KernelRequirements::Conversion {
                                input: OperandRequirement::new(source_format, 2),
                                output: OperandRequirement::new(destination_format, 2),
                                memory_relations: vec![MemoryRelation::DistinctElements(vec![
                                    MemoryOperand::Input(0),
                                    MemoryOperand::Output,
                                ])],
                            },
                        ),
                    ));
                } else {
                    append_logical_span_copies(
                        &self.shards,
                        &staging,
                        &destination,
                        tile,
                        &mut after_exchange,
                    )?;
                }
            }
        }
        for (tile, copy) in before_exchange {
            self.append_local_copy(tiles, tile, copy)?;
        }
        self.append_ordered_phase(transfers, provenance, exchange_order, tiles)?;
        for (tile, copy) in after_exchange {
            self.append_local_copy(tiles, tile, copy)?;
        }
        for (tile, run) in after_exchange_kernels {
            self.append_kernel(tiles, tile, run)?;
        }
        Ok(())
    }

    fn push_conversion_staging(
        &mut self,
        destination: LowShardId,
    ) -> LowLoweringResult<LowShardId> {
        let destination = &self.shards[destination.index() as usize];
        let mut extents = destination.extents.clone();
        let tile = destination.tile;
        let shape = destination.tensor_type.shape.clone();
        let precision = destination.tensor_type.format.precision;
        for extent in &mut extents {
            extent.physical_end = extent.logical_end;
        }
        self.push_shard(LowShard {
            id: LowShardId(0),
            tile,
            tensor_type: TensorType {
                shape,
                format: crate::TensorFormat {
                    precision,
                    layout: Layout {
                        order: ElementOrder::RowMajor,
                        tiling: TensorTiling::replicated(1),
                        memory_class: MemoryClass::Ipu21Standard,
                    },
                },
            },
            extents,
            definition: ShardDefinition::Staging,
        })
    }

    fn logical_view(&self, shard: LowShardId) -> ShardView {
        let mut view = self.full_view(shard);
        for extent in &mut view.extents {
            extent.physical_end = extent.logical_end;
        }
        view
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
                distribution,
                orientation,
            } => self.lower_blocked_gemm(
                operation,
                initialize.clone(),
                accumulate.clone(),
                *inner_block,
                *output_column_block,
                *orientation,
                *distribution,
                &plan.requirements,
                tiles,
            ),
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
            OperatorDispatch::SplitHeads => {
                self.lower_split_heads(operation, &plan.operator, tiles)
            }
        }
    }

    fn lower_split_heads(
        &mut self,
        operation: &MidOperation,
        operator: &crate::MidOperator,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let crate::MidOperator::SplitHeads(options) = operator else {
            return Err(LowLoweringError::InvalidOperatorPlan);
        };
        let [input] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidOperatorPlan);
        };
        let [result] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        let input_type = self
            .value_shards(*input)?
            .first()
            .map(|shard| self.shards[shard.index() as usize].tensor_type.clone())
            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
        if input_type.shape.0.len() != 3 || options.heads == 0 {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let head_width = input_type.shape.0[2] / options.heads;
        if head_width == 0 || head_width * options.heads != input_type.shape.0[2] {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let output_shards = self.value_shards(*result)?.to_vec();
        let output_type = output_shards
            .first()
            .map(|shard| self.shards[shard.index() as usize].tensor_type.clone())
            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
        let original_sources = self.value_shards(*input)?.to_vec();
        let direct_panel_exchange = input_type
            .format
            .supports_f16_micro_panel_exchange(&output_type.format);
        tracing::debug!(
            source = ?operation.source.map(OperationId::index),
            source_order = ?input_type.format.layout.order,
            destination_order = ?output_type.format.layout.order,
            direct_panel_exchange,
            "selected split-head exchange strategy"
        );
        if direct_panel_exchange {
            let mappings = self.split_head_mappings(
                &original_sources,
                &output_shards,
                options.heads,
                head_width,
            )?;
            if let Some(mappings) = self.f16_micro_panel_mappings(mappings)? {
                tracing::info!(
                    source = ?operation.source.map(OperationId::index),
                    mappings = mappings.len(),
                    "lowering split transform as physical micro-panel exchange"
                );
                return self.lower_mapped_views(
                    mappings,
                    false,
                    ExchangeOrder::Physical,
                    WorkProvenance {
                        operation: operation.source,
                        value: Some(*result),
                        reason: WorkReason::OperatorInputs,
                    },
                    tiles,
                );
            }
        }

        let source_shards = if matches!(
            input_type.format.layout.order,
            ElementOrder::Amp(AmpOrder::Output | AmpOrder::TransposedLeft)
        ) {
            self.unpack_amp_to_row_major(
                *input,
                WorkProvenance {
                    operation: operation.source,
                    value: Some(*result),
                    reason: WorkReason::OperatorKernel,
                },
                tiles,
            )?
            .ok_or(LowLoweringError::InvalidOperatorPlan)?
        } else {
            self.value_shards(*input)?.to_vec()
        };

        let mappings =
            self.split_head_mappings(&source_shards, &output_shards, options.heads, head_width)?;
        self.lower_mapped_views(
            mappings,
            true,
            ExchangeOrder::Semantic,
            WorkProvenance {
                operation: operation.source,
                value: Some(*result),
                reason: WorkReason::OperatorInputs,
            },
            tiles,
        )
    }

    fn split_head_mappings(
        &self,
        source_shards: &[LowShardId],
        output_shards: &[LowShardId],
        heads: u32,
        head_width: u32,
    ) -> LowLoweringResult<Vec<(ShardView, ShardView)>> {
        let mut mappings = Vec::new();
        for &output in output_shards {
            let output_extents = self.shards[output.index() as usize].extents.clone();
            let tile = self.shards[output.index() as usize].tile;
            if output_extents.len() != 3 {
                return Err(LowLoweringError::InvalidOperatorPlan);
            }
            // A legal layout may assign several complete attention streams to
            // one tile. Split that allocation into per-stream views here: the
            // batch/head reshape is semantic, but it does not require the
            // ownership layout itself to shard the stream axis one-by-one.
            for stream in output_extents[0].start..output_extents[0].logical_end {
                let mut stream_extents = output_extents.clone();
                stream_extents[0].start = stream;
                stream_extents[0].logical_end = stream + 1;
                stream_extents[0].physical_end = stream + 1;
                let (target, column_base) =
                    split_head_source_extents(&stream_extents, heads, head_width)?;
                for (mut source_extents, source) in
                    self.intersecting_shard_set(&source_shards, &target, tile)
                {
                    let mut destination_extents = vec![
                        stream_extents[0],
                        source_extents[1],
                        ShardExtent {
                            axis: 2,
                            start: source_extents[2].start - column_base,
                            logical_end: source_extents[2].logical_end - column_base,
                            physical_end: source_extents[2].logical_end - column_base,
                        },
                    ];
                    let source_shard = &self.shards[source.index() as usize];
                    let complete_head = source_extents[2].start == column_base
                        && source_extents[2].logical_end == column_base + head_width
                        && source_shard.extents[2].start == column_base
                        && source_shard.extents[2].logical_end == column_base + head_width;
                    if complete_head {
                        let source_padding = source_shard.extents[2]
                            .physical_end
                            .saturating_sub(source_extents[2].logical_end);
                        let destination_padding = output_extents[2]
                            .physical_end
                            .saturating_sub(output_extents[2].logical_end);
                        let padding = source_padding.min(destination_padding);
                        source_extents[2].physical_end += padding;
                        destination_extents[2].physical_end += padding;
                    }
                    let source_view = ShardView {
                        shard: source,
                        extents: source_extents,
                    };
                    let destination_view = ShardView {
                        shard: output,
                        extents: destination_extents,
                    };
                    mappings.push((source_view, destination_view));
                }
            }
        }
        Ok(mappings)
    }

    /// Splits corresponding views at each allocation's F16 micro-panel
    /// boundaries. Within every resulting rectangle the source and
    /// destination have identical physical traversal, even when their outer
    /// panel sequence and tile ownership differ.
    fn f16_micro_panel_mappings(
        &self,
        mappings: Vec<(ShardView, ShardView)>,
    ) -> LowLoweringResult<Option<Vec<(ShardView, ShardView)>>> {
        let mut split = Vec::new();
        for (source, destination) in mappings {
            let source_shard = &self.shards[source.shard.index() as usize];
            let destination_shard = &self.shards[destination.shard.index() as usize];
            let pieces = split_mapping_at_panel_boundaries(
                source_shard,
                source,
                destination_shard,
                destination,
            )?;
            for (source, destination) in pieces {
                let source_spans = view_byte_spans(source_shard, &source)?;
                let destination_spans = view_byte_spans(destination_shard, &destination)?;
                let valid_spans = source_spans
                    .iter()
                    .chain(&destination_spans)
                    .all(|span| span.offset & 0b11 == 0 && span.bytes & 0b11 == 0);
                let source_bytes = source_spans.iter().map(|span| span.bytes).sum::<u32>();
                let destination_bytes =
                    destination_spans.iter().map(|span| span.bytes).sum::<u32>();
                if !valid_spans || source_bytes != destination_bytes {
                    return Ok(None);
                }
                split.push((source, destination));
            }
        }
        Ok(Some(split))
    }

    fn build_attention_tasks(
        &mut self,
        query: MidValueId,
        result: MidValueId,
        shape: AttentionBufferShape,
    ) -> LowLoweringResult<Vec<AttentionTask>> {
        let value_row_block = u16::try_from(shape.physical_staging_rows)
            .map_err(|_| LowLoweringError::InvalidOperatorPlan)?;
        let outputs = self.value_shards(result)?.to_vec();
        let mut tasks = Vec::with_capacity(outputs.len());
        for output in outputs {
            let tile = self.shards[output.index() as usize].tile;
            let rank = self.shards[output.index() as usize].extents.len();
            if rank != 3 {
                return Err(LowLoweringError::InvalidOperatorPlan);
            }
            let rows = self.shards[output.index() as usize].extents[rank - 2].physical_end
                - self.shards[output.index() as usize].extents[rank - 2].start;
            let value_dimension = *self.shards[output.index() as usize]
                .tensor_type
                .shape
                .0
                .last()
                .ok_or(LowLoweringError::InvalidOperatorPlan)?;
            if rows == 0 || rows > shape.query_block_rows {
                return Err(LowLoweringError::InvalidOperatorPlan);
            }
            let canonical_query = self.local_shard(query, tile)?;
            let query_dimension = *self.shards[canonical_query.index() as usize]
                .tensor_type
                .shape
                .0
                .last()
                .ok_or(LowLoweringError::InvalidOperatorPlan)?;
            let deferred_query = self.has_deferred_value(query);
            let query_shard = if deferred_query {
                self.push_attention_buffer(
                    tile,
                    rows,
                    rows,
                    query_dimension,
                    shape.padded_query_dimension,
                    ElementOrder::Amp(AmpOrder::Left),
                )?
            } else {
                canonical_query
            };
            let direct_query = deferred_query
                && (self.deferred_supports_physical_exchange(query, query_shard)
                    || self.deferred_panel_benefits_from_word_exchange(
                        query,
                        self.shards[output.index() as usize].extents[rank - 3].start,
                        self.shards[output.index() as usize].extents[rank - 2].start,
                        rows,
                        0,
                        query_dimension,
                        query_shard,
                    )?);
            let query_receive = (deferred_query && !direct_query)
                .then(|| {
                    self.push_attention_buffer(
                        tile,
                        rows,
                        rows,
                        query_dimension,
                        query_dimension,
                        ElementOrder::RowMajor,
                    )
                })
                .transpose()?;
            let scratch = self.push_attention_scratch(
                tile,
                rows,
                shape.scratch_columns,
                Precision::F16,
                ElementOrder::Amp(AmpOrder::Left),
                MemoryClass::Ipu21Interleaved,
            )?;
            let key_staging = self.push_attention_buffer(
                tile,
                shape.logical_staging_rows,
                shape.physical_staging_rows,
                query_dimension,
                shape.padded_query_dimension,
                ElementOrder::Amp(AmpOrder::TransposedRight),
            )?;
            self.shards[key_staging.index() as usize].definition = ShardDefinition::ExchangeStaging;
            let weights = self.push_attention_scratch(
                tile,
                rows,
                shape.state_columns,
                Precision::F16,
                ElementOrder::Amp(AmpOrder::Left),
                MemoryClass::Ipu21Standard,
            )?;
            if shape.reuse_key_staging_for_state
                && crate::shard_storage_bytes(&self.shards[weights.index() as usize])?
                    <= crate::shard_storage_bytes(&self.shards[key_staging.index() as usize])?
            {
                // Materialized QK consumes the packed K matrix before softmax
                // starts. Reinterpret that now-dead standard-memory allocation
                // as probabilities plus row state so the PV kernel retains its
                // proven standard-load path without a second large buffer.
                self.shards[weights.index() as usize].definition =
                    ShardDefinition::Alias(key_staging);
            }
            let value_staging = self.push_attention_buffer(
                tile,
                shape.logical_staging_rows,
                shape.physical_staging_rows,
                value_dimension,
                shape.padded_value_dimension,
                ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                    row_block: value_row_block,
                    column_block: AMP_COLUMN_MICRO as u16,
                }),
            )?;
            self.shards[value_staging.index() as usize].definition =
                ShardDefinition::ExchangeStaging;
            tasks.push(AttentionTask {
                tile,
                head: self.shards[output.index() as usize].extents[rank - 3].start,
                query_row_start: self.shards[output.index() as usize].extents[rank - 2].start,
                query_rows: rows,
                query_dimension,
                value_dimension,
                query: query_shard,
                query_receive,
                output,
                scratch,
                weights,
                key_staging,
                value_staging,
            });
        }
        Ok(tasks)
    }

    fn materialize_attention_queries(
        &mut self,
        query: MidValueId,
        tasks: &[AttentionTask],
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        if !self.has_deferred_value(query) {
            return Ok(());
        }
        let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
        let mut local_copies = Vec::new();
        let physical = tasks.iter().all(|task| task.query_receive.is_none());
        let physical = physical
            && tasks
                .iter()
                .all(|task| self.deferred_supports_physical_exchange(query, task.query));
        let direct = tasks.iter().all(|task| task.query_receive.is_none());
        if direct && !physical {
            for task in tasks {
                if self.shard_has_padding(task.query) {
                    self.append_fill_zero(tiles, task.query, provenance.clone())?;
                }
            }
        }
        for task in tasks {
            self.gather_deferred_panel(
                query,
                task.head,
                task.query_row_start,
                task.query_rows,
                0,
                task.query_dimension,
                task.query_receive.unwrap_or(task.query),
                if physical {
                    ExchangeOrder::Physical
                } else {
                    ExchangeOrder::Semantic
                },
                &mut transfers,
                &mut local_copies,
            )?;
        }
        for (tile, copy) in local_copies {
            self.append_local_copy(tiles, tile, copy)?;
        }
        self.append_ordered_phase(
            transfers,
            provenance,
            if physical {
                ExchangeOrder::Physical
            } else {
                ExchangeOrder::Semantic
            },
            tiles,
        )?;
        if !physical {
            for task in tasks {
                self.append_attention_rearrange(
                    tiles,
                    task.tile,
                    task.query_receive
                        .ok_or(LowLoweringError::InvalidOperatorPlan)?,
                    task.query,
                    WorkProvenance {
                        operation: provenance.operation,
                        value: provenance.value,
                        reason: WorkReason::OperatorKernel,
                    },
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_attention_blocks(
        &mut self,
        key: MidValueId,
        value: MidValueId,
        tasks: &[AttentionTask],
        key_rows: u32,
        block_rows: u32,
        padded_query_dimension: u32,
        padded_value_dimension: u32,
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<Vec<PreparedAttentionBlock>> {
        let blocks = key_rows.div_ceil(block_rows);
        let key_destinations = tasks.iter().fold(
            BTreeMap::<u32, Vec<LowShardId>>::new(),
            |mut destinations, task| {
                destinations
                    .entry(task.head)
                    .or_default()
                    .push(task.key_staging);
                destinations
            },
        );
        let value_destinations = tasks.iter().fold(
            BTreeMap::<u32, Vec<LowShardId>>::new(),
            |mut destinations, task| {
                destinations
                    .entry(task.head)
                    .or_default()
                    .push(task.value_staging);
                destinations
            },
        );
        let key_panel_count = padded_query_dimension.div_ceil(AMP_COLUMN_MICRO);
        let value_panel_count = padded_value_dimension.div_ceil(AMP_COLUMN_MICRO);
        let mut semantic_gathers = BTreeMap::<ShardView, Vec<ShardView>>::new();
        let mut physical_gathers = BTreeMap::<ShardView, Vec<ShardView>>::new();
        let mut prepared = Vec::new();
        for block in 0..blocks {
            let row_start = block * block_rows;
            let valid_rows = key_rows.saturating_sub(row_start).min(block_rows);
            let owner_offset = block.saturating_mul(key_panel_count + value_panel_count);
            let key_panels = self.prepare_distributed_attention_panels(
                key,
                &key_destinations,
                row_start,
                valid_rows,
                tasks[0].query_dimension,
                padded_query_dimension,
                ElementOrder::Amp(AmpOrder::TransposedRight),
                owner_offset,
                &mut semantic_gathers,
                &mut physical_gathers,
                provenance.clone(),
                tiles,
            )?;
            let row_block =
                u16::try_from(block_rows).map_err(|_| LowLoweringError::InvalidOperatorPlan)?;
            let value_panels = self.prepare_distributed_attention_panels(
                value,
                &value_destinations,
                row_start,
                valid_rows,
                tasks[0].value_dimension,
                padded_value_dimension,
                ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                    row_block,
                    column_block: AMP_COLUMN_MICRO as u16,
                }),
                owner_offset + key_panel_count,
                &mut semantic_gathers,
                &mut physical_gathers,
                provenance.clone(),
                tiles,
            )?;
            prepared.push(PreparedAttentionBlock {
                row_start,
                valid_rows,
                key_panels,
                value_panels,
            });
        }
        self.append_mixed_phase(semantic_gathers, physical_gathers, provenance, tiles)?;
        for block in &prepared {
            for panel in block.key_panels.iter().chain(&block.value_panels) {
                if let Some(row_major) = panel.row_major {
                    self.append_attention_rearrange(
                        tiles,
                        panel.tile,
                        row_major,
                        panel.packed,
                        WorkProvenance {
                            operation: provenance.operation,
                            value: provenance.value,
                            reason: WorkReason::OperatorKernel,
                        },
                    )?;
                }
            }
        }
        Ok(prepared)
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_blocked_attention(
        &mut self,
        operation: &MidOperation,
        query_key: TileKernelSpec,
        probability_value: TileKernelSpec,
        query_block_rows: u32,
        key_block_rows: u32,
        padded_query_dimension: u32,
        padded_value_dimension: u32,
        requirements: &OperatorRequirements,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [query, key, value] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidOperatorPlan);
        };
        let [result] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        if key_block_rows != AMP_INNER_BLOCK || query_block_rows == 0 {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let key_shards = self.value_shards(*key)?.to_vec();
        let value_shards = self.value_shards(*value)?.to_vec();
        if key_shards.len() != value_shards.len() {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        if self.has_deferred_value(*key) != self.has_deferred_value(*value) {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let tasks = self.build_attention_tasks(
            *query,
            *result,
            AttentionBufferShape {
                query_block_rows,
                logical_staging_rows: key_block_rows,
                physical_staging_rows: key_block_rows,
                scratch_columns: padded_value_dimension.max(key_block_rows),
                state_columns: key_block_rows + 16,
                padded_query_dimension,
                padded_value_dimension,
                reuse_key_staging_for_state: false,
            },
        )?;
        let key_rows = self.shards[self.value_shards(*key)?[0].index() as usize]
            .tensor_type
            .shape
            .0[1];
        let blocks = usize::try_from(key_rows.div_ceil(key_block_rows))
            .map_err(|_| LowLoweringError::IdOverflow)?;
        if blocks == 0 {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let exchange_provenance = WorkProvenance {
            operation: operation.source,
            value: Some(*key),
            reason: WorkReason::OperatorInputs,
        };
        let kernel_provenance = WorkProvenance {
            operation: operation.source,
            value: Some(*result),
            reason: WorkReason::OperatorKernel,
        };
        self.materialize_attention_queries(*query, &tasks, exchange_provenance, tiles)?;
        let deferred_key_value = self.has_deferred_value(*key);
        let prepared_blocks = if deferred_key_value {
            self.prepare_attention_blocks(
                *key,
                *value,
                &tasks,
                key_rows,
                key_block_rows,
                padded_query_dimension,
                padded_value_dimension,
                exchange_provenance,
                tiles,
            )?
        } else {
            Vec::new()
        };
        for block in 0..blocks {
            let block_start =
                u32::try_from(block).map_err(|_| LowLoweringError::IdOverflow)? * key_block_rows;
            let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
            let mut task_sources = Vec::with_capacity(tasks.len());
            if deferred_key_value {
                let prepared = prepared_blocks
                    .get(block)
                    .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                self.append_prepared_panel_broadcasts(
                    &prepared.key_panels,
                    0,
                    &mut transfers,
                    tiles,
                )?;
                self.append_prepared_panel_broadcasts(
                    &prepared.value_panels,
                    0,
                    &mut transfers,
                    tiles,
                )?;
                self.append_phase(transfers, exchange_provenance, tiles)?;
                task_sources.extend(
                    tasks
                        .iter()
                        .map(|task| (task.key_staging, task.value_staging, prepared.valid_rows)),
                );
            } else {
                for task in &tasks {
                    let source_matches = |candidate: &&LowShardId| {
                        let shard = &self.shards[candidate.index() as usize];
                        shard.extents[0].start == task.head && shard.extents[1].start == block_start
                    };
                    let key_source = *key_shards
                        .iter()
                        .find(source_matches)
                        .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                    let value_source = *value_shards
                        .iter()
                        .find(source_matches)
                        .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                    let valid_key_rows = self.shards[key_source.index() as usize].extents[1]
                        .logical_end
                        .saturating_sub(block_start);
                    let mut operands = Vec::with_capacity(2);
                    for (source, destination) in [
                        (key_source, task.key_staging),
                        (value_source, task.value_staging),
                    ] {
                        if self.shards[source.index() as usize].tile == task.tile {
                            operands.push(source);
                        } else {
                            transfers
                                .entry(self.full_view(source))
                                .or_default()
                                .push(self.full_view(destination));
                            operands.push(destination);
                        }
                    }
                    task_sources.push((operands[0], operands[1], valid_key_rows));
                }
                self.append_phase(transfers, exchange_provenance, tiles)?;
            }
            for (task, (key_operand, value_operand, valid_key_rows)) in
                tasks.iter().zip(task_sources)
            {
                let score_view = self.narrow_view(task.scratch, &[(1, 0, key_block_rows)])?;
                self.append_kernel(
                    tiles,
                    task.tile,
                    KernelRun::new(
                        kernel_provenance,
                        TileKernel::Planned(query_key.clone()),
                        vec![
                            KernelOperand {
                                views: vec![self.full_view(task.query)],
                            },
                            KernelOperand {
                                views: vec![self.full_view(key_operand)],
                            },
                        ],
                        score_view.clone(),
                        KernelRequirements::Operator(requirements.clone()),
                    ),
                )?;
                self.append_kernel(
                    tiles,
                    task.tile,
                    KernelRun::new(
                        kernel_provenance,
                        TileKernel::Planned(TileKernelSpec::AttentionSoftmax {
                            head_dimension: task.query_dimension,
                            key_columns: valid_key_rows,
                            padded_key_columns: key_block_rows,
                        }),
                        vec![KernelOperand {
                            views: vec![score_view],
                        }],
                        self.full_view(task.weights),
                        KernelRequirements::Operator(requirements.clone()),
                    ),
                )?;
                let probability_view = self.narrow_view(task.weights, &[(1, 0, key_block_rows)])?;
                let block_value_view =
                    self.narrow_view(task.scratch, &[(1, 0, padded_value_dimension)])?;
                self.append_kernel(
                    tiles,
                    task.tile,
                    KernelRun::new(
                        kernel_provenance,
                        TileKernel::Planned(probability_value.clone()),
                        vec![
                            KernelOperand {
                                views: vec![probability_view],
                            },
                            KernelOperand {
                                views: vec![self.full_view(value_operand)],
                            },
                        ],
                        block_value_view.clone(),
                        KernelRequirements::Operator(requirements.clone()),
                    ),
                )?;
                self.append_kernel(
                    tiles,
                    task.tile,
                    KernelRun::new(
                        kernel_provenance,
                        TileKernel::Planned(TileKernelSpec::AttentionMerge {
                            value_dimension: task.value_dimension,
                            padded_value_dimension,
                            key_block_columns: key_block_rows,
                            initial: block == 0,
                            final_block: block + 1 == blocks,
                        }),
                        vec![
                            KernelOperand {
                                views: vec![block_value_view],
                            },
                            KernelOperand {
                                views: vec![self.full_view(task.weights)],
                            },
                        ],
                        self.full_view(task.output),
                        KernelRequirements::Operator(requirements.clone()),
                    ),
                )?;
            }
        }
        Ok(())
    }

    fn append_materialized_attention_input(
        &mut self,
        operand: AttentionOperand,
        sources: &[LowShardId],
        prepared: &[PreparedAttentionBlock],
        tasks: &[AttentionTask],
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
        if prepared.is_empty() {
            for task in tasks {
                let destination = match operand {
                    AttentionOperand::Key => task.key_staging,
                    AttentionOperand::Value => task.value_staging,
                };
                let matching_sources = sources
                    .iter()
                    .copied()
                    .filter(|source| {
                        self.shards[source.index() as usize].extents[0].start == task.head
                    })
                    .collect::<Vec<_>>();
                for source in matching_sources {
                    let source_view = self.full_view(source);
                    let row_extent = self.shards[source.index() as usize].extents[1];
                    let destination_view = self.narrow_view(
                        destination,
                        &[(0, row_extent.start, row_extent.physical_end)],
                    )?;
                    let source_tile = self.shards[source.index() as usize].tile;
                    if source_tile == task.tile {
                        let mut copies = Vec::new();
                        append_span_copies(
                            &self.shards,
                            &source_view,
                            &destination_view,
                            task.tile,
                            &mut copies,
                        )?;
                        for (tile, copy) in copies {
                            self.append_local_copy(tiles, tile, copy)?;
                        }
                    } else {
                        transfers
                            .entry(source_view)
                            .or_default()
                            .push(destination_view);
                    }
                }
            }
        } else {
            for block in prepared {
                let panels = match operand {
                    AttentionOperand::Key => &block.key_panels,
                    AttentionOperand::Value => &block.value_panels,
                };
                self.append_prepared_panel_broadcasts(
                    panels,
                    block.row_start,
                    &mut transfers,
                    tiles,
                )?;
            }
        }
        self.append_physical_phase(transfers, provenance, tiles)
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_materialized_attention(
        &mut self,
        operation: &MidOperation,
        query_key: TileKernelSpec,
        probability_value: TileKernelSpec,
        query_block_rows: u32,
        padded_key_rows: u32,
        padded_query_dimension: u32,
        padded_value_dimension: u32,
        requirements: &OperatorRequirements,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [query, key, value] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidOperatorPlan);
        };
        let [result] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        if query_block_rows == 0
            || padded_key_rows == 0
            || !padded_key_rows.is_multiple_of(AMP_INNER_BLOCK)
            || self.has_deferred_value(*key) != self.has_deferred_value(*value)
        {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let key_shards = self.value_shards(*key)?.to_vec();
        let value_shards = self.value_shards(*value)?.to_vec();
        if key_shards.len() != value_shards.len() {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let key_rows = self.shards[key_shards[0].index() as usize]
            .tensor_type
            .shape
            .0[1];
        if key_rows == 0 || key_rows > padded_key_rows {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let tasks = self.build_attention_tasks(
            *query,
            *result,
            AttentionBufferShape {
                query_block_rows,
                logical_staging_rows: key_rows,
                physical_staging_rows: padded_key_rows,
                scratch_columns: padded_key_rows.max(padded_value_dimension),
                state_columns: padded_key_rows + AMP_COLUMN_MICRO,
                padded_query_dimension,
                padded_value_dimension,
                reuse_key_staging_for_state: true,
            },
        )?;
        if tasks.is_empty() {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let exchange_provenance = WorkProvenance {
            operation: operation.source,
            value: Some(*key),
            reason: WorkReason::OperatorInputs,
        };
        let kernel_provenance = WorkProvenance {
            operation: operation.source,
            value: Some(*result),
            reason: WorkReason::OperatorKernel,
        };
        self.materialize_attention_queries(*query, &tasks, exchange_provenance, tiles)?;
        let prepared = if self.has_deferred_value(*key) {
            self.prepare_attention_blocks(
                *key,
                *value,
                &tasks,
                key_rows,
                AMP_INNER_BLOCK,
                padded_query_dimension,
                padded_value_dimension,
                exchange_provenance,
                tiles,
            )?
        } else {
            Vec::new()
        };
        self.append_materialized_attention_input(
            AttentionOperand::Key,
            &key_shards,
            &prepared,
            &tasks,
            exchange_provenance,
            tiles,
        )?;
        for task in &tasks {
            let scores = self.narrow_view(task.scratch, &[(1, 0, padded_key_rows)])?;
            self.append_kernel(
                tiles,
                task.tile,
                KernelRun::new(
                    kernel_provenance,
                    TileKernel::Planned(query_key.clone()),
                    vec![
                        KernelOperand {
                            views: vec![self.full_view(task.query)],
                        },
                        KernelOperand {
                            views: vec![self.full_view(task.key_staging)],
                        },
                    ],
                    scores.clone(),
                    KernelRequirements::Operator(requirements.clone()),
                ),
            )?;
            self.append_kernel(
                tiles,
                task.tile,
                KernelRun::new(
                    kernel_provenance,
                    TileKernel::Planned(TileKernelSpec::AttentionSoftmax {
                        head_dimension: task.query_dimension,
                        key_columns: key_rows,
                        padded_key_columns: padded_key_rows,
                    }),
                    vec![KernelOperand {
                        views: vec![scores],
                    }],
                    self.full_view(task.weights),
                    KernelRequirements::Operator(requirements.clone()),
                ),
            )?;
        }
        self.append_materialized_attention_input(
            AttentionOperand::Value,
            &value_shards,
            &prepared,
            &tasks,
            exchange_provenance,
            tiles,
        )?;
        for task in &tasks {
            let probabilities = self.narrow_view(task.weights, &[(1, 0, padded_key_rows)])?;
            let block_value = self.narrow_view(task.scratch, &[(1, 0, padded_value_dimension)])?;
            self.append_kernel(
                tiles,
                task.tile,
                KernelRun::new(
                    kernel_provenance,
                    TileKernel::Planned(probability_value.clone()),
                    vec![
                        KernelOperand {
                            views: vec![probabilities],
                        },
                        KernelOperand {
                            views: vec![self.full_view(task.value_staging)],
                        },
                    ],
                    block_value.clone(),
                    KernelRequirements::Operator(requirements.clone()),
                ),
            )?;
            self.append_kernel(
                tiles,
                task.tile,
                KernelRun::new(
                    kernel_provenance,
                    TileKernel::Planned(TileKernelSpec::AttentionMerge {
                        value_dimension: task.value_dimension,
                        padded_value_dimension,
                        key_block_columns: padded_key_rows,
                        initial: true,
                        final_block: true,
                    }),
                    vec![
                        KernelOperand {
                            views: vec![block_value],
                        },
                        KernelOperand {
                            views: vec![self.full_view(task.weights)],
                        },
                    ],
                    self.full_view(task.output),
                    KernelRequirements::Operator(requirements.clone()),
                ),
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn deferred_panel_mappings(
        &self,
        value: MidValueId,
        stream: u32,
        row_start: u32,
        rows: u32,
        column_start: u32,
        columns: u32,
        destination: LowShardId,
    ) -> LowLoweringResult<Vec<(ShardView, ShardView)>> {
        let deferred_root = self
            .deferred_root(value)
            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
        let deferred = self
            .deferred_values
            .get(&deferred_root)
            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
        let deferred_shards = deferred.shards.clone();
        let logical_type = &self.shards[self.value_shards(value)?[0].index() as usize].tensor_type;
        let source_type = &self.shards[deferred.shards[0].index() as usize].tensor_type;
        let logical_target = [
            (stream, stream + 1),
            (row_start, row_start + rows),
            (column_start, column_start + columns),
        ];
        let mapping = deferred
            .transform
            .map_slice(&source_type.shape, &logical_type.shape, &logical_target)
            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
        let target = mapping
            .source_ranges
            .iter()
            .copied()
            .enumerate()
            .map(|(axis, (start, end))| {
                Ok(ShardExtent {
                    axis: u16::try_from(axis).map_err(|_| LowLoweringError::IdOverflow)?,
                    start,
                    logical_end: end,
                    physical_end: end,
                })
            })
            .collect::<LowLoweringResult<Vec<_>>>()?;
        let destination_tile = self.shards[destination.index() as usize].tile;
        let mut covered = 0u64;
        let mut mappings = Vec::new();
        for (source_extents, source) in
            self.intersecting_shard_set(&deferred_shards, &target, destination_tile)
        {
            let destination_extents = mapping
                .destination_source_axes
                .iter()
                .enumerate()
                .map(|(destination_axis, &source_axis)| {
                    let source = source_extents
                        .get(source_axis)
                        .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                    let base = target
                        .get(source_axis)
                        .ok_or(LowLoweringError::InvalidOperatorPlan)?
                        .start;
                    Ok(ShardExtent {
                        axis: u16::try_from(destination_axis)
                            .map_err(|_| LowLoweringError::IdOverflow)?,
                        start: source.start - base,
                        logical_end: source.logical_end - base,
                        physical_end: source.logical_end - base,
                    })
                })
                .collect::<LowLoweringResult<Vec<_>>>()?;
            covered = covered.saturating_add(
                u64::from(source_extents[1].logical_end - source_extents[1].start)
                    * u64::from(source_extents[2].logical_end - source_extents[2].start),
            );
            let source_view = ShardView {
                shard: source,
                extents: source_extents,
            };
            let destination_view = ShardView {
                shard: destination,
                extents: destination_extents,
            };
            mappings.push((source_view, destination_view));
        }
        if covered != u64::from(rows) * u64::from(columns) {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        Ok(mappings)
    }

    fn mapping_word_exchange_fragments(
        &self,
        mappings: &[(ShardView, ShardView)],
    ) -> LowLoweringResult<Option<u64>> {
        let maximum_bytes = ipu_exchange::MAX_TRANSFER_WORDS
            .checked_mul(4)
            .ok_or(LowLoweringError::IdOverflow)?;
        let mut fragments = 0u64;
        for (source, destination) in mappings {
            let source_spans =
                logical_view_byte_spans(&self.shards[source.shard.index() as usize], source)?;
            let destination_spans = logical_view_byte_spans(
                &self.shards[destination.shard.index() as usize],
                destination,
            )?;
            let aligned = source_spans
                .iter()
                .chain(&destination_spans)
                .all(|span| span.offset & 0b11 == 0 && span.bytes & 0b11 == 0);
            let source_bytes = source_spans.iter().map(|span| span.bytes).sum::<u32>();
            let destination_bytes = destination_spans.iter().map(|span| span.bytes).sum::<u32>();
            if !aligned || source_bytes != destination_bytes {
                tracing::trace!(
                    source = ?source,
                    destination = ?destination,
                    source_order = ?self.shards[source.shard.index() as usize]
                        .tensor_type.format.layout.order,
                    destination_order = ?self.shards[destination.shard.index() as usize]
                        .tensor_type.format.layout.order,
                    source_spans = ?source_spans,
                    destination_spans = ?destination_spans,
                    aligned,
                    source_bytes,
                    destination_bytes,
                    "deferred logical fragment cannot be exchanged directly"
                );
                return Ok(None);
            }
            let mut source_index = 0usize;
            let mut destination_index = 0usize;
            let mut source_offset = 0u32;
            let mut destination_offset = 0u32;
            while source_index < source_spans.len() && destination_index < destination_spans.len() {
                let source_remaining = source_spans[source_index].bytes - source_offset;
                let destination_remaining =
                    destination_spans[destination_index].bytes - destination_offset;
                let bytes = source_remaining
                    .min(destination_remaining)
                    .min(maximum_bytes);
                if bytes == 0 || bytes & 0b11 != 0 {
                    return Ok(None);
                }
                fragments = fragments.saturating_add(1);
                source_offset += bytes;
                destination_offset += bytes;
                if source_offset == source_spans[source_index].bytes {
                    source_index += 1;
                    source_offset = 0;
                }
                if destination_offset == destination_spans[destination_index].bytes {
                    destination_index += 1;
                    destination_offset = 0;
                }
            }
            if source_index != source_spans.len()
                || destination_index != destination_spans.len()
                || source_offset != 0
                || destination_offset != 0
            {
                return Ok(None);
            }
        }
        Ok(Some(fragments))
    }

    fn mappings_benefit_from_word_exchange(
        &self,
        mappings: &[(ShardView, ShardView)],
        destination: LowShardId,
    ) -> LowLoweringResult<bool> {
        let Some(fragments) = self.mapping_word_exchange_fragments(mappings)? else {
            return Ok(false);
        };
        let shard = &self.shards[destination.index() as usize];
        let bytes = u64::from(crate::shard_storage_bytes(shard)?);
        let elements = bytes.div_ceil(shard.tensor_type.format.precision.bytes().max(1));
        let packed_cycles = crate::cost::row_major_pack_cycles(&shard.tensor_type, elements);
        let clear_cycles = if self.shard_has_padding(destination) {
            crate::cost::IPU21_TARGET_COSTS
                .kernel_launch_cycles
                .saturating_add(bytes.div_ceil(8 * 6))
        } else {
            0
        };
        let fragment_cycles = fragments
            .saturating_mul(crate::cost::IPU21_LOGICAL_FRAGMENT_CYCLES)
            .saturating_add(clear_cycles);
        let direct = fragment_cycles < packed_cycles;
        tracing::trace!(
            destination = destination.index(),
            fragments,
            fragment_cycles,
            packed_cycles,
            direct,
            "selected logical conversion materialization"
        );
        Ok(direct)
    }

    #[allow(clippy::too_many_arguments)]
    fn deferred_panel_benefits_from_word_exchange(
        &self,
        value: MidValueId,
        stream: u32,
        row_start: u32,
        rows: u32,
        column_start: u32,
        columns: u32,
        destination: LowShardId,
    ) -> LowLoweringResult<bool> {
        let mappings = self.deferred_panel_mappings(
            value,
            stream,
            row_start,
            rows,
            column_start,
            columns,
            destination,
        )?;
        self.mappings_benefit_from_word_exchange(&mappings, destination)
    }

    #[allow(clippy::too_many_arguments)]
    fn gather_deferred_panel(
        &self,
        value: MidValueId,
        stream: u32,
        row_start: u32,
        rows: u32,
        column_start: u32,
        columns: u32,
        destination: LowShardId,
        order: ExchangeOrder,
        transfers: &mut BTreeMap<ShardView, Vec<ShardView>>,
        local_copies: &mut Vec<(u16, LocalCopy)>,
    ) -> LowLoweringResult<()> {
        let destination_tile = self.shards[destination.index() as usize].tile;
        let mappings = self.deferred_panel_mappings(
            value,
            stream,
            row_start,
            rows,
            column_start,
            columns,
            destination,
        )?;
        for (source_view, destination_view) in mappings {
            let mappings = if order == ExchangeOrder::Physical {
                self.f16_micro_panel_mappings(vec![(source_view, destination_view)])?
                    .ok_or(LowLoweringError::InvalidOperatorPlan)?
            } else {
                vec![(source_view, destination_view)]
            };
            for (source_view, destination_view) in mappings {
                if self.shards[source_view.shard.index() as usize].tile == destination_tile {
                    if order == ExchangeOrder::Physical {
                        append_span_copies(
                            &self.shards,
                            &source_view,
                            &destination_view,
                            destination_tile,
                            local_copies,
                        )?;
                    } else {
                        append_logical_span_copies(
                            &self.shards,
                            &source_view,
                            &destination_view,
                            destination_tile,
                            local_copies,
                        )?;
                    }
                } else {
                    transfers
                        .entry(source_view)
                        .or_default()
                        .push(destination_view);
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_distributed_attention_panels(
        &mut self,
        value: MidValueId,
        destinations: &BTreeMap<u32, Vec<LowShardId>>,
        block_start: u32,
        valid_rows: u32,
        logical_columns: u32,
        physical_columns: u32,
        order: ElementOrder,
        owner_offset: u32,
        semantic_gathers: &mut BTreeMap<ShardView, Vec<ShardView>>,
        physical_gathers: &mut BTreeMap<ShardView, Vec<ShardView>>,
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<Vec<PreparedDistributedPanel>> {
        let panels = physical_columns.div_ceil(AMP_COLUMN_MICRO);
        if panels == 0 || valid_rows == 0 {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let mut packed_panels = Vec::new();
        for (&stream, stream_destinations) in destinations {
            for panel in 0..panels {
                let column_start = panel * AMP_COLUMN_MICRO;
                let panel_columns = logical_columns
                    .saturating_sub(column_start)
                    .min(AMP_COLUMN_MICRO);
                if panel_columns == 0 {
                    continue;
                }
                let owner = usize::try_from(owner_offset.saturating_add(panel))
                    .map_err(|_| LowLoweringError::IdOverflow)?
                    % stream_destinations.len();
                let tile = stream_destinations[owner];
                let tile = self.shards[tile.index() as usize].tile;
                let packed = self.push_attention_buffer(
                    tile,
                    valid_rows,
                    AMP_INNER_BLOCK,
                    panel_columns,
                    AMP_COLUMN_MICRO,
                    order,
                )?;
                let physical = self.deferred_supports_physical_exchange(value, packed);
                let word_exchange = !physical
                    && self.deferred_panel_benefits_from_word_exchange(
                        value,
                        stream,
                        block_start,
                        valid_rows,
                        column_start,
                        panel_columns,
                        packed,
                    )?;
                if word_exchange && self.shard_has_padding(packed) {
                    self.append_fill_zero(tiles, packed, provenance.clone())?;
                }
                let row_major = if physical || word_exchange {
                    None
                } else {
                    Some(self.push_attention_buffer(
                        tile,
                        valid_rows,
                        valid_rows,
                        panel_columns,
                        panel_columns,
                        ElementOrder::RowMajor,
                    )?)
                };
                let gather_destination = row_major.unwrap_or(packed);
                let mut local_copies = Vec::new();
                self.gather_deferred_panel(
                    value,
                    stream,
                    block_start,
                    valid_rows,
                    column_start,
                    panel_columns,
                    gather_destination,
                    if physical {
                        ExchangeOrder::Physical
                    } else {
                        ExchangeOrder::Semantic
                    },
                    if physical {
                        physical_gathers
                    } else {
                        semantic_gathers
                    },
                    &mut local_copies,
                )?;
                for (tile, copy) in local_copies {
                    self.append_local_copy(tiles, tile, copy)?;
                }
                packed_panels.push(PreparedDistributedPanel {
                    panel,
                    row_major,
                    packed,
                    tile,
                    destinations: stream_destinations.clone(),
                });
            }
        }
        Ok(packed_panels)
    }

    fn append_prepared_panel_broadcasts(
        &mut self,
        panels: &[PreparedDistributedPanel],
        destination_row_start: u32,
        broadcasts: &mut BTreeMap<ShardView, Vec<ShardView>>,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        for panel in panels {
            let source = self.full_view(panel.packed);
            let source_rows = source.extents[0].physical_end - source.extents[0].start;
            let column_start = panel.panel * AMP_COLUMN_MICRO;
            for &destination in &panel.destinations {
                let destination_tile = self.shards[destination.index() as usize].tile;
                let destination_view = self.narrow_view(
                    destination,
                    &[
                        (
                            0,
                            destination_row_start,
                            destination_row_start + source_rows,
                        ),
                        (1, column_start, column_start + AMP_COLUMN_MICRO),
                    ],
                )?;
                if panel.tile == destination_tile {
                    let mut copies = Vec::new();
                    append_span_copies(
                        &self.shards,
                        &source,
                        &destination_view,
                        panel.tile,
                        &mut copies,
                    )?;
                    for (tile, copy) in copies {
                        self.append_local_copy(tiles, tile, copy)?;
                    }
                } else {
                    broadcasts
                        .entry(source.clone())
                        .or_insert_with(Vec::new)
                        .push(destination_view);
                }
            }
        }
        Ok(())
    }

    fn append_attention_rearrange(
        &mut self,
        tiles: &mut [TileWorkList],
        tile: u16,
        source: LowShardId,
        destination: LowShardId,
        provenance: WorkProvenance,
    ) -> LowLoweringResult<()> {
        let input = self.shards[source.index() as usize]
            .tensor_type
            .format
            .clone();
        let output = self.shards[destination.index() as usize]
            .tensor_type
            .format
            .clone();
        self.append_kernel(
            tiles,
            tile,
            KernelRun::new(
                provenance,
                TileKernel::Planned(TileKernelSpec::Rearrange {
                    from: input.layout.clone(),
                    to: output.layout.clone(),
                }),
                vec![KernelOperand {
                    views: vec![self.full_view(source)],
                }],
                self.full_view(destination),
                KernelRequirements::Conversion {
                    input: OperandRequirement::new(input, 2),
                    output: OperandRequirement::new(output, 2),
                    memory_relations: Vec::new(),
                },
            ),
        )
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
                    memory_relations: Vec::new(),
                }),
            ),
        )
    }

    fn push_attention_scratch(
        &mut self,
        tile: u16,
        rows: u32,
        columns: u32,
        precision: Precision,
        order: ElementOrder,
        memory_class: MemoryClass,
    ) -> LowLoweringResult<LowShardId> {
        self.push_shard(LowShard {
            id: LowShardId(0),
            tile,
            tensor_type: TensorType::new(
                [rows, columns],
                precision,
                Layout {
                    order,
                    tiling: TensorTiling::replicated(1),
                    memory_class,
                },
            ),
            extents: vec![
                ShardExtent {
                    axis: 0,
                    start: 0,
                    logical_end: rows,
                    physical_end: rows,
                },
                ShardExtent {
                    axis: 1,
                    start: 0,
                    logical_end: columns,
                    physical_end: columns,
                },
            ],
            definition: ShardDefinition::Staging,
        })
    }

    fn push_packed_buffer(
        &mut self,
        tile: u16,
        elements: u32,
        precision: Precision,
        definition: ShardDefinition,
    ) -> LowLoweringResult<LowShardId> {
        self.push_shard(LowShard {
            id: LowShardId(0),
            tile,
            tensor_type: TensorType::new(
                [elements],
                precision,
                Layout {
                    order: ElementOrder::RowMajor,
                    tiling: TensorTiling::replicated(1),
                    memory_class: MemoryClass::Ipu21Standard,
                },
            ),
            extents: vec![ShardExtent {
                axis: 0,
                start: 0,
                logical_end: elements,
                physical_end: elements,
            }],
            definition,
        })
    }

    fn push_attention_buffer(
        &mut self,
        tile: u16,
        logical_rows: u32,
        physical_rows: u32,
        logical_columns: u32,
        physical_columns: u32,
        order: ElementOrder,
    ) -> LowLoweringResult<LowShardId> {
        self.push_shard(LowShard {
            id: LowShardId(0),
            tile,
            tensor_type: TensorType::new(
                [logical_rows, logical_columns],
                Precision::F16,
                Layout {
                    order,
                    tiling: TensorTiling::replicated(1),
                    memory_class: MemoryClass::Ipu21Standard,
                },
            ),
            extents: vec![
                ShardExtent {
                    axis: 0,
                    start: 0,
                    logical_end: logical_rows,
                    physical_end: physical_rows,
                },
                ShardExtent {
                    axis: 1,
                    start: 0,
                    logical_end: logical_columns,
                    physical_end: physical_columns,
                },
            ],
            definition: ShardDefinition::Staging,
        })
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

    #[allow(clippy::too_many_arguments)]
    fn lower_blocked_gemm(
        &mut self,
        operation: &MidOperation,
        initialize: TileKernelSpec,
        accumulate: TileKernelSpec,
        inner_block: u32,
        output_column_block: u32,
        orientation: crate::GemmOrientation,
        distribution: GemmDistribution,
        requirements: &OperatorRequirements,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        if let GemmDistribution::ParallelReduction {
            row_partitions,
            column_partitions,
            inner_partitions,
            result_row_partitions,
            result_column_partitions,
            reduction_staging,
        } = distribution
        {
            return self.lower_parallel_reduction_gemm(
                operation,
                initialize,
                accumulate,
                inner_block,
                output_column_block,
                orientation,
                row_partitions,
                column_partitions,
                inner_partitions,
                result_row_partitions,
                result_column_partitions,
                reduction_staging,
                requirements,
                tiles,
            );
        }
        if orientation != crate::GemmOrientation::Normal {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
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
        let left_layout = left_type.format.layout.resolve(&left_type.shape)?;
        let output_layout = output_type.format.layout.resolve(&output_type.shape)?;
        let inner_extent = left_layout.padded_shape().0[left_rank - 1];
        let column_extent = output_layout.padded_shape().0[output_rank - 1];
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
                                            pattern: LocalCopyPattern::Contiguous,
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
    fn lower_parallel_reduction_gemm(
        &mut self,
        operation: &MidOperation,
        initialize: TileKernelSpec,
        accumulate: TileKernelSpec,
        inner_block: u32,
        output_column_block: u32,
        orientation: crate::GemmOrientation,
        row_partitions: u16,
        column_partitions: u16,
        inner_partitions: u16,
        result_row_partitions: u16,
        result_column_partitions: u16,
        reduction_staging: crate::ReductionStaging,
        requirements: &OperatorRequirements,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [semantic_left_value, semantic_right_value] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidOperatorPlan);
        };
        let [output_value] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        if inner_block == 0
            || output_column_block == 0
            || row_partitions == 0
            || column_partitions == 0
            || inner_partitions < 2
            || result_row_partitions == 0
            || result_column_partitions == 0
            || result_row_partitions.saturating_mul(result_column_partitions) > inner_partitions
        {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let (left_value, right_value, left_requirement, right_requirement) = match orientation {
            crate::GemmOrientation::Normal => (
                semantic_left_value,
                semantic_right_value,
                &requirements.inputs[0],
                &requirements.inputs[1],
            ),
            crate::GemmOrientation::Swapped => (
                semantic_right_value,
                semantic_left_value,
                &requirements.inputs[1],
                &requirements.inputs[0],
            ),
        };
        let mut kernel_requirements = requirements.clone();
        if orientation == crate::GemmOrientation::Swapped {
            kernel_requirements.inputs.swap(0, 1);
        }
        let left_shards = self.value_shards(*left_value)?.to_vec();
        let right_shards = self.value_shards(*right_value)?.to_vec();
        let output_shards = self.value_shards(*output_value)?.to_vec();
        let left_rank = self.shards[left_shards[0].index() as usize].extents.len();
        let right_rank = self.shards[right_shards[0].index() as usize].extents.len();
        let output_rank = self.shards[output_shards[0].index() as usize].extents.len();
        if left_rank < 2 || output_rank < 2 {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let (
            left_row_axis,
            left_inner_axis,
            right_inner_axis,
            right_column_axis,
            output_row_axis,
            output_column_axis,
        ) = match orientation {
            crate::GemmOrientation::Normal => (
                left_rank - 2,
                left_rank - 1,
                right_rank - 2,
                right_rank - 1,
                output_rank - 2,
                output_rank - 1,
            ),
            crate::GemmOrientation::Swapped => (
                left_rank - 1,
                left_rank - 2,
                right_rank - 1,
                right_rank - 2,
                output_rank - 1,
                output_rank - 2,
            ),
        };
        let output_type = self.shards[output_shards[0].index() as usize]
            .tensor_type
            .clone();
        let logical_columns = output_type.shape.0[output_column_axis];
        let output_layout = output_type.format.layout.resolve(&output_type.shape)?;
        let physical_columns = output_layout.padded_shape().0[output_column_axis];
        let column_grain = output_type
            .format
            .layout
            .tiling
            .axes
            .iter()
            .find(|axis| axis.axis.resolve(output_rank).ok() == Some(output_column_axis))
            .map(|axis| axis.block_size)
            .filter(|grain| *grain != 0)
            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
        let column_tiling = output_type
            .format
            .layout
            .tiling
            .axes
            .iter()
            .find(|axis| axis.axis.resolve(output_rank).ok() == Some(output_column_axis))
            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
        let column_blocks = physical_columns / column_grain;
        if !physical_columns.is_multiple_of(column_grain)
            || column_blocks < u32::from(column_partitions)
        {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let short_blocks = column_blocks / u32::from(column_partitions);
        let long_partitions = column_blocks % u32::from(column_partitions);
        let columns = (0..u32::from(column_partitions))
            .map(|partition| {
                if column_tiling.partitions == column_partitions {
                    return output_layout
                        .axis_bounds(output_column_axis, partition)
                        .ok_or(LowLoweringError::InvalidOperatorPlan);
                }
                let start_blocks = partition
                    .saturating_mul(short_blocks)
                    .saturating_add(partition.min(long_partitions));
                let blocks = short_blocks + u32::from(partition < long_partitions);
                let start = start_blocks.saturating_mul(column_grain);
                let physical_end = start_blocks
                    .saturating_add(blocks)
                    .saturating_mul(column_grain);
                Ok((
                    start,
                    physical_end.min(logical_columns).max(start),
                    physical_end,
                ))
            })
            .collect::<LowLoweringResult<Vec<_>>>()?;
        let mut partial_type = output_type.clone();
        let partial_tiles = row_partitions.saturating_mul(column_partitions);
        partial_type.format.layout = match (orientation, output_type.format.layout.order) {
            (crate::GemmOrientation::Normal, ElementOrder::Amp(AmpOrder::Left)) => {
                Layout::amp_left_result_grid(
                    output_column_block,
                    partial_tiles,
                    row_partitions,
                    column_partitions,
                    crate::mid::GridOrder::ColumnsFast,
                )
            }
            (crate::GemmOrientation::Swapped, ElementOrder::Amp(AmpOrder::TransposedLeft)) => {
                Layout::amp_transposed_left_result_grid(
                    output_column_block,
                    partial_tiles,
                    row_partitions,
                    column_partitions,
                    crate::mid::GridOrder::ColumnsFast,
                )
            }
            (crate::GemmOrientation::Normal, _) => Layout::amp_output_grid(
                output_column_block,
                partial_tiles,
                row_partitions,
                column_partitions,
                crate::mid::GridOrder::ColumnsFast,
            ),
            (crate::GemmOrientation::Swapped, _) => Layout::amp_transposed_output_grid(
                output_column_block,
                partial_tiles,
                row_partitions,
                column_partitions,
                crate::mid::GridOrder::ColumnsFast,
            ),
        };
        if let Some(axis) = partial_type
            .format
            .layout
            .tiling
            .axes
            .iter_mut()
            .find(|axis| axis.axis.resolve(output_rank).ok() == Some(output_column_axis))
        {
            axis.block_size = column_grain;
            axis.padding_multiple = column_grain;
            if column_tiling.partitions == column_partitions {
                axis.block_size = column_tiling.block_size;
                axis.padding_multiple = column_tiling.padding_multiple;
                axis.shard_padding_multiple = column_tiling.shard_padding_multiple;
            }
        }

        let mut replica_groups = BTreeMap::<Vec<(u32, u32)>, Vec<LowShardId>>::new();
        for left in left_shards.iter().copied() {
            let key = self.shards[left.index() as usize]
                .extents
                .iter()
                .map(|extent| (extent.start, extent.physical_end))
                .collect::<Vec<_>>();
            replica_groups.entry(key).or_default().push(left);
        }
        let mut replica_columns = BTreeMap::<LowShardId, u16>::new();
        for replicas in replica_groups.values_mut() {
            replicas.sort_unstable_by_key(|shard| self.shards[shard.index() as usize].tile);
            if replicas.len() != usize::from(column_partitions) {
                return Err(LowLoweringError::InvalidOperatorPlan);
            }
            for (column, shard) in replicas.iter().copied().enumerate() {
                replica_columns.insert(
                    shard,
                    u16::try_from(column).map_err(|_| LowLoweringError::IdOverflow)?,
                );
            }
        }
        {
            let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
            let mut local_copies = Vec::<(u16, LocalCopy)>::new();
            let mut gemm_runs = Vec::<(u16, KernelRun)>::new();
            let mut partials = BTreeMap::<Vec<(u32, u32)>, Vec<(u16, ShardView)>>::new();
            let mut resident_lefts = BTreeMap::<LowShardId, ShardView>::new();
            let mut weight_staging = BTreeMap::<(u16, LowShardId), LowShardId>::new();
            for (output_column, &(column_start, logical_column_end, column_end)) in
                columns.iter().enumerate()
            {
                let output_column =
                    u32::try_from(output_column).map_err(|_| LowLoweringError::IdOverflow)?;
                let local_output_columns = column_end - column_start;
                if local_output_columns == 0
                    || local_output_columns > output_column_block
                    || !local_output_columns.is_multiple_of(crate::mid::AMP_COLUMN_MICRO)
                {
                    return Err(LowLoweringError::InvalidOperatorPlan);
                }
                for left in left_shards.iter().copied() {
                    let left_shard = self.shards[left.index() as usize].clone();
                    let resident_left = if let Some(view) = resident_lefts.get(&left) {
                        view.clone()
                    } else {
                        let restrictions = left_shard
                            .extents
                            .iter()
                            .enumerate()
                            .map(|(axis, extent)| (axis, extent.start, extent.physical_end))
                            .collect::<Vec<_>>();
                        let view = self.dispatch_input_view(
                            *left_value,
                            left_shard.tile,
                            &restrictions,
                            &mut transfers,
                            &mut local_copies,
                        )?;
                        if left_requirement.materialization
                            != crate::OperandMaterialization::DispatchSlices
                            && view.shard != left
                        {
                            return Err(LowLoweringError::InvalidOperatorPlan);
                        }
                        resident_lefts.insert(left, view.clone());
                        view
                    };
                    let inner = left_shard.extents[left_inner_axis];
                    if !(inner.physical_end - inner.start).is_multiple_of(inner_block) {
                        return Err(LowLoweringError::InvalidOperatorPlan);
                    }
                    if replica_columns.get(&left).copied().map(u32::from) != Some(output_column) {
                        continue;
                    }
                    let partial_layout = partial_type.format.layout.resolve(&partial_type.shape)?;
                    let padded_output = partial_layout.padded_shape();
                    let mut extents = partial_type
                        .shape
                        .0
                        .iter()
                        .zip(&padded_output.0)
                        .enumerate()
                        .map(|(axis, (&logical_end, &physical_end))| ShardExtent {
                            axis: u16::try_from(axis).unwrap_or(u16::MAX),
                            start: 0,
                            logical_end,
                            physical_end,
                        })
                        .collect::<Vec<_>>();
                    if orientation == crate::GemmOrientation::Normal {
                        for axis in 0..output_rank.saturating_sub(2) {
                            extents[axis] = left_shard.extents[axis];
                            extents[axis].axis =
                                u16::try_from(axis).map_err(|_| LowLoweringError::IdOverflow)?;
                        }
                    }
                    extents[output_row_axis] = left_shard.extents[left_row_axis];
                    extents[output_row_axis].axis =
                        u16::try_from(output_row_axis).map_err(|_| LowLoweringError::IdOverflow)?;
                    extents[output_column_axis] = ShardExtent {
                        axis: u16::try_from(output_column_axis)
                            .map_err(|_| LowLoweringError::IdOverflow)?,
                        start: column_start,
                        logical_end: logical_column_end,
                        physical_end: column_end,
                    };
                    let partial_key = extents
                        .iter()
                        .map(|extent| (extent.start, extent.physical_end))
                        .collect::<Vec<_>>();
                    let direct_output = output_shards.iter().copied().find(|output| {
                        let shard = &self.shards[output.index() as usize];
                        shard.tile == left_shard.tile
                            && shard.tensor_type.format.layout.order
                                == partial_type.format.layout.order
                            && shard.tensor_type.format.layout.memory_class
                                == partial_type.format.layout.memory_class
                            && shard.extents.iter().zip(&extents).all(|(owner, partial)| {
                                owner.start <= partial.start
                                    && owner.logical_end >= partial.logical_end
                                    && owner.physical_end >= partial.physical_end
                            })
                    });
                    let partial = if let Some(output) = direct_output {
                        ShardView {
                            shard: output,
                            extents: extents.clone(),
                        }
                    } else {
                        let partial = self.push_shard(LowShard {
                            id: LowShardId(0),
                            tile: left_shard.tile,
                            tensor_type: partial_type.clone(),
                            extents,
                            definition: ShardDefinition::Staging,
                        })?;
                        self.full_view(partial)
                    };
                    partials
                        .entry(partial_key)
                        .or_default()
                        .push((left_shard.tile, partial.clone()));

                    let source_panel_block = right_requirement.format.layout.order.clone();
                    let source_panel_block = match source_panel_block {
                        ElementOrder::BlockMajor(
                            BlockMajorOrder::Matrix {
                                row_block: block, ..
                            }
                            | BlockMajorOrder::TransposedMatrix {
                                row_block: block, ..
                            },
                        ) => u32::from(block),
                        _ => AMP_INNER_BLOCK,
                    };
                    let first_panel_end = inner
                        .start
                        .saturating_add(source_panel_block)
                        .min(inner.physical_end);
                    let first_source = self
                        .matrix_shards_for_block(
                            &right_shards,
                            right_column_axis,
                            right_inner_axis,
                            column_start,
                            column_end,
                            inner.start,
                            first_panel_end,
                        )
                        .next()
                        .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                    let mut weight_type = self.shards[first_source.index() as usize]
                        .tensor_type
                        .clone();
                    weight_type.format.layout.memory_class = crate::MemoryClass::Ipu21Interleaved;
                    let mut weight_extents =
                        self.shards[first_source.index() as usize].extents.clone();
                    weight_extents[right_inner_axis].start = inner.start;
                    weight_extents[right_inner_axis].logical_end = inner.logical_end;
                    weight_extents[right_inner_axis].physical_end = inner.physical_end;
                    let source_inner =
                        self.shards[first_source.index() as usize].extents[right_inner_axis];
                    let source_covers_compute_inner = source_inner.start <= inner.start
                        && source_inner.physical_end >= inner.physical_end;
                    let stage_local_sources =
                        right_requirement.local_staging == crate::LocalOperandStaging::MatchRemote;
                    let weights = if self.shards[first_source.index() as usize].tile
                        == left_shard.tile
                        && source_covers_compute_inner
                        && !stage_local_sources
                    {
                        None
                    } else {
                        let key = (left_shard.tile, first_source);
                        if let Some(staging) = weight_staging.get(&key).copied() {
                            Some(staging)
                        } else {
                            let staging = self.push_shard(LowShard {
                                id: LowShardId(0),
                                tile: left_shard.tile,
                                tensor_type: weight_type,
                                extents: weight_extents,
                                definition: ShardDefinition::ExchangeStaging,
                            })?;
                            weight_staging.insert(key, staging);
                            Some(staging)
                        }
                    };

                    for (block_index, inner_start) in (inner.start..inner.physical_end)
                        .step_by(inner_block as usize)
                        .enumerate()
                    {
                        let inner_end = inner_start + inner_block;
                        let mut sources = Vec::new();
                        for panel_start in
                            (inner_start..inner_end).step_by(source_panel_block as usize)
                        {
                            let panel_end = panel_start + source_panel_block;
                            let source = self
                                .matrix_shards_for_block(
                                    &right_shards,
                                    right_column_axis,
                                    right_inner_axis,
                                    column_start,
                                    column_end,
                                    panel_start,
                                    panel_end,
                                )
                                .next()
                                .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                            let target_view = self.narrow_view(
                                source,
                                &[
                                    (right_inner_axis, panel_start, panel_end),
                                    (right_column_axis, column_start, column_end),
                                ],
                            )?;
                            let source_is_local =
                                self.shards[source.index() as usize].tile == left_shard.tile;
                            let consume_direct = source_is_local && !stage_local_sources;
                            if !consume_direct {
                                let destination_view = self.narrow_view(
                                    weights.ok_or(LowLoweringError::InvalidOperatorPlan)?,
                                    &[
                                        (right_inner_axis, panel_start, panel_end),
                                        (right_column_axis, column_start, column_end),
                                    ],
                                )?;
                                if source_is_local {
                                    append_logical_span_copies(
                                        &self.shards,
                                        &target_view,
                                        &destination_view,
                                        left_shard.tile,
                                        &mut local_copies,
                                    )?;
                                } else {
                                    transfers
                                        .entry(target_view.clone())
                                        .or_default()
                                        .push(destination_view);
                                }
                            }
                            sources.push((target_view, consume_direct));
                        }

                        if sources.len() > 1 && sources.iter().any(|(_, local)| *local) {
                            for (panel_index, (source_view, local)) in
                                sources.into_iter().enumerate()
                            {
                                let panel_start = inner_start
                                    + u32::try_from(panel_index)
                                        .map_err(|_| LowLoweringError::IdOverflow)?
                                        * source_panel_block;
                                let panel_end = panel_start + source_panel_block;
                                let left_view = self.narrow_view(
                                    resident_left.shard,
                                    &[(left_inner_axis, panel_start, panel_end)],
                                )?;
                                let mut kernel = if block_index == 0 && panel_index == 0 {
                                    initialize.clone()
                                } else {
                                    accumulate.clone()
                                };
                                if let TileKernelSpec::Gemm {
                                    weights: load,
                                    inner_block: kernel_inner_block,
                                    output_columns: kernel_output_columns,
                                    ..
                                } = &mut kernel
                                {
                                    *kernel_inner_block = source_panel_block;
                                    *kernel_output_columns = local_output_columns;
                                    let selected = if local {
                                        source_view.shard
                                    } else {
                                        weights.ok_or(LowLoweringError::InvalidOperatorPlan)?
                                    };
                                    *load = if self.shards[selected.index() as usize]
                                        .tensor_type
                                        .format
                                        .layout
                                        .memory_class
                                        == crate::MemoryClass::Ipu21Standard
                                    {
                                        crate::GemmWeightLoad::Standard
                                    } else {
                                        crate::GemmWeightLoad::Interleaved
                                    };
                                }
                                let weight_view = if local {
                                    source_view
                                } else {
                                    self.narrow_view(
                                        weights.ok_or(LowLoweringError::InvalidOperatorPlan)?,
                                        &[
                                            (right_inner_axis, panel_start, panel_end),
                                            (right_column_axis, column_start, column_end),
                                        ],
                                    )?
                                };
                                gemm_runs.push((
                                    left_shard.tile,
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
                                                views: vec![weight_view],
                                            },
                                        ],
                                        partial.clone(),
                                        KernelRequirements::Operator(kernel_requirements.clone()),
                                    ),
                                ));
                            }
                            continue;
                        }

                        let left_view = self.narrow_view(
                            resident_left.shard,
                            &[(left_inner_axis, inner_start, inner_end)],
                        )?;
                        let mut kernel = if block_index == 0 {
                            initialize.clone()
                        } else {
                            accumulate.clone()
                        };
                        if let TileKernelSpec::Gemm {
                            weights: load,
                            inner_block: kernel_inner_block,
                            output_columns: kernel_output_columns,
                            ..
                        } = &mut kernel
                        {
                            *kernel_inner_block = inner_block;
                            *kernel_output_columns = local_output_columns;
                            let selected = if sources.len() == 1 && sources[0].1 {
                                sources[0].0.shard
                            } else {
                                weights.ok_or(LowLoweringError::InvalidOperatorPlan)?
                            };
                            *load = if self.shards[selected.index() as usize]
                                .tensor_type
                                .format
                                .layout
                                .memory_class
                                == crate::MemoryClass::Ipu21Standard
                            {
                                crate::GemmWeightLoad::Standard
                            } else {
                                crate::GemmWeightLoad::Interleaved
                            };
                        }
                        let weight_view = if sources.len() == 1 && sources[0].1 {
                            sources.pop().expect("one source").0
                        } else {
                            self.narrow_view(
                                weights.ok_or(LowLoweringError::InvalidOperatorPlan)?,
                                &[
                                    (right_inner_axis, inner_start, inner_end),
                                    (right_column_axis, column_start, column_end),
                                ],
                            )?
                        };
                        let run = KernelRun::new(
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
                                    views: vec![weight_view],
                                },
                            ],
                            partial.clone(),
                            KernelRequirements::Operator(kernel_requirements.clone()),
                        );
                        gemm_runs.push((left_shard.tile, run));
                    }
                }
            }
            self.append_phase(
                transfers,
                WorkProvenance {
                    operation: operation.source,
                    value: Some(*right_value),
                    reason: WorkReason::OperatorInput {
                        input: match orientation {
                            crate::GemmOrientation::Normal => 1,
                            crate::GemmOrientation::Swapped => 0,
                        },
                    },
                },
                tiles,
            )?;
            for (tile, copy) in local_copies {
                self.append_local_copy(tiles, tile, copy)?;
            }
            for (tile, run) in gemm_runs {
                self.append_kernel(tiles, tile, run)?;
            }

            tracing::debug!(
                partial_groups = partials.len(),
                output_shards = output_shards.len(),
                result_row_partitions,
                result_column_partitions,
                "prepared parallel GEMM partials"
            );

            let remote_partials_per_stage = match reduction_staging {
                crate::ReductionStaging::Complete => inner_partitions.saturating_sub(1),
                crate::ReductionStaging::Streamed => 1,
            };
            let reduction_stages = inner_partitions
                .saturating_sub(1)
                .div_ceil(remote_partials_per_stage.max(1));
            let mut reduction_transfers = (0..reduction_stages)
                .map(|_| BTreeMap::<ShardView, Vec<ShardView>>::new())
                .collect::<Vec<_>>();
            let mut seed_copies = Vec::<(u16, LocalCopy)>::new();
            let mut reduction_runs = (0..reduction_stages)
                .map(|_| Vec::<(u16, KernelRun)>::new())
                .collect::<Vec<_>>();
            let mut result_copies = Vec::<(u16, LocalCopy)>::new();
            let mut reduction_roots = 0usize;
            for contributors in partials.into_values() {
                let Some((_, complete)) = contributors.first() else {
                    return Err(LowLoweringError::InvalidOperatorPlan);
                };
                let expected = complete
                    .extents
                    .iter()
                    .try_fold(1u64, |elements, extent| {
                        elements.checked_mul(u64::from(extent.physical_end - extent.start))
                    })
                    .ok_or(LowLoweringError::IdOverflow)?;
                let mut covered = 0u64;
                for output in output_shards.iter().copied() {
                    let owner = self.shards[output.index() as usize].clone();
                    let intersection =
                        intersect_extents_with_shared_padding(&owner.extents, &complete.extents);
                    let Some(intersection) = intersection else {
                        continue;
                    };
                    let elements = intersection
                        .iter()
                        .try_fold(1u32, |elements, extent| {
                            elements.checked_mul(extent.physical_end - extent.start)
                        })
                        .ok_or(LowLoweringError::IdOverflow)?;
                    if elements == 0 || !elements.is_multiple_of(8) {
                        return Err(LowLoweringError::InvalidOperatorPlan);
                    }
                    covered = covered
                        .checked_add(u64::from(elements))
                        .ok_or(LowLoweringError::IdOverflow)?;

                    let initial = self.push_packed_buffer(
                        owner.tile,
                        elements,
                        Precision::F16,
                        ShardDefinition::Staging,
                    )?;
                    let remote_elements = elements
                        .checked_mul(u32::from(remote_partials_per_stage))
                        .ok_or(LowLoweringError::IdOverflow)?;
                    let remote = self.push_packed_buffer(
                        owner.tile,
                        remote_elements,
                        Precision::F16,
                        ShardDefinition::ExchangeStaging,
                    )?;
                    let result = self.push_packed_buffer(
                        owner.tile,
                        elements,
                        Precision::F16,
                        ShardDefinition::Staging,
                    )?;
                    let seed = contributors
                        .iter()
                        .position(|(tile, _)| *tile == owner.tile)
                        .unwrap_or(0);
                    let source_view = |partial: &ShardView| ShardView {
                        shard: partial.shard,
                        extents: intersection.clone(),
                    };
                    let seed_source = source_view(&contributors[seed].1);
                    if contributors[seed].0 == owner.tile {
                        append_span_copies(
                            &self.shards,
                            &seed_source,
                            &self.full_view(initial),
                            owner.tile,
                            &mut seed_copies,
                        )?;
                    } else {
                        reduction_transfers[0]
                            .entry(seed_source)
                            .or_default()
                            .push(self.full_view(initial));
                    }

                    let remote_contributors = contributors
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| *index != seed)
                        .map(|(_, (_, partial))| partial)
                        .collect::<Vec<_>>();
                    for (stage, chunk) in remote_contributors
                        .chunks(usize::from(remote_partials_per_stage))
                        .enumerate()
                    {
                        for (slot, partial) in chunk.iter().enumerate() {
                            let start = u32::try_from(slot)
                                .map_err(|_| LowLoweringError::IdOverflow)?
                                .checked_mul(elements)
                                .ok_or(LowLoweringError::IdOverflow)?;
                            let end = start
                                .checked_add(elements)
                                .ok_or(LowLoweringError::IdOverflow)?;
                            reduction_transfers[stage]
                                .entry(source_view(partial))
                                .or_default()
                                .push(ShardView {
                                    shard: remote,
                                    extents: vec![ShardExtent {
                                        axis: 0,
                                        start,
                                        logical_end: end,
                                        physical_end: end,
                                    }],
                                });
                        }
                        let (accumulator, stage_result) = if stage.is_multiple_of(2) {
                            (initial, result)
                        } else {
                            (result, initial)
                        };
                        reduction_runs[stage].push((
                            owner.tile,
                            KernelRun::new(
                                WorkProvenance {
                                    operation: operation.source,
                                    value: Some(*output_value),
                                    reason: WorkReason::OperatorKernel,
                                },
                                TileKernel::Planned(TileKernelSpec::ReductionSum {
                                    partials: u16::try_from(chunk.len() + 1)
                                        .map_err(|_| LowLoweringError::IdOverflow)?,
                                }),
                                vec![
                                    KernelOperand {
                                        views: vec![self.full_view(accumulator)],
                                    },
                                    KernelOperand {
                                        views: vec![self.full_view(remote)],
                                    },
                                ],
                                self.full_view(stage_result),
                                KernelRequirements::Operator(requirements.clone()),
                            ),
                        ));
                    }
                    let final_result = if usize::from(reduction_stages).is_multiple_of(2) {
                        initial
                    } else {
                        result
                    };
                    append_span_copies(
                        &self.shards,
                        &self.full_view(final_result),
                        &ShardView {
                            shard: output,
                            extents: intersection,
                        },
                        owner.tile,
                        &mut result_copies,
                    )?;
                    reduction_roots += 1;
                }
                if covered != expected {
                    return Err(LowLoweringError::InvalidOperatorPlan);
                }
            }
            for (stage, (transfers, runs)) in reduction_transfers
                .into_iter()
                .zip(reduction_runs)
                .enumerate()
            {
                self.append_physical_phase(
                    transfers,
                    WorkProvenance {
                        operation: operation.source,
                        value: Some(*output_value),
                        reason: WorkReason::OperatorInputs,
                    },
                    tiles,
                )?;
                if stage == 0 {
                    for (tile, copy) in seed_copies.drain(..) {
                        self.append_local_copy(tiles, tile, copy)?;
                    }
                }
                for (tile, run) in runs {
                    self.append_kernel(tiles, tile, run)?;
                }
            }
            for (tile, copy) in result_copies {
                self.append_local_copy(tiles, tile, copy)?;
            }
            tracing::debug!(reduction_roots, "materialized packed parallel reduction");
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
        let left_layout = left_type.format.layout.resolve(&left_type.shape)?;
        let output_layout = output_type.format.layout.resolve(&output_type.shape)?;
        let inner_extent = left_layout.padded_shape().0[left_rank - 1];
        let column_extent = output_layout.padded_shape().0[output_rank - 1];
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
                                                pattern: LocalCopyPattern::Contiguous,
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
        let body = self.lower_region(&repeat.body.operations, &repeat.body.yields, false)?;
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
    let mut pending = Vec::new();
    let mut source_index = 0usize;
    let mut destination_index = 0usize;
    let mut source_offset = 0u32;
    let mut destination_offset = 0u32;
    while source_index < source_spans.len() && destination_index < destination_spans.len() {
        let source_span = source_spans[source_index];
        let destination_span = destination_spans[destination_index];
        let bytes =
            (source_span.bytes - source_offset).min(destination_span.bytes - destination_offset);
        pending.push(LocalCopy {
            source: source.shard,
            source_offset: source_span.offset + source_offset,
            destination: destination.shard,
            destination_offset: destination_span.offset + destination_offset,
            bytes,
            pattern: LocalCopyPattern::Contiguous,
        });
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
    copies.extend(
        coalesce_local_copies(pending)
            .into_iter()
            .map(|copy| (tile, copy)),
    );
    Ok(())
}

const PARALLEL_STRIDED_COPY_MAX_BYTES: u32 = 512;

fn coalesce_local_copies(copies: Vec<LocalCopy>) -> Vec<LocalCopy> {
    let mut coalesced = Vec::new();
    let mut index = 0;
    while index < copies.len() {
        let first = &copies[index];
        let Some(second) = copies.get(index + 1) else {
            coalesced.push(first.clone());
            break;
        };
        if first.source != second.source
            || first.destination != second.destination
            || first.bytes != second.bytes
            || first.bytes == 0
            || !first.bytes.is_multiple_of(8)
        {
            coalesced.push(first.clone());
            index += 1;
            continue;
        }
        let source_stride = second.source_offset.saturating_sub(first.source_offset);
        let destination_stride = second
            .destination_offset
            .saturating_sub(first.destination_offset);
        if source_stride == 0 || destination_stride == 0 {
            coalesced.push(first.clone());
            index += 1;
            continue;
        }
        let mut end = index + 2;
        while let Some(copy) = copies.get(end) {
            let previous = &copies[end - 1];
            if copy.source != first.source
                || copy.destination != first.destination
                || copy.bytes != first.bytes
                || copy.source_offset.checked_sub(previous.source_offset) != Some(source_stride)
                || copy
                    .destination_offset
                    .checked_sub(previous.destination_offset)
                    != Some(destination_stride)
            {
                break;
            }
            end += 1;
        }
        let rows = u32::try_from(end - index).unwrap_or(u32::MAX);
        // Larger strided regions are deliberately left as contiguous rows:
        // spreading them over workers loses more to bank contention than it
        // saves in call overhead on IPU21.
        if first.bytes.saturating_mul(rows) > PARALLEL_STRIDED_COPY_MAX_BYTES {
            coalesced.extend(copies[index..end].iter().cloned());
            index = end;
            continue;
        }
        if source_stride == first.bytes && destination_stride == first.bytes {
            let mut copy = first.clone();
            copy.bytes = copy.bytes.saturating_mul(rows);
            coalesced.push(copy);
        } else {
            let mut copy = first.clone();
            copy.bytes = copy.bytes.saturating_mul(rows);
            copy.pattern = LocalCopyPattern::Strided {
                rows,
                row_bytes: first.bytes,
                source_stride,
                destination_stride,
            };
            coalesced.push(copy);
        }
        index = end;
    }
    coalesced
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

fn intersect_extents_with_shared_padding(
    left: &[ShardExtent],
    right: &[ShardExtent],
) -> Option<Vec<ShardExtent>> {
    if left.len() != right.len() {
        return None;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let start = left.start.max(right.start);
            let logical_end = left.logical_end.min(right.logical_end);
            (start < logical_end).then(|| {
                let shared_tail =
                    if logical_end == left.logical_end && logical_end == right.logical_end {
                        left.physical_end
                            .saturating_sub(left.logical_end)
                            .min(right.physical_end.saturating_sub(right.logical_end))
                    } else {
                        0
                    };
                ShardExtent {
                    axis: left.axis,
                    start,
                    logical_end,
                    physical_end: logical_end + shared_tail,
                }
            })
        })
        .collect()
}

fn shard_extents(tensor_type: &TensorType) -> LowLoweringResult<Vec<(u16, Vec<ShardExtent>)>> {
    Ok(tensor_type
        .format
        .layout
        .resolve(&tensor_type.shape)?
        .shard_extents()
        .into_iter()
        .map(|shard| (shard.tile, shard.extents))
        .collect())
}

fn split_head_source_extents(
    output: &[ShardExtent],
    heads: u32,
    head_width: u32,
) -> LowLoweringResult<(Vec<ShardExtent>, u32)> {
    if output.len() != 3
        || heads == 0
        || output[0].logical_end - output[0].start != 1
        || output[2].logical_end > head_width
    {
        return Err(LowLoweringError::InvalidOperatorPlan);
    }
    let stream = output[0].start;
    let batch = stream / heads;
    let head = stream % heads;
    let column_base = head
        .checked_mul(head_width)
        .ok_or(LowLoweringError::IdOverflow)?;
    Ok((
        vec![
            ShardExtent {
                axis: 0,
                start: batch,
                logical_end: batch + 1,
                physical_end: batch + 1,
            },
            ShardExtent {
                axis: 1,
                start: output[1].start,
                logical_end: output[1].logical_end,
                physical_end: output[1].logical_end,
            },
            ShardExtent {
                axis: 2,
                start: column_base + output[2].start,
                logical_end: column_base + output[2].logical_end,
                physical_end: column_base + output[2].logical_end,
            },
        ],
        column_base,
    ))
}

fn split_mapping_at_panel_boundaries(
    source_shard: &LowShard,
    mut source: ShardView,
    destination_shard: &LowShard,
    mut destination: ShardView,
) -> LowLoweringResult<Vec<(ShardView, ShardView)>> {
    let source_rank = source.extents.len();
    let destination_rank = destination.extents.len();
    let outer_elements = |extents: &[ShardExtent]| {
        extents[..extents.len().saturating_sub(2)]
            .iter()
            .try_fold(1_u32, |elements, extent| {
                elements.checked_mul(extent.logical_end - extent.start)
            })
    };
    if source_rank < 2
        || destination_rank < 2
        || source_shard.extents.len() != source_rank
        || destination_shard.extents.len() != destination_rank
        || outer_elements(&source.extents) != Some(1)
        || outer_elements(&destination.extents) != Some(1)
    {
        return Err(LowLoweringError::InvalidOperatorPlan);
    }

    let aligned_ranges = |source: ShardExtent,
                          source_shard: ShardExtent,
                          destination: ShardExtent,
                          destination_shard: ShardExtent|
     -> LowLoweringResult<Vec<(ShardExtent, ShardExtent)>> {
        let logical_width = source.logical_end - source.start;
        if logical_width != destination.logical_end - destination.start {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let width = source.physical_end - source.start;
        if width != destination.physical_end - destination.start {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let mut ranges = Vec::new();
        let mut offset = 0;
        while offset < width {
            let source_position = source
                .start
                .checked_sub(source_shard.start)
                .and_then(|start| start.checked_add(offset))
                .ok_or(LowLoweringError::InvalidOperatorPlan)?;
            let destination_position = destination
                .start
                .checked_sub(destination_shard.start)
                .and_then(|start| start.checked_add(offset))
                .ok_or(LowLoweringError::InvalidOperatorPlan)?;
            let source_remaining = AMP_COLUMN_MICRO - source_position % AMP_COLUMN_MICRO;
            let destination_remaining = AMP_COLUMN_MICRO - destination_position % AMP_COLUMN_MICRO;
            let length = (width - offset)
                .min(source_remaining)
                .min(destination_remaining);
            let source_start = source.start + offset;
            let destination_start = destination.start + offset;
            ranges.push((
                ShardExtent {
                    axis: source.axis,
                    start: source_start,
                    logical_end: source
                        .logical_end
                        .min(source_start + length)
                        .max(source_start),
                    physical_end: source_start + length,
                },
                ShardExtent {
                    axis: destination.axis,
                    start: destination_start,
                    logical_end: destination
                        .logical_end
                        .min(destination_start + length)
                        .max(destination_start),
                    physical_end: destination_start + length,
                },
            ));
            offset += length;
        }
        Ok(ranges)
    };

    let source_row_axis = source_rank - 2;
    let source_column_axis = source_rank - 1;
    let destination_row_axis = destination_rank - 2;
    let destination_column_axis = destination_rank - 1;

    // The global row tail can finish part-way through a micro-panel while
    // both allocations contain padding through the same panel boundary.
    // Carry that padding with the useful values so the direct physical
    // exchange remains word-aligned. A split head's column tail is not
    // extended because the following source columns may belong to another
    // head rather than padding.
    let source_rows = source.extents[source_row_axis];
    let destination_rows = destination.extents[destination_row_axis];
    if source_rows.logical_end == source_shard.tensor_type.shape.0[source_row_axis]
        && destination_rows.logical_end
            == destination_shard.tensor_type.shape.0[destination_row_axis]
    {
        let source_panel_tail = (AMP_COLUMN_MICRO
            - (source_rows.logical_end - source_shard.extents[source_row_axis].start)
                % AMP_COLUMN_MICRO)
            % AMP_COLUMN_MICRO;
        let destination_panel_tail = (AMP_COLUMN_MICRO
            - (destination_rows.logical_end
                - destination_shard.extents[destination_row_axis].start)
                % AMP_COLUMN_MICRO)
            % AMP_COLUMN_MICRO;
        let padding = source_panel_tail
            .min(destination_panel_tail)
            .min(source_shard.extents[source_row_axis].physical_end - source_rows.logical_end)
            .min(
                destination_shard.extents[destination_row_axis].physical_end
                    - destination_rows.logical_end,
            );
        source.extents[source_row_axis].physical_end += padding;
        destination.extents[destination_row_axis].physical_end += padding;
    }

    let rows = aligned_ranges(
        source.extents[source_row_axis],
        source_shard.extents[source_row_axis],
        destination.extents[destination_row_axis],
        destination_shard.extents[destination_row_axis],
    )?;
    let columns = aligned_ranges(
        source.extents[source_column_axis],
        source_shard.extents[source_column_axis],
        destination.extents[destination_column_axis],
        destination_shard.extents[destination_column_axis],
    )?;
    let mut pieces = Vec::with_capacity(rows.len().saturating_mul(columns.len()));
    for (source_row, destination_row) in rows {
        for &(source_column, destination_column) in &columns {
            let mut source_extents = source.extents.clone();
            let mut destination_extents = destination.extents.clone();
            source_extents[source_row_axis] = source_row;
            source_extents[source_column_axis] = source_column;
            destination_extents[destination_row_axis] = destination_row;
            destination_extents[destination_column_axis] = destination_column;
            pieces.push((
                ShardView {
                    shard: source.shard,
                    extents: source_extents,
                },
                ShardView {
                    shard: destination.shard,
                    extents: destination_extents,
                },
            ));
        }
    }
    Ok(pieces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AccumulationPrecision, AxisTiling, ComputeGraph, ElementOrder, GemmDistribution,
        GemmKernelMode, GemmWeightLoad, GridOrder, Ipu21CostModel, Layout, MemoryClass,
        MidOperator, OperandRequirement, OperatorCandidate, OperatorDispatch, Padding,
        PipelineConfig, Precision, TensorAxis, TensorFormat, TensorTiling, TileKernelSpec, lower,
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
    fn randomized_linear_shards_cover_flat_storage_once_in_balanced_grains() {
        let mut random = fastrand::Rng::with_seed(0x666c_6174_5f73_6864);
        for case in 0..CASES {
            let rank = random.usize(2..=4);
            let grain = 1_u32 << random.u32(1..=5);
            let mut shape = (0..rank - 1).map(|_| random.u32(1..=5)).collect::<Vec<_>>();
            shape.push(grain * random.u32(1..=8));
            let elements = shape
                .iter()
                .map(|&extent| u64::from(extent))
                .product::<u64>();
            let grains = elements / u64::from(grain);
            let tiles = random.u16(1..=u16::try_from(grains.min(64)).unwrap());
            let tensor = TensorType::new(
                shape.clone(),
                Precision::F16,
                Layout::logical_linear(tiles, grain),
            );
            let shards = shard_extents(&tensor).unwrap();
            let mut coverage = vec![0_u8; usize::try_from(elements).unwrap()];
            let mut tile_elements = vec![0_u64; usize::from(tiles)];
            for (tile, extents) in shards {
                assert_eq!(extents.len(), rank, "case {case}");
                assert!(
                    extents[..rank - 1]
                        .iter()
                        .all(|extent| extent.logical_end == extent.start + 1),
                    "case {case}"
                );
                let mut row = 0_u64;
                for (axis, extent) in extents[..rank - 1].iter().enumerate() {
                    row = row * u64::from(shape[axis]) + u64::from(extent.start);
                }
                let columns = &extents[rank - 1];
                let width = u64::from(shape[rank - 1]);
                for column in columns.start..columns.logical_end {
                    let index = usize::try_from(row * width + u64::from(column)).unwrap();
                    coverage[index] += 1;
                    tile_elements[usize::from(tile)] += 1;
                }
            }
            assert!(coverage.into_iter().all(|count| count == 1), "case {case}");
            assert!(
                tile_elements
                    .iter()
                    .all(|count| count % u64::from(grain) == 0),
                "case {case}"
            );
            assert!(
                tile_elements.iter().max().unwrap() - tile_elements.iter().min().unwrap()
                    <= u64::from(grain),
                "case {case}"
            );
        }
    }

    #[test]
    fn randomized_parallel_reduction_gemms_lower_to_packed_reductions() {
        let mut random = fastrand::Rng::with_seed(0x7472_6565_5f6b_7370);
        for case in 0..CASES {
            let output_columns = [64, 128][random.usize(0..2)];
            let inner_partitions = random.u16(2..=4);
            let column_partitions = random.u16(1..=3);
            let row_partitions = random.u16(inner_partitions..=8);
            let tiles = inner_partitions * column_partitions * row_partitions;
            let rows_per_partition = random.u32(1..=4);
            let rows = u32::from(row_partitions) * rows_per_partition;
            let inner = u32::from(inner_partitions)
                * 64
                * random.u32(1..=u32::from(row_partitions / inner_partitions));
            let columns = u32::from(column_partitions) * output_columns;
            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", [1, rows, inner]).unwrap();
            let right = graph.parameter("right", [1, inner, columns]).unwrap();
            let product = graph.gemm(left, right).unwrap();
            graph.set_outputs([product]).unwrap();
            let operator = MidOperator::Gemm {
                options: Default::default(),
                multiply: Precision::F16,
                accumulate: AccumulationPrecision::F32,
            };
            let kernel = |mode| TileKernelSpec::Gemm {
                multiply: Precision::F16,
                accumulate: AccumulationPrecision::F32,
                mode,
                weights: GemmWeightLoad::Interleaved,
                inner_block: 64,
                output_columns,
            };
            let left_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::amp_left_parallel_grid(
                    64,
                    tiles,
                    row_partitions,
                    column_partitions,
                    inner_partitions,
                ),
            };
            let right_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::block_major_matrix_storage(
                    64,
                    output_columns,
                    column_partitions,
                    inner_partitions,
                    1,
                    MemoryClass::Ipu21Interleaved,
                ),
            };
            let (result_row_partitions, result_column_partitions) = if random.bool() {
                (1, 1)
            } else if random.bool() && rows_per_partition >= u32::from(inner_partitions) {
                (inner_partitions, 1)
            } else {
                (1, inner_partitions)
            };
            let storage_rows = row_partitions.saturating_mul(result_row_partitions);
            let storage_columns = column_partitions.saturating_mul(result_column_partitions);
            let reduction_staging = if random.bool() {
                crate::ReductionStaging::Complete
            } else {
                crate::ReductionStaging::Streamed
            };
            let output_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::amp_left_result_grid(
                    if result_column_partitions > 1 {
                        crate::mid::AMP_COLUMN_MICRO
                    } else {
                        output_columns
                    },
                    storage_rows * storage_columns,
                    storage_rows,
                    storage_columns,
                    crate::mid::GridOrder::ColumnsFast,
                ),
            };
            let candidate = OperatorCandidate::new(
                operator,
                [
                    OperandRequirement::new(left_format.clone(), 32),
                    OperandRequirement::new(right_format.clone(), 32),
                ],
                OperandRequirement::new(output_format, 32),
            )
            .with_dispatch(OperatorDispatch::BlockedGemm {
                initialize: kernel(GemmKernelMode::Initialize),
                accumulate: kernel(GemmKernelMode::Accumulate),
                inner_block: 64,
                output_column_block: output_columns,
                orientation: crate::GemmOrientation::Normal,
                distribution: GemmDistribution::ParallelReduction {
                    row_partitions,
                    column_partitions,
                    inner_partitions,
                    result_row_partitions,
                    result_column_partitions,
                    reduction_staging,
                },
            });
            let mut config = PipelineConfig::new(tiles)
                .with_input(left, left_format)
                .with_input(right, right_format);
            config.operator_candidates = vec![candidate];
            let mid = lower(&graph, &config, &Ipu21CostModel)
                .unwrap_or_else(|error| panic!("case {case}: {error}"));
            let low = lower_to_tiles(&mid, &config)
                .unwrap_or_else(|error| {
                    panic!(
                        "case {case}: {error}; rows={rows} inner={inner} columns={columns} grid={row_partitions}x{column_partitions}x{inner_partitions}"
                    )
                });
            let reduction_runs = low
                .kernel_runs
                .iter()
                .filter(|run| {
                    matches!(
                        run.kernel,
                        TileKernel::Planned(TileKernelSpec::ReductionSum { .. })
                    )
                })
                .collect::<Vec<_>>();
            assert!(!reduction_runs.is_empty(), "case {case}");
            assert!(
                reduction_runs.iter().all(|run| {
                    matches!(
                        run.kernel,
                        TileKernel::Planned(TileKernelSpec::ReductionSum { partials })
                            if partials == match reduction_staging {
                                crate::ReductionStaging::Complete => inner_partitions,
                                crate::ReductionStaging::Streamed => 2,
                            }
                    ) && run.inputs.len() == 2
                }),
                "case {case}"
            );
            assert!(
                low.exchange_phases.len() <= usize::from(inner_partitions).saturating_add(2),
                "case {case}"
            );
            let parameter_shards = low
                .inputs
                .iter()
                .find(|input| input.kind == crate::GraphInputKind::Parameter)
                .unwrap()
                .shards
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            let direct_parameter_runs = low
                .kernel_runs
                .iter()
                .filter(|run| {
                    matches!(run.kernel, TileKernel::Planned(TileKernelSpec::Gemm { .. }))
                        && run.inputs[1]
                            .views
                            .iter()
                            .any(|view| parameter_shards.contains(&view.shard))
                })
                .count();
            assert!(direct_parameter_runs > 0, "case {case}");
            if (result_row_partitions, result_column_partitions) != (1, 1) {
                let output_shards = low.outputs[0]
                    .shards
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>();
                let packed_results = reduction_runs
                    .iter()
                    .map(|run| run.output.shard)
                    .collect::<BTreeSet<_>>();
                let copied_outputs = low
                    .local_copies
                    .iter()
                    .filter(|copy| packed_results.contains(&copy.source))
                    .map(|copy| copy.destination)
                    .collect::<BTreeSet<_>>();
                assert!(
                    output_shards.is_subset(&copied_outputs),
                    "case {case}: every distributed result shard must receive a packed result"
                );
            }
        }
    }

    #[test]
    fn randomized_parameter_owner_groups_pack_independently_of_compute_tiles() {
        let mut random = fastrand::Rng::with_seed(0x7061_7261_6d73);
        for case in 0..CASES {
            let owner_tiles = 1_u16 << random.u32(1..=3);
            let compute_tiles = owner_tiles * 2;
            let inner = u32::from(owner_tiles) * 64;
            let rows = u32::from(compute_tiles) * random.u32(1..=4);
            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", [rows, inner]).unwrap();
            let right0 = graph.parameter("right.0", [inner, 64]).unwrap();
            let right1 = graph.parameter("right.1", [inner, 64]).unwrap();
            let output0 = graph.gemm(left, right0).unwrap();
            let output1 = graph.gemm(left, right1).unwrap();
            graph.set_outputs([output0, output1]).unwrap();

            let left_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::amp_left(64, compute_tiles),
            };
            let right_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::block_major_matrix_storage(
                    64,
                    64,
                    1,
                    owner_tiles,
                    1,
                    MemoryClass::Ipu21Standard,
                ),
            };
            let output_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::amp_left_result(compute_tiles),
            };
            let mut config = PipelineConfig::new(compute_tiles)
                .with_input(left, left_format.clone())
                .with_input(right0, right_format.clone())
                .with_input(right1, right_format.clone());
            config.operator_candidates = vec![OperatorCandidate::new(
                MidOperator::Gemm {
                    options: crate::GemmOptions::default(),
                    multiply: Precision::F16,
                    accumulate: crate::AccumulationPrecision::F16,
                },
                [
                    OperandRequirement::new(left_format, 32).with_access_tail(16),
                    OperandRequirement::new(right_format, 32),
                ],
                OperandRequirement::new(output_format, 32),
            )];

            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            assert!(mid.operations.iter().all(|operation| {
                operation.operator_plan.as_ref().is_none_or(|plan| {
                    plan.requirements.inputs[1].format.layout.tiling.tile_count == owner_tiles
                        && plan.requirements.output.format.layout.tiling.tile_count == compute_tiles
                })
            }));
            let low = lower_to_tiles(&mid, &config).unwrap();
            let parameter_tiles = |name: &str| {
                low.inputs
                    .iter()
                    .find(|input| input.name == name)
                    .unwrap()
                    .shards
                    .iter()
                    .map(|shard| low.shards[shard.index() as usize].tile)
                    .collect::<BTreeSet<_>>()
            };
            let first = parameter_tiles("right.0");
            let second = parameter_tiles("right.1");
            assert_eq!(first.len(), usize::from(owner_tiles), "case {case}");
            assert_eq!(second.len(), usize::from(owner_tiles), "case {case}");
            assert!(first.is_disjoint(&second), "case {case}");
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
            for run in
                low.tiles
                    .iter()
                    .flat_map(|tile| low.work(tile))
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
            {
                let output = &low.shards[run.output.shard.index() as usize];
                let flattens_outer_rows = matches!(
                    output.tensor_type.format.layout.order,
                    ElementOrder::Amp(AmpOrder::Left | AmpOrder::Output)
                );
                assert!(
                    flattens_outer_rows
                        || run.output.extents[..run.output.extents.len() - 2]
                            .iter()
                            .all(|extent| extent.physical_end - extent.start == 1),
                    "case {case}"
                );
            }
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
                    assert_ne!(
                        low.shards[input.shard.index() as usize].definition,
                        ShardDefinition::Unmaterialized,
                        "case {case}"
                    );
                    let inner = input.extents.last().unwrap();
                    let TileKernel::Planned(TileKernelSpec::Gemm { inner_block, .. }) = &run.kernel
                    else {
                        continue;
                    };
                    assert!(
                        inner.physical_end - inner.start <= *inner_block,
                        "case {case}"
                    );
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
                layout: Layout::amp_left_grid(
                    64,
                    tiles,
                    row_partitions,
                    column_partitions,
                    crate::mid::GridOrder::ColumnsFast,
                ),
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
            let resolved = layout.resolve(&tensor_type.shape).unwrap();
            let padded = resolved.padded_shape();
            let shards = shard_extents(&tensor_type).unwrap();
            assert_eq!(shards.len(), usize::from(tile_count), "case {case}");

            for (axis, partitions, block) in [
                (0, row_partitions, row_block),
                (1, column_partitions, column_block),
            ] {
                let ranges = shards
                    .iter()
                    .map(|(_, extents)| (extents[axis].start, extents[axis].physical_end))
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
    fn randomized_partition_padding_preserves_logical_groups() {
        let mut random = fastrand::Rng::with_seed(0x6772_6f75_705f_7064);
        for case in 0..CASES * 8 {
            let groups = random.u16(1..=16);
            let group_width = random.u32(1..=127);
            let rows = random.u32(1..=16);
            let physical_multiple = 1_u32 << random.u32(1..=4);
            let physical_width = group_width.div_ceil(physical_multiple) * physical_multiple;
            let physical_blocks = physical_width / physical_multiple;
            let partitions_per_group =
                random.u16(1..=u16::try_from(physical_blocks.min(8)).unwrap());
            let partitions = groups * partitions_per_group;
            let layout = Layout {
                order: ElementOrder::RowMajor,
                tiling: TensorTiling {
                    tile_count: partitions,
                    replicas: 1,
                    axes: vec![
                        AxisTiling::new(
                            TensorAxis::FromEnd(1),
                            partitions,
                            physical_multiple,
                            Padding::Zero,
                        )
                        .with_padding_groups(groups),
                    ],
                },
                memory_class: MemoryClass::Ipu21Standard,
            };
            let tensor = TensorType::new(
                [rows, u32::from(groups) * group_width],
                Precision::F16,
                layout,
            );
            let shards = shard_extents(&tensor).unwrap();
            assert_eq!(shards.len(), usize::from(partitions), "case {case}");
            for group in 0..groups {
                let group_base = u32::from(group) * group_width;
                let group_shards = &shards[usize::from(group * partitions_per_group)
                    ..usize::from((group + 1) * partitions_per_group)];
                let mut cursor = group_base;
                let mut allocated = 0;
                for (_, extents) in group_shards {
                    assert_eq!(extents[1].start, cursor, "case {case}");
                    assert!(
                        extents[1].logical_end <= group_base + group_width,
                        "case {case}"
                    );
                    cursor = extents[1].logical_end;
                    allocated += extents[1].physical_end - extents[1].start;
                    assert_eq!(
                        crate::shard_storage_bytes(&LowShard {
                            id: LowShardId(0),
                            tile: 0,
                            tensor_type: tensor.clone(),
                            extents: extents.clone(),
                            definition: ShardDefinition::Staging,
                        })
                        .unwrap(),
                        rows * (extents[1].physical_end - extents[1].start) * 2,
                        "case {case}"
                    );
                }
                assert_eq!(cursor, group_base + group_width, "case {case}");
                assert_eq!(allocated, physical_width, "case {case}");
            }
            assert_eq!(
                crate::estimate::physical_elements(&tensor.shape, &tensor.format.layout),
                u64::from(rows) * u64::from(groups) * u64::from(physical_width),
                "case {case}"
            );
            assert_eq!(
                crate::estimate::maximum_shard_bytes(&tensor),
                u64::from(rows)
                    * u64::from(
                        physical_blocks.div_ceil(u32::from(partitions_per_group))
                            * physical_multiple,
                    )
                    * 2,
                "case {case}"
            );
        }
    }

    #[test]
    fn randomized_padded_intersections_do_not_claim_adjacent_groups() {
        let mut random = fastrand::Rng::with_seed(0x7064_5f69_6e74_6572);
        for case in 0..CASES * 8 {
            let width = random.u32(1..=127);
            let padding = random.u32(1..=31);
            let group = random.u32(0..=30);
            let start = group * width;
            let owned = ShardExtent {
                axis: 0,
                start,
                logical_end: start + width,
                physical_end: start + width + padding,
            };
            let matching = intersect_extents_with_shared_padding(&[owned], &[owned]).unwrap();
            assert_eq!(matching, vec![owned], "case {case}");

            let adjacent = ShardExtent {
                axis: 0,
                start: start + width,
                logical_end: start + width * 2,
                physical_end: start + width * 2 + padding,
            };
            assert!(
                intersect_extents_with_shared_padding(&[owned], &[adjacent]).is_none(),
                "case {case}"
            );

            let narrower_padding = random.u32(0..=padding);
            let narrower = ShardExtent {
                physical_end: owned.logical_end + narrower_padding,
                ..owned
            };
            let intersection =
                intersect_extents_with_shared_padding(&[owned], &[narrower]).unwrap();
            assert_eq!(
                intersection[0].physical_end,
                owned.logical_end + narrower_padding,
                "case {case}"
            );
        }
    }

    #[test]
    fn randomized_gemm_grid_orders_align_operands_and_pair_shared_payloads() {
        let mut random = fastrand::Rng::with_seed(0x6772_6964_5f6f_7264);
        for case in 0..CASES {
            let row_partitions = 1_u16 << random.u32(1..=3);
            let column_partitions = 1_u16 << random.u32(1..=3);
            let tiles = row_partitions * column_partitions;
            let rows = u32::from(row_partitions) * random.u32(1..=8);
            let columns = u32::from(column_partitions) * 64 * random.u32(1..=3);
            let inner = 64 * random.u32(1..=4);
            for order in [GridOrder::ColumnsFast, GridOrder::RowsFast] {
                let left = TensorType::new(
                    [rows, inner],
                    Precision::F16,
                    Layout::amp_left_grid(64, tiles, row_partitions, column_partitions, order),
                );
                let right = TensorType::new(
                    [inner, columns],
                    Precision::F16,
                    Layout::block_major_matrix_grid(
                        64,
                        64,
                        tiles,
                        row_partitions,
                        column_partitions,
                        order,
                    ),
                );
                let output = TensorType::new(
                    [rows, columns],
                    Precision::F16,
                    Layout::amp_output_grid(64, tiles, row_partitions, column_partitions, order),
                );
                let left = shard_extents(&left).unwrap();
                let right = shard_extents(&right).unwrap();
                let output = shard_extents(&output).unwrap();
                for tile in 0..usize::from(tiles) {
                    assert_eq!(left[tile].1[0], output[tile].1[0], "case {case}");
                    assert_eq!(right[tile].1[1], output[tile].1[1], "case {case}");
                }
                for tile in (0..usize::from(tiles)).step_by(2) {
                    let shared_axis = match order {
                        GridOrder::ColumnsFast => 0,
                        GridOrder::RowsFast => 1,
                    };
                    assert_eq!(
                        output[tile].1[shared_axis],
                        output[tile + 1].1[shared_axis],
                        "case {case}"
                    );
                }
            }
        }
    }

    #[test]
    fn randomized_split_head_mappings_are_bijective_rectangles() {
        let mut random = fastrand::Rng::with_seed(0x6d61_7070_6564_5f68);
        for _ in 0..CASES * 8 {
            let batch = random.u32(1..=8);
            let heads = random.u32(1..=32);
            let rows = random.u32(1..=256);
            let width = random.u32(1..=128);
            let stream = random.u32(0..batch * heads);
            let row_start = random.u32(0..rows);
            let row_end = random.u32(row_start + 1..=rows);
            let column_start = random.u32(0..width);
            let column_end = random.u32(column_start + 1..=width);
            let output = vec![
                ShardExtent {
                    axis: 0,
                    start: stream,
                    logical_end: stream + 1,
                    physical_end: stream + 1,
                },
                ShardExtent {
                    axis: 1,
                    start: row_start,
                    logical_end: row_end,
                    physical_end: row_end,
                },
                ShardExtent {
                    axis: 2,
                    start: column_start,
                    logical_end: column_end,
                    physical_end: column_end,
                },
            ];
            let (source, base) = split_head_source_extents(&output, heads, width).unwrap();

            assert_eq!(source[0].start, stream / heads);
            assert_eq!(source[1], output[1]);
            assert_eq!(source[2].start, base + column_start);
            assert_eq!(source[2].logical_end, base + column_end);
            assert_eq!(
                source
                    .iter()
                    .map(|extent| extent.logical_end - extent.start)
                    .product::<u32>(),
                output
                    .iter()
                    .map(|extent| extent.logical_end - extent.start)
                    .product::<u32>()
            );
        }
    }

    #[test]
    fn randomized_micro_panel_mappings_carry_word_aligned_row_padding() {
        let mut random = fastrand::Rng::with_seed(0x7061_6464_6564_5f72);
        for case in 0..CASES * 8 {
            let rows = random.u32(1..=AMP_INNER_BLOCK);
            let panel_rows = rows.div_ceil(AMP_COLUMN_MICRO) * AMP_COLUMN_MICRO;
            let source_rows = if random.bool() {
                panel_rows
            } else {
                AMP_INNER_BLOCK
            };
            let source = LowShard {
                id: LowShardId(0),
                tile: 0,
                tensor_type: TensorType::new(
                    [rows, AMP_COLUMN_MICRO],
                    Precision::F16,
                    Layout {
                        order: ElementOrder::Amp(AmpOrder::TransposedLeft),
                        tiling: TensorTiling::replicated(1),
                        memory_class: MemoryClass::Ipu21Standard,
                    },
                ),
                extents: vec![
                    ShardExtent {
                        axis: 0,
                        start: 0,
                        logical_end: rows,
                        physical_end: source_rows,
                    },
                    ShardExtent {
                        axis: 1,
                        start: 0,
                        logical_end: AMP_COLUMN_MICRO,
                        physical_end: AMP_COLUMN_MICRO,
                    },
                ],
                definition: ShardDefinition::ExchangeStaging,
            };
            let destination = LowShard {
                id: LowShardId(1),
                tile: 1,
                tensor_type: TensorType::new(
                    [rows, AMP_COLUMN_MICRO],
                    Precision::F16,
                    Layout {
                        order: ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                            row_block: AMP_INNER_BLOCK as u16,
                            column_block: AMP_COLUMN_MICRO as u16,
                        }),
                        tiling: TensorTiling::replicated(1),
                        memory_class: MemoryClass::Ipu21Standard,
                    },
                ),
                extents: vec![
                    ShardExtent {
                        axis: 0,
                        start: 0,
                        logical_end: rows,
                        physical_end: AMP_INNER_BLOCK,
                    },
                    ShardExtent {
                        axis: 1,
                        start: 0,
                        logical_end: AMP_COLUMN_MICRO,
                        physical_end: AMP_COLUMN_MICRO,
                    },
                ],
                definition: ShardDefinition::ExchangeStaging,
            };
            let logical_view = |shard: &LowShard| ShardView {
                shard: shard.id,
                extents: shard
                    .extents
                    .iter()
                    .copied()
                    .map(|mut extent| {
                        extent.physical_end = extent.logical_end;
                        extent
                    })
                    .collect(),
            };
            let mappings = split_mapping_at_panel_boundaries(
                &source,
                logical_view(&source),
                &destination,
                logical_view(&destination),
            )
            .unwrap_or_else(|error| panic!("case {case}, rows {rows}: {error}"));
            let source_bytes = mappings
                .iter()
                .flat_map(|(view, _)| view_byte_spans(&source, view).unwrap())
                .map(|span| {
                    assert_eq!(span.offset & 0b11, 0, "case {case}, rows {rows}");
                    assert_eq!(span.bytes & 0b11, 0, "case {case}, rows {rows}");
                    span.bytes
                })
                .sum::<u32>();
            let destination_bytes = mappings
                .iter()
                .flat_map(|(_, view)| view_byte_spans(&destination, view).unwrap())
                .map(|span| span.bytes)
                .sum::<u32>();
            assert_eq!(source_bytes, panel_rows * AMP_COLUMN_MICRO * 2);
            assert_eq!(destination_bytes, source_bytes, "case {case}, rows {rows}");
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
            assert!(low.exchange_phases.iter().all(|phase| {
                phase.provenance.operation.is_some() && phase.provenance.value.is_some()
            }));
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
                let mut initialized_columns = std::collections::BTreeSet::new();
                for run in gemms {
                    assert_eq!(run.provenance.reason, WorkReason::OperatorKernel);
                    assert!(run.provenance.operation.is_some());
                    assert!(run.provenance.value.is_some());
                    let TileKernel::Planned(TileKernelSpec::Gemm {
                        mode,
                        inner_block: kernel_inner,
                        output_columns: kernel_columns,
                        ..
                    }) = run.kernel
                    else {
                        unreachable!()
                    };
                    let output_key = run
                        .output
                        .extents
                        .iter()
                        .map(|extent| (extent.start, extent.physical_end))
                        .collect::<Vec<_>>();
                    assert_eq!(
                        mode,
                        if initialized_columns.insert(output_key) {
                            crate::GemmKernelMode::Initialize
                        } else {
                            crate::GemmKernelMode::Accumulate
                        },
                        "case {case}"
                    );
                    assert_eq!(run.inputs.len(), 2);
                    assert!(run.inputs.iter().all(|operand| operand.views.len() == 1));
                    assert!(
                        run.inputs[0].views[0]
                            .extents
                            .iter()
                            .any(|extent| { extent.physical_end - extent.start == kernel_inner })
                    );
                    assert!(
                        run.output
                            .extents
                            .iter()
                            .any(|extent| { extent.physical_end - extent.start == kernel_columns })
                    );
                }
            }
        }
    }

    #[test]
    fn randomized_odd_capacities_use_nonempty_active_tile_subsets() {
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
            let selected_tiles = mid.values[result.index() as usize]
                .tensor_type
                .format
                .layout
                .tiling
                .tile_count;
            assert!(selected_tiles <= capacity, "case {case}");

            let low = lower_to_tiles(&mid, &config).unwrap();
            assert_eq!(low.tile_count, capacity, "case {case}");
            assert_eq!(low.outputs[0].shards.len(), usize::from(selected_tiles));
            for &shard in &low.outputs[0].shards {
                assert!(
                    low.shards[shard.index() as usize]
                        .extents
                        .iter()
                        .all(|extent| extent.start < extent.logical_end),
                    "case {case} capacity={capacity} selected={selected_tiles} shard={:?} type={:?}",
                    low.shards[shard.index() as usize].extents,
                    mid.values[result.index() as usize].tensor_type,
                );
            }
            assert!(low.tiles.iter().all(|tile| tile.tile < capacity));
        }
    }

    #[test]
    fn randomized_resident_blocked_weights_lower_without_panel_copies() {
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
            let mut config = PipelineConfig::new(tiles)
                .with_automatic_input(left, Precision::F16)
                .with_automatic_input(right, Precision::F16);
            config.operator_candidates.retain(|candidate| {
                matches!(
                    candidate.dispatch,
                    OperatorDispatch::BlockedGemm {
                        distribution: GemmDistribution::OutputStationary,
                        ..
                    }
                ) && candidate.inputs.get(1).is_some_and(|requirement| {
                    requirement.format.layout.order
                        == crate::ElementOrder::BlockMajor(crate::BlockMajorOrder::Matrix {
                            row_block: 64,
                            column_block: crate::mid::AMP_COLUMN_MICRO as u16,
                        })
                        && requirement.format.layout.tiling.tile_count == tiles
                        && requirement.format.layout.memory_class == MemoryClass::Ipu21Interleaved
                })
            });
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let operation = mid
                .operations
                .iter()
                .find(|operation| matches!(operation.kind, MidOperationKind::Operator(_)))
                .unwrap();
            let right_type = &mid.values[operation.inputs[1].index() as usize].tensor_type;
            assert_eq!(
                right_type.format.layout.order,
                crate::ElementOrder::BlockMajor(crate::BlockMajorOrder::Matrix {
                    row_block: 64,
                    column_block: crate::mid::AMP_COLUMN_MICRO as u16,
                })
            );
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
    fn randomized_partially_sharded_weight_grids_preserve_storage() {
        let mut random = fastrand::Rng::with_seed(0x7374_726d_6765_6d6d);
        for _ in 0..32 {
            let row_partitions = 1_u16 << random.u32(1..=2);
            let inner_partitions = 1_u16 << random.u32(1..=row_partitions.ilog2());
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
                layout: Layout::amp_left_grid(
                    64,
                    tiles,
                    row_partitions,
                    column_partitions,
                    crate::mid::GridOrder::ColumnsFast,
                ),
            };
            let right_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::block_major_matrix_storage(
                    64,
                    64,
                    column_partitions,
                    inner_partitions,
                    row_partitions / inner_partitions,
                    crate::MemoryClass::Ipu21Standard,
                ),
            };
            let output_format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::amp_left_result_grid(
                    64,
                    tiles,
                    row_partitions,
                    column_partitions,
                    crate::mid::GridOrder::ColumnsFast,
                ),
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

            let expected_weight_bytes = inner
                .div_ceil(u32::from(inner_partitions))
                .saturating_mul(columns.div_ceil(u32::from(column_partitions)))
                .saturating_mul(2);
            assert!(low.inputs[1].shards.iter().all(|shard| {
                crate::shard_storage_bytes(&low.shards[shard.index() as usize])
                    == Ok(expected_weight_bytes)
            }));
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
                    ShardDefinition::WritableAlias(carried.argument)
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
    fn randomized_repeats_alias_fresh_results_after_the_last_carried_use() {
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
            let low = lower_to_tiles(&mid, &config).unwrap();
            for tile in &low.tiles {
                let repeat = low
                    .work(tile)
                    .find_map(|work| match work {
                        TileWorkRef::Repeat(repeat) => Some(repeat),
                        _ => None,
                    })
                    .unwrap();
                assert_eq!(
                    low.shards[repeat.carried[0].yielded.index() as usize].definition,
                    ShardDefinition::WritableAlias(repeat.carried[0].argument),
                    "case {case}"
                );
            }
        }
    }

    fn contains_phase(program: &LowProgram, list: &TileWorkList, phase: ExchangePhaseId) -> bool {
        program.work(list).any(|work| match work {
            TileWorkRef::Exchange(candidate) => candidate == phase,
            TileWorkRef::Repeat(repeat) => contains_phase(program, &repeat.body, phase),
            TileWorkRef::Kernel(_) | TileWorkRef::LocalCopy(_) | TileWorkRef::Checkpoint(..) => {
                false
            }
        })
    }
}
