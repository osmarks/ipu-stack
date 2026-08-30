//! Logical per-tile schedule produced from the layout-aware mid-level IR.
//!
//! Tensor shards have tile identities and rectangular physical extents, and
//! work is ordered per tile. Exchanges still refer to logical shards rather
//! than SRAM addresses; kernel runs still name a selected kernel kind rather
//! than a linked symbol. Placement and final code generation resolve those
//! remaining choices.

use crate::PipelineConfig;
use crate::conversion::{ConversionStrategy, DeferredTransform};
use crate::graph::{GraphInputKind, OperationId};
use crate::kernel::TileKernelSpec;
use crate::layout::{
    AMP_COLUMN_MICRO, AMP_INNER_BLOCK, BlockedOrder, Layout, LayoutError, MemoryClass,
    NativeKernelOrder, ShardExtent, StorageOrder, TensorRegion, TensorTiling, TensorType,
};
use crate::mid::{MidGraph, MidOperation, MidOperationKind, MidRepeat, MidValueId};
use crate::operator::{
    GemmDistribution, GemmKernelMode, MemoryOperand, MemorySpaceRequirements, OperandRequirement,
    OperatorDispatch, OperatorRequirements, OutputAliasing, PointwiseInputMapping, Precision,
};
use crate::storage::{ByteSpan, StorageError, logical_view_byte_spans, view_byte_spans};
use ipu_target::hardware::HardwareTarget;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

mod attention;
mod gemm;
mod pointwise;
#[cfg(test)]
mod tests;

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
    pub extents: TensorRegion,
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
    pub extents: TensorRegion,
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ExchangeOrder {
    /// Preserve tensor coordinates, converting between physical layouts.
    #[default]
    Semantic,
    /// Preserve allocation order, treating both views as packed byte spans.
    Physical,
    /// Use copy geometry selected from the resolved layouts by mid-level
    /// conversion planning.
    Planned(crate::CopyGeometry),
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
pub enum KernelRequirements {
    Operator(OperatorRequirements),
    Conversion {
        input: OperandRequirement,
        output: OperandRequirement,
        memory_space: crate::MemorySpaceRequirements,
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
    pub kernel: TileKernelSpec,
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
        profiling = ?config.profiling
    )
)]
pub fn lower_to_tiles(graph: &MidGraph, config: &PipelineConfig) -> LowLoweringResult<LowProgram> {
    if config.tile_count == 0 {
        return Err(LowLoweringError::EmptyTileGroup);
    }
    let mut state = LoweringState::new(graph, config.tile_count, config.target)?;
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

enum DeferredValue {
    Conversion(MidValueId),
    View(DeferredTransform, Vec<LowShardId>),
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
    panel_rows: u32,
    logical_staging_rows: u32,
    physical_staging_rows: u32,
    scratch_columns: u32,
    state_columns: u32,
    padded_query_dimension: u32,
    padded_value_dimension: u32,
    reuse_key_staging_for_state: bool,
}

fn gemm_kernel_spec(
    family: crate::GemmKernelFamily,
    mode: crate::GemmKernelMode,
    block: crate::GemmBlockShape,
    rows: u32,
) -> TileKernelSpec {
    TileKernelSpec::Gemm {
        multiply: family.multiply,
        accumulate: family.accumulate,
        mode,
        weights: family.weights,
        inner_block: block.inner,
        output_columns: block.output_columns,
        rows,
    }
}

fn gemm_kernel_rows(output: &ShardView, order: StorageOrder) -> LowLoweringResult<u32> {
    let rank = output.extents.len();
    let column_axis = rank
        .checked_sub(
            if matches!(
                order,
                StorageOrder::Native(
                    NativeKernelOrder::TransposedOutput | NativeKernelOrder::TransposedLeft
                )
            ) {
                2
            } else {
                1
            },
        )
        .ok_or(LowLoweringError::InvalidOperatorPlan)?;
    output
        .extents
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != column_axis)
        .try_fold(1u32, |rows, (_, extent)| {
            rows.checked_mul(extent.physical_end - extent.start)
        })
        .filter(|rows| *rows != 0)
        .ok_or(LowLoweringError::IdOverflow)
}

fn rearrange_kernel_spec(
    from: Layout,
    to: Layout,
    input: &ShardView,
    output: &ShardView,
) -> LowLoweringResult<TileKernelSpec> {
    let view = if from.order == StorageOrder::Linear {
        output
    } else {
        input
    };
    let rank = view.extents.len();
    if rank < 2 {
        return Err(LowLoweringError::InvalidConversionPlan);
    }
    let rows = view.extents[rank - 2];
    let columns = view.extents[rank - 1];
    let matrices = view.extents[..rank - 2]
        .iter()
        .try_fold(1u32, |product, extent| {
            product.checked_mul(extent.physical_end - extent.start)
        })
        .ok_or(LowLoweringError::IdOverflow)?;
    Ok(TileKernelSpec::Rearrange {
        from,
        to,
        matrices,
        logical_rows: rows.logical_end - rows.start,
        physical_rows: rows.physical_end - rows.start,
        logical_columns: columns.logical_end - columns.start,
        physical_columns: columns.physical_end - columns.start,
    })
}

impl AttentionBufferShape {
    fn from_plan(plan: &crate::AttentionPlan, key_rows: u32) -> Self {
        let padded_query_dimension = plan.padding.query_dimension;
        let padded_value_dimension = plan.padding.value_dimension;
        match plan.blocking {
            crate::AttentionBlocking::Flash {
                query_rows,
                key_rows,
            } => Self {
                query_block_rows: query_rows,
                panel_rows: key_rows,
                logical_staging_rows: key_rows,
                physical_staging_rows: key_rows,
                scratch_columns: padded_value_dimension.max(key_rows),
                state_columns: key_rows + 16,
                padded_query_dimension,
                padded_value_dimension,
                reuse_key_staging_for_state: false,
            },
            crate::AttentionBlocking::Materialized {
                query_rows,
                padded_key_rows,
            } => Self {
                query_block_rows: query_rows,
                panel_rows: AMP_INNER_BLOCK,
                logical_staging_rows: key_rows,
                physical_staging_rows: padded_key_rows,
                scratch_columns: padded_key_rows.max(padded_value_dimension),
                state_columns: padded_key_rows + AMP_COLUMN_MICRO,
                padded_query_dimension,
                padded_value_dimension,
                reuse_key_staging_for_state: true,
            },
        }
    }
}

#[derive(Clone, Copy)]
enum AttentionOperand {
    Key,
    Value,
}

struct LoweringState {
    target: HardwareTarget,
    tile_count: u16,
    shards: Vec<LowShard>,
    canonical: Vec<Vec<LowShardId>>,
    phases: Vec<ExchangePhase>,
    kernel_runs: Vec<KernelRun>,
    local_copies: Vec<LocalCopy>,
    repeat_runs: Vec<RepeatRun>,
    kernel_metadata: Vec<Arc<KernelRunMetadata>>,
    deferred_values: BTreeMap<MidValueId, DeferredValue>,
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

    fn new(graph: &MidGraph, tile_count: u16, target: HardwareTarget) -> LowLoweringResult<Self> {
        let mut state = Self {
            target,
            tile_count,
            shards: Vec::new(),
            canonical: vec![Vec::new(); graph.values.len()],
            phases: Vec::new(),
            kernel_runs: Vec::new(),
            local_copies: Vec::new(),
            repeat_runs: Vec::new(),
            kernel_metadata: Vec::new(),
            deferred_values: BTreeMap::new(),
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
                            extents: extents.clone().into(),
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
                    extents: extents.into(),
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

    fn value_shards(&self, value: MidValueId) -> LowLoweringResult<&[LowShardId]> {
        self.canonical
            .get(value.index() as usize)
            .filter(|shards| !shards.is_empty())
            .map(Vec::as_slice)
            .ok_or(LowLoweringError::UnknownValue(value))
    }

    fn deferred_view(&self, mut value: MidValueId) -> Option<&DeferredValue> {
        loop {
            match self.deferred_values.get(&value)? {
                DeferredValue::Conversion(source) => value = *source,
                view @ DeferredValue::View(..) => return Some(view),
            }
        }
    }

    fn deferred_supports_physical_exchange(
        &self,
        value: MidValueId,
        destination: LowShardId,
    ) -> bool {
        let Some(DeferredValue::View(_, shards)) = self.deferred_view(value) else {
            return false;
        };
        let Some(source) = shards.first() else {
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
            if self.defer_conversion(
                operation,
                operations.get(index + 1),
                operations,
                retained_values,
                &mut tiles,
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
                    MidOperationKind::Operator(_)
                        | MidOperationKind::View(..)
                        | MidOperationKind::Repeat(_)
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
                    StorageOrder::Native(NativeKernelOrder::Output) => {
                        let columns = source.extents[2];
                        (columns.physical_end - columns.start).is_multiple_of(AMP_COLUMN_MICRO)
                    }
                    StorageOrder::Native(NativeKernelOrder::TransposedLeft) => {
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
                    rearrange_kernel_spec(
                        source.tensor_type.format.layout.clone(),
                        self.shards[staging.index() as usize]
                            .tensor_type
                            .format
                            .layout
                            .clone(),
                        &self.full_view(source_shard),
                        &self.full_view(staging),
                    )?,
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
                        memory_space: MemorySpaceRequirements::default(),
                    },
                ),
            )?;
            staging_shards.push(staging);
        }
        Ok(Some(staging_shards))
    }

    fn defer_conversion(
        &mut self,
        operation: &MidOperation,
        next: Option<&MidOperation>,
        operations: &[MidOperation],
        retained_values: &[MidValueId],
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<bool> {
        let (transform, strategy, materialization) = match &operation.kind {
            MidOperationKind::View(transform, strategy, _) => (
                Some(*transform),
                *strategy,
                crate::OperandMaterialization::DispatchSlices,
            ),
            MidOperationKind::Rearrange(strategy, materialization, _) => {
                (None, *strategy, *materialization)
            }
            _ => return Ok(false),
        };
        if !strategy.uses_intersections() {
            return Ok(false);
        }
        if materialization != crate::OperandMaterialization::DispatchSlices {
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
        let next = if transform.is_some() {
            operations
                .iter()
                .find(|candidate| candidate.inputs.contains(result))
        } else {
            next
        };
        let Some(next) = next else {
            return Ok(false);
        };
        let Some(input_index) = next.inputs.iter().position(|input| input == result) else {
            return Ok(false);
        };
        let streamable = next
            .operator_plan()
            .and_then(|plan| plan.requirements.inputs.get(input_index))
            .is_some_and(|requirement| {
                requirement.materialization == crate::OperandMaterialization::DispatchSlices
            });
        if !streamable {
            return Ok(false);
        }
        if let Some(transform) = transform {
            let source_shards = self.value_shards(*source)?.to_vec();
            let source_format = &self.shards[source_shards[0].index() as usize]
                .tensor_type
                .format;
            let result_format = &self.shards[self.value_shards(*result)?[0].index() as usize]
                .tensor_type
                .format;
            let direct = source_format.supports_f16_micro_panel_exchange(result_format);
            let shards = if direct
                || !matches!(
                    source_format.layout.order,
                    StorageOrder::Native(
                        NativeKernelOrder::Output | NativeKernelOrder::TransposedLeft
                    )
                ) {
                source_shards
            } else {
                self.unpack_amp_to_row_major(
                    *source,
                    operation_provenance(operation, &operation.kind),
                    tiles,
                )?
                .ok_or(LowLoweringError::InvalidConversionPlan)?
            };
            self.deferred_values
                .insert(*result, DeferredValue::View(transform, shards));
        } else {
            self.deferred_values
                .insert(*result, DeferredValue::Conversion(*source));
        }
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
        let (strategy, _) = operation
            .conversion()
            .ok_or(LowLoweringError::InvalidConversionPlan)?;
        match strategy {
            ConversionStrategy::LocalKernel => self.lower_local_conversion(operation, kind, tiles),
            ConversionStrategy::DirectRetile
            | ConversionStrategy::DirectLogical
            | ConversionStrategy::StageLogicalThenTransform => {
                self.lower_intersection_conversion(operation, kind, strategy, tiles)
            }
        }
    }

    fn lower_local_conversion(
        &mut self,
        operation: &MidOperation,
        kind: &MidOperationKind,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [input] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidConversionPlan);
        };
        let [result] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        let input_format = self.shards[self.value_shards(*input)?[0].index() as usize]
            .tensor_type
            .format
            .clone();
        let output_format = self.shards[self.value_shards(*result)?[0].index() as usize]
            .tensor_type
            .format
            .clone();
        let static_kernel = match kind {
            MidOperationKind::CastPrecision => Some(TileKernelSpec::Cast {
                from: input_format.precision,
                to: output_format.precision,
            }),
            MidOperationKind::View(..) | MidOperationKind::Rearrange(..) => None,
            MidOperationKind::Operator(_) | MidOperationKind::Repeat(_) => {
                return Err(LowLoweringError::InvalidConversionPlan);
            }
        };
        for output in self.value_shards(*result)?.to_vec() {
            let tile = self.shards[output.index() as usize].tile;
            let input = self.local_shard(*input, tile)?;
            let input_view = self.full_view(input);
            let output_view = self.full_view(output);
            let kernel = match kind {
                MidOperationKind::View(..) | MidOperationKind::Rearrange(..) => {
                    rearrange_kernel_spec(
                        input_format.layout.clone(),
                        output_format.layout.clone(),
                        &input_view,
                        &output_view,
                    )?
                }
                _ => static_kernel
                    .clone()
                    .ok_or(LowLoweringError::InvalidConversionPlan)?,
            };
            self.append_kernel(
                tiles,
                tile,
                KernelRun::new(
                    operation_provenance(operation, kind),
                    kernel,
                    vec![KernelOperand {
                        views: vec![input_view],
                    }],
                    output_view,
                    KernelRequirements::Conversion {
                        input: OperandRequirement::new(input_format.clone(), 8),
                        output: OperandRequirement::new(output_format.clone(), 8),
                        memory_space: MemorySpaceRequirements::default(),
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
        strategy: ConversionStrategy,
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
        let mappings = match kind {
            MidOperationKind::View(_, _, mappings)
            | MidOperationKind::Rearrange(_, _, mappings) => mappings,
            _ => return Err(LowLoweringError::InvalidConversionPlan),
        };
        let staged = strategy == ConversionStrategy::StageLogicalThenTransform;
        let staging = if staged {
            outputs
                .iter()
                .map(|&output| self.push_conversion_staging(output))
                .collect::<LowLoweringResult<Vec<_>>>()?
        } else {
            outputs.clone()
        };
        if strategy == ConversionStrategy::DirectLogical {
            for &output in &outputs {
                if self.shard_has_padding(output) {
                    self.append_fill_zero(tiles, output, operation_provenance(operation, kind))?;
                }
            }
        }
        if mappings.iter().any(|mapping| mapping.copies.is_empty()) {
            return Err(LowLoweringError::InvalidConversionPlan);
        }
        let bound = mappings
            .iter()
            .map(|mapping| {
                let source = inputs
                    .get(mapping.source_shard as usize)
                    .copied()
                    .ok_or(LowLoweringError::InvalidConversionPlan)?;
                let destination = staging
                    .get(mapping.destination_shard as usize)
                    .copied()
                    .ok_or(LowLoweringError::InvalidConversionPlan)?;
                if self.shards[source.index() as usize].extents != mapping.source_storage
                    || self.shards[destination.index() as usize].extents
                        != mapping.destination_storage
                {
                    tracing::error!(
                        source_shard = mapping.source_shard,
                        destination_shard = mapping.destination_shard,
                        actual_source = ?self.shards[source.index() as usize].extents,
                        planned_source = ?mapping.source_storage,
                        actual_destination = ?self.shards[destination.index() as usize].extents,
                        planned_destination = ?mapping.destination_storage,
                        "selected conversion geometry did not bind to low shards"
                    );
                    return Err(LowLoweringError::InvalidConversionPlan);
                }
                Ok((mapping, source, destination))
            })
            .collect::<LowLoweringResult<Vec<_>>>()?;
        let mut transfers = BTreeMap::<(ShardView, crate::CopyGeometry), Vec<ShardView>>::new();
        let mut source_copies = Vec::new();
        let mut local_copies = Vec::new();
        let mut destination_copies = Vec::new();
        for (mapping, source, destination) in bound {
            let source_tile = self.shards[source.index() as usize].tile;
            let destination_tile = self.shards[destination.index() as usize].tile;
            let (transfer_source, transfer_destination) = if mapping.source_copies.is_empty() {
                (source, destination)
            } else {
                let bytes = mapping
                    .copies
                    .iter()
                    .try_fold(0_u64, |bytes, geometry| bytes.checked_add(geometry.bytes()))
                    .and_then(|bytes| u32::try_from(bytes).ok())
                    .ok_or(LowLoweringError::IdOverflow)?;
                let precision = self.shards[source.index() as usize]
                    .tensor_type
                    .format
                    .precision;
                let source_staging = self.push_transfer_staging(
                    source_tile,
                    precision,
                    bytes,
                    ShardDefinition::Staging,
                )?;
                let destination_staging = self.push_transfer_staging(
                    destination_tile,
                    precision,
                    bytes,
                    ShardDefinition::ExchangeStaging,
                )?;
                self.append_fill_zero(
                    tiles,
                    source_staging,
                    operation_provenance(operation, kind),
                )?;
                for geometry in &mapping.source_copies {
                    source_copies.extend(
                        planned_local_copies(source, source_staging, geometry)?
                            .into_iter()
                            .map(|copy| (source_tile, copy)),
                    );
                }
                for geometry in &mapping.destination_copies {
                    destination_copies.extend(
                        planned_local_copies(destination_staging, destination, geometry)?
                            .into_iter()
                            .map(|copy| (destination_tile, copy)),
                    );
                }
                (source_staging, destination_staging)
            };
            let source_view = if transfer_source == source {
                ShardView {
                    shard: source,
                    extents: mapping.source_region.clone(),
                }
            } else {
                self.full_view(transfer_source)
            };
            let destination_view = if transfer_destination == destination {
                ShardView {
                    shard: destination,
                    extents: mapping.destination_region.clone(),
                }
            } else {
                self.full_view(transfer_destination)
            };
            for geometry in &mapping.copies {
                if source_tile == destination_tile {
                    local_copies.extend(
                        planned_local_copies(transfer_source, transfer_destination, geometry)?
                            .into_iter()
                            .map(|copy| (destination_tile, copy)),
                    );
                } else {
                    transfers
                        .entry((source_view.clone(), geometry.clone()))
                        .or_default()
                        .push(destination_view.clone());
                }
            }
        }
        for (tile, copy) in source_copies {
            self.append_local_copy(tiles, tile, copy)?;
        }
        self.append_planned_phase(transfers, operation_provenance(operation, kind), tiles)?;
        for (tile, copy) in local_copies.into_iter().chain(destination_copies) {
            self.append_local_copy(tiles, tile, copy)?;
        }
        if staged {
            for (&staging, &destination) in staging.iter().zip(&outputs) {
                let source_format = self.shards[staging.index() as usize]
                    .tensor_type
                    .format
                    .clone();
                let destination_format = self.shards[destination.index() as usize]
                    .tensor_type
                    .format
                    .clone();
                let supported_destination = match destination_format.layout.order {
                    StorageOrder::Native(
                        NativeKernelOrder::Left | NativeKernelOrder::TransposedRight,
                    ) => true,
                    StorageOrder::Blocked(order) => order.is_matrix(),
                    StorageOrder::Linear | StorageOrder::Native(_) => false,
                };
                if source_format.precision != crate::Precision::F16
                    || source_format.layout.order != StorageOrder::Linear
                    || !supported_destination
                {
                    return Err(LowLoweringError::InvalidConversionPlan);
                }
                let tile = self.shards[destination.index() as usize].tile;
                self.append_kernel(
                    tiles,
                    tile,
                    KernelRun::new(
                        operation_provenance(operation, kind),
                        rearrange_kernel_spec(
                            source_format.layout.clone(),
                            destination_format.layout.clone(),
                            &self.full_view(staging),
                            &self.full_view(destination),
                        )?,
                        vec![KernelOperand {
                            views: vec![self.full_view(staging)],
                        }],
                        self.full_view(destination),
                        KernelRequirements::Conversion {
                            input: OperandRequirement::new(source_format, 2),
                            output: OperandRequirement::new(destination_format, 2),
                            memory_space: MemorySpaceRequirements::default()
                                .with_distinct_elements([
                                    MemoryOperand::Input(0),
                                    MemoryOperand::Output,
                                ]),
                        },
                    ),
                )?;
            }
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
                        order: StorageOrder::Linear,
                        tiling: TensorTiling::replicated(1),
                        memory_class: MemoryClass::Standard,
                    },
                },
            },
            extents,
            definition: ShardDefinition::Staging,
        })
    }

    fn push_transfer_staging(
        &mut self,
        tile: u16,
        precision: Precision,
        bytes: u32,
        definition: ShardDefinition,
    ) -> LowLoweringResult<LowShardId> {
        let element_bytes = precision.bytes() as u32;
        let elements = bytes.div_ceil(element_bytes);
        self.push_shard(LowShard {
            id: LowShardId(0),
            tile,
            tensor_type: TensorType::new(
                [elements],
                precision,
                Layout::row_major(TensorTiling::replicated(1)),
            ),
            extents: vec![ShardExtent {
                axis: 0,
                start: 0,
                logical_end: elements,
                physical_end: elements,
            }]
            .into(),
            definition,
        })
    }

    fn lower_operator(
        &mut self,
        operation: &MidOperation,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let plan = operation
            .operator_plan()
            .ok_or(LowLoweringError::MissingOperatorPlan)?;
        match &plan.dispatch {
            OperatorDispatch::Pointwise(input_mapping) => {
                let kernel = match plan.operator {
                    crate::MidOperator::Gelu => TileKernelSpec::Gelu,
                    crate::MidOperator::Add(_) => TileKernelSpec::Add,
                    _ => return Err(LowLoweringError::InvalidOperatorPlan),
                };
                self.lower_pointwise(operation, kernel, *input_mapping, &plan.requirements, tiles)
            }
            OperatorDispatch::BlockedGemm(dispatch) => {
                self.lower_blocked_gemm(operation, dispatch, &plan.requirements, tiles)
            }
            OperatorDispatch::Attention(attention) => match attention.blocking {
                crate::AttentionBlocking::Flash { .. } => {
                    self.lower_blocked_attention(operation, attention, &plan.requirements, tiles)
                }
                crate::AttentionBlocking::Materialized { .. } => self.lower_materialized_attention(
                    operation,
                    attention,
                    &plan.requirements,
                    tiles,
                ),
            },
        }
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

    fn deferred_region_mappings(
        &self,
        value: MidValueId,
        logical_target: &TensorRegion,
        destination: LowShardId,
    ) -> LowLoweringResult<Vec<(ShardView, ShardView)>> {
        let mut source_value = value;
        let (source_shards, target, source_axes) = loop {
            match self.deferred_values.get(&source_value) {
                Some(DeferredValue::Conversion(source)) => source_value = *source,
                Some(DeferredValue::View(transform, shards)) => {
                    let logical_type =
                        &self.shards[self.value_shards(value)?[0].index() as usize].tensor_type;
                    let source_type = &self.shards[shards[0].index() as usize].tensor_type;
                    let mapping = transform
                        .map_slice(&source_type.shape, &logical_type.shape, logical_target)
                        .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                    break (shards.clone(), mapping.source, mapping.source_axes);
                }
                None => {
                    break (
                        self.value_shards(source_value)?.to_vec(),
                        logical_target.clone(),
                        (0..logical_target.len()).map(Some).collect(),
                    );
                }
            }
        };
        let destination_tile = self.shards[destination.index() as usize].tile;
        let destination_extents = &self.shards[destination.index() as usize].extents;
        if destination_extents.len() != source_axes.iter().flatten().count() {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let mut covered = 0u64;
        let mut mappings = Vec::new();
        for (source_extents, source) in
            self.intersecting_shard_set(&source_shards, &target, destination_tile)
        {
            let mapped_extents = source_axes
                .iter()
                .flatten()
                .enumerate()
                .map(|(destination_axis, &source_axis)| {
                    let source = source_extents
                        .get(source_axis)
                        .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                    let base = target
                        .get(source_axis)
                        .ok_or(LowLoweringError::InvalidOperatorPlan)?
                        .start;
                    let destination_base = destination_extents
                        .get(destination_axis)
                        .ok_or(LowLoweringError::InvalidOperatorPlan)?
                        .start;
                    Ok(ShardExtent {
                        axis: u16::try_from(destination_axis)
                            .map_err(|_| LowLoweringError::IdOverflow)?,
                        start: destination_base + source.start - base,
                        logical_end: destination_base + source.logical_end - base,
                        physical_end: destination_base + source.logical_end - base,
                    })
                })
                .collect::<LowLoweringResult<Vec<_>>>()?;
            covered = covered
                .saturating_add(TensorRegion::new(source_extents.clone()).logical_elements());
            let source_view = ShardView {
                shard: source,
                extents: source_extents.into(),
            };
            let destination_view = ShardView {
                shard: destination,
                extents: mapped_extents.into(),
            };
            mappings.push((source_view, destination_view));
        }
        if covered != target.logical_elements() {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        Ok(mappings)
    }

    fn mapping_word_exchange_fragments(
        &self,
        mappings: &[(ShardView, ShardView)],
    ) -> LowLoweringResult<Option<u64>> {
        let maximum_bytes = self
            .target
            .exchange()
            .maximum_transfer_words
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
            self.target
                .costs()
                .kernel_launch_cycles
                .saturating_add(bytes.div_ceil(8 * 6))
        } else {
            0
        };
        let fragment_cycles = fragments
            .saturating_mul(self.target.costs().logical_fragment_cycles)
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

    fn materialize_deferred_region(
        &self,
        value: MidValueId,
        region: &TensorRegion,
        destination: LowShardId,
        order: ExchangeOrder,
        transfers: &mut BTreeMap<ShardView, Vec<ShardView>>,
        local_copies: &mut Vec<(u16, LocalCopy)>,
    ) -> LowLoweringResult<()> {
        let destination_tile = self.shards[destination.index() as usize].tile;
        let mappings = self.deferred_region_mappings(value, region, destination)?;
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
                TileKernelSpec::FillZero,
                Vec::new(),
                self.full_view(shard),
                KernelRequirements::Operator(OperatorRequirements {
                    inputs: Vec::new(),
                    output,
                    output_aliasing: crate::OutputAliasing::Fresh,
                    memory_space: MemorySpaceRequirements::default(),
                }),
            ),
        )
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
                    order: order.clone(),
                }
            })
            .collect::<Vec<_>>();
        self.append_exchange_phase(transfers, provenance, tiles)
    }

    fn append_planned_phase(
        &mut self,
        transfers: BTreeMap<(ShardView, crate::CopyGeometry), Vec<ShardView>>,
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let transfers = transfers
            .into_iter()
            .map(|((source, geometry), mut destinations)| {
                destinations.sort_unstable();
                destinations.dedup();
                LogicalExchange {
                    source,
                    destinations,
                    order: ExchangeOrder::Planned(geometry),
                }
            })
            .collect();
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
                    order: order.clone(),
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
                    StorageOrder::Native(NativeKernelOrder::Left | NativeKernelOrder::Output)
                )
            });
        if matches!(run.kernel, TileKernelSpec::Gemm { .. })
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

fn planned_local_copies(
    source: LowShardId,
    destination: LowShardId,
    geometry: &crate::CopyGeometry,
) -> LowLoweringResult<Vec<LocalCopy>> {
    let mut inner = geometry.clone();
    let retain_inner = inner.dimensions.first().is_some_and(|dimension| {
        inner.contiguous_bytes.is_multiple_of(8)
            && u64::from(inner.contiguous_bytes) * u64::from(dimension.count) <= 512
    });
    let outer = inner.dimensions.split_off(usize::from(retain_inner));
    let offsets = crate::CopyGeometry {
        dimensions: outer,
        ..inner.clone()
    }
    .offsets()
    .ok_or(LowLoweringError::IdOverflow)?;
    offsets
        .into_iter()
        .map(|(source_offset, destination_offset)| {
            Ok(LocalCopy {
                source,
                source_offset,
                destination,
                destination_offset,
                bytes: u32::try_from(inner.bytes()).map_err(|_| LowLoweringError::IdOverflow)?,
                pattern: inner.dimensions.first().map_or(
                    LocalCopyPattern::Contiguous,
                    |dimension| LocalCopyPattern::Strided {
                        rows: dimension.count,
                        row_bytes: inner.contiguous_bytes,
                        source_stride: dimension.source_stride,
                        destination_stride: dimension.destination_stride,
                    },
                ),
            })
        })
        .collect()
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
                .operator_plan()
                .and_then(|plan| plan.requirements.inputs.get(index));
            if let Some(requirement) = requirement {
                alignment = alignment.max(requirement.allocation.alignment);
                access_tail = access_tail.max(requirement.allocation.access_tail_bytes);
            } else if operation.conversion().is_some() {
                alignment = alignment.max(8);
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
            MidOperationKind::CastPrecision => WorkReason::PrecisionCast,
            MidOperationKind::Rearrange(..) | MidOperationKind::View(..) => {
                WorkReason::LayoutRearrangement
            }
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

fn shard_extents(tensor_type: &TensorType) -> LowLoweringResult<Vec<(u16, TensorRegion)>> {
    Ok(tensor_type
        .format
        .layout
        .resolve(&tensor_type.shape)?
        .shard_extents()
        .into_iter()
        .map(|shard| (shard.tile, shard.extents))
        .collect())
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
