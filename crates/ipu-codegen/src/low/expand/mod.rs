//! Expand selected whole-device primitives into tile-local calls and movement.

mod cache;
mod emit;
mod exchange_grouping;
pub(crate) use cache::ExpansionCache;
mod primitive;

mod materialize;

mod buffers;
mod conversion;
mod copies;
mod gemm;
mod mapping;
mod ownership;
mod pointwise;
mod reduce;
mod repeat;
use crate::graph::OperationId;
use crate::low::*;
use crate::storage::{ByteSpan, StorageError};
use crate::{
    AMP_COLUMN_MICRO, AmpOrder, AxisFactorView, ConversionStrategy, CopyOrder, ElementOrder,
    KernelRequirements, Layout, LayoutError, MemoryClass, MidOperation, MidOperationKind,
    MidProgram, MidRepeat, MidValueId, OutputAliasing, Precision, ShardExtent, TensorTiling,
    TensorType, TileKernelSpec,
};
#[cfg(test)]
pub use copies::view_byte_spans;
pub(crate) use copies::view_byte_traversal;
use copies::*;
pub use copies::{logical_view_byte_spans, shard_storage_bytes};
use gemm::*;
use mapping::*;
use ownership::CopyRegions;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ExpansionError {
    #[error("tile expansion requires a nonzero tile count")]
    EmptyTileGroup,
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

pub type ExpansionResult<T> = Result<T, ExpansionError>;

#[cfg(test)]
pub(crate) fn expand_tiles(
    graph: &MidProgram,
    checkpoints: bool,
) -> ExpansionResult<Arc<TileGraph>> {
    expand_tiles_cached(graph, checkpoints, Arc::new(ExpansionCache::default()))
}

#[cfg(test)]
pub(crate) fn expand_tiles_cached(
    graph: &MidProgram,
    checkpoints: bool,
    cache: Arc<ExpansionCache>,
) -> ExpansionResult<Arc<TileGraph>> {
    expand_tiles_analyzed(
        graph,
        checkpoints,
        cache,
        &mut crate::estimate::GeometryAnalysis::default(),
    )
}

pub(crate) fn expand_tiles_analyzed(
    graph: &MidProgram,
    checkpoints: bool,
    cache: Arc<ExpansionCache>,
    analysis: &mut crate::estimate::GeometryAnalysis,
) -> ExpansionResult<Arc<TileGraph>> {
    if graph.tile_count == 0 {
        return Err(ExpansionError::EmptyTileGroup);
    }
    let start = Instant::now();
    let mut state = TileGraphBuilder::new(graph)?;
    state.cache = cache;
    let body = state.build_region(&graph.operations, checkpoints)?;
    let inputs = graph
        .inputs
        .iter()
        .map(|input| {
            Ok(ProgramInput {
                name: input.name.clone(),
                kind: input.kind,
                value: input.value,
                shards: state.value_shards(input.value)?.to_vec(),
            })
        })
        .collect::<ExpansionResult<_>>()?;
    let outputs = graph
        .outputs
        .iter()
        .map(|value| {
            Ok(ValueBlocks {
                value: *value,
                shards: state.value_shards(*value)?.to_vec(),
            })
        })
        .collect::<ExpansionResult<_>>()?;
    let values = graph
        .values
        .iter()
        .filter_map(|value| {
            let shards = &state.canonical[value.id.index() as usize];
            (!shards.is_empty()).then(|| ValueBlocks {
                value: value.id,
                shards: shards.clone(),
            })
        })
        .collect();
    let mut program = TileGraph {
        tile_count: graph.tile_count,
        shards: state.shards,
        exchange_phases: state.phases,
        inputs,
        body,
        kernel_runs: state.kernel_runs,
        local_copies: state.local_copies,
        values,
        outputs,
        logical_values: graph.values.clone(),
        checkpoints: graph
            .operations
            .iter()
            .enumerate()
            .filter(|(index, operation)| {
                graph
                    .operations
                    .get(index + 1)
                    .is_none_or(|next| next.source != operation.source)
            })
            .filter_map(|(_, operation)| {
                operation
                    .source
                    .map(|source| (source, operation.results.clone()))
            })
            .collect(),
        estimated_cycles: graph.estimated_cycles,
        estimated_exchange_cycles: graph.estimated_exchange_cycles,
    };
    let build_time = start.elapsed();
    let start = Instant::now();
    crate::low::passes::simplify(&mut program);
    let simplify_time = start.elapsed();
    let start = Instant::now();
    let cycles = crate::estimate::program_cycles_analyzed(&program, None, analysis)?;
    tracing::debug!(
        shards = program.shards.len(),
        exchange_phases = program.exchange_phases.len(),
        build_ms = build_time.as_secs_f64() * 1000.0,
        simplify_ms = simplify_time.as_secs_f64() * 1000.0,
        cost_ms = start.elapsed().as_secs_f64() * 1000.0,
        "built and costed logical tile schedule"
    );
    program.estimated_cycles = cycles.total;
    program.estimated_exchange_cycles = cycles.exchange;
    Ok(Arc::new(program))
}

struct TileGraphBuilder {
    cache: Arc<ExpansionCache>,
    tile_count: u16,
    storage_groups: Vec<MidValueId>,
    shards: Vec<BlockValue>,
    canonical: Vec<Vec<BlockValueId>>,
    phases: Vec<ExchangePhase>,
    kernel_runs: Vec<KernelRun>,
    local_copies: Vec<LocalCopy>,
    kernel_metadata: Vec<Arc<KernelRunMetadata>>,
    materialized_views: BTreeMap<BlockValueId, ShardView>,
}

impl TileGraphBuilder {
    fn new(graph: &MidProgram) -> ExpansionResult<Self> {
        let tile_count = graph.tile_count;
        let mut state = Self {
            cache: Arc::new(ExpansionCache::default()),
            tile_count,
            storage_groups: graph
                .values
                .iter()
                .map(|value| value.storage_group)
                .collect(),
            shards: Vec::new(),
            canonical: vec![Vec::new(); graph.values.len()],
            phases: Vec::new(),
            kernel_runs: Vec::new(),
            local_copies: Vec::new(),
            kernel_metadata: Vec::new(),
            materialized_views: BTreeMap::new(),
        };
        let mut used = graph
            .inputs
            .iter()
            .map(|input| input.value)
            .chain(graph.outputs.iter().copied())
            .collect::<BTreeSet<_>>();
        fn uses(operations: &[MidOperation], used: &mut BTreeSet<MidValueId>) {
            for operation in operations {
                used.extend(operation.inputs.iter().chain(&operation.results).copied());
                for deferred in operation.deferred_inputs().iter().flatten() {
                    used.extend([deferred.source, deferred.producer]);
                }
                if let MidOperationKind::Repeat(repeat) = &operation.kind {
                    used.extend(repeat.iterated_inputs.iter().flatten().copied());
                    used.extend(
                        repeat
                            .body
                            .arguments
                            .iter()
                            .chain(&repeat.body.yields)
                            .copied(),
                    );
                    uses(&repeat.body.operations, used);
                }
            }
        }
        uses(&graph.operations, &mut used);
        for value in &graph.values {
            if !used.contains(&value.id) {
                continue;
            }
            let layout = &value.tensor_type.format.layout;
            layout.validate_tile_count(tile_count)?;
            let extents = layout.shard_extents(&value.tensor_type.shape)?;
            let mut value_shards = Vec::with_capacity(extents.len());
            for (owner, extents) in extents {
                let id = state.push_shard(BlockValue {
                    id: BlockValueId(0),
                    tile: ((usize::from(owner) + usize::from(value.tile_offset))
                        % usize::from(tile_count)) as u16,
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

    fn build_region(
        &mut self,
        operations: &[MidOperation],
        checkpoints: bool,
    ) -> ExpansionResult<BlockRegion> {
        let mut tiles = BlockRegion::default();
        let mut checkpoint = 0u8;
        let mut index = 0;
        while index < operations.len() {
            let operation = &operations[index];
            let started = Instant::now();
            let copies = crate::mid::independent_copy_prefix(
                &operations[index..],
                checkpoints,
                &self.storage_groups,
            );
            let sums = crate::mid::independent_sum_prefix(
                &operations[index..],
                checkpoints,
                &self.storage_groups,
            );
            let group = copies.max(sums);
            let lowered = if sums > 1 {
                let mut batch = reduce::SumBatch::default();
                for operation in &operations[index..index + sums] {
                    let MidOperationKind::Primitive(crate::Primitive::Sum { axis, staging }) =
                        operation.kind
                    else {
                        unreachable!()
                    };
                    self.prepare_sum(operation, usize::from(axis), staging, &mut batch)?;
                }
                self.append_sum_batch(
                    batch,
                    WorkProvenance {
                        operation: None,
                        value: None,
                        reason: WorkReason::OperatorKernel,
                    },
                    &mut tiles,
                )
            } else if copies > 1 {
                let mut batch = conversion::MaterializationBatch::default();
                for operation in &operations[index..index + group] {
                    let MidOperationKind::Primitive(crate::Primitive::Copy {
                        mapping,
                        reuse_local,
                    }) = &operation.kind
                    else {
                        unreachable!()
                    };
                    self.prepare_copy_tensor(
                        operation,
                        mapping,
                        *reuse_local,
                        &mut batch,
                        &mut tiles,
                    )?;
                }
                let mut provenance = operation_provenance(operation);
                provenance.value = None;
                if operations[index..index + group]
                    .iter()
                    .any(|next| next.source != operation.source)
                {
                    provenance.operation = None;
                }
                self.append_materialization(batch, provenance, &mut tiles)
            } else {
                match &operation.kind {
                    MidOperationKind::Primitive(primitive) => {
                        self.build_primitive(operation, primitive, &mut tiles)
                    }
                    MidOperationKind::Repeat(repeat) => {
                        self.build_repeat(operation, repeat, &mut tiles)
                    }
                    MidOperationKind::Operator { .. } => Err(ExpansionError::InvalidOperatorPlan),
                    MidOperationKind::Convert(plan) => {
                        self.build_conversion(operation, plan, &mut tiles)
                    }
                }
            };
            if let Err(error) = lowered {
                tracing::debug!(
                    operation = index,
                    source = ?operation.source.map(OperationId::index),
                    kind = ?operation.kind,
                    plan = ?operation.operator_plan(),
                    conversion = ?operation.conversion_plan(),
                    inputs = ?operation.inputs,
                    results = ?operation.results,
                    ?error,
                    "failed to lower mid operation to tile work"
                );
                return Err(error);
            }
            if checkpoints
                && operations
                    .get(index + group.max(1))
                    .is_none_or(|next| next.source != operation.source)
                && let Some(source) = operation.source
            {
                tiles
                    .operations
                    .push(BlockOperation::Checkpoint(source, checkpoint));
                checkpoint ^= 1;
            }
            tracing::debug!(
                operation = index,
                source = ?operation.source.map(OperationId::index),
                elapsed_ms = started.elapsed().as_millis() as u64,
                shards = self.shards.len(),
                exchange_phases = self.phases.len(),
                "lowered mid operation to tile work"
            );
            index += group.max(1);
        }
        Ok(tiles)
    }

    fn append_fill_zero(
        &mut self,
        tiles: &mut BlockRegion,
        shard: BlockValueId,
        provenance: WorkProvenance,
    ) -> ExpansionResult<()> {
        let bytes = crate::shard_storage_bytes(&self.shards[shard.index() as usize])?;
        self.append_zero_range(
            tiles,
            shard,
            crate::ByteSpan { offset: 0, bytes },
            false,
            provenance,
        )
    }

    /// All these clears precede the copies. Overwriting a covered gap is safe
    /// and cheaper than another launch when the gap is sufficiently small.
    fn append_copy_clears(
        &mut self,
        tiles: &mut BlockRegion,
        shard: BlockValueId,
        ranges: &[crate::ByteSpan],
        provenance: WorkProvenance,
    ) -> ExpansionResult<()> {
        let block = &self.shards[shard.index() as usize];
        let padding = crate::low::copy::uncovered_copy_bytes(
            block.storage(),
            &[crate::CopyMapping {
                source: block.storage(),
                source_extents: &block.extents,
                destination_extents: &block.extents,
            }],
            CopyOrder::Semantic,
        )?;
        let padding_only = !padding.is_empty() && ranges == padding;
        let mut ranges = ranges.iter().copied().peekable();
        let launch_bytes = crate::estimate::IPU21_TARGET_COSTS.kernel_launch_cycles * 48;
        while let Some(mut range) = ranges.next() {
            while let Some(next) = ranges.peek()
                && u64::from(next.offset - (range.offset + range.bytes)) <= launch_bytes
            {
                range.bytes = next.offset + next.bytes - range.offset;
                ranges.next();
            }
            self.append_zero_range(tiles, shard, range, padding_only, provenance)?;
        }
        Ok(())
    }

    fn append_zero_range(
        &mut self,
        tiles: &mut BlockRegion,
        shard: BlockValueId,
        range: crate::ByteSpan,
        padding_only: bool,
        provenance: WorkProvenance,
    ) -> ExpansionResult<()> {
        let tile = self.shards[shard.index() as usize].tile;
        {
            let run = self.kernel_run(
                provenance,
                TileKernelSpec::FillZero {
                    offset: range.offset,
                    bytes: range.bytes,
                    padding_only,
                },
                Vec::new(),
                self.full_view(shard),
            )?;
            self.append_kernel(tiles, tile, run)
        }
    }
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
            MidOperationKind::Primitive(crate::Primitive::Copy { .. }) => {
                WorkReason::OperatorInputs
            }
            MidOperationKind::Operator { .. } | MidOperationKind::Primitive(_) => {
                WorkReason::OperatorKernel
            }
            MidOperationKind::Repeat(_) => WorkReason::Repeat,
        },
    }
}

#[cfg(test)]
mod tests;
