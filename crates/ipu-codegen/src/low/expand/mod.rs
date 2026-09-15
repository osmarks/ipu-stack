//! Expand selected whole-device primitives into tile-local calls and movement.

mod emit;
use crate::storage::GeometryCache;
mod compute;

mod buffers;
mod cast;
mod copies;
mod gemm;
mod mapping;
mod movement;
mod ownership;
mod reduce;
mod repeat;
#[cfg(test)]
use super::{view_byte_spans, view_byte_traversal};
use crate::graph::OperationId;
use crate::low::*;
use crate::mid::Compute;
#[cfg(test)]
use crate::storage::ByteSpan;
use crate::storage::StorageError;
use crate::{
    AMP_COLUMN_MICRO, AmpOrder, AxisFactorView, CopyOrder, CopyPolicy, ElementOrder, Layout,
    LayoutError, MemoryClass, MidOperation, MidOperationKind, MidProgram, MidRepeat, MidValueId,
    Precision, ShardExtent, TensorTiling, TensorType, TileKernelSpec,
};
use copies::*;

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
    #[error("copy policy is incompatible with its mapping or bindings")]
    InvalidCopyPlan,
    #[error("repeat structure is inconsistent with its inputs, arguments, yields, or results")]
    InvalidRepeat,
    #[error("repeat carried value {0} cannot alias its body argument")]
    RepeatRequiresInPlace(usize),
    #[error("too many logical shards or exchange phases")]
    IdOverflow,
    #[error("invalid tensor layout: {0}")]
    Layout(#[from] LayoutError),
    #[error("invalid kernel binding: {0}")]
    Kernel(#[from] crate::kernel::KernelError),
    #[error("invalid tensor storage view: {0}")]
    Storage(#[from] StorageError),
}

pub type ExpansionResult<T> = Result<T, ExpansionError>;

#[cfg(test)]
pub(crate) fn expand_tiles(
    graph: &MidProgram,
    checkpoints: bool,
) -> ExpansionResult<Arc<TileGraph>> {
    expand_tiles_cached(graph, checkpoints, Arc::new(GeometryCache::default()))
}

pub(crate) fn expand_tiles_cached(
    graph: &MidProgram,
    checkpoints: bool,
    cache: Arc<GeometryCache>,
) -> ExpansionResult<Arc<TileGraph>> {
    if graph.tile_count == 0 {
        return Err(ExpansionError::EmptyTileGroup);
    }
    let mut state = TileGraphBuilder::new(graph, Arc::clone(&cache))?;
    let body = state.build_region(&graph.operations, checkpoints)?;
    for value in graph
        .inputs
        .iter()
        .map(|input| input.value)
        .chain(graph.outputs.iter().copied())
    {
        state.value_views(value)?;
    }
    let mut program = TileGraph {
        tile_count: graph.tile_count,
        requires_finite_scratch: false,
        shards: state.shards,
        exchange_phases: state.phases,
        inputs: graph.inputs.clone(),
        body,
        kernel_runs: state.kernel_runs,
        local_copies: state.local_copies,
        value_views: state.bindings,
        outputs: graph.outputs.clone(),
        logical_values: state.logical_values,
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
    };
    crate::low::passes::run(&mut program, &cache)?;
    tracing::debug!(
        shards = program.shards.len(),
        exchange_phases = program.exchange_phases.len(),
        "built logical tile schedule"
    );
    Ok(Arc::new(program))
}

struct TileGraphBuilder {
    cache: Arc<GeometryCache>,
    tile_count: u16,
    storage_groups: Vec<MidValueId>,
    shards: Vec<BlockValue>,
    bindings: Vec<Vec<ShardView>>,
    logical_values: Vec<crate::MidValue>,
    exported_values: BTreeSet<MidValueId>,
    required_storage: BTreeSet<MidValueId>,
    phases: Vec<ExchangePhase>,
    kernel_runs: Vec<KernelRun>,
    local_copies: Vec<crate::kernel::CopyRun>,
    kernel_metadata: Vec<Arc<KernelRunMetadata>>,
}

impl TileGraphBuilder {
    fn new(graph: &MidProgram, cache: Arc<GeometryCache>) -> ExpansionResult<Self> {
        let tile_count = graph.tile_count;
        let mut state = Self {
            cache,
            tile_count,
            storage_groups: graph
                .values
                .iter()
                .map(|value| value.storage_group)
                .collect(),
            shards: Vec::new(),
            bindings: vec![Vec::new(); graph.values.len()],
            logical_values: graph.values.clone(),
            exported_values: graph.outputs.iter().copied().collect(),
            required_storage: BTreeSet::new(),
            phases: Vec::new(),
            kernel_runs: Vec::new(),
            local_copies: Vec::new(),
            kernel_metadata: Vec::new(),
        };
        let mut used = graph
            .inputs
            .iter()
            .map(|input| input.value)
            .chain(graph.outputs.iter().copied())
            .collect::<BTreeSet<_>>();
        fn uses(
            operations: &[MidOperation],
            used: &mut BTreeSet<MidValueId>,
            bindings: &mut BTreeSet<MidValueId>,
        ) {
            for operation in operations {
                used.extend(operation.read_values().chain(&operation.results).copied());
                if let MidOperationKind::Compute(compute) = &operation.kind {
                    bindings
                        .extend(compute::allocation_inputs(compute, &operation.inputs).copied());
                }
                if let MidOperationKind::Repeat(repeat) = &operation.kind {
                    used.extend(
                        repeat
                            .body
                            .arguments
                            .iter()
                            .chain(&repeat.body.yields)
                            .copied(),
                    );
                    bindings.extend(
                        operation
                            .read_values()
                            .chain(&operation.results)
                            .chain(&repeat.body.arguments)
                            .chain(&repeat.body.yields)
                            .copied(),
                    );
                    uses(&repeat.body.operations, used, bindings);
                }
            }
        }
        uses(&graph.operations, &mut used, &mut state.required_storage);
        for value in &graph.values {
            if !used.contains(&value.id) {
                continue;
            }
            let layout = &value.tensor_type.format.layout;
            layout.validate_tile_count(tile_count)?;
            value
                .owners
                .validate(layout.tiling.tile_count, tile_count)?;
            let extents = layout.shard_extents(&value.tensor_type.shape)?;
            let mut value_views = Vec::with_capacity(extents.len());
            for (owner, extents) in extents {
                let id = state.push_shard(BlockValue {
                    id: BlockValueId(0),
                    tile: value.owners.tile(owner, tile_count).ok_or(
                        crate::tensor::LayoutError::InvalidOwnerMap {
                            owners: layout.tiling.tile_count,
                            tiles: tile_count,
                        },
                    )?,
                    tensor_type: value.tensor_type.clone(),
                    extents,
                    definition: ShardDefinition::Value(value.id),
                })?;
                value_views.push(state.full_view(id));
            }
            state.bindings[value.id.index() as usize] = value_views;
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
                    let MidOperationKind::Compute(Compute::Sum { axis, staging }) = operation.kind
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
            } else if copies != 0 {
                let mut batch = movement::MaterializationBatch::default();
                for operation in &operations[index..index + group] {
                    self.prepare_copy_tensor(operation, &mut batch, &mut tiles)?;
                }
                let mut provenance = operation_provenance(operation);
                if copies > 1 {
                    provenance.value = None;
                }
                if operations[index..index + group]
                    .iter()
                    .any(|next| next.source != operation.source)
                {
                    provenance.operation = None;
                }
                self.append_materialization(batch, provenance, &mut tiles)
            } else {
                match &operation.kind {
                    MidOperationKind::Compute(compute) => {
                        self.build_compute(operation, compute, &mut tiles)
                    }
                    MidOperationKind::Copy { .. } => {
                        unreachable!("copy prefix includes its first operation")
                    }
                    MidOperationKind::Repeat(repeat) => {
                        self.build_repeat(operation, repeat, &mut tiles)
                    }
                }
            };
            if let Err(error) = lowered {
                tracing::debug!(
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
        let padding = crate::storage::uncovered_bytes(
            block.storage(),
            [block.extents.as_slice()],
            CopyOrder::Semantic,
        )?;
        let padding_only = !padding.is_empty() && ranges == padding;
        let bytes = shard_storage_bytes(block)?;
        let mut ranges = ranges
            .iter()
            .map(|range| {
                let start = range.offset / 8 * 8;
                let end = (u64::from(range.offset) + u64::from(range.bytes)).div_ceil(8) * 8;
                crate::ByteSpan {
                    offset: start,
                    bytes: end.min(u64::from(bytes)) as u32 - start,
                }
            })
            .peekable();
        let launch_bytes = crate::estimate::IPU21_TARGET_COSTS.kernel_launch_cycles * 48;
        while let Some(mut range) = ranges.next() {
            while let Some(next) = ranges.peek()
                && u64::from(next.offset.saturating_sub(range.offset + range.bytes)) <= launch_bytes
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
            let run = self.bind_kernel(
                provenance,
                TileKernelSpec::FillZero {
                    offset: range.offset,
                    bytes: range.bytes,
                    padding_only,
                },
                Vec::new(),
                vec![self.full_view(shard)],
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
            MidOperationKind::Compute(Compute::Kernel {
                kernel: TileKernelSpec::Cast { .. },
                ..
            }) => WorkReason::PrecisionCast,
            MidOperationKind::Copy {
                policy: CopyPolicy::Automatic,
                ..
            } => WorkReason::OperatorInputs,
            MidOperationKind::Copy { .. } => WorkReason::LayoutRearrangement,
            MidOperationKind::Compute(_) => WorkReason::OperatorKernel,
            MidOperationKind::Repeat(_) => WorkReason::Repeat,
        },
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "tests/output_bindings.rs"]
mod output_binding_tests;

#[cfg(test)]
#[path = "tests/mapping_semantics.rs"]
mod mapping_semantics;

#[cfg(test)]
#[path = "tests/primitive_fragments.rs"]
mod primitive_fragments;
