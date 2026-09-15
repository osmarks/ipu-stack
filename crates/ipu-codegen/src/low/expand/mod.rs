//! Expand selected whole-device primitives into tile-local calls and movement.
//!
//! This module owns builder state, region dispatch, and shared graph construction.
//! `compute` binds resident operands to calls; `copy` realizes movement and its
//! initialization; `repeat` constructs loop bodies and their storage bindings.

use crate::mid::MidOperationKind;
use crate::storage::GeometryCache;
mod compute;

mod copy;
mod repeat;
#[cfg(test)]
use super::{view_byte_spans, view_byte_traversal};
use crate::graph::OperationId;
use crate::low::*;

#[cfg(test)]
use crate::storage::ByteSpan;
use crate::storage::StorageError;
use crate::{
    AMP_COLUMN_MICRO, AmpOrder, AxisFactorView, CopyOrder, CopyPolicy, ElementOrder, Layout,
    LayoutError, MemoryClass, MidOperation, MidProgram, MidRepeat, MidValueId, PackingPolicy,
    Precision, ShardExtent, TensorTiling, TensorType,
};
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
    expand_tiles_cached(
        graph,
        checkpoints,
        false,
        Arc::new(GeometryCache::default()),
    )
}

pub(crate) fn expand_tiles_cached(
    graph: &MidProgram,
    checkpoints: bool,
    reuse_cast_inputs: bool,
    cache: Arc<GeometryCache>,
) -> ExpansionResult<Arc<TileGraph>> {
    if graph.tile_count == 0 {
        return Err(ExpansionError::EmptyTileGroup);
    }
    let mut state = TileGraphBuilder::new(graph, Arc::clone(&cache))?;
    state.program.body = state.build_region(&graph.operations, checkpoints)?;
    for value in graph
        .inputs
        .iter()
        .map(|input| input.value)
        .chain(graph.outputs.iter().copied())
    {
        state.value_views(value)?;
    }
    let mut program = state.program;
    crate::low::passes::run(&mut program, &cache, reuse_cast_inputs)?;
    tracing::debug!(
        shards = program.shards.len(),
        exchange_phases = program.exchange_phases.len(),
        "built logical tile schedule"
    );
    Ok(Arc::new(program))
}

struct TileGraphBuilder {
    cache: Arc<GeometryCache>,
    program: TileGraph,
    storage_groups: Vec<MidValueId>,
    kernel_metadata: Vec<Arc<KernelRunMetadata>>,
}

impl TileGraphBuilder {
    fn new(graph: &MidProgram, cache: Arc<GeometryCache>) -> ExpansionResult<Self> {
        let tile_count = graph.tile_count;
        let mut state = Self {
            cache,
            program: TileGraph {
                tile_count: graph.tile_count,
                requires_finite_scratch: false,
                shards: Vec::new(),
                exchange_phases: Vec::new(),
                inputs: graph.inputs.clone(),
                body: BlockRegion::default(),
                kernel_runs: Vec::new(),
                local_copies: Vec::new(),
                value_views: vec![Vec::new(); graph.values.len()],
                outputs: graph.outputs.clone(),
                logical_values: graph.values.clone(),
                checkpoints: graph
                    .operations
                    .chunk_by(|a, b| a.source == b.source)
                    .filter_map(|group| {
                        let operation = group.last()?;
                        operation
                            .source
                            .map(|source| (source, operation.results.clone()))
                    })
                    .collect(),
            },
            storage_groups: graph
                .values
                .iter()
                .map(|value| value.storage_group)
                .collect(),
            kernel_metadata: Vec::new(),
        };
        let mut used = graph
            .inputs
            .iter()
            .map(|input| input.value)
            .chain(graph.outputs.iter().copied())
            .collect::<BTreeSet<_>>();
        fn uses(operations: &[MidOperation], used: &mut BTreeSet<MidValueId>) {
            for operation in operations {
                used.extend(operation.read_values().chain(&operation.results).copied());
                if let MidOperationKind::Repeat(repeat) = &operation.kind {
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
            state.program.value_views[value.id.index() as usize] = value_views;
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
            let group = copies;
            let lowered = if copies != 0 {
                self.lower_copies(&operations[index..index + group], &mut tiles)
            } else {
                match &operation.kind {
                    MidOperationKind::Copy { .. } => {
                        unreachable!("copy prefix includes its first operation")
                    }
                    MidOperationKind::Repeat(repeat) => {
                        self.build_repeat(operation, repeat, &mut tiles)
                    }
                    _ => self.build_compute(operation, &mut tiles),
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
                shards = self.program.shards.len(),
                exchange_phases = self.program.exchange_phases.len(),
                "lowered mid operation to tile work"
            );
            index += group.max(1);
        }
        Ok(tiles)
    }

    // Shared graph construction: bind calls, assign table IDs, and register
    // resident value views. Operator-specific selection lives in the child modules.
    fn bind_kernel(
        &mut self,
        provenance: WorkProvenance,
        kernel: MidOperationKind,
        inputs: Vec<ShardView>,
        outputs: Vec<ShardView>,
    ) -> ExpansionResult<KernelRun> {
        Ok(KernelRun::bind(
            provenance,
            kernel,
            inputs,
            outputs,
            &self.program.shards,
            &mut self.kernel_metadata,
        )?)
    }

    fn push_shard(&mut self, mut shard: BlockValue) -> ExpansionResult<BlockValueId> {
        let id = BlockValueId(
            u32::try_from(self.program.shards.len()).map_err(|_| ExpansionError::IdOverflow)?,
        );
        shard.id = id;
        self.program.shards.push(shard);
        Ok(id)
    }

    fn value_views(&self, value: MidValueId) -> ExpansionResult<&[ShardView]> {
        self.program
            .value_views
            .get(value.index() as usize)
            .filter(|shards| !shards.is_empty())
            .map(Vec::as_slice)
            .ok_or(ExpansionError::UnknownValue(value))
    }

    fn full_view(&self, shard: BlockValueId) -> ShardView {
        ShardView {
            shard,
            extents: self.program.shards[shard.index() as usize].extents.clone(),
        }
    }

    /// Bind a whole allocation when a write/region ABI needs its exact strides.
    /// Read consumers use value_views and keep their logical selection.
    fn allocation_shards(&self, value: MidValueId) -> ExpansionResult<Vec<BlockValueId>> {
        self.value_views(value)?
            .iter()
            .map(|view| {
                if view.extents != self.program.shards[view.shard.index() as usize].extents {
                    return Err(ExpansionError::InvalidOperatorPlan);
                }
                Ok(view.shard)
            })
            .collect()
    }

    fn append_exchange_phase(
        &mut self,
        transfers: Vec<LogicalExchange>,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        if transfers.is_empty() {
            return Ok(());
        }
        let id = ExchangePhaseId(
            u32::try_from(self.program.exchange_phases.len())
                .map_err(|_| ExpansionError::IdOverflow)?,
        );
        self.program.exchange_phases.push(ExchangePhase {
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
        tiles.operations.push(BlockOperation::Exchange(id));
        Ok(())
    }

    fn append_kernel(
        &mut self,
        tiles: &mut BlockRegion,
        tile: u16,
        run: KernelRun,
    ) -> ExpansionResult<()> {
        let id = KernelRunId(
            u32::try_from(self.program.kernel_runs.len())
                .map_err(|_| ExpansionError::IdOverflow)?,
        );
        self.program.kernel_runs.push(run);
        tiles
            .operations
            .push(BlockOperation::Compute { tile, run: id });
        Ok(())
    }

    fn append_local_copy(
        &mut self,
        tiles: &mut BlockRegion,
        tile: u16,
        copy: crate::kernel::CopyRun,
    ) -> ExpansionResult<()> {
        let id = LocalCopyId(
            u32::try_from(self.program.local_copies.len())
                .map_err(|_| ExpansionError::IdOverflow)?,
        );
        self.program.local_copies.push(copy);
        tiles
            .operations
            .push(BlockOperation::Copy { tile, copy: id });
        Ok(())
    }
}

fn operation_provenance(operation: &MidOperation) -> WorkProvenance {
    WorkProvenance {
        operation: operation.source,
        value: operation.results.first().copied(),
        reason: match &operation.kind {
            MidOperationKind::Cast { .. } => WorkReason::PrecisionCast,
            MidOperationKind::Copy {
                policy: CopyPolicy::Automatic,
                ..
            } => WorkReason::OperatorInputs,
            MidOperationKind::Copy { .. } => WorkReason::LayoutRearrangement,
            MidOperationKind::Repeat(_) => WorkReason::Repeat,
            _ => WorkReason::OperatorKernel,
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
