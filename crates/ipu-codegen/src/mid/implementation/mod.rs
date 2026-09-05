//! Build executable mid blocks from an implementation candidate.

mod emit;

mod attention_blocked;
mod attention_materialized;
mod attention_panels;
mod deferred;
mod views;

mod attention;
mod buffers;
mod conversion;
mod copies;
mod gemm;
mod gemm_parallel;
mod gemm_streamed;
mod mapping;
mod pointwise;
mod reduce;
mod repeat;
use super::block::*;
use crate::graph::OperationId;
use crate::mid::{
    AMP_COLUMN_MICRO, AMP_INNER_BLOCK, AmpOrder, AxisFactorView, BlockMajorOrder,
    ConversionStrategy, CopyOrder, CopyPattern, ElementOrder, GemmDistribution,
    ImplementationCandidate, Layout, LayoutError, MemoryClass, MemoryOperand, MidOperation,
    MidOperationKind, MidRepeat, MidValueId, OperandRequirement, OperatorDispatch, OutputAliasing,
    PointwiseInputMapping, Precision, ShardExtent, StorageRequirements, TensorTiling, TensorType,
    TileKernelSpec,
};
use crate::storage::{ByteSpan, StorageError};
use attention::*;
use copies::*;
pub use copies::{logical_view_byte_spans, shard_storage_bytes, view_byte_spans};
use deferred::*;
use gemm::*;
use mapping::*;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BlockBuildError {
    #[error("mid block construction requires a nonzero tile count")]
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

pub type BlockBuildResult<T> = Result<T, BlockBuildError>;

pub(crate) fn build_blocks(graph: &ImplementationCandidate) -> BlockBuildResult<Arc<MidProgram>> {
    if graph.tile_count == 0 {
        return Err(BlockBuildError::EmptyTileGroup);
    }
    let mut state = BlockBuilder::new(graph)?;
    let body = state.build_region(&graph.operations, true)?;
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
        .collect::<BlockBuildResult<_>>()?;
    let outputs = graph
        .outputs
        .iter()
        .map(|value| {
            Ok(ValueBlocks {
                value: *value,
                shards: state.value_shards(*value)?.to_vec(),
            })
        })
        .collect::<BlockBuildResult<_>>()?;
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
    tracing::info!(
        shards = state.shards.len(),
        exchange_phases = state.phases.len(),
        "built logical tile schedule"
    );
    let mut program = MidProgram {
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
            .filter_map(|operation| {
                matches!(
                    operation.kind,
                    MidOperationKind::Operator { .. } | MidOperationKind::Repeat(_)
                )
                .then(|| {
                    operation
                        .source
                        .map(|source| (source, operation.results.clone()))
                })
                .flatten()
            })
            .collect(),
        estimated_cycles: graph.estimated_cycles,
        estimated_exchange_cycles: graph.estimated_exchange_cycles,
    };
    super::passes::simplify(&mut program);
    Ok(Arc::new(program))
}

type ShardIntersections = Vec<(Vec<ShardExtent>, Vec<BlockValueId>)>;

struct BlockBuilder {
    tile_count: u16,
    shards: Vec<BlockValue>,
    canonical: Vec<Vec<BlockValueId>>,
    phases: Vec<ExchangePhase>,
    kernel_runs: Vec<KernelRun>,
    local_copies: Vec<LocalCopy>,
    kernel_metadata: Vec<Arc<KernelRunMetadata>>,
    deferred_conversions: BTreeMap<MidValueId, MidValueId>,
    deferred_values: BTreeMap<MidValueId, DeferredValue>,
    intersection_cache: BTreeMap<(MidValueId, Vec<ShardExtent>), ShardIntersections>,
}

impl BlockBuilder {
    fn new(graph: &ImplementationCandidate) -> BlockBuildResult<Self> {
        let tile_count = graph.tile_count;
        let mut state = Self {
            tile_count,
            shards: Vec::new(),
            canonical: vec![Vec::new(); graph.values.len()],
            phases: Vec::new(),
            kernel_runs: Vec::new(),
            local_copies: Vec::new(),
            kernel_metadata: Vec::new(),
            deferred_conversions: BTreeMap::new(),
            deferred_values: BTreeMap::new(),
            intersection_cache: BTreeMap::new(),
        };
        for value in &graph.values {
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

    fn intersecting_shards(
        &mut self,
        source: MidValueId,
        target: &[ShardExtent],
        local_tile: u16,
    ) -> BlockBuildResult<Vec<(Vec<ShardExtent>, BlockValueId)>> {
        let key = (source, target.to_vec());
        if !self.intersection_cache.contains_key(&key) {
            let groups = self.shard_intersection_groups(self.value_shards(source)?, target);
            self.intersection_cache.insert(key.clone(), groups);
        }
        Ok(self.select_intersections(&self.intersection_cache[&key], local_tile))
    }

    fn shard_intersection_groups(
        &self,
        sources: &[BlockValueId],
        target: &[ShardExtent],
    ) -> ShardIntersections {
        let mut groups = BTreeMap::<Vec<ShardExtent>, Vec<BlockValueId>>::new();
        for &source in sources {
            if let Some(extents) =
                intersect_extents(&self.shards[source.index() as usize].extents, target)
            {
                groups.entry(extents).or_default().push(source);
            }
        }
        groups.into_iter().collect()
    }

    fn select_intersections(
        &self,
        groups: &ShardIntersections,
        local_tile: u16,
    ) -> Vec<(Vec<ShardExtent>, BlockValueId)> {
        groups
            .iter()
            .map(|(extents, candidates)| {
                let selected = candidates
                    .iter()
                    .copied()
                    .find(|source| self.shards[source.index() as usize].tile == local_tile)
                    .unwrap_or(candidates[0]);
                (extents.clone(), selected)
            })
            .collect()
    }

    fn intersecting_shard_set(
        &self,
        sources: &[BlockValueId],
        target: &[ShardExtent],
        local_tile: u16,
    ) -> Vec<(Vec<ShardExtent>, BlockValueId)> {
        self.select_intersections(&self.shard_intersection_groups(sources, target), local_tile)
    }

    fn build_region(
        &mut self,
        operations: &[MidOperation],
        checkpoints: bool,
    ) -> BlockBuildResult<BlockRegion> {
        let mut tiles = BlockRegion::default();
        let mut checkpoint = 0u8;
        for (index, operation) in operations.iter().enumerate() {
            let started = Instant::now();
            if self.build_deferred_output(operation, &mut tiles)? {
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
                    self.build_repeat(operation, repeat, &mut tiles)
                }
                MidOperationKind::Operator { plan, .. } => {
                    self.build_operator(operation, plan, &mut tiles)
                }
                MidOperationKind::Convert(plan) => {
                    self.build_conversion(operation, plan, &mut tiles)
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
                tiles
                    .operations
                    .push(BlockOperation::Checkpoint(source, checkpoint));
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

    fn build_operator(
        &mut self,
        operation: &MidOperation,
        plan: &crate::OperatorPlan,
        tiles: &mut BlockRegion,
    ) -> BlockBuildResult<()> {
        match &plan.dispatch {
            OperatorDispatch::Pointwise {
                kernel,
                input_mapping,
            } => self.build_pointwise(
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
                    return Err(BlockBuildError::InvalidOperatorPlan);
                };
                let kernel = |mode| TileKernelSpec::Gemm {
                    multiply,
                    accumulate,
                    mode,
                    weights: crate::GemmWeightLoad::Standard,
                    inner_block: *inner_block,
                    output_columns: *output_column_block,
                };
                self.build_blocked_gemm(
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
            } => self.build_blocked_attention(
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
            } => self.build_materialized_attention(
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
            OperatorDispatch::View => self.build_view(operation, &plan.operator, tiles),
        }
    }

    fn shard_has_padding(&self, shard: BlockValueId) -> bool {
        self.shards[shard.index() as usize]
            .extents
            .iter()
            .any(|extent| extent.logical_end < extent.physical_end)
    }

    fn append_fill_zero(
        &mut self,
        tiles: &mut BlockRegion,
        shard: BlockValueId,
        provenance: WorkProvenance,
    ) -> BlockBuildResult<()> {
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
                StorageRequirements {
                    inputs: Vec::new(),
                    output,
                    output_aliasing: crate::OutputAliasing::Fresh,
                    distinct_elements: Vec::new(),
                },
            ),
        )
    }

    fn dispatch_input_view(
        &mut self,
        value: MidValueId,
        tile: u16,
        ranges: &[(usize, u32, u32)],
        transfers: &mut BTreeMap<ShardView, Vec<ShardView>>,
        local_copies: &mut Vec<(u16, LocalCopy)>,
    ) -> BlockBuildResult<ShardView> {
        let target = self.local_shard(value, tile)?;
        let target_view = self.narrow_view(target, ranges)?;
        let Some(source_value) = self.deferred_conversions.get(&value).copied() else {
            return Ok(target_view);
        };

        let staging = self.push_shard(BlockValue {
            id: BlockValueId(0),
            tile,
            tensor_type: self.shards[target.index() as usize].tensor_type.clone(),
            extents: target_view.extents.clone(),
            definition: ShardDefinition::ExchangeStaging,
        })?;
        let intersections = self.intersecting_shards(source_value, &target_view.extents, tile)?;
        if intersections.is_empty() {
            return Err(BlockBuildError::InvalidConversionPlan);
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
                append_span_copies(
                    &self.shards,
                    &source_view,
                    &destination_view,
                    tile,
                    local_copies,
                    CopyOrder::Semantic,
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
