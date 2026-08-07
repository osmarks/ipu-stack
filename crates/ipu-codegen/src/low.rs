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
    PipelineConfig, TensorType, TileKernelSpec,
};
use std::collections::BTreeMap;

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
    ExchangeCopy(LowShardId),
    Alias(LowShardId),
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

/// One source shard may be sent to several destinations by a later exchange
/// implementation. Each destination is a distinct resident-copy shard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalExchange {
    pub source: ShardView,
    pub destinations: Vec<LowShardId>,
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
pub struct KernelRun {
    pub provenance: WorkProvenance,
    pub kernel: TileKernel,
    pub inputs: Vec<KernelOperand>,
    pub output: ShardView,
    pub requirements: KernelRequirements,
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
    Kernel(KernelRun),
    Repeat(RepeatRun),
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
    pub tiles: Vec<TileWorkList>,
    pub outputs: Vec<LowValue>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LowLoweringError {
    #[error("low-level lowering requires a nonzero tile count")]
    EmptyTileGroup,
    #[error("value {value:?} declares {declared} tiles, but the schedule uses {scheduled}")]
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
    let tiles = state.lower_region(&graph.operations)?;
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
        outputs,
    })
}

struct LoweringState {
    tile_count: u16,
    shards: Vec<LowShard>,
    canonical: Vec<Vec<LowShardId>>,
    phases: Vec<ExchangePhase>,
}

impl LoweringState {
    fn new(graph: &MidGraph, tile_count: u16) -> LowLoweringResult<Self> {
        let mut state = Self {
            tile_count,
            shards: Vec::new(),
            canonical: vec![Vec::new(); graph.values.len()],
            phases: Vec::new(),
        };
        for value in &graph.values {
            if value.tensor_type.format.layout.tiling.tile_count != tile_count {
                return Err(LowLoweringError::TileCountMismatch {
                    value: value.id,
                    declared: value.tensor_type.format.layout.tiling.tile_count,
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

    fn value_shards(&self, value: MidValueId) -> LowLoweringResult<&[LowShardId]> {
        self.canonical
            .get(value.index() as usize)
            .filter(|shards| !shards.is_empty())
            .map(Vec::as_slice)
            .ok_or(LowLoweringError::UnknownValue(value))
    }

    fn local_shard(&self, value: MidValueId, tile: u16) -> LowLoweringResult<LowShardId> {
        self.value_shards(value)?
            .iter()
            .copied()
            .find(|shard| self.shards[shard.index() as usize].tile == tile)
            .ok_or(LowLoweringError::UnknownValue(value))
    }

    fn lower_region(
        &mut self,
        operations: &[MidOperation],
    ) -> LowLoweringResult<Vec<TileWorkList>> {
        let mut tiles = (0..self.tile_count)
            .map(|tile| TileWorkList {
                tile,
                work: Vec::new(),
            })
            .collect::<Vec<_>>();
        for operation in operations {
            match &operation.kind {
                MidOperationKind::Repeat(repeat) => {
                    self.lower_repeat(operation, repeat, &mut tiles)?;
                }
                MidOperationKind::Operator(_) => self.lower_operator(operation, &mut tiles)?,
                kind => self.lower_conversion(operation, kind, &mut tiles)?,
            }
        }
        Ok(tiles)
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
        &self,
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
        for output in self.value_shards(*result)? {
            let tile = self.shards[output.index() as usize].tile;
            let input = self.local_shard(*input, tile)?;
            tiles[usize::from(tile)]
                .work
                .push(TileWork::Kernel(KernelRun {
                    provenance: operation_provenance(operation, kind),
                    kernel: TileKernel::Planned(plan.kernel.clone()),
                    inputs: vec![KernelOperand {
                        views: vec![self.full_view(input)],
                    }],
                    output: self.full_view(*output),
                    requirements: KernelRequirements::Conversion {
                        input: plan.input.clone(),
                        output: plan.output.clone(),
                    },
                }));
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
        let mut transfers = BTreeMap::<ShardView, Vec<LowShardId>>::new();
        let mut runs = Vec::new();
        for output in outputs {
            let tile = self.shards[output.index() as usize].tile;
            for source in &inputs {
                let Some(extents) = intersect_extents(
                    &self.shards[source.index() as usize].extents,
                    &self.shards[output.index() as usize].extents,
                ) else {
                    continue;
                };
                let source_view = ShardView {
                    shard: *source,
                    extents: extents.clone(),
                };
                let resident = if self.shards[source.index() as usize].tile == tile {
                    source_view.clone()
                } else {
                    let copy = self.push_shard(LowShard {
                        id: LowShardId(0),
                        tile,
                        tensor_type: self.shards[source.index() as usize].tensor_type.clone(),
                        extents: extents.clone(),
                        definition: ShardDefinition::ExchangeCopy(*source),
                    })?;
                    transfers.entry(source_view).or_default().push(copy);
                    self.full_view(copy)
                };
                runs.push((
                    tile,
                    KernelRun {
                        provenance: operation_provenance(operation, kind),
                        kernel: TileKernel::Planned(plan.kernel.clone()),
                        inputs: vec![KernelOperand {
                            views: vec![resident],
                        }],
                        output: ShardView {
                            shard: output,
                            extents,
                        },
                        requirements: KernelRequirements::Conversion {
                            input: plan.input.clone(),
                            output: plan.output.clone(),
                        },
                    },
                ));
            }
        }
        self.append_phase(transfers, operation_provenance(operation, kind), tiles)?;
        for (tile, run) in runs {
            tiles[usize::from(tile)].work.push(TileWork::Kernel(run));
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
            OperatorDispatch::Pointwise { kernel } => {
                self.lower_pointwise(operation, kernel.clone(), &plan.requirements, tiles)
            }
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
        requirements: &OperatorRequirements,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [result] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        let outputs = self.value_shards(*result)?.to_vec();
        let mut transfers =
            vec![BTreeMap::<ShardView, Vec<LowShardId>>::new(); operation.inputs.len()];
        let mut runs = Vec::with_capacity(outputs.len());
        for output in outputs {
            let tile = self.shards[output.index() as usize].tile;
            let inputs = operation
                .inputs
                .iter()
                .enumerate()
                .map(|(index, input)| {
                    let source_view =
                        if matches!(&kernel, TileKernelSpec::Add | TileKernelSpec::Gelu) {
                            self.value_shards(*input)?
                                .iter()
                                .find_map(|source| self.broadcast_view(*source, output))
                                .ok_or(LowLoweringError::InvalidOperatorPlan)?
                        } else {
                            self.full_view(self.local_shard(*input, tile)?)
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
                            definition: ShardDefinition::ExchangeCopy(source_view.shard),
                        })?;
                        transfers[index].entry(source_view).or_default().push(copy);
                        self.full_view(copy)
                    };
                    Ok(KernelOperand { views: vec![view] })
                })
                .collect::<LowLoweringResult<_>>()?;
            runs.push((
                tile,
                KernelRun {
                    provenance: WorkProvenance {
                        operation: operation.source,
                        value: operation.results.first().copied(),
                        reason: WorkReason::OperatorKernel,
                    },
                    kernel: TileKernel::Planned(kernel.clone()),
                    inputs,
                    output: self.full_view(output),
                    requirements: KernelRequirements::Operator(requirements.clone()),
                },
            ));
        }
        for (index, transfers) in transfers.into_iter().enumerate() {
            self.append_phase(
                transfers,
                WorkProvenance {
                    operation: operation.source,
                    value: operation.inputs.get(index).copied(),
                    reason: WorkReason::OperatorInput {
                        input: u16::try_from(index).map_err(|_| LowLoweringError::IdOverflow)?,
                    },
                },
                tiles,
            )?;
        }
        for (tile, run) in runs {
            tiles[usize::from(tile)].work.push(TileWork::Kernel(run));
        }
        Ok(())
    }

    fn broadcast_view(&self, source: LowShardId, output: LowShardId) -> Option<ShardView> {
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
                let required = output_shard.extents[offset + axis];
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

        for column_start in (0..column_extent).step_by(output_column_block as usize) {
            let column_end = column_start + output_column_block;
            for inner_start in (0..inner_extent).step_by(inner_block as usize) {
                let inner_end = inner_start + inner_block;
                let mut transfers = BTreeMap::<ShardView, Vec<LowShardId>>::new();
                let mut runs = Vec::with_capacity(output_shards.len());
                for output in &output_shards {
                    let tile = self.shards[output.index() as usize].tile;
                    let left = self.local_shard(*left_value, tile)?;
                    let left_view =
                        self.narrow_view(left, &[(left_rank - 1, inner_start, inner_end)])?;
                    let right = right_shards
                        .iter()
                        .copied()
                        .find(|shard| {
                            let extents = &self.shards[shard.index() as usize].extents;
                            let columns = extents[extents.len() - 1];
                            columns.start <= column_start && columns.physical_end >= column_end
                        })
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
                        right_view
                    } else {
                        let copy = self.push_shard(LowShard {
                            id: LowShardId(0),
                            tile,
                            tensor_type: self.shards[right.index() as usize].tensor_type.clone(),
                            extents: right_view.extents.clone(),
                            definition: ShardDefinition::ExchangeCopy(right),
                        })?;
                        transfers.entry(right_view).or_default().push(copy);
                        self.full_view(copy)
                    };
                    let output_view =
                        self.narrow_view(*output, &[(output_rank - 1, column_start, column_end)])?;
                    runs.push((
                        tile,
                        KernelRun {
                            provenance: WorkProvenance {
                                operation: operation.source,
                                value: Some(*output_value),
                                reason: WorkReason::OperatorKernel,
                            },
                            kernel: TileKernel::Planned(if inner_start == 0 {
                                initialize.clone()
                            } else {
                                accumulate.clone()
                            }),
                            inputs: vec![
                                KernelOperand {
                                    views: vec![left_view],
                                },
                                KernelOperand {
                                    views: vec![resident_right],
                                },
                            ],
                            output: output_view,
                            requirements: KernelRequirements::Operator(requirements.clone()),
                        },
                    ));
                }
                self.append_phase(
                    transfers,
                    WorkProvenance {
                        operation: operation.source,
                        value: Some(*right_value),
                        reason: WorkReason::OperatorInput { input: 1 },
                    },
                    tiles,
                )?;
                for (tile, run) in runs {
                    tiles[usize::from(tile)].work.push(TileWork::Kernel(run));
                }
            }
        }
        Ok(())
    }

    fn append_phase(
        &mut self,
        transfers: BTreeMap<ShardView, Vec<LowShardId>>,
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
        let body = self.lower_region(&repeat.body.operations)?;
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
            tiles[usize::from(tile)]
                .work
                .push(TileWork::Repeat(RepeatRun {
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
                }));
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
    let mut stride = u32::from(layout.tiling.replicas);
    let axes = layout
        .tiling
        .axes
        .iter()
        .map(|tiling| {
            let result = (tiling.axis.resolve(rank)?, tiling, stride);
            stride = stride
                .checked_mul(u32::from(tiling.partitions))
                .ok_or(LowLoweringError::IdOverflow)?;
            Ok(result)
        })
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
        AxisTiling, ComputeGraph, ElementOrder, Layout, MemoryClass, Padding, PipelineConfig,
        Precision, TensorAxis, TensorFormat, TensorTiling, ToyCostModel, lower,
    };

    const CASES: usize = 32;

    fn format(tiles: u16) -> TensorFormat {
        TensorFormat {
            precision: Precision::F16,
            layout: Layout::row_sharded(tiles),
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
                .with_input(left, format(tiles))
                .with_input(right, format(tiles));
            let mid = lower(&graph, &config, &ToyCostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();

            assert_eq!(low.tiles.len(), usize::from(tiles), "case {case}");
            for tile in &low.tiles {
                for work in &tile.work {
                    if let TileWork::Kernel(run) = work {
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
                assert!(low.tiles.iter().all(|tile| contains_phase(tile, phase.id)));
                for transfer in &phase.transfers {
                    assert!(transfer.destinations.iter().all(|destination| matches!(
                        low.shards[destination.index() as usize].definition,
                        ShardDefinition::ExchangeCopy(source) if source == transfer.source.shard
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
            let mid = lower(&graph, &config, &ToyCostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();

            assert!(
                low.exchange_phases.iter().any(|phase| matches!(
                    phase.provenance.reason,
                    WorkReason::OperatorInput { input: 0 }
                )),
                "case {case}"
            );
            for tile in &low.tiles {
                let add = tile
                    .work
                    .iter()
                    .find_map(|work| match work {
                        TileWork::Kernel(
                            run @ KernelRun {
                                kernel: TileKernel::Planned(TileKernelSpec::Add),
                                ..
                            },
                        ) => Some(run),
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
                .with_input(left, format(tiles))
                .with_input(right, format(tiles));
            let mid = lower(&graph, &config, &ToyCostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();

            assert!(low.exchange_phases.iter().all(|phase| {
                phase.provenance.operation.is_some() && phase.provenance.value.is_some()
            }));

            for tile in &low.tiles {
                let gemms = tile
                    .work
                    .iter()
                    .filter_map(|work| match work {
                        TileWork::Kernel(
                            run @ KernelRun {
                                kernel: TileKernel::Planned(TileKernelSpec::Gemm { .. }),
                                ..
                            },
                        ) => Some(run),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    gemms.len(),
                    (inner_blocks * column_blocks) as usize,
                    "case {case}"
                );
                for (index, run) in gemms.into_iter().enumerate() {
                    assert_eq!(run.provenance.reason, WorkReason::OperatorKernel);
                    assert!(run.provenance.operation.is_some());
                    assert!(run.provenance.value.is_some());
                    let TileKernel::Planned(TileKernelSpec::Gemm { mode, .. }) = run.kernel else {
                        unreachable!()
                    };
                    let inner_index = index as u32 % inner_blocks;
                    assert_eq!(
                        mode,
                        if inner_index == 0 {
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
                    assert_eq!(output_columns.physical_end - output_columns.start, 64);
                }
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
            let mid = lower(&graph, &config, &ToyCostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();

            for tile in &low.tiles {
                let repeats = tile
                    .work
                    .iter()
                    .filter_map(|work| match work {
                        TileWork::Repeat(repeat) => Some(repeat),
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
                    repeats[0]
                        .body
                        .work
                        .iter()
                        .any(|work| matches!(work, TileWork::Kernel(_)))
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
            let mid = lower(&graph, &config, &ToyCostModel).unwrap();
            assert_eq!(
                lower_to_tiles(&mid, &config),
                Err(LowLoweringError::RepeatRequiresInPlace(0)),
                "case {case}"
            );
        }
    }

    fn contains_phase(list: &TileWorkList, phase: ExchangePhaseId) -> bool {
        list.work.iter().any(|work| match work {
            TileWork::Exchange(candidate) => *candidate == phase,
            TileWork::Repeat(repeat) => contains_phase(&repeat.body, phase),
            TileWork::Kernel(_) => false,
        })
    }
}
