//! Logical per-tile schedule produced from the layout-aware mid-level IR.
//!
//! Tensor shards have tile identities and rectangular physical extents, and
//! work is ordered per tile. Exchanges still refer to logical shards rather
//! than SRAM addresses; kernel runs still name a selected kernel kind rather
//! than a linked symbol. Placement and final code generation resolve those
//! remaining choices.

use crate::graph::{GraphInputKind, OperationId};
use crate::mid::{
    KernelRequirements, Layout, LayoutError, MidGraph, MidKernel, MidOperation, MidOperationKind,
    MidRepeat, MidValueId, Precision, TensorType,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LowShardId(u32);

impl LowShardId {
    pub const fn index(self) -> u32 {
        self.0
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardExtent {
    pub axis: u16,
    pub start: u32,
    pub logical_end: u32,
    pub physical_end: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardDefinition {
    Value(MidValueId),
    ExchangeCopy(LowShardId),
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
    pub source: LowShardId,
    pub destinations: Vec<LowShardId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExchangePhase {
    pub id: ExchangePhaseId,
    pub transfers: Vec<LogicalExchange>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LowKernel {
    Kernel(MidKernel),
    Cast { from: Precision, to: Precision },
    Rearrange { from: Layout, to: Layout },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelOperand {
    /// Shards resident on the execution tile which jointly represent this
    /// operand. The first lowering intentionally uses a conservative gather.
    pub shards: Vec<LowShardId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelRun {
    pub source: Option<OperationId>,
    pub kernel: LowKernel,
    pub inputs: Vec<KernelOperand>,
    pub output: LowShardId,
    pub requirements: Option<KernelRequirements>,
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepeatRun {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LowLoweringConfig {
    pub tile_count: u16,
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
    #[error("repeat structure is inconsistent with its inputs, arguments, yields, or results")]
    InvalidRepeat,
    #[error("too many logical shards or exchange phases")]
    IdOverflow,
    #[error("invalid tensor layout: {0}")]
    Layout(#[from] LayoutError),
}

pub type LowLoweringResult<T> = Result<T, LowLoweringError>;

/// Produces a logical per-tile schedule. Its conservative residency policy
/// gathers every shard of every operand onto each tile that runs an output
/// shard. Later cost-driven scheduling can narrow transfers without changing
/// the placement/code-generation boundary.
pub fn lower_to_tiles(
    graph: &MidGraph,
    config: &LowLoweringConfig,
) -> LowLoweringResult<LowProgram> {
    if config.tile_count == 0 {
        return Err(LowLoweringError::EmptyTileGroup);
    }
    let mut state = LoweringState::new(graph, *config)?;
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
    config: LowLoweringConfig,
    shards: Vec<LowShard>,
    canonical: Vec<Vec<LowShardId>>,
    phases: Vec<ExchangePhase>,
}

impl LoweringState {
    fn new(graph: &MidGraph, config: LowLoweringConfig) -> LowLoweringResult<Self> {
        let mut state = Self {
            config,
            shards: Vec::new(),
            canonical: vec![Vec::new(); graph.values.len()],
            phases: Vec::new(),
        };
        for value in &graph.values {
            if value.tensor_type.format.layout.tiling.tile_count != config.tile_count {
                return Err(LowLoweringError::TileCountMismatch {
                    value: value.id,
                    declared: value.tensor_type.format.layout.tiling.tile_count,
                    scheduled: config.tile_count,
                });
            }
            let extents = shard_extents(&value.tensor_type)?;
            let mut value_shards = Vec::with_capacity(usize::from(config.tile_count));
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
        let mut tiles = (0..self.config.tile_count)
            .map(|tile| TileWorkList {
                tile,
                work: Vec::new(),
            })
            .collect::<Vec<_>>();
        let mut resident = BTreeMap::<(LowShardId, u16), LowShardId>::new();
        for operation in operations {
            match &operation.kind {
                MidOperationKind::Repeat(repeat) => {
                    self.lower_repeat(operation, repeat, &mut tiles)?;
                }
                kind => self.lower_operation(operation, kind, &mut resident, &mut tiles)?,
            }
        }
        Ok(tiles)
    }

    fn lower_operation(
        &mut self,
        operation: &MidOperation,
        kind: &MidOperationKind,
        resident: &mut BTreeMap<(LowShardId, u16), LowShardId>,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [result] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        let output_shards = self.value_shards(*result)?.to_vec();
        let input_shards = operation
            .inputs
            .iter()
            .map(|value| Ok((*value, self.value_shards(*value)?.to_vec())))
            .collect::<LowLoweringResult<Vec<_>>>()?;
        let mut transfers = BTreeMap::<LowShardId, Vec<LowShardId>>::new();
        let mut runs = Vec::with_capacity(output_shards.len());
        for output in output_shards {
            let tile = self.shards[output.index() as usize].tile;
            let mut operands = Vec::with_capacity(input_shards.len());
            for (_, sources) in &input_shards {
                let mut local = Vec::with_capacity(sources.len());
                for source in sources {
                    local.push(self.ensure_resident(*source, tile, resident, &mut transfers)?);
                }
                operands.push(KernelOperand { shards: local });
            }
            runs.push((
                tile,
                KernelRun {
                    source: operation.source,
                    kernel: match kind {
                        MidOperationKind::Kernel(kernel) => LowKernel::Kernel(*kernel),
                        MidOperationKind::CastPrecision { from, to } => LowKernel::Cast {
                            from: *from,
                            to: *to,
                        },
                        MidOperationKind::Rearrange { from, to } => LowKernel::Rearrange {
                            from: from.clone(),
                            to: to.clone(),
                        },
                        MidOperationKind::Repeat(_) => unreachable!(),
                    },
                    inputs: operands,
                    output,
                    requirements: operation.kernel_requirements.clone(),
                },
            ));
        }
        if !transfers.is_empty() {
            let id = ExchangePhaseId(
                u32::try_from(self.phases.len()).map_err(|_| LowLoweringError::IdOverflow)?,
            );
            self.phases.push(ExchangePhase {
                id,
                transfers: transfers
                    .into_iter()
                    .map(|(source, destinations)| LogicalExchange {
                        source,
                        destinations,
                    })
                    .collect(),
            });
            for tile in &mut *tiles {
                tile.work.push(TileWork::Exchange(id));
            }
        }
        for (tile, run) in runs {
            tiles[usize::from(tile)].work.push(TileWork::Kernel(run));
        }
        Ok(())
    }

    fn ensure_resident(
        &mut self,
        source: LowShardId,
        tile: u16,
        resident: &mut BTreeMap<(LowShardId, u16), LowShardId>,
        transfers: &mut BTreeMap<LowShardId, Vec<LowShardId>>,
    ) -> LowLoweringResult<LowShardId> {
        let source_shard = &self.shards[source.index() as usize];
        if source_shard.tile == tile {
            return Ok(source);
        }
        if let Some(copy) = resident.get(&(source, tile)) {
            return Ok(*copy);
        }
        let copy = self.push_shard(LowShard {
            id: LowShardId(0),
            tile,
            tensor_type: source_shard.tensor_type.clone(),
            extents: source_shard.extents.clone(),
            definition: ShardDefinition::ExchangeCopy(source),
        })?;
        resident.insert((source, tile), copy);
        transfers.entry(source).or_default().push(copy);
        Ok(copy)
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
        let body = self.lower_region(&repeat.body.operations)?;
        for tile in 0..self.config.tile_count {
            let carried = (0..repeat.carried_inputs)
                .map(|index| {
                    Ok(RepeatCarried {
                        initial: self.local_shard(operation.inputs[index], tile)?,
                        argument: self.local_shard(repeat.body.arguments[index], tile)?,
                        yielded: self.local_shard(repeat.body.yields[index], tile)?,
                        result: self.local_shard(operation.results[index], tile)?,
                    })
                })
                .collect::<LowLoweringResult<_>>()?;
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
                    Ok(RepeatIterated {
                        inputs: values
                            .iter()
                            .map(|value| self.local_shard(*value, tile))
                            .collect::<LowLoweringResult<_>>()?,
                        argument: self
                            .local_shard(repeat.body.arguments[expected_inputs + index], tile)?,
                    })
                })
                .collect::<LowLoweringResult<_>>()?;
            tiles[usize::from(tile)]
                .work
                .push(TileWork::Repeat(RepeatRun {
                    count: repeat.count,
                    carried,
                    invariants,
                    iterated,
                    body: Box::new(body[usize::from(tile)].clone()),
                }));
        }
        Ok(())
    }
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
        AxisTiling, ComputeGraph, ElementOrder, Layout, LoweringConfig, MemoryClass, Padding,
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
            let config = LoweringConfig::new(tiles)
                .with_input(left, format(tiles))
                .with_input(right, format(tiles));
            let mid = lower(&graph, &config, &ToyCostModel).unwrap();
            let low = lower_to_tiles(&mid, &LowLoweringConfig { tile_count: tiles }).unwrap();

            assert_eq!(low.tiles.len(), usize::from(tiles), "case {case}");
            for tile in &low.tiles {
                for work in &tile.work {
                    if let TileWork::Kernel(run) = work {
                        assert_eq!(low.shards[run.output.index() as usize].tile, tile.tile);
                        assert!(
                            run.inputs
                                .iter()
                                .flat_map(|operand| &operand.shards)
                                .all(|shard| {
                                    low.shards[shard.index() as usize].tile == tile.tile
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
                        ShardDefinition::ExchangeCopy(source) if source == transfer.source
                    )));
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
            let mut config = LoweringConfig::new(tiles).with_input(carried, format(tiles));
            for parameter in parameters {
                config.inputs.insert(parameter, format(tiles));
            }
            let mid = lower(&graph, &config, &ToyCostModel).unwrap();
            let low = lower_to_tiles(&mid, &LowLoweringConfig { tile_count: tiles }).unwrap();

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

    fn contains_phase(list: &TileWorkList, phase: ExchangePhaseId) -> bool {
        list.work.iter().any(|work| match work {
            TileWork::Exchange(candidate) => *candidate == phase,
            TileWork::Repeat(repeat) => contains_phase(&repeat.body, phase),
            TileWork::Kernel(_) => false,
        })
    }
}
