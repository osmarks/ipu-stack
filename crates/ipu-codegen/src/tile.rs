//! Final lowering from logical per-tile work to address-resolved programs.

use crate::{
    ExchangePatch, ExchangePhaseId, ExchangeStep, KernelBuildPlan, LowProgram, LowShardId,
    PhysicalExchangePhase, PlacedExchangeRow, Placement, RepeatPointer, RepeatRun, RepeatStep,
    StepProfile, TileAddress, TileProgram, TileStep, TileWorkList, TileWorkRef,
    materialize_kernel_run,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RepeatExchangeStrategy {
    #[default]
    PatchInPlace,
    SeparateRows,
}

/// Address-resolution context which finalizes one tile on demand.
///
/// Package generation uses this to emit and discard each logical tile program
/// instead of retaining every tile's expanded instruction steps at once.
pub struct TileProgramLowering<'a> {
    program: &'a LowProgram,
    placement: &'a Placement,
    kernels: &'a KernelBuildPlan,
    exchanges: &'a [PhysicalExchangePhase],
    phases: BTreeMap<ExchangePhaseId, &'a PhysicalExchangePhase>,
    exchange_code_base: u32,
    exchange_code_end: u32,
    execution_tile_count: u16,
    repeat_exchanges: RepeatExchangeStrategy,
    repeat_phase_counts: BTreeMap<ExchangePhaseId, u32>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TileLoweringError {
    #[error(transparent)]
    Kernel(#[from] crate::KernelMaterializationError),
    #[error("tile schedule refers to an unknown exchange phase")]
    UnknownExchange,
    #[error("exchange phase has no row for tile {0}")]
    MissingExchangeRow(u16),
    #[error("nested finalized repeats are not yet supported")]
    NestedRepeat,
    #[error("tile-program address arithmetic overflowed")]
    Overflow,
    #[error("repeat iterated input has no placed first block")]
    InvalidRepeat,
    #[error("execution has {execution} tiles, fewer than the {scheduled} scheduled tiles")]
    MissingExecutionTiles { scheduled: u16, execution: u16 },
    #[error(
        "exchange rows occupy 0x{start:x}..0x{end:x}, beyond executable tile SRAM at 0x{limit:x}"
    )]
    ExchangeCodeMemory { start: u32, end: u32, limit: u32 },
    #[error(
        "tile {tile} local copy {source_shard:?}+{source_offset} -> {destination_shard:?}+{destination_offset} has invalid addresses or byte count {bytes}"
    )]
    InvalidLocalCopy {
        tile: u16,
        source_shard: LowShardId,
        source_offset: u32,
        destination_shard: LowShardId,
        destination_offset: u32,
        bytes: u32,
    },
}

struct PlacedExchange {
    active: bool,
    program: PlacedExchangeRow,
    patches: Vec<ExchangePatch>,
    iteration_programs: Vec<PlacedExchangeRow>,
}

impl<'a> TileProgramLowering<'a> {
    pub fn new(
        program: &'a LowProgram,
        placement: &'a Placement,
        exchanges: &'a [PhysicalExchangePhase],
        kernels: &'a KernelBuildPlan,
        exchange_code_base: u32,
        execution_tile_count: u16,
        repeat_exchanges: RepeatExchangeStrategy,
    ) -> Result<Self, TileLoweringError> {
        if execution_tile_count < program.tile_count {
            return Err(TileLoweringError::MissingExecutionTiles {
                scheduled: program.tile_count,
                execution: execution_tile_count,
            });
        }
        let repeat_phase_counts = repeat_phase_counts(program)?;
        let cursor = exchange_code_base
            .checked_add(compact_exchange_table_bytes(
                program,
                exchanges,
                execution_tile_count,
                repeat_exchanges,
            )?)
            .ok_or(TileLoweringError::Overflow)?;
        let executable_memory_end = ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT;
        if cursor > executable_memory_end {
            return Err(TileLoweringError::ExchangeCodeMemory {
                start: exchange_code_base,
                end: cursor,
                limit: executable_memory_end,
            });
        }
        let phases = exchanges
            .iter()
            .map(|phase| (phase.id, phase))
            .collect::<BTreeMap<_, _>>();
        Ok(Self {
            program,
            placement,
            kernels,
            exchanges,
            phases,
            exchange_code_base,
            exchange_code_end: cursor,
            execution_tile_count,
            repeat_exchanges,
            repeat_phase_counts,
        })
    }

    pub const fn exchange_code_end(&self) -> u32 {
        self.exchange_code_end
    }

    pub fn lower_tile(&self, tile: u16) -> Result<TileProgram, TileLoweringError> {
        if tile >= self.execution_tile_count {
            return Err(TileLoweringError::MissingExecutionTiles {
                scheduled: tile.saturating_add(1),
                execution: self.execution_tile_count,
            });
        }
        if tile < self.program.tile_count {
            let work = &self.program.tiles[usize::from(tile)];
            let (rows, _) = layout_exchange_rows(
                self.exchanges,
                tile,
                self.program.tile_count,
                self.exchange_code_base,
                &self.repeat_phase_counts,
                self.repeat_exchanges,
            )?;
            return Ok(TileProgram {
                tile,
                steps: lower_work(
                    self.program,
                    work,
                    self.placement,
                    self.kernels,
                    &self.phases,
                    &rows,
                    &BTreeMap::new(),
                    false,
                    None,
                    self.repeat_exchanges,
                )?,
            });
        }
        let (rows, _) = layout_exchange_rows(
            self.exchanges,
            tile,
            self.program.tile_count,
            self.exchange_code_base,
            &self.repeat_phase_counts,
            self.repeat_exchanges,
        )?;
        Ok(TileProgram {
            tile,
            steps: lower_inactive_work(
                self.program,
                &self.program.tiles[0],
                &rows,
                None,
                self.repeat_exchanges,
            )?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_work(
    program: &LowProgram,
    tile: &TileWorkList,
    placement: &Placement,
    kernels: &KernelBuildPlan,
    phases: &BTreeMap<ExchangePhaseId, &PhysicalExchangePhase>,
    exchange_rows: &BTreeMap<ExchangePhaseId, PlacedExchange>,
    overrides: &BTreeMap<LowShardId, TileAddress>,
    inside_repeat: bool,
    repeat_iteration: Option<usize>,
    repeat_exchanges: RepeatExchangeStrategy,
) -> Result<Vec<TileStep>, TileLoweringError> {
    let mut steps = Vec::new();
    for work in program.work(tile) {
        let step = match work {
            TileWorkRef::Exchange(id) => {
                phases.get(&id).ok_or(TileLoweringError::UnknownExchange)?;
                let placed = exchange_rows
                    .get(&id)
                    .ok_or(TileLoweringError::UnknownExchange)?;
                TileStep::Exchange(ExchangeStep {
                    active: placed.active,
                    program: repeat_iteration
                        .and_then(|iteration| placed.iteration_programs.get(iteration))
                        .unwrap_or(&placed.program)
                        .clone(),
                    patches: (inside_repeat
                        && repeat_exchanges == RepeatExchangeStrategy::PatchInPlace)
                        .then(|| placed.patches.clone())
                        .unwrap_or_default(),
                    profile: StepProfile::default(),
                })
            }
            TileWorkRef::LocalCopy(copy) => {
                let invalid = || TileLoweringError::InvalidLocalCopy {
                    tile: tile.tile,
                    source_shard: copy.source,
                    source_offset: copy.source_offset,
                    destination_shard: copy.destination,
                    destination_offset: copy.destination_offset,
                    bytes: copy.bytes,
                };
                let source = placement
                    .shard_addresses
                    .get(&copy.source)
                    .and_then(|address| address.checked_add(copy.source_offset))
                    .ok_or_else(&invalid)?;
                let destination = placement
                    .shard_addresses
                    .get(&copy.destination)
                    .and_then(|address| address.checked_add(copy.destination_offset))
                    .ok_or_else(&invalid)?;
                let (symbol, arguments) = local_copy_call(copy.bytes).ok_or_else(invalid)?;
                TileStep::Compute(crate::ComputeStep {
                    symbol: symbol.into(),
                    output_address: TileAddress::Absolute(destination),
                    input_addresses: vec![TileAddress::Absolute(source)],
                    arguments,
                    profile: StepProfile::default(),
                })
            }
            TileWorkRef::Kernel(run) => TileStep::Compute(materialize_kernel_run(
                run,
                &program.shards,
                &placement.shard_addresses,
                kernels,
                overrides,
            )?),
            TileWorkRef::Repeat(repeat) => {
                if inside_repeat {
                    return Err(TileLoweringError::NestedRepeat);
                }
                if repeat_exchanges == RepeatExchangeStrategy::SeparateRows {
                    steps.extend(lower_repeat_separate_rows(
                        program,
                        repeat,
                        placement,
                        kernels,
                        phases,
                        exchange_rows,
                        repeat_exchanges,
                    )?);
                    continue;
                }
                TileStep::Repeat(lower_repeat(
                    program,
                    repeat,
                    placement,
                    kernels,
                    phases,
                    exchange_rows,
                )?)
            }
        };
        steps.push(step);
    }
    Ok(steps)
}

fn local_copy_call(bytes: u32) -> Option<(&'static str, Vec<u32>)> {
    if bytes >= 6 * 8 && bytes.is_multiple_of(8) {
        let words = bytes / 8;
        Some((crate::COPY_U64_SYMBOL, vec![words / 6, words % 6]))
    } else if bytes != 0 && bytes.is_multiple_of(4) {
        Some((crate::COPY_U32_SYMBOL, vec![bytes / 4]))
    } else {
        None
    }
}

fn lower_repeat(
    program: &LowProgram,
    repeat: &RepeatRun,
    placement: &Placement,
    kernels: &KernelBuildPlan,
    phases: &BTreeMap<ExchangePhaseId, &PhysicalExchangePhase>,
    exchange_rows: &BTreeMap<ExchangePhaseId, PlacedExchange>,
) -> Result<RepeatStep, TileLoweringError> {
    let mut overrides = BTreeMap::new();
    let mut pointers = Vec::with_capacity(repeat.iterated.len());
    for (index, iterated) in repeat.iterated.iter().enumerate() {
        let initial_address = iterated
            .inputs
            .first()
            .and_then(|input| placement.shard_addresses.get(input))
            .copied()
            .ok_or(TileLoweringError::InvalidRepeat)?;
        for (iteration, input) in iterated.inputs.iter().enumerate() {
            let expected = initial_address
                .checked_add(
                    iterated
                        .stride_bytes
                        .checked_mul(
                            u32::try_from(iteration).map_err(|_| TileLoweringError::Overflow)?,
                        )
                        .ok_or(TileLoweringError::Overflow)?,
                )
                .ok_or(TileLoweringError::Overflow)?;
            if placement.shard_addresses.get(input).copied() != Some(expected) {
                return Err(TileLoweringError::InvalidRepeat);
            }
        }
        let index = u16::try_from(index).map_err(|_| TileLoweringError::Overflow)?;
        let address = TileAddress::RepeatPointer { index, offset: 0 };
        if !iterated.stride_bytes.is_multiple_of(4) {
            return Err(TileLoweringError::InvalidRepeat);
        }
        overrides.insert(iterated.argument, address);
        pointers.push(RepeatPointer {
            initial_address,
            stride_bytes: iterated.stride_bytes,
        });
    }
    let body = lower_work(
        program,
        &repeat.body,
        placement,
        kernels,
        phases,
        exchange_rows,
        &overrides,
        true,
        None,
        RepeatExchangeStrategy::PatchInPlace,
    )?;
    for step in &body {
        let TileStep::Exchange(exchange) = step else {
            continue;
        };
        if exchange
            .patches
            .iter()
            .any(|patch| patch.values.words.len() != repeat.count as usize)
        {
            return Err(TileLoweringError::InvalidRepeat);
        }
    }
    Ok(RepeatStep {
        count: repeat.count,
        iterated_pointers: pointers,
        body,
        profile: StepProfile::default(),
    })
}

fn lower_repeat_separate_rows(
    program: &LowProgram,
    repeat: &RepeatRun,
    placement: &Placement,
    kernels: &KernelBuildPlan,
    phases: &BTreeMap<ExchangePhaseId, &PhysicalExchangePhase>,
    exchange_rows: &BTreeMap<ExchangePhaseId, PlacedExchange>,
    repeat_exchanges: RepeatExchangeStrategy,
) -> Result<Vec<TileStep>, TileLoweringError> {
    let mut steps = Vec::new();
    for iteration in 0..repeat.count {
        let mut overrides = BTreeMap::new();
        for iterated in &repeat.iterated {
            let initial_address = iterated
                .inputs
                .first()
                .and_then(|input| placement.shard_addresses.get(input))
                .copied()
                .ok_or(TileLoweringError::InvalidRepeat)?;
            let address = initial_address
                .checked_add(
                    iterated
                        .stride_bytes
                        .checked_mul(iteration)
                        .ok_or(TileLoweringError::Overflow)?,
                )
                .ok_or(TileLoweringError::Overflow)?;
            if placement
                .shard_addresses
                .get(&iterated.inputs[iteration as usize])
                .copied()
                != Some(address)
            {
                return Err(TileLoweringError::InvalidRepeat);
            }
            overrides.insert(iterated.argument, TileAddress::Absolute(address));
        }
        steps.extend(lower_work(
            program,
            &repeat.body,
            placement,
            kernels,
            phases,
            exchange_rows,
            &overrides,
            true,
            Some(iteration as usize),
            repeat_exchanges,
        )?);
    }
    Ok(steps)
}

fn lower_inactive_work(
    program: &LowProgram,
    work: &TileWorkList,
    exchange_rows: &BTreeMap<ExchangePhaseId, PlacedExchange>,
    repeat_iteration: Option<usize>,
    repeat_exchanges: RepeatExchangeStrategy,
) -> Result<Vec<TileStep>, TileLoweringError> {
    let mut steps = Vec::new();
    for work in program.work(work) {
        match work {
            TileWorkRef::Exchange(id) => steps.push(TileStep::Exchange(ExchangeStep {
                active: exchange_rows[&id].active,
                program: repeat_iteration
                    .and_then(|iteration| exchange_rows[&id].iteration_programs.get(iteration))
                    .unwrap_or(&exchange_rows[&id].program)
                    .clone(),
                patches: Vec::new(),
                profile: StepProfile::default(),
            })),
            TileWorkRef::Repeat(repeat) => {
                if repeat_exchanges == RepeatExchangeStrategy::SeparateRows {
                    for iteration in 0..repeat.count as usize {
                        steps.extend(lower_inactive_work(
                            program,
                            &repeat.body,
                            exchange_rows,
                            Some(iteration),
                            repeat_exchanges,
                        )?);
                    }
                } else {
                    steps.push(TileStep::Repeat(RepeatStep {
                        count: repeat.count,
                        iterated_pointers: Vec::new(),
                        body: lower_inactive_work(
                            program,
                            &repeat.body,
                            exchange_rows,
                            None,
                            repeat_exchanges,
                        )?,
                        profile: StepProfile::default(),
                    }));
                }
            }
            TileWorkRef::Kernel(_) | TileWorkRef::LocalCopy(_) => {}
        }
    }
    Ok(steps)
}

fn align_up(value: u32, alignment: u32) -> Result<u32, TileLoweringError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(TileLoweringError::Overflow)
}

pub fn compact_exchange_table_bytes(
    program: &LowProgram,
    exchanges: &[PhysicalExchangePhase],
    execution_tile_count: u16,
    strategy: RepeatExchangeStrategy,
) -> Result<u32, TileLoweringError> {
    let scheduled_tile_count = program.tile_count;
    let repeat_counts = repeat_phase_counts(program)?;
    let mut maximum = 0;
    for tile in 0..execution_tile_count {
        let mut bytes = 0u32;
        for phase in exchanges {
            let words = if tile < scheduled_tile_count {
                phase
                    .programs
                    .get(usize::from(tile))
                    .ok_or(TileLoweringError::MissingExchangeRow(tile))?
                    .len()
            } else {
                crate::inactive_exchange_program().len()
            };
            let row_bytes = u32::try_from(words)
                .map_err(|_| TileLoweringError::Overflow)?
                .checked_mul(4)
                .ok_or(TileLoweringError::Overflow)?;
            let extra_bytes = match strategy {
                RepeatExchangeStrategy::PatchInPlace if tile < scheduled_tile_count => phase
                    .repeat_patches[usize::from(tile)]
                .iter()
                .try_fold(0u32, |total, patch| {
                    total
                        .checked_add(
                            u32::try_from(patch.values.len())
                                .map_err(|_| TileLoweringError::Overflow)?
                                .checked_mul(4)
                                .ok_or(TileLoweringError::Overflow)?,
                        )
                        .ok_or(TileLoweringError::Overflow)
                })?,
                RepeatExchangeStrategy::SeparateRows => repeat_counts
                    .get(&phase.id)
                    .copied()
                    .unwrap_or(1)
                    .saturating_sub(1)
                    .checked_mul(row_bytes)
                    .ok_or(TileLoweringError::Overflow)?,
                RepeatExchangeStrategy::PatchInPlace => 0,
            };
            bytes = bytes
                .checked_add(row_bytes)
                .and_then(|bytes| bytes.checked_add(extra_bytes))
                .ok_or(TileLoweringError::Overflow)?;
        }
        maximum = maximum.max(bytes);
    }
    Ok(maximum)
}

fn layout_exchange_rows(
    exchanges: &[PhysicalExchangePhase],
    tile: u16,
    scheduled_tile_count: u16,
    base: u32,
    repeat_counts: &BTreeMap<ExchangePhaseId, u32>,
    strategy: RepeatExchangeStrategy,
) -> Result<(BTreeMap<ExchangePhaseId, PlacedExchange>, u32), TileLoweringError> {
    let mut cursor = align_up(base, 4)?;
    let mut result = BTreeMap::new();
    if exchanges.is_empty() {
        return Ok((result, cursor));
    }
    for phase in exchanges {
        let (active, base_program) = if tile < scheduled_tile_count {
            let index = usize::from(tile);
            let active = *phase
                .active
                .get(index)
                .ok_or(TileLoweringError::MissingExchangeRow(tile))?;
            phase
                .programs
                .get(index)
                .cloned()
                .map(|program| (active, program))
                .ok_or(TileLoweringError::MissingExchangeRow(tile))?
        } else {
            (false, crate::inactive_exchange_program())
        };
        let row_bytes = u32::try_from(base_program.len())
            .map_err(|_| TileLoweringError::Overflow)?
            .checked_mul(4)
            .ok_or(TileLoweringError::Overflow)?;
        let source_patches = (tile < scheduled_tile_count)
            .then(|| &phase.repeat_patches[usize::from(tile)][..])
            .unwrap_or_default();
        let iteration_count = repeat_counts.get(&phase.id).copied();
        let mut iteration_programs = Vec::new();
        let mut patches = Vec::new();
        let program = if strategy == RepeatExchangeStrategy::SeparateRows
            && let Some(count) = iteration_count
        {
            for iteration in 0..count as usize {
                let mut words = base_program.clone();
                for patch in source_patches {
                    let value = *patch
                        .values
                        .get(iteration)
                        .ok_or(TileLoweringError::InvalidRepeat)?;
                    *words
                        .get_mut(patch.word_offset as usize)
                        .ok_or(TileLoweringError::InvalidRepeat)? = value;
                }
                iteration_programs.push(PlacedExchangeRow {
                    address: cursor,
                    words,
                });
                cursor = cursor
                    .checked_add(row_bytes)
                    .ok_or(TileLoweringError::Overflow)?;
            }
            iteration_programs
                .first()
                .cloned()
                .ok_or(TileLoweringError::InvalidRepeat)?
        } else {
            let program = PlacedExchangeRow {
                address: cursor,
                words: base_program,
            };
            cursor = cursor
                .checked_add(row_bytes)
                .ok_or(TileLoweringError::Overflow)?;
            for patch in source_patches {
                let address = cursor;
                cursor = cursor
                    .checked_add(
                        u32::try_from(patch.values.len())
                            .map_err(|_| TileLoweringError::Overflow)?
                            .checked_mul(4)
                            .ok_or(TileLoweringError::Overflow)?,
                    )
                    .ok_or(TileLoweringError::Overflow)?;
                patches.push(ExchangePatch {
                    word_offset: patch.word_offset,
                    values: PlacedExchangeRow {
                        address,
                        words: patch.values.clone(),
                    },
                });
            }
            program
        };
        result.insert(
            phase.id,
            PlacedExchange {
                active,
                program,
                patches,
                iteration_programs,
            },
        );
    }
    Ok((result, cursor))
}

fn repeat_phase_counts(
    program: &LowProgram,
) -> Result<BTreeMap<ExchangePhaseId, u32>, TileLoweringError> {
    let mut counts = BTreeMap::new();
    for repeat in &program.repeat_runs {
        for work in program.work(&repeat.body) {
            let TileWorkRef::Exchange(phase) = work else {
                continue;
            };
            if counts
                .insert(phase, repeat.count)
                .is_some_and(|count| count != repeat.count)
            {
                return Err(TileLoweringError::InvalidRepeat);
            }
        }
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ComputeGraph, Ipu21CostModel, Layout, PipelineConfig, Precision, TensorFormat, lower,
        lower_exchanges, lower_to_tiles, place,
    };
    use ipu_exchange::{RETURN_M10_INSTRUCTION, Topology};

    #[test]
    fn randomized_gemms_finalize_to_address_resolved_tile_programs() {
        let mut random = fastrand::Rng::with_seed(0x7469_6c65);
        for _ in 0..32 {
            let tiles = 1_u16 << random.u32(0..=3);
            let rows = u32::from(tiles) * random.u32(1..=8);
            let columns = random.u32(1..=2) * 64;
            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", [rows, 64]).unwrap();
            let right = graph.parameter("right", [64, columns]).unwrap();
            let output = graph.gemm(left, right).unwrap();
            graph.set_outputs([output]).unwrap();
            let config = PipelineConfig::new(tiles)
                .with_active_tile_counts([tiles])
                .with_input(
                    left,
                    TensorFormat {
                        precision: Precision::F16,
                        layout: Layout::amp_left(64, tiles),
                    },
                )
                .with_input(
                    right,
                    TensorFormat {
                        precision: Precision::F16,
                        layout: Layout::amp_right(64, tiles),
                    },
                );
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();
            let placement = place(&low).unwrap();
            let kernels = KernelBuildPlan::from_program(&low).unwrap();
            let exchanges = lower_exchanges(
                &low,
                &placement,
                &Topology::c600(),
                crate::ExchangeLoweringOptions::default(),
            )
            .unwrap();
            let filler_tiles = random.u16(1..=4);
            let lowering = TileProgramLowering::new(
                &low,
                &placement,
                &exchanges,
                &kernels,
                0x4d000,
                tiles + filler_tiles,
                RepeatExchangeStrategy::PatchInPlace,
            )
            .unwrap();
            assert!(lowering.exchange_code_end() >= 0x4d000);
            for tile in 0..tiles {
                let program = lowering.lower_tile(tile).unwrap();
                assert!(
                    program
                        .steps
                        .iter()
                        .any(|step| matches!(step, TileStep::Compute(_)))
                );
                for step in &program.steps {
                    if let TileStep::Exchange(exchange) = step {
                        assert_eq!(exchange.program.words.last(), Some(&RETURN_M10_INSTRUCTION));
                    }
                }
            }
            for tile in tiles..tiles + filler_tiles {
                let program = lowering.lower_tile(tile).unwrap();
                assert_eq!(program.steps.len(), exchanges.len());
                assert!(program.steps.iter().all(|step| matches!(
                    step,
                    TileStep::Exchange(exchange)
                        if !exchange.active
                )));
            }
        }
    }

    #[test]
    fn randomized_local_copy_calls_never_launch_zero_work_workers() {
        let mut random = fastrand::Rng::with_seed(0x636f_7079);
        for _ in 0..1_000 {
            let words = random.u32(1..=4_096);
            let bytes = words * 4;
            let (symbol, arguments) = local_copy_call(bytes).unwrap();
            if symbol == crate::COPY_U64_SYMBOL {
                assert!(arguments[0] != 0);
                assert_eq!((arguments[0] * 6 + arguments[1]) * 8, bytes);
                assert!(arguments[1] < 6);
            } else {
                assert_eq!(symbol, crate::COPY_U32_SYMBOL);
                assert_eq!(arguments, [words]);
            }
        }
    }
}
