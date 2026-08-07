//! Final lowering from logical per-tile work to address-resolved programs.

use crate::{
    ExchangePhaseId, ExchangeStep, KernelBuildPlan, LowProgram, LowShardId, PhysicalExchangePhase,
    Placement, RepeatPointer, RepeatRun, RepeatStep, StepProfile, TileAddress, TileProgram,
    TileStep, TileWork, TileWorkList, materialize_kernel_run_with_addresses,
};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TilePrograms {
    pub programs: Vec<TileProgram>,
    pub exchange_code_end: u32,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TileLoweringError {
    #[error(transparent)]
    Kernel(#[from] crate::KernelMaterializationError),
    #[error("tile schedule refers to an unknown exchange phase")]
    UnknownExchange,
    #[error("exchange phase has no row for tile {0}")]
    MissingExchangeRow(u16),
    #[error("repeat contains an iterated exchange source; per-iteration row tables are required")]
    IteratedExchange,
    #[error("nested finalized repeats are not yet supported")]
    NestedRepeat,
    #[error("tile-program address arithmetic overflowed")]
    Overflow,
    #[error("repeat iterated input has no placed first block")]
    InvalidRepeat,
    #[error("execution has {execution} tiles, fewer than the {scheduled} scheduled tiles")]
    MissingExecutionTiles { scheduled: u16, execution: u16 },
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

pub fn lower_to_tile_programs(
    program: &LowProgram,
    placement: &Placement,
    exchanges: &[PhysicalExchangePhase],
    kernels: &KernelBuildPlan,
    exchange_code_base: u32,
) -> Result<TilePrograms, TileLoweringError> {
    lower_to_tile_programs_with_fill(
        program,
        placement,
        exchanges,
        kernels,
        exchange_code_base,
        program.tile_count,
    )
}

pub fn lower_to_tile_programs_with_fill(
    program: &LowProgram,
    placement: &Placement,
    exchanges: &[PhysicalExchangePhase],
    kernels: &KernelBuildPlan,
    exchange_code_base: u32,
    execution_tile_count: u16,
) -> Result<TilePrograms, TileLoweringError> {
    if execution_tile_count < program.tile_count {
        return Err(TileLoweringError::MissingExecutionTiles {
            scheduled: program.tile_count,
            execution: execution_tile_count,
        });
    }
    let mut cursor = align_up(exchange_code_base, 4)?;
    let mut phase_addresses = BTreeMap::<ExchangePhaseId, u32>::new();
    for phase in exchanges {
        phase_addresses.insert(phase.id, cursor);
        let maximum_words = phase
            .rows
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0)
            .max(ipu_exchange::PLAN_WORDS);
        cursor = cursor
            .checked_add(
                u32::try_from(maximum_words)
                    .map_err(|_| TileLoweringError::Overflow)?
                    .checked_mul(4)
                    .ok_or(TileLoweringError::Overflow)?,
            )
            .ok_or(TileLoweringError::Overflow)?;
    }
    let phases = exchanges
        .iter()
        .map(|phase| (phase.id, phase))
        .collect::<BTreeMap<_, _>>();
    let mut programs = program
        .tiles
        .iter()
        .map(|tile| {
            Ok(TileProgram {
                tile: tile.tile,
                steps: lower_work(
                    program,
                    tile,
                    placement,
                    kernels,
                    &phases,
                    &phase_addresses,
                    &BTreeMap::new(),
                    false,
                )?,
            })
        })
        .collect::<Result<Vec<_>, TileLoweringError>>()?;
    for tile in program.tile_count..execution_tile_count {
        programs.push(TileProgram {
            tile,
            steps: exchanges
                .iter()
                .map(|phase| {
                    Ok(TileStep::Exchange(ExchangeStep {
                        address: *phase_addresses
                            .get(&phase.id)
                            .ok_or(TileLoweringError::UnknownExchange)?,
                        row: crate::inactive_exchange_row(),
                        profile: StepProfile::default(),
                    }))
                })
                .collect::<Result<Vec<_>, TileLoweringError>>()?,
        });
    }
    Ok(TilePrograms {
        programs,
        exchange_code_end: cursor,
    })
}

#[allow(clippy::too_many_arguments)]
fn lower_work(
    program: &LowProgram,
    tile: &TileWorkList,
    placement: &Placement,
    kernels: &KernelBuildPlan,
    phases: &BTreeMap<ExchangePhaseId, &PhysicalExchangePhase>,
    phase_addresses: &BTreeMap<ExchangePhaseId, u32>,
    overrides: &BTreeMap<LowShardId, TileAddress>,
    inside_repeat: bool,
) -> Result<Vec<TileStep>, TileLoweringError> {
    let mut steps = Vec::new();
    for work in &tile.work {
        let step = match work {
            TileWork::Exchange(id) => {
                let phase = phases.get(id).ok_or(TileLoweringError::UnknownExchange)?;
                if inside_repeat
                    && program.exchange_phases[id.index() as usize]
                        .transfers
                        .iter()
                        .any(|transfer| overrides.contains_key(&transfer.source.shard))
                {
                    return Err(TileLoweringError::IteratedExchange);
                }
                TileStep::Exchange(ExchangeStep {
                    address: *phase_addresses
                        .get(id)
                        .ok_or(TileLoweringError::UnknownExchange)?,
                    row: phase
                        .rows
                        .get(usize::from(tile.tile))
                        .cloned()
                        .ok_or(TileLoweringError::MissingExchangeRow(tile.tile))?,
                    profile: StepProfile::default(),
                })
            }
            TileWork::LocalCopy(copy) => {
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
                let (symbol, arguments) = if copy.bytes != 0 && copy.bytes.is_multiple_of(8) {
                    let words = copy.bytes / 8;
                    (crate::COPY_U64_SYMBOL, vec![words / 6, words % 6])
                } else if copy.bytes != 0 && copy.bytes.is_multiple_of(4) {
                    (crate::COPY_U32_SYMBOL, vec![copy.bytes / 4])
                } else {
                    return Err(invalid());
                };
                TileStep::Compute(crate::ComputeStep {
                    symbol: symbol.into(),
                    output_address: TileAddress::Absolute(destination),
                    input_addresses: vec![TileAddress::Absolute(source)],
                    arguments,
                    profile: StepProfile::default(),
                })
            }
            TileWork::Kernel(run) => TileStep::Compute(materialize_kernel_run_with_addresses(
                run,
                &program.shards,
                &placement.shard_addresses,
                kernels,
                overrides,
            )?),
            TileWork::Repeat(repeat) => {
                if inside_repeat {
                    return Err(TileLoweringError::NestedRepeat);
                }
                TileStep::Repeat(lower_repeat(
                    program,
                    repeat,
                    placement,
                    kernels,
                    phases,
                    phase_addresses,
                )?)
            }
        };
        steps.push(step);
    }
    Ok(steps)
}

fn lower_repeat(
    program: &LowProgram,
    repeat: &RepeatRun,
    placement: &Placement,
    kernels: &KernelBuildPlan,
    phases: &BTreeMap<ExchangePhaseId, &PhysicalExchangePhase>,
    phase_addresses: &BTreeMap<ExchangePhaseId, u32>,
) -> Result<RepeatStep, TileLoweringError> {
    let mut overrides = BTreeMap::new();
    let mut pointers = Vec::with_capacity(repeat.iterated.len());
    for (index, iterated) in repeat.iterated.iter().enumerate() {
        let first = *iterated
            .inputs
            .first()
            .ok_or(TileLoweringError::InvalidRepeat)?;
        let initial_address = placement
            .shard_addresses
            .get(&first)
            .copied()
            .ok_or(TileLoweringError::InvalidRepeat)?;
        let index = u16::try_from(index).map_err(|_| TileLoweringError::Overflow)?;
        overrides.insert(
            iterated.argument,
            TileAddress::RepeatPointer { index, offset: 0 },
        );
        pointers.push(RepeatPointer {
            initial_address,
            stride_bytes: iterated.stride_bytes,
        });
    }
    Ok(RepeatStep {
        count: repeat.count,
        iterated_pointers: pointers,
        body: lower_work(
            program,
            &repeat.body,
            placement,
            kernels,
            phases,
            phase_addresses,
            &overrides,
            true,
        )?,
        profile: StepProfile::default(),
    })
}

fn align_up(value: u32, alignment: u32) -> Result<u32, TileLoweringError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(TileLoweringError::Overflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ComputeGraph, Layout, PipelineConfig, Precision, TensorFormat, ToyCostModel, lower,
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
            let mid = lower(&graph, &config, &ToyCostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();
            let placement = place(&low).unwrap();
            let kernels = KernelBuildPlan::from_program(&low).unwrap();
            let exchanges = lower_exchanges(&low, &placement, &Topology::c600()).unwrap();
            let filler_tiles = random.u16(1..=4);
            let finalized = lower_to_tile_programs_with_fill(
                &low,
                &placement,
                &exchanges,
                &kernels,
                0x4d000,
                tiles + filler_tiles,
            )
            .unwrap();
            assert_eq!(finalized.programs.len(), usize::from(tiles + filler_tiles));
            for program in &finalized.programs[..usize::from(tiles)] {
                assert!(
                    program
                        .steps
                        .iter()
                        .any(|step| matches!(step, TileStep::Compute(_)))
                );
                for step in &program.steps {
                    if let TileStep::Exchange(exchange) = step {
                        assert_eq!(exchange.row.last(), Some(&RETURN_M10_INSTRUCTION));
                    }
                }
            }
            for program in &finalized.programs[usize::from(tiles)..] {
                assert_eq!(program.steps.len(), exchanges.len());
                assert!(program.steps.iter().all(|step| matches!(
                    step,
                    TileStep::Exchange(exchange)
                        if exchange.row[0] == ipu_exchange::SANS_INACTIVE_INSTRUCTION
                )));
            }
        }
    }
}
