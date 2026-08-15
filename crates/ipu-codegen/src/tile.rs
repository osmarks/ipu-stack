//! Final lowering from logical per-tile work to address-resolved programs.

use crate::{
    ExchangePatch, ExchangePhaseId, ExchangeSetupPatch, ExchangeStep, KernelBuildPlan, LowProgram,
    LowShardId, PhysicalExchangePhase, PlacedExchangeRow, Placement, RepeatPointer, RepeatRun,
    RepeatStep, StepProfile, TileAddress, TileProgram, TileStep, TileWorkList, TileWorkRef,
    materialize_kernel_run,
};
use std::collections::BTreeMap;

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
    validate_exchange_placement: bool,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TileLoweringError {
    #[error(transparent)]
    Kernel(#[from] crate::KernelMaterializationError),
    #[error(transparent)]
    ExchangeDiagnostic(#[from] crate::ExchangeLoweringError),
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
        "tile {tile} phase {phase:?} exchange row at 0x{row_address:x} shares an SRAM element with transfer {transfer} {kind:?} data at 0x{data_address:x}"
    )]
    ExchangeRowDataConflict {
        tile: u16,
        phase: ExchangePhaseId,
        row_address: u32,
        transfer: u32,
        kind: crate::ExchangeActivityKind,
        data_address: u32,
    },
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
    incoming_base: u32,
    program: PlacedExchangeRow,
    setup_patch: Option<ExchangeSetupPatch>,
    repeat_patches: Vec<ExchangePatch>,
}

impl<'a> TileProgramLowering<'a> {
    pub fn new(
        program: &'a LowProgram,
        placement: &'a Placement,
        exchanges: &'a [PhysicalExchangePhase],
        kernels: &'a KernelBuildPlan,
        exchange_code_base: u32,
        execution_tile_count: u16,
        validate_exchange_placement: bool,
    ) -> Result<Self, TileLoweringError> {
        if execution_tile_count < program.tile_count {
            return Err(TileLoweringError::MissingExecutionTiles {
                scheduled: program.tile_count,
                execution: execution_tile_count,
            });
        }
        let cursor = exchange_code_base
            .checked_add(compact_exchange_table_bytes(
                exchanges,
                execution_tile_count,
                program.tile_count,
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
            validate_exchange_placement,
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
                self.validate_exchange_placement,
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
                )?,
            });
        }
        let (rows, _) = layout_exchange_rows(
            self.exchanges,
            tile,
            self.program.tile_count,
            self.exchange_code_base,
            self.validate_exchange_placement,
        )?;
        Ok(TileProgram {
            tile,
            steps: lower_inactive_work(self.program, &self.program.tiles[0], &rows)?,
        })
    }
}

fn placed_local_copy(
    program: &LowProgram,
    placement: &Placement,
    copy: &crate::LocalCopy,
) -> Result<(u32, u32), TileLoweringError> {
    let source_shard = &program.shards[copy.source.index() as usize];
    let invalid = || TileLoweringError::InvalidLocalCopy {
        tile: source_shard.tile,
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
    Ok((source, destination))
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
                    incoming_base: placed.incoming_base,
                    program: placed.program.clone(),
                    setup_patch: placed.setup_patch.clone(),
                    repeat_patches: inside_repeat
                        .then(|| placed.repeat_patches.clone())
                        .unwrap_or_default(),
                    profile: StepProfile::default(),
                })
            }
            TileWorkRef::LocalCopy(copy) => {
                let (source, destination) = placed_local_copy(program, placement, copy)?;
                let (symbol, arguments) =
                    local_copy_call(copy).ok_or_else(|| TileLoweringError::InvalidLocalCopy {
                        tile: tile.tile,
                        source_shard: copy.source,
                        source_offset: copy.source_offset,
                        destination_shard: copy.destination,
                        destination_offset: copy.destination_offset,
                        bytes: copy.bytes,
                    })?;
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
                TileStep::Repeat(lower_repeat(
                    program,
                    repeat,
                    placement,
                    kernels,
                    phases,
                    exchange_rows,
                )?)
            }
            TileWorkRef::Checkpoint(operation, breakpoint) => {
                TileStep::Checkpoint(crate::CheckpointStep {
                    operation: operation.index(),
                    breakpoint,
                    profile: StepProfile::default(),
                })
            }
        };
        steps.push(step);
    }
    Ok(steps)
}

fn local_copy_call(copy: &crate::LocalCopy) -> Option<(&'static str, Vec<u32>)> {
    let bytes = copy.bytes;
    if let crate::LocalCopyPattern::Strided {
        rows,
        row_bytes,
        source_stride,
        destination_stride,
    } = copy.pattern
    {
        return (rows >= 2
            && row_bytes != 0
            && row_bytes.is_multiple_of(8)
            && row_bytes.checked_mul(rows) == Some(bytes))
        .then(|| {
            (
                crate::COPY_STRIDED_U64_SYMBOL,
                vec![row_bytes / 8, rows, source_stride, destination_stride],
            )
        });
    }
    if bytes >= 6 * 8 && bytes.is_multiple_of(8) {
        let words = bytes / 8;
        Some((crate::COPY_U64_SYMBOL, vec![words / 6, words % 6]))
    } else if bytes != 0 && bytes.is_multiple_of(4) {
        Some((crate::COPY_U32_SYMBOL, vec![bytes / 4]))
    } else if bytes != 0 && bytes.is_multiple_of(2) {
        Some((crate::COPY_U16_SYMBOL, vec![bytes / 2]))
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
    )?;
    for step in &body {
        let TileStep::Exchange(exchange) = step else {
            continue;
        };
        if exchange
            .repeat_patches
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

fn lower_inactive_work(
    program: &LowProgram,
    work: &TileWorkList,
    exchange_rows: &BTreeMap<ExchangePhaseId, PlacedExchange>,
) -> Result<Vec<TileStep>, TileLoweringError> {
    let mut steps = Vec::new();
    for work in program.work(work) {
        match work {
            TileWorkRef::Exchange(id) => steps.push(TileStep::Exchange(ExchangeStep {
                active: exchange_rows[&id].active,
                incoming_base: exchange_rows[&id].incoming_base,
                program: exchange_rows[&id].program.clone(),
                setup_patch: exchange_rows[&id].setup_patch.clone(),
                repeat_patches: Vec::new(),
                profile: StepProfile::default(),
            })),
            TileWorkRef::Repeat(repeat) => steps.push(TileStep::Repeat(RepeatStep {
                count: repeat.count,
                iterated_pointers: Vec::new(),
                body: lower_inactive_work(program, &repeat.body, exchange_rows)?,
                profile: StepProfile::default(),
            })),
            TileWorkRef::Kernel(_) | TileWorkRef::LocalCopy(_) => {}
            TileWorkRef::Checkpoint(operation, breakpoint) => {
                steps.push(TileStep::Checkpoint(crate::CheckpointStep {
                    operation: operation.index(),
                    breakpoint,
                    profile: StepProfile::default(),
                }))
            }
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
    exchanges: &[PhysicalExchangePhase],
    execution_tile_count: u16,
    scheduled_tile_count: u16,
) -> Result<u32, TileLoweringError> {
    let mut maximum = 0;
    for tile in 0..execution_tile_count {
        let (_, end) = layout_exchange_rows(exchanges, tile, scheduled_tile_count, 0, false)?;
        maximum = maximum.max(end);
    }
    Ok(maximum)
}

/// Returns the final row address for one tile and physical exchange phase in
/// the same compact table layout used by package generation.
pub fn compact_exchange_row_address(
    exchanges: &[PhysicalExchangePhase],
    tile: u16,
    scheduled_tile_count: u16,
    base: u32,
    phase: ExchangePhaseId,
) -> Result<u32, TileLoweringError> {
    let (rows, _) = layout_exchange_rows(exchanges, tile, scheduled_tile_count, base, false)?;
    rows.get(&phase)
        .map(|row| row.program.address)
        .ok_or(TileLoweringError::UnknownExchange)
}

fn layout_exchange_rows(
    exchanges: &[PhysicalExchangePhase],
    tile: u16,
    scheduled_tile_count: u16,
    base: u32,
    validate_placement: bool,
) -> Result<(BTreeMap<ExchangePhaseId, PlacedExchange>, u32), TileLoweringError> {
    // SENDPICP carries its two receive-control values in the following word.
    // Keep every exchange program naturally aligned so the phase builder can
    // place these two-word instructions without knowing final SRAM addresses.
    let mut cursor = align_up(base, 8)?;
    let mut result = BTreeMap::new();
    if exchanges.is_empty() {
        return Ok((result, cursor));
    }
    struct SharedRow {
        program: PlacedExchangeRow,
        offsets: Option<PlacedExchangeRow>,
    }

    let mut key_counts = BTreeMap::<(Vec<u32>, Option<u32>), usize>::new();
    for phase in exchanges {
        let program = if tile < scheduled_tile_count {
            phase
                .programs
                .get(usize::from(tile))
                .ok_or(TileLoweringError::MissingExchangeRow(tile))?
        } else {
            continue;
        };
        let has_repeat_patches = !phase.repeat_patches[usize::from(tile)].is_empty();
        let key = (
            ipu_exchange::normalized_exchange_address_words(program),
            has_repeat_patches.then_some(phase.id.index()),
        );
        *key_counts.entry(key).or_default() += 1;
    }
    let mut shared = BTreeMap::<(Vec<u32>, Option<u32>), SharedRow>::new();
    for phase in exchanges {
        let (active, base_program) = if tile < scheduled_tile_count {
            let index = usize::from(tile);
            let active = *phase
                .active
                .get(index)
                .ok_or(TileLoweringError::MissingExchangeRow(tile))?;
            let program = phase
                .programs
                .get(index)
                .cloned()
                .ok_or(TileLoweringError::MissingExchangeRow(tile))?;
            (active, program)
        } else {
            (false, crate::inactive_exchange_program())
        };
        let has_repeat_patches =
            tile < scheduled_tile_count && !phase.repeat_patches[usize::from(tile)].is_empty();
        let key = (
            ipu_exchange::normalized_exchange_address_words(&base_program),
            has_repeat_patches.then_some(phase.id.index()),
        );
        let shared_count = key_counts.get(&key).copied().unwrap_or(1);
        let setup_entries = (shared_count > 1)
            .then(|| {
                key.0
                    .iter()
                    .zip(&base_program)
                    .enumerate()
                    .filter_map(|(offset, (&normalized, &target))| {
                        (normalized != target).then_some((offset, target))
                    })
                    .map(|(offset, target)| {
                        Ok((
                            u32::try_from(offset)
                                .map_err(|_| TileLoweringError::Overflow)?
                                .checked_mul(4)
                                .ok_or(TileLoweringError::Overflow)?,
                            target,
                        ))
                    })
                    .collect::<Result<Vec<_>, TileLoweringError>>()
            })
            .transpose()?
            .unwrap_or_default();
        let (program, offsets) = if let Some(shared) = shared.get(&key) {
            (shared.program.clone(), shared.offsets.clone())
        } else {
            cursor = align_up(cursor, 8)?;
            let address = cursor;
            let words = if shared_count > 1 {
                key.0.clone()
            } else {
                base_program.clone()
            };
            cursor = cursor
                .checked_add(
                    u32::try_from(words.len())
                        .map_err(|_| TileLoweringError::Overflow)?
                        .checked_mul(4)
                        .ok_or(TileLoweringError::Overflow)?,
                )
                .ok_or(TileLoweringError::Overflow)?;
            let program = PlacedExchangeRow { address, words };
            let offsets = if setup_entries.is_empty() {
                None
            } else {
                let words = setup_entries
                    .iter()
                    .map(|&(offset, _)| offset)
                    .collect::<Vec<_>>();
                let address = cursor;
                cursor = cursor
                    .checked_add(
                        u32::try_from(words.len())
                            .map_err(|_| TileLoweringError::Overflow)?
                            .checked_mul(4)
                            .ok_or(TileLoweringError::Overflow)?,
                    )
                    .ok_or(TileLoweringError::Overflow)?;
                Some(PlacedExchangeRow { address, words })
            };
            shared.insert(
                key.clone(),
                SharedRow {
                    program: program.clone(),
                    offsets: offsets.clone(),
                },
            );
            (program, offsets)
        };
        let setup_patch = if setup_entries.is_empty() {
            None
        } else {
            let words = setup_entries
                .into_iter()
                .map(|(_, value)| value)
                .collect::<Vec<_>>();
            let address = cursor;
            cursor = cursor
                .checked_add(
                    u32::try_from(words.len())
                        .map_err(|_| TileLoweringError::Overflow)?
                        .checked_mul(4)
                        .ok_or(TileLoweringError::Overflow)?,
                )
                .ok_or(TileLoweringError::Overflow)?;
            Some(ExchangeSetupPatch {
                offsets: offsets.expect("a shared exchange row has patch offsets"),
                values: PlacedExchangeRow { address, words },
            })
        };
        let mut repeat_patches = Vec::new();
        if tile < scheduled_tile_count {
            for patch in &phase.repeat_patches[usize::from(tile)] {
                let address = cursor;
                cursor = cursor
                    .checked_add(
                        u32::try_from(patch.values.len())
                            .map_err(|_| TileLoweringError::Overflow)?
                            .checked_mul(4)
                            .ok_or(TileLoweringError::Overflow)?,
                    )
                    .ok_or(TileLoweringError::Overflow)?;
                repeat_patches.push(ExchangePatch {
                    word_offset: patch.word_offset,
                    values: PlacedExchangeRow {
                        address,
                        words: patch.values.clone(),
                    },
                });
            }
        }
        if active && validate_placement {
            let diagnostic = crate::diagnose_exchange_tile(phase, tile, program.address)?;
            if let Some(conflict) = diagnostic
                .activities
                .iter()
                .find(|activity| activity.conflicts_with_row)
            {
                return Err(TileLoweringError::ExchangeRowDataConflict {
                    tile,
                    phase: phase.id,
                    row_address: program.address,
                    transfer: conflict.activity.transfer,
                    kind: conflict.activity.kind,
                    data_address: conflict.activity.address,
                });
            }
        }
        result.insert(
            phase.id,
            PlacedExchange {
                active,
                incoming_base: if tile < scheduled_tile_count {
                    phase.incoming_bases[usize::from(tile)]
                } else {
                    0
                },
                program,
                setup_patch,
                repeat_patches,
            },
        );
    }
    Ok((result, cursor))
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
                        layout: Layout::block_major_matrix(64, tiles),
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
                false,
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
            let copy = crate::LocalCopy {
                source: crate::LowShardId::from_index(0),
                source_offset: 0,
                destination: crate::LowShardId::from_index(1),
                destination_offset: 0,
                bytes,
                pattern: crate::LocalCopyPattern::Contiguous,
            };
            let (symbol, arguments) = local_copy_call(&copy).unwrap();
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
