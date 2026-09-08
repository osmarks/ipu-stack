//! Price executable work and compose per-tile timelines across barriers/repeats.

use super::{IPU21_TARGET_COSTS as TARGET, *};
use crate::{BlockOperation, BlockRegion, ExpansionResult, KernelRun, TileGraph};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ProgramCycles {
    pub total: u64,
    pub exchange: u64,
}

/// Exact exchange horizons composed with the emitted compute/copy timeline.
pub(crate) fn scheduled_program_cycles(
    program: &TileGraph,
    phases: &[crate::PhysicalExchangePhase],
) -> ExpansionResult<ProgramCycles> {
    let mut cycles = vec![0; program.exchange_phases.len()];
    for phase in phases {
        cycles[phase.id.index() as usize] =
            u64::from(phase.event_cycles).saturating_add(TARGET.exchange_phase_cycles);
    }
    program_cycles(program, Some(&cycles))
}

/// Prefix and tail are tile-local. The middle starts at the first barrier and
/// ends at the last one. This permits exact repeat composition without unrolling
/// and without introducing a barrier at an operation or repeat boundary.
struct Timeline {
    prefix: Vec<u64>,
    middle: Option<u64>,
    tail: Vec<u64>,
    exchange: u64,
}

impl Timeline {
    fn new(tiles: usize) -> Self {
        Self {
            prefix: vec![0; tiles],
            middle: None,
            tail: vec![0; tiles],
            exchange: 0,
        }
    }

    fn local(&mut self, tile: usize, cycles: u64) {
        let times = if self.middle.is_some() {
            &mut self.tail
        } else {
            &mut self.prefix
        };
        times[tile] = times[tile].saturating_add(cycles);
    }

    fn barrier(&mut self, cycles: u64) {
        self.middle = Some(self.middle.map_or(cycles, |middle| {
            middle
                .saturating_add(maximum(&self.tail))
                .saturating_add(cycles)
        }));
        self.tail.fill(0);
        self.exchange = self.exchange.saturating_add(cycles);
    }

    fn repeat(&mut self, body: Self, count: u64) {
        if count == 0 {
            return;
        }
        if let Some(middle) = body.middle {
            for (tile, &cycles) in body.prefix.iter().enumerate() {
                self.local(tile, cycles);
            }
            let between = body
                .tail
                .iter()
                .zip(&body.prefix)
                .map(|(tail, prefix)| tail.saturating_add(*prefix))
                .max()
                .unwrap_or(0);
            // barrier() accounts this as exchange; replace that bookkeeping with
            // the body's actual exchange contribution after composing latency.
            let exchange = self.exchange;
            self.barrier(
                middle.saturating_add(between.saturating_add(middle).saturating_mul(count - 1)),
            );
            self.exchange = exchange.saturating_add(body.exchange.saturating_mul(count));
            for (tile, &cycles) in body.tail.iter().enumerate() {
                self.local(tile, cycles);
            }
        } else {
            for (tile, &cycles) in body.prefix.iter().enumerate() {
                self.local(tile, cycles.saturating_mul(count));
            }
        }
    }

    fn cycles(self) -> ProgramCycles {
        ProgramCycles {
            total: maximum(&self.prefix)
                .saturating_add(self.middle.unwrap_or(0))
                .saturating_add(maximum(&self.tail)),
            exchange: self.exchange,
        }
    }
}

fn maximum(values: &[u64]) -> u64 {
    values.iter().copied().max().unwrap_or(0)
}

pub(crate) fn program_cycles(
    program: &TileGraph,
    exchange: Option<&[u64]>,
) -> ExpansionResult<ProgramCycles> {
    let phases = if let Some(costs) = exchange {
        costs.to_vec()
    } else {
        program
            .exchange_phases
            .iter()
            .map(|phase| {
                let traffic = phase_traffic(program, phase)?;
                // Fragmented transfers also consume routing/pointer events.
                // Use the same calibration as materialization selection;
                // bandwidth alone makes scattered views look nearly free.
                Ok(super::cycles::exchange_endpoint_cycles(&traffic, 1).max(
                    traffic
                        .maximum_fragments()
                        .saturating_mul(super::IPU21_LOGICAL_FRAGMENT_CYCLES),
                ))
            })
            .collect::<ExpansionResult<Vec<_>>>()?
    };
    fn region(program: &TileGraph, body: &BlockRegion, phases: &[u64]) -> Timeline {
        let mut timeline = Timeline::new(usize::from(program.tile_count));
        for operation in &body.operations {
            match operation {
                BlockOperation::Compute { tile, run } => timeline.local(
                    usize::from(*tile),
                    kernel_cycles(&program.kernel_runs[run.0 as usize]),
                ),
                BlockOperation::Copy { tile, copy } => {
                    let copy = &program.local_copies[copy.0 as usize];
                    // The strided helper assigns complete rows to six workers.
                    // Its inner loop has six instructions per 64-bit word and
                    // five instructions between rows; short rows cannot attain
                    // the contiguous-copy bandwidth.
                    let work = match copy.pattern {
                        crate::CopyPattern::Contiguous => {
                            u64::from(copy.bytes).div_ceil(TARGET.local_copy_bytes_per_cycle)
                        }
                        crate::CopyPattern::Strided {
                            rows, row_bytes, ..
                        } => u64::from(rows)
                            .div_ceil(6)
                            .saturating_mul(6)
                            .saturating_mul(
                                u64::from(row_bytes)
                                    .div_ceil(8)
                                    .saturating_mul(6)
                                    .saturating_add(5),
                            ),
                    };
                    timeline.local(
                        usize::from(*tile),
                        work.saturating_add(TARGET.local_copy_call_cycles),
                    );
                }
                BlockOperation::Exchange(phase) => timeline.barrier(phases[phase.index() as usize]),
                BlockOperation::Repeat(repeat) => timeline.repeat(
                    region(program, &repeat.body, phases),
                    u64::from(repeat.count),
                ),
                BlockOperation::Checkpoint(..) => {}
            }
        }
        timeline
    }
    Ok(region(program, &program.body, &phases).cycles())
}

pub(super) fn phase_traffic(
    program: &TileGraph,
    phase: &crate::ExchangePhase,
) -> ExpansionResult<ExchangeEndpointTraffic> {
    geometry_traffic(program, phase, true, None)
}

fn geometry_traffic(
    program: &TileGraph,
    phase: &crate::ExchangePhase,
    shared_tx_lane: bool,
    mut storage: Option<&mut ExchangeStoragePhase>,
) -> ExpansionResult<ExchangeEndpointTraffic> {
    let mut traffic = ExchangeEndpointTraffic::default();
    for transfer in &phase.transfers {
        let source = &program.shards[transfer.source.shard.index() as usize];
        let spans = match transfer.span_order(&program.shards) {
            crate::CopyOrder::Physical => crate::view_byte_spans,
            crate::CopyOrder::Semantic => crate::logical_view_byte_spans,
        };
        let source_spans = spans(source, &transfer.source)?;
        let bytes = source_spans.iter().map(|span| u64::from(span.bytes)).sum();
        let mut outgoing_fragments = 0;
        let mut outgoing_long_fragments = 0;
        for destination in &transfer.destinations {
            let target = &program.shards[destination.shard.index() as usize];
            let mut fragments = 0u64;
            let mut long_fragments = 0u64;
            crate::for_each_copy_span(
                &source_spans,
                &spans(target, destination)?,
                |_, offset, bytes| {
                    if let Some(storage) = storage.as_deref_mut() {
                        let max_bytes = u64::from(ipu_exchange::MAX_TRANSFER_WORDS) * 4;
                        let mut remaining = u64::from(bytes);
                        let mut address =
                            (u64::from(destination.shard.index()) << 32) + u64::from(offset);
                        while remaining != 0 {
                            let chunk = remaining.min(max_bytes);
                            long_fragments += u64::from(chunk > 256);
                            storage.connection(
                                source.tile,
                                target.tile,
                                chunk,
                                false,
                                transfer.destinations.len(),
                            );
                            storage.receive(target.tile, address, chunk);
                            address += chunk;
                            remaining -= chunk;
                        }
                    }
                    fragments = fragments.saturating_add(
                        u64::from(bytes).div_ceil(u64::from(ipu_exchange::MAX_TRANSFER_WORDS) * 4),
                    );
                    Ok(())
                },
            )?;
            outgoing_fragments = outgoing_fragments.max(fragments);
            outgoing_long_fragments = outgoing_long_fragments.max(long_fragments);
            traffic.add_incoming(target.tile, bytes, fragments);
        }
        if let Some(storage) = storage.as_deref_mut() {
            storage.send(source.tile, outgoing_fragments, outgoing_long_fragments);
        }
        // A multicast source is sent once, rather than once per receiver.
        traffic.add_outgoing(
            if shared_tx_lane {
                source.tile / 2
            } else {
                source.tile
            },
            bytes,
            outgoing_fragments,
        );
    }

    Ok(traffic)
}

fn kernel_cycles(run: &KernelRun) -> u64 {
    let tensor = |view: &crate::ShardView, format: &crate::TensorFormat| TensorType {
        shape: TensorShape(
            view.extents
                .iter()
                .map(|extent| extent.physical_end - extent.start)
                .collect(),
        ),
        format: format.clone(),
    };
    let inputs = run
        .inputs
        .iter()
        .zip(&run.requirements.inputs)
        .filter_map(|(operand, access)| {
            operand
                .views
                .first()
                .map(|view| tensor(view, &access.format))
        })
        .collect::<Vec<_>>();
    super::primitive::kernel_cycles(
        &run.kernel,
        &inputs,
        &tensor(&run.output, &run.requirements.output.format),
    )
}

pub(crate) fn program_footprint(program: &TileGraph) -> ExpansionResult<ExchangeFootprint> {
    // Storage belongs to a tile, not to a shared transmit lane. Count both
    // endpoint roles conservatively (bidi encoding may later combine them).
    // Sum each tile across static phases before taking the maximum; Repeat
    // execution counts do not multiply its stored table.
    let mut chunks = vec![0u64; usize::from(program.tile_count)];
    fn iterated_sources(region: &BlockRegion, sources: &mut HashSet<crate::BlockValueId>) {
        for operation in &region.operations {
            if let BlockOperation::Repeat(repeat) = operation {
                sources.extend(
                    repeat
                        .bindings
                        .iter()
                        .flat_map(|binding| &binding.iterated)
                        .map(|binding| binding.argument),
                );
                iterated_sources(&repeat.body, sources);
            }
        }
    }
    let mut iterated = HashSet::new();
    iterated_sources(&program.body, &mut iterated);
    let mut table = ExchangeStorageEstimator::new(program.tile_count);
    for phase in &program.exchange_phases {
        let mut storage = ExchangeStoragePhase::new(program.tile_count);
        let traffic = geometry_traffic(program, phase, false, Some(&mut storage))?;
        for transfer in &phase.transfers {
            if iterated.contains(&transfer.source.shard) {
                storage
                    .disable_sharing(program.shards[transfer.source.shard.index() as usize].tile);
            }
        }
        table.add(storage);
        for (tile, load) in traffic
            .outgoing_lanes
            .iter()
            .enumerate()
            .chain(traffic.incoming_tiles.iter().enumerate())
        {
            let count = &mut chunks[tile];
            *count = count.saturating_add(load.fragments);
        }
    }
    Ok(ExchangeFootprint {
        phases: program.exchange_phases.len() as u64,
        maximum_transfer_chunks_per_tile: chunks.into_iter().max().unwrap_or(0),
        encoded_row_bytes: Some(table.maximum_bytes()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduled_phase_prices_follow_repeat_execution_counts() {
        use crate::*;
        let provenance = WorkProvenance {
            operation: None,
            value: None,
            reason: WorkReason::OperatorKernel,
        };
        let phase = ExchangePhaseId(0);
        let body = BlockRegion {
            operations: vec![BlockOperation::Exchange(phase)],
        };
        let mut program = TileGraph {
            tile_count: 2,
            shards: vec![],
            kernel_runs: vec![],
            local_copies: vec![],
            exchange_phases: vec![ExchangePhase {
                id: phase,
                provenance,
                transfers: vec![],
            }],
            body: BlockRegion {
                operations: vec![BlockOperation::Repeat(Box::new(BlockRepeat {
                    provenance,
                    count: 7,
                    bindings: vec![],
                    body,
                }))],
            },
            inputs: vec![],
            outputs: vec![],
            values: vec![],
            logical_values: vec![],
            checkpoints: vec![],
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        };
        assert_eq!(
            program_cycles(&program, Some(&[123])).unwrap(),
            ProgramCycles {
                total: 861,
                exchange: 861
            }
        );
        assert_eq!(program_footprint(&program).unwrap().phases, 1);

        // Identical payloads with many routing fragments must not receive the
        // same analytical price. Scheduled prices replace that approximation.
        let full = vec![ShardExtent {
            axis: 0,
            start: 0,
            logical_end: 4096,
            physical_end: 4096,
        }];
        program.shards = (0..2)
            .map(|tile| BlockValue {
                id: BlockValueId::from_index(tile),
                tile: tile as u16,
                tensor_type: TensorType::new(
                    [4096],
                    Precision::F16,
                    Layout::row_major(TensorTiling::replicated(2)),
                ),
                extents: full.clone(),
                definition: ShardDefinition::Value(crate::MidValueId::from_index(tile)),
            })
            .collect();
        let transfer = |extents: Vec<ShardExtent>| LogicalExchange {
            source: ShardView {
                shard: BlockValueId::from_index(0),
                extents: extents.clone(),
            },
            destinations: vec![ShardView {
                shard: BlockValueId::from_index(1),
                extents,
            }],
            order: crate::CopyOrder::Physical,
        };
        program.exchange_phases[0].transfers = vec![transfer(full)];
        let footprint = program_footprint(&program).unwrap();
        // A contiguous 8 KiB span needs one transfer, not the 32 fragments
        // assumed by the mid cycle heuristic's 256-byte payload.
        assert_eq!(footprint.maximum_transfer_chunks_per_tile, 1);
        assert_eq!(footprint.estimated_row_bytes(), 24);
        let mut distributed = program.clone();
        distributed.tile_count = 4;
        for tile in 2..4 {
            let mut shard = distributed.shards[tile - 2].clone();
            shard.id = BlockValueId::from_index(tile as u32);
            shard.tile = tile as u16;
            distributed.shards.push(shard);
        }
        let mut other = distributed.exchange_phases[0].clone();
        other.id = ExchangePhaseId(1);
        other.transfers[0].source.shard = BlockValueId::from_index(2);
        other.transfers[0].destinations[0].shard = BlockValueId::from_index(3);
        distributed.exchange_phases.push(other);
        let distributed_rows = program_footprint(&distributed).unwrap();
        assert_eq!(distributed_rows.maximum_transfer_chunks_per_tile, 1);
        assert_eq!(distributed_rows.estimated_row_bytes(), 32);
        let BlockOperation::Repeat(repeat) = &mut program.body.operations[0] else {
            unreachable!()
        };
        repeat.count = 1000;
        assert_eq!(program_footprint(&program).unwrap(), footprint);
        let BlockOperation::Repeat(repeat) = &mut program.body.operations[0] else {
            unreachable!()
        };
        repeat.count = 7;
        let contiguous = program_cycles(&program, None).unwrap();
        program.exchange_phases[0].transfers = (0..64)
            .map(|i| {
                transfer(vec![ShardExtent {
                    axis: 0,
                    start: i * 2,
                    logical_end: i * 2 + 2,
                    physical_end: i * 2 + 2,
                }])
            })
            .collect();
        assert!(program_cycles(&program, None).unwrap().total > contiguous.total);
        assert_eq!(program_cycles(&program, Some(&[123])).unwrap().total, 861);
    }

    #[test]
    fn repeated_timelines_match_unrolled_execution() {
        let mut random = fastrand::Rng::with_seed(0x74696d656c696e65);
        for _ in 0..1000 {
            let tiles = random.usize(1..8);
            let count = random.u64(0..8);
            let prefix = (0..tiles).map(|_| random.u64(0..100)).collect::<Vec<_>>();
            let events = (0..random.usize(0..30))
                .map(|_| (random.usize(0..=tiles), random.u64(0..100)))
                .collect::<Vec<_>>();
            let mut compact = Timeline::new(tiles);
            for (tile, &cycles) in prefix.iter().enumerate() {
                compact.local(tile, cycles);
            }
            let mut body = Timeline::new(tiles);
            for &(tile, cycles) in &events {
                if tile == tiles {
                    body.barrier(cycles);
                } else {
                    body.local(tile, cycles);
                }
            }
            compact.repeat(body, count);
            let mut clocks = prefix;
            let mut exchange = 0;
            for _ in 0..count {
                for &(tile, cycles) in &events {
                    if tile == tiles {
                        let end = maximum(&clocks) + cycles;
                        clocks.fill(end);
                        exchange += cycles;
                    } else {
                        clocks[tile] += cycles;
                    }
                }
            }
            assert_eq!(
                compact.cycles(),
                ProgramCycles {
                    total: maximum(&clocks),
                    exchange
                }
            );
        }
    }
}
