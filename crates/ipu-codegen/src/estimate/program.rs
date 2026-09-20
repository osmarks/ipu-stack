//! Price executable work and compose per-tile timelines across barriers/repeats.

use super::{IPU21_TARGET_COSTS as TARGET, *};
use crate::{BlockOperation, BlockRegion, ExpansionResult, KernelRun, TileGraph};
// Contracts are already interned by expansion. Keep the few physical shape
// variants under each contract, comparing borrowed widths without allocation.
type KernelCosts<'a> =
    std::collections::HashMap<*const crate::low::KernelRunMetadata, Vec<(&'a KernelRun, u64)>>;
fn cached_kernel_cycles<'a>(run: &'a KernelRun, costs: &mut KernelCosts<'a>) -> u64 {
    fn shapes(run: &KernelRun) -> impl Iterator<Item = &[crate::ShardExtent]> {
        run.inputs
            .iter()
            .chain(&run.outputs)
            .map(|view| view.extents.as_slice())
    }
    let variants = costs
        .entry(std::sync::Arc::as_ptr(&run.metadata))
        .or_default();
    let found = variants.iter().find(|(other, _)| {
        run.inputs.len() == other.inputs.len()
            && shapes(run).zip(shapes(other)).all(|(a, b)| {
                a.len() == b.len()
                    && a.iter()
                        .zip(b)
                        .all(|(a, b)| a.physical_end - a.start == b.physical_end - b.start)
            })
    });
    if let Some((_, cycles)) = found {
        #[cfg(test)]
        assert_eq!(*cycles, run.call(None).map_or(u64::MAX, |call| call.cycles));
        return *cycles;
    }
    let cycles = run.call(None).map_or(u64::MAX, |call| call.cycles);
    variants.push((run, cycles));
    cycles
}

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

    fn barrier(&mut self, cycles: u64, exchange: u64) {
        self.middle = Some(self.middle.map_or(cycles, |middle| {
            middle
                .saturating_add(maximum(&self.tail))
                .saturating_add(cycles)
        }));
        self.tail.fill(0);
        self.exchange = self.exchange.saturating_add(exchange);
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
            self.barrier(
                middle.saturating_add(between.saturating_add(middle).saturating_mul(count - 1)),
                body.exchange.saturating_mul(count),
            );
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
    let geometry = GeometryCache::default();
    let estimated;
    let phases = if let Some(costs) = exchange {
        costs
    } else {
        estimated = program
            .exchange_phases
            .iter()
            .map(|phase| {
                let traffic = geometry_traffic(program, phase, None, &geometry)?;
                let cycles = super::cycles::exchange_endpoint_cycles(&traffic, 1);
                tracing::debug!(phase = phase.id.index(), source = ?phase.provenance.operation,
                    bytes = traffic.maximum_payload_bytes(), fragments = traffic.maximum_fragments(),
                    controls = traffic.maximum_controls(), cycles, "estimated logical exchange");
                Ok(cycles)
            })
            .collect::<ExpansionResult<Vec<_>>>()?;
        &estimated
    };
    fn region<'a>(
        program: &'a TileGraph,
        body: &BlockRegion,
        phases: &[u64],
        kernels: &mut KernelCosts<'a>,
    ) -> Timeline {
        let mut timeline = Timeline::new(usize::from(program.tile_count));
        for operation in &body.operations {
            match operation {
                BlockOperation::Compute { tile, run } => timeline.local(
                    usize::from(*tile),
                    cached_kernel_cycles(&program.kernel_runs[run.0 as usize], kernels),
                ),
                BlockOperation::Copy { tile, copy } => timeline.local(
                    usize::from(*tile),
                    program.local_copies[copy.0 as usize].call().cycles,
                ),
                BlockOperation::Exchange(phase) => {
                    let cycles = phases[phase.index() as usize];
                    timeline.barrier(cycles, cycles);
                }
                BlockOperation::Repeat(repeat) => timeline.repeat(
                    region(program, &repeat.body, phases, kernels),
                    u64::from(repeat.count),
                ),
                BlockOperation::Checkpoint(..) => {}
            }
        }
        timeline
    }
    Ok(region(
        program,
        &program.body,
        phases,
        &mut std::collections::HashMap::new(),
    )
    .cycles())
}

fn geometry_traffic(
    program: &TileGraph,
    phase: &crate::ExchangePhase,
    mut storage: Option<&mut ExchangeStoragePhase>,
    geometry: &GeometryCache,
) -> ExpansionResult<ExchangeEndpointTraffic> {
    #[cfg(test)]
    let mut expected_storage = storage.as_deref().cloned();
    let mut traffic = ExchangeEndpointTraffic::default();
    let mut receive_end = vec![None; usize::from(program.tile_count)];
    for transfer in &phase.transfers {
        let source = &program.shards[transfer.source.shard.index() as usize];
        let order = transfer.span_order(&program.shards);
        let source_geometry = transfer
            .source
            .bind(&program.shards)?
            .geometry(geometry, order)?;
        let bytes = source_geometry.traversal.byte_len();
        let mut outgoing_fragments = 0;
        let mut outgoing_long_fragments = 0;
        for destination in &transfer.destinations {
            let target = &program.shards[destination.shard.index() as usize];
            let target_geometry = destination
                .bind(&program.shards)?
                .geometry(geometry, order)?;
            let copy = geometry.pair(&source_geometry, &target_geometry)?;
            let mut fragments = 0;
            let mut long_fragments = 0;
            // Relative allocation/offset identities expose pointer continuation
            // without placement. This follows the supplied transfer order;
            // scheduling can change that order, pairing and control overlap.
            let mut resets = 0;
            for [_, row] in &copy.rows {
                let limit = crate::exchange::MAX_TRANSFER_WORDS * 4;
                fragments += u64::from(row.rows) * u64::from(row.bytes.div_ceil(limit));
                long_fragments += u64::from(row.rows)
                    * (u64::from(row.bytes / limit) * u64::from(limit > 256)
                        + u64::from(row.bytes % limit > 256));
                let address = (u64::from(destination.shard.index()) << 32) + u64::from(row.offset);
                if let Some(storage) = storage.as_deref_mut() {
                    storage.connection_rows(
                        source.tile,
                        target.tile,
                        address,
                        row.bytes,
                        row.rows,
                        row.stride,
                        transfer.destinations.len(),
                        crate::exchange::MAX_TRANSFER_WORDS * 4,
                    );
                }
                let end = &mut receive_end[usize::from(target.tile)];
                resets += u64::from(*end != Some(address))
                    + u64::from(row.rows.saturating_sub(1)) * u64::from(row.stride != row.bytes);
                *end = Some(
                    address
                        + u64::from(row.rows - 1) * u64::from(row.stride)
                        + u64::from(row.bytes),
                );
            }
            outgoing_fragments = outgoing_fragments.max(fragments);
            outgoing_long_fragments = outgoing_long_fragments.max(long_fragments);
            traffic.add_receive(target.tile, bytes, fragments, resets);
        }
        if let Some(storage) = storage.as_deref_mut() {
            storage.send(source.tile, outgoing_fragments, outgoing_long_fragments);
        }
        // A multicast source is sent once, rather than once per receiver.
        traffic.add_outgoing(source.tile, bytes, outgoing_fragments);
    }

    #[cfg(test)]
    {
        let expected = enumerated_geometry_traffic(program, phase, expected_storage.as_mut())?;
        assert_eq!(traffic, expected);
        assert_eq!(storage.as_deref(), expected_storage.as_ref());
    }
    Ok(traffic)
}

#[cfg(test)]
pub(crate) fn program_footprint(program: &TileGraph) -> ExpansionResult<ExchangeFootprint> {
    program_footprint_analyzed(program, &GeometryCache::default())
}

#[cfg(test)]
pub(crate) fn program_footprint_analyzed(
    program: &TileGraph,
    geometry: &GeometryCache,
) -> ExpansionResult<ExchangeFootprint> {
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
        let traffic = geometry_traffic(program, phase, Some(&mut storage), geometry)?;
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
fn enumerated_geometry_traffic(
    program: &TileGraph,
    phase: &crate::ExchangePhase,
    mut storage: Option<&mut ExchangeStoragePhase>,
) -> ExpansionResult<ExchangeEndpointTraffic> {
    let mut traffic = ExchangeEndpointTraffic::default();
    let mut receive_end = vec![None; usize::from(program.tile_count)];
    for transfer in &phase.transfers {
        let source = &program.shards[transfer.source.shard.index() as usize];
        let order = transfer.span_order(&program.shards);
        let source_spans = crate::view_byte_traversal(source, &transfer.source, order)?;
        let bytes = source_spans.byte_len();
        let mut outgoing_fragments = 0;
        let mut outgoing_long_fragments = 0;
        for destination in &transfer.destinations {
            let target = &program.shards[destination.shard.index() as usize];
            let mut fragments = 0u64;
            let mut resets = 0;
            let mut long_fragments = 0u64;
            let target_spans = crate::view_byte_traversal(target, destination, order)?;
            {
                crate::for_each_copy_span(
                    source_spans.spans(),
                    target_spans.spans(),
                    |_, offset, bytes| {
                        let address =
                            (u64::from(destination.shard.index()) << 32) + u64::from(offset);
                        let end = &mut receive_end[usize::from(target.tile)];
                        resets += u64::from(*end != Some(address));
                        *end = Some(address + u64::from(bytes));
                        if let Some(storage) = storage.as_deref_mut() {
                            let max_bytes = u64::from(crate::exchange::MAX_TRANSFER_WORDS) * 4;
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
                            u64::from(bytes)
                                .div_ceil(u64::from(crate::exchange::MAX_TRANSFER_WORDS) * 4),
                        );
                        Ok(())
                    },
                )?;
            }
            outgoing_fragments = outgoing_fragments.max(fragments);
            outgoing_long_fragments = outgoing_long_fragments.max(long_fragments);
            traffic.add_receive(target.tile, bytes, fragments, resets);
        }
        if let Some(storage) = storage.as_deref_mut() {
            storage.send(source.tile, outgoing_fragments, outgoing_long_fragments);
        }
        // A multicast source is sent once, rather than once per receiver.
        traffic.add_outgoing(source.tile, bytes, outgoing_fragments);
    }

    Ok(traffic)
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
            requires_finite_scratch: false,
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
            value_views: vec![],
            logical_values: vec![],
            checkpoints: vec![],
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
        program.exchange_phases[0].transfers = (0..2048)
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
                    body.barrier(cycles, cycles);
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
