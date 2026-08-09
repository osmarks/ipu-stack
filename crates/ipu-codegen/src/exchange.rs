//! Physical exchange programs generated from logical shard transfers.

use crate::{
    ExchangePhaseId, LowProgram, Placement, ShardDefinition, logical_view_byte_spans,
    view_byte_spans,
};
use ipu_exchange::{
    MAX_TRANSFER_WORDS, MulticastPlan, PLAN_WORDS, PlanProgramBuilder, RETURN_M10_INSTRUCTION,
    SANS_INACTIVE_INSTRUCTION, SYNC_ANS_INSTRUCTION, Topology, finalize_point_receiver,
    offset_plan, patch_receiver_address, patch_sender_address, plan_event_cycles,
};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalExchangePhase {
    pub id: ExchangePhaseId,
    /// Executable supervisor row indexed by logical tile.
    pub rows: Vec<Vec<u32>>,
    pub event_cycles: u32,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ExchangeLoweringError {
    #[error(transparent)]
    Exchange(#[from] ipu_exchange::ExchangeError),
    #[error(transparent)]
    Storage(#[from] crate::StorageError),
    #[error("exchange refers to an unplaced shard")]
    UnplacedShard,
    #[error("exchange destination is not writable")]
    InvalidDestination,
    #[error("exchange payload is not a nonempty whole number of words")]
    UnalignedPayload,
    #[error("exchange source and destination byte counts differ")]
    SizeMismatch,
    #[error("exchange address arithmetic overflowed")]
    Overflow,
}

pub fn lower_exchanges(
    program: &LowProgram,
    placement: &Placement,
    topology: &Topology,
) -> Result<Vec<PhysicalExchangePhase>, ExchangeLoweringError> {
    program
        .exchange_phases
        .iter()
        .map(|phase| {
            let mut builders = BTreeMap::<u16, PlanProgramBuilder>::new();
            let mut horizon = 0u32;
            let mut tile_availability = vec![0u32; usize::from(program.tile_count)];
            for transfer in &phase.transfers {
                let source = &program.shards[transfer.source.shard.index() as usize];
                let logical_order = transfer.destinations.iter().any(|view| {
                    program.shards[view.shard.index() as usize]
                        .tensor_type
                        .format
                        .layout
                        .order
                        != source.tensor_type.format.layout.order
                });
                let source_base = placement
                    .shard_addresses
                    .get(&source.id)
                    .copied()
                    .ok_or(ExchangeLoweringError::UnplacedShard)?;
                let destinations = transfer
                    .destinations
                    .iter()
                    .map(|view| {
                        let shard = &program.shards[view.shard.index() as usize];
                        if matches!(shard.definition, ShardDefinition::Alias(_)) {
                            return Err(ExchangeLoweringError::InvalidDestination);
                        }
                        Ok((
                            shard.tile,
                            placement
                                .shard_addresses
                                .get(&view.shard)
                                .copied()
                                .ok_or(ExchangeLoweringError::UnplacedShard)?,
                            if logical_order {
                                logical_view_byte_spans(shard, view)?
                            } else {
                                view_byte_spans(shard, view)?
                            },
                        ))
                    })
                    .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
                if destinations.is_empty() {
                    return Err(ExchangeLoweringError::SizeMismatch);
                }
                let source_spans = if logical_order {
                    logical_view_byte_spans(source, &transfer.source)?
                } else {
                    view_byte_spans(source, &transfer.source)?
                };
                let source_bytes = source_spans.iter().try_fold(0u32, |total, span| {
                    total
                        .checked_add(span.bytes)
                        .ok_or(ExchangeLoweringError::Overflow)
                })?;
                for (_, _, spans) in &destinations {
                    let destination_bytes = spans.iter().try_fold(0u32, |total, span| {
                        total
                            .checked_add(span.bytes)
                            .ok_or(ExchangeLoweringError::Overflow)
                    })?;
                    if destination_bytes != source_bytes {
                        return Err(ExchangeLoweringError::SizeMismatch);
                    }
                }
                let mut source_index = 0usize;
                let mut source_offset = 0u32;
                let mut destination_positions = vec![(0usize, 0u32); destinations.len()];
                while source_index < source_spans.len() {
                    let source_span = source_spans[source_index];
                    if source_span.bytes == 0 || source_span.offset & 3 != 0 {
                        return Err(ExchangeLoweringError::UnalignedPayload);
                    }
                    let mut chunk_bytes = (source_span.bytes - source_offset).min(
                        MAX_TRANSFER_WORDS
                            .checked_mul(4)
                            .ok_or(ExchangeLoweringError::Overflow)?,
                    );
                    for ((index, offset), (_, _, spans)) in
                        destination_positions.iter().zip(&destinations)
                    {
                        let span = spans
                            .get(*index)
                            .ok_or(ExchangeLoweringError::SizeMismatch)?;
                        if span.offset & 3 != 0 {
                            return Err(ExchangeLoweringError::UnalignedPayload);
                        }
                        chunk_bytes = chunk_bytes.min(span.bytes - *offset);
                    }
                    if chunk_bytes == 0 || chunk_bytes & 3 != 0 {
                        return Err(ExchangeLoweringError::UnalignedPayload);
                    }
                    let source_address = source_base
                        .checked_add(source_span.offset)
                        .and_then(|address| address.checked_add(source_offset))
                        .ok_or(ExchangeLoweringError::Overflow)?;
                    let destination_entries = destinations
                        .iter()
                        .zip(&destination_positions)
                        .map(|((tile, base, spans), (index, offset))| {
                            let span = spans
                                .get(*index)
                                .ok_or(ExchangeLoweringError::SizeMismatch)?;
                            Ok((
                                *tile,
                                base.checked_add(span.offset)
                                    .and_then(|address| address.checked_add(*offset))
                                    .ok_or(ExchangeLoweringError::Overflow)?,
                            ))
                        })
                        .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
                    let schedule_offset = std::iter::once(source.tile)
                        .chain(destination_entries.iter().map(|entry| entry.0))
                        .map(|tile| tile_availability[usize::from(tile)])
                        .max()
                        .unwrap_or(0);
                    let schedule_offset = if schedule_offset == 0 {
                        0
                    } else {
                        schedule_offset
                            .checked_add(1)
                            .ok_or(ExchangeLoweringError::Overflow)?
                    };
                    let transfer_end = append_transfer(
                        topology,
                        ScheduledTransfer {
                            source: source.tile,
                            destinations: &destination_entries,
                            source_address,
                            words: chunk_bytes / 4,
                            schedule_offset,
                        },
                        &mut horizon,
                        &mut builders,
                    )?;
                    for tile in std::iter::once(source.tile)
                        .chain(destination_entries.iter().map(|entry| entry.0))
                    {
                        tile_availability[usize::from(tile)] = transfer_end;
                    }
                    source_offset += chunk_bytes;
                    if source_offset == source_span.bytes {
                        source_index += 1;
                        source_offset = 0;
                    }
                    for ((index, offset), (_, _, spans)) in
                        destination_positions.iter_mut().zip(&destinations)
                    {
                        *offset += chunk_bytes;
                        if *offset == spans[*index].bytes {
                            *index += 1;
                            *offset = 0;
                        }
                    }
                }
                if destination_positions
                    .iter()
                    .zip(&destinations)
                    .any(|((index, offset), (_, _, spans))| *index != spans.len() || *offset != 0)
                {
                    return Err(ExchangeLoweringError::SizeMismatch);
                }
            }
            let rows = (0..program.tile_count)
                .map(|tile| match builders.remove(&tile) {
                    Some(builder) => Ok(builder.finish(horizon)?),
                    None => Ok(inactive_exchange_row()),
                })
                .collect::<Result<Vec<_>, ipu_exchange::ExchangeError>>()?;
            Ok(PhysicalExchangePhase {
                id: phase.id,
                rows,
                event_cycles: horizon,
            })
        })
        .collect()
}

struct ScheduledTransfer<'a> {
    source: u16,
    destinations: &'a [(u16, u32)],
    source_address: u32,
    words: u32,
    schedule_offset: u32,
}

fn append_transfer(
    topology: &Topology,
    transfer: ScheduledTransfer<'_>,
    horizon: &mut u32,
    builders: &mut BTreeMap<u16, PlanProgramBuilder>,
) -> Result<u32, ExchangeLoweringError> {
    let ScheduledTransfer {
        source,
        destinations,
        source_address,
        words,
        schedule_offset,
    } = transfer;
    if words == 0 {
        return Err(ExchangeLoweringError::UnalignedPayload);
    }
    let tiles = destinations.iter().map(|entry| entry.0).collect::<Vec<_>>();
    let mut plan = if tiles.len() == 1 && schedule_offset == 0 {
        let point = topology.point_to_point(source, tiles[0], words)?;
        MulticastPlan {
            sender: point.sender,
            receivers: vec![finalize_point_receiver(
                &point.receiver,
                topology.physical(source)?,
            )?],
        }
    } else {
        let mut plan = topology.multicast(source, &tiles, words, 0)?;
        offset_plan(&mut plan.sender, schedule_offset)?;
        for receiver in &mut plan.receivers {
            offset_plan(receiver, schedule_offset)?;
        }
        plan
    };
    patch_sender_address(&mut plan.sender, source_address)?;
    for (row, (_, address)) in plan.receivers.iter_mut().zip(destinations) {
        patch_receiver_address(row, *address)?;
    }
    builders
        .entry(source)
        .or_default()
        .append_scheduled_row(&plan.sender)?;
    for (&tile, row) in tiles.iter().zip(&plan.receivers) {
        builders
            .entry(tile)
            .or_default()
            .append_scheduled_row(row)?;
    }
    let transfer_end = std::iter::once(&plan.sender)
        .chain(&plan.receivers)
        .map(|row| plan_event_cycles(row))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(schedule_offset);
    *horizon = (*horizon).max(transfer_end);
    Ok(transfer_end)
}

pub fn inactive_exchange_row() -> Vec<u32> {
    let mut row = vec![0; PLAN_WORDS];
    row[0] = SANS_INACTIVE_INSTRUCTION;
    row[1] = SYNC_ANS_INSTRUCTION;
    row[2] = RETURN_M10_INSTRUCTION;
    row
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ComputeGraph, Ipu21CostModel, Layout, PipelineConfig, Precision, TensorFormat, lower,
        lower_to_tiles, place,
    };

    #[test]
    fn randomized_gemm_exchanges_produce_one_executable_row_per_tile() {
        let mut random = fastrand::Rng::with_seed(0x6578_6368);
        for _ in 0..32 {
            let tiles = 1_u16 << random.u32(1..=3);
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
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();
            let placement = place(&low).unwrap();
            let phases = lower_exchanges(&low, &placement, &Topology::c600()).unwrap();
            assert_eq!(phases.len(), low.exchange_phases.len());
            for phase in phases {
                assert_eq!(phase.rows.len(), usize::from(tiles));
                assert!(phase.event_cycles != 0);
                for row in phase.rows {
                    if row[0] == SANS_INACTIVE_INSTRUCTION {
                        assert_eq!(row[2], RETURN_M10_INSTRUCTION);
                    } else {
                        assert_eq!(row[0], ipu_exchange::SYNC_SUPERVISOR_INSTRUCTION);
                        assert_eq!(row.last(), Some(&RETURN_M10_INSTRUCTION));
                    }
                }
            }
        }
    }
}
