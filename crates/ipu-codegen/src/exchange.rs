//! Physical exchange programs generated from logical shard transfers.

use crate::{ExchangePhaseId, LowProgram, Placement, ShardDefinition, view_byte_spans};
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
    #[error("exchange destination is not an exchange-copy shard")]
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
                let source_base = placement
                    .shard_addresses
                    .get(&source.id)
                    .copied()
                    .ok_or(ExchangeLoweringError::UnplacedShard)?;
                let destinations = transfer
                    .destinations
                    .iter()
                    .map(|id| {
                        let shard = &program.shards[id.index() as usize];
                        if !matches!(shard.definition, ShardDefinition::ExchangeStaging) {
                            return Err(ExchangeLoweringError::InvalidDestination);
                        }
                        Ok((
                            shard.tile,
                            placement
                                .shard_addresses
                                .get(id)
                                .copied()
                                .ok_or(ExchangeLoweringError::UnplacedShard)?,
                            crate::shard_storage_bytes(shard)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
                let expected = destinations
                    .first()
                    .map(|entry| entry.2)
                    .ok_or(ExchangeLoweringError::SizeMismatch)?;
                if destinations.iter().any(|entry| entry.2 != expected) {
                    return Err(ExchangeLoweringError::SizeMismatch);
                }
                let mut packed_offset = 0u32;
                for span in view_byte_spans(source, &transfer.source)? {
                    if span.bytes == 0 || span.bytes & 3 != 0 || span.offset & 3 != 0 {
                        return Err(ExchangeLoweringError::UnalignedPayload);
                    }
                    let mut chunk_offset = 0u32;
                    while chunk_offset < span.bytes {
                        let chunk_bytes = (span.bytes - chunk_offset).min(
                            MAX_TRANSFER_WORDS
                                .checked_mul(4)
                                .ok_or(ExchangeLoweringError::Overflow)?,
                        );
                        let source_address = source_base
                            .checked_add(span.offset)
                            .and_then(|address| address.checked_add(chunk_offset))
                            .ok_or(ExchangeLoweringError::Overflow)?;
                        let destination_entries = destinations
                            .iter()
                            .map(|(tile, base, _)| {
                                Ok((
                                    *tile,
                                    base.checked_add(packed_offset)
                                        .and_then(|address| address.checked_add(chunk_offset))
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
                        chunk_offset = chunk_offset
                            .checked_add(chunk_bytes)
                            .ok_or(ExchangeLoweringError::Overflow)?;
                    }
                    packed_offset = packed_offset
                        .checked_add(span.bytes)
                        .ok_or(ExchangeLoweringError::Overflow)?;
                }
                if packed_offset != expected {
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
