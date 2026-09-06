//! Exchange row inspection, endpoint pressure and critical-chain reporting.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExchangeActivityDiagnostic {
    pub activity: ExchangeActivity,
    pub memory_elements: Vec<ExchangeMemoryElement>,
    pub conflicts_with_row: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExchangeTileDiagnostic {
    pub phase: ExchangePhaseId,
    pub tile: u16,
    pub row_address: u32,
    pub row_elements: Vec<ExchangeMemoryElement>,
    pub program: ipu_exchange::diagnostic::PlanProgramDiagnostic,
    pub activities: Vec<ExchangeActivityDiagnostic>,
}

pub fn diagnose_exchange_tile(
    phase: &PhysicalExchangePhase,
    tile: u16,
    row_address: u32,
) -> Result<ExchangeTileDiagnostic, ExchangeLoweringError> {
    let program = phase
        .programs
        .get(usize::from(tile))
        .ok_or(ExchangeLoweringError::DiagnosticTile(tile))?;
    let row_words = u32::try_from(program.len()).map_err(|_| ExchangeLoweringError::Overflow)?;
    let row_elements = effective_memory_elements(row_address, row_words);
    let activities = phase
        .activities
        .get(usize::from(tile))
        .ok_or(ExchangeLoweringError::DiagnosticTile(tile))?
        .iter()
        .copied()
        .map(|activity| {
            let memory_elements = effective_memory_elements(activity.address, activity.words);
            let conflicts_with_row = memory_elements
                .iter()
                .any(|element| row_elements.contains(element));
            ExchangeActivityDiagnostic {
                activity,
                memory_elements,
                conflicts_with_row,
            }
        })
        .collect();
    Ok(ExchangeTileDiagnostic {
        phase: phase.id,
        tile,
        row_address,
        row_elements,
        program: ipu_exchange::diagnostic::diagnose_plan_program(program, Some(row_address))?,
        activities,
    })
}

#[derive(Clone, Debug, Default)]
struct TilePressure {
    send_roles: u32,
    receive_roles: u32,
    send_words: u64,
    receive_words: u64,
    last_transfer: Option<usize>,
}

#[derive(Clone, Debug)]
struct ScheduledTransferDiagnostic {
    source: u16,
    source_address: u32,
    destinations: Vec<(u16, u32)>,
    words: u32,
    start: u32,
    end: u32,
    blocking_tile: u16,
    predecessor: Option<usize>,
}

pub(super) struct PhaseDiagnostics {
    tiles: Vec<TilePressure>,
    transfers: Vec<ScheduledTransferDiagnostic>,
    source_words: u64,
    destination_words: u64,
    multicast_chunks: usize,
    maximum_fanout: usize,
    pub(super) maximum_endpoint_roles: usize,
}

impl PhaseDiagnostics {
    pub(super) fn new(tile_count: u16) -> Self {
        Self {
            tiles: vec![TilePressure::default(); usize::from(tile_count)],
            transfers: Vec::new(),
            source_words: 0,
            destination_words: 0,
            multicast_chunks: 0,
            maximum_fanout: 0,
            maximum_endpoint_roles: 0,
        }
    }

    pub(super) fn record(
        &mut self,
        source: u16,
        source_address: u32,
        destinations: &[(u16, u32)],
        words: u32,
        start: u32,
        end: u32,
        blocking_tile: u16,
    ) {
        let id = self.transfers.len();
        let predecessor = self.tiles[usize::from(blocking_tile)].last_transfer;
        let source_pressure = &mut self.tiles[usize::from(source)];
        source_pressure.send_roles += 1;
        source_pressure.send_words += u64::from(words);
        source_pressure.last_transfer = Some(id);
        for &(tile, _) in destinations {
            let pressure = &mut self.tiles[usize::from(tile)];
            pressure.receive_roles += 1;
            pressure.receive_words += u64::from(words);
            pressure.last_transfer = Some(id);
        }
        self.source_words += u64::from(words);
        self.destination_words += u64::from(words) * destinations.len() as u64;
        self.multicast_chunks += usize::from(destinations.len() > 1);
        self.maximum_fanout = self.maximum_fanout.max(destinations.len());
        self.transfers.push(ScheduledTransferDiagnostic {
            source,
            source_address,
            destinations: destinations.to_vec(),
            words,
            start,
            end,
            blocking_tile,
            predecessor,
        });
    }

    pub(super) fn emit(
        &self,
        phase: u32,
        provenance: &crate::WorkProvenance,
        horizon: u32,
        tile_availability: &[TileAvailability],
        builder: &PhaseProgramBuilder,
    ) {
        let role_word_lower_bound = self
            .tiles
            .iter()
            .map(|tile| tile.send_words.max(tile.receive_words))
            .max()
            .unwrap_or(0);
        let mut busiest_tiles = self
            .tiles
            .iter()
            .enumerate()
            .filter(|(_, tile)| tile.send_roles != 0 || tile.receive_roles != 0)
            .map(|(tile, pressure)| {
                let encoded_end = builder.tile_event_cycles(tile as u16).unwrap_or(0);
                (
                    tile as u16,
                    pressure.send_roles,
                    pressure.receive_roles,
                    pressure.send_words,
                    pressure.receive_words,
                    tile_availability[tile].send,
                    tile_availability[tile].receive,
                    encoded_end,
                    horizon.saturating_sub(encoded_end),
                )
            })
            .collect::<Vec<_>>();
        busiest_tiles.sort_unstable_by_key(|tile| {
            (
                Reverse(tile.3 + tile.4),
                Reverse(tile.5.max(tile.6)),
                tile.0,
            )
        });
        busiest_tiles.truncate(8);

        let active_builders = builder.active_tile_count() as u64;
        let total_final_padding = (0..builder.tile_count())
            .filter_map(|tile| builder.tile_event_cycles(tile).ok())
            .filter(|cycles| *cycles != 0)
            .map(|cycles| u64::from(horizon.saturating_sub(cycles)))
            .sum::<u64>();
        let maximum_final_padding = (0..builder.tile_count())
            .filter_map(|tile| builder.tile_event_cycles(tile).ok())
            .filter(|cycles| *cycles != 0)
            .map(|cycles| horizon.saturating_sub(cycles))
            .max()
            .unwrap_or(0);
        let maximum_scheduled_wait = self
            .tiles
            .iter()
            .enumerate()
            .map(|(tile, pressure)| {
                let send_wait =
                    u64::from(tile_availability[tile].send).saturating_sub(pressure.send_words);
                let receive_wait = u64::from(tile_availability[tile].receive)
                    .saturating_sub(pressure.receive_words);
                send_wait.max(receive_wait)
            })
            .max()
            .unwrap_or(0);

        let critical_transfer = self
            .transfers
            .iter()
            .enumerate()
            .max_by_key(|(_, transfer)| transfer.end)
            .map(|(id, _)| id);
        let mut critical_chain = Vec::new();
        let mut cursor = critical_transfer;
        while let Some(id) = cursor {
            critical_chain.push(id);
            cursor = self.transfers[id].predecessor;
        }
        critical_chain.reverse();
        let critical_chain_length = critical_chain.len();
        let critical_chain_tail = critical_chain
            .iter()
            .rev()
            .take(12)
            .rev()
            .map(|&id| {
                let transfer = &self.transfers[id];
                (
                    id,
                    transfer.source,
                    transfer.source_address,
                    &transfer.destinations,
                    transfer.words,
                    transfer.start,
                    transfer.end,
                    transfer.blocking_tile,
                )
            })
            .collect::<Vec<_>>();

        tracing::info!(
            phase,
            ?provenance,
            scheduled_chunks = self.transfers.len(),
            multicast_chunks = self.multicast_chunks,
            maximum_fanout = self.maximum_fanout,
            maximum_endpoint_roles = self.maximum_endpoint_roles,
            source_words = self.source_words,
            destination_words = self.destination_words,
            role_word_lower_bound_cycles = role_word_lower_bound,
            scheduled_horizon_cycles = horizon,
            scheduler_excess_cycles = u64::from(horizon).saturating_sub(role_word_lower_bound),
            maximum_scheduled_wait_cycles = maximum_scheduled_wait,
            mean_final_padding_cycles = total_final_padding
                .checked_div(active_builders)
                .unwrap_or(0),
            maximum_final_padding_cycles = maximum_final_padding,
            critical_chain_length,
            ?critical_chain_tail,
            ?busiest_tiles,
            "exchange scheduler diagnostics"
        );
    }
}
