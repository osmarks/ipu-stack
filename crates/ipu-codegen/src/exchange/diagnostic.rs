//! Exchange row inspection, endpoint pressure and critical-chain reporting.
pub use super::program::diagnostic::*;
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
    pub program: crate::exchange::diagnostic::PlanProgramDiagnostic,
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
    let row_words =
        u32::try_from(program.words().len()).map_err(|_| ExchangeLoweringError::Overflow)?;
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
        program: crate::exchange::diagnostic::diagnose_plan_program(
            program.words(),
            Some(row_address),
        )?,
        activities,
    })
}
