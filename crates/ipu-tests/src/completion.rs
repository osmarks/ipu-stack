//! Terminal-state checks shared by model and standalone kernel diagnostics.
use anyhow::{Result, bail};
use ipu_package::Application;
use ipu_runtime::Runtime;
use std::time::{Duration, Instant};

pub fn diagnose_completion(
    runtime: &Runtime,
    application: &Application,
    timeout: Duration,
) -> Result<()> {
    let completion_pc = application
        .debug_symbols
        .iter()
        .find(|symbol| symbol.name == ipu_codegen::COMPLETED_SYMBOL)
        .map(|symbol| symbol.address);
    let deadline = Instant::now() + timeout;
    let mut completed = std::collections::BTreeSet::new();
    loop {
        for tile in &application.tiles {
            let physical = u16::try_from(tile.physical_tile)?;
            if completed.contains(&physical) {
                continue;
            }
            let device = runtime.device();
            let state = device.tile_context_state(physical, 0)?;
            let terminal_fault = if state == 3 && completion_pc.is_some() {
                // Branching to zero in the runtime's completion routine can
                // remain visible as INVALID_PC after the final host exchange.
                // Do not confuse a fault elsewhere with successful completion.
                let exception = ipu_driver::TileException::from_status(
                    device.read_tile_context_status(physical, 0)?,
                );
                exception == ipu_driver::TileException::InvalidProgramCounter
                    && Some(device.read_tile_program_counter(physical, 0)?) == completion_pc
                    && device.read_tile_word(physical, tile.diagnostic_address)? == 1
            } else {
                false
            };
            if state == 0 || terminal_fault {
                completed.insert(physical);
            }
        }
        if completed.len() == application.tiles.len() {
            break;
        }
        if Instant::now() >= deadline {
            bail!(
                "supervisors did not complete: {}",
                summarize_states(&supervisor_states(runtime, application)?)
            );
        }
        std::thread::sleep(Duration::from_micros(100));
    }

    let mut active_workers = Vec::new();
    for tile in &application.tiles {
        let physical = u16::try_from(tile.physical_tile)?;
        for context in 1..=6 {
            let state = runtime.device().tile_context_state(physical, context)?;
            if state != 0 && active_workers.len() < 16 {
                active_workers.push((physical, context, state));
            }
        }
    }
    if !active_workers.is_empty() {
        bail!("workers did not halt: {active_workers:?}");
    }
    Ok(())
}

pub fn supervisor_states(runtime: &Runtime, application: &Application) -> Result<Vec<(u16, u32)>> {
    application
        .tiles
        .iter()
        .map(|tile| {
            let physical = u16::try_from(tile.physical_tile)?;
            Ok((physical, runtime.device().tile_context_state(physical, 0)?))
        })
        .collect()
}

pub fn summarize_states(states: &[(u16, u32)]) -> String {
    let mut counts = [0usize; 4];
    let mut unexpected = Vec::new();
    for &(tile, state) in states {
        if let Some(count) = counts.get_mut(state as usize) {
            *count += 1;
        }
        if state != 0 && unexpected.len() < 16 {
            unexpected.push((tile, state));
        }
    }
    format!("counts={counts:?} firstUnexpected={unexpected:?}")
}
