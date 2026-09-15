//! Bounded SRAM placement search after package support storage is reserved.

use crate::low::LowProgram;
use crate::package::PackageBuildResult;

pub(super) struct AddressProposal {
    pub placement: crate::Placement,
    pub baseline_score: u128,
    pub score: u128,
    pub offset: u32,
}

pub(super) fn propose_exchange_placement(
    program: &LowProgram,
    available_ranges: &[(u32, u32)],
    auxiliary: &[Vec<crate::place::AuxiliaryRequest>],
    baseline: &crate::Placement,
) -> PackageBuildResult<Option<AddressProposal>> {
    if program.exchange_phases.is_empty() {
        return Ok(None);
    }
    let conflicts = crate::place::ExchangeConflicts::new(program)?;
    let baseline_score = conflicts.score(baseline);
    if baseline_score == 0 {
        return Ok(None);
    }
    let mut best: Option<(u128, u32, crate::Placement)> = None;
    for offset in (4096..ipu_target::ipu21::memory::IPU21_INTERLEAVED_ELEMENT_SIZE).step_by(4096) {
        let candidate = match crate::place::place_with_auxiliary(
            program,
            available_ranges,
            offset,
            auxiliary,
        ) {
            Ok(candidate) => candidate,
            Err(crate::PlacementError::OutOfMemory { .. }) => continue,
            Err(error) => return Err(error.into()),
        };
        let score = conflicts.score(&candidate);
        tracing::debug!(offset, score = %score, "scored exchange placement");
        if score < baseline_score
            && best.as_ref().is_none_or(|(best_score, best_offset, _)| {
                (score, offset) < (*best_score, *best_offset)
            })
        {
            best = Some((score, offset, candidate));
        }
    }
    Ok(best.map(|(score, offset, placement)| AddressProposal {
        placement,
        score,
        offset,
        baseline_score,
    }))
}

pub(super) fn exchange_cycles(
    program: &LowProgram,
    phases: &[crate::PhysicalExchangePhase],
) -> u64 {
    let multiplicities = program.exchange_multiplicities();
    phases
        .iter()
        .map(|phase| {
            u64::from(phase.event_cycles).saturating_mul(multiplicities[phase.id.index() as usize])
        })
        .fold(0, u64::saturating_add)
}
