//! Instrument selected tile work and describe it for the execution profiler.
use super::*;

pub(super) fn profile_binding(
    program: &LowProgram,
    physical_to_logical: &[u16],
    address: u32,
) -> PackageBuildResult<Binding> {
    let mut file_offset = 0u64;
    let mut sample_count = 0u32;
    let slices = physical_to_logical
        .iter()
        .enumerate()
        .filter_map(|(physical, &logical)| {
            let steps = if logical < program.tile_count {
                profile_step_count(program, &program.tiles[usize::from(logical)])
            } else {
                inactive_profile_work(program).len()
            };
            (steps != 0).then_some((physical, steps))
        })
        .map(|(physical, steps)| {
            let samples = u32::try_from(steps + 1)?;
            let size = u64::from(samples)
                .checked_mul(4)
                .ok_or_else(|| invalid("profile binding size overflow"))?;
            let slice = RegionSlice {
                tile: u32::try_from(physical)?,
                tile_address: address,
                file_offset,
                size,
            };
            file_offset = file_offset
                .checked_add(size)
                .ok_or_else(|| invalid("profile binding offset overflow"))?;
            sample_count = sample_count
                .checked_add(samples)
                .ok_or_else(|| invalid("profile binding sample count overflow"))?;
            Ok(slice)
        })
        .collect::<PackageBuildResult<Vec<_>>>()?;
    Ok(Binding {
        name: PROFILE_CYCLES_BINDING.into(),
        dtype: "u32".into(),
        shape: vec![sample_count],
        slices,
    })
}

pub(super) fn instrument_profile(
    program: &LowProgram,
    exchanges: &[crate::PhysicalExchangePhase],
    logical_tile: u16,
    physical_tile: u32,
    tile_program: &mut crate::TileProgram,
    address: u32,
) -> PackageBuildResult<TileProfilePlan> {
    let mut plans = Vec::with_capacity(tile_program.steps.len());
    if logical_tile < program.tile_count {
        let schedule = program
            .work(&program.tiles[usize::from(logical_tile)])
            .collect::<Vec<_>>();
        if schedule.len() != tile_program.steps.len() {
            return Err(invalid("tile profile work does not match finalized steps"));
        }
        let same_call = std::iter::once(false).chain(tile_program.steps.windows(2).map(|pair| {
            matches!((&pair[0], &pair[1]), (crate::TileStep::Compute(a), crate::TileStep::Compute(b))
                if a.symbol == b.symbol && a.arguments == b.arguments)
        })).collect::<Vec<_>>();
        for (index, (&work, step)) in schedule.iter().zip(&mut tile_program.steps).enumerate() {
            if index != 0 && profile_work_can_merge(schedule[index - 1], work) {
                continue;
            }
            let following = schedule[index + 1..].iter().find_map(|work| match work {
                crate::TileWorkRef::Kernel(run) => Some(&run.provenance),
                crate::TileWorkRef::Repeat(repeat) => Some(&repeat.provenance),
                crate::TileWorkRef::Exchange(_)
                | crate::TileWorkRef::LocalCopy(_)
                | crate::TileWorkRef::Checkpoint(..) => None,
            });
            step_profile(step).before = Some(profile_address(address, plans.len())?);
            let mut description = profile_step(
                program,
                exchanges,
                logical_tile,
                index,
                work,
                step,
                following,
            )?;
            let invocations = schedule[index + 1..]
                .iter()
                .take_while(|&&next| profile_work_can_merge(work, next))
                .count()
                + 1;
            description.metadata.push(ProfileMetadata {
                name: "invocations".into(),
                value: invocations.to_string(),
            });
            if let crate::TileStep::Compute(call) = step {
                // Rendering may group calls with different sizes. Only groups
                // with identical executable ABIs can be averaged for costing.
                description.metadata.extend([
                    ProfileMetadata {
                        name: "uniformInvocations".into(),
                        value: same_call[index + 1..index + invocations]
                            .iter()
                            .all(|same| *same)
                            .to_string(),
                    },
                    ProfileMetadata {
                        name: "arguments".into(),
                        value: format!("{:?}", call.arguments),
                    },
                ]);
            }
            description.local_index = u32::try_from(plans.len())?;
            plans.push(description);
        }
    } else {
        let schedule = inactive_profile_work(program);
        if schedule.len() != tile_program.steps.len() {
            return Err(invalid(
                "inactive tile profile does not match finalized steps",
            ));
        }
        for (index, (work, step)) in schedule
            .into_iter()
            .zip(&mut tile_program.steps)
            .enumerate()
        {
            if let (crate::TileWorkRef::Checkpoint(operation, _), crate::TileStep::Checkpoint(_)) =
                (work, &*step)
            {
                step_profile(step).before = Some(profile_address(address, index)?);
                plans.push(ProfileStep {
                    local_index: u32::try_from(index)?,
                    phase: u32::try_from(index)?,
                    epoch: 0,
                    operation: format!("operation.{}", operation.index()),
                    kind: ProfileStepKind::Idle,
                    kernel: "diagnostic-checkpoint".into(),
                    metadata: Vec::new(),
                    exchange_activities: Vec::new(),
                    exchange_event_cycles: 0,
                });
                continue;
            }
            let (phase, provenance) = match (work, &*step) {
                (crate::TileWorkRef::Exchange(id), crate::TileStep::Exchange(_)) => {
                    let phase = &program.exchange_phases[id.index() as usize];
                    (0x8000_0000 | id.index(), &phase.provenance)
                }
                (crate::TileWorkRef::Repeat(repeat), crate::TileStep::Repeat(_)) => {
                    (u32::try_from(index)?, &repeat.provenance)
                }
                _ => return Err(invalid("inactive tile contains executable work")),
            };
            step_profile(step).before = Some(profile_address(address, index)?);
            plans.push(inactive_tile_description(index, phase, provenance)?);
        }
    }
    if let Some(last) = tile_program.steps.last_mut() {
        step_profile(last).after = Some(profile_address(address, plans.len())?);
    }
    Ok(TileProfilePlan {
        physical_tile,
        steps: plans,
    })
}

fn inactive_profile_work(program: &LowProgram) -> Vec<crate::TileWorkRef<'_>> {
    program
        .tiles
        .first()
        .into_iter()
        .flat_map(|tile| program.work(tile))
        .filter(|work| {
            matches!(
                work,
                crate::TileWorkRef::Exchange(_)
                    | crate::TileWorkRef::Repeat(_)
                    | crate::TileWorkRef::Checkpoint(..)
            )
        })
        .collect()
}

pub(super) fn profile_step_count(program: &LowProgram, tile: &crate::TileWorkList) -> usize {
    let mut previous = None;
    let mut count = 0;
    for work in program.work(tile) {
        if previous.is_none_or(|previous| !profile_work_can_merge(previous, work)) {
            count += 1;
        }
        previous = Some(work);
    }
    count
}

fn profile_work_can_merge(
    previous: crate::TileWorkRef<'_>,
    current: crate::TileWorkRef<'_>,
) -> bool {
    matches!(
        (previous, current),
        (crate::TileWorkRef::Kernel(previous), crate::TileWorkRef::Kernel(current))
            if previous.kernel == current.kernel && previous.provenance == current.provenance
    ) || matches!(
        (previous, current),
        (
            crate::TileWorkRef::LocalCopy(previous),
            crate::TileWorkRef::LocalCopy(current)
        ) if previous.bytes == current.bytes && previous.pattern == current.pattern
    )
}

#[allow(clippy::too_many_arguments)]
fn profile_step(
    program: &LowProgram,
    exchanges: &[crate::PhysicalExchangePhase],
    logical_tile: u16,
    index: usize,
    work: crate::TileWorkRef<'_>,
    step: &mut crate::TileStep,
    following: Option<&crate::WorkProvenance>,
) -> PackageBuildResult<ProfileStep> {
    match (work, step) {
        (crate::TileWorkRef::Exchange(id), crate::TileStep::Exchange(exchange)) => {
            let phase = &program.exchange_phases[id.index() as usize];
            if !exchange.active {
                exchange_synchronization_description(
                    index,
                    0x8000_0000 | id.index(),
                    &phase.provenance,
                )
            } else {
                let mut description = profile_description(
                    index,
                    0x8000_0000 | id.index(),
                    &phase.provenance,
                    ProfileStepKind::Exchange,
                    "exchange",
                )?;
                let physical = exchanges
                    .get(id.index() as usize)
                    .ok_or_else(|| invalid("profile exchange phase is missing"))?;
                description.exchange_activities = physical
                    .activities
                    .get(usize::from(logical_tile))
                    .ok_or_else(|| invalid("profile exchange tile is missing"))?
                    .iter()
                    .map(|activity| ProfileExchangeActivity {
                        fanout: activity.fanout,
                        paired: activity.paired,
                        kind: match activity.kind {
                            crate::ExchangeActivityKind::Send => ProfileExchangeActivityKind::Send,
                            crate::ExchangeActivityKind::Receive => {
                                ProfileExchangeActivityKind::Receive
                            }
                            crate::ExchangeActivityKind::PartnerBusy => {
                                ProfileExchangeActivityKind::PartnerBusy
                            }
                        },
                        start_cycle: activity.start_cycle,
                        end_cycle: activity.end_cycle,
                    })
                    .collect();
                description.exchange_event_cycles = physical.event_cycles;
                Ok(description)
            }
        }
        (crate::TileWorkRef::Kernel(run), crate::TileStep::Compute(compute)) => {
            let mut description = profile_description(
                index,
                u32::try_from(index)?,
                &run.provenance,
                ProfileStepKind::Compute,
                &compute.symbol,
            )?;
            description.metadata.push(ProfileMetadata {
                name: "kernelSpec".into(),
                value: format!("{:?}", run.kernel),
            });
            description.metadata.push(ProfileMetadata {
                name: "outputElements".into(),
                value: view_logical_elements(&run.output).to_string(),
            });
            for (operand, input) in run.inputs.iter().enumerate() {
                description.metadata.push(ProfileMetadata {
                    name: format!("input{operand}Elements"),
                    value: input
                        .views
                        .iter()
                        .map(view_logical_elements)
                        .sum::<u64>()
                        .to_string(),
                });
            }
            Ok(description)
        }
        (crate::TileWorkRef::LocalCopy(copy), crate::TileStep::Compute(compute)) => {
            if let Some(provenance) = following {
                let mut description = profile_description(
                    index,
                    u32::try_from(index)?,
                    provenance,
                    ProfileStepKind::Compute,
                    &compute.symbol,
                )?;
                description.metadata[0].value = "LocalCopy".into();
                description.metadata.extend([
                    ProfileMetadata {
                        name: "bytes".into(),
                        value: copy.bytes.to_string(),
                    },
                    ProfileMetadata {
                        name: "pattern".into(),
                        value: format!("{:?}", copy.pattern),
                    },
                ]);
                Ok(description)
            } else {
                Ok(ProfileStep {
                    local_index: u32::try_from(index)?,
                    phase: u32::try_from(index)?,
                    epoch: 0,
                    operation: String::new(),
                    kind: ProfileStepKind::Compute,
                    kernel: compute.symbol.clone(),
                    metadata: vec![
                        ProfileMetadata {
                            name: "reason".into(),
                            value: "LocalCopy".into(),
                        },
                        ProfileMetadata {
                            name: "bytes".into(),
                            value: copy.bytes.to_string(),
                        },
                        ProfileMetadata {
                            name: "pattern".into(),
                            value: format!("{:?}", copy.pattern),
                        },
                    ],
                    exchange_activities: Vec::new(),
                    exchange_event_cycles: 0,
                })
            }
        }
        (crate::TileWorkRef::Repeat(repeat), crate::TileStep::Repeat(_)) => profile_description(
            index,
            u32::try_from(index)?,
            &repeat.provenance,
            ProfileStepKind::Compute,
            "repeat",
        ),
        (crate::TileWorkRef::Checkpoint(operation, _), crate::TileStep::Checkpoint(_)) => {
            Ok(ProfileStep {
                local_index: u32::try_from(index)?,
                phase: u32::try_from(index)?,
                epoch: 0,
                operation: format!("operation.{}", operation.index()),
                kind: ProfileStepKind::Synchronization,
                kernel: "diagnostic-checkpoint".into(),
                metadata: Vec::new(),
                exchange_activities: Vec::new(),
                exchange_event_cycles: 0,
            })
        }
        _ => Err(invalid(
            "tile profile work kind does not match finalized step",
        )),
    }
}

fn view_logical_elements(view: &crate::ShardView) -> u64 {
    view.extents.iter().fold(1u64, |elements, extent| {
        elements.saturating_mul(u64::from(extent.logical_end.saturating_sub(extent.start)))
    })
}

fn exchange_synchronization_description(
    index: usize,
    phase: u32,
    provenance: &crate::WorkProvenance,
) -> PackageBuildResult<ProfileStep> {
    let mut description = profile_description(
        index,
        phase,
        provenance,
        ProfileStepKind::Synchronization,
        "sync",
    )?;
    description.metadata[0].value = "ExchangeBarrier".into();
    Ok(description)
}

fn inactive_tile_description(
    index: usize,
    phase: u32,
    provenance: &crate::WorkProvenance,
) -> PackageBuildResult<ProfileStep> {
    let mut description =
        profile_description(index, phase, provenance, ProfileStepKind::Idle, "idle")?;
    description.metadata[0].value = "InactiveTile".into();
    Ok(description)
}

fn profile_description(
    index: usize,
    phase: u32,
    provenance: &crate::WorkProvenance,
    kind: ProfileStepKind,
    kernel: &str,
) -> PackageBuildResult<ProfileStep> {
    let mut metadata = vec![ProfileMetadata {
        name: "reason".into(),
        value: format!("{:?}", provenance.reason),
    }];
    if let Some(value) = provenance.value {
        metadata.push(ProfileMetadata {
            name: "value".into(),
            value: value.index().to_string(),
        });
    }
    Ok(ProfileStep {
        local_index: u32::try_from(index)?,
        phase,
        epoch: 0,
        operation: provenance
            .operation
            .map(|operation| format!("operation.{}", operation.index()))
            .unwrap_or_default(),
        kind,
        kernel: kernel.into(),
        metadata,
        exchange_activities: Vec::new(),
        exchange_event_cycles: 0,
    })
}

fn step_profile(step: &mut crate::TileStep) -> &mut crate::StepProfile {
    match step {
        crate::TileStep::Exchange(exchange) => &mut exchange.profile,
        crate::TileStep::Compute(compute) => &mut compute.profile,
        crate::TileStep::Repeat(repeat) => &mut repeat.profile,
        crate::TileStep::Checkpoint(checkpoint) => &mut checkpoint.profile,
    }
}

fn profile_address(base: u32, index: usize) -> PackageBuildResult<u32> {
    base.checked_add(
        u32::try_from(index)?
            .checked_mul(4)
            .ok_or_else(|| invalid("profile address overflow"))?,
    )
    .ok_or_else(|| invalid("profile address overflow"))
}
