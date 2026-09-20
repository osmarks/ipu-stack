//! Instrument selected tile work and describe it for the execution profiler.
use super::*;

pub(super) fn profile_binding(
    program: &LowGraph,
    physical_to_logical: &[u16],
    addresses: &[u32],
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
            (steps != 0).then_some((physical, logical, steps))
        })
        .map(|(physical, logical, steps)| {
            let samples = u32::try_from(steps + 1)?;
            let size = u64::from(samples)
                .checked_mul(4)
                .ok_or_else(|| invalid("profile binding size overflow"))?;
            let slice = RegionSlice {
                tile: u32::try_from(physical)?,
                tile_address: addresses[usize::from(logical)],
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
    program: &LowGraph,
    exchanges: &[crate::PhysicalExchangePhase],
    logical_tile: u16,
    physical_tile: u32,
    tile_program: &mut crate::TileProgram,
    address: u32,
) -> PackageBuildResult<TileProfilePlan> {
    let mut plans = Vec::with_capacity(tile_program.steps.len());
    if logical_tile < program.tile_count {
        instrument_active_steps(
            program,
            exchanges,
            logical_tile,
            &program.tiles[usize::from(logical_tile)].work,
            &mut tile_program.steps,
            address,
            &mut plans,
        )?;
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
            if let (
                crate::BlockOperation::Checkpoint(operation, _),
                crate::TileStep::Checkpoint(_),
            ) = (work, &*step)
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
                (crate::BlockOperation::Exchange(id), crate::TileStep::Exchange(_)) => {
                    let phase = &program.exchange_phases[id.index() as usize];
                    (0x8000_0000 | id.index(), &phase.provenance)
                }
                (crate::BlockOperation::Repeat(repeat), crate::TileStep::Repeat(_)) => (
                    u32::try_from(index)?,
                    &program.repeat_runs[*repeat].provenance,
                ),
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

#[allow(clippy::too_many_arguments)]
fn instrument_active_steps(
    program: &LowGraph,
    exchanges: &[crate::PhysicalExchangePhase],
    logical_tile: u16,
    schedule: &[crate::BlockOperation<usize>],
    steps: &mut [crate::TileStep],
    address: u32,
    plans: &mut Vec<ProfileStep>,
) -> PackageBuildResult<()> {
    if schedule.len() != steps.len() {
        return Err(invalid(
            "repeat profile work does not match finalized steps",
        ));
    }
    let same_call = std::iter::once(false).chain(steps.windows(2).map(|pair| {
            matches!((&pair[0], &pair[1]), (crate::TileStep::Compute(a), crate::TileStep::Compute(b))
                if a.symbol == b.symbol && a.arguments == b.arguments)
        })).collect::<Vec<_>>();
    for (index, (work, step)) in schedule.iter().zip(steps.iter_mut()).enumerate() {
        if let (crate::BlockOperation::Repeat(repeat), crate::TileStep::Repeat(finalized)) =
            (work, &mut *step)
            && let repeat = &program.repeat_runs[*repeat]
            && !repeat.body.work.is_empty()
        {
            let first = plans.len();
            instrument_active_steps(
                program,
                exchanges,
                logical_tile,
                &repeat.body.work,
                &mut finalized.body,
                address,
                plans,
            )?;
            let epoch = plans[..first]
                .iter()
                .map(|plan| plan.epoch)
                .max()
                .unwrap_or(0)
                + 1;
            for description in &mut plans[first..] {
                description.epoch = epoch;
            }
            if repeat.count > 1 {
                let mut remainder = profile_description(
                    plans.len(),
                    u32::try_from(index)?,
                    &repeat.provenance,
                    ProfileStepKind::Compute,
                    "repeat-remainder",
                )?;
                remainder.metadata.push(ProfileMetadata {
                    name: "iterations".into(),
                    value: (repeat.count - 1).to_string(),
                });
                plans.push(remainder);
            }
            continue;
        }
        if index != 0 && profile_work_can_merge(program, &schedule[index - 1], work) {
            continue;
        }
        let following = schedule[index + 1..].iter().find_map(|work| match work {
            crate::BlockOperation::Compute { run, .. } => {
                Some(&program.kernel_runs[run.0 as usize].provenance)
            }
            crate::BlockOperation::Repeat(repeat) => Some(&program.repeat_runs[*repeat].provenance),
            crate::BlockOperation::Exchange(_)
            | crate::BlockOperation::Copy { .. }
            | crate::BlockOperation::Checkpoint(..) => None,
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
            .take_while(|&next| profile_work_can_merge(program, work, next))
            .count()
            + 1;
        super::profile_work::append_work_estimate(
            &mut description.metadata,
            program,
            &schedule[index..index + invocations],
        );
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
    if let Some(last) = steps.last_mut() {
        step_profile(last).after = Some(profile_address(address, plans.len())?);
    }
    Ok(())
}

pub(super) fn inactive_profile_work(program: &LowGraph) -> Vec<&crate::BlockOperation<usize>> {
    program
        .tiles
        .first()
        .into_iter()
        .flat_map(|tile| tile.work.iter())
        .filter(|work| {
            matches!(
                work,
                crate::BlockOperation::Exchange(_)
                    | crate::BlockOperation::Repeat(_)
                    | crate::BlockOperation::Checkpoint(..)
            )
        })
        .collect()
}

pub(super) fn profile_step_count(program: &LowGraph, tile: &crate::TileWorkList) -> usize {
    let mut previous = None;
    let mut count = 0;
    for work in tile.work.iter() {
        if previous.is_none_or(|previous| !profile_work_can_merge(program, previous, work)) {
            count += match work {
                crate::BlockOperation::Repeat(repeat) => {
                    let repeat = &program.repeat_runs[*repeat];
                    if repeat.body.work.is_empty() {
                        1 // The loop executes, but has no first-iteration body samples.
                    } else {
                        profile_step_count(program, &repeat.body) + usize::from(repeat.count > 1)
                    }
                }
                _ => 1,
            };
        }
        previous = Some(work);
    }
    count
}

fn profile_work_can_merge(
    program: &LowGraph,
    previous: &crate::BlockOperation<usize>,
    current: &crate::BlockOperation<usize>,
) -> bool {
    match (previous, current) {
        (
            crate::BlockOperation::Compute { run: a, .. },
            crate::BlockOperation::Compute { run: b, .. },
        ) => {
            let (a, b) = (
                &program.kernel_runs[a.0 as usize],
                &program.kernel_runs[b.0 as usize],
            );
            a.kernel == b.kernel && a.provenance == b.provenance
        }
        (
            crate::BlockOperation::Copy { copy: a, .. },
            crate::BlockOperation::Copy { copy: b, .. },
        ) => {
            let (a, b) = (
                &program.local_copies[a.0 as usize],
                &program.local_copies[b.0 as usize],
            );
            a.symbol() == b.symbol()
                && a.movement().bytes == b.movement().bytes
                && a.movement().pattern == b.movement().pattern
        }
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn profile_step(
    program: &LowGraph,
    exchanges: &[crate::PhysicalExchangePhase],
    logical_tile: u16,
    index: usize,
    work: &crate::BlockOperation<usize>,
    step: &mut crate::TileStep,
    following: Option<&crate::WorkProvenance>,
) -> PackageBuildResult<ProfileStep> {
    match (work, step) {
        (crate::BlockOperation::Exchange(id), crate::TileStep::Exchange(exchange)) => {
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
        (crate::BlockOperation::Compute { run, .. }, crate::TileStep::Compute(compute)) => {
            let run = &program.kernel_runs[run.0 as usize];
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
                value: view_logical_elements(&run.outputs[0]).to_string(),
            });
            for (operand, input) in run.inputs.iter().enumerate() {
                description.metadata.push(ProfileMetadata {
                    name: format!("input{operand}Elements"),
                    value: view_logical_elements(input).to_string(),
                });
            }
            Ok(description)
        }
        (crate::BlockOperation::Copy { copy, .. }, crate::TileStep::Compute(compute)) => {
            let copy = program.local_copies[copy.0 as usize].movement();
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
        (crate::BlockOperation::Repeat(repeat), crate::TileStep::Repeat(_)) => {
            let repeat = &program.repeat_runs[*repeat];
            let mut description = profile_description(
                index,
                u32::try_from(index)?,
                &repeat.provenance,
                ProfileStepKind::Idle,
                "repeat",
            )?;
            description.metadata.push(ProfileMetadata {
                name: "iterations".into(),
                value: repeat.count.to_string(),
            });
            Ok(description)
        }
        (crate::BlockOperation::Checkpoint(operation, _), crate::TileStep::Checkpoint(_)) => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn active_tiles_with_empty_repeat_bodies_have_complete_samples() {
        for count in [1, 3] {
            let mut graph = crate::HighGraph::new();
            let input = graph.host_input("input", [8, 16]).unwrap();
            let output = graph
                .repeat(count, [input], [], [], |body, args| {
                    Ok(vec![body.gelu(args.carried[0])?])
                })
                .unwrap()[0];
            graph.set_outputs([output]).unwrap();
            let config = crate::PipelineConfig::new(1).with_input(
                input,
                crate::TensorFormat {
                    precision: crate::Precision::F16,
                    layout: crate::Layout::row_sharded(1),
                },
            );
            let mut mid = crate::planner::test_support::lower(
                &graph,
                &config,
                &crate::estimate::Ipu21CostModel,
            )
            .unwrap();
            // A valid whole-device program may leave active tiles unused by a region.
            mid.tile_count = 2;
            let low = crate::lower_to_tiles(&crate::expand_tiles(&mid).unwrap(), false);
            // Readback is ordered physically, while allocation is indexed logically.
            // Include an inactive execution tile as well as both active tiles.
            let addresses = [0x58000, 0x60000, 0x68000];
            let binding = profile_binding(&low, &[2, 0, 1], &addresses).unwrap();
            for slice in &binding.slices {
                let logical = [2, 0, 1][slice.tile as usize];
                assert_eq!(slice.tile_address, addresses[logical]);
                let steps = if logical < 2 {
                    profile_step_count(&low, &low.tiles[logical])
                } else {
                    inactive_profile_work(&low).len()
                };
                assert_eq!(slice.size, (steps as u64 + 1) * 4);
            }
            assert_eq!(binding.slices.len(), 3);
            let placement = crate::place(&low).unwrap();
            let exchanges = crate::lower_exchanges(&low, &placement, &Topology::c600()).unwrap();
            let lowering =
                crate::TileProgramLowering::new(&low, &placement, &exchanges, 0x60000, 2, false)
                    .unwrap();
            for tile in 0..2 {
                let mut program = lowering.lower_tile(tile).unwrap();
                let expected_count = profile_step_count(&low, &low.tiles[tile as usize]);
                let base = 0x58000;
                let profile =
                    instrument_profile(&low, &exchanges, tile, u32::from(tile), &mut program, base)
                        .unwrap();
                assert_eq!(profile.steps.len(), expected_count);
                fn samples(steps: &mut [crate::TileStep], addresses: &mut BTreeSet<u32>) {
                    for step in steps {
                        let profile = *step_profile(step);
                        addresses.extend(profile.before);
                        addresses.extend(profile.after);
                        if let crate::TileStep::Repeat(repeat) = step {
                            samples(&mut repeat.body, addresses);
                        }
                    }
                }
                let mut written = BTreeSet::new();
                samples(&mut program.steps, &mut written);
                let expected = (0..=expected_count)
                    .map(|i| base + i as u32 * 4)
                    .collect::<BTreeSet<_>>();
                assert_eq!(written, expected, "count={count} tile={tile}");
                assert!(
                    expected_count > 0,
                    "an empty repeat still executes on the tile"
                );
            }
        }
    }

    #[test]
    fn repeat_profiles_first_iteration_and_aggregates_remainder() {
        let mut graph = crate::HighGraph::new();
        let input = graph.host_input("input", [8, 16]).unwrap();
        let output = graph
            .repeat(3, [input], [], [], |body, arguments| {
                let first = body.gelu(arguments.carried[0])?;
                Ok(vec![body.gelu(first)?])
            })
            .unwrap()[0];
        graph.set_outputs([output]).unwrap();
        let config = crate::PipelineConfig::new(1).with_input(
            input,
            crate::TensorFormat {
                precision: crate::Precision::F16,
                layout: crate::Layout::row_sharded(1),
            },
        );
        let mid =
            crate::planner::test_support::lower(&graph, &config, &crate::estimate::Ipu21CostModel)
                .unwrap();
        let low = crate::lower_to_tiles(&crate::expand_tiles(&mid).unwrap(), false);
        let placement = crate::place(&low).unwrap();
        let exchanges = crate::lower_exchanges(&low, &placement, &Topology::c600()).unwrap();
        let lowering =
            crate::TileProgramLowering::new(&low, &placement, &exchanges, 0x60000, 1, false)
                .unwrap();
        let mut program = lowering.lower_tile(0).unwrap();
        let count = profile_step_count(&low, &low.tiles[0]);
        let base = 0x58000;
        let profile = instrument_profile(&low, &exchanges, 0, 0, &mut program, base).unwrap();
        assert_eq!(profile.steps.len(), count);
        let epochs = profile
            .steps
            .iter()
            .map(|step| step.epoch)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(epochs, [0, 1].into_iter().collect());
        let crate::TileStep::Repeat(repeat) = &program.steps[0] else {
            panic!("loop was unrolled");
        };
        assert_eq!(repeat.count, 3);
        let mut body = repeat.body.clone();
        let body_end = step_profile(body.last_mut().unwrap()).after.unwrap();
        assert_eq!(body_end, base + u32::try_from((count - 1) * 4).unwrap());
        assert_eq!(repeat.profile.after, Some(base + count as u32 * 4));
        assert_eq!(profile.steps.last().unwrap().kernel, "repeat-remainder");
        assert!(
            profile
                .steps
                .last()
                .unwrap()
                .metadata
                .iter()
                .any(|entry| entry.name == "iterations" && entry.value == "2")
        );
    }
}
