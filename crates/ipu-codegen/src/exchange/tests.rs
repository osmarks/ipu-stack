use super::matching::maximum_ready_matching;
use super::*;
use crate::estimate::Ipu21CostModel;
use crate::exchange::diagnostic::sender_address_instruction_groups;
use crate::exchange::patch_sender_instruction;
use crate::planner::test_support::lower;
use crate::{HighGraph, Layout, PipelineConfig, Precision, TensorFormat, lower_to_tiles, place};
use ipu_target::ipu21::instruction::RETURN_M10_INSTRUCTION;

#[test]
fn loopback_packet_boundaries_preserve_repeat_sources() {
    for source in [0, 745, 1238, 1471] {
        for width in [ExchangeItemWidth::Word32, ExchangeItemWidth::Paired64] {
            for count in 1..=64 {
                let mut destinations = vec![
                    source,
                    Topology::c600().paired_logical(source).unwrap(),
                    0,
                    1,
                    1470,
                    1471,
                ];
                destinations.sort_unstable();
                destinations.dedup();
                if width == ExchangeItemWidth::Paired64
                    && crate::exchange::paired_multicast(
                        &Topology::c600(),
                        source,
                        &destinations,
                        count,
                    )
                    .is_err()
                {
                    // Primitive paired controls already reject some short
                    // lengths; production width selection retains ordinary TX.
                    continue;
                }
                let problem: Vec<PendingTransfer> = vec![transfer(
                    source,
                    vec![0x65000, 0x75000],
                    destinations.iter().map(|&tile| (tile, 0x98000)).collect(),
                    count * width.item_words(),
                    width,
                )];
                let packets =
                    packet::split_self_receive_conflicts(&Topology::c600(), problem.clone())
                        .unwrap();
                let mut offset = 0;
                if source == 745 && width == ExchangeItemWidth::Word32 && count == 28 {
                    assert_eq!(
                        packets.len(),
                        2,
                        "the pretrained-plan regression must split"
                    );
                    let mut cache = ExchangeScheduleCache::default();
                    let (selected, run) = select_exchange_schedule(
                        1472,
                        &problem,
                        std::num::NonZeroU32::new(1024),
                        &mut cache,
                    )
                    .unwrap();
                    validate_exchange_schedule(1472, &selected, &run).unwrap();
                    let mut relocated = problem.clone();
                    for address in &mut relocated[0].source_addresses {
                        *address += 0x100;
                    }
                    let (selected, run) = select_exchange_schedule(
                        1472,
                        &relocated,
                        std::num::NonZeroU32::new(1024),
                        &mut cache,
                    )
                    .unwrap();

                    validate_exchange_schedule(1472, &selected, &run).unwrap();
                }
                for p in &packets {
                    assert_eq!(p.source_addresses, vec![0x65000 + offset, 0x75000 + offset]);
                    assert_eq!(p.source_offset, offset);
                    assert!(
                        p.destinations
                            .iter()
                            .all(|&(_, address)| address == 0x98000 + offset)
                    );
                    offset += p.words * 4;
                }
                assert_eq!(offset, problem[0].words * 4);
                let run = schedule_exchange_problem_with_priority(
                    1472,
                    &problem,
                    ExchangeSchedulingPriority::BalancedStreams(1024),
                )
                .unwrap();
                validate_exchange_schedule(1472, &problem, &run).unwrap();
            }
        }
    }
}

#[test]
fn multicast_loopback_schedules_both_roles_and_rejects_bank_aliases() {
    for (words, width) in [
        (1, ExchangeItemWidth::Word32),
        (52, ExchangeItemWidth::Word32),
        (65, ExchangeItemWidth::Word32),
        (512, ExchangeItemWidth::Word32),
        (128, ExchangeItemWidth::Paired64),
        (512, ExchangeItemWidth::Paired64),
    ] {
        let mut problem: Vec<PendingTransfer> = vec![transfer(
            0,
            vec![0x65000],
            [0, 1, 2, 3]
                .into_iter()
                .map(|tile| (tile, 0x98000))
                .collect(),
            words,
            width,
        )];
        let run = schedule_exchange_problem(4, &problem).unwrap();
        validate_exchange_schedule(4, &problem, &run).unwrap();
        assert!(
            run.activities[0]
                .iter()
                .any(|a| a.kind == ExchangeActivityKind::Send)
        );
        assert!(
            run.activities[0]
                .iter()
                .any(|a| a.kind == ExchangeActivityKind::Receive)
        );
        problem[0].destinations[0].1 = 0x65004;
        assert!(schedule_exchange_problem(4, &problem).is_err());
        // Every alternative source address in a repeated phase must be safe.
        problem[0].destinations[0].1 = 0x98000;
        problem[0].source_addresses.push(0x98004);
        problem[0].refresh_source_elements();
        assert!(schedule_exchange_problem(4, &problem).is_err());
    }
}

#[test]
fn randomized_ready_matchings_have_maximum_cardinality() {
    let mut random = fastrand::Rng::with_seed(0x6d61_7463_6869_6e67);
    for _ in 0..128 {
        let tile_count = random.usize(2..=8);
        let mut transfers = Vec::new();
        let mut adjacency = vec![Vec::new(); tile_count];
        for (source, edges) in adjacency.iter_mut().enumerate() {
            for destination in 0..tile_count {
                if source == destination || !random.bool() {
                    continue;
                }
                let index = transfers.len();
                transfers.push(transfer(
                    source as u16,
                    vec![0x1_0000],
                    vec![(destination as u16, 0x4_0000)],
                    1,
                    ExchangeItemWidth::Word32,
                ));
                edges.push(index);
            }
        }
        let problem: Vec<PendingTransfer> = transfers;
        let pending = problem.clone();
        let source_order = (0..tile_count).collect::<Vec<_>>();
        let matching = maximum_ready_matching(&pending, &adjacency, &source_order, tile_count);
        let mut matched_sources = BTreeSet::new();
        let mut matched_destinations = BTreeSet::new();
        for &index in &matching {
            assert!(matched_sources.insert(pending[index].source));
            assert!(matched_destinations.insert(pending[index].destinations[0].0));
        }

        let mut reachable = vec![false; 1usize << tile_count];
        reachable[0] = true;
        for edges in &adjacency {
            let mut next = reachable.clone();
            for (mask, &is_reachable) in reachable.iter().enumerate() {
                if !is_reachable {
                    continue;
                }
                for &index in edges {
                    let destination = usize::from(pending[index].destinations[0].0);
                    if mask & (1 << destination) == 0 {
                        next[mask | (1 << destination)] = true;
                    }
                }
            }
            reachable = next;
        }
        let expected = reachable
            .iter()
            .enumerate()
            .filter(|(_, reachable)| **reachable)
            .map(|(mask, _)| mask.count_ones() as usize)
            .max()
            .unwrap_or(0);
        assert_eq!(matching.len(), expected);
    }
}

#[test]
fn randomized_matching_wave_orders_preserve_memory_dependencies() {
    let mut random = fastrand::Rng::with_seed(0x7761_7665_5f64_6167);
    for _ in 0..64 {
        let tile_count = random.u16(8..=24);
        let waves = random.u16(4..=8);
        let mut transfers = Vec::new();
        for wave in 0..waves {
            let shift = random.u16(1..tile_count);
            for source in 0..tile_count {
                transfers.push(transfer(
                    source,
                    vec![0x1_0000 + u32::from(random.u16(0..=wave)) * 0x100],
                    vec![(
                        (source + shift) % tile_count,
                        0x4_0000 + u32::from(random.u16(0..=wave)) * 0x100,
                    )],
                    random.u32(1..=64),
                    ExchangeItemWidth::Word32,
                ));
            }
        }
        let problem: Vec<PendingTransfer> = transfers;
        let pending = problem.clone();
        let incumbent = (0..pending.len()).collect::<Vec<_>>();
        let order = matching::order(&SchedulingProblem::new(&pending, tile_count), &incumbent)
            .expect("balanced point-to-point phases have a matching-wave candidate");
        assert_eq!(
            Some(order.clone()),
            matching::reference_matching_wave_order(
                &SchedulingProblem::new(&pending, tile_count),
                &incumbent
            )
        );
        let mut positions = vec![usize::MAX; pending.len()];
        for (position, &index) in order.iter().enumerate() {
            assert_eq!(positions[index], usize::MAX);
            positions[index] = position;
        }
        assert!(positions.iter().all(|&position| position != usize::MAX));
        for (before, after) in memory_dependencies(&pending, tile_count) {
            assert!(positions[before] < positions[after]);
        }
    }
}

#[test]
fn independent_sends_pipeline_before_previous_payload_arrives() {
    let problem: Vec<PendingTransfer> = [0, 4]
        .into_iter()
        .enumerate()
        .map(|(index, source)| {
            transfer(
                source,
                vec![0x1_0000],
                vec![(2, 0x4_0000 + index as u32 * 0x100)],
                16,
                ExchangeItemWidth::Word32,
            )
        })
        .collect();
    let run = schedule_exchange_problem(8, &problem).unwrap();
    validate_exchange_schedule(8, &problem, &run).unwrap();
    let mut receives = run.activities[2]
        .iter()
        .filter(|a| a.kind == ExchangeActivityKind::Receive)
        .collect::<Vec<_>>();
    receives.sort_by_key(|a| a.start_cycle);
    assert_eq!(receives.len(), 2);
    let second_send = run
        .activities
        .iter()
        .flatten()
        .find(|a| a.kind == ExchangeActivityKind::Send && a.transfer == receives[1].transfer)
        .unwrap();
    assert!(
        second_send.start_cycle < receives[0].end_cycle,
        "independent send should enter the route before the previous receive finishes"
    );
    assert!(receives[1].start_cycle >= receives[0].end_cycle);
}

#[test]
fn randomized_schedules_are_deterministic_and_valid() {
    let mut random = fastrand::Rng::with_seed(0x736e_6170_7368_6f74);
    for phase in 0..32 {
        let tile_count = random.u16(2..=16);
        let transfer_count = random.usize(1..=64);
        let transfers = (0..transfer_count)
            .map(|_| {
                let source = random.u16(0..tile_count);
                let destination_count = random.usize(1..=usize::from(tile_count.min(4) - 1));
                let mut tiles = BTreeSet::new();
                while tiles.len() != destination_count {
                    let tile = random.u16(0..tile_count);
                    if tile != source {
                        tiles.insert(tile);
                    }
                }
                let mut source_addresses = vec![0x1_0000 + random.u32(0..32) * 0x100];
                if random.bool() {
                    source_addresses.push(0x4_2000 + random.u32(0..2) * 0x4000);
                }
                transfer(
                    source,
                    source_addresses,
                    tiles
                        .into_iter()
                        .map(|tile| (tile, 0x4_0000 + random.u32(0..2) * 0x4000))
                        .collect(),
                    random.u32(1..=64),
                    ExchangeItemWidth::Word32,
                )
            })
            .collect();
        let problem: Vec<PendingTransfer> = transfers;
        for priority in [
            ExchangeSchedulingPriority::Automatic,
            ExchangeSchedulingPriority::RemainingDirectional,
            ExchangeSchedulingPriority::Streams(64),
            ExchangeSchedulingPriority::Streams(1024),
            ExchangeSchedulingPriority::BalancedStreams(64),
            ExchangeSchedulingPriority::BalancedStreams(1024),
        ] {
            let first =
                schedule_exchange_problem_with_priority(tile_count, &problem, priority).unwrap();
            validate_exchange_schedule(tile_count, &problem, &first).unwrap();
            let second =
                schedule_exchange_problem_with_priority(tile_count, &problem, priority).unwrap();
            validate_exchange_schedule(tile_count, &problem, &second).unwrap();
            assert_eq!(first, second);
        }
        let topology = Topology::new(
            (0..tile_count)
                .map(ipu_target::c600::logical_to_physical)
                .collect(),
        )
        .unwrap();
        let pending = problem.clone();
        let (receive_counts, incoming_bases) = receive_configuration(&pending, tile_count).unwrap();
        let scheduling = SchedulingProblem::new(&pending, tile_count);
        let baseline = streams::schedule(
            &topology,
            &scheduling,
            &incoming_bases,
            &receive_counts,
            64,
            true,
        )
        .unwrap();
        let optimized = optimize_pending_schedule(
            &topology,
            &pending,
            &incoming_bases,
            &receive_counts,
            tile_count,
            std::num::NonZeroU32::new(64),
        )
        .unwrap();
        let (maximum, total) = encoded_row_storage(&baseline).unwrap();
        let (new_maximum, new_total) = encoded_row_storage(&optimized.schedule).unwrap();
        assert_eq!((new_maximum, new_total), (maximum, total));
        assert_eq!(optimized.schedule.horizon, baseline.horizon);
        let run = optimized
            .schedule
            .into_phase(ExchangePhaseId::from_index(phase), incoming_bases)
            .unwrap();
        validate_exchange_schedule(tile_count, &problem, &run).unwrap();
    }
}

#[test]
fn width_selection_pairs_receivers_with_independent_addresses() {
    let topology = Topology::c600();
    let problem: Vec<PendingTransfer> = [0, 4]
        .into_iter()
        .map(|source| {
            transfer(
                source,
                vec![0x8_0000],
                (if source == 0 {
                    [0, 1]
                } else {
                    [source + 2, source + 3]
                })
                .into_iter()
                .map(|tile| (tile, 0x8_8000 + u32::from(tile & 1) * 0x4000))
                .collect(),
                64,
                ExchangeItemWidth::Word32,
            )
        })
        .collect();
    let pending = problem.clone();
    let ordinary = optimize_owned_pending(&topology, pending.clone(), 8, None).unwrap();
    let selected = select_transfer_widths(0, &topology, pending, 8, None).unwrap();
    assert!(
        selected
            .pending
            .iter()
            .all(|transfer| transfer.width == ExchangeItemWidth::Paired64)
    );
    assert!(selected.optimized.schedule.horizon < ordinary.optimized.schedule.horizon);
    let paired = selected.pending.clone();
    let run = schedule_exchange_problem(8, &paired).unwrap();
    validate_exchange_schedule(8, &paired, &run).unwrap();
}

#[test]
fn borrowed_transmit_lane_allows_receive_but_excludes_local_send() {
    let topology =
        Topology::new((0..64).map(ipu_target::c600::logical_to_physical).collect()).unwrap();
    for source in [0, 1] {
        let partner = source ^ 1;
        let transfer = |source, destinations: &[u16], words, width| {
            transfer(
                source,
                vec![0x8_0000],
                destinations.iter().map(|&tile| (tile, 0x8_8000)).collect(),
                words,
                width,
            )
        };
        let problem: Vec<PendingTransfer> = vec![
            transfer(source, &[2, 3], 4096, ExchangeItemWidth::Paired64),
            transfer(4, &[partner], 2048, ExchangeItemWidth::Word32),
            transfer(partner, &[5], 256, ExchangeItemWidth::Word32),
        ];
        let pending = problem.clone();
        let (counts, bases) = receive_configuration(&pending, 64).unwrap();
        for order in [[0, 1, 2], [1, 0, 2]] {
            let schedule = materialize_schedule_order(
                &topology,
                &SchedulingProblem::new(&pending, 64),
                &bases,
                &counts,
                &order,
                false,
            )
            .unwrap();
            let activities = &schedule.activities[usize::from(partner)];
            let activity = |kind| activities.iter().find(|a| a.kind == kind).unwrap();
            let borrowed = activity(ExchangeActivityKind::PartnerBusy);
            let receive = activity(ExchangeActivityKind::Receive);
            let send = activity(ExchangeActivityKind::Send);
            assert!(
                receive.start_cycle < borrowed.end_cycle
                    && borrowed.start_cycle < receive.end_cycle
            );
            assert!(send.start_cycle >= borrowed.end_cycle);
            assert!(endpoint_work_lower_bound(&pending, 64) <= schedule.horizon);
        }
        let run = schedule_exchange_problem(64, &problem).unwrap();
        validate_exchange_schedule(64, &problem, &run).unwrap();
    }
}

#[test]
fn randomized_eligible_physical_pairs_use_double_width_transfers() {
    let topology = Topology::c600();
    let tile_count = topology.tile_count() as u16;
    let all_pairs = (0..tile_count)
        .filter_map(|tile| {
            let paired = topology.paired_logical(tile).ok()?;
            (tile < paired).then_some([tile, paired])
        })
        .collect::<Vec<_>>();
    let mut random = fastrand::Rng::with_seed(0x7769_6465_7061_6972);

    for case in 0..128 {
        let source = random.u16(0..tile_count);
        let source_pair = topology.paired_logical(source).unwrap();
        let mut candidates = all_pairs
            .iter()
            .copied()
            .filter(|pair| !pair.contains(&source) && !pair.contains(&source_pair))
            .collect::<Vec<_>>();
        random.shuffle(&mut candidates);
        let pair_count = random.usize(1..=candidates.len().min(16));
        let destination_base = if case & 1 == 0 { 0x1_8000 } else { 0x4_4000 };
        let destinations = candidates[..pair_count]
            .iter()
            .enumerate()
            .flat_map(|(index, pair)| {
                let address = destination_base + (index as u32) * 0x200;
                pair.map(|tile| (tile, address))
            })
            .collect::<Vec<_>>();
        let words = random.u32(64..=512) * 2;
        let source_address = if random.bool() { 0x1_0000 } else { 0x4_2000 };
        let original = PendingTransfer {
            source,
            source_shard: BlockValueId::from_index(u32::from(source)),
            source_offset: random.u32(0..16) * 8,
            destinations: destinations.clone(),
            source_addresses: vec![source_address],
            source_elements: effective_memory_elements(source_address, words),
            words,
            width: ExchangeItemWidth::Word32,
            reserved_source: None,
        };
        let paired_tiles = destinations
            .iter()
            .map(|&(tile, _)| tile)
            .collect::<Vec<_>>();
        let paired_is_encodable =
            crate::exchange::paired_multicast(&topology, source, &paired_tiles, (words & !1) / 2)
                .is_ok();
        let alternatives =
            paired_transfer_alternatives(std::slice::from_ref(&original), &topology, tile_count)
                .unwrap();
        if !paired_is_encodable {
            assert!(alternatives[0].is_none());
            continue;
        }
        let paired = alternatives[0]
            .as_ref()
            .expect("eligible transfer must have a double-width part");
        assert_eq!(paired.source, source);
        assert_eq!(paired.reserved_source, Some(source_pair));
        assert_eq!(paired.words, words & !1);
        let mut paired_destinations = paired.destinations.clone();
        paired_destinations.sort_unstable();
        let mut expected_destinations = destinations.clone();
        expected_destinations.sort_unstable();
        assert_eq!(paired_destinations, expected_destinations);
        assert_eq!(paired.source_addresses, original.source_addresses);

        for failure in 0..4 {
            let mut inexact = original.clone();
            match failure {
                0 => inexact.words -= 1,
                1 => {
                    inexact.destinations.pop();
                }
                // Every Repeat source must support the selected width.
                2 => inexact.source_addresses.push(source_address + 4),
                _ => inexact.destinations[0].1 += 4,
            }
            assert!(
                paired_transfer_alternatives(std::slice::from_ref(&inexact), &topology, tile_count)
                    .unwrap()[0]
                    .is_none()
            );
        }
    }
}

#[test]
fn gemm_smoke_reblocking_uses_word_aligned_exchange() {
    let tiles = 64;
    let mut graph = HighGraph::new();
    let left = graph.host_input("left", [1, 64, 64]).unwrap();
    let right = graph.parameter("right", [1, 64, 4096]).unwrap();
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
                layout: Layout::block_major_matrix(64, tiles),
            },
        );
    let mid = crate::planner::build::baseline(
        &graph,
        &config,
        &Ipu21CostModel,
        &crate::planner::cache::FragmentCache::default(),
    )
    .unwrap();
    let expanded = crate::low::expand::expand_tiles(&mid, true).unwrap();
    let low = lower_to_tiles(&expanded, false);
    let placement = place(&low).unwrap();
    let phases = lower_exchanges(&low, &placement, &Topology::c600()).unwrap();
    assert!(!phases.is_empty());
}

#[test]
fn randomized_gemm_exchanges_produce_one_executable_row_per_tile() {
    let mut random = fastrand::Rng::with_seed(0x6578_6368);
    for _ in 0..32 {
        let tiles = 1_u16 << random.u32(1..=3);
        let rows = u32::from(tiles) * random.u32(1..=8);
        let columns = random.u32(1..=2) * 64;
        let mut graph = HighGraph::new();
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
                    layout: Layout::block_major_matrix(64, tiles),
                },
            );
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(
            &crate::expand_tiles(&mid).unwrap(),
            config.diagnostic_checkpoints,
        );
        let placement = place(&low).unwrap();
        let phases = lower_exchanges(&low, &placement, &Topology::c600()).unwrap();
        assert_eq!(phases.len(), low.exchange_phases.len());
        for phase in phases {
            assert_eq!(phase.programs.len(), usize::from(tiles));
            assert_eq!(phase.active.len(), usize::from(tiles));
            assert_eq!(phase.tile_event_cycles.len(), usize::from(tiles));
            assert_eq!(phase.activities.len(), usize::from(tiles));
            assert!(phase.event_cycles != 0);
            assert!(phase.activities.iter().flatten().next().is_some());
            for activities in &phase.activities {
                for activity in activities {
                    assert!(activity.start_cycle < activity.end_cycle);
                    assert!(activity.end_cycle <= phase.event_cycles);
                }
            }
            for ((active, program), local_cycles) in phase
                .active
                .iter()
                .zip(&phase.programs)
                .zip(&phase.tile_event_cycles)
            {
                let program = program.words();
                assert_eq!(program.last(), Some(&RETURN_M10_INSTRUCTION));
                assert_eq!(*active, program.len() > 1);
                assert_eq!(*active, *local_cycles != 0);
                assert_eq!(plan_event_cycles(program).unwrap(), *local_cycles);
                assert!(*local_cycles <= phase.event_cycles);
                assert!(
                    !program.contains(&ipu_target::ipu21::instruction::SYNC_SUPERVISOR_INSTRUCTION)
                );
            }
        }
    }
}

#[test]
fn dense_repeated_parameter_broadcasts_have_relocatable_exchange_rows() {
    use crate::estimate::Ipu21CostModel;
    use crate::{HighGraph, Layout, PipelineConfig, Precision, TensorFormat};
    let mut graph = HighGraph::new();
    let x = graph.host_input("x", [729, 1152]).unwrap();
    let parameters = (0..27)
        .map(|i| graph.parameter(format!("bias.{i}"), [1152]).unwrap())
        .collect::<Vec<_>>();
    let sequence = graph.value_sequence("bias", parameters.clone()).unwrap();
    let output = graph
        .repeat(27, [x], [], [sequence], |body, args| {
            Ok(vec![body.add(args.carried[0], args.iterated[0])?])
        })
        .unwrap()[0];
    graph.set_outputs([output]).unwrap();
    let mut config = PipelineConfig::new(1472).with_input(
        x,
        TensorFormat {
            precision: Precision::F16,
            layout: Layout::row_sharded(729),
        },
    );
    for parameter in parameters {
        config.inputs.insert(
            parameter,
            TensorFormat {
                precision: Precision::F16,
                layout: Layout::logical_linear(9, 128),
            },
        );
    }
    let mid = crate::planner::test_support::lower(&graph, &config, &Ipu21CostModel).unwrap();
    let expanded = crate::expand_tiles(&mid).unwrap();
    let low = crate::lower_to_tiles(&expanded, false);
    let placement = crate::place(&low).unwrap();
    let exchanges = crate::lower_exchanges(&low, &placement, &Topology::c600()).unwrap();
    assert!(
        exchanges
            .iter()
            .flat_map(|p| &p.outgoing_bases)
            .any(Option::is_some)
    );
    for phase in &exchanges {
        for (tile, base) in phase.outgoing_bases.iter().enumerate() {
            let Some((shard, offset)) = base else {
                continue;
            };
            assert!(phase.repeat_patches[tile].is_empty());
            let base = placement.shard_addresses[shard] + offset;
            let row = phase.programs[tile].words();
            let groups = sender_address_instruction_groups(row).unwrap();
            let sends = phase.activities[tile]
                .iter()
                .filter(|a| a.kind == ExchangeActivityKind::Send);
            for (group, send) in groups.into_iter().zip(sends) {
                for (word, offset) in group {
                    let mut expected = row[word];
                    patch_sender_instruction(&mut expected, send.address - base + offset).unwrap();
                    assert_eq!(row[word], expected);
                }
            }
        }
    }
}

#[test]
fn repeat_sources_follow_execution_order_when_sends_fill_earlier_gaps() {
    let topology =
        Topology::new((0..3).map(ipu_target::c600::logical_to_physical).collect()).unwrap();
    let pending = (0..2)
        .map(|i| {
            let address = 0x10000 + i * 0x4000;
            PendingTransfer {
                source: 0,
                source_shard: BlockValueId::from_index(i),
                source_offset: 0,
                source_addresses: vec![address, address + 256],
                source_elements: effective_memory_elements(address, 8),
                destinations: vec![(i as u16 + 1, 0x80000)],
                words: 8,
                width: ExchangeItemWidth::Word32,
                reserved_source: None,
            }
        })
        .collect::<Vec<_>>();
    let (counts, bases) = receive_configuration(&pending, 3).unwrap();
    let mut schedule = MaterializedSchedule::new(3, &pending);
    let mut predecessors = vec![TilePredecessor::default(); 3];
    schedule
        .append(
            &topology,
            &pending,
            &bases,
            &counts,
            0,
            1000,
            false,
            &mut predecessors,
        )
        .unwrap();
    schedule
        .append(
            &topology,
            &pending,
            &bases,
            &counts,
            1,
            0,
            false,
            &mut predecessors,
        )
        .unwrap();
    schedule.finish_horizon();
    let sends = schedule.activities[0]
        .iter()
        .filter(|activity| activity.kind == ExchangeActivityKind::Send)
        .collect::<Vec<_>>();
    assert_eq!(
        sends
            .iter()
            .map(|activity| activity.transfer)
            .collect::<Vec<_>>(),
        vec![1, 0]
    );
    let programs = schedule.builder.finish().unwrap();
    let row = programs.programs[0].as_ref().unwrap().words();
    let groups = sender_address_instruction_groups(row).unwrap();
    assert_eq!(groups.len(), sends.len());
    for (group, activity) in groups.iter().zip(sends) {
        for &(word, offset) in group {
            let mut instruction = row[word];
            patch_sender_instruction(
                &mut instruction,
                pending[activity.transfer as usize].source_address() + offset,
            )
            .unwrap();
            assert_eq!(instruction, row[word]);
        }
    }
}

#[test]
fn repeat_base_selects_tile_local_displacements_with_exceptions() {
    let id = BlockValueId::from_index;
    let make = |source, shard, address: u32, stride: u32| PendingTransfer {
        source,
        source_shard: id(shard),
        source_offset: 8,
        destinations: vec![(3, 0x80000)],
        source_addresses: (0..3).map(|i| address + i * stride).collect(),
        source_elements: vec![],
        words: 16,
        width: ExchangeItemWidth::Word32,
        reserved_source: None,
    };
    let mut transfers = vec![
        make(0, 0, 0x60008, 256),
        make(0, 1, 0x64008, 256),
        make(1, 2, 0x68008, 512),
        make(2, 3, 0x70008, 0),
    ];
    let addresses = BTreeMap::from([
        (id(0), 0x60000),
        (id(1), 0x64000),
        (id(2), 0x68000),
        (id(3), 0x70000),
    ]);
    let expected = vec![Some((id(0), 8)), Some((id(2), 8)), None, None];
    assert_eq!(
        repeat_outgoing_bases(&transfers, &vec![1; transfers.len()], &addresses, 4),
        expected
    );
    transfers.reverse();
    assert_eq!(
        repeat_outgoing_bases(&transfers, &vec![1; transfers.len()], &addresses, 4),
        expected
    );
    // Stationary sends use zero base independently of the moving group.
    transfers.push(make(0, 3, 0x70008, 0));
    assert_eq!(
        repeat_outgoing_bases(&transfers, &vec![1; transfers.len()], &addresses, 4),
        expected
    );
    // More stationary address words no longer discourage relocation.
    assert_eq!(
        repeat_outgoing_bases(&transfers, &[1, 1, 1, 1, 3], &addresses, 4),
        expected
    );
    transfers.pop();
    // An irregular exception is patched, not treated as sharing the base.
    transfers[2].source_addresses[2] += 4;
    assert_eq!(
        repeat_outgoing_bases(&transfers, &vec![1; transfers.len()], &addresses, 4),
        expected
    );
    // A low stationary address no longer prevents relative moving addresses.
    transfers.push(make(0, 3, 0x50008, 0));
    assert_eq!(
        repeat_outgoing_bases(&transfers, &vec![1; transfers.len()], &addresses, 4),
        expected
    );
}

#[test]
fn paired_repeat_base_preserves_encoded_alignment() {
    let shard = BlockValueId::from_index(0);
    let mut transfer = PendingTransfer {
        source: 0,
        source_shard: shard,
        source_offset: 4,
        source_addresses: vec![0x60004, 0x60104],
        source_elements: vec![],
        destinations: vec![(1, 0x80000)],
        words: 16,
        width: ExchangeItemWidth::Word32,
        reserved_source: None,
    };
    let addresses = BTreeMap::from([(shard, 0x60000)]);
    assert_eq!(
        repeat_outgoing_bases(&[transfer.clone()], &[1], &addresses, 2)[0],
        Some((shard, 4))
    );
    let mut paired = transfer.clone();
    paired.source_addresses.iter_mut().for_each(|a| *a += 4);
    paired.source_offset += 4;
    paired.width = ExchangeItemWidth::Paired64;
    assert_eq!(
        repeat_outgoing_bases(&[transfer.clone(), paired.clone()], &[1, 1], &addresses, 2),
        vec![None; 2]
    );
    transfer.source_addresses.iter_mut().for_each(|a| *a += 4);
    assert_eq!(
        repeat_outgoing_bases(&[transfer, paired], &[1, 1], &addresses, 2)[0],
        Some((shard, 8))
    );
}

#[test]
fn coalescing_preserves_loopback_dependencies_in_every_repeat_iteration() {
    let make = |offset: u32, destination, source_addresses: Vec<u32>| {
        let mut t = PendingTransfer {
            source: 0,
            source_shard: BlockValueId::from_index(0),
            source_offset: offset,
            source_addresses,
            source_elements: vec![],
            destinations: vec![(0, destination)],
            words: 16,
            width: ExchangeItemWidth::Word32,
            reserved_source: None,
        };
        t.refresh_source_elements();
        t
    };
    // The second Repeat binding would read the first transfer's destination.
    let a = make(0, 0x64040, vec![0x60000, 0x64000]);
    let b = make(64, 0x64080, vec![0x60040, 0x64040]);
    assert_eq!(coalesce_pending_transfers(vec![a, b]).len(), 2);
    let a = make(0, 0x68000, vec![0x60000, 0x64000]);
    let b = make(64, 0x68040, vec![0x60040, 0x64040]);
    let merged = coalesce_pending_transfers(vec![a, b]);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].words, 32);
    assert_eq!(
        merged[0].source_elements,
        [
            effective_memory_elements(0x60000, 32),
            effective_memory_elements(0x64000, 32)
        ]
        .concat()
    );
}

#[test]
fn compact_streams_order_inputs_before_ready_forwarders() {
    let transfer = |source, address, destination, words| {
        transfer(
            source,
            vec![address],
            vec![(destination, 0x64000)],
            words,
            ExchangeItemWidth::Word32,
        )
    };
    let phase = vec![
        transfer(0, 0x60000, 1, 16),
        transfer(2, 0x60000, 3, 16),
        transfer(1, 0x64000, 2, 1024),
    ];
    let pending = phase.clone();
    let problem = SchedulingProblem::new(&pending, 4);
    for balanced in [false, true] {
        let order = streams::order(&problem, 1024, balanced);
        assert_eq!(order.last(), Some(&2));
    }
}

#[test]
fn phase_finalization_rejects_a_stale_schedule_horizon() {
    let mut schedule = MaterializedSchedule::new(4, &[]);
    schedule.horizon = 1;
    assert!(matches!(
        schedule.into_phase(ExchangePhaseId::from_index(0), vec![0; 4]),
        Err(ExchangeLoweringError::Invariant(_))
    ));
}

pub(super) fn transfer(
    source: u16,
    source_addresses: Vec<u32>,
    destinations: Vec<(u16, u32)>,
    words: u32,
    width: ExchangeItemWidth,
) -> PendingTransfer {
    let mut transfer = PendingTransfer {
        source,
        source_shard: BlockValueId::from_index(u32::from(source)),
        source_offset: 0,
        source_addresses,
        destinations,
        words,
        width,
        source_elements: Vec::new(),
        reserved_source: (width == ExchangeItemWidth::Paired64)
            .then(|| Topology::c600().paired_logical(source).unwrap()),
    };
    transfer.refresh_source_elements();
    transfer
}
pub(super) fn select_exchange_schedule(
    tile_count: u16,
    transfers: &[PendingTransfer],
    stream_words: Option<std::num::NonZeroU32>,
    cache: &mut ExchangeScheduleCache,
) -> Result<(Vec<PendingTransfer>, PhysicalExchangePhase), ExchangeLoweringError> {
    let selected = select_phase(
        ExchangePhaseId::from_index(0),
        &Topology::c600(),
        transfers.to_vec(),
        tile_count,
        stream_words,
        cache,
    )?;
    Ok((
        selected.pending,
        selected
            .optimized
            .schedule
            .into_phase(ExchangePhaseId::from_index(0), selected.incoming_bases)?,
    ))
}
pub(super) fn schedule_exchange_problem(
    tile_count: u16,
    transfers: &[PendingTransfer],
) -> Result<PhysicalExchangePhase, ExchangeLoweringError> {
    schedule_exchange_problem_with_priority(
        tile_count,
        transfers,
        ExchangeSchedulingPriority::Automatic,
    )
}
pub(super) fn schedule_exchange_problem_with_priority(
    tile_count: u16,
    transfers: &[PendingTransfer],
    priority: ExchangeSchedulingPriority,
) -> Result<PhysicalExchangePhase, ExchangeLoweringError> {
    let topology = Topology::c600();
    let pending = packet::split_self_receive_conflicts(&topology, transfers.to_vec())?;
    let (counts, bases) = receive_configuration(&pending, tile_count)?;
    let problem = SchedulingProblem::new(&pending, tile_count);
    let schedule = match priority {
        ExchangeSchedulingPriority::Streams(words)
        | ExchangeSchedulingPriority::BalancedStreams(words) => streams::schedule(
            &topology,
            &problem,
            &bases,
            &counts,
            words,
            matches!(priority, ExchangeSchedulingPriority::BalancedStreams(_)),
        )?,
        _ => {
            let initial = greedy::schedule(&topology, &problem, &bases, &counts, priority)?;
            improve_pending_schedule(&topology, &problem, &bases, &counts, initial, "test")?
                .schedule
        }
    };
    schedule.into_phase(ExchangePhaseId::from_index(0), bases)
}
/// Checks that scheduled activities and encoded rows preserve the input
/// transfer set and obey per-tile bus and SRAM-element hazards.
pub(super) fn validate_exchange_schedule(
    tile_count: u16,
    transfers: &[PendingTransfer],
    phase: &PhysicalExchangePhase,
) -> Result<(), ExchangeLoweringError> {
    let packets = packet::split_self_receive_conflicts(&Topology::c600(), transfers.to_vec())?;
    let transfers = &packets;
    let fail = |message| ExchangeLoweringError::Invariant(message);
    let size = usize::from(tile_count);
    for (name, length) in [
        ("active", phase.active.len()),
        ("programs", phase.programs.len()),
        ("incoming bases", phase.incoming_bases.len()),
        ("tile horizons", phase.tile_event_cycles.len()),
        ("activities", phase.activities.len()),
        ("repeat patches", phase.repeat_patches.len()),
        ("outgoing bases", phase.outgoing_bases.len()),
    ] {
        if length != size {
            return Err(fail(format!(
                "phase {} has {length} {name} entries for {tile_count} tiles",
                phase.id.index()
            )));
        }
    }
    if phase
        .repeat_patches
        .iter()
        .any(|patches| !patches.is_empty())
        || phase.outgoing_bases.iter().any(Option::is_some)
    {
        return Err(fail(format!(
            "standalone phase {} unexpectedly contains repeat relocation",
            phase.id.index()
        )));
    }
    let maximum_horizon = phase.tile_event_cycles.iter().copied().max().unwrap_or(0);
    if phase.event_cycles != maximum_horizon {
        return Err(fail(format!(
            "phase {} horizon {} differs from maximum tile horizon {maximum_horizon}",
            phase.id.index(),
            phase.event_cycles
        )));
    }

    let mut send_counts = vec![0usize; transfers.len()];
    let mut partner_busy_counts = vec![0usize; transfers.len()];
    let mut receive_counts = transfers
        .iter()
        .map(|transfer| vec![0usize; transfer.destinations.len()])
        .collect::<Vec<_>>();
    let reserved_paired_sources = transfers
        .iter()
        .filter(|transfer| transfer.width == ExchangeItemWidth::Paired64)
        .map(|transfer| Topology::c600().paired_logical(transfer.source))
        .collect::<Result<BTreeSet<_>, _>>()?;
    for tile in 0..size {
        let decoded =
            crate::exchange::diagnostic::diagnose_plan_program(phase.programs[tile].words(), None)?;
        if decoded.event_cycles != phase.tile_event_cycles[tile] {
            return Err(fail(format!(
                "phase {} tile {tile} decoded horizon {} differs from {}",
                phase.id.index(),
                decoded.event_cycles,
                phase.tile_event_cycles[tile]
            )));
        }
        let tile_u16 = u16::try_from(tile).map_err(|_| ExchangeLoweringError::Overflow)?;
        let expected_active =
            !phase.activities[tile].is_empty() || reserved_paired_sources.contains(&tile_u16);
        if phase.active[tile] != expected_active
            || phase.active[tile] != (phase.tile_event_cycles[tile] != 0)
        {
            return Err(fail(format!(
                "phase {} tile {tile} has inconsistent active state",
                phase.id.index()
            )));
        }
        for activity in &phase.activities[tile] {
            if activity.start_cycle > activity.end_cycle
                || activity.end_cycle > activity.memory_end_cycle
                || activity.memory_end_cycle > phase.tile_event_cycles[tile]
            {
                return Err(fail(format!(
                    "phase {} tile {tile} transfer {} has invalid cycle interval",
                    phase.id.index(),
                    activity.transfer
                )));
            }
            let transfer_index =
                usize::try_from(activity.transfer).map_err(|_| ExchangeLoweringError::Overflow)?;
            let transfer = transfers.get(transfer_index).ok_or_else(|| {
                fail(format!(
                    "phase {} tile {tile} references missing transfer {}",
                    phase.id.index(),
                    activity.transfer
                ))
            })?;
            if activity.words != transfer.words {
                return Err(fail(format!(
                    "phase {} tile {tile} transfer {transfer_index} has wrong word count",
                    phase.id.index()
                )));
            }
            match activity.kind {
                ExchangeActivityKind::Send => {
                    if usize::from(transfer.source) != tile
                        || activity.address != transfer.source_addresses[0]
                    {
                        return Err(fail(format!(
                            "phase {} transfer {transfer_index} has a mismatched send activity",
                            phase.id.index()
                        )));
                    }
                    send_counts[transfer_index] += 1;
                }
                ExchangeActivityKind::Receive => {
                    let destination = transfer
                        .destinations
                        .iter()
                        .position(|destination| {
                            usize::from(destination.0) == tile
                                && destination.1 == activity.address
                        })
                        .ok_or_else(|| {
                            fail(format!(
                                "phase {} transfer {transfer_index} has an unexpected receive activity on tile {tile}",
                                phase.id.index()
                            ))
                        })?;
                    receive_counts[transfer_index][destination] += 1;
                }
                ExchangeActivityKind::PartnerBusy => {
                    let expected = (transfer.width == ExchangeItemWidth::Paired64)
                        .then(|| Topology::c600().paired_logical(transfer.source))
                        .transpose()?;
                    if expected != Some(tile_u16)
                        || activity.address != transfer.source_addresses[0]
                    {
                        return Err(fail(format!(
                            "phase {} transfer {transfer_index} has a mismatched partner-busy activity",
                            phase.id.index()
                        )));
                    }
                    partner_busy_counts[transfer_index] += 1;
                }
            }
        }
        for kind in [ExchangeActivityKind::Send, ExchangeActivityKind::Receive] {
            let mut intervals = phase.activities[tile]
                .iter()
                .filter(|activity| activity.kind == kind)
                .map(|activity| (activity.start_cycle, activity.end_cycle))
                .collect::<Vec<_>>();
            intervals.sort_unstable();
            if intervals.windows(2).any(|pair| pair[1].0 < pair[0].1) {
                return Err(fail(format!(
                    "phase {} tile {tile} has overlapping {kind:?} bus intervals",
                    phase.id.index()
                )));
            }
        }
        let sends = phase.activities[tile]
            .iter()
            .filter(|activity| activity.kind == ExchangeActivityKind::Send);
        for send in sends {
            let transfer = &transfers[send.transfer as usize];
            for receive in phase.activities[tile]
                .iter()
                .filter(|activity| activity.kind == ExchangeActivityKind::Receive)
            {
                let overlaps = send.start_cycle < receive.memory_end_cycle
                    && receive.start_cycle < send.memory_end_cycle;
                if overlaps
                    && transfer.source_addresses.iter().any(|&address| {
                        spans_share_effective_memory_element(
                            address,
                            send.words,
                            receive.address,
                            receive.words,
                        )
                    })
                {
                    return Err(fail(format!(
                        "phase {} tile {tile} overlaps send/receive access to one SRAM element",
                        phase.id.index()
                    )));
                }
            }
        }
        for partner_busy in phase.activities[tile]
            .iter()
            .filter(|activity| activity.kind == ExchangeActivityKind::PartnerBusy)
        {
            if phase.activities[tile].iter().any(|activity| {
                activity.transfer != partner_busy.transfer
                    && activity.kind != ExchangeActivityKind::Receive
                    && activity.start_cycle < partner_busy.end_cycle
                    && partner_busy.start_cycle < activity.end_cycle
            }) {
                return Err(fail(format!(
                    "phase {} tile {tile} overlaps borrowed and local transmit intervals",
                    phase.id.index()
                )));
            }
        }
    }
    for (index, count) in send_counts.into_iter().enumerate() {
        if count != 1 {
            return Err(fail(format!(
                "phase {} transfer {index} has {count} send activities",
                phase.id.index()
            )));
        }
    }
    for (index, count) in partner_busy_counts.into_iter().enumerate() {
        let expected = usize::from(transfers[index].width == ExchangeItemWidth::Paired64);
        if count != expected {
            return Err(fail(format!(
                "phase {} transfer {index} has {count} partner-busy activities, expected {expected}",
                phase.id.index()
            )));
        }
    }
    for (transfer, counts) in receive_counts.into_iter().enumerate() {
        if counts.into_iter().any(|count| count != 1) {
            return Err(fail(format!(
                "phase {} transfer {transfer} does not have exactly one activity per destination",
                phase.id.index()
            )));
        }
    }
    Ok(())
}
