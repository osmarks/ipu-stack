use super::order::maximum_ready_matching;
use super::*;
use crate::{
    ComputeGraph, Ipu21CostModel, Layout, PipelineConfig, Precision, TensorFormat, lower,
    lower_to_tiles, place,
};

#[test]
fn grouped_ready_queue_matches_eager_priority() {
    let mut random = fastrand::Rng::with_seed(0x7061697273);
    for case in 0..16 {
        let tiles = if case & 4 == 0 { 8 } else { 32 };
        let transfers = (0..1024)
            .map(|index| {
                let source = random.u16(0..tiles);
                let destination = (source + random.u16(1..tiles)) % tiles;
                let words = random.u32(1..=512);
                let source_address = if case % 2 == 0 {
                    0
                } else {
                    0x80000 + random.u32(0..16) * 4096
                };
                let destination_address = if case % 2 == 0 {
                    0x80000 + index * 4096
                } else {
                    0x80000 + random.u32(0..16) * 4096
                };
                let mut destinations = vec![(destination, destination_address)];
                if case & 2 != 0 {
                    let other = (destination + 1) % tiles;
                    if other != source {
                        destinations.push((other, destination_address));
                    }
                }
                PendingTransfer {
                    source,
                    source_shard: BlockValueId::from_index(u32::from(source)),
                    source_offset: 0,
                    source_addresses: vec![source_address],
                    source_elements: effective_memory_elements(source_address, words),
                    destinations,
                    words,
                    width: ExchangeItemWidth::Word32,
                    reserved_source: None,
                }
            })
            .collect::<Vec<_>>();
        let problem = SchedulingProblem::new(&transfers, tiles);
        for priority in [
            ExchangeSchedulingPriority::Automatic,
            ExchangeSchedulingPriority::Combined,
            ExchangeSchedulingPriority::Directional,
            ExchangeSchedulingPriority::RemainingCombined,
            ExchangeSchedulingPriority::RemainingDirectional,
        ] {
            let mut grouped = TransferScheduler::with_priority(&problem, priority);
            assert!(!grouped.ready_groups.is_empty());
            assert!(grouped.ready_groups.len() <= 2 * usize::from(tiles * (tiles - 1)));
            let mut reference = TransferScheduler::with_priority(&problem, priority);
            reference.ready = std::mem::take(&mut reference.ready_groups)
                .into_iter()
                .flatten()
                .collect();
            reference.transfer_group.clear();
            let mut availability = vec![TileAvailability::default(); usize::from(tiles)];
            while let Some(actual) = grouped.next(&availability) {
                let expected = reference
                    .ready
                    .iter()
                    .map(|&entry| reference.refresh(entry, &availability))
                    .max()
                    .unwrap();
                assert_eq!(
                    actual,
                    (
                        expected.index.0,
                        reference.dependency_ready[expected.index.0]
                    )
                );
                reference
                    .ready
                    .retain(|entry| entry.index != expected.index);
                let (index, dependency) = actual;
                let transfer = &transfers[index];
                let source = usize::from(transfer.source);
                let completion = transfer
                    .destinations
                    .iter()
                    .map(|&(tile, _)| availability[usize::from(tile)].receive)
                    .chain([availability[source].send, dependency])
                    .max()
                    .unwrap()
                    + transfer.words;
                availability[source].send = completion;
                for &(tile, _) in &transfer.destinations {
                    availability[usize::from(tile)].receive = completion;
                }
                grouped.complete(index, completion);
                reference.complete(index, completion);
            }
            assert!(grouped.is_complete());
            assert!(reference.is_complete());
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
        let mut problem = ExchangeScheduleProblem {
            phase: 0,
            transfers: vec![ExchangeScheduleTransfer {
                source: 0,
                source_addresses: vec![0x65000],
                destinations: [0, 1, 2, 3]
                    .into_iter()
                    .map(|tile| ExchangeScheduleDestination {
                        tile,
                        address: 0x98000,
                    })
                    .collect(),
                words,
                width,
            }],
        };
        let run = schedule_exchange_problem(4, &problem).unwrap();
        validate_exchange_schedule(4, &problem, &run.phase).unwrap();
        assert!(
            run.phase.activities[0]
                .iter()
                .any(|a| a.kind == ExchangeActivityKind::Send)
        );
        assert!(
            run.phase.activities[0]
                .iter()
                .any(|a| a.kind == ExchangeActivityKind::Receive)
        );
        problem.transfers[0].destinations[0].address = 0x65004;
        assert!(schedule_exchange_problem(4, &problem).is_err());
        // Every alternative source address in a repeated phase must be safe.
        problem.transfers[0].destinations[0].address = 0x98000;
        problem.transfers[0].source_addresses.push(0x98004);
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
                transfers.push(ExchangeScheduleTransfer {
                    source: source as u16,
                    source_addresses: vec![0x1_0000],
                    destinations: vec![ExchangeScheduleDestination {
                        tile: destination as u16,
                        address: 0x4_0000,
                    }],
                    words: 1,
                    width: ExchangeItemWidth::Word32,
                });
                edges.push(index);
            }
        }
        let problem = ExchangeScheduleProblem {
            phase: 0,
            transfers,
        };
        let pending = pending_from_problem(tile_count as u16, &problem).unwrap();
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
                transfers.push(ExchangeScheduleTransfer {
                    source,
                    source_addresses: vec![0x1_0000 + u32::from(random.u16(0..=wave)) * 0x100],
                    destinations: vec![ExchangeScheduleDestination {
                        tile: (source + shift) % tile_count,
                        address: 0x4_0000 + u32::from(random.u16(0..=wave)) * 0x100,
                    }],
                    words: random.u32(1..=64),
                    width: ExchangeItemWidth::Word32,
                });
            }
        }
        let problem = ExchangeScheduleProblem {
            phase: 0,
            transfers,
        };
        let pending = pending_from_problem(tile_count, &problem).unwrap();
        let incumbent = (0..pending.len()).collect::<Vec<_>>();
        let order = point_to_point_matching_wave_order(
            &SchedulingProblem::new(&pending, tile_count),
            &incumbent,
        )
        .expect("balanced point-to-point phases have a matching-wave candidate");
        assert_eq!(
            Some(order.clone()),
            order::reference_matching_wave_order(
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
    let problem = ExchangeScheduleProblem {
        phase: 0,
        transfers: [0, 4]
            .into_iter()
            .enumerate()
            .map(|(index, source)| ExchangeScheduleTransfer {
                source,
                source_addresses: vec![0x1_0000],
                destinations: vec![ExchangeScheduleDestination {
                    tile: 2,
                    address: 0x4_0000 + index as u32 * 0x100,
                }],
                words: 16,
                width: ExchangeItemWidth::Word32,
            })
            .collect(),
    };
    let run = schedule_exchange_problem(8, &problem).unwrap();
    validate_exchange_schedule(8, &problem, &run.phase).unwrap();
    let mut receives = run.phase.activities[2]
        .iter()
        .filter(|a| a.kind == ExchangeActivityKind::Receive)
        .collect::<Vec<_>>();
    receives.sort_by_key(|a| a.start_cycle);
    assert_eq!(receives.len(), 2);
    let second_send = run
        .phase
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
fn randomized_captured_schedule_replays_are_deterministic_and_valid() {
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
                ExchangeScheduleTransfer {
                    source,
                    source_addresses,
                    destinations: tiles
                        .into_iter()
                        .map(|tile| ExchangeScheduleDestination {
                            tile,
                            address: 0x4_0000 + random.u32(0..2) * 0x4000,
                        })
                        .collect(),
                    words: random.u32(1..=64),
                    width: ExchangeItemWidth::Word32,
                }
            })
            .collect();
        let problem = ExchangeScheduleProblem { phase, transfers };
        for priority in [
            ExchangeSchedulingPriority::Automatic,
            ExchangeSchedulingPriority::RemainingDirectional,
            ExchangeSchedulingPriority::Streams(64),
            ExchangeSchedulingPriority::Streams(1024),
        ] {
            let first =
                schedule_exchange_problem_with_priority(tile_count, &problem, priority).unwrap();
            validate_exchange_schedule(tile_count, &problem, &first.phase).unwrap();
            let second =
                schedule_exchange_problem_with_priority(tile_count, &problem, priority).unwrap();
            validate_exchange_schedule(tile_count, &problem, &second.phase).unwrap();
            assert_eq!(first.phase, second.phase);
            assert_eq!(first.initial_horizon, second.initial_horizon);
            assert_eq!(first.endpoint_lower_bound, second.endpoint_lower_bound);
            assert_eq!(
                first.neighborhood_improvements,
                second.neighborhood_improvements
            );
        }
    }
}

#[test]
fn width_selection_pairs_receivers_with_independent_addresses() {
    let topology = Topology::c600();
    let problem = ExchangeScheduleProblem {
        phase: 0,
        transfers: [0, 4]
            .into_iter()
            .map(|source| ExchangeScheduleTransfer {
                source,
                source_addresses: vec![0x8_0000],
                destinations: (if source == 0 {
                    [0, 1]
                } else {
                    [source + 2, source + 3]
                })
                .into_iter()
                .map(|tile| ExchangeScheduleDestination {
                    tile,
                    address: 0x8_8000 + u32::from(tile & 1) * 0x4000,
                })
                .collect(),
                words: 64,
                width: ExchangeItemWidth::Word32,
            })
            .collect(),
    };
    let pending = pending_from_problem(8, &problem).unwrap();
    let ordinary = optimize_owned_pending(&topology, pending.clone(), 8).unwrap();
    let selected = select_transfer_widths(0, &topology, pending, 8).unwrap();
    assert!(
        selected
            .pending
            .iter()
            .all(|transfer| transfer.width == ExchangeItemWidth::Paired64)
    );
    assert!(selected.optimized.schedule.horizon < ordinary.optimized.schedule.horizon);
    let paired = schedule_problem(0, &selected.pending);
    let run = schedule_exchange_problem(8, &paired).unwrap();
    validate_exchange_schedule(8, &paired, &run.phase).unwrap();
}

#[test]
fn borrowed_transmit_lane_allows_receive_but_excludes_local_send() {
    let topology = Topology::new(
        (0..64)
            .map(ipu_exchange::c600_logical_to_physical)
            .collect(),
    )
    .unwrap();
    for source in [0, 1] {
        let partner = source ^ 1;
        let transfer = |source, destinations: &[u16], words, width| ExchangeScheduleTransfer {
            source,
            source_addresses: vec![0x8_0000],
            destinations: destinations
                .iter()
                .map(|&tile| ExchangeScheduleDestination {
                    tile,
                    address: 0x8_8000,
                })
                .collect(),
            words,
            width,
        };
        let problem = ExchangeScheduleProblem {
            phase: 0,
            transfers: vec![
                transfer(source, &[2, 3], 4096, ExchangeItemWidth::Paired64),
                transfer(4, &[partner], 2048, ExchangeItemWidth::Word32),
                transfer(partner, &[5], 256, ExchangeItemWidth::Word32),
            ],
        };
        let pending = pending_from_problem(64, &problem).unwrap();
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
        validate_exchange_schedule(64, &problem, &run.phase).unwrap();
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
        let paired_is_encodable = topology
            .paired_multicast(source, &paired_tiles, (words & !1) / 2)
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

        let mut inexact = original.clone();
        if random.bool() {
            inexact.words -= 1;
        } else {
            inexact.destinations.pop();
        }
        assert!(
            paired_transfer_alternatives(std::slice::from_ref(&inexact), &topology, tile_count)
                .unwrap()[0]
                .is_none()
        );
    }
}

#[test]
fn randomized_transfer_schedules_preserve_hazards_without_same_role_overlap() {
    let mut random = fastrand::Rng::with_seed(0x736c_6f74);
    for _ in 0..64 {
        let tile_count = random.u16(2..=32);
        let transfer_count = random.usize(1..=256);
        let transfers = (0..transfer_count)
            .map(|_| {
                let source = random.u16(0..tile_count);
                let receiver_count = random.usize(1..=usize::from(tile_count.min(8) - 1));
                let mut receivers = Vec::with_capacity(receiver_count);
                while receivers.len() != receiver_count {
                    let tile = random.u16(0..tile_count);
                    if tile != source && !receivers.contains(&tile) {
                        receivers.push(tile);
                    }
                }
                let words = random.u32(1..=MAX_TRANSFER_WORDS);
                PendingTransfer {
                    source,
                    source_shard: BlockValueId::from_index(u32::from(source)),
                    source_offset: 0,
                    destinations: receivers.into_iter().map(|tile| (tile, 0)).collect(),
                    source_addresses: vec![0],
                    source_elements: effective_memory_elements(0, words),
                    words,
                    width: ExchangeItemWidth::Word32,
                    reserved_source: None,
                }
            })
            .collect::<Vec<_>>();
        let dependencies = memory_dependencies(&transfers, tile_count);
        let problem = SchedulingProblem::new(&transfers, tile_count);
        let mut scheduler = TransferScheduler::new(&problem);
        let mut availability = vec![TileAvailability::default(); usize::from(tile_count)];
        let mut occurrences = vec![0u8; transfers.len()];
        let mut intervals = vec![(0u32, 0u32); transfers.len()];
        while let Some((index, dependency_ready)) = scheduler.next(&availability) {
            occurrences[index] += 1;
            let transfer = &transfers[index];
            let start = std::iter::once(dependency_ready)
                .chain(std::iter::once(
                    availability[usize::from(transfer.source)].send,
                ))
                .chain(
                    transfer
                        .destinations
                        .iter()
                        .map(|&(tile, _)| availability[usize::from(tile)].receive),
                )
                .max()
                .unwrap_or(0);
            let end = start.saturating_add(transfers[index].words);
            intervals[index] = (start, end);
            availability[usize::from(transfer.source)].send = end;
            for &(tile, _) in &transfer.destinations {
                availability[usize::from(tile)].receive = end;
            }
            scheduler.complete(index, end);
        }
        assert!(scheduler.is_complete());
        assert!(occurrences.into_iter().all(|count| count == 1));
        for &(before, after) in &dependencies {
            assert!(intervals[before].1 <= intervals[after].0);
        }
        for tile in 0..tile_count {
            let mut send_intervals = transfers
                .iter()
                .enumerate()
                .filter(|(_, transfer)| transfer.source == tile)
                .map(|(index, _)| intervals[index])
                .collect::<Vec<_>>();
            send_intervals.sort_unstable();
            assert!(send_intervals.windows(2).all(|pair| pair[0].1 <= pair[1].0));
            let mut receive_intervals = transfers
                .iter()
                .enumerate()
                .filter(|(_, transfer)| {
                    transfer
                        .destinations
                        .iter()
                        .any(|&(destination, _)| destination == tile)
                })
                .map(|(index, _)| intervals[index])
                .collect::<Vec<_>>();
            receive_intervals.sort_unstable();
            assert!(
                receive_intervals
                    .windows(2)
                    .all(|pair| pair[0].1 <= pair[1].0)
            );
        }

        let mut incumbent = MaterializedSchedule::new(tile_count, &transfers);
        incumbent.order.extend(0..transfers.len());
        let mut last_transfer = vec![None; usize::from(tile_count)];
        for (index, transfer) in transfers.iter().enumerate() {
            let predecessor = transfer
                .tiles()
                .filter_map(|tile| last_transfer[usize::from(tile)])
                .max();
            for tile in transfer.tiles() {
                last_transfer[usize::from(tile)] = Some(index);
            }
            incumbent.timings[index] = Some(MaterializedTiming {
                start: index as u32,
                end: index as u32 + 1,
                blocking_tile: transfer.source,
                predecessor,
            });
        }
        let repaired = critical_neighborhood_order(
            &SchedulingProblem::new(&transfers, tile_count),
            &incumbent,
            true,
        )
        .unwrap();
        let mut repaired_positions = vec![usize::MAX; transfers.len()];
        for (position, &index) in repaired.iter().enumerate() {
            assert_eq!(repaired_positions[index], usize::MAX);
            repaired_positions[index] = position;
        }
        assert!(
            repaired_positions
                .iter()
                .all(|position| *position != usize::MAX)
        );
        for &(before, after) in &dependencies {
            assert!(repaired_positions[before] < repaired_positions[after]);
        }
    }
}

#[test]
fn gemm_smoke_reblocking_uses_word_aligned_exchange() {
    let tiles = 64;
    let mut graph = ComputeGraph::new();
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
    let mid = crate::mid::lower_finalists(&graph, &config, &Ipu21CostModel, 1)
        .unwrap()
        .remove(0);
    let expanded = crate::low::expand::expand_tiles(&mid, true).unwrap();
    let low = lower_to_tiles(&expanded, false);
    let placement = place(&low).unwrap();
    let phases = lower_exchanges(&low, &placement, &Topology::c600(), false).unwrap();
    assert!(!phases.phases.is_empty());
}

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
                    layout: Layout::block_major_matrix(64, tiles),
                },
            );
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(
            &crate::expand_tiles(&mid).unwrap(),
            config.diagnostic_checkpoints,
        );
        let placement = place(&low).unwrap();
        let phases = lower_exchanges(&low, &placement, &Topology::c600(), false)
            .unwrap()
            .phases;
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
                assert_eq!(program.last(), Some(&RETURN_M10_INSTRUCTION));
                assert_eq!(*active, program.len() > 1);
                assert_eq!(*active, *local_cycles != 0);
                assert_eq!(plan_event_cycles(program).unwrap(), *local_cycles);
                assert!(*local_cycles <= phase.event_cycles);
                assert!(!program.contains(&ipu_exchange::SYNC_SUPERVISOR_INSTRUCTION));
            }
        }
    }
}
